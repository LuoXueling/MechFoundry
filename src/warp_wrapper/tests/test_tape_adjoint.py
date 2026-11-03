"""Tests for adjoint tape support (implicit differentiation via wp.Tape).

Run with:
    conda activate mechfoundry-warp117
    pytest src/warp_wrapper/tests/test_tape_adjoint.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.neoHookean import NeoHookean
from core.utils import solve_linear_system

wp.init()
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Loss kernels
# ---------------------------------------------------------------------------
@wp.kernel
def _squared_l2_loss_2d(
    u: wp.array(dtype=wp.vec2f),
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(u[i], u[i]))


@wp.kernel
def _squared_l2_loss_3d(
    u: wp.array(dtype=wp.vec3f),
    loss: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    wp.atomic_add(loss, 0, wp.dot(u[i], u[i]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _forward_with_tape(solver, dim):
    """Run forward with tape, compute ||u||^2 loss, and backward."""
    tape = wp.Tape()
    solver.forward(tape=tape)

    n = solver.space.node_count()
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=solver.device)
    kernel = _squared_l2_loss_2d if dim == 2 else _squared_l2_loss_3d
    with tape:
        wp.launch(
            kernel, dim=n, inputs=[solver.u.dof_values, loss], device=solver.device
        )

    tape.backward(loss=loss)
    return tape, loss


# ---------------------------------------------------------------------------
# Test 1 — Smoke tests
# ---------------------------------------------------------------------------
class TestTapeSmoke:
    """Forward + loss + backward runs without error."""

    def test_smoke_2d(self, tmp_path):
        solver = NeoHookean(
            "tests.examples.two_d_affine_tension.config",
            override_params={"output_root": str(tmp_path)},
        )
        tape, loss = _forward_with_tape(solver, dim=2)

        assert solver._tape_residual is not None, "R_tape not stored"
        assert solver._tape_residual.grad is not None, "R_tape.grad not allocated"
        R_grad_np = solver._tape_residual.grad.numpy()
        assert R_grad_np.shape[0] == solver.space.node_count()
        # R_tape.grad should be non-trivial (not all zeros)
        assert np.linalg.norm(R_grad_np) > 0.0, "R_tape.grad is all zeros"

    def test_smoke_3d(self, tmp_path):
        solver = NeoHookean(
            "tests.examples.three_d_affine_tension.config",
            override_params={"output_root": str(tmp_path)},
        )
        tape, loss = _forward_with_tape(solver, dim=3)

        assert solver._tape_residual is not None
        assert solver._tape_residual.grad is not None
        assert solver._tape_residual.dtype._wp_scalar_type_ == wp.float32
        assert solver.dirichlet_free_mask_wp.dtype._wp_scalar_type_ == wp.float32
        R_grad_np = solver._tape_residual.grad.numpy()
        assert R_grad_np.shape[0] == solver.space.node_count()
        assert np.linalg.norm(R_grad_np) > 0.0


# ---------------------------------------------------------------------------
# Test 2 — Adjoint solve correctness
# ---------------------------------------------------------------------------
class TestAdjointSolveCorrectness:
    """Verify R_tape.grad matches an independently computed adjoint vector.

    For loss L = ||u||^2, dL/du = 2u.  The adjoint solves
        K * x = -masked(dL/du)
    and sets R_tape.grad = x.  We replicate this solve independently and
    compare the results.
    """

    @staticmethod
    def _check_adjoint(solver, dim):
        """Run tape backward and verify R_tape.grad against independent solve."""
        tape, loss = _forward_with_tape(solver, dim)

        n = solver.space.node_count()
        vecd = wp.types.vector(length=dim, dtype=wp.float64)

        # ---- Expected R_tape.grad via independent solve ----
        # dL/du = 2*u
        u_np = solver.u.dof_values.numpy().astype(np.float64)  # (N, dim)
        dLdu_np = 2.0 * u_np

        # Mask Dirichlet DOFs (free_mask is 1.0 for free, 0.0 for Dirichlet)
        dirichlet_mask_np = solver.dirichlet_free_mask_wp.numpy()
        dLdu_np *= dirichlet_mask_np

        # Negate: RHS = -masked(dL/du)
        rhs_np = -dLdu_np

        # Convert to warp
        rhs_wp = wp.array(
            np.ascontiguousarray(rhs_np), dtype=vecd, device=solver.device
        )

        # Solve K * x = rhs  (K is already projected from the tape pass)
        K = solver._last_K
        assert K is not None, "_last_K should be set after forward"
        x_expected = solve_linear_system(
            K,
            rhs_wp,
            method="cg",
            device=solver.device,
            dim=dim,
            _cache={},
        )

        # Mask expected result on Dirichlet DOFs (should already be ~0)
        x_np = x_expected.numpy()
        x_np *= dirichlet_mask_np

        # ---- Actual R_tape.grad from tape backward ----
        R_grad_np = solver._tape_residual.grad.numpy()

        # ---- Compare on free DOFs ----
        # Both should be zero on Dirichlet DOFs, non-trivial on free DOFs
        free_mask_flat = dirichlet_mask_np.ravel()
        free_idx = np.where(free_mask_flat > 0.5)[0]
        assert len(free_idx) > 0, "No free DOFs found"

        actual_free = R_grad_np.ravel()[free_idx]
        expected_free = x_np.ravel()[free_idx]

        # Use relative tolerance — values can span several orders of magnitude
        np.testing.assert_allclose(
            actual_free,
            expected_free,
            rtol=1e-3,
            atol=1e-10,
            err_msg="R_tape.grad does not match independent adjoint solve",
        )

    def test_adjoint_correctness_2d(self, tmp_path):
        solver = NeoHookean(
            "tests.examples.two_d_affine_tension.config",
            override_params={"output_root": str(tmp_path)},
        )
        self._check_adjoint(solver, dim=2)

    def test_adjoint_correctness_3d(self, tmp_path):
        solver = NeoHookean(
            "tests.examples.three_d_affine_tension.config",
            override_params={"output_root": str(tmp_path)},
        )
        self._check_adjoint(solver, dim=3)


# ---------------------------------------------------------------------------
# Test 3 — Finite difference verification of compliance sensitivity
# ---------------------------------------------------------------------------
class TestComplianceFD:
    """Verify forward solve sensitivity dL/dE via central finite differences.

    For a linear-ish problem, the compliance C = u^T f is proportional to 1/E.
    Perturbing E should change the solution proportionally.
    """

    def test_compliance_fd_2d(self, tmp_path):
        config = "tests.examples.two_d_right_boundary_force.config"
        E_base = 30.0
        dE = E_base * 1e-4  # small perturbation

        def _run(E_val, subdir):
            out = str(tmp_path / subdir)
            solver = NeoHookean(
                config,
                # This test checks material sensitivity, not an AMG backend.
                # Its severe single-step traction produces an indefinite
                # intermediate tangent that is not a reliable native-f32
                # solver case; keep its matrix-free path explicitly f64.
                override_params={
                    "output_root": out,
                    "E": E_val,
                    "linear_solver": "matrix_free_cg",
                    "use_mixed_precision": True,
                },
            )
            solver.forward()
            u_np = solver.get_displacement()
            # Compliance proxy: sum of squared displacements
            loss_val = float(np.sum(u_np ** 2))
            return loss_val

        loss_plus = _run(E_base + dE, "plus")
        loss_minus = _run(E_base - dE, "minus")
        fd_grad = (loss_plus - loss_minus) / (2.0 * dE)

        # For stiffer material (larger E), displacements decrease → loss decreases
        # So dL/dE should be negative
        assert (
            fd_grad < 0.0
        ), f"Expected negative dL/dE for compliance-like loss, got {fd_grad:.6e}"

        # Verify the magnitude is reasonable (not zero, not wildly large)
        assert abs(fd_grad) > 1e-10, f"FD gradient is essentially zero: {fd_grad:.6e}"
        assert abs(fd_grad) < 1e6, f"FD gradient is unreasonably large: {fd_grad:.6e}"
