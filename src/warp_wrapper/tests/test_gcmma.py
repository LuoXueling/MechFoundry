"""Tests for the GPU-native GCMMA (CCSAMMASolver) implementation.

Includes:
- Analytical 3-variable, 2-constraint test from NLopt reference
- Unconstrained minimization test
- Penalty parameter initialization and decay test
- Conservatism loop smoke test

Run with:
    conda activate mechfoundry-warp117
    pytest src/warp_wrapper/tests/test_gcmma.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.mma_solver import CCSAMMASolver

wp.init()

# Use CUDA device (warp 1.11.x CPU backend has compilation issues)
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Test 1: Analytical 3-variable, 2-constraint problem
# ---------------------------------------------------------------------------


class TestAnalytical3Var2Con:
    """Minimize f0 = x1 + x2 + x3
    subject to:
        f1 = 8/x1 + 1/x2 - 4 <= 0
        f2 = 1/x2 + 2/x3 - 3 <= 0
    bounds: 0.1 <= xi <= 10.0
    """

    @staticmethod
    def _eval(x_np):
        """Evaluate objective and constraints from numpy array."""
        f0 = float(np.sum(x_np))
        f1 = 8.0 / x_np[0] + 1.0 / x_np[1] - 4.0
        f2 = 1.0 / x_np[1] + 2.0 / x_np[2] - 3.0
        return f0, [f1, f2]

    @staticmethod
    def _grad(x_np):
        """Compute gradients analytically."""
        grad_f0 = np.ones(3, dtype=np.float32)
        grad_f1 = np.array(
            [
                -8.0 / x_np[0] ** 2,
                -1.0 / x_np[1] ** 2,
                0.0,
            ],
            dtype=np.float32,
        )
        grad_f2 = np.array(
            [
                0.0,
                -1.0 / x_np[1] ** 2,
                -2.0 / x_np[2] ** 2,
            ],
            dtype=np.float32,
        )
        return grad_f0, [grad_f1, grad_f2]

    def test_convergence_pure_mma(self):
        """Test convergence without conservatism loop (pure MMA mode)."""
        n = 3
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=2,
            xmin=0.1,
            xmax=10.0,
            move_limit=0.2,
            device=DEVICE,
        )

        x_np = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=DEVICE)

        for it in range(80):
            x_np = x.numpy()
            f0, f_i = self._eval(x_np)
            grad_f0_np, grad_f_i_np = self._grad(x_np)

            grad_f0 = wp.array(grad_f0_np, dtype=wp.float32, device=DEVICE)
            grad_f_i = [
                wp.array(g, dtype=wp.float32, device=DEVICE) for g in grad_f_i_np
            ]

            print(f"Iter {it}: f0={f0:.4f}, f1={f_i[0]:.4f}, f2={f_i[1]:.4f}")

            x_new = mma.step(
                x=x,
                f0=f0,
                grad_f0=grad_f0,
                f_i=f_i,
                grad_f_i=grad_f_i,
                eval_func=None,  # pure MMA
            )
            x = wp.clone(x_new)

        x_final = x.numpy()
        f0_final, f_i_final = self._eval(x_final)

        # Check constraints are approximately satisfied
        assert f_i_final[0] <= 0.05, f"Constraint 1 violated: {f_i_final[0]}"
        assert f_i_final[1] <= 0.05, f"Constraint 2 violated: {f_i_final[1]}"

        # Optimal is around f0 ~ 4.57 (KKT solution)
        assert f0_final < 6.0, f"Objective too high: {f0_final}"

    def test_convergence_with_gcmma(self):
        """Test convergence with conservatism loop (GCMMA mode)."""
        n = 3
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=2,
            xmin=0.1,
            xmax=10.0,
            move_limit=1,
            max_inner=15,
            device=DEVICE,
        )

        x_np = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=DEVICE)

        def eval_func(x_trial):
            x_trial_np = x_trial.numpy()
            f0_new, f_i_new = self._eval(x_trial_np)
            print(
                f"  eval_func called with x={x_trial_np}, returns f0={f0_new:.4f}, f_i={f_i_new}"
            )
            return f0_new, f_i_new

        for it in range(80):
            x_np = x.numpy()
            f0, f_i = self._eval(x_np)
            grad_f0_np, grad_f_i_np = self._grad(x_np)

            grad_f0 = wp.array(grad_f0_np, dtype=wp.float32, device=DEVICE)
            grad_f_i = [
                wp.array(g, dtype=wp.float32, device=DEVICE) for g in grad_f_i_np
            ]

            print(f"Iter {it}: f0={f0:.4f}, f1={f_i[0]:.4f}, f2={f_i[1]:.4f}")

            x_new = mma.step(
                x=x,
                f0=f0,
                grad_f0=grad_f0,
                f_i=f_i,
                grad_f_i=grad_f_i,
                eval_func=eval_func,
            )
            x = wp.clone(x_new)

        x_final = x.numpy()
        f0_final, f_i_final = self._eval(x_final)

        # GCMMA should converge better than pure MMA
        assert f_i_final[0] <= 0.05, f"Constraint 1 violated: {f_i_final[0]}"
        assert f_i_final[1] <= 0.05, f"Constraint 2 violated: {f_i_final[1]}"
        assert f0_final < 6.0, f"Objective too high: {f0_final}"


# ---------------------------------------------------------------------------
# Test 2: Unconstrained minimization
# ---------------------------------------------------------------------------


class TestUnconstrained:
    """Minimize f0 = sum((x - target)^2) with no constraints."""

    def test_unconstrained_quadratic(self):
        """Pure MMA on a simple quadratic moves toward the optimum."""
        n = 10
        target = 0.5
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=0,
            xmin=0.0,
            xmax=1.0,
            move_limit=0.2,
            device=DEVICE,
        )

        x_np = np.full(n, 0.9, dtype=np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=DEVICE)

        for it in range(50):
            x_np = x.numpy()
            f0 = float(np.sum((x_np - target) ** 2))
            grad_f0_np = (2.0 * (x_np - target)).astype(np.float32)
            grad_f0 = wp.array(grad_f0_np, dtype=wp.float32, device=DEVICE)

            x_new = mma.step(
                x=x,
                f0=f0,
                grad_f0=grad_f0,
                f_i=None,
                grad_f_i=None,
                eval_func=None,
            )
            x = wp.clone(x_new)

        x_final = x.numpy()
        # Should be close to target=0.5
        assert np.allclose(
            x_final, target, atol=0.05
        ), f"Expected ~{target}, got {x_final}"


# ---------------------------------------------------------------------------
# Test 3: Penalty parameter initialization and decay
# ---------------------------------------------------------------------------


class TestPenaltyParams:
    """Verify penalty parameters behave as expected."""

    def test_penalty_init_from_gradient(self):
        """rho_min should be initialized from max gradient magnitude."""
        n = 5
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=1,
            xmin=0.0,
            xmax=1.0,
            move_limit=0.2,
            device=DEVICE,
        )

        x = wp.array(np.full(n, 0.5, dtype=np.float32), device=DEVICE)
        # Gradient with max magnitude = 100.0
        grad_f0 = wp.array(
            np.array([100.0, -50.0, 10.0, -1.0, 0.5], dtype=np.float32),
            device=DEVICE,
        )
        grad_f1 = wp.array(
            np.array([0.1, 0.2, 0.3, 0.4, 200.0], dtype=np.float32),
            device=DEVICE,
        )

        mma.step(
            x=x,
            f0=1.0,
            grad_f0=grad_f0,
            f_i=[0.5],
            grad_f_i=[grad_f1],
            eval_func=None,
        )

        # rho_min = max(1e-5 * 100.0, 1e-12) = 1e-3
        assert abs(mma.rho_min - 1e-3) < 1e-10, f"rho_min={mma.rho_min}"
        assert abs(mma.rho - mma.rho_min) < 1e-10

        # rhoc_min[0] = max(1e-5 * 200.0, 1e-12) = 2e-3
        assert abs(mma.rhoc_min[0] - 2e-3) < 1e-10, f"rhoc_min={mma.rhoc_min[0]}"
        assert abs(mma.rhoc[0] - mma.rhoc_min[0]) < 1e-10

    def test_penalty_decay(self):
        """Penalty decays by 0.1x per iteration, bounded by rho_min."""
        n = 3
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=0,
            xmin=0.0,
            xmax=1.0,
            move_limit=0.2,
            device=DEVICE,
        )

        x = wp.array(np.full(n, 0.5, dtype=np.float32), device=DEVICE)
        grad_f0 = wp.array(np.ones(n, dtype=np.float32), device=DEVICE)

        # First call initializes rho
        mma.step(x=x, f0=1.0, grad_f0=grad_f0)
        rho_after_init = mma.rho
        rho_min = mma.rho_min

        # After init, rho = rho_min = max(1e-5 * 1.0, 1e-12) = 1e-5
        assert abs(rho_after_init - 1e-5) < 1e-12

        # Artificially increase rho to test decay
        mma.rho = 1.0

        # Second call decays: new_rho = max(0.1 * 1.0, rho_min) = 0.1
        mma.step(x=x, f0=1.0, grad_f0=grad_f0)
        assert abs(mma.rho - 0.1) < 1e-10, f"rho after decay: {mma.rho}"

        # Third call: max(0.1 * 0.1, rho_min) = max(0.01, 1e-5) = 0.01
        mma.step(x=x, f0=1.0, grad_f0=grad_f0)
        assert abs(mma.rho - 0.01) < 1e-10, f"rho after second decay: {mma.rho}"


# ---------------------------------------------------------------------------
# Test 4: Sigma oscillation tracking
# ---------------------------------------------------------------------------


class TestSigmaTracking:
    """Verify sigma adapts based on oscillation behavior."""

    def test_sigma_initialization(self):
        """First two iterations should use sigma_init * range."""
        n = 2
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=0,
            xmin=0.0,
            xmax=1.0,
            move_limit=0.2,
            device=DEVICE,
        )

        x = wp.array(np.array([0.5, 0.5], dtype=np.float32), device=DEVICE)
        grad = wp.array(np.array([1.0, -1.0], dtype=np.float32), device=DEVICE)

        mma.step(x=x, f0=1.0, grad_f0=grad)

        sigma_np = mma.sigma.numpy()
        expected = (0.2 / 0.9) * (1.0 - 0.0)  # move/0.9 * range
        assert np.allclose(
            sigma_np, expected, atol=1e-5
        ), f"sigma={sigma_np}, expected={expected}"


# ---------------------------------------------------------------------------
# Test 5: Conservatism loop increases penalties
# ---------------------------------------------------------------------------


class TestConservatismLoop:
    """Verify inner loop increases penalties when approximation is not conservative."""

    def test_penalty_increase_on_non_conservative(self):
        """When eval_func returns values larger than the approximation,
        rho should increase."""
        n = 3
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=1,
            xmin=0.1,
            xmax=10.0,
            move_limit=0.3,
            max_inner=3,
            device=DEVICE,
        )

        x_np = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=DEVICE)
        grad_f0 = wp.array(np.ones(n, dtype=np.float32), device=DEVICE)
        grad_f1 = wp.array(
            np.array([-0.32, -0.04, 0.0], dtype=np.float32), device=DEVICE
        )

        # eval_func that always returns values much larger than approximation
        # to force penalty increase
        call_count = [0]

        def eval_func(x_trial):
            call_count[0] += 1
            return 1e6, [1e6]  # way larger than any approximation

        rho_before = mma.rho
        mma.step(
            x=x,
            f0=15.0,
            grad_f0=grad_f0,
            f_i=[-0.2],
            grad_f_i=[grad_f1],
            eval_func=eval_func,
        )
        rho_after = mma.rho

        # Conservatism loop should have run and increased rho
        assert call_count[0] > 0, "eval_func was never called"
        assert (
            rho_after > rho_before
        ), f"rho should have increased: before={rho_before}, after={rho_after}"


# ---------------------------------------------------------------------------
# Test 6: Dynamic buffer allocation for constraints
# ---------------------------------------------------------------------------


class TestDynamicBuffers:
    """Verify constraint buffers grow dynamically."""

    def test_ensure_constraint_buffers(self):
        """Passing more constraints than initially allocated should work."""
        n = 5
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=0,
            xmin=0.0,
            xmax=1.0,
            move_limit=0.2,
            device=DEVICE,
        )

        assert len(mma.Pi) == 0

        x = wp.array(np.full(n, 0.5, dtype=np.float32), device=DEVICE)
        grad = wp.array(np.ones(n, dtype=np.float32), device=DEVICE)
        grad_c = wp.array(np.ones(n, dtype=np.float32) * 0.1, device=DEVICE)

        # Now call with 2 constraints despite init with 0
        x_new = mma.step(
            x=x,
            f0=1.0,
            grad_f0=grad,
            f_i=[0.5, -0.3],
            grad_f_i=[grad_c, grad_c],
        )

        # Buffers should have grown
        assert len(mma.Pi) >= 2
        assert len(mma.Qi) >= 2
        assert len(mma.vi) >= 2

        # Should return valid result
        x_np = x_new.numpy()
        assert all(np.isfinite(x_np)), f"Non-finite result: {x_np}"


# ---------------------------------------------------------------------------
# Test 7: GPU dual path produces same results as CPU path
# ---------------------------------------------------------------------------


class TestGPUDual:
    """Verify gpu_dual=True produces identical results to the CPU path."""

    @staticmethod
    def _eval_3var(x_np):
        f0 = float(np.sum(x_np))
        f1 = 8.0 / x_np[0] + 1.0 / x_np[1] - 4.0
        f2 = 1.0 / x_np[1] + 2.0 / x_np[2] - 3.0
        return f0, [f1, f2]

    @staticmethod
    def _grad_3var(x_np):
        grad_f0 = np.ones(3, dtype=np.float32)
        grad_f1 = np.array(
            [-8.0 / x_np[0] ** 2, -1.0 / x_np[1] ** 2, 0.0], dtype=np.float32
        )
        grad_f2 = np.array(
            [0.0, -1.0 / x_np[1] ** 2, -2.0 / x_np[2] ** 2], dtype=np.float32
        )
        return grad_f0, [grad_f1, grad_f2]

    def test_gpu_dual_matches_cpu_constrained(self):
        """GPU dual solve matches CPU dual solve on the 3-var problem."""
        n = 3
        x0_np = np.array([5.0, 5.0, 5.0], dtype=np.float32)

        results = {}
        for label, gpu_dual in [("cpu", False), ("gpu", True)]:
            mma = CCSAMMASolver(
                n_vars=n,
                n_constraints=2,
                xmin=0.1,
                xmax=10.0,
                move_limit=0.2,
                device=DEVICE,
                gpu_dual=gpu_dual,
            )
            x = wp.array(x0_np.copy(), dtype=wp.float32, device=DEVICE)

            for it in range(30):
                x_np = x.numpy()
                f0, f_i = self._eval_3var(x_np)
                gf0, gfi = self._grad_3var(x_np)
                grad_f0 = wp.array(gf0, dtype=wp.float32, device=DEVICE)
                grad_f_i = [wp.array(g, dtype=wp.float32, device=DEVICE) for g in gfi]
                x_new = mma.step(
                    x=x,
                    f0=f0,
                    grad_f0=grad_f0,
                    f_i=f_i,
                    grad_f_i=grad_f_i,
                    eval_func=None,
                )
                x = wp.clone(x_new)

            results[label] = x.numpy()

        np.testing.assert_allclose(
            results["gpu"],
            results["cpu"],
            atol=1e-4,
            err_msg="GPU dual and CPU dual diverged on constrained problem",
        )

    def test_gpu_dual_unconstrained(self):
        """GPU dual path works correctly for unconstrained problems (m=0)."""
        n = 10
        target = 0.5
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=0,
            xmin=0.0,
            xmax=1.0,
            move_limit=0.2,
            device=DEVICE,
            gpu_dual=True,
        )

        x = wp.array(np.full(n, 0.9, dtype=np.float32), device=DEVICE)
        for it in range(50):
            x_np = x.numpy()
            f0 = float(np.sum((x_np - target) ** 2))
            grad_f0 = wp.array(
                (2.0 * (x_np - target)).astype(np.float32),
                dtype=wp.float32,
                device=DEVICE,
            )
            x_new = mma.step(x=x, f0=f0, grad_f0=grad_f0)
            x = wp.clone(x_new)

        x_final = x.numpy()
        assert np.allclose(
            x_final, target, atol=0.05
        ), f"Expected ~{target}, got {x_final}"

    def test_gpu_dual_with_gcmma(self):
        """GPU dual path works with conservatism loop (GCMMA)."""
        n = 3
        mma = CCSAMMASolver(
            n_vars=n,
            n_constraints=2,
            xmin=0.1,
            xmax=10.0,
            move_limit=1,
            max_inner=15,
            device=DEVICE,
            gpu_dual=True,
        )

        x_np = np.array([5.0, 5.0, 5.0], dtype=np.float32)
        x = wp.array(x_np, dtype=wp.float32, device=DEVICE)

        def eval_func(x_trial):
            xnp = x_trial.numpy()
            return self._eval_3var(xnp)

        for it in range(80):
            x_np = x.numpy()
            f0, f_i = self._eval_3var(x_np)
            gf0, gfi = self._grad_3var(x_np)
            grad_f0 = wp.array(gf0, dtype=wp.float32, device=DEVICE)
            grad_f_i = [wp.array(g, dtype=wp.float32, device=DEVICE) for g in gfi]
            x_new = mma.step(
                x=x,
                f0=f0,
                grad_f0=grad_f0,
                f_i=f_i,
                grad_f_i=grad_f_i,
                eval_func=eval_func,
            )
            x = wp.clone(x_new)

        x_final = x.numpy()
        f0_final, f_i_final = self._eval_3var(x_final)
        assert f_i_final[0] <= 0.05, f"Constraint 1 violated: {f_i_final[0]}"
        assert f_i_final[1] <= 0.05, f"Constraint 2 violated: {f_i_final[1]}"
        assert f0_final < 6.0, f"Objective too high: {f0_final}"


if __name__ == "__main__":
    solver = TestAnalytical3Var2Con()
    # solver.test_convergence_pure_mma()
    solver.test_convergence_with_gcmma()
