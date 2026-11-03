"""Tests for the phase_field design-variable mode of ShapeOptTrimmed.

Minimal design: SDF stays in ``e.dof_values`` and the forward Nitsche path
is identical to ``design_variable="sdf"``.  PF mode adds a nodal φ tape
field and a persistent hidden φ state that:
  - pass unit φ to the interior residual, preserving equilibrium while
    exposing its implicit/adjoint derivative,
  - tracks the freshly GD-updated SDF at boundary nodes (|SDF| < band·ε),
  - receives the chain-rule φ-gradient (∂J/∂SDF · dSDF/dφ) at interior nodes,
  - is inverted (atanh) and used as the bg-grid scatter source in
    ``_post_optimizer_step_remesh_impl`` so MC + winding-number reinit can
    reveal interior voids carved by the φ-gradient.

Test coverage:
1. φ↔SDF roundtrip (pure math).
2. Iter-0 forward is identical between SDF and PF modes (forward unchanged).
3. PF-mode hidden φ is initialized to ``tanh(SDF/ε)`` and tracks the SDF
   at boundary nodes after one iteration.
4. Selective reprojection band-mask semantics (pure math).
5. Internal void & through-hole identification by warp's MC + winding number.
6. Internal void & through-hole increase strain energy under Nitsche
   (verifies the SDF pipeline correctly reflects the geometry change).
7. PF-mode regression: a few iterations decrease the objective and stay
   close to SDF mode (since forward path is unchanged).

Run:
    conda activate mechfoundry-warp117
    pytest src/warp_wrapper/tests/test_phase_field_design.py -v
"""

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import warp as wp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_octree_mesh import make_uniform_hex_mesh

wp.init()

# Reuse the beam-SDF helper from the mixed-precision suite.
from tests.test_nitsche_mixed import _make_beam_sdf  # noqa: E402


CONFIG = "configs.config_shape_opt_cantilever"
BEAM_LO = (-0.75, -0.25, -0.25)
BEAM_HI = (0.75, 0.25, 0.25)


def _common_overrides(output_dir, design_variable="sdf", **extra):
    return {
        "phase_field_epsilon": None,  # set by caller
        "use_mixed_precision": True,
        "reinit_sdf": False,
        "max_iter": 20,
        "relative_tol": 1e-6,
        "absolute_tol": 1e-8,
        "line_search_enabled": False,
        "load_steps": 1,
        "linear_solver": "cg",
        "cg_backend_assembled": "amgx",
        "ew_enabled": False,
        "target_volume_fraction": 0.5,
        "shape_opt_w_vol": 10.0,
        "optimizer": "gd",
        "gd_lr": 1e-3,
        "output_per_step": 0,
        "output_root": output_dir,
        "design_variable": design_variable,
        "dirichlet_bcs_dict": {
            lambda x, y, z: torch.isclose(x, torch.full_like(x, BEAM_LO[0])): 0.0,
        },
        "neumann_bcs_dict": {
            lambda x, y, z: torch.isclose(x, torch.full_like(x, BEAM_HI[0])): (
                0.0,
                -0.001,
                0.0,
            ),
        },
        **extra,
    }


def _make_problem_inputs(nx=16, ny=8, nz=8, pad=0.25):
    lo = [BEAM_LO[i] - pad for i in range(3)]
    hi = [BEAM_HI[i] + pad for i in range(3)]
    nodes, elements = make_uniform_hex_mesh(nx, ny, nz, lo, hi)
    h_min = min((hi[i] - lo[i]) / [nx, ny, nz][i] for i in range(3))
    epsilon = 0.5 * h_min
    sdf = _make_beam_sdf(nodes, BEAM_LO, BEAM_HI)
    return nodes, elements, sdf, epsilon


def _octree_mesh_dict(nodes, elements, sdf):
    return {
        "nodes": nodes.copy() if isinstance(nodes, np.ndarray) else nodes.clone(),
        "elements": (
            elements.copy() if isinstance(elements, np.ndarray) else elements.clone()
        ),
        "sdf": sdf.clone() if isinstance(sdf, torch.Tensor) else torch.tensor(sdf).clone(),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPhiSdfRoundtrip:
    """Pure-math roundtrip of the φ ↔ SDF helpers — no FEM."""

    def test_roundtrip(self):
        from core.shape_opt import _phi_to_sdf, _sdf_to_phi

        torch.manual_seed(0)
        phi = torch.rand(500).clamp(1e-4, 1.0 - 1e-4).double()
        eps = 0.01
        d = _phi_to_sdf(phi, eps)
        phi_back = _sdf_to_phi(d, eps)
        err = (phi_back - phi).abs().max().item()
        assert err < 1e-8, f"Roundtrip error {err:.3e} exceeds 1e-8"


class TestForwardIdentical:
    """PF uses a unit-valued shadow φ in the interior residual.  Therefore
    raw SE, raw volume, and u remain identical between modes (up to AMGX noise).
    """

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_iter0_forward_identical(self):
        from core.shape_opt import ShapeOptTrimmed

        nodes, elements, sdf, epsilon = _make_problem_inputs()

        def _make(mode, sub):
            ov = _common_overrides(self.tmp + sub, design_variable=mode)
            ov["phase_field_epsilon"] = epsilon
            ov["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)
            return ShapeOptTrimmed(CONFIG, override_params=ov)

        s_sdf = _make("sdf", "/sdf")
        s_pf = _make("phase_field", "/pf")

        tape_s = wp.Tape()
        s_sdf.e.dof_values.requires_grad = True
        s_sdf.step(tape=tape_s)
        with tape_s:
            s_sdf._objective_function(tape_s)

        tape_p = wp.Tape()
        s_pf.e.dof_values.requires_grad = True
        s_pf.step(tape=tape_p)
        residual_phi = wp.to_torch(s_pf._inner.e_phi.dof_values).detach()
        torch.testing.assert_close(residual_phi, torch.ones_like(residual_phi))
        with tape_p:
            s_pf._objective_function(tape_p)

        # Tight numerical match. Hidden-phi evaluates the standard interior at
        # exact phi=1 rather than the SDF projection's saturated near-one value.
        assert s_sdf._last_se_raw == pytest.approx(
            s_pf._inner._last_se_raw, rel=1e-10
        )
        assert s_sdf._last_vol_raw == pytest.approx(
            s_pf._inner._last_vol_raw, rel=1e-8
        )
        u_s = wp.to_torch(s_sdf._inner.u.dof_values).detach().cpu().numpy()
        u_p = wp.to_torch(s_pf._inner.u.dof_values).detach().cpu().numpy()
        rel = np.linalg.norm(u_s - u_p) / max(np.linalg.norm(u_s), 1e-30)
        # AMGX may produce 1e-9 noise; anything below 1e-6 is effectively identical.
        assert rel < 1e-6, f"Displacement diverged: rel={rel:.3e}"


class TestBodyForceReinitLifecycle:
    """A nonzero design-weighted body cache must follow every inner remesh."""

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    @pytest.mark.parametrize("mode", ["sdf", "phase_field"])
    def test_two_steps_rebuild_body_cache_on_topology_change(self, mode):
        from core.shape_opt import ShapeOptTrimmed

        nodes, elements, sdf, epsilon = _make_problem_inputs()
        overrides = _common_overrides(
            self.tmp + "/" + mode,
            design_variable=mode,
            B=(0.0, -0.01, 0.0),
            warm_start_u=False,
        )
        overrides["phase_field_epsilon"] = epsilon
        overrides["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)
        solver = ShapeOptTrimmed(CONFIG, override_params=overrides)

        # First step creates the trimmed inner instance and its initial cache.
        solver.step()
        old_inner = solver._inner
        old_body = old_inner._mesh_cache["body"]
        old_n_elems = int(old_inner._prev_octree_elements.shape[0])
        assert old_body is not None

        # Move the free end inward by two element layers so the second step's
        # in-place inner reinitialization has guaranteed different topology,
        # while the left clamp and body-loaded cantilever remain well posed.
        shortened_sdf = _make_beam_sdf(
            nodes,
            BEAM_LO,
            (BEAM_HI[0] - 0.25, BEAM_HI[1], BEAM_HI[2]),
        )
        solver._copy_design_tensor(shortened_sdf, solver.e.dof_values)
        solver.step()

        inner = solver._inner
        mc = inner._mesh_cache
        assert inner is old_inner  # exercise reinit_with_sdf(), not replacement
        assert int(inner._prev_octree_elements.shape[0]) != old_n_elems
        assert mc["body"] is not None
        assert mc["body"] is not old_body
        assert mc["body"].shape[0] == inner.space.node_count()

        if mode == "phase_field":
            phi = wp.to_torch(inner.e_phi.dof_values).detach()
            torch.testing.assert_close(phi, torch.ones_like(phi))

        # The cache assembled after all current-mesh fields are ready must
        # equal a fresh assembly, and both residual entry points must agree.
        fresh_body = inner._assemble_body_forces(mc["domain"], mc["test"])
        torch.testing.assert_close(
            wp.to_torch(mc["body"]),
            wp.to_torch(fresh_body),
            rtol=1e-10,
            atol=1e-12,
        )
        residual_cached = inner._evaluate_residual(
            mc["domain"],
            mc["test"],
            mc["boundary"],
            mc["bd_test"],
            neumann_load=mc["neumann"],
            body_force=mc["body"],
        )
        residual_fresh = inner._evaluate_residual(
            mc["domain"],
            mc["test"],
            mc["boundary"],
            mc["bd_test"],
            neumann_load=mc["neumann"],
            body_force=None,
        )
        torch.testing.assert_close(
            wp.to_torch(residual_cached),
            wp.to_torch(residual_fresh),
            rtol=1e-10,
            atol=1e-12,
        )


class TestShadowAndSDFSplit:
    """Verify the gradient split that enables interior carving without
    competing with SDF evolution:
      - ∂J/∂SDF at interior is zeroed (SDF drives boundary only).
      - ∂J/∂e_phi at interior is non-zero (shadow drives interior).
    """

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_sdf_grad_interior_zero_and_shadow_grad_interior_nonzero(self):
        from core.shape_opt import ShapeOptTrimmed

        nodes, elements, sdf, epsilon = _make_problem_inputs()
        overrides = _common_overrides(
            self.tmp, design_variable="phase_field", phi_reproject_band_k=3.0
        )
        overrides["phase_field_epsilon"] = epsilon
        overrides["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)
        # Heavier load so the gradient signal is above noise.
        overrides["neumann_bcs_dict"] = {
            lambda x, y, z: torch.isclose(
                x, torch.full_like(x, BEAM_HI[0])
            ): (0.0, -0.05, 0.0),
        }
        overrides["shape_opt_w_vol"] = 10.0

        s = ShapeOptTrimmed(CONFIG, override_params=overrides)

        tape = wp.Tape()
        s.e.dof_values.requires_grad = True
        s.step(tape=tape)
        with tape:
            obj = s._objective_function(tape)
        tape.zero()
        if s.e.dof_values.grad is not None:
            s.e.dof_values.grad.zero_()
        tape.backward(loss=obj)

        # The unit-valued interior residual φ must itself receive a tape
        # gradient.  This is the implicit/adjoint term; the same e_phi field
        # also receives the direct objective term.
        residual_phi_grad = wp.to_torch(s._inner.e_phi.dof_values.grad).detach().cpu()
        inner_sdf = wp.to_torch(s._inner.e.dof_values).detach().cpu()
        inner_deep = inner_sdf < -3.0 * epsilon
        assert int(inner_deep.sum()) >= 3
        assert residual_phi_grad[inner_deep].abs().max().item() > 1e-14

        s._post_tape_backward()

        sdf_t = wp.to_torch(s.e.dof_values).detach().cpu().to(torch.float64)
        g_sdf = s._last_e_grad.cpu()
        g_phi = s._last_e_phi_grad.cpu() if hasattr(s, "_last_e_phi_grad") else None

        deep_interior = sdf_t < -3.0 * epsilon
        boundary = sdf_t.abs() < 3.0 * epsilon
        assert int(deep_interior.sum()) >= 3
        assert int(boundary.sum()) >= 5

        # SDF gradient: zero at deep interior (zero-out in _post_tape_backward),
        # non-zero at boundary band (from primary backward).
        assert g_sdf[deep_interior].abs().max().item() == 0.0, (
            "∂J/∂SDF at deep interior must be zeroed in PF mode"
        )
        assert g_sdf[boundary].abs().max().item() > 1e-10, (
            "∂J/∂SDF at boundary must still drive the SDF evolution"
        )

        # Shadow φ gradient: non-zero at deep interior (that's the whole
        # point!).  Absolute magnitude depends on mesh size, load, W weights
        # — we just check it's meaningfully above float noise.  The primary
        # SDF gradient saturates to 0 at deep interior via chain rule; the
        # shadow doesn't (φ appears linearly in the integrand).
        assert g_phi is not None
        g_phi_deep_max = g_phi[deep_interior].abs().max().item()
        assert g_phi_deep_max > 1e-14, (
            f"Shadow ∂J/∂e_phi at deep interior is noise: {g_phi_deep_max:.3e}"
        )
        print(
            f"Interior split: g_sdf[deep]=0 (zeroed), "
            f"g_phi[deep] max={g_phi_deep_max:.3e}, "
            f"g_sdf[boundary] max={g_sdf[boundary].abs().max():.3e}"
        )

    def test_residual_phi_adjoint_preserves_forward_and_sdf_gradient(self):
        """Unit residual phi changes only the hidden-phi sensitivity.

        Disable the volume term so every gradient in this comparison comes
        from strain energy.  SDF mode supplies the direct SE derivative;
        phase-field mode additionally supplies the interior residual/adjoint
        derivative through the same e_phi field.
        """
        from core.shape_opt import ShapeOptTrimmed

        nodes, elements, sdf, epsilon = _make_problem_inputs()

        def _run(mode, subdir):
            overrides = _common_overrides(
                self.tmp + subdir,
                design_variable=mode,
                shape_opt_w_vol=0.0,
            )
            overrides["phase_field_epsilon"] = epsilon
            overrides["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)
            overrides["neumann_bcs_dict"] = {
                lambda x, y, z: torch.isclose(
                    x, torch.full_like(x, BEAM_HI[0])
                ): (0.0, -0.05, 0.0),
            }
            solver = ShapeOptTrimmed(CONFIG, override_params=overrides)
            tape = wp.Tape()
            solver.e.dof_values.requires_grad = True
            solver.step(tape=tape)
            u = wp.to_torch(solver._inner.u.dof_values).detach().cpu().clone()
            with tape:
                objective = solver._objective_function(tape)
            tape.zero()
            if solver.e.dof_values.grad is not None:
                solver.e.dof_values.grad.zero_()
            tape.backward(loss=objective)
            inner = solver._inner
            return {
                "u": u,
                "sdf": wp.to_torch(inner.e.dof_values).detach().cpu().clone(),
                "sdf_grad": wp.to_torch(inner.e.dof_values.grad).detach().cpu().clone(),
                "phi_grad": wp.to_torch(inner.e_phi.dof_values.grad)
                .detach()
                .cpu()
                .clone(),
            }

        sdf_run = _run("sdf", "/sdf_se_only")
        phi_run = _run("phase_field", "/phi_se_only")

        u_rel = torch.linalg.vector_norm(
            phi_run["u"] - sdf_run["u"]
        ) / torch.linalg.vector_norm(sdf_run["u"]).clamp_min(1e-30)
        sdf_grad_rel = torch.linalg.vector_norm(
            phi_run["sdf_grad"] - sdf_run["sdf_grad"]
        ) / torch.linalg.vector_norm(sdf_run["sdf_grad"]).clamp_min(1e-30)

        # The unit phi multiplier is exactly one, so equilibrium is unchanged.
        assert float(u_rel) < 1e-6
        # Its SDF-chain contribution is suppressed by saturated dphi/dSDF in
        # the non-boundary interior elements where this residual is integrated.
        assert float(sdf_grad_rel) < 1e-6

        deep = phi_run["sdf"] < -3.0 * epsilon
        assert int(deep.sum()) >= 3
        adjoint_phi = phi_run["phi_grad"] - sdf_run["phi_grad"]
        adjoint_deep_max = adjoint_phi[deep].abs().max().item()
        direct_deep_max = sdf_run["phi_grad"][deep].abs().max().item()
        assert adjoint_deep_max > 1e-6 * max(direct_deep_max, 1e-30)

        print(
            f"SE-only shared-phi check: u_rel={float(u_rel):.3e}, "
            f"sdf_grad_rel={float(sdf_grad_rel):.3e}, "
            f"direct_phi_deep_max={direct_deep_max:.3e}, "
            f"adjoint_phi_deep_max={adjoint_deep_max:.3e}"
        )


class TestSelectiveReprojection:
    """Unit test for the band-mask formula (pure math)."""

    def test_band_mask(self):
        from core.shape_opt import _sdf_to_phi

        eps = 0.05
        band_k = 3.0
        d = torch.tensor(
            [-5.0 * eps, -2.0 * eps, 0.0, 1.5 * eps, 4.0 * eps], dtype=torch.float64
        )
        phi_current = torch.tensor(
            [0.42, 0.31, 0.77, 0.11, 0.02], dtype=torch.float64
        )
        phi_reproj = _sdf_to_phi(d, eps)
        mask = d > -band_k * eps
        phi_new = torch.where(mask, phi_reproj, phi_current)
        # Deep interior preserved.
        assert phi_new[0].item() == phi_current[0].item()
        # Everyone else overwritten.
        for i in range(1, 5):
            assert phi_new[i].item() == pytest.approx(phi_reproj[i].item(), abs=1e-12)


class TestVoidIdentification:
    """Verify wp.MarchingCubes + wp.mesh_query_point_sign_winding_number can
    identify both internal voids (cavities) and through-holes from a φ field.
    """

    def _build_phi_grid_with_void(self, N=64, void_center=(0.0, 0.0, 0.0), void_r=0.15):
        eps = 0.04
        coord = torch.linspace(-1.0, 1.0, N, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(coord, coord, coord, indexing="ij")
        bx = torch.maximum(torch.tensor(-0.75) - X, X - torch.tensor(0.75))
        by = torch.maximum(torch.tensor(-0.25) - Y, Y - torch.tensor(0.25))
        bz = torch.maximum(torch.tensor(-0.25) - Z, Z - torch.tensor(0.25))
        beam_outside = torch.sqrt(
            torch.clamp(bx, min=0.0) ** 2
            + torch.clamp(by, min=0.0) ** 2
            + torch.clamp(bz, min=0.0) ** 2
        )
        beam_inside = torch.minimum(
            torch.maximum(bx, torch.maximum(by, bz)), torch.zeros_like(bx)
        )
        d_beam = beam_outside + beam_inside
        cx, cy, cz = void_center
        d_sphere = torch.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) - void_r
        d_with_void = torch.maximum(d_beam, -d_sphere)
        phi = (0.5 * (1.0 - torch.tanh(d_with_void / eps))).clamp(0.0, 1.0)
        return phi.cuda(), d_with_void.cuda(), eps

    def _build_phi_grid_with_hole(self, N=64, hole_axis_z=0.0, hole_r=0.12):
        eps = 0.04
        coord = torch.linspace(-1.0, 1.0, N, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(coord, coord, coord, indexing="ij")
        bx = torch.maximum(torch.tensor(-0.75) - X, X - torch.tensor(0.75))
        by = torch.maximum(torch.tensor(-0.25) - Y, Y - torch.tensor(0.25))
        bz = torch.maximum(torch.tensor(-0.25) - Z, Z - torch.tensor(0.25))
        beam_outside = torch.sqrt(
            torch.clamp(bx, min=0.0) ** 2
            + torch.clamp(by, min=0.0) ** 2
            + torch.clamp(bz, min=0.0) ** 2
        )
        beam_inside = torch.minimum(
            torch.maximum(bx, torch.maximum(by, bz)), torch.zeros_like(bx)
        )
        d_beam = beam_outside + beam_inside
        d_cyl = torch.sqrt(Y ** 2 + (Z - hole_axis_z) ** 2) - hole_r
        d_with_hole = torch.maximum(d_beam, -d_cyl)
        phi = (0.5 * (1.0 - torch.tanh(d_with_hole / eps))).clamp(0.0, 1.0)
        return phi.cuda(), d_with_hole.cuda(), eps

    def _query_winding_sdf(self, phi_grid, query_pts):
        from core.analytical_sdf import (
            extract_surface,
            build_warp_mesh,
            compute_nodal_sdf,
        )

        mc_field = (0.5 - phi_grid).contiguous()
        verts, faces, probe = extract_surface(mc_field, domain_length=2.0)
        bbox_diag = float((verts.max(dim=0).values - verts.min(dim=0).values).norm().item())
        wp_mesh, negate, _keep = build_warp_mesh(
            verts, faces, probe, bbox_diag=bbox_diag
        )
        sdf_q = compute_nodal_sdf(
            wp_mesh, query_pts.cuda().float(), negate, max_dist=10.0 * bbox_diag
        )
        return sdf_q.cpu()

    def test_internal_void_classified_as_exterior(self):
        phi, _, _ = self._build_phi_grid_with_void()
        query_pts = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.9, 0.0, 0.0]]
        )
        sdf_q = self._query_winding_sdf(phi, query_pts)
        print(
            f"Void-test SDF: void_center={sdf_q[0]:+.4f}, "
            f"body_interior={sdf_q[1]:+.4f}, body_exterior={sdf_q[2]:+.4f}"
        )
        assert sdf_q[0] > 0.05
        assert sdf_q[1] < -0.02
        assert sdf_q[2] > 0.05

    def test_through_hole_classified_as_exterior(self):
        phi, _, _ = self._build_phi_grid_with_hole()
        query_pts = torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.18, 0.0], [0.0, 0.0, 0.4]]
        )
        sdf_q = self._query_winding_sdf(phi, query_pts)
        print(
            f"Hole-test SDF: hole_axis={sdf_q[0]:+.4f}, "
            f"body_interior={sdf_q[1]:+.4f}, body_exterior={sdf_q[2]:+.4f}"
        )
        assert sdf_q[0] > 0.05
        assert sdf_q[1] < -0.02
        assert sdf_q[2] > 0.05


class TestVoidStrainEnergyImpact:
    """Verify Nitsche correctly reflects voids/through-holes in compliance.

    Mode-agnostic: uses ``design_variable="sdf"`` for both intact and carved
    setups since the forward path is identical between modes.  The carved
    geometry must (a) shrink the inner trimmed mesh and (b) increase SE.
    """

    NX, NY, NZ = 48, 24, 24
    PAD = 0.25

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def _make_sdfs(self, nodes, void_kind):
        nodes_t = torch.as_tensor(nodes, dtype=torch.float32)
        x, y, z = nodes_t[:, 0], nodes_t[:, 1], nodes_t[:, 2]
        d_intact = _make_beam_sdf(nodes, BEAM_LO, BEAM_HI)
        if void_kind == "sphere":
            cx, cy, cz = -0.4, 0.0, 0.0
            d_inner = torch.sqrt(
                (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
            ) - 0.18
        elif void_kind == "cylinder":
            d_inner = torch.sqrt((x + 0.4) ** 2 + z ** 2) - 0.18
        else:
            raise ValueError(void_kind)
        d_carved = torch.maximum(d_intact, -d_inner)
        return d_intact, d_carved

    def _se_for_sdf(self, sdf_tensor, suffix):
        from core.shape_opt import ShapeOptTrimmed

        lo = [BEAM_LO[i] - self.PAD for i in range(3)]
        hi = [BEAM_HI[i] + self.PAD for i in range(3)]
        nodes, elements = make_uniform_hex_mesh(
            self.NX, self.NY, self.NZ, lo, hi
        )
        h_min = min(
            (hi[i] - lo[i]) / [self.NX, self.NY, self.NZ][i] for i in range(3)
        )
        epsilon = 0.5 * h_min

        overrides = _common_overrides(self.tmp + suffix, design_variable="sdf")
        overrides["phase_field_epsilon"] = epsilon
        overrides["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf_tensor)
        overrides["neumann_bcs_dict"] = {
            lambda x, y, z: torch.isclose(
                x, torch.full_like(x, BEAM_HI[0])
            ): (0.0, -0.05, 0.0),
        }
        s = ShapeOptTrimmed(CONFIG, override_params=overrides)
        tape = wp.Tape()
        s.e.dof_values.requires_grad = True
        s.step(tape=tape)
        with tape:
            s._objective_function(tape)
        n_active = int(s._trim_active_indices.shape[0])
        return s._last_se_raw, n_active

    def _nodes(self):
        lo = [BEAM_LO[i] - self.PAD for i in range(3)]
        hi = [BEAM_HI[i] + self.PAD for i in range(3)]
        nodes, _ = make_uniform_hex_mesh(self.NX, self.NY, self.NZ, lo, hi)
        return nodes

    def test_internal_void_increases_se(self):
        nodes = self._nodes()
        d_intact, d_void = self._make_sdfs(nodes, "sphere")
        se_intact, n_intact = self._se_for_sdf(d_intact, "/intact_sphere")
        se_void, n_void = self._se_for_sdf(d_void, "/with_void")
        print(
            f"Spherical-void SE: intact={se_intact:.6e} ({n_intact} active), "
            f"with_void={se_void:.6e} ({n_void} active), "
            f"SE ratio={se_void/se_intact:.4f}"
        )
        assert n_void < n_intact, (
            f"Inner mesh did not shrink: {n_intact} → {n_void}"
        )
        assert se_void > se_intact

    def test_through_hole_increases_se(self):
        nodes = self._nodes()
        d_intact, d_hole = self._make_sdfs(nodes, "cylinder")
        se_intact, n_intact = self._se_for_sdf(d_intact, "/intact_cyl")
        se_hole, n_hole = self._se_for_sdf(d_hole, "/with_hole")
        print(
            f"Through-hole SE: intact={se_intact:.6e} ({n_intact} active), "
            f"with_hole={se_hole:.6e} ({n_hole} active), "
            f"SE ratio={se_hole/se_intact:.4f}"
        )
        assert n_hole < n_intact, (
            f"Inner mesh did not shrink: {n_intact} → {n_hole}"
        )
        assert se_hole > se_intact


class TestPFEPhiTracking:
    """The PF-mode ``e_phi`` field: allocated alongside ``e``, fed by the
    φ-direct standard integrand on tape, and updated each iter so:
      (a) interior nodes accumulate ∂J/∂e_phi-driven dips across iterations
          (this signal is non-zero at deep interior — the SDF gradient
          saturates to ≈0 there); and
      (b) boundary-band nodes track ``tanh(SDF/ε)`` consistently with SDF.
    """

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_e_phi_allocated_and_initialised(self):
        from core.shape_opt import ShapeOptTrimmed, _sdf_to_phi

        nodes, elements, sdf, epsilon = _make_problem_inputs()
        overrides = _common_overrides(self.tmp, design_variable="phase_field")
        overrides["phase_field_epsilon"] = epsilon
        overrides["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)

        s = ShapeOptTrimmed(CONFIG, override_params=overrides)
        assert s._design_variable == "phase_field"
        assert s.e_phi is not None
        # At init e_phi = tanh(SDF/ε).  Tolerance allows for the float32
        # _phi_nodal that _build_mesh_state stores before assigning the field.
        e_phi_init = wp.to_torch(s.e_phi.dof_values).detach().cpu()
        sdf_init = wp.to_torch(s.e.dof_values).detach().cpu()
        expected = _sdf_to_phi(sdf_init.to(torch.float64), epsilon)
        err = (e_phi_init.to(torch.float64) - expected).abs().max().item()
        assert err < 1e-6, f"e_phi init mismatch {err:.3e}"

    def test_hidden_phi_boundary_tracks_interior_dips_persist(self):
        from core.shape_opt import ShapeOptTrimmed, _sdf_to_phi

        nodes, elements, sdf, epsilon = _make_problem_inputs()
        overrides = _common_overrides(
            self.tmp,
            design_variable="phase_field",
            phi_reproject_band_k=3.0,
            use_mixed_precision=False,
        )
        overrides["phase_field_epsilon"] = epsilon
        overrides["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)

        s = ShapeOptTrimmed(CONFIG, override_params=overrides)
        sdf_t = wp.to_torch(s.e.dof_values).detach()

        # First call: lazy-initializes _hidden_phi from current SDF.
        s._pf_update_hidden_phi()
        assert s._hidden_phi is not None and s._hidden_phi.shape[0] == sdf_t.shape[0]
        assert s._hidden_phi.dtype == torch.float32

        # Carve a deep-interior dip into the persistent _hidden_phi.
        deep = sdf_t < -3.0 * epsilon
        n_deep = int(deep.sum())
        assert n_deep >= 3, f"Need ≥3 deep-interior nodes, got {n_deep}"
        dip_idx = torch.where(deep)[0][: min(10, n_deep)]
        s._hidden_phi[dip_idx] = 0.4

        # Stub zero φ-gradient: only the boundary-tracking branch should run.
        s._last_e_phi_grad = torch.zeros_like(sdf_t)
        s._pf_update_hidden_phi()

        # Boundary band of _hidden_phi tracks tanh(SDF/ε).
        band = sdf_t.abs() < 3.0 * epsilon
        expected_boundary = _sdf_to_phi(sdf_t, epsilon)
        err = (s._hidden_phi[band] - expected_boundary[band]).abs().max().item()
        assert err < 2e-7, f"Boundary tracking error {err:.3e}"
        # Deep-interior dip survives (no gradient → no update).
        assert s._hidden_phi[dip_idx].max().item() == pytest.approx(0.4, abs=1e-7)

        # Same dip, hit again with another zero-gradient update, must still
        # persist (no reset across calls — accumulation across iterations).
        s._pf_update_hidden_phi()
        assert s._hidden_phi[dip_idx].max().item() == pytest.approx(0.4, abs=1e-7), (
            "Interior dip was reset between iterations — accumulation broken"
        )

    def test_hidden_phi_start_iteration_only_delays_interior_gradient(self):
        from core.shape_opt import ShapeOptTrimmed

        nodes, elements, sdf, epsilon = _make_problem_inputs()
        overrides = _common_overrides(
            self.tmp,
            design_variable="phase_field",
            pf_phi_lr=0.1,
            pf_phi_start_iter=5,
            use_mixed_precision=False,
        )
        overrides["phase_field_epsilon"] = epsilon
        overrides["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)
        s = ShapeOptTrimmed(CONFIG, override_params=overrides)
        sdf_t = wp.to_torch(s.e.dof_values).detach()
        s._pf_update_hidden_phi()
        deep = sdf_t < -3.0 * epsilon
        before = s._hidden_phi.clone()
        s._last_e_phi_grad = torch.ones_like(sdf_t)

        s._pf_update_iteration = 4
        s._pf_update_hidden_phi()
        torch.testing.assert_close(s._hidden_phi[deep], before[deep])

        s._pf_update_iteration = 5
        s._pf_update_hidden_phi()
        torch.testing.assert_close(
            s._hidden_phi[deep], before[deep] - 0.1, rtol=0, atol=2e-7
        )


class TestThinFeatureNeighbourSelection:
    """Mixed-depth edge selection is floating-point and unit invariant."""

    @staticmethod
    def _run(scale):
        from core.shape_opt import ShapeOptTrimmed

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nodes = [[0.0, 0.0, 0.0]]
        elements = []

        def add_hex(x0, x1, center_local):
            coords = [
                [x0, 0.0, 0.0], [x1, 0.0, 0.0],
                [x1, 0.1, 0.0], [x0, 0.1, 0.0],
                [x0, 0.0, 0.1], [x1, 0.0, 0.1],
                [x1, 0.1, 0.1], [x0, 0.1, 0.1],
            ]
            elem = []
            for local, xyz in enumerate(coords):
                if local == center_local:
                    elem.append(0)
                else:
                    elem.append(len(nodes))
                    nodes.append(xyz)
            elements.append(elem)

        # Long candidates deliberately precede short candidates. The old
        # integer composite key therefore selected 0.06 for sub-unit models.
        add_hex(0.0, 0.06, 0)
        add_hex(0.0, 0.03, 0)
        add_hex(-0.06, 0.0, 1)
        add_hex(-0.03, 0.0, 1)

        nodes_t = torch.tensor(nodes, dtype=torch.float32, device=device) * scale
        elements_t = torch.tensor(elements, dtype=torch.int32, device=device)
        phi = torch.full((len(nodes),), 0.62, dtype=torch.float32, device=device)
        phi[0] = 0.42
        sdf = torch.full_like(phi, -1.0 * scale)

        s = object.__new__(ShapeOptTrimmed)
        s._prev_octree_nodes = nodes_t
        s._prev_octree_elements = elements_t
        s._epsilon = 0.015 * scale
        s.params = {"phi_reproject_band_k": 3.0}
        s.design_dtype = wp.float32
        s.e = SimpleNamespace(dof_values=wp.from_torch(sdf, dtype=wp.float32))
        s.log = lambda *_args, **_kwargs: None

        out = s._pf_filter_thin_features(phi)
        assert out.device.type == device.type
        return out[0].item()

    def test_shortest_mixed_depth_edge_is_unit_invariant(self):
        # Two 0.03 edges give 0.024 below-length, below 2*eps=0.03, so the
        # dip is rejected. Choosing either 0.06 edge would give 0.036.
        assert self._run(1.0) == pytest.approx(0.51, abs=1e-7)
        assert self._run(1000.0) == pytest.approx(0.51, abs=1e-7)
