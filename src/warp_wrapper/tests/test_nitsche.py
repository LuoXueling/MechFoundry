"""Tests for the diffused Nitsche method implementation (single precision).

Mixed precision tests live in test_nitsche_mixed.py because warp JIT-compiles
different kernel variants for float32 vs float64 geometry and running both
in one process causes codegen conflicts.

Run with:
    conda activate mechfoundry-warp117
    pytest src/warp_wrapper/tests/test_nitsche.py -v

Covers:
  1. Phase field properties (clamping, transition)
  2. Quadrature consistency (ExplicitQuadrature == default when no refinement)
  3. Beta stabilization parameter sanity
  4. Nitsche solver convergence
  5. Adaptive quadrature with geometric_func sub-cell refinement
  6. Nitsche vs classical FEM comparison on a sphere
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch
import pytest
import warp as wp
import math

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_octree_mesh import make_uniform_hex_mesh

wp.init()

# ---------------------------------------------------------------------------
# Import trimesh pipeline (same approach as test_trimesh_octree.py)
# ---------------------------------------------------------------------------
PROJECT_SRC = ROOT_DIR.parent.parent / "src"
TESTS_DIR = Path(__file__).resolve().parent


def _load_source_module(name: str, path: Path, strip_relative: bool = False):
    source = path.read_text()
    if strip_relative:
        lines = source.split("\n")
        clean = []
        for line in lines:
            if line.strip().startswith("from ."):
                clean.append("pass  # skipped relative import")
            else:
                clean.append(line)
        source = "\n".join(clean)
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def _import_project_modules():
    he_mod = _load_source_module(
        "_proj_he", PROJECT_SRC / "triposg_wrapper" / "hierarchical_extraction.py"
    )
    utils_mod = _load_source_module(
        "_proj_utils", PROJECT_SRC / "triposg_wrapper" / "utils.py", strip_relative=True
    )
    ts_path = PROJECT_SRC / "triposg_wrapper" / "trimesh_sdf.py"
    ts_source = ts_path.read_text()
    lines = ts_source.split("\n")
    clean = [
        "pass  # skipped relative import" if line.strip().startswith("from .") else line
        for line in lines
    ]
    ts_mod = types.ModuleType("_proj_trimesh_sdf")
    ts_mod.__file__ = str(ts_path)
    ts_mod.__dict__["hierarchical_extraction"] = he_mod.hierarchical_extraction
    ts_mod.__dict__["_sdf2octree_fn"] = utils_mod.sdf2octree
    exec(compile("\n".join(clean), str(ts_path), "exec"), ts_mod.__dict__)
    return ts_mod, utils_mod.sdf2octree


try:
    import trimesh

    trimesh_sdf, sdf2octree = _import_project_modules()
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False

# ---------------------------------------------------------------------------
# Solver overrides
# ---------------------------------------------------------------------------
_OVERRIDES = dict(
    use_mixed_precision=False,
    relative_tol=1e-5,
    absolute_tol=1e-6,
    max_iter=30,
    linear_solver="cg",
    cg_backend_assembled="amgx",
    ew_enabled=False,
    line_search_enabled=False,
    load_steps=1,
)

_RESIDUAL_TOL = (
    1e-4  # Float32 accumulation floors at ~1e-5; mixed precision tests use 1e-6
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cube_octree_mesh(n=4, domain=(-1.0, 1.0)):
    """Create a uniform hex mesh on [-1,1]^3."""
    lo, hi = domain
    nodes, elements = make_uniform_hex_mesh(n, n, n, [lo, lo, lo], [hi, hi, hi])
    return nodes, elements


def _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5, truncate=0.01):
    """Compute truncated SDF for an inner cube [inner_lo, inner_hi]^3."""
    nodes_t = torch.as_tensor(nodes, dtype=torch.float32)
    dx = torch.maximum(inner_lo - nodes_t[:, 0], nodes_t[:, 0] - inner_hi)
    dy = torch.maximum(inner_lo - nodes_t[:, 1], nodes_t[:, 1] - inner_hi)
    dz = torch.maximum(inner_lo - nodes_t[:, 2], nodes_t[:, 2] - inner_hi)
    outside = torch.sqrt(
        torch.clamp(dx, min=0.0) ** 2
        + torch.clamp(dy, min=0.0) ** 2
        + torch.clamp(dz, min=0.0) ** 2
    )
    inside = torch.minimum(
        torch.maximum(dx, torch.maximum(dy, dz)), torch.zeros_like(dx)
    )
    sdf = outside + inside
    return sdf.clamp(-truncate, truncate)


def _make_sphere_trimesh_octree(init_depth, octree_depth, radius=0.5):
    """Create an octree mesh from a trimesh sphere via hierarchical extraction."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    return trimesh_sdf.trimesh_to_octree(
        mesh,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_depth,
        octree_depth=octree_depth,
        domain_length=2.0,
        device="cuda",
    )


# ===================================================================
# Category 1 — Phase field
# ===================================================================
class TestPhaseField:
    """Test phase field computation from SDF."""

    def test_phi_range(self):
        """Phi should be in [0, 1]."""
        nodes, elements = _make_cube_octree_mesh(n=4)
        sdf = _make_sdf_for_inner_cube(nodes)
        epsilon = 0.005
        phi = (0.5 * (1.0 - torch.tanh(sdf / epsilon))).clamp(0.0, 1.0)
        assert phi.min() >= 0.0
        assert phi.max() <= 1.0

    def test_phi_inside_outside(self):
        """Phi ≈ 1 inside cube, ≈ 0 outside (far from boundary)."""
        nodes, elements = _make_cube_octree_mesh(n=10)
        sdf = _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5, truncate=0.5)
        epsilon = 0.005
        phi = (0.5 * (1.0 - torch.tanh(sdf / epsilon))).clamp(0.0, 1.0)
        nodes_t = torch.as_tensor(nodes, dtype=torch.float32)

        inside_mask = (
            (nodes_t[:, 0].abs() < 0.1)
            & (nodes_t[:, 1].abs() < 0.1)
            & (nodes_t[:, 2].abs() < 0.1)
        )
        if inside_mask.any():
            assert phi[inside_mask].min() > 0.99

        outside_mask = (
            (nodes_t[:, 0].abs() > 0.8)
            | (nodes_t[:, 1].abs() > 0.8)
            | (nodes_t[:, 2].abs() > 0.8)
        )
        if outside_mask.any():
            assert phi[outside_mask].max() < 0.01


# ===================================================================
# Category 2 — Quadrature consistency
# ===================================================================
class TestQuadratureConsistency:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_same_mesh_same_quadrature(self):
        """ExplicitQuadrature builds successfully with 8 QPs per cell."""
        nodes, elements = _make_cube_octree_mesh(n=3)
        sdf = _make_sdf_for_inner_cube(nodes)
        epsilon = 0.05

        from core.nitsche_field import NitscheNeoHookean

        s = NitscheNeoHookean(
            "configs.config_nitsche_example",
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": epsilon,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -1.0)): 0.0,
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )
        assert s._correction_points_wp is not None
        assert s._qp_per_cell == 8

    def test_same_mesh_same_quadrature_trimmed(self):
        """ExplicitQuadrature builds successfully after trim_exterior compaction."""
        nodes, elements = _make_cube_octree_mesh(n=3)
        sdf = _make_sdf_for_inner_cube(nodes)
        epsilon = 0.05

        from core.nitsche_field import NitscheNeoHookean

        s = NitscheNeoHookean(
            "configs.config_nitsche_example",
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": epsilon,
                "trim_exterior": True,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )
        assert s._correction_points_wp is not None
        assert s._qp_per_cell == 8


# ===================================================================
# Category 3 — Beta sanity
# ===================================================================
class TestBetaSanity:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def _make_solver(self):
        nodes, elements = _make_cube_octree_mesh(n=8)
        sdf = _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5)
        epsilon = 0.02

        from core.nitsche_field import NitscheNeoHookean

        return NitscheNeoHookean(
            "configs.config_nitsche_example",
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": epsilon,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )

    def test_beta_nonnegative(self):
        s = self._make_solver()
        beta = s._beta_element.cpu().numpy()
        assert (beta >= -1e-10).all(), f"Found negative beta: {beta.min()}"

    def test_beta_zero_outside_boundary(self):
        s = self._make_solver()
        beta = s._beta_element.cpu().numpy()
        non_boundary = ~s._boundary_elem_mask.cpu().numpy()
        assert (beta[non_boundary] == 0.0).all()

    def test_beta_positive_at_boundary(self):
        s = self._make_solver()
        beta = s._beta_element.cpu().numpy()
        boundary = s._boundary_elem_mask.cpu().numpy()
        if boundary.any():
            assert (beta[boundary] > 0.0).any()

    def test_beta_eigensolve_uses_float32(self, monkeypatch):
        """The native-f32 path must not silently assemble beta matrices in f64."""
        from core.nitsche_field import NitscheNeoHookean

        dtypes = []
        original = NitscheNeoHookean._eigenvalue_beta_batch

        def capture_dtype(self, *args, **kwargs):
            # compute_dtype is the penultimate positional argument.
            dtypes.append(args[-2])
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            NitscheNeoHookean, "_eigenvalue_beta_batch", capture_dtype
        )
        self._make_solver()
        assert dtypes
        assert set(dtypes) == {torch.float32}


class TestBetaSanityTrimmed:
    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def _make_solver(self):
        nodes, elements = _make_cube_octree_mesh(n=8)
        sdf = _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5)
        epsilon = 0.02

        from core.nitsche_field import NitscheNeoHookean

        return NitscheNeoHookean(
            "configs.config_nitsche_example",
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": epsilon,
                "trim_exterior": True,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )

    def test_beta_nonnegative(self):
        s = self._make_solver()
        beta = s._beta_element.cpu().numpy()
        assert (beta >= -1e-10).all(), f"Found negative beta: {beta.min()}"

    def test_beta_zero_outside_boundary(self):
        s = self._make_solver()
        beta = s._beta_element.cpu().numpy()
        non_boundary = ~s._boundary_elem_mask.cpu().numpy()
        assert (beta[non_boundary] == 0.0).all()

    def test_beta_positive_at_boundary(self):
        s = self._make_solver()
        beta = s._beta_element.cpu().numpy()
        boundary = s._boundary_elem_mask.cpu().numpy()
        if boundary.any():
            assert (beta[boundary] > 0.0).any()


class TestMinElementSize:
    """``_min_element_size`` (beta floor, selector tolerance) is the finest leaf
    edge of the mesh actually solved, whatever its units and however trimmed."""

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def _make_solver(self, nodes, elements, sdf, epsilon, fixed_x, **extra):
        from core.nitsche_field import NitscheNeoHookean

        return NitscheNeoHookean(
            "configs.config_nitsche_example",
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": epsilon,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, fixed_x)): 0.0,
                },
                "neumann_bcs_dict": {},
                **extra,
                **_OVERRIDES,
            },
        )

    def test_physical_units_without_domain_length(self):
        """A 0.1-unit cube with no ``domain_length`` key: 0.1/8, not 2/8."""
        nodes, elements = _make_cube_octree_mesh(n=8, domain=(0.0, 0.1))
        sdf = _make_sdf_for_inner_cube(nodes, inner_lo=0.025, inner_hi=0.075)
        s = self._make_solver(nodes, elements, sdf, epsilon=0.00625, fixed_x=0.025)
        assert s._min_element_size == pytest.approx(0.1 / 8, rel=1e-5)

    def test_trimmed_mesh_spanning_few_leaves(self):
        """Trimmed to 8 of 16 leaves across: still 2/16, not 2/8."""
        nodes, elements = _make_cube_octree_mesh(n=16)
        sdf = _make_sdf_for_inner_cube(
            nodes, inner_lo=-0.25, inner_hi=0.25, truncate=1.0
        )
        s = self._make_solver(
            nodes, elements, sdf, epsilon=0.0625, fixed_x=-0.25, trim_exterior=True
        )
        # The premise: trimming leaves [-0.5, 0.5]^3, under 0.71 of the cube.
        assert float(s._domain_span.max()) == pytest.approx(1.0, rel=1e-5)
        assert s._min_element_size == pytest.approx(2.0 / 16, rel=1e-5)


# ===================================================================
# Category 4 — Convergence
# ===================================================================
class TestNitscheConvergence:

    CONFIG = "configs.config_nitsche_example"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_converges_simple(self):
        """Nitsche solver converges with active Dirichlet + Neumann terms."""
        nodes, elements = _make_cube_octree_mesh(n=8)
        sdf = _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5)
        epsilon = 0.02

        from core.nitsche_field import NitscheNeoHookean

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": epsilon,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.5)): (
                        0.01,
                        0.0,
                        0.0,
                    ),
                },
                **_OVERRIDES,
            },
        )
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"

        # Verify Nitsche terms are active
        assert (s._psi_D_qp.numpy() > 0.5).sum() > 0, "No Dirichlet QPs active"
        assert (s._psi_N_qp.numpy() > 0.5).sum() > 0, "No Neumann QPs active"

        # Check non-trivial displacement
        u = s.u.dof_values.numpy().reshape(-1, 3)
        assert np.abs(u).max() > 1e-6, "Displacement is essentially zero"

    def test_converges_simple_trimmed(self):
        """Nitsche solver converges after trim_exterior compacts the mesh."""
        nodes, elements = _make_cube_octree_mesh(n=8)
        sdf = _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5)
        epsilon = 0.02

        from core.nitsche_field import NitscheNeoHookean

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": epsilon,
                "trim_exterior": True,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.5)): (
                        0.01,
                        0.0,
                        0.0,
                    ),
                },
                **_OVERRIDES,
            },
        )
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"

        # Verify Nitsche terms are active
        assert (s._psi_D_qp.numpy() > 0.5).sum() > 0, "No Dirichlet QPs active"
        assert (s._psi_N_qp.numpy() > 0.5).sum() > 0, "No Neumann QPs active"

        # Check non-trivial displacement
        u = s.u.dof_values.numpy().reshape(-1, 3)
        assert np.abs(u).max() > 1e-6, "Displacement is essentially zero"


class TestSharpExteriorClaim:
    """Under sharp_volumetric a Dirichlet region claims a wholly exterior element whole.

    Only active QPs may do the claiming.  Padding slots sit at the origin with
    parent element 0, so a region containing the origin used to claim element 0
    whole through them, pinning an exterior element nowhere near the region.
    Calls ``_evaluate_bcs_at_positions`` on hand-built QPs; no mesh is needed.
    """

    def test_padding_does_not_claim_its_element(self):
        from core.nitsche_field import NitscheNeoHookean

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        solver = types.SimpleNamespace(
            params={"sharp_volumetric": True, "nitsche_neumann_rescale": False},
            device=dev, dim=3, use_mixed_precision=False,
            _original_dirichlet_bcs_dict={(lambda x, y, z: x < 0.1): 0.0},
            _original_neumann_bcs_dict={},
        )
        # Element 0: exterior, far from the region, its last slot padding.
        # Element 1: exterior, one QP inside the region.  Element 2: interior.
        qp_phys = torch.tensor([[5.0, 0, 0], [5.1, 0, 0], [0.0, 0, 0],
                                [0.05, 1, 0], [0.2, 1, 0],
                                [3.0, 2, 0], [3.1, 2, 0]], device=dev)
        qp_elem = torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.int32, device=dev)
        qp_active = torch.tensor([1, 1, 0, 1, 1, 1, 1], dtype=torch.int32, device=dev)
        qp_phi = torch.tensor([0.1, 0.1, 0.0, 0.1, 0.1, 0.9, 0.9], device=dev)

        _, _, psi_D, _, _ = NitscheNeoHookean._evaluate_bcs_at_positions(
            solver, qp_phys, qp_active, torch.zeros(7, 3, device=dev),
            qp_elem=qp_elem, qp_phi=qp_phi,
        )
        claimed = psi_D[:, 0].bool().tolist()
        assert claimed == [False, False, False, True, True, False, False], claimed


class TestNeumannRescale:
    """The assembled Neumann load is traction x the region's loaded area.

    Without rescaling, the band the BC predicate claims keeps only part of the
    traction kernel, and the 2x2x2 Gauss sum depends on alignment.  Here the
    unscaled load is 10 % over the reference area, and the rescaled one
    matches it to 3e-5.  The reference is the MC surface's area (0.595 of the
    true 0.64 face on this 3-cells-across cube), since a plain predicate has
    no other surface to be measured on.
    """

    CONFIG = "configs.config_nitsche_example"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def _applied_force(self, rescale, spec=None, reference=None):
        """Total load assembled at u = 0, and the region's reference area."""
        nodes, elements = _make_cube_octree_mesh(n=8)
        # Faces at +-0.4 lie between the 0.25-spaced node planes.
        sdf = _make_sdf_for_inner_cube(nodes, -0.4, 0.4, truncate=10.0)

        from core.nitsche_field import NitscheNeoHookean

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": 0.125,
                "nitsche_neumann_rescale": rescale,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.4)): 0.0,
                },
                "neumann_bcs_dict": {
                    (spec or (lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.4)))): (
                        0.01,
                        0.0,
                        0.0,
                    ),
                },
                **({"bc_reference_surface": reference} if reference else {}),
                **_OVERRIDES,
            },
        )
        # At u = u_bar = 0 the stress and every Dirichlet term vanish, so the
        # residual is minus the applied load.
        s._current_preprocessed_e = s._preprocess_optimized()
        mc = s._mesh_cache
        R = s._evaluate_residual(
            mc["domain"], mc["test"], mc["boundary"], mc["bd_test"],
            neumann_load=mc["neumann"], body_force=mc["body"],
        )
        force = -wp.to_torch(R).double().sum(dim=0).cpu().numpy()
        with s._widened_bc_isclose():
            area = s._neumann_region_area(next(iter(s._original_neumann_bcs_dict)))
        return force, area

    def test_load_is_traction_times_area(self):
        force, area = self._applied_force(rescale=True)
        np.testing.assert_allclose(force, [0.01 * area, 0.0, 0.0], rtol=1e-4, atol=1e-12)

    def test_switch_restores_unscaled_load(self):
        force, area = self._applied_force(rescale=False)
        assert abs(force[0] / (0.01 * area) - 1.0) > 0.05

    @pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
    def test_selector_measured_on_reference_surface(self):
        """A selector region is measured on ``bc_reference_surface``.

        The solver resolves the selector on its own MC surface, whose +x face
        is short of the true one on this coarse cube.  Measured on the exact
        cube, the region is the true 0.8 x 0.8 face, and so is the load.
        """
        from core.bc_surface_selectors import HighestSelector

        box = trimesh.creation.box(extents=(0.8, 0.8, 0.8))
        force, area = self._applied_force(
            rescale=True,
            spec=HighestSelector(axis="x", fraction=0.05),
            reference={"vertices": np.asarray(box.vertices), "faces": np.asarray(box.faces)},
        )
        assert area == pytest.approx(0.64, rel=1e-6)
        np.testing.assert_allclose(force, [0.01 * 0.64, 0.0, 0.0], rtol=1e-4, atol=1e-12)


# ===================================================================
# Category 6 — Adaptive quadrature with refined quad_octree_mesh
# ===================================================================
@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestAdaptiveQuadrature:
    """Test adaptive quadrature via internal sub-cell refinement with geometric_func."""

    CONFIG = "configs.config_nitsche_example"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_refined_quad_mesh_converges(self):
        """Nitsche with geometric_func adaptive quadrature converges."""
        coarse = _make_sphere_trimesh_octree(init_depth=3, octree_depth=3, radius=0.5)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")

        from core.nitsche_field import NitscheNeoHookean

        epsilon = 0.02

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": coarse,
                "geometric_func": geometric_func,
                "quad_refinement_depth": 2,
                "phase_field_epsilon": epsilon,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.5)): (
                        0.01,
                        0.0,
                        0.0,
                    ),
                },
                **_OVERRIDES,
            },
        )

        # Adaptive quadrature should be active
        assert s._qp_uses_child

        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"

        # Verify non-trivial displacement
        u = s.u.dof_values.numpy().reshape(-1, 3)
        assert np.abs(u).max() > 1e-6

    def test_refined_quad_mesh_converges_trimmed(self):
        """Nitsche with geometric_func adaptive quadrature converges after trimming."""
        coarse = _make_sphere_trimesh_octree(init_depth=3, octree_depth=3, radius=0.5)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")

        from core.nitsche_field import NitscheNeoHookean

        epsilon = 0.02

        # Use native CG backend (AMGX can crash with very small trimmed meshes
        # when reusing handles from prior tests in the same process).
        overrides = {**_OVERRIDES, "cg_backend_assembled": "native"}

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": coarse,
                "geometric_func": geometric_func,
                "quad_refinement_depth": 2,
                "phase_field_epsilon": epsilon,
                "trim_exterior": True,
                "beta_max": 2000.0,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.5)): (
                        0.01,
                        0.0,
                        0.0,
                    ),
                },
                **overrides,
            },
        )

        has_chunks = hasattr(s, "_nitsche_chunks") and s._nitsche_chunks
        has_flat = (
            hasattr(s, "_correction_points_wp") and s._correction_points_wp is not None
        )
        assert has_chunks or (s._qp_per_cell > 8 and has_flat)

        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"

        u = s.u.dof_values.numpy().reshape(-1, 3)
        assert np.abs(u).max() > 1e-6


# ===================================================================
# Category 7 — Nitsche vs classical FEM comparison
# ===================================================================
@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestNitscheVsClassical:
    """Compare Nitsche solution on octree mesh vs classical FEM on conforming mesh.

    Uses a cube [-0.5, 0.5]^3 embedded in [-1,1]^3. Classical FEM uses a
    conforming mesh of the cube with standard hard Dirichlet BCs. Nitsche
    uses trim_exterior to remove void elements and enforces BCs via diffuse
    penalty/consistency/symmetry terms on the cube surface.

    For uniform compression (clamp x=-0.5, push x=+0.5), the exact solution
    is a linear displacement field u_x = delta * (x + 0.5). Both methods
    should produce this to within discretization + diffuse-boundary error.
    """

    CONFIG = "configs.config_nitsche_example"
    HALF = 0.5

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    DELTA = 0.005
    BOUNDS = (-1.005, -1.005, -1.005, 1.005, 1.005, 1.005)
    STATIONS = [0.0, 0.05, 0.1, 0.15]

    def _solve_classical(self):
        """Solve classical FEM on conforming mesh; cached per test."""
        from core.neoHookean import OctreeNeoHookean

        hs = self.HALF
        nodes_cl, elems_cl = make_uniform_hex_mesh(
            16, 16, 16, [-hs, -hs, -hs], [hs, hs, hs]
        )
        solver = OctreeNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/classical",
                "octree_mesh": {"nodes": nodes_cl, "elements": elems_cl},
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -hs)): 0.0,
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, hs)): (
                        self.DELTA,
                        0.0,
                        0.0,
                    ),
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )
        solver.forward()
        return solver, np.asarray(nodes_cl)

    def _solve_nitsche(self, label, coarse_depth=6, sharp_volumetric=True):
        """Solve Nitsche FEM on trimmed octree mesh with adaptive quadrature."""
        from core.nitsche_field import NitscheNeoHookean

        hs = self.HALF
        mesh = __import__("trimesh").creation.box(extents=(2 * hs,) * 3)
        coarse = trimesh_sdf.trimesh_to_octree(
            mesh,
            bounds=self.BOUNDS,
            init_octree_depth=coarse_depth,
            octree_depth=coarse_depth,
            domain_length=2.0,
            device="cuda",
        )
        h_coarse = 2.0 / (2 ** coarse_depth)
        h_quad = h_coarse / 2.0  # 1 level of refinement
        epsilon = h_quad / 2.0
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")
        params = {
            "output_root": self.tmp + "/" + label,
            "octree_mesh": coarse,
            "phase_field_epsilon": epsilon,
            "sharp_volumetric": sharp_volumetric,
            "heuristic_beta": True,
            "phi_floor": 1e-4,
            "tangent_f64": True,
            "geometric_func": geometric_func,
            "quad_refinement_depth": 1,
            "dirichlet_bcs_dict": {
                lambda x, y, z: torch.isclose(x, torch.full_like(x, -hs)): 0.0,
                lambda x, y, z: torch.isclose(x, torch.full_like(x, hs)): (
                    self.DELTA,
                    0.0,
                    0.0,
                ),
            },
            "neumann_bcs_dict": {},
            **_OVERRIDES,
        }
        solver = NitscheNeoHookean(self.CONFIG, override_params=params)
        solver.forward()
        return solver

    def _compute_ratios(self, cl_solver, nodes_cl_np, ni_solver):
        """Compute Nitsche/classical ux ratios at x-stations."""
        u_cl = cl_solver.u.dof_values.numpy().reshape(-1, 3)
        u_ni = ni_solver.u.dof_values.numpy().reshape(-1, 3)
        ni_nodes = ni_solver.params.octree_mesh["nodes"]
        nodes_ni_np = (
            ni_nodes.cpu().numpy()
            if isinstance(ni_nodes, torch.Tensor)
            else np.asarray(ni_nodes)
        )
        phi_ni = ni_solver._phi_nodal.cpu().numpy()
        inside = phi_ni > 0.5
        ratios = {}
        for xp in self.STATIONS:
            cl_m = np.abs(nodes_cl_np[:, 0] - xp) < 0.06
            ni_m = (np.abs(nodes_ni_np[:, 0] - xp) < 0.1) & inside
            assert ni_m.any(), f"No Nitsche interior nodes at x={xp}"
            ratios[xp] = u_ni[ni_m, 0].mean() / u_cl[cl_m, 0].mean()
        return ratios, u_ni, nodes_ni_np, inside

    def test_affine_compression_sharp_volumetric(self):
        """Trimmed Nitsche (sharp volumetric) vs classical FEM: ratio within [0.95, 1.05]."""
        cl_solver, nodes_cl_np = self._solve_classical()
        ni_solver = self._solve_nitsche("nitsche_sharp", sharp_volumetric=True)

        assert cl_solver.final_residual < _RESIDUAL_TOL
        assert ni_solver.final_residual < _RESIDUAL_TOL

        # Classical matches analytical at midplane
        ux_analytical = self.DELTA / 2.0
        mid_cl = np.abs(nodes_cl_np[:, 0]) < 0.06
        u_cl = cl_solver.u.dof_values.numpy().reshape(-1, 3)
        assert abs(u_cl[mid_cl, 0].mean() - ux_analytical) < 1e-4

        # Nitsche/classical ratio at interior x-stations
        ratios, u_ni, nodes_ni_np, inside = self._compute_ratios(
            cl_solver, nodes_cl_np, ni_solver
        )
        for xp, ratio in ratios.items():
            assert (
                0.95 < ratio < 1.05
            ), f"x={xp:+.1f}: ratio {ratio:.3f} outside [0.95, 1.05]"

        # Transverse displacement is small
        mid_ni = (np.abs(nodes_ni_np[:, 0]) < 0.1) & inside
        ux_ni = np.abs(u_ni[mid_ni, 0]).mean()
        uy_ni = np.abs(u_ni[mid_ni, 1]).mean()
        uz_ni = np.abs(u_ni[mid_ni, 2]).mean()
        assert uy_ni < 0.5 * ux_ni
        assert uz_ni < 0.5 * ux_ni

    def test_affine_compression_diffused_volumetric(self):
        """Trimmed Nitsche (diffused volumetric) vs classical FEM: ratio within [0.95, 1.05]."""
        cl_solver, nodes_cl_np = self._solve_classical()
        ni_solver = self._solve_nitsche("nitsche_diffused", sharp_volumetric=False)

        assert cl_solver.final_residual < _RESIDUAL_TOL
        assert ni_solver.final_residual < _RESIDUAL_TOL
        assert ni_solver._qp_uses_child, "Adaptive quadrature should be active"

        ratios, u_ni, nodes_ni_np, inside = self._compute_ratios(
            cl_solver, nodes_cl_np, ni_solver
        )
        for xp, ratio in ratios.items():
            assert (
                0.95 < ratio < 1.05
            ), f"x={xp:+.1f}: ratio {ratio:.3f} outside [0.95, 1.05]"


# ===================================================================
# Category 8 — Chunked vs flat integration path equivalence
# ===================================================================
@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestChunkedVsFlat:
    """Verify chunked and flat integration paths produce the same solution.

    Runs the same Nitsche problem twice — once forcing flat and once forcing
    chunked — and checks that DOF values, residuals, and beta parameters
    match to within floating-point tolerance.
    """

    CONFIG = "configs.config_nitsche_example"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def _solve(self, label, force_chunked):
        """Solve a sphere compression problem with flat or chunked path."""
        from core.nitsche_field import NitscheNeoHookean

        mesh = __import__("trimesh").creation.icosphere(subdivisions=3, radius=0.5)
        coarse = trimesh_sdf.trimesh_to_octree(
            mesh,
            bounds=(-1.005,) * 3 + (1.005,) * 3,
            init_octree_depth=3,
            octree_depth=3,
            domain_length=2.0,
            device="cuda",
        )
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")
        epsilon = 0.02
        solver = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/" + label,
                "octree_mesh": coarse,
                "geometric_func": geometric_func,
                "quad_refinement_depth": 2,
                "phase_field_epsilon": epsilon,
                "heuristic_beta": True,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.5)): (
                        0.005,
                        0.0,
                        0.0,
                    ),
                },
                "neumann_bcs_dict": {},
                "_force_chunked": force_chunked,
                **_OVERRIDES,
            },
        )
        solver.forward()
        return solver

    def test_flat_vs_chunked_same_solution(self):
        """Flat and chunked paths produce nearly identical DOF values."""
        flat_solver = self._solve("flat", force_chunked=False)
        chunked_solver = self._solve("chunked", force_chunked=True)

        assert flat_solver.final_residual < _RESIDUAL_TOL
        assert chunked_solver.final_residual < _RESIDUAL_TOL

        # Verify correct paths were used
        assert not flat_solver._nitsche_chunks, "Flat solver should not use chunks"
        assert chunked_solver._nitsche_chunks, "Chunked solver should use chunks"

        # Compare DOF values
        u_flat = flat_solver.u.dof_values.numpy().reshape(-1, 3)
        u_chunk = chunked_solver.u.dof_values.numpy().reshape(-1, 3)

        # Mixed absolute + relative error (handles near-zero DOFs)
        abs_err = np.linalg.norm(u_flat - u_chunk, axis=1)
        mag_flat = np.linalg.norm(u_flat, axis=1)
        # relative where displacement is significant, absolute otherwise
        atol = 1e-7
        rtol = 0.01
        err = abs_err / np.maximum(mag_flat, atol / rtol)
        assert err.max() < rtol, (
            f"Max DOF error {err.max():.6f} > {rtol} " f"between flat and chunked paths"
        )

        # Beta should also match
        beta_flat = flat_solver._beta_element.cpu().numpy()
        beta_chunk = chunked_solver._beta_element.cpu().numpy()
        np.testing.assert_allclose(beta_flat, beta_chunk, rtol=1e-5)
