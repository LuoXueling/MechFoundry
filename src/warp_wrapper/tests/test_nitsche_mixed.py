"""Tests for the diffused Nitsche method with MIXED PRECISION (float64 geometry).

Warp JIT-compiles different kernel variants for float32 vs float64 geometry.
Running both precisions in one process causes codegen conflicts, so mixed
precision Nitsche tests live here, separate from test_nitsche.py.

Run with:
    conda activate mechfoundry-warp117
    pytest src/warp_wrapper/tests/test_nitsche_mixed.py -v
"""

import sys
import types
from pathlib import Path

import numpy as np
import torch
import pytest
import warp as wp


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_octree_mesh import make_uniform_hex_mesh

wp.init()


@wp.kernel
def _contract_vec3d(
    values: wp.array(dtype=wp.vec3d),
    seed: wp.array(dtype=wp.vec3d),
    out: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    wp.atomic_add(out, 0, wp.dot(values[i], seed[i]))

# ---------------------------------------------------------------------------
# Import trimesh pipeline (same approach as test_trimesh_octree.py)
# ---------------------------------------------------------------------------
PROJECT_SRC = ROOT_DIR.parent.parent / "src"


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
# Solver overrides — MIXED PRECISION
# ---------------------------------------------------------------------------
_OVERRIDES = dict(
    use_mixed_precision=True,
    relative_tol=1e-8,
    absolute_tol=1e-10,
    max_iter=30,
    linear_solver="cg",
    cg_backend_assembled="amgx",
    ew_enabled=False,
    line_search_enabled=True,
    load_steps=1,
)

_RESIDUAL_TOL = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cube_octree_mesh(n=4, domain=(-1.0, 1.0)):
    lo, hi = domain
    nodes, elements = make_uniform_hex_mesh(n, n, n, [lo, lo, lo], [hi, hi, hi])
    return nodes, elements


def _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5, truncate=0.01):
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
# Mixed precision convergence
# ===================================================================
class TestMixedPrecision:
    """Verify Nitsche works with use_mixed_precision=True."""

    CONFIG = "configs.config_nitsche_example"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_mixed_precision_converges(self):
        """Nitsche solver converges in mixed-precision (f64 integrands, f32 K)."""
        nodes, elements = _make_cube_octree_mesh(n=4)
        sdf = _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5, truncate=0.5)
        epsilon = 0.2

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

        # Verify f64 integrand scalar dtype
        assert s._integrand_scalar_dtype == wp.float64

        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"

        # Verify active BCs and non-trivial result
        assert (s._psi_D_qp.numpy() > 0.5).sum() > 0
        u = s.u.dof_values.numpy().reshape(-1, 3)
        assert np.abs(u).max() > 1e-6

    def test_partial_dirichlet_masks_and_values_are_componentwise(self):
        """Two partial BCs must not clamp or average the third component."""
        nodes, elements = _make_cube_octree_mesh(n=8)
        sdf = _make_sdf_for_inner_cube(nodes, truncate=0.5)

        from core.nitsche_field import NitscheNeoHookean

        left_x = lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5))
        left_y = lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5))
        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/partial_components",
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": 0.2,
                "dirichlet_bcs_dict": {
                    left_x: ([0], 2.0),
                    left_y: ([1], 4.0),
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )

        if getattr(s, "_nitsche_chunks", None):
            masks = np.concatenate([c.psi_D_qp.numpy() for c in s._nitsche_chunks])
            values = np.concatenate([c.u_bar_qp.numpy() for c in s._nitsche_chunks])
        else:
            masks = s._psi_D_qp.numpy()
            values = s._u_bar_qp.numpy()

        active = masks.any(axis=1)
        assert active.any()
        assert np.all(masks[active] == np.asarray([1.0, 1.0, 0.0]))
        assert np.allclose(values[active, 0], 2.0)
        assert np.allclose(values[active, 1], 4.0)
        assert np.allclose(values[active, 2], 0.0)

    def test_partial_dirichlet_reacts_only_in_selected_component(self):
        """A free rigid translation must produce no Nitsche reaction.

        With an x-only zero-displacement condition, a constant z translation
        has zero strain and zero projected Dirichlet gap.  Its assembled
        residual must therefore vanish.  The same translation in x has a
        nonzero projected gap and must produce a reaction.  Comparing these
        finite perturbations also exercises the component projection used by
        the consistent tangent, rather than merely inspecting stored masks.
        """
        nodes, elements = _make_cube_octree_mesh(n=8)
        sdf = _make_sdf_for_inner_cube(nodes, truncate=0.5)

        from core.nitsche_field import NitscheNeoHookean

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/partial_component_reaction",
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": 0.2,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(
                        x, torch.full_like(x, -0.5)
                    ): ([0], 0.0),
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )
        mesh_cache = s._mesh_cache
        s.current_load_scale = 1.0
        s._current_preprocessed_e = s.e

        def residual_for_translation(component):
            displacement = torch.zeros(
                s.space.node_count(), 3, dtype=torch.float64, device=s.device
            )
            displacement[:, component] = 1.0e-2
            wp.copy(
                src=wp.from_torch(displacement.contiguous(), dtype=s.dtype),
                dest=s.u.dof_values,
            )
            residual = s._evaluate_residual(
                mesh_cache["domain"],
                mesh_cache["test"],
                mesh_cache["boundary"],
                mesh_cache["bd_test"],
            )
            return np.linalg.norm(residual.numpy().reshape(-1, 3))

        free_reaction = residual_for_translation(2)
        constrained_reaction = residual_for_translation(0)

        assert constrained_reaction > 1.0e-6
        assert free_reaction < max(1.0e-10, constrained_reaction * 1.0e-8)

    def test_phase_weighted_body_force_changes_sdf_and_hidden_phi_vjps(self):
        """The taped body residual must differentiate both design paths."""
        nodes, elements = _make_cube_octree_mesh(n=8)
        sdf = _make_sdf_for_inner_cube(nodes, truncate=0.5)

        from core.nitsche_field import NitscheNeoHookean

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/body_vjp",
                "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
                "phase_field_epsilon": 0.05,
                "B": (0.0, -2.0, 0.0),
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(
                        x, torch.full_like(x, -0.5)
                    ): 0.0,
                },
                "neumann_bcs_dict": {},
                **_OVERRIDES,
            },
        )
        domain = s._mesh_cache["domain"]
        test = s._mesh_cache["test"]
        generator = torch.Generator(device=s.device).manual_seed(7)
        seed_t = torch.randn(
            s.space.node_count(),
            3,
            dtype=torch.float64,
            device=s.device,
            generator=generator,
        )
        seed = wp.from_torch(seed_t.contiguous(), dtype=s.dtype)
        mc = s._mesh_cache
        s.current_load_scale = 1.0

        # SDF boundary path: the only recorded operation is body-load
        # assembly because u=0 makes the neo-Hookean internal residual zero.
        s.e.dof_values.requires_grad = True
        tape_sdf = wp.Tape()
        s._current_preprocessed_e = s._preprocess_optimized(tape=tape_sdf)
        s._pre_tape_residual(tape_sdf)
        if getattr(s, "_nitsche_chunks", None):
            for chunk in s._nitsche_chunks:
                chunk.phi_qp.retain_grad = True
        else:
            s._phi_qp.retain_grad = True
        residual_sdf = s._evaluate_residual(
            domain,
            test,
            mc["boundary"],
            mc["bd_test"],
            neumann_load=None,
            body_force=mc["body"],
            tape=tape_sdf,
        )
        total_body_y = float(residual_sdf.numpy()[:, 1].sum())
        assert 0.0 < total_body_y < 8.0
        loss_sdf = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=s.device)
        with tape_sdf:
            wp.launch(
                _contract_vec3d,
                dim=s.space.node_count(),
                inputs=[residual_sdf, seed],
                outputs=[loss_sdf],
                device=s.device,
            )
        tape_sdf.backward(loss=loss_sdf)
        residual_sdf_t = wp.to_torch(residual_sdf).detach().clone()
        g_sdf = wp.to_torch(s.e.dof_values.grad).detach().clone()
        assert torch.isfinite(g_sdf).all()
        if getattr(s, "_nitsche_chunks", None):
            g_qp_max = max(
                wp.to_torch(c.phi_qp.grad).abs().max().item()
                for c in s._nitsche_chunks
            )
        else:
            g_qp_max = wp.to_torch(s._phi_qp.grad).abs().max().item()
        assert g_qp_max > 1e-8

        # Omitting the optional cache must preserve both the forward residual
        # and the body-load VJP.  This specifically exercises the supported
        # body_force=None helper path.
        s.e.dof_values.grad.zero_()
        tape_sdf_uncached = wp.Tape()
        s._current_preprocessed_e = s._preprocess_optimized(tape=tape_sdf_uncached)
        s._pre_tape_residual(tape_sdf_uncached)
        residual_sdf_uncached = s._evaluate_residual(
            domain,
            test,
            mc["boundary"],
            mc["bd_test"],
            neumann_load=None,
            body_force=None,
            tape=tape_sdf_uncached,
        )
        loss_sdf_uncached = wp.zeros(
            1, dtype=wp.float64, requires_grad=True, device=s.device
        )
        with tape_sdf_uncached:
            wp.launch(
                _contract_vec3d,
                dim=s.space.node_count(),
                inputs=[residual_sdf_uncached, seed],
                outputs=[loss_sdf_uncached],
                device=s.device,
            )
        tape_sdf_uncached.backward(loss=loss_sdf_uncached)
        torch.testing.assert_close(
            wp.to_torch(residual_sdf_uncached), residual_sdf_t, rtol=1e-11, atol=1e-12
        )
        torch.testing.assert_close(
            wp.to_torch(s.e.dof_values.grad), g_sdf, rtol=1e-10, atol=1e-12
        )

        # Hidden-phi interior path: use the same Nitsche instance with a unit
        # e_phi field. Forward load stays unchanged, while its residual VJP is
        # nonzero on standard (non-boundary) elements.
        s._use_phase_field_design = lambda: True
        s.e_phi = s.e_space.make_field()
        ones = torch.ones(
            s.e_space.node_count(), dtype=torch.float64, device=s.device
        )
        wp.copy(
            src=wp.from_torch(ones.contiguous(), dtype=wp.float64),
            dest=s.e_phi.dof_values,
        )
        s.e_phi.dof_values.requires_grad = True
        tape_phi = wp.Tape()
        s._current_preprocessed_e = s.e_phi
        s._pre_tape_residual(tape_phi)
        residual_phi = s._evaluate_residual(
            domain,
            test,
            mc["boundary"],
            mc["bd_test"],
            neumann_load=None,
            body_force=mc["body"],
            tape=tape_phi,
        )
        loss_phi = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=s.device)
        with tape_phi:
            wp.launch(
                _contract_vec3d,
                dim=s.space.node_count(),
                inputs=[residual_phi, seed],
                outputs=[loss_phi],
                device=s.device,
            )
        tape_phi.backward(loss=loss_phi)
        residual_phi_t = wp.to_torch(residual_phi).detach().clone()
        g_phi = wp.to_torch(s.e_phi.dof_values.grad).detach().clone()
        assert torch.isfinite(g_phi).all()
        assert g_phi.abs().max().item() > 1e-8

        s.e_phi.dof_values.grad.zero_()
        tape_phi_uncached = wp.Tape()
        s._current_preprocessed_e = s.e_phi
        s._pre_tape_residual(tape_phi_uncached)
        residual_phi_uncached = s._evaluate_residual(
            domain,
            test,
            mc["boundary"],
            mc["bd_test"],
            neumann_load=None,
            body_force=None,
            tape=tape_phi_uncached,
        )
        loss_phi_uncached = wp.zeros(
            1, dtype=wp.float64, requires_grad=True, device=s.device
        )
        with tape_phi_uncached:
            wp.launch(
                _contract_vec3d,
                dim=s.space.node_count(),
                inputs=[residual_phi_uncached, seed],
                outputs=[loss_phi_uncached],
                device=s.device,
            )
        tape_phi_uncached.backward(loss=loss_phi_uncached)
        torch.testing.assert_close(
            wp.to_torch(residual_phi_uncached), residual_phi_t, rtol=1e-11, atol=1e-12
        )
        torch.testing.assert_close(
            wp.to_torch(s.e_phi.dof_values.grad), g_phi, rtol=1e-10, atol=1e-12
        )


# ===================================================================
# Adaptive quadrature + mixed precision
# ===================================================================
@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestAdaptiveQuadratureMixed:
    """Test adaptive quadrature + mixed precision."""

    CONFIG = "configs.config_nitsche_example"

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_refined_quad_mesh_mixed_precision(self):
        """Adaptive quadrature + mixed precision converges."""
        coarse = _make_sphere_trimesh_octree(init_depth=3, octree_depth=3, radius=0.5)
        fine = _make_sphere_trimesh_octree(init_depth=3, octree_depth=4, radius=0.5)

        from core.nitsche_field import NitscheNeoHookean

        epsilon = 0.02

        s = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp,
                "octree_mesh": coarse,
                "quad_octree_mesh": fine,
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

        assert s._integrand_scalar_dtype == wp.float64
        s.forward()
        assert s.final_residual < _RESIDUAL_TOL, f"residual {s.final_residual:.3e}"


# ===================================================================
# Cantilever beam: classical vs Nitsche comparison (mixed precision)
# ===================================================================
def _make_beam_trimesh_octree(init_depth, octree_depth, extents=(1.5, 0.5, 0.5)):
    """Create octree mesh from a beam box centered at origin."""
    mesh = trimesh.creation.box(extents=extents)
    return trimesh_sdf.trimesh_to_octree(
        mesh,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_depth,
        octree_depth=octree_depth,
        domain_length=2.0,
        device="cuda",
    )


@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestCantileverBeamComparison:
    """Compare cantilever beam: classical OctreeNeoHookean vs NitscheNeoHookean.

    Geometry: 1.5 x 0.5 x 0.5 beam centered at origin.
    Grid-aligned: 0.75/(2/2^d) = 3·2^(d-3) and 0.25/(2/2^d) = 2^(d-3),
    both integer for d >= 3.

    BCs:
      - Dirichlet: clamped at x = -0.75
      - Neumann: small downward load (0, -0.001, 0) on full tip face x=0.75

    Both solved with mixed precision (small deformation).
    """

    CONFIG = "configs.config_nitsche_example"

    BEAM_LO = [-0.75, -0.25, -0.25]
    BEAM_HI = [0.75, 0.25, 0.25]

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_cantilever_beam(self):
        """Classical conforming FEM vs Nitsche on same cantilever beam.

        Small downward tip load for small-deformation regime.
        Uses adaptive quadrature + sharp_volumetric + directional Neumann
        (-grad_phi . n_face) for high accuracy.
        """
        # ---- Classical FEM on conforming uniform hex mesh ----
        from core.neoHookean import OctreeNeoHookean

        nodes_cl, elems_cl = make_uniform_hex_mesh(
            48, 16, 16, self.BEAM_LO, self.BEAM_HI
        )

        classical = OctreeNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/classical",
                "octree_mesh": {"nodes": nodes_cl, "elements": elems_cl},
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.75)): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.75)): (
                        0.0,
                        -0.001,
                        0.0,
                    ),
                },
                **_OVERRIDES,
            },
        )
        classical.forward()

        # ---- Nitsche FEM on octree mesh from trimesh ----
        from core.nitsche_field import NitscheNeoHookean

        beam_mesh = trimesh.creation.box(extents=(1.5, 0.5, 0.5))
        coarse = _make_beam_trimesh_octree(init_depth=6, octree_depth=6)
        h_coarse = 2.0 / (2**6)
        h_quad = h_coarse / 2.0
        epsilon = h_quad / 2.0
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(
            beam_mesh, device="cuda"
        )

        nitsche = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/nitsche",
                "octree_mesh": coarse,
                "phase_field_epsilon": epsilon,
                "sharp_volumetric": True,
                "heuristic_beta": True,
                "phi_floor": 1e-4,
                "tangent_f64": True,
                "geometric_func": geometric_func,
                "quad_refinement_depth": 1,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.75)): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.75)): (
                        0.0,
                        -0.001,
                        0.0,
                    ),
                },
                **_OVERRIDES,
            },
        )
        nitsche.forward()

        # ---- Both converge ----
        assert (
            classical.final_residual < _RESIDUAL_TOL
        ), f"Classical residual {classical.final_residual:.3e}"
        assert (
            nitsche.final_residual < _RESIDUAL_TOL
        ), f"Nitsche residual {nitsche.final_residual:.3e}"

        # ---- Extract solutions ----
        u_cl = classical.u.dof_values.numpy().reshape(-1, 3)
        nodes_cl_np = np.asarray(nodes_cl)

        u_ni = nitsche.u.dof_values.numpy().reshape(-1, 3)
        ni_nodes = coarse["nodes"]
        nodes_ni_np = (
            ni_nodes.cpu().numpy()
            if isinstance(ni_nodes, torch.Tensor)
            else np.asarray(ni_nodes)
        )
        phi_ni = nitsche._phi_nodal.cpu().numpy()
        inside_mask = phi_ni > 0.9

        # ---- Check 1: Tip deflects downward in both ----
        cl_tip = np.abs(nodes_cl_np[:, 0] - 0.75) < 0.02
        tip_uy_cl = u_cl[cl_tip, 1].mean()

        ni_tip = (np.abs(nodes_ni_np[:, 0] - 0.75) < 0.15) & inside_mask
        assert ni_tip.any(), "No Nitsche nodes found near beam tip"
        tip_uy_ni = u_ni[ni_tip, 1].mean()

        assert tip_uy_cl < 0, f"Classical tip should deflect down: uy={tip_uy_cl:.4e}"
        assert tip_uy_ni < 0, f"Nitsche tip should deflect down: uy={tip_uy_ni:.4e}"

        # ---- Check 2: Clamped end nearly zero in classical ----
        cl_fixed = np.abs(nodes_cl_np[:, 0] - (-0.75)) < 0.02
        assert np.abs(u_cl[cl_fixed]).max() < 1e-8, "Classical clamped end not zero"

        # ---- Check 3: uy deflection profile at interior x-stations ----
        # Grid-aligned geometry (0.75, 0.25) + directional Neumann fix give
        # interior ratios in [0.95, 1.05].  Tolerance [0.88, 1.12].
        stations = [0.0, 0.25, 0.5]
        for xp in stations:
            cl_m = np.abs(nodes_cl_np[:, 0] - xp) < 0.06
            ni_m = (np.abs(nodes_ni_np[:, 0] - xp) < 0.1) & inside_mask
            assert ni_m.any(), f"No Nitsche interior nodes at x={xp}"
            uy_cl = u_cl[cl_m, 1].mean()
            uy_ni = u_ni[ni_m, 1].mean()
            ratio = uy_ni / uy_cl if abs(uy_cl) > 1e-15 else 0.0
            assert 0.88 < ratio < 1.12, (
                f"x={xp:+.2f}: Nitsche/classical uy ratio {ratio:.3f} "
                f"outside [0.88, 1.12] (cl={uy_cl:.4e}, ni={uy_ni:.4e})"
            )

        # ---- Check 4: Tip ratio ----
        # Tip nodes are on the Neumann face — slightly lower accuracy due
        # to diffuse BC enforcement at the loaded boundary.
        tip_ratio = abs(tip_uy_ni / tip_uy_cl)
        assert 0.88 < tip_ratio < 1.05, (
            f"Tip deflection ratio {tip_ratio:.3f} outside [0.88, 1.05]: "
            f"classical={tip_uy_cl:.4e}, nitsche={tip_uy_ni:.4e}"
        )


# ===================================================================
# Twisting cube: small deformation (π/36 rotation, ~5°)
# ===================================================================
# Rotation constants (π/36 about z-axis through cube center)
_TWIST_THETA = np.pi / 36
_TWIST_COS = np.cos(_TWIST_THETA)
_TWIST_SIN = np.sin(_TWIST_THETA)
_TWIST_CX, _TWIST_CY = 0.0, 0.0  # center of rotation


def _ux_top(x, y, z):
    return (x - _TWIST_CX) * (_TWIST_COS - 1.0) - (y - _TWIST_CY) * _TWIST_SIN


def _uy_top(x, y, z):
    return (x - _TWIST_CX) * _TWIST_SIN + (y - _TWIST_CY) * (_TWIST_COS - 1.0)


def _uz_top(x, y, z):
    return 0.0 * x  # return tensor-compatible zero


# Override for twisting cube (small deformation).
_TWIST_OVERRIDES = dict(
    use_mixed_precision=True,
    relative_tol=1e-8,
    absolute_tol=1e-10,
    max_iter=30,
    linear_solver="cg",
    cg_backend_assembled="amgx",
    ew_enabled=False,
    line_search_enabled=True,
    load_steps=1,
    phi_floor=1e-2,
    tangent_f64=True,
)


def _make_cube_trimesh_octree(init_depth, octree_depth, half_side=0.75):
    """Create octree mesh from a cube [-half_side, half_side]^3."""
    mesh = trimesh.creation.box(extents=(2 * half_side,) * 3)
    return trimesh_sdf.trimesh_to_octree(
        mesh,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_depth,
        octree_depth=octree_depth,
        domain_length=2.0,
        device="cuda",
    )


@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestTwistingCube:
    """Small-deformation twisting cube: classical vs Nitsche.

    Geometry: [-0.75, 0.75]^3 cube (grid-aligned at any octree depth >= 3).
    BCs:
      - Bottom (z = -0.75): fully clamped
      - Top (z = 0.75): rigid rotation of π/36 (~5°) about z-axis
    1 load step, E=10, ν=0.3.
    """

    CONFIG = "configs.config_nitsche_example"
    HALF_SIDE = 0.75

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_twisting_cube(self):
        """Classical conforming FEM vs Nitsche on small-deformation twisting cube.

        Uses sharp_volumetric + heuristic_beta for accurate Nitsche results.
        Grid-aligned geometry (0.75) enables [0.95, 1.05] tolerance.
        """
        hs = self.HALF_SIDE

        # ---- Classical FEM on conforming uniform hex mesh ----
        from core.neoHookean import OctreeNeoHookean

        nodes_cl, elems_cl = make_uniform_hex_mesh(
            20, 20, 20, [-hs, -hs, -hs], [hs, hs, hs]
        )

        classical = OctreeNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/classical_twist",
                "octree_mesh": {"nodes": nodes_cl, "elements": elems_cl},
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, -hs)): 0.0,
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, hs)): (
                        _ux_top,
                        _uy_top,
                        _uz_top,
                    ),
                },
                "neumann_bcs_dict": {},
                **_TWIST_OVERRIDES,
            },
        )
        classical.forward()

        # ---- Nitsche FEM on octree mesh from trimesh ----
        from core.nitsche_field import NitscheNeoHookean

        cube_mesh = trimesh.creation.box(extents=(2 * hs,) * 3)
        coarse = _make_cube_trimesh_octree(init_depth=6, octree_depth=6, half_side=hs)
        h_coarse = 2.0 / (2**6)
        h_quad = h_coarse / 2.0
        epsilon = h_quad / 2.0
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(
            cube_mesh, device="cuda"
        )

        twist_overrides = dict(_TWIST_OVERRIDES)
        twist_overrides["load_steps"] = 3

        nitsche = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/nitsche_twist",
                "octree_mesh": coarse,
                "phase_field_epsilon": epsilon,
                "sharp_volumetric": True,
                "heuristic_beta": True,
                "geometric_func": geometric_func,
                "quad_refinement_depth": 1,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, -hs)): 0.0,
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, hs)): (
                        _ux_top,
                        _uy_top,
                        _uz_top,
                    ),
                },
                "neumann_bcs_dict": {},
                **twist_overrides,
            },
        )
        nitsche.forward()

        # ---- Both converge ----
        assert (
            classical.final_residual < 1e-4
        ), f"Classical residual {classical.final_residual:.3e}"
        assert (
            nitsche.final_residual < 1e-4
        ), f"Nitsche residual {nitsche.final_residual:.3e}"

        # ---- Extract solutions ----
        u_cl = classical.u.dof_values.numpy().reshape(-1, 3)
        nodes_cl_np = np.asarray(nodes_cl)

        u_ni = nitsche.u.dof_values.numpy().reshape(-1, 3)
        ni_nodes = coarse["nodes"]
        nodes_ni_np = (
            ni_nodes.cpu().numpy()
            if isinstance(ni_nodes, torch.Tensor)
            else np.asarray(ni_nodes)
        )
        phi_ni = nitsche._phi_nodal.cpu().numpy()
        inside_mask = phi_ni > 0.9

        # ---- Check 1: Top face rotates in both ----
        cl_top = np.abs(nodes_cl_np[:, 2] - hs) < 0.01
        assert (
            np.abs(u_cl[cl_top, 0]).max() > 1e-4
        ), "Classical top should have non-trivial x-displacement"

        ni_top = (np.abs(nodes_ni_np[:, 2] - hs) < 0.15) & inside_mask
        assert ni_top.any(), "No Nitsche nodes near top face"
        assert (
            np.abs(u_ni[ni_top, 0]).max() > 1e-4
        ), "Nitsche top should have non-trivial x-displacement"

        # ---- Check 2: Bottom is clamped in classical ----
        cl_bot = np.abs(nodes_cl_np[:, 2] - (-hs)) < 0.01
        assert np.abs(u_cl[cl_bot]).max() < 1e-6, "Classical bottom not zero"

        # ---- Check 3: In-plane rotation |u_xy| profile at z-slices ----
        # Grid-aligned geometry (0.75) tightens tolerance from [0.88, 1.12]
        # to [0.91, 1.05].  Remaining error is from the diffuse Dirichlet
        # enforcement near the clamped face.
        z_stations = [-0.3, 0.0, 0.3]
        for zp in z_stations:
            cl_m = np.abs(nodes_cl_np[:, 2] - zp) < 0.06
            ni_m = (np.abs(nodes_ni_np[:, 2] - zp) < 0.1) & inside_mask
            assert ni_m.any(), f"No Nitsche interior nodes at z={zp}"
            uxy_cl = np.sqrt(u_cl[cl_m, 0] ** 2 + u_cl[cl_m, 1] ** 2).mean()
            uxy_ni = np.sqrt(u_ni[ni_m, 0] ** 2 + u_ni[ni_m, 1] ** 2).mean()
            ratio = uxy_ni / uxy_cl if uxy_cl > 1e-15 else 0.0
            assert 0.91 < ratio < 1.05, (
                f"z={zp:+.1f}: Nitsche/classical |uxy| ratio {ratio:.3f} "
                f"outside [0.91, 1.05] (cl={uxy_cl:.4e}, ni={uxy_ni:.4e})"
            )

        # ---- Check 4: Axial uz should be small at midplane ----
        # The twist is purely in-plane; uz should be negligible at z=0.
        ni_mid = (np.abs(nodes_ni_np[:, 2]) < 0.1) & inside_mask
        uxy_mid_ni = np.sqrt(u_ni[ni_mid, 0] ** 2 + u_ni[ni_mid, 1] ** 2).mean()
        uz_mid_ni = np.abs(u_ni[ni_mid, 2]).mean()
        assert (
            uz_mid_ni < 0.3 * uxy_mid_ni
        ), f"Midplane uz={uz_mid_ni:.4e} too large vs |uxy|={uxy_mid_ni:.4e}"

    def test_nitsche_only(self):
        """Nitsche solver only — isolate from classical solver kernel cache.

        Uses trim_exterior=True with standard quadrature (no quad_octree_mesh).
        """
        hs = self.HALF_SIDE
        from core.nitsche_field import NitscheNeoHookean

        coarse = _make_cube_trimesh_octree(init_depth=4, octree_depth=5, half_side=hs)
        epsilon = 0.03

        overrides = dict(_TWIST_OVERRIDES)
        overrides["phi_floor"] = 0.1
        overrides["load_steps"] = 3

        nitsche = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/nitsche_only",
                "octree_mesh": coarse,
                "phase_field_epsilon": epsilon,
                "trim_exterior": True,
                "beta_max": 2000.0,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, -hs)): 0.0,
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, hs)): (
                        _ux_top,
                        _uy_top,
                        _uz_top,
                    ),
                },
                "neumann_bcs_dict": {},
                **overrides,
            },
        )
        nitsche.forward()
        assert (
            nitsche.final_residual < 1e-4
        ), f"Nitsche residual {nitsche.final_residual:.3e}"

    def test_nitsche_trim_exterior(self):
        """Nitsche solver with trim_exterior=True — removes void elements.

        After trimming, no hard Dirichlet BCs are applied; all boundary
        conditions are enforced purely through the Nitsche method.
        Uses small uniform compression (z-displacement on top face).

        NOTE: adaptive quadrature (quad_octree_mesh) is NOT used here because
        the two-integration split has a quadrature mismatch between standard
        and correction that prevents convergence for pure-Nitsche (no hard BC)
        problems.  Adaptive quadrature + trim_exterior is a known limitation.
        Adaptive quadrature itself is tested by test_refined_quad_mesh_mixed_precision.
        """
        hs = self.HALF_SIDE
        from core.nitsche_field import NitscheNeoHookean

        coarse = _make_cube_trimesh_octree(init_depth=3, octree_depth=4, half_side=hs)
        epsilon = 0.06

        overrides = dict(_TWIST_OVERRIDES)
        overrides["tangent_f64"] = True
        overrides["load_steps"] = 1
        overrides["max_iter"] = 30
        # Higher phi_floor stabilizes exterior DOFs of boundary elements that
        # have no hard BCs and negligible Nitsche constraint (|grad_phi| ≈ 0).
        overrides["phi_floor"] = 0.1

        # Small uniform compression: clamp bottom, push top down by 0.001
        nitsche = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/nitsche_trim",
                "octree_mesh": coarse,
                "phase_field_epsilon": epsilon,
                "trim_exterior": True,
                "beta_max": 2000.0,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, -hs)): 0.0,
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, hs)): (
                        0.0,
                        0.0,
                        -0.001,
                    ),
                },
                "neumann_bcs_dict": {},
                **overrides,
            },
        )
        nitsche.forward()
        assert (
            nitsche.final_residual < 1e-4
        ), f"Nitsche trim residual {nitsche.final_residual:.3e}"

        # Verify non-trivial displacement inside the geometry
        u = nitsche.u.dof_values.numpy().reshape(-1, 3)
        phi = nitsche._phi_nodal.cpu().numpy()
        inside = phi > 0.3
        assert (
            np.abs(u[inside]).max() > 1e-4
        ), f"Displacement too small: {np.abs(u[inside]).max():.4e}"


# ===================================================================
# Rotation box: grid-aligned geometry for high-accuracy comparison
# ===================================================================
# [-0.75, 0.75]^3 box — 0.75 aligns with octree grid at any depth >= 3
# (0.75 / (2/2^d) = 3*2^(d-3), always integer for d >= 3).
# This eliminates grid-mismatch error and allows [0.95, 1.05] tolerance.

_ROT_THETA = np.pi / 4 / 50  # same small rotation as compare_uniform_octree
_ROT_COS = np.cos(_ROT_THETA)
_ROT_SIN = np.sin(_ROT_THETA)


def _rot_ux_top(x, y, z):
    """x-displacement: rotation about z-axis through origin."""
    return x * (_ROT_COS - 1.0) - y * _ROT_SIN


def _rot_uy_top(x, y, z):
    return x * _ROT_SIN + y * (_ROT_COS - 1.0)


def _rot_uz_top(x, y, z):
    return 0.0 * x


_ROT_OVERRIDES = dict(
    use_mixed_precision=True,
    relative_tol=1e-8,
    absolute_tol=1e-10,
    max_iter=30,
    linear_solver="cg",
    cg_backend_assembled="amgx",
    ew_enabled=False,
    line_search_enabled=True,
    load_steps=1,
    phi_floor=1e-4,
    tangent_f64=True,
)


def _make_box_trimesh_octree(init_depth, octree_depth, half_side=0.75):
    mesh = trimesh.creation.box(extents=(2 * half_side,) * 3)
    return mesh, trimesh_sdf.trimesh_to_octree(
        mesh,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_depth,
        octree_depth=octree_depth,
        domain_length=2.0,
        device="cuda",
    )


@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestRotationBoxComparison:
    """Classical vs Nitsche on grid-aligned [-0.75, 0.75]^3 box.

    Replicates the compare_uniform_octree.py scenario as a unit test.
    Both diffused and sharp volume are tested against classical FEM.

    BCs:
      - Bottom (z = -0.75): fully clamped
      - Top (z = +0.75): small rotation (π/200) about z-axis
    """

    CONFIG = "configs.config_nitsche_example"
    HS = 0.75

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def _solve_classical(self, n=48):
        """Classical FEM on uniform hex mesh."""
        from core.neoHookean import OctreeNeoHookean

        hs = self.HS
        nodes, elems = make_uniform_hex_mesh(n, n, n, [-hs, -hs, -hs], [hs, hs, hs])
        solver = OctreeNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + "/classical_rot",
                "octree_mesh": {"nodes": nodes, "elements": elems},
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, -hs)): 0.0,
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, hs)): (
                        _rot_ux_top,
                        _rot_uy_top,
                        _rot_uz_top,
                    ),
                },
                "neumann_bcs_dict": {},
                **_ROT_OVERRIDES,
            },
        )
        solver.forward()
        return solver

    def _solve_nitsche(self, label, sharp_volumetric=True):
        """Nitsche FEM on octree mesh from trimesh box."""
        from core.nitsche_field import NitscheNeoHookean

        hs = self.HS
        mesh, coarse = _make_box_trimesh_octree(
            init_depth=6, octree_depth=6, half_side=hs
        )
        h_coarse = 2.0 / (2**6)
        h_quad = h_coarse / 2.0
        epsilon = h_quad / 2.0
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")

        solver = NitscheNeoHookean(
            self.CONFIG,
            override_params={
                "output_root": self.tmp + f"/{label}",
                "octree_mesh": coarse,
                "phase_field_epsilon": epsilon,
                "sharp_volumetric": sharp_volumetric,
                "heuristic_beta": True,
                "geometric_func": geometric_func,
                "quad_refinement_depth": 1,
                "trim_exterior": True,
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, -hs)): 0.0,
                    lambda x, y, z: torch.isclose(z, torch.full_like(z, hs)): (
                        _rot_ux_top,
                        _rot_uy_top,
                        _rot_uz_top,
                    ),
                },
                "neumann_bcs_dict": {},
                **_ROT_OVERRIDES,
            },
        )
        solver.forward()
        return solver

    def _compare_solutions(self, cl, ni, label, tol_lo=0.95, tol_hi=1.05):
        """Compare classical vs Nitsche solutions at interior z-stations."""
        assert cl.final_residual < _RESIDUAL_TOL
        assert ni.final_residual < _RESIDUAL_TOL

        u_cl = cl.u.dof_values.numpy().reshape(-1, 3)
        n_cl = np.asarray(cl.params.octree_mesh["nodes"])

        u_ni = ni.u.dof_values.numpy().reshape(-1, 3)
        n_ni_t = ni.params.octree_mesh["nodes"]
        n_ni = (
            n_ni_t.cpu().numpy()
            if isinstance(n_ni_t, torch.Tensor)
            else np.asarray(n_ni_t)
        )
        phi = ni._phi_nodal.cpu().numpy()
        inside = phi > 0.9

        hs = self.HS
        # Use z-stations in upper half — Nitsche under-enforces BC near
        # the clamped face, so lower stations have larger error.
        for zfrac in [0.5, 0.6, 0.7]:
            z_val = -hs + zfrac * 2 * hs
            cl_m = np.abs(n_cl[:, 2] - z_val) < 0.03
            ni_m = (np.abs(n_ni[:, 2] - z_val) < 0.05) & inside
            assert ni_m.any(), f"No Nitsche nodes at z={z_val}"
            mag_cl = np.linalg.norm(u_cl[cl_m], axis=1).mean()
            mag_ni = np.linalg.norm(u_ni[ni_m], axis=1).mean()
            ratio = mag_ni / mag_cl if mag_cl > 1e-15 else 0.0
            assert tol_lo < ratio < tol_hi, (
                f"{label} z={z_val:+.2f}: |u| ratio {ratio:.4f} "
                f"outside [{tol_lo}, {tol_hi}] "
                f"(cl={mag_cl:.4e}, ni={mag_ni:.4e})"
            )

    def test_rotation_sharp_volume(self):
        """Sharp volume Nitsche vs classical on grid-aligned box."""
        cl = self._solve_classical()
        ni = self._solve_nitsche("nitsche_sharp", sharp_volumetric=True)
        self._compare_solutions(cl, ni, "sharp", tol_lo=0.95, tol_hi=1.05)

    def test_rotation_diffused_volume(self):
        """Diffused volume Nitsche vs classical on grid-aligned box."""
        cl = self._solve_classical()
        ni = self._solve_nitsche("nitsche_diffused", sharp_volumetric=False)
        self._compare_solutions(cl, ni, "diffused", tol_lo=0.95, tol_hi=1.05)


# ===================================================================
# Refined vs non-refined gradient comparison for ShapeOpt
# ===================================================================


def _make_beam_sdf(nodes, lo=(-0.75, -0.25, -0.25), hi=(0.75, 0.25, 0.25)):
    """Signed distance to a rectangular box [lo, hi]. Negative inside."""
    nodes_t = torch.as_tensor(nodes, dtype=torch.float32)
    dx = torch.maximum(
        torch.tensor(lo[0]) - nodes_t[:, 0], nodes_t[:, 0] - torch.tensor(hi[0])
    )
    dy = torch.maximum(
        torch.tensor(lo[1]) - nodes_t[:, 1], nodes_t[:, 1] - torch.tensor(hi[1])
    )
    dz = torch.maximum(
        torch.tensor(lo[2]) - nodes_t[:, 2], nodes_t[:, 2] - torch.tensor(hi[2])
    )
    outside = torch.sqrt(
        torch.clamp(dx, min=0.0) ** 2
        + torch.clamp(dy, min=0.0) ** 2
        + torch.clamp(dz, min=0.0) ** 2
    )
    inside = torch.minimum(
        torch.maximum(dx, torch.maximum(dy, dz)), torch.zeros_like(dx)
    )
    return outside + inside


def _one_iteration_gradient(problem):
    """Run one forward + tape + backward pass, return SDF gradient as numpy."""
    tape = wp.Tape()
    problem.e.dof_values.requires_grad = True
    problem.step(tape=tape)
    with tape:
        obj = problem._objective_function(tape)
    tape.zero()
    tape.backward(loss=obj)
    problem._post_tape_backward()
    return problem.e.dof_values.grad.numpy().copy()


@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestRefinedGradientComparison:
    """Verify refined vs non-refined quadrature produce close SDF gradients.

    Uses a small cantilever beam with ShapeOpt. Both setups share the same
    coarse uniform hex mesh and BCs. The refined path adds geometric_func
    (trimesh SDF) + quad_refinement_depth=1 for boundary elements.

    For a box geometry, the analytical SDF is piecewise linear, so trilinear
    interpolation from coarse nodes is nearly exact. The gradients should
    therefore be very close between the two paths.
    """

    CONFIG = "configs.config_shape_opt_cantilever"

    BEAM_LO = (-0.75, -0.25, -0.25)
    BEAM_HI = (0.75, 0.25, 0.25)

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_refined_vs_nonrefined_gradient(self):
        """First-iteration SDF gradients match between refined and non-refined."""
        from core.shape_opt import ShapeOpt

        # ---- Small uniform hex mesh with padding ----
        nx, ny, nz = 16, 8, 8
        pad = 0.25
        lo = [self.BEAM_LO[i] - pad for i in range(3)]
        hi = [self.BEAM_HI[i] + pad for i in range(3)]
        nodes, elements = make_uniform_hex_mesh(nx, ny, nz, lo, hi)

        hx = (hi[0] - lo[0]) / nx
        hy = (hi[1] - lo[1]) / ny
        hz = (hi[2] - lo[2]) / nz
        h_min = min(hx, hy, hz)
        epsilon = 0.5 * h_min

        sdf = _make_beam_sdf(nodes, self.BEAM_LO, self.BEAM_HI)

        common = {
            "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
            "phase_field_epsilon": epsilon,
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
            "output_per_step": 0,
            "dirichlet_bcs_dict": {
                lambda x, y, z: torch.isclose(
                    x, torch.full_like(x, self.BEAM_LO[0])
                ): 0.0,
            },
            "neumann_bcs_dict": {
                lambda x, y, z: torch.isclose(x, torch.full_like(x, self.BEAM_HI[0])): (
                    0.0,
                    -0.001,
                    0.0,
                ),
            },
        }

        # ---- Non-refined ----
        prob_base = ShapeOpt(
            self.CONFIG,
            override_params={
                **common,
                "output_root": self.tmp + "/base",
            },
        )
        grad_base = _one_iteration_gradient(prob_base)

        # ---- Refined (trimesh geometric_func + quad_refinement_depth=1) ----
        beam_mesh = trimesh.creation.box(
            extents=(
                self.BEAM_HI[0] - self.BEAM_LO[0],
                self.BEAM_HI[1] - self.BEAM_LO[1],
                self.BEAM_HI[2] - self.BEAM_LO[2],
            )
        )
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(
            beam_mesh, device="cuda"
        )

        prob_ref = ShapeOpt(
            self.CONFIG,
            override_params={
                **common,
                "output_root": self.tmp + "/refined",
                "geometric_func": geometric_func,
                "quad_refinement_depth": 1,
            },
        )
        grad_ref = _one_iteration_gradient(prob_ref)

        # ---- Compare ----
        # Cosine similarity (global direction agreement)
        dot = np.dot(grad_base.ravel(), grad_ref.ravel())
        norm_base = np.linalg.norm(grad_base)
        norm_ref = np.linalg.norm(grad_ref)
        cosine_sim = dot / (norm_base * norm_ref + 1e-30)

        # Norm ratio (magnitudes should be within a factor of 2)
        norm_ratio = norm_ref / (norm_base + 1e-30)

        print(
            f"Gradient comparison: cosine_sim={cosine_sim:.6f}, "
            f"norm_ratio={norm_ratio:.6f}, "
            f"norm_base={norm_base:.4e}, norm_ref={norm_ref:.4e}"
        )

        assert cosine_sim > 0.95, (
            f"Cosine similarity {cosine_sim:.4f} < 0.95 — gradients point "
            f"in different directions"
        )
        assert 0.5 < norm_ratio < 2.0, (
            f"Norm ratio {norm_ratio:.4f} outside [0.5, 2.0] — gradient "
            f"magnitudes diverge too much"
        )


# ===================================================================
# Sharp vs diffuse volumetric gradient comparison for ShapeOpt
# ===================================================================


class TestSharpVsDiffuseGradient:
    """Verify sharp_volumetric=True and False produce close SDF gradients.

    Uses a small cantilever beam on a uniform hex mesh.  Both setups share
    the same geometry, BCs, and mesh — only sharp_volumetric differs.
    sharp_volumetric=True auto-enables trim_exterior, which removes elements
    fully outside the domain.
    """

    CONFIG = "configs.config_shape_opt_cantilever"

    BEAM_LO = (-0.75, -0.25, -0.25)
    BEAM_HI = (0.75, 0.25, 0.25)

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_sharp_vs_diffuse_gradient(self):
        """First-iteration SDF gradients match between sharp and diffuse."""
        from core.shape_opt import ShapeOpt

        # ---- Denser uniform hex mesh with padding ----
        # Need enough interior elements so sharp vs diffuse comparison is
        # meaningful (boundary elements are a small fraction of total).
        nx, ny, nz = 32, 16, 16
        pad = 0.25
        lo = [self.BEAM_LO[i] - pad for i in range(3)]
        hi = [self.BEAM_HI[i] + pad for i in range(3)]
        nodes, elements = make_uniform_hex_mesh(nx, ny, nz, lo, hi)

        hx = (hi[0] - lo[0]) / nx
        hy = (hi[1] - lo[1]) / ny
        hz = (hi[2] - lo[2]) / nz
        h_min = min(hx, hy, hz)
        epsilon = 0.5 * h_min

        sdf = _make_beam_sdf(nodes, self.BEAM_LO, self.BEAM_HI)

        common = {
            "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
            "phase_field_epsilon": epsilon,
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
            "output_per_step": 0,
            "dirichlet_bcs_dict": {
                lambda x, y, z: torch.isclose(
                    x, torch.full_like(x, self.BEAM_LO[0])
                ): 0.0,
            },
            "neumann_bcs_dict": {
                lambda x, y, z: torch.isclose(
                    x, torch.full_like(x, self.BEAM_HI[0])
                ): (0.0, -0.001, 0.0),
            },
        }

        # ---- Diffuse ----
        prob_diffuse = ShapeOpt(
            self.CONFIG,
            override_params={
                **common,
                "output_root": self.tmp + "/diffuse",
                "sharp_volumetric": False,
            },
        )
        grad_diffuse = _one_iteration_gradient(prob_diffuse)

        # ---- Sharp (auto-enables trim_exterior) ----
        prob_sharp = ShapeOpt(
            self.CONFIG,
            override_params={
                **common,
                "output_root": self.tmp + "/sharp",
                "sharp_volumetric": True,
            },
        )
        grad_sharp = _one_iteration_gradient(prob_sharp)

        # ---- Compare on common nodes ----
        # sharp_volumetric auto-enables trim_exterior, which compacts nodes.
        # Match nodes by physical position to compare gradients.
        pos_d = prob_diffuse._prev_octree_nodes.cpu().numpy()
        pos_s = prob_sharp._prev_octree_nodes.cpu().numpy()

        from scipy.spatial import cKDTree

        tree = cKDTree(pos_d)
        dist, idx = tree.query(pos_s)
        matched = dist < 1e-8
        assert matched.sum() > 0.5 * len(pos_s), "Too few matched nodes"

        g_d = grad_diffuse[idx[matched]]
        g_s = grad_sharp[matched]

        dot = np.dot(g_d.ravel(), g_s.ravel())
        norm_d = np.linalg.norm(g_d)
        norm_s = np.linalg.norm(g_s)
        cosine_sim = dot / (norm_d * norm_s + 1e-30)
        norm_ratio = norm_s / (norm_d + 1e-30)

        print(
            f"Sharp vs diffuse: cosine_sim={cosine_sim:.6f}, "
            f"norm_ratio={norm_ratio:.6f}, "
            f"norm_diffuse={norm_d:.4e}, norm_sharp={norm_s:.4e}, "
            f"matched={matched.sum()}/{len(pos_s)} nodes"
        )

        # With the compute-graph-preserving fix (phi instead of 1.0 inside),
        # sharp and diffuse produce close gradients. The residual difference
        # comes from pre-baked quadrature weights (off tape) that legitimately
        # differ: sharp uses H(phi-0.5) while diffuse uses smooth phi, giving
        # slightly different forward solutions and thus adjoint contributions.
        assert cosine_sim > 0.85, (
            f"Cosine similarity {cosine_sim:.4f} < 0.85 — sharp and diffuse "
            f"gradients point in different directions"
        )
        assert 0.5 < norm_ratio < 2.0, (
            f"Norm ratio {norm_ratio:.4f} outside [0.5, 2.0] — gradient "
            f"magnitudes diverge too much"
        )


def _gd_iter(solver, label, it, lr=1e-3):
    """Run one forward → objective → backward → GD step on *solver*.

    Returns ``(obj_val, se_raw, vol_raw, grad_np, e_np)``.
    """
    tape = wp.Tape()
    solver.e.dof_values.requires_grad = True
    solver.step(tape=tape)
    with tape:
        obj = solver._objective_function(tape)
    tape.zero()
    solver.e.dof_values.grad.zero_()
    tape.backward(loss=obj)
    solver._post_tape_backward()
    obj_val = float(obj.numpy()[0])
    se_raw = solver._last_se_raw
    vol_raw = solver._last_vol_raw
    grad_np = wp.to_torch(solver.e.dof_values.grad).detach().cpu().clone().numpy()
    e_np = wp.to_torch(solver.e.dof_values).detach().cpu().clone().numpy()
    print(
        f"{label} iter {it}: obj={obj_val:.6e} SE={se_raw:.6e} vol={vol_raw:.6e}"
        f"  |grad|={np.linalg.norm(grad_np):.6e}  |e|={np.linalg.norm(e_np):.6e}"
    )
    grad = wp.clone(solver.e.dof_values.grad)
    wp.fem.linalg.array_axpy(x=grad, y=solver.e.dof_values, alpha=wp.float64(-lr))
    return obj_val, se_raw, vol_raw, grad_np, e_np


@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not available")
class TestShapeOptTrimmed:
    """Verify ShapeOptTrimmed produces same results as standalone ShapeOpt(trim_exterior=True).

    Both instances use the same mesh, BCs, and solver settings. Over 3 iterations
    the raw SE and volume values should match, since the inner solver is identical.
    """

    CONFIG = "configs.config_shape_opt_cantilever"
    BEAM_LO = (-0.75, -0.25, -0.25)
    BEAM_HI = (0.75, 0.25, 0.25)

    @pytest.fixture(autouse=True)
    def _tmp(self, tmp_path):
        self.tmp = str(tmp_path)

    def test_trimmed_vs_standalone(self):
        """Raw SE/volume match between ShapeOptTrimmed and standalone ShapeOpt."""
        from core.shape_opt import ShapeOpt, ShapeOptTrimmed

        nx, ny, nz = 16, 8, 8
        pad = 0.25
        lo = [self.BEAM_LO[i] - pad for i in range(3)]
        hi = [self.BEAM_HI[i] + pad for i in range(3)]
        nodes, elements = make_uniform_hex_mesh(nx, ny, nz, lo, hi)

        hx = (hi[0] - lo[0]) / nx
        hy = (hi[1] - lo[1]) / ny
        hz = (hi[2] - lo[2]) / nz
        h_min = min(hx, hy, hz)
        epsilon = 0.5 * h_min

        sdf = _make_beam_sdf(nodes, self.BEAM_LO, self.BEAM_HI)

        # Each instance needs its own octree_mesh copy because
        # _trim_exterior_elements modifies the dict in-place.
        def _make_common(output_dir, **extra):
            return {
                "octree_mesh": {
                    "nodes": nodes.copy() if isinstance(nodes, np.ndarray) else nodes.clone(),
                    "elements": elements.copy() if isinstance(elements, np.ndarray) else elements.clone(),
                    "sdf": sdf.clone() if isinstance(sdf, torch.Tensor) else torch.tensor(sdf).clone(),
                },
                "phase_field_epsilon": epsilon,
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
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(
                        x, torch.full_like(x, self.BEAM_LO[0])
                    ): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(
                        x, torch.full_like(x, self.BEAM_HI[0])
                    ): (0.0, -0.001, 0.0),
                },
                **extra,
            }

        # ---- Standalone ShapeOpt with trim_exterior=True ----
        standalone = ShapeOpt(
            self.CONFIG,
            override_params=_make_common(
                self.tmp + "/standalone", trim_exterior=True
            ),
        )

        # ---- ShapeOptTrimmed (untrimmed outer, trimmed inner) ----
        trimmed = ShapeOptTrimmed(
            self.CONFIG,
            override_params=_make_common(self.tmp + "/trimmed"),
        )

        # ---- Verify first iteration SE/volume match ----
        tape_s = wp.Tape()
        standalone.e.dof_values.requires_grad = True
        standalone.step(tape=tape_s)
        with tape_s:
            standalone._objective_function(tape_s)
        se_s = standalone._last_se_raw
        vol_s = standalone._last_vol_raw

        tape_t = wp.Tape()
        trimmed.e.dof_values.requires_grad = True
        trimmed.step(tape=tape_t)
        with tape_t:
            trimmed._objective_function(tape_t)
        se_t = trimmed._inner._last_se_raw
        vol_t = trimmed._inner._last_vol_raw

        # Verify node mapping correctness (inner created after first step)
        inner_nodes_t = trimmed._inner._prev_octree_nodes.cpu()
        outer_nodes_t = trimmed._prev_octree_nodes.cpu()
        active_idx = trimmed._trim_active_indices.long().cpu()
        mapped = outer_nodes_t[active_idx]
        assert torch.allclose(mapped, inner_nodes_t, atol=1e-6), (
            "Node mapping mismatch between outer and inner mesh"
        )

        print(
            f"Iter 0: standalone SE={se_s:.6e} vol={vol_s:.6e}, "
            f"trimmed SE={se_t:.6e} vol={vol_t:.6e}"
        )
        if abs(se_s) > 1e-15:
            assert abs(se_t - se_s) / abs(se_s) < 1e-4, "SE mismatch at iter 0"
        if abs(vol_s) > 1e-15:
            assert abs(vol_t - vol_s) / abs(vol_s) < 1e-4, "Volume mismatch at iter 0"

        # ---- Run both solvers in lock-step for comparison ----
        del tape_s, tape_t
        n_iters = 3
        objs_trimmed = []
        objs_standalone = []
        for it in range(n_iters):
            obj_t, _, _, grad_t, e_t = _gd_iter(trimmed, "Trimmed", it)
            obj_s, _, _, grad_s, e_s = _gd_iter(standalone, "Standalone", it)
            objs_trimmed.append(obj_t)
            objs_standalone.append(obj_s)

            # Compare SDF at active nodes
            active = trimmed._trim_active_indices.long().cpu().numpy()
            e_t_active = e_t[active]  # outer e at active nodes (BEFORE GD step — already applied above)
            sdf_diff = np.linalg.norm(e_t_active - e_s)
            sdf_norm = np.linalg.norm(e_s)

            # Compare gradients at active nodes
            grad_t_active = grad_t[active]
            grad_diff = np.linalg.norm(grad_t_active - grad_s)
            grad_norm = np.linalg.norm(grad_s)

            obj_rel = abs(obj_t - obj_s) / abs(obj_s) if abs(obj_s) > 0 else 0
            print(f"  Iter {it} comparison: obj_rel={obj_rel:.6e}"
                  f"  sdf_diff={sdf_diff:.6e} (rel={sdf_diff/sdf_norm:.6e})"
                  f"  grad_diff={grad_diff:.6e} (rel={grad_diff/grad_norm:.6e})")

        assert objs_trimmed[-1] < objs_trimmed[0], (
            f"Trimmed objective did not decrease: {objs_trimmed[0]:.6e} -> {objs_trimmed[-1]:.6e}"
        )
        assert objs_standalone[-1] < objs_standalone[0], (
            f"Standalone objective did not decrease: "
            f"{objs_standalone[0]:.6e} -> {objs_standalone[-1]:.6e}"
        )
        final_rel = abs(objs_trimmed[-1] - objs_standalone[-1]) / abs(objs_standalone[-1])
        assert final_rel < 0.05, (
            f"Final objective diverged too much: standalone={objs_standalone[-1]:.6e} "
            f"trimmed={objs_trimmed[-1]:.6e} rel={final_rel:.6e}"
        )

    def test_fd_gradient_check(self):
        """Finite-difference validation of tape gradient at iter 0.

        Uses a single solver so normalization references (from
        ``_normalized_scalar``) are set once from the unperturbed forward and
        shared across all FD perturbations.

        Interior nodes (away from SDF=0) should match to <1%.
        Boundary nodes (SDF~0) have a known ~10% gap because adaptive
        quadrature, beta, and standard weights are rebuilt off-tape.
        """
        from core.shape_opt import ShapeOpt

        nx, ny, nz = 16, 8, 8
        pad = 0.25
        lo = [self.BEAM_LO[i] - pad for i in range(3)]
        hi = [self.BEAM_HI[i] + pad for i in range(3)]
        nodes, elements = make_uniform_hex_mesh(nx, ny, nz, lo, hi)
        h_min = min((hi[i] - lo[i]) / [nx, ny, nz][i] for i in range(3))
        epsilon = 0.5 * h_min
        sdf = _make_beam_sdf(nodes, self.BEAM_LO, self.BEAM_HI)

        def _make_params(output_dir, **extra):
            return {
                "octree_mesh": {
                    "nodes": nodes.copy() if isinstance(nodes, np.ndarray) else nodes.clone(),
                    "elements": elements.copy() if isinstance(elements, np.ndarray) else elements.clone(),
                    "sdf": sdf.clone() if isinstance(sdf, torch.Tensor) else torch.tensor(sdf).clone(),
                },
                "phase_field_epsilon": epsilon,
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
                "dirichlet_bcs_dict": {
                    lambda x, y, z: torch.isclose(
                        x, torch.full_like(x, self.BEAM_LO[0])
                    ): 0.0,
                },
                "neumann_bcs_dict": {
                    lambda x, y, z: torch.isclose(
                        x, torch.full_like(x, self.BEAM_HI[0])
                    ): (0.0, -0.001, 0.0),
                },
                **extra,
            }

        # Forward + backward to get tape gradient and set normalization refs
        sa = ShapeOpt(
            self.CONFIG,
            override_params=_make_params(self.tmp + "/fd", trim_exterior=True),
        )
        tape = wp.Tape()
        sa.e.dof_values.requires_grad = True
        sa.step(tape=tape)
        with tape:
            obj = sa._objective_function(tape)
        tape.zero()
        tape.backward(loss=obj)
        sa._post_tape_backward()

        grad_tape = wp.to_torch(sa.e.dof_values.grad).detach().cpu().clone().numpy()
        obj_base = float(obj.numpy()[0])
        e_base = wp.to_torch(sa.e.dof_values).detach().cpu().clone()

        # Pick test nodes: largest-gradient, a mid-gradient, a small-gradient
        sorted_idx = np.argsort(np.abs(grad_tape))[::-1]
        check_nodes = [
            sorted_idx[0],                   # boundary (SDF~0)
            sorted_idx[len(sorted_idx) // 4],  # interior
            sorted_idx[len(sorted_idx) // 2],  # interior
        ]

        eps_fd = 1e-4
        for node in check_nodes:
            fd_objs = []
            for sign in [+1, -1]:
                e_pert = e_base.clone()
                e_pert[node] += sign * eps_fd
                wp.copy(
                    src=wp.from_torch(e_pert.cuda(), dtype=wp.float64),
                    dest=sa.e.dof_values,
                )
                sa.step(tape=None)
                obj_pert = sa._objective_function(None)
                fd_objs.append(float(obj_pert.numpy()[0]))

            fd_grad = (fd_objs[0] - fd_objs[1]) / (2 * eps_fd)
            tape_val = grad_tape[node]

            if abs(fd_grad) < 1e-12:
                print(f"  node={node}: tape={tape_val:.6e}  FD~0, skipping")
                continue

            ratio = tape_val / fd_grad
            print(f"  node={node}: tape={tape_val:.6e}  FD={fd_grad:.6e}  ratio={ratio:.4f}")

            # Boundary nodes (SDF~0): allow up to 15% due to off-tape
            # dependencies (beta, adaptive quadrature, standard weights).
            # Interior nodes: expect <1% error.
            sdf_val = e_base[node].item()
            if abs(sdf_val) < epsilon:
                assert abs(ratio - 1.0) < 0.15, (
                    f"Boundary node {node} (SDF={sdf_val:.4f}): "
                    f"tape/FD ratio={ratio:.4f}, expected within 15%"
                )
            else:
                assert abs(ratio - 1.0) < 0.01, (
                    f"Interior node {node} (SDF={sdf_val:.4f}): "
                    f"tape/FD ratio={ratio:.4f}, expected within 1%"
                )

        # Restore base state
        wp.copy(
            src=wp.from_torch(e_base.cuda(), dtype=wp.float64),
            dest=sa.e.dof_values,
        )

    # ------------------------------------------------------------------
    # Remeshing incremental tests (Steps 2–4)
    # ------------------------------------------------------------------

    def _make_remesh_overrides(self, output_dir, **extra):
        """Build override_params with remeshing_trimesh for the small test geometry."""
        import trimesh as _trimesh

        nx, ny, nz = 16, 8, 8
        pad = 0.25
        lo = [self.BEAM_LO[i] - pad for i in range(3)]
        hi = [self.BEAM_HI[i] + pad for i in range(3)]
        nodes, elements = make_uniform_hex_mesh(nx, ny, nz, lo, hi)
        hx = (hi[0] - lo[0]) / nx
        hy = (hi[1] - lo[1]) / ny
        hz = (hi[2] - lo[2]) / nz
        h_min = min(hx, hy, hz)
        epsilon = 0.5 * h_min

        beam_center = [(self.BEAM_LO[i] + self.BEAM_HI[i]) / 2 for i in range(3)]
        beam_extents = [self.BEAM_HI[i] - self.BEAM_LO[i] for i in range(3)]
        box = _trimesh.creation.box(extents=beam_extents)
        box.apply_translation(beam_center)
        box = box.subdivide().subdivide()
        clip_bbox = (tuple(lo), tuple(hi))

        depth = extra.pop("remesh_octree_depth", 4)
        init_depth = extra.pop("remesh_init_octree_depth", 2)
        return {
            "remeshing": True,
            "remesh_octree_depth": depth,
            "remesh_init_octree_depth": init_depth,
            "remeshing_trimesh": {
                "mesh": box,
                "mesh_center": beam_center,
                "h_target": h_min,
                "clip_bbox": clip_bbox,
            },
            "phase_field_epsilon": epsilon,
            "use_mixed_precision": True,
            "reinit_sdf": True,
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
            "dirichlet_bcs_dict": {
                lambda x, y, z: torch.isclose(x, torch.full_like(x, self.BEAM_LO[0])): 0.0,
            },
            "neumann_bcs_dict": {
                lambda x, y, z: torch.isclose(x, torch.full_like(x, self.BEAM_HI[0])): (0.0, -0.001, 0.0),
            },
            **extra,
        }

    @pytest.fixture(scope="class")
    def trimmed_baseline_obj(self, tmp_path_factory):
        """Uniform-hex baseline (init_depth==final_depth): fixed mesh, Sussman reinit.

        All five paths share the same MC-reinitialized box as their starting
        point (via _init_remeshing_state), so their iter-0 objectives are
        identical.  After 3 GD iters the objectives should remain within 1% of
        each other.  This fixture returns the uniform-hex result as the
        reference value.
        """
        from core.shape_opt import ShapeOptTrimmed

        tmp = str(tmp_path_factory.mktemp("remesh_baseline"))
        overrides = self._make_remesh_overrides(
            tmp,
            remesh_octree_depth=4,
            remesh_init_octree_depth=4,  # same → uniform hex, no rebuild
        )
        solver = ShapeOptTrimmed(self.CONFIG, override_params=overrides)
        objs = [_gd_iter(solver, "baseline", i)[0] for i in range(3)]
        return objs[-1]

    def _run_remesh_test(self, tmp_path, trimmed_baseline_obj, label, **extra):
        """Shared body for all remeshing tests.

        Runs 3 GD iters and asserts:
        - objective decreases monotonically
        - no NaN
        - final objective within 1% of the uniform-hex baseline
        """
        import math
        from core.shape_opt import ShapeOptTrimmed

        overrides = self._make_remesh_overrides(str(tmp_path), **extra)
        solver = ShapeOptTrimmed(self.CONFIG, override_params=overrides)
        objs = [_gd_iter(solver, label, i)[0] for i in range(3)]
        assert objs[-1] < objs[0], f"{label}: objective did not decrease: {objs}"
        assert not any(math.isnan(o) for o in objs), f"{label}: NaN in objectives: {objs}"
        rel = abs(objs[-1] - trimmed_baseline_obj) / max(abs(trimmed_baseline_obj), 1e-12)
        assert rel < 0.001, (
            f"{label}: final obj {objs[-1]:.4e} deviates {rel:.2%} from baseline "
            f"{trimmed_baseline_obj:.4e} (threshold 0.1%)"
        )

    def test_neumann_rescale_kept_across_remeshing(self, tmp_path):
        """The Neumann rescale factors are computed once and never recomputed.

        Every GD iteration remeshes the inner solver (``reinit_with_sdf``).  The
        factors from its first build must survive those remeshes, and each
        rebuilt traction must still carry them.
        """
        from core.shape_opt import ShapeOptTrimmed

        overrides = self._make_remesh_overrides(
            str(tmp_path), remeshing_method="marchingcubes"
        )
        solver = ShapeOptTrimmed(self.CONFIG, override_params=overrides)
        _gd_iter(solver, "RescaleKept", 0)
        inner = solver._inner
        first = dict(inner._neumann_rescale_factors)
        assert len(first) == 1 and next(iter(first.values())) != 1.0
        for i in (1, 2):
            _gd_iter(solver, "RescaleKept", i)
            assert solver._inner is inner
            assert inner._neumann_rescale_factors == first
            loaded = wp.to_torch(inner._psi_N_qp) > 0.5
            t_bar = wp.to_torch(inner._t_bar_qp)[loaded]
            factor = next(iter(first.values()))
            assert torch.allclose(
                t_bar[:, 1], torch.full_like(t_bar[:, 1], -0.001 * factor), rtol=1e-5
            )

    def test_uniform_hex_baseline(self, tmp_path, trimmed_baseline_obj):
        """Uniform-hex path (init==final) matches itself — sanity check."""
        self._run_remesh_test(
            tmp_path, trimmed_baseline_obj, "UniformHex",
            remesh_octree_depth=4,
            remesh_init_octree_depth=4,
        )

    def test_remeshing_interpolate(self, tmp_path, trimmed_baseline_obj):
        self._run_remesh_test(tmp_path, trimmed_baseline_obj, "RemeshInterp",
                              remeshing_method="interpolate")

    def test_remeshing_marchingcubes(self, tmp_path, trimmed_baseline_obj):
        self._run_remesh_test(tmp_path, trimmed_baseline_obj, "RemeshMC",
                              remeshing_method="marchingcubes")

    def test_reinit_marchingcubes(self, tmp_path, trimmed_baseline_obj):
        self._run_remesh_test(tmp_path, trimmed_baseline_obj, "ReinitMC",
                              remeshing_method="marchingcubes",
                              reinit_sdf_method="marchingcubes")

    def test_bc_graft(self, tmp_path, trimmed_baseline_obj):
        self._run_remesh_test(tmp_path, trimmed_baseline_obj, "BCGraft",
                              remeshing_method="marchingcubes",
                              reinit_sdf_method="marchingcubes",
                              sdf_bc_fix_method="graft")

    def test_bc_zip(self, tmp_path, trimmed_baseline_obj):
        self._run_remesh_test(tmp_path, trimmed_baseline_obj, "BCZip",
                              remeshing_method="marchingcubes",
                              reinit_sdf_method="marchingcubes",
                              sdf_bc_fix_method="zip")

    def test_warp_mesh_init_matches_trimesh_init(self, tmp_path, trimmed_baseline_obj):
        """Path A with warp.Mesh input should match trimesh input within 1%.

        Steps:
        1. Build a trimesh solver (path A, trimesh).
        2. Extract the initial warp BVH from it and pass as warp.Mesh input.
        3. Run 3 iters on both; assert < 1% deviation.
        """
        import math
        import warp as wp
        from core.shape_opt import ShapeOptTrimmed

        # --- Path A (trimesh) ---
        tmp_a = str(tmp_path / "warp_a")
        overrides_a = self._make_remesh_overrides(
            tmp_a,
            remesh_octree_depth=4,
            remesh_init_octree_depth=2,
            remeshing_method="marchingcubes",
            reinit_sdf_method="marchingcubes",
        )
        solver_a = ShapeOptTrimmed(self.CONFIG, override_params=overrides_a)

        # Extract the warp.Mesh from solver_a's initial BVH.
        wp_mesh = solver_a._mc_wp_mesh
        negate = solver_a._mc_wp_negate
        bbox_diag = solver_a._mc_bbox_diag_phys
        overrides_a_meta = self._make_remesh_overrides(
            tmp_a,  # reuse same dir — only used for metadata extraction here
            remesh_octree_depth=4,
            remesh_init_octree_depth=2,
        )
        mesh_center = list(overrides_a_meta["remeshing_trimesh"]["mesh_center"])
        h_target = float(overrides_a_meta["remeshing_trimesh"]["h_target"])
        clip_bbox = overrides_a_meta["remeshing_trimesh"]["clip_bbox"]

        # --- Path A (warp.Mesh) ---
        tmp_b = str(tmp_path / "warp_b")
        overrides_b = self._make_remesh_overrides(
            tmp_b,
            remesh_octree_depth=4,
            remesh_init_octree_depth=2,
            remeshing_method="marchingcubes",
            reinit_sdf_method="marchingcubes",
        )
        # Replace trimesh with the warp.Mesh extracted above.
        overrides_b["remeshing_trimesh"]["mesh"] = wp_mesh
        overrides_b["remeshing_trimesh"]["mesh_center"] = mesh_center
        solver_b = ShapeOptTrimmed(self.CONFIG, override_params=overrides_b)

        objs_a = [_gd_iter(solver_a, "TrimeshA", i)[0] for i in range(3)]
        objs_b = [_gd_iter(solver_b, "WarpMesh", i)[0] for i in range(3)]

        assert not any(math.isnan(o) for o in objs_a + objs_b), (
            f"NaN: trimesh={objs_a}, warp.Mesh={objs_b}"
        )
        for i, (oa, ob) in enumerate(zip(objs_a, objs_b)):
            rel = abs(oa - ob) / max(abs(oa), 1e-12)
            assert rel < 0.01, (
                f"iter {i}: trimesh={oa:.6e}, warp.Mesh={ob:.6e}, diff {rel:.1%}"
            )

    def test_geo_func_init_matches_trimesh_init(self, tmp_path, trimmed_baseline_obj):
        """Path B (geo_func) should give same results as path A (trimesh) for 3 iters.

        Steps:
        1. Initialize a solver via path A (trimesh), extract the initial MC
           surface BVH as a geo_func + axis/geo_min metadata.
        2. Initialize a second solver via path B (geo_func) using those.
        3. Run 3 GD iters on both; assert objectives agree within 1%.
        """
        import math
        from core.shape_opt import ShapeOptTrimmed

        # --- Path A: trimesh init ---
        tmp_a = str(tmp_path / "path_a")
        overrides_a = self._make_remesh_overrides(
            tmp_a,
            remesh_octree_depth=4,
            remesh_init_octree_depth=2,
            remeshing_method="marchingcubes",
            reinit_sdf_method="marchingcubes",
        )
        solver_a = ShapeOptTrimmed(self.CONFIG, override_params=overrides_a)

        # Extract path B inputs from solver_a's initial MC state.
        # _mc_geo_func() captures the BVH as a closure — safe to use after
        # solver_a has been further modified by GD iters.
        geo_func = solver_a._mc_geo_func()
        axis_scale = list(solver_a._remesh_axis_scale)
        geo_min = list(solver_a._remesh_geo_min)
        h_target = float(solver_a._sussman_grid_spacing[0])
        clip_bbox = solver_a._remesh_clip_bbox

        # --- Path B: geo_func init ---
        tmp_b = str(tmp_path / "path_b")
        overrides_b = self._make_remesh_overrides(
            tmp_b,
            remesh_octree_depth=4,
            remesh_init_octree_depth=2,
            remeshing_method="marchingcubes",
            reinit_sdf_method="marchingcubes",
        )
        # Replace remeshing_trimesh with remeshing_geo_func.
        del overrides_b["remeshing_trimesh"]
        overrides_b["remeshing_geo_func"] = {
            "geo_func": geo_func,
            "axis_scale": axis_scale,
            "geo_min": geo_min,
            "h_target": h_target,
            "clip_bbox": clip_bbox,
        }
        solver_b = ShapeOptTrimmed(self.CONFIG, override_params=overrides_b)

        # Run 3 GD iters on both and compare objectives.
        objs_a = [_gd_iter(solver_a, "PathA", i)[0] for i in range(3)]
        objs_b = [_gd_iter(solver_b, "PathB", i)[0] for i in range(3)]

        assert not any(math.isnan(o) for o in objs_a + objs_b), (
            f"NaN in objectives: A={objs_a}, B={objs_b}"
        )
        for i, (oa, ob) in enumerate(zip(objs_a, objs_b)):
            rel = abs(oa - ob) / max(abs(oa), 1e-12)
            assert rel < 0.01, (
                f"iter {i}: PathA={oa:.6e}, PathB={ob:.6e}, "
                f"relative diff {rel:.1%} > 1%"
            )
        # Also check each path's final obj is within 1% of the uniform-hex baseline.
        for label, objs in [("PathA", objs_a), ("PathB", objs_b)]:
            rel = abs(objs[-1] - trimmed_baseline_obj) / max(abs(trimmed_baseline_obj), 1e-12)
            assert rel < 0.01, (
                f"{label}: final obj {objs[-1]:.4e} deviates {rel:.1%} "
                f"from baseline {trimmed_baseline_obj:.4e}"
            )
