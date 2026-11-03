"""Comprehensive tests for the trimesh → SDF → octree → FEA pipeline.

Compares octree FEA solutions against regular-grid solutions and analytical
affine displacement fields.  Uses ``trimesh.proximity.signed_distance`` via
:mod:`src.triposg_wrapper.trimesh_sdf` as the geometric function for hierarchical octree
extraction.

Test categories
---------------
A.  SDF correctness (sphere, box)
B.  Mesh generation (uniform, non-uniform, element volumes)
C.  FEA analytical match (parametrised over affine gradients)
D.  Octree vs regular-grid comparison
E.  Depth convergence
F.  Hanging-node constraint verification
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — same convention as test_octree_mesh.py
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent  # src/warp_wrapper/
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.neoHookean import OctreeNeoHookean  # noqa: E402

import trimesh  # noqa: E402
import trimesh.proximity  # noqa: E402

PROJECT_SRC = ROOT_DIR.parent.parent / "src"  # project-level src/

# ---------------------------------------------------------------------------
# Import helpers from project-level src/ (avoid shadowed 'src' namespace)
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent


def _load_source_module(name: str, path: Path, strip_relative: bool = False):
    """Load a .py file as a module, optionally stripping ``from .`` imports."""
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
    """Load project-level sdf2octree, hierarchical_extraction, and trimesh_sdf."""
    # 1. hierarchical_extraction (no problematic deps)
    he_mod = _load_source_module(
        "_proj_he", PROJECT_SRC / "triposg_wrapper" / "hierarchical_extraction.py"
    )

    # 2. sdf2octree from utils.py (strip TripoSG relative imports)
    utils_mod = _load_source_module(
        "_proj_utils", PROJECT_SRC / "triposg_wrapper" / "utils.py", strip_relative=True
    )

    # 3. trimesh_sdf (strip relative imports, inject dependencies)
    ts_path = PROJECT_SRC / "triposg_wrapper" / "trimesh_sdf.py"
    ts_source = ts_path.read_text()
    lines = ts_source.split("\n")
    clean = []
    for line in lines:
        if line.strip().startswith("from ."):
            clean.append("pass  # skipped relative import")
        else:
            clean.append(line)
    ts_mod = types.ModuleType("_proj_trimesh_sdf")
    ts_mod.__file__ = str(ts_path)
    # Inject deps before exec so the module's functions find them
    ts_mod.__dict__["hierarchical_extraction"] = he_mod.hierarchical_extraction
    ts_mod.__dict__["_sdf2octree_fn"] = utils_mod.sdf2octree
    exec(compile("\n".join(clean), str(ts_path), "exec"), ts_mod.__dict__)

    return ts_mod, utils_mod.sdf2octree


trimesh_sdf, sdf2octree = _import_project_modules()

# Import test helpers from sibling test file
_spec = importlib.util.spec_from_file_location(
    "test_octree_mesh", TESTS_DIR / "test_octree_mesh.py"
)
_tom = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tom)
make_uniform_hex_mesh = _tom.make_uniform_hex_mesh
make_cube_config_overrides = _tom.make_cube_config_overrides

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

AFFINE_GRADIENTS = {
    "shear": ((0.0, 0.015, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "tension": ((0.01, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "compression": ((-0.009, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "mixed": (
        (0.004, -0.002, 0.0015),
        (0.0035, 0.0025, -0.002),
        (-0.001, 0.002, 0.003),
    ),
}


def _make_affine_config_overrides(tmp, gradient, octree_mesh):
    """Config for [-1,1]^3 with affine ``u = F x`` BCs on all 6 faces."""
    F = np.array(gradient, dtype=np.float64)

    def ux(x, y, z):
        return F[0, 0] * x + F[0, 1] * y + F[0, 2] * z

    def uy(x, y, z):
        return F[1, 0] * x + F[1, 1] * y + F[1, 2] * z

    def uz(x, y, z):
        return F[2, 0] * x + F[2, 1] * y + F[2, 2] * z

    return {
        "output_root": tmp,
        "width": 2.0,
        "height": 2.0,
        "depth": 2.0,
        "nx": 3,
        "ny": 3,
        "nz": 3,
        "octree_mesh": octree_mesh,
        "dirichlet_bcs_dict": {
            lambda x, y, z: np.isclose(x, -1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(x, 1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(y, -1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(y, 1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(z, -1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(z, 1.0): (ux, uy, uz),
        },
    }


def _to_numpy(t):
    """Convert a tensor (torch or warp) to a numpy array."""
    if hasattr(t, "cpu"):
        return t.cpu().numpy()
    return np.asarray(t)


def _classify_elements(octree):
    """Partition elements into coarse and fine sets by edge length.

    Returns ``(coarse_indices, fine_indices, edge_lengths)`` where the
    coarse/fine split is at the midpoint of the unique edge-length spectrum.
    Also returns the unique edge-length values found.
    """
    nodes_np = _to_numpy(octree["nodes"]).astype(np.float64)
    elems_np = _to_numpy(octree["elements"])

    # Use edge 0→1 length as element-size proxy (axis-aligned hex)
    lengths = np.linalg.norm(
        nodes_np[elems_np[:, 1]] - nodes_np[elems_np[:, 0]], axis=1
    )
    unique_lengths = np.unique(np.round(lengths, 6))

    if len(unique_lengths) < 2:
        # Uniform mesh — everything is the same size
        return np.arange(len(elems_np)), np.array([], dtype=int), unique_lengths

    threshold = (unique_lengths[0] + unique_lengths[-1]) / 2.0
    coarse = np.where(lengths > threshold)[0]
    fine = np.where(lengths <= threshold)[0]
    return coarse, fine, unique_lengths


def _nodes_in_elements(octree, elem_indices):
    """Return the sorted unique node indices belonging to *elem_indices*."""
    elems_np = _to_numpy(octree["elements"])
    return np.unique(elems_np[elem_indices].ravel())


def _verify_hanging_constraints(disp, octree, tol=1e-8):
    """Assert edge/face hanging-node constraints are satisfied in *disp*."""
    edge_h = octree.get("edge_hanging", None)
    if edge_h is not None and len(edge_h) > 0:
        edge_np = _to_numpy(edge_h)
        for row in edge_np:
            slave, m1, m2 = int(row[0]), int(row[1]), int(row[2])
            avg = 0.5 * (disp[m1] + disp[m2])
            err = np.linalg.norm(disp[slave] - avg)
            assert (
                err < tol
            ), f"Edge hanging constraint violated: slave={slave}, err={err:.3e}"

    face_h = octree.get("face_hanging", None)
    if face_h is not None and len(face_h) > 0:
        face_np = _to_numpy(face_h)
        for row in face_np:
            slave = int(row[0])
            masters = [int(row[i]) for i in range(1, 5)]
            avg = sum(disp[m] for m in masters) / 4.0
            err = np.linalg.norm(disp[slave] - avg)
            assert (
                err < tol
            ), f"Face hanging constraint violated: slave={slave}, err={err:.3e}"


def _make_uniform_trimesh_octree(depth: int):
    """Return a uniform octree dict generated via the trimesh pipeline.

    Uses an icosphere whose surface sits well inside the bounds so that
    ``hierarchical_extraction`` with ``init_depth == octree_depth`` produces
    a uniform grid (no refinement loop).
    """
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.7)
    return trimesh_sdf.trimesh_to_octree(
        mesh,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=depth,
        octree_depth=depth,
        domain_length=2.0,
        device="cuda",
    )


def _make_nonuniform_trimesh_octree(init_depth: int = 2, octree_depth: int = 3):
    """Return a non-uniform octree dict with hanging nodes from a sphere."""
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.7)
    return trimesh_sdf.trimesh_to_octree(
        mesh,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_depth,
        octree_depth=octree_depth,
        domain_length=2.0,
        device="cuda",
    )


# ===================================================================
# Category A — SDF correctness
# ===================================================================


class TestSDFCorrectness:
    """Verify signed-distance values for known shapes."""

    def test_sphere_sdf_correctness(self):
        """SDF for an icosphere at known interior / surface / exterior points."""
        radius = 0.7
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")

        test_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # centre — deep inside
                [0.0, 0.0, radius],  # on surface
                [0.0, 0.0, 1.0],  # outside
                [radius, 0.0, 0.0],  # on surface
            ],
            dtype=torch.float16,
            device="cuda",
        ).unsqueeze(
            0
        )  # (1, 4, 3)

        sdf = geometric_func(test_points).flatten().float().cpu().numpy()

        # Standard convention: negative inside, positive outside
        assert sdf[0] < -0.5, f"Origin should be deep inside: {sdf[0]}"
        assert abs(sdf[1]) < 0.1, f"Surface point SDF should be ~0: {sdf[1]}"
        assert sdf[2] > 0.2, f"Exterior point should have positive SDF: {sdf[2]}"
        assert abs(sdf[3]) < 0.1, f"Surface point SDF should be ~0: {sdf[3]}"

        # Approximate check vs analytical SDF = |x| - r for a sphere
        for i, pt in enumerate(test_points.squeeze(0).float().cpu().numpy()):
            expected = np.linalg.norm(pt) - radius
            assert (
                abs(sdf[i] - expected) < 0.1
            ), f"Point {i}: got {sdf[i]:.4f}, expected ~{expected:.4f}"

    def test_box_sdf_correctness(self):
        """SDF for an axis-aligned box at known points."""
        half = 0.7
        mesh = trimesh.creation.box(extents=(2 * half, 2 * half, 2 * half))
        geometric_func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")

        test_points = torch.tensor(
            [
                [0.0, 0.0, 0.0],  # centre — inside
                [half, 0.0, 0.0],  # on face — surface
                [1.0, 0.0, 0.0],  # outside
            ],
            dtype=torch.float16,
            device="cuda",
        ).unsqueeze(0)

        sdf = geometric_func(test_points).flatten().float().cpu().numpy()

        assert sdf[0] < -0.3, f"Centre should be inside: {sdf[0]}"
        assert abs(sdf[1]) < 0.1, f"Face centre should be ~0: {sdf[1]}"
        assert sdf[2] > 0.2, f"Exterior should be positive: {sdf[2]}"

    def test_sign_convention_auto_detect(self):
        """Auto-detected convention always gives negative-inside SDF."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
        func = trimesh_sdf.make_trimesh_geometric_func(mesh, device="cuda")
        origin = torch.tensor([[[0.0, 0.0, 0.0]]], dtype=torch.float16, device="cuda")
        sdf_val = func(origin).item()
        assert sdf_val < 0, f"Auto convention should give negative inside: {sdf_val}"


# ===================================================================
# Category B — Mesh generation
# ===================================================================


class TestMeshGeneration:
    """Verify octree mesh structure produced by the trimesh pipeline."""

    def test_uniform_octree_from_trimesh(self):
        """init_depth == octree_depth → uniform mesh, correct counts, no hanging."""
        depth = 3
        octree = _make_uniform_trimesh_octree(depth)

        nodes = octree["nodes"]
        elements = octree["elements"]

        expected_elements = (2 ** depth) ** 3  # 512
        expected_nodes = (2 ** depth + 1) ** 3  # 729
        assert (
            elements.shape[0] == expected_elements
        ), f"Expected {expected_elements} elements, got {elements.shape[0]}"
        assert (
            nodes.shape[0] == expected_nodes
        ), f"Expected {expected_nodes} nodes, got {nodes.shape[0]}"

        # No hanging nodes for uniform mesh
        assert "edge_hanging" not in octree
        assert "face_hanging" not in octree

        # All elements should have identical edge length (uniform)
        _, _, unique_lengths = _classify_elements(octree)
        assert (
            len(unique_lengths) == 1
        ), f"Uniform mesh should have one element size, got {unique_lengths}"

    def test_nonuniform_octree_has_both_coarse_and_fine(self):
        """Sphere surface → non-uniform mesh with both coarse and fine elements."""
        octree = _make_nonuniform_trimesh_octree(init_depth=2, octree_depth=3)

        n_coarse_limit = (2 ** 2) ** 3  # 64 (all-coarse)
        n_fine_limit = (2 ** 3) ** 3  # 512 (all-fine)
        n_elem = octree["elements"].shape[0]
        assert n_coarse_limit < n_elem < n_fine_limit, (
            f"Expected between {n_coarse_limit} and {n_fine_limit} elements, "
            f"got {n_elem}"
        )

        # Must have exactly 2 distinct element sizes (coarse + fine)
        coarse_idx, fine_idx, unique_lengths = _classify_elements(octree)
        assert (
            len(unique_lengths) == 2
        ), f"Expected 2 element sizes (coarse + fine), got {unique_lengths}"
        assert len(coarse_idx) > 0, "No coarse elements found"
        assert len(fine_idx) > 0, "No fine elements found"

        # Coarse edge length should be 2× fine edge length (one level of refinement)
        ratio = unique_lengths[-1] / unique_lengths[0]
        assert (
            abs(ratio - 2.0) < 0.01
        ), f"Coarse/fine edge ratio should be 2, got {ratio:.4f}"

        # Must have edge AND face hanging nodes at coarse-fine interface
        n_eh = octree.get("edge_hanging", torch.empty(0)).shape[0]
        n_fh = octree.get("face_hanging", torch.empty(0)).shape[0]
        assert n_eh > 0, "Non-uniform octree should have edge hanging nodes"
        assert n_fh > 0, "Non-uniform octree should have face hanging nodes"

    def test_element_volumes_positive_both_sizes(self):
        """All hex elements (coarse AND fine) should have positive volume."""
        octree = _make_nonuniform_trimesh_octree(init_depth=2, octree_depth=3)

        nodes_np = _to_numpy(octree["nodes"]).astype(np.float64)
        elems_np = _to_numpy(octree["elements"])
        coarse_idx, fine_idx, _ = _classify_elements(octree)

        # Check coarse elements
        for i in coarse_idx:
            elem = elems_np[i]
            p0, p1, p3, p4 = (
                nodes_np[elem[0]],
                nodes_np[elem[1]],
                nodes_np[elem[3]],
                nodes_np[elem[4]],
            )
            vol = np.dot(np.cross(p1 - p0, p3 - p0), p4 - p0)
            assert vol > -1e-10, f"Coarse element {i} has non-positive volume: {vol}"

        # Check fine elements
        for i in fine_idx:
            elem = elems_np[i]
            p0, p1, p3, p4 = (
                nodes_np[elem[0]],
                nodes_np[elem[1]],
                nodes_np[elem[3]],
                nodes_np[elem[4]],
            )
            vol = np.dot(np.cross(p1 - p0, p3 - p0), p4 - p0)
            assert vol > -1e-10, f"Fine element {i} has non-positive volume: {vol}"

    def test_hanging_nodes_lie_on_coarse_fine_interface(self):
        """Hanging nodes should be shared between coarse and fine elements."""
        octree = _make_nonuniform_trimesh_octree(init_depth=2, octree_depth=3)
        coarse_idx, fine_idx, _ = _classify_elements(octree)

        coarse_nodes = set(_nodes_in_elements(octree, coarse_idx).tolist())
        fine_nodes = set(_nodes_in_elements(octree, fine_idx).tolist())

        # Hanging nodes (slaves) must appear in fine elements
        edge_h = octree.get("edge_hanging", None)
        if edge_h is not None and len(edge_h) > 0:
            for row in _to_numpy(edge_h):
                slave = int(row[0])
                assert (
                    slave in fine_nodes
                ), f"Edge hanging slave {slave} not in fine elements"
                # Masters must appear in coarse elements
                m1, m2 = int(row[1]), int(row[2])
                assert (
                    m1 in coarse_nodes
                ), f"Edge hanging master {m1} not in coarse elements"
                assert (
                    m2 in coarse_nodes
                ), f"Edge hanging master {m2} not in coarse elements"

        face_h = octree.get("face_hanging", None)
        if face_h is not None and len(face_h) > 0:
            for row in _to_numpy(face_h):
                slave = int(row[0])
                assert (
                    slave in fine_nodes
                ), f"Face hanging slave {slave} not in fine elements"
                for j in range(1, 5):
                    m = int(row[j])
                    assert (
                        m in coarse_nodes
                    ), f"Face hanging master {m} not in coarse elements"


# ===================================================================
# Category C — FEA analytical match (affine problems)
# ===================================================================


class TestFEAAnalytical:
    """Verify FEA solutions on trimesh-derived octree match ``u = F x``."""

    @pytest.mark.parametrize("grad_name, grad_values", list(AFFINE_GRADIENTS.items()))
    def test_affine_analytical_uniform_trimesh(self, grad_name, grad_values, tmp_path):
        """Uniform trimesh octree (depth=3): ``u = F x`` on all faces."""
        octree = _make_uniform_trimesh_octree(3)
        F = np.array(grad_values, dtype=np.float64)
        overrides = _make_affine_config_overrides(str(tmp_path), grad_values, octree)

        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides,
        )
        s.forward()
        assert s.final_residual < 1e-5, f"Residual too high: {s.final_residual:.3e}"

        disp = s.get_displacement()
        pos = s.space.node_positions().numpy()
        expected = (F @ pos.T).T

        np.testing.assert_allclose(
            disp,
            expected,
            atol=1e-4,
            rtol=5e-3,
            err_msg=f"Affine {grad_name}: displacement mismatch (uniform)",
        )

    @pytest.mark.parametrize(
        "grad_name, grad_values",
        [
            ("shear", AFFINE_GRADIENTS["shear"]),
            ("mixed", AFFINE_GRADIENTS["mixed"]),
        ],
    )
    def test_affine_analytical_nonuniform_trimesh(
        self, grad_name, grad_values, tmp_path
    ):
        """Non-uniform trimesh octree with hanging nodes: ``u = F x``.

        Verifies the analytical solution separately for nodes belonging to
        coarse elements, fine elements, and hanging (interface) nodes.
        """
        octree = _make_nonuniform_trimesh_octree(init_depth=2, octree_depth=3)
        F = np.array(grad_values, dtype=np.float64)
        overrides = _make_affine_config_overrides(str(tmp_path), grad_values, octree)

        # Ensure the mesh really has both levels
        coarse_idx, fine_idx, unique_lengths = _classify_elements(octree)
        assert (
            len(coarse_idx) > 0 and len(fine_idx) > 0
        ), "Test requires both coarse and fine elements"

        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides,
        )
        s.forward()
        assert s.final_residual < 1e-5, f"Residual too high: {s.final_residual:.3e}"

        disp = s.get_displacement()
        pos = s.space.node_positions().numpy()
        expected = (F @ pos.T).T

        # Global check
        np.testing.assert_allclose(
            disp,
            expected,
            atol=5e-4,
            rtol=5e-3,
            err_msg=f"Affine {grad_name}: displacement mismatch (non-uniform)",
        )

        # Check nodes in coarse-only region
        coarse_node_ids = _nodes_in_elements(octree, coarse_idx)
        fine_node_ids = _nodes_in_elements(octree, fine_idx)
        coarse_only = np.setdiff1d(coarse_node_ids, fine_node_ids)
        if len(coarse_only) > 0:
            err_coarse = np.abs(disp[coarse_only] - expected[coarse_only]).max()
            assert (
                err_coarse < 5e-4
            ), f"Affine {grad_name}: coarse-region L∞ = {err_coarse:.3e}"

        # Check nodes in fine-only region
        fine_only = np.setdiff1d(fine_node_ids, coarse_node_ids)
        if len(fine_only) > 0:
            err_fine = np.abs(disp[fine_only] - expected[fine_only]).max()
            assert (
                err_fine < 5e-4
            ), f"Affine {grad_name}: fine-region L∞ = {err_fine:.3e}"

        # Check hanging (slave) nodes explicitly
        hanging_slaves = set()
        edge_h = octree.get("edge_hanging", None)
        if edge_h is not None and len(edge_h) > 0:
            hanging_slaves.update(int(r[0]) for r in _to_numpy(edge_h))
        face_h = octree.get("face_hanging", None)
        if face_h is not None and len(face_h) > 0:
            hanging_slaves.update(int(r[0]) for r in _to_numpy(face_h))

        if hanging_slaves:
            h_ids = np.array(sorted(hanging_slaves))
            err_hanging = np.abs(disp[h_ids] - expected[h_ids]).max()
            assert (
                err_hanging < 5e-4
            ), f"Affine {grad_name}: hanging-node L∞ = {err_hanging:.3e}"


# ===================================================================
# Category D — Octree vs regular-grid comparison
# ===================================================================


class TestOctreeVsRegularGrid:
    """Compare FEA solutions on trimesh octree vs dense uniform grid."""

    def test_trimesh_octree_vs_regular_grid(self, tmp_path):
        """Shared-node displacement comparison between octree and regular grid."""
        from scipy.spatial import cKDTree

        shear = 0.015
        depth = 3
        n = 2 ** depth  # 8 cells per direction

        # --- Regular (uniform) grid baseline ---
        reg_nodes, reg_elements = make_uniform_hex_mesh(
            n, n, n, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]
        )
        reg_mesh = {"nodes": reg_nodes, "elements": reg_elements}
        overrides_reg = make_cube_config_overrides(
            str(tmp_path / "reg"), shear, reg_mesh
        )
        s_reg = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides_reg,
        )
        s_reg.forward()
        disp_reg = s_reg.get_displacement()
        pos_reg = s_reg.space.node_positions().numpy()

        # --- Non-uniform trimesh octree ---
        octree = _make_nonuniform_trimesh_octree(init_depth=2, octree_depth=depth)
        overrides_oct = make_cube_config_overrides(str(tmp_path / "oct"), shear, octree)
        s_oct = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides_oct,
        )
        s_oct.forward()
        disp_oct = s_oct.get_displacement()
        pos_oct = s_oct.space.node_positions().numpy()

        # --- Compare at shared nodes ---
        tree = cKDTree(pos_reg)
        dists, indices = tree.query(pos_oct)
        shared = dists < 1e-6
        n_shared = shared.sum()
        assert n_shared > 0, "No shared nodes found between grids"

        linf = np.abs(disp_oct[shared] - disp_reg[indices[shared]]).max()
        assert (
            linf < 5e-3
        ), f"Displacement mismatch at {n_shared} shared nodes: L∞ = {linf:.3e}"


# ===================================================================
# Category E — Depth convergence
# ===================================================================


class TestConvergence:
    """Verify error decreases (or stays near zero) with depth refinement."""

    def test_depth_convergence(self, tmp_path):
        """Affine shear on uniform trimesh octree at depths 2, 3, 4."""
        F = np.array(AFFINE_GRADIENTS["shear"], dtype=np.float64)
        errors = []
        depths = [2, 3, 4]

        for d in depths:
            octree = _make_uniform_trimesh_octree(d)
            overrides = _make_affine_config_overrides(
                str(tmp_path / f"d{d}"),
                AFFINE_GRADIENTS["shear"],
                octree,
            )
            s = OctreeNeoHookean(
                "tests.examples.three_d_affine_shear.config",
                override_params=overrides,
            )
            s.forward()
            assert (
                s.final_residual < 1e-4
            ), f"Depth {d}: residual too high ({s.final_residual:.3e})"

            disp = s.get_displacement()
            pos = s.space.node_positions().numpy()
            expected = (F @ pos.T).T
            err = np.abs(disp - expected).max()
            errors.append(err)

        # Error should not increase with refinement
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i - 1] * 1.5 + 1e-6, (
                f"Error increased: depth {depths[i-1]}→{depths[i]}: "
                f"{errors[i-1]:.3e} → {errors[i]:.3e}"
            )


# ===================================================================
# Category F — Hanging-node constraints
# ===================================================================


class TestHangingConstraints:
    """Verify hanging-node constraints are satisfied in FEA solutions."""

    def test_hanging_constraints_satisfied_trimesh_octree(self, tmp_path):
        """Shear BCs on non-uniform sphere octree."""
        octree = _make_nonuniform_trimesh_octree(init_depth=2, octree_depth=3)
        overrides = make_cube_config_overrides(str(tmp_path), 0.015, octree)
        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides,
        )
        s.forward()
        assert s.final_residual < 1e-5

        disp = s.get_displacement()
        _verify_hanging_constraints(disp, octree, tol=1e-8)

    @pytest.mark.parametrize(
        "grad_name, grad_values",
        [
            ("shear", AFFINE_GRADIENTS["shear"]),
            ("mixed", AFFINE_GRADIENTS["mixed"]),
        ],
    )
    def test_hanging_constraints_various_gradients(
        self, grad_name, grad_values, tmp_path
    ):
        """Hanging constraints hold across different deformation gradients."""
        octree = _make_nonuniform_trimesh_octree(init_depth=2, octree_depth=3)
        overrides = _make_affine_config_overrides(str(tmp_path), grad_values, octree)
        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides,
        )
        s.forward()
        assert s.final_residual < 1e-5

        disp = s.get_displacement()
        _verify_hanging_constraints(disp, octree, tol=1e-8)
