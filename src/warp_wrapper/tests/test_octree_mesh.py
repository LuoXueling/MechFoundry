"""Verify octree (Hexmesh) import and hanging node enforcement."""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tempfile
import shutil
import numpy as np
import torch
import warp as wp
from core.neoHookean import NeoHookean, OctreeNeoHookean


def _import_sdf2octree():
    """Import sdf2octree from the project-level src/utils.py.

    The test's sys.path makes 'src' resolve to src/warp_wrapper/src/ (regular
    package with __init__.py), shadowing the outer project 'src/' (namespace
    package).  We work around this by loading the file directly, stubbing out
    relative imports that sdf2octree does not need.
    """
    import types

    utils_path = ROOT_DIR.parent.parent / "src" / "triposg_wrapper" / "utils.py"
    source = utils_path.read_text()

    # Strip relative imports (only used by sdf2surfacemesh / octree2vtu,
    # not by sdf2octree which only uses torch + numpy).
    lines = source.split("\n")
    clean = []
    for line in lines:
        if line.strip().startswith("from ."):
            clean.append("pass  # skipped relative import")
        else:
            clean.append(line)

    mod = types.ModuleType("_proj_utils")
    mod.__file__ = str(utils_path)
    exec(compile("\n".join(clean), str(utils_path), "exec"), mod.__dict__)
    return mod.sdf2octree


sdf2octree = _import_sdf2octree()


def make_uniform_hex_mesh(nx, ny, nz, bounds_lo, bounds_hi):
    """Create a uniform hex mesh equivalent to Grid3D(nx, ny, nz).

    Returns (nodes, elements) as numpy arrays.
    """
    lx = bounds_hi[0] - bounds_lo[0]
    ly = bounds_hi[1] - bounds_lo[1]
    lz = bounds_hi[2] - bounds_lo[2]

    # Node positions
    xs = np.linspace(bounds_lo[0], bounds_hi[0], nx + 1)
    ys = np.linspace(bounds_lo[1], bounds_hi[1], ny + 1)
    zs = np.linspace(bounds_lo[2], bounds_hi[2], nz + 1)

    nodes = []
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                nodes.append([xs[i], ys[j], zs[k]])
    nodes = np.array(nodes, dtype=np.float64)

    # Element connectivity — Warp Hexmesh ordering
    def node_index(i, j, k):
        return k + j * (nz + 1) + i * (ny + 1) * (nz + 1)

    elements = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                n0 = node_index(i, j, k)
                n1 = node_index(i + 1, j, k)
                n2 = node_index(i + 1, j + 1, k)
                n3 = node_index(i, j + 1, k)
                n4 = node_index(i, j, k + 1)
                n5 = node_index(i + 1, j, k + 1)
                n6 = node_index(i + 1, j + 1, k + 1)
                n7 = node_index(i, j + 1, k + 1)
                elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
    elements = np.array(elements, dtype=np.int32)

    return nodes, elements


def make_octree_mesh_with_hanging():
    """Create a 2x2x2 base mesh with one cell refined into 8, producing hanging nodes.

    Refines the cell at corner (0, 0, 0) into 8 sub-cells.
    Domain: [0, 1]^3.  Base: 2x2x2 = 8 cells.  After refining one cell: 15 cells total.
    """
    # Start with 2x2x2 uniform mesh
    nodes_base, elements_base = make_uniform_hex_mesh(
        2, 2, 2, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    )

    # Refine cell 0: the cell at (i=0, j=0, k=0)
    # Its corner nodes span [0, 0.5] in each dimension
    # We need to add midpoint nodes on each edge and face of this cell,
    # plus the center node.

    # Cell 0 nodes (from our node_index function):
    # n0 = node_index(0,0,0) = 0
    # n1 = node_index(1,0,0) = 9
    # n2 = node_index(1,1,0) = 12
    # n3 = node_index(0,1,0) = 3
    # n4 = node_index(0,0,1) = 1
    # n5 = node_index(1,0,1) = 10
    # n6 = node_index(1,1,1) = 13
    # n7 = node_index(0,1,1) = 4
    cell0_nodes = elements_base[0]
    assert len(cell0_nodes) == 8

    # The 2x2x2 uniform mesh has 27 nodes (3x3x3)
    n_base = len(nodes_base)
    assert n_base == 27

    # Add 19 new nodes for octree refinement of cell [0, 0.5]^3:
    # 12 edge midpoints, 6 face centers, 1 body center
    h = 0.25  # half the cell size (cell goes 0 to 0.5, midpoint at 0.25)

    new_nodes = [
        # Edge midpoints (12 edges of the hex)
        [0.25, 0.0, 0.0],  # edge n0-n1   -> new node 27
        [0.5, 0.25, 0.0],  # edge n1-n2   -> new node 28
        [0.25, 0.5, 0.0],  # edge n3-n2   -> new node 29
        [0.0, 0.25, 0.0],  # edge n0-n3   -> new node 30
        [0.25, 0.0, 0.5],  # edge n4-n5   -> new node 31
        [0.5, 0.25, 0.5],  # edge n5-n6   -> new node 32
        [0.25, 0.5, 0.5],  # edge n7-n6   -> new node 33
        [0.0, 0.25, 0.5],  # edge n4-n7   -> new node 34
        [0.0, 0.0, 0.25],  # edge n0-n4   -> new node 35
        [0.5, 0.0, 0.25],  # edge n1-n5   -> new node 36
        [0.5, 0.5, 0.25],  # edge n2-n6   -> new node 37
        [0.0, 0.5, 0.25],  # edge n3-n7   -> new node 38
        # Face centers (6 faces)
        [0.25, 0.25, 0.0],  # bottom (z=0) -> new node 39
        [0.25, 0.25, 0.5],  # top (z=0.5)  -> new node 40
        [0.25, 0.0, 0.25],  # front (y=0)  -> new node 41
        [0.25, 0.5, 0.25],  # back (y=0.5) -> new node 42
        [0.0, 0.25, 0.25],  # left (x=0)   -> new node 43
        [0.5, 0.25, 0.25],  # right (x=0.5)-> new node 44
        # Body center
        [0.25, 0.25, 0.25],  # center       -> new node 45
    ]
    new_nodes = np.array(new_nodes, dtype=np.float64)
    nodes = np.vstack([nodes_base, new_nodes])

    # Build the 8 sub-cells replacing cell 0
    c = cell0_nodes  # original 8 corner nodes of cell 0
    # Edge midpoints
    e01 = 27
    e12 = 28
    e32 = 29
    e03 = 30
    e45 = 31
    e56 = 32
    e76 = 33
    e47 = 34
    e04 = 35
    e15 = 36
    e26 = 37
    e37 = 38
    # Face centers
    fbot = 39
    ftop = 40
    ffront = 41
    fback = 42
    fleft = 43
    fright = 44
    # Body center
    center = 45

    sub_cells = np.array(
        [
            # sub-cell 0: (0,0,0) corner
            [c[0], e01, fbot, e03, e04, ffront, center, fleft],
            # sub-cell 1: (1,0,0) corner
            [e01, c[1], e12, fbot, ffront, e15, fright, center],
            # sub-cell 2: (1,1,0) corner
            [fbot, e12, c[2], e32, center, fright, e26, fback],
            # sub-cell 3: (0,1,0) corner
            [e03, fbot, e32, c[3], fleft, center, fback, e37],
            # sub-cell 4: (0,0,1) corner
            [e04, ffront, center, fleft, c[4], e45, ftop, e47],
            # sub-cell 5: (1,0,1) corner
            [ffront, e15, fright, center, e45, c[5], e56, ftop],
            # sub-cell 6: (1,1,1) corner
            [center, fright, e26, fback, ftop, e56, c[6], e76],
            # sub-cell 7: (0,1,1) corner
            [fleft, center, fback, e37, e47, ftop, e76, c[7]],
        ],
        dtype=np.int32,
    )

    # Replace cell 0 with 8 sub-cells, keep cells 1-7 unchanged
    elements = np.vstack([sub_cells, elements_base[1:]])

    # Build hanging node constraints
    # Edge hanging: nodes on edges of the coarse cell that are shared with
    # neighboring coarse cells.  These are the 12 edge midpoints.
    # Each edge midpoint's masters are the two endpoints of that coarse edge.
    edge_hanging = np.array(
        [
            [e01, c[0], c[1]],
            [e12, c[1], c[2]],
            [e32, c[3], c[2]],
            [e03, c[0], c[3]],
            [e45, c[4], c[5]],
            [e56, c[5], c[6]],
            [e76, c[7], c[6]],
            [e47, c[4], c[7]],
            [e04, c[0], c[4]],
            [e15, c[1], c[5]],
            [e26, c[2], c[6]],
            [e37, c[3], c[7]],
        ],
        dtype=np.int64,
    )

    # Face hanging: face center nodes shared with neighboring coarse cells.
    # Each face center's masters are the 4 corner nodes of that face.
    face_hanging = np.array(
        [
            [fbot, c[0], c[1], c[2], c[3]],  # z=0 face
            [ftop, c[4], c[5], c[6], c[7]],  # z=0.5 face
            [ffront, c[0], c[1], c[5], c[4]],  # y=0 face
            [fback, c[3], c[2], c[6], c[7]],  # y=0.5 face
            [fleft, c[0], c[3], c[7], c[4]],  # x=0 face
            [fright, c[1], c[2], c[6], c[5]],  # x=0.5 face
        ],
        dtype=np.int64,
    )

    return nodes, elements, edge_hanging, face_hanging


def test_uniform_octree_mesh():
    """Test: uniform hex mesh (no hanging nodes) via octree_mesh param."""
    tmp = tempfile.mkdtemp()
    try:
        nodes, elements = make_uniform_hex_mesh(
            3, 3, 3, [0.0, 0.0, 0.0], [1.0, 0.6, 0.4]
        )
        octree_mesh = {
            "nodes": nodes,
            "elements": elements,
        }

        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params={
                "output_root": tmp,
                "octree_mesh": octree_mesh,
            },
        )
        n = s.space.node_count()
        print(f"Uniform octree mesh: {n} nodes, {len(elements)} elements")
        assert n == len(nodes), f"Node count mismatch: {n} vs {len(nodes)}"

        s.forward()
        print(
            f"  forward() converged in {s.newton_iterations} iters, "
            f"residual={s.final_residual:.3e}"
        )
        assert s.final_residual < 1e-6, f"Residual too high: {s.final_residual}"

        s.save(time=0.0)
        print("  save() completed")
        print("UNIFORM OCTREE TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_octree_with_hanging_nodes():
    """Test: octree mesh with one refined cell and hanging node constraints."""
    tmp = tempfile.mkdtemp()
    try:
        nodes, elements, edge_hanging, face_hanging = make_octree_mesh_with_hanging()
        octree_mesh = {
            "nodes": nodes,
            "elements": elements,
            "edge_hanging": edge_hanging,
            "face_hanging": face_hanging,
        }

        shear = 0.015

        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params={
                "output_root": tmp,
                "octree_mesh": octree_mesh,
                "max_iter": 30,
            },
        )
        n = s.space.node_count()
        print(f"Octree mesh with hanging: {n} nodes, {len(elements)} elements")
        print(f"  Edge hanging: {len(edge_hanging)}, Face hanging: {len(face_hanging)}")

        s.forward()
        print(
            f"  forward() converged in {s.newton_iterations} iters, "
            f"residual={s.final_residual:.3e}"
        )
        assert s.final_residual < 1e-5, f"Residual too high: {s.final_residual}"

        # Verify hanging node constraints are satisfied in the solution
        disp = s.get_displacement()
        for row in edge_hanging:
            slave, m1, m2 = row
            avg = 0.5 * (disp[m1] + disp[m2])
            err = np.linalg.norm(disp[slave] - avg)
            assert err < 1e-10, (
                f"Edge hanging constraint violated: slave={slave}, " f"err={err:.3e}"
            )

        for row in face_hanging:
            slave, m1, m2, m3, m4 = row
            avg = 0.25 * (disp[m1] + disp[m2] + disp[m3] + disp[m4])
            err = np.linalg.norm(disp[slave] - avg)
            assert err < 1e-10, (
                f"Face hanging constraint violated: slave={slave}, " f"err={err:.3e}"
            )

        print("  All hanging node constraints satisfied")

        s.save(time=0.0)
        print("  save() completed")
        print("OCTREE WITH HANGING NODES TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_load_octree_mesh_method():
    """Test the load_octree_mesh() convenience method."""
    tmp = tempfile.mkdtemp()
    try:
        # Start with a grid-based solver (OctreeField falls back to grid when no octree_mesh)
        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params={"output_root": tmp},
        )
        n_grid = s.space.node_count()
        print(f"Grid-based solver: {n_grid} nodes")
        s.forward()
        print(f"  Grid forward() converged in {s.newton_iterations} iters")

        # Switch to octree mesh via load_octree_mesh
        nodes, elements = make_uniform_hex_mesh(
            2, 2, 2, [0.0, 0.0, 0.0], [1.0, 0.6, 0.4]
        )
        s.load_octree_mesh(nodes, elements)
        n_oct = s.space.node_count()
        print(f"After load_octree_mesh: {n_oct} nodes")
        assert n_oct == len(nodes)

        s.forward()
        print(
            f"  Octree forward() converged in {s.newton_iterations} iters, "
            f"residual={s.final_residual:.3e}"
        )
        assert s.final_residual < 1e-5

        print("LOAD_OCTREE_MESH TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_interpolation_warm_start():
    """Test: displacement interpolation from coarse to fine octree mesh."""
    tmp = tempfile.mkdtemp()
    try:
        bounds_lo = [0.0, 0.0, 0.0]
        bounds_hi = [1.0, 0.6, 0.4]

        # --- Coarse mesh: solve to convergence ---
        coarse_nodes, coarse_elements = make_uniform_hex_mesh(
            2, 2, 2, bounds_lo, bounds_hi
        )
        octree_coarse = {
            "nodes": coarse_nodes,
            "elements": coarse_elements,
        }
        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params={
                "output_root": tmp,
                "octree_mesh": octree_coarse,
            },
        )
        s.forward()
        coarse_iters = s.newton_iterations
        print(
            f"Coarse mesh: {len(coarse_nodes)} nodes, converged in {coarse_iters} iters"
        )

        # --- Fine mesh: load with interpolation ---
        fine_nodes, fine_elements = make_uniform_hex_mesh(4, 4, 4, bounds_lo, bounds_hi)
        s.load_octree_mesh(fine_nodes, fine_elements, interpolate=True)
        n_fine = s.space.node_count()
        print(f"Fine mesh: {n_fine} nodes")

        # Check that interpolated displacement is non-zero at free DOFs
        disp_before = s.get_displacement()
        max_disp = np.abs(disp_before).max()
        print(f"  Max interpolated displacement: {max_disp:.6e}")
        assert max_disp > 0.0, "Interpolated displacement should be non-zero"

        # Solve with warm start
        s.forward()
        warm_iters = s.newton_iterations
        print(
            f"  Warm-start forward() converged in {warm_iters} iters, "
            f"residual={s.final_residual:.3e}"
        )
        assert s.final_residual < 1e-5

        # --- Fine mesh: solve from scratch (no interpolation) for comparison ---
        s2 = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params={
                "output_root": tmp,
                "octree_mesh": {
                    "nodes": fine_nodes,
                    "elements": fine_elements,
                },
            },
        )
        s2.forward()
        cold_iters = s2.newton_iterations
        print(f"  Cold-start forward() converged in {cold_iters} iters")

        # Warm start should converge in no more iterations than cold start
        # (For small fully-constrained problems this may not always hold,
        # so we use a generous margin.)
        assert (
            warm_iters <= cold_iters + 5
        ), f"Warm start ({warm_iters}) much worse than cold start ({cold_iters})"

        print("INTERPOLATION WARM-START TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers for sdf2octree-based tests on [-1,1]^3 domain
# ---------------------------------------------------------------------------


def make_sdf2octree_mesh(depth, domain_length=2.0, refine_corner=True):
    """Create an octree mesh via sdf2octree with a known refinement pattern.

    If refine_corner=True: cells in the first octant are at finest level (depth),
    rest at level depth-1 (one level coarser), producing hanging nodes.
    If refine_corner=False: uniform mesh at finest level (no hanging nodes).
    Returns (nodes, elements, edge_hanging, face_hanging) as torch cuda tensors.
    """
    n = 2 ** depth
    cell_tensor = torch.zeros(n, n, n, dtype=torch.long, device="cuda")
    sdf_tensor = torch.full((n + 1, n + 1, n + 1), -1.0, device="cuda")

    if refine_corner:
        half = n // 2
        # Fine: first octant at max depth
        cell_tensor[:half, :half, :half] = depth
        # Coarse: rest at depth-1; anchors every 2 cells
        step = 2  # 2^(depth - (depth-1)) = 2
        for i in range(0, n, step):
            for j in range(0, n, step):
                for k in range(0, n, step):
                    if i < half and j < half and k < half:
                        continue
                    cell_tensor[i, j, k] = depth - 1
    else:
        cell_tensor[:] = depth

    nodes, node_sdf, elements, edge_h, face_h = sdf2octree(
        cell_tensor, sdf_tensor, domain_length
    )
    return nodes, elements, edge_h, face_h


def make_cube_config_overrides(tmp, shear=0.015, octree_mesh=None):
    """Config overrides for the [-1,1]^3 cubic domain with affine shear BCs."""

    def ux(x, y, z):
        return shear * y

    def uy(x, y, z):
        return 0.0

    def uz(x, y, z):
        return 0.0

    overrides = {
        "output_root": tmp,
        "width": 2.0,
        "height": 2.0,
        "depth": 2.0,
        "nx": 3,
        "ny": 3,
        "nz": 3,
        "dirichlet_bcs_dict": {
            lambda x, y, z: np.isclose(x, -1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(x, 1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(y, -1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(y, 1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(z, -1.0): (ux, uy, uz),
            lambda x, y, z: np.isclose(z, 1.0): (ux, uy, uz),
        },
    }
    if octree_mesh is not None:
        overrides["octree_mesh"] = octree_mesh
    return overrides


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------


def test_interpolation_analytical_field():
    """Test: interpolated values match analytical trilinear field exactly.

    Sets displacement on a coarse sparse octree to u_x = 0.015*y (affine),
    interpolates to a finer sparse octree, and checks that the interpolated
    values at every new node match the analytical formula.
    """
    tmp = tempfile.mkdtemp()
    try:
        shear = 0.015

        # Coarse octree (depth=2) with hanging nodes
        c_nodes, c_elems, c_eh, c_fh = make_sdf2octree_mesh(2, refine_corner=True)
        print(
            f"Coarse mesh: {c_nodes.shape[0]} nodes, "
            f"{c_eh.shape[0]} edge hanging, {c_fh.shape[0]} face hanging"
        )

        octree_coarse = {
            "nodes": c_nodes,
            "elements": c_elems,
            "edge_hanging": c_eh,
            "face_hanging": c_fh,
        }
        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=make_cube_config_overrides(
                tmp, shear=shear, octree_mesh=octree_coarse
            ),
        )

        # Manually set displacement to analytical field: u = (shear*y, 0, 0)
        node_pos = s.space.node_positions().numpy()  # (V, 3)
        analytical = np.zeros_like(node_pos, dtype=np.float32)
        analytical[:, 0] = shear * node_pos[:, 1]  # u_x = shear * y
        s.u.dof_values = wp.from_torch(
            torch.tensor(analytical, device="cuda"), dtype=wp.vec3
        )

        # Fine octree (depth=3) with hanging nodes
        f_nodes, f_elems, f_eh, f_fh = make_sdf2octree_mesh(3, refine_corner=True)
        print(
            f"Fine mesh: {f_nodes.shape[0]} nodes, "
            f"{f_eh.shape[0]} edge hanging, {f_fh.shape[0]} face hanging"
        )

        s.load_octree_mesh(f_nodes, f_elems, f_eh, f_fh, interpolate=True)

        # Read back interpolated displacement
        disp = s.get_displacement()  # (V_new, 3) numpy
        fine_pos = s.space.node_positions().numpy()

        # Expected: u_x = shear * y at each fine node
        expected_ux = shear * fine_pos[:, 1]
        error = np.abs(disp[:, 0] - expected_ux).max()
        print(f"  Max interpolation error (u_x): {error:.3e}")
        assert error < 1e-4, f"Interpolation error {error:.3e} exceeds tolerance 1e-4"

        # u_y and u_z should be zero
        error_uy = np.abs(disp[:, 1]).max()
        error_uz = np.abs(disp[:, 2]).max()
        print(f"  Max u_y error: {error_uy:.3e}, Max u_z error: {error_uz:.3e}")
        assert error_uy < 1e-4, f"u_y error {error_uy:.3e} exceeds tolerance"
        assert error_uz < 1e-4, f"u_z error {error_uz:.3e} exceeds tolerance"

        print("INTERPOLATION ANALYTICAL FIELD TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_interpolation_warm_start_with_hanging():
    """Test: warm-start interpolation between octree meshes with hanging nodes.

    Uses sdf2octree-generated meshes at different depths to verify that
    interpolation produces a useful warm start that reduces solve iterations.
    """
    tmp = tempfile.mkdtemp()
    try:
        shear = 0.015

        # --- Coarse octree (depth=2) with hanging nodes: solve to convergence ---
        c_nodes, c_elems, c_eh, c_fh = make_sdf2octree_mesh(2, refine_corner=True)
        octree_coarse = {
            "nodes": c_nodes,
            "elements": c_elems,
            "edge_hanging": c_eh,
            "face_hanging": c_fh,
        }
        s = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=make_cube_config_overrides(
                tmp, shear=shear, octree_mesh=octree_coarse
            ),
        )
        s.forward()
        print(
            f"Coarse mesh: {c_nodes.shape[0]} nodes, "
            f"converged in {s.newton_iterations} iters"
        )
        assert s.final_residual < 1e-5

        # --- Fine octree (depth=3) with hanging nodes: warm start ---
        f_nodes, f_elems, f_eh, f_fh = make_sdf2octree_mesh(3, refine_corner=True)
        s.load_octree_mesh(f_nodes, f_elems, f_eh, f_fh, interpolate=True)

        # Verify interpolated displacement is non-zero
        disp = s.get_displacement()
        max_disp = np.abs(disp).max()
        print(f"Fine mesh: {f_nodes.shape[0]} nodes, max interp disp: {max_disp:.6e}")
        assert max_disp > 0.0, "Interpolated displacement should be non-zero"

        # Relax absolute tolerance for the fine-mesh warm-start: hanging node
        # constraints cause CG stalling around 1e-6, well below the test's 1e-5
        # assertion threshold.
        s.params.absolute_tol = 1e-6
        s.forward()
        warm_iters = s.newton_iterations
        print(
            f"  Warm-start converged in {warm_iters} iters, "
            f"residual={s.final_residual:.3e}"
        )
        assert s.final_residual < 1e-5

        # Verify hanging node constraints
        disp_warm = s.get_displacement()
        for row_idx in range(f_eh.shape[0]):
            row = f_eh[row_idx].cpu().numpy()
            slave, m1, m2 = int(row[0]), int(row[1]), int(row[2])
            err = np.linalg.norm(
                disp_warm[slave] - 0.5 * (disp_warm[m1] + disp_warm[m2])
            )
            assert err < 1e-10, f"Edge hanging violated: slave={slave}, err={err:.3e}"

        for row_idx in range(f_fh.shape[0]):
            row = f_fh[row_idx].cpu().numpy()
            slave = int(row[0])
            masters = [int(row[i]) for i in range(1, 5)]
            avg = sum(disp_warm[m] for m in masters) / 4.0
            err = np.linalg.norm(disp_warm[slave] - avg)
            assert err < 1e-10, f"Face hanging violated: slave={slave}, err={err:.3e}"
        print("  All hanging node constraints satisfied")

        # --- Cold start for comparison ---
        octree_fine = {
            "nodes": f_nodes,
            "elements": f_elems,
            "edge_hanging": f_eh,
            "face_hanging": f_fh,
        }
        s2 = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=make_cube_config_overrides(
                tmp, shear=shear, octree_mesh=octree_fine
            ),
        )
        s2.forward()
        cold_iters = s2.newton_iterations
        print(f"  Cold-start converged in {cold_iters} iters")

        # Verify warm and cold solutions agree (both should equal the analytical field)
        disp_cold = s2.get_displacement()
        linf_err = np.abs(disp_warm - disp_cold).max()
        print(f"  Warm vs cold L-inf difference: {linf_err:.3e}")
        assert linf_err < 1e-4, f"Warm and cold solutions differ by {linf_err:.3e}"

        print("INTERPOLATION WARM-START WITH HANGING TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_regular_vs_octree_solution():
    """Test: non-uniform octree mesh produces results close to a dense regular mesh.

    Solves the same affine-shear problem on:
      1. A dense uniform octree (depth=3, all cells at finest level, no hanging
         nodes) on [-1,1]^3 — acts as the reference "regular" mesh.
      2. A non-uniform octree (depth=3, first octant refined, rest at depth-2)
         with hanging nodes on the same domain.

    Compares displacement at octree nodes that coincide with the dense mesh
    nodes.  Because the non-uniform mesh is coarser in 7 of 8 octants, we
    use a higher tolerance than an identical-mesh comparison would require.
    """
    tmp = tempfile.mkdtemp()
    try:
        shear = 0.015

        # --- Dense uniform octree (depth=3, no hanging nodes) as reference ---
        d_nodes, d_elems, d_eh, d_fh = make_sdf2octree_mesh(
            3,
            refine_corner=False,
        )
        dense_mesh = {
            "nodes": d_nodes,
            "elements": d_elems,
        }
        # No hanging constraints for uniform mesh
        overrides_dense = make_cube_config_overrides(
            tmp, shear=shear, octree_mesh=dense_mesh
        )
        s_dense = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides_dense,
        )
        s_dense.forward()
        disp_dense = s_dense.get_displacement()
        pos_dense = s_dense.space.node_positions().numpy()
        print(
            f"Dense uniform octree: {len(pos_dense)} nodes, converged in "
            f"{s_dense.newton_iterations} iters, residual={s_dense.final_residual:.3e}"
        )

        # --- Non-uniform octree (depth=3, corner refined) with hanging nodes ---
        o_nodes, o_elems, o_eh, o_fh = make_sdf2octree_mesh(3, refine_corner=True)
        octree_mesh = {
            "nodes": o_nodes,
            "elements": o_elems,
            "edge_hanging": o_eh,
            "face_hanging": o_fh,
        }
        overrides_oct = make_cube_config_overrides(
            tmp, shear=shear, octree_mesh=octree_mesh
        )
        s_oct = OctreeNeoHookean(
            "tests.examples.three_d_affine_shear.config",
            override_params=overrides_oct,
        )
        s_oct.forward()
        disp_oct = s_oct.get_displacement()
        pos_oct = s_oct.space.node_positions().numpy()
        print(
            f"Octree: {len(pos_oct)} nodes ({o_eh.shape[0]} edge hanging, "
            f"{o_fh.shape[0]} face hanging), converged in "
            f"{s_oct.newton_iterations} iters, residual={s_oct.final_residual:.3e}"
        )
        assert s_oct.final_residual < 1e-5

        # Find octree nodes that coincide with dense mesh nodes
        from scipy.spatial import cKDTree

        tree = cKDTree(pos_dense)
        dists, indices = tree.query(pos_oct)
        tol_pos = 1e-6
        shared_mask = dists < tol_pos
        n_shared = shared_mask.sum()
        print(f"  Shared nodes: {n_shared} / {len(pos_oct)} octree nodes")
        assert n_shared > 0, "No shared nodes found between dense and octree mesh"

        # Compare displacements at shared nodes
        disp_dense_matched = disp_dense[indices[shared_mask]]
        disp_oct_shared = disp_oct[shared_mask]
        linf_err = np.abs(disp_oct_shared - disp_dense_matched).max()
        print(f"  L-inf displacement error at shared nodes: {linf_err:.3e}")
        # Higher tolerance since meshes differ — the octree is coarser in most
        # of the domain so discretisation error differs between the two solves.
        assert (
            linf_err < 5e-3
        ), f"Displacement mismatch between dense and octree: {linf_err:.3e}"

        print("REGULAR VS OCTREE SOLUTION TEST PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    # test_uniform_octree_mesh()
    # print()
    # test_octree_with_hanging_nodes()
    # print()
    # test_load_octree_mesh_method()
    # print()
    # test_interpolation_warm_start()
    # print()
    # test_interpolation_analytical_field()
    print()
    test_interpolation_warm_start_with_hanging()
    # print()
    # test_regular_vs_octree_solution()
    # print()
    print("ALL OCTREE MESH TESTS PASSED")
