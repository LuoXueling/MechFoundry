"""Compare classical FEM on standard hex mesh vs Nitsche on trimesh-generated octree.

Three solver variants:
  1. Standard: body-fitted hex mesh (ground truth)
  2. Nitsche:  axis-aligned box embedded in octree (partially conforming)
  3. Rotated:  45°-rotated box embedded in the same octree (fully non-conforming)

All use the same twisting problem:
  - Bottom face: fully clamped
  - Top face: rigid rotation θ about z-axis through center

Leaf-local solver implementation used by ``run.py``.
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# Fix sys.path: Python adds the script's directory (src/warp_wrapper/) as
# sys.path[0].  Replace with the project root so namespace imports work.
LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
sys.path.insert(0, os.fspath(LEAF_DIR))
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

import numpy as np
import torch
import trimesh

from src.triposg_wrapper.trimesh_sdf import (
    trimesh_to_octree,
    make_trimesh_geometric_func,
)

# ---------------------------------------------------------------------------
# Geometry and boundary conditions
# ---------------------------------------------------------------------------

HALF_SIDE = 0.75  # grid-aligned: 0.75/(2/2^d) = integer for d >= 3
OCTREE_DEPTH = 6
QUAD_REFINEMENT_DEPTH = 1

THETA = np.pi / 4
COS_T = np.cos(THETA)
SIN_T = np.sin(THETA)

# ---------------------------------------------------------------------------
# Rotation for the non-conforming variant
# ---------------------------------------------------------------------------
# 45° rotation about the x-axis.  The y- and z-faces become non-conforming
# while the x-faces stay aligned.
_a = np.pi / 4
ROT_MAT = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(_a), -np.sin(_a)],
        [0.0, np.sin(_a), np.cos(_a)],
    ]
)
ROT_INV = ROT_MAT.T  # global → local

# Bounding box for Rx(45°): y,z extents = hs * √2.
# HALF_SIDE = 0.75 → max extent ≈ 1.061, needs domain > 2.0.
# Use domain_length = ceil(2 * hs * √2 * 2^d) / 2^d * 2 to grid-align.
_ROT_BBOX_HALF = HALF_SIDE * np.sqrt(2.0)  # ≈ 1.0607
# Round up to next multiple of h_coarse = domain_length / 2^depth.
# We pick domain_length such that _ROT_BBOX_HALF < domain_length / 2.
# 2 * 1.0607 = 2.121 → use 2.25 (grid-aligns at depth 6: 2.25/64 = 0.03515625)
DOMAIN_LENGTH_ROTATED = 2.25

# Displacement matrix for the rotated top face:
#   u(x) = M @ x  where  M = R @ (R_theta - I) @ R^T
# R_theta is the small twist about the local z-axis.
_R_theta = np.array([[COS_T, -SIN_T, 0], [SIN_T, COS_T, 0], [0, 0, 1]])
DISP_MAT = ROT_MAT @ (_R_theta - np.eye(3)) @ ROT_INV


# Rotation about origin — used by Nitsche (domain centered at origin)
def ux_top(x, y, z):
    return x * (COS_T - 1.0) - y * SIN_T


def uy_top(x, y, z):
    return x * SIN_T + y * (COS_T - 1.0)


def uz_top(x, y, z):
    return 0.0 * x  # tensor-compatible zero


# Rotation about face center (hs, hs) — used by standard (domain [0, 2*hs]^3)
def _ux_top_standard(x, y, z):
    return (x - HALF_SIDE) * (COS_T - 1.0) - (y - HALF_SIDE) * SIN_T


def _uy_top_standard(x, y, z):
    return (x - HALF_SIDE) * SIN_T + (y - HALF_SIDE) * (COS_T - 1.0)


def _uz_top_standard(x, y, z):
    return 0.0


# Rotated BC: u = DISP_MAT @ x (rotation about local z, expressed in global frame)
_M = DISP_MAT  # alias for shorter closures


def _ux_top_rotated(x, y, z):
    return _M[0, 0] * x + _M[0, 1] * y + _M[0, 2] * z


def _uy_top_rotated(x, y, z):
    return _M[1, 0] * x + _M[1, 1] * y + _M[1, 2] * z


def _uz_top_rotated(x, y, z):
    return _M[2, 0] * x + _M[2, 1] * y + _M[2, 2] * z


# Shared solver settings
SOLVER_CONFIG = dict(
    E=10.0,
    nu=0.3,
    B=(0.0, 0.0, 0.0),
    relative_tol=1e-5,
    absolute_tol=1e-8,
    max_iter=50,
    linear_solver="cg",
    cg_backend_assembled="amgx",
    amgx_preconditioner="amg",
    load_steps=20,
    use_mixed_precision=True,
)


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------


def run_standard(n_element, output_dir="output/compare_standard"):
    """Classical FEM on a standard n x n x n hex mesh in [0, 2*hs]^3."""
    from core.neoHookean import NeoHookean

    os.makedirs(output_dir, exist_ok=True)
    hs = HALF_SIDE

    overrides = {
        "project_name": "compare_standard",
        "output_root": output_dir,
        "dim": 3,
        "q_deg": 2,
        "elem_order": 1,
        "width": 2 * hs,
        "height": 2 * hs,
        "depth": 2 * hs,
        "nx": n_element,
        "ny": n_element,
        "nz": n_element,
        "device": "cuda",
        "dirichlet_bcs_dict": {
            lambda x, y, z: torch.isclose(z, torch.zeros_like(z)): 0.0,
            lambda x, y, z: torch.isclose(z, torch.full_like(z, 2 * hs)): (
                _ux_top_standard,
                _uy_top_standard,
                _uz_top_standard,
            ),
        },
        "neumann_bcs_dict": {},
        **SOLVER_CONFIG,
    }

    print(f"\n{'='*60}")
    print(f"STANDARD MESH: {n_element}x{n_element}x{n_element} elements")
    print(f"{'='*60}")

    solver = NeoHookean(
        "config", override_params=overrides
    )

    t0 = time.perf_counter()
    solver.forward()
    t_solve = time.perf_counter() - t0
    solver.save()

    print(f"  Newton iterations: {solver.newton_iterations}")
    print(f"  Final residual:    {solver.final_residual:.3e}")
    print(f"  Solve time:        {t_solve:.2f}s")

    # Build node positions analytically to match the structured-grid ordering.
    nx, ny, nz = n_element, n_element, n_element
    xs = np.linspace(0, 2 * hs, nx + 1)
    ys = np.linspace(0, 2 * hs, ny + 1)
    zs = np.linspace(0, 2 * hs, nz + 1)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    solver._cached_node_positions = np.stack(
        [gx.ravel(), gy.ravel(), gz.ravel()], axis=1
    )

    return solver, t_solve


def run_nitsche(
    output_dir="output/compare_nitsche",
    epsilon=None,
    quad_refinement_depth=QUAD_REFINEMENT_DEPTH,
    quad_refinement_depth_interior=None,
    octree_depth=OCTREE_DEPTH,
    init_octree_depth=3,
    use_mixed_precision=None,
    tangent_f64=None,
    cg_backend_assembled=None,
):
    """Nitsche FEM on a [-hs, hs]^3 box embedded in [-1,1]^3 octree."""
    from core.nitsche_field import NitscheNeoHookean

    os.makedirs(output_dir, exist_ok=True)
    hs = HALF_SIDE
    depth = octree_depth

    mesh = trimesh.creation.box(extents=(2 * hs,) * 3)
    coarse = trimesh_to_octree(
        mesh,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        init_octree_depth=init_octree_depth,
        octree_depth=depth,
        domain_length=2.0,
        device="cuda",
    )

    h_coarse = 2.0 / (2 ** depth)
    h_quad = h_coarse / (2 ** quad_refinement_depth)
    if epsilon is None:
        epsilon = h_quad / 2.0

    ri = quad_refinement_depth_interior
    ri_label = f", ri={ri}" if ri is not None else ""
    print(f"\n{'='*60}")
    print(
        f"NITSCHE: {2*hs}^3 box, depth={depth}, ref_depth={quad_refinement_depth}{ri_label}"
    )
    print(f"{'='*60}")
    print(f"  h_coarse={h_coarse:.6f}, epsilon={epsilon:.6f}")
    n_octree = coarse["elements"].shape[0]
    n_uniform = (2 ** depth) ** 3
    print(
        f"  octree elements: {n_octree} (vs {n_uniform} uniform, {n_octree/n_uniform:.1%})"
    )

    geometric_func = make_trimesh_geometric_func(mesh, device="cuda")

    overrides = {
        "project_name": "compare_nitsche",
        "output_root": output_dir,
        "dim": 3,
        "q_deg": 2,
        "elem_order": 1,
        "octree_mesh": coarse,
        "phase_field_epsilon": epsilon,
        "device": "cuda",
        "use_mixed_precision": True,
        "tangent_f64": True,
        "line_search_enabled": True,
        "ew_enabled": False,
        "phi_floor": 1e-4,
        "trim_exterior": True,
        "sharp_volumetric": True,
        "heuristic_beta": False,
        "geometric_func": geometric_func,
        "quad_refinement_depth": quad_refinement_depth,
        "dirichlet_bcs_dict": {
            lambda x, y, z: torch.isclose(z, torch.full_like(z, -hs)): 0.0,
            lambda x, y, z: torch.isclose(z, torch.full_like(z, hs)): (
                ux_top,
                uy_top,
                uz_top,
            ),
        },
        "neumann_bcs_dict": {},
        **SOLVER_CONFIG,
    }
    # Optional arguments let precision/backend validation exercise this exact
    # production benchmark without mutating the module-wide showcase defaults.
    if use_mixed_precision is not None:
        overrides["use_mixed_precision"] = use_mixed_precision
    if tangent_f64 is not None:
        overrides["tangent_f64"] = tangent_f64
    if cg_backend_assembled is not None:
        overrides["cg_backend_assembled"] = cg_backend_assembled
    if ri is not None:
        overrides["quad_refinement_depth_interior"] = ri

    solver = NitscheNeoHookean(
        "config",
        override_params=overrides,
    )

    t0 = time.perf_counter()
    solver.forward()
    t_solve = time.perf_counter() - t0
    solver.save()

    print(f"  Newton iterations: {solver.newton_iterations}")
    print(f"  Final residual:    {solver.final_residual:.3e}")
    print(f"  Solve time:        {t_solve:.2f}s")

    diagnose_grad_phi(solver)
    return solver, t_solve, epsilon


def run_nitsche_rotated(
    output_dir="output/compare_nitsche_rotated",
    epsilon=None,
    quad_refinement_depth=QUAD_REFINEMENT_DEPTH,
    quad_refinement_depth_interior=None,
    octree_depth=OCTREE_DEPTH,
    init_octree_depth=3,
    use_mixed_precision=None,
    tangent_f64=None,
    cg_backend_assembled=None,
):
    """Nitsche FEM on a ROTATED box embedded in a larger octree domain.

    The cube is rotated 45° about x so that y- and z-faces are non-conforming.
    The domain is expanded beyond [-1,1]^3 to fit the rotated bounding box.
    BCs are transformed so the physical problem (twist about local z) is identical.
    """
    from core.nitsche_field import NitscheNeoHookean

    os.makedirs(output_dir, exist_ok=True)
    hs = HALF_SIDE
    depth = octree_depth
    dl = DOMAIN_LENGTH_ROTATED
    half_dl = dl / 2.0

    # Build rotated box mesh
    T = np.eye(4)
    T[:3, :3] = ROT_MAT
    mesh = trimesh.creation.box(extents=(2 * hs,) * 3, transform=T)

    coarse = trimesh_to_octree(
        mesh,
        bounds=(-half_dl, -half_dl, -half_dl, half_dl, half_dl, half_dl),
        init_octree_depth=init_octree_depth,
        octree_depth=depth,
        domain_length=dl,
        device="cuda",
    )

    h_coarse = dl / (2 ** depth)
    h_quad = h_coarse / (2 ** quad_refinement_depth)
    if epsilon is None:
        epsilon = h_quad / 2.0

    # BC tolerance: these predicates use abs, not isclose, so the solver's
    # widened isclose never reaches them -- match its _bc_match_tolerance()
    # here.  The old max(2 eps, 1.5 h) clamped a strip 1.5 h up the side faces.
    bc_tol = max(1.5 * epsilon, 0.75 * h_coarse)

    ri = quad_refinement_depth_interior
    ri_label = f", ri={ri}" if ri is not None else ""
    print(f"\n{'='*60}")
    print(
        f"NITSCHE ROTATED: {2*hs}^3 box, domain={dl}^3, depth={depth}, "
        f"ref_depth={quad_refinement_depth}{ri_label}"
    )
    print(f"{'='*60}")
    print(f"  h_coarse={h_coarse:.6f}, epsilon={epsilon:.6f}")
    print(f"  bc_tol={bc_tol:.6f}")
    print(f"  bbox of rotated cube: {mesh.bounds.tolist()}")
    n_octree = coarse["elements"].shape[0]
    n_uniform = (2 ** depth) ** 3
    print(
        f"  octree elements: {n_octree} (vs {n_uniform} uniform, {n_octree/n_uniform:.1%})"
    )

    geometric_func = make_trimesh_geometric_func(mesh, device="cuda")

    # Rotated face predicates: z_local = ROT_INV[2,:] · (x,y,z)
    _r2 = ROT_INV[2]  # third row of R^T
    _r20, _r21, _r22 = float(_r2[0]), float(_r2[1]), float(_r2[2])
    _hs = float(hs)
    _tol = float(bc_tol)

    def pred_bottom(x, y, z):
        z_local = _r20 * x + _r21 * y + _r22 * z
        return torch.abs(z_local - (-_hs)) < _tol

    def pred_top(x, y, z):
        z_local = _r20 * x + _r21 * y + _r22 * z
        return torch.abs(z_local - _hs) < _tol

    overrides = {
        "project_name": "compare_nitsche_rotated",
        "output_root": output_dir,
        "dim": 3,
        "q_deg": 2,
        "elem_order": 1,
        "octree_mesh": coarse,
        "phase_field_epsilon": epsilon,
        "device": "cuda",
        "use_mixed_precision": True,
        "tangent_f64": True,
        "line_search_enabled": True,
        "ew_enabled": False,
        "phi_floor": 1e-4,
        "trim_exterior": True,
        "sharp_volumetric": True,
        "heuristic_beta": False,
        "geometric_func": geometric_func,
        "quad_refinement_depth": quad_refinement_depth,
        "dirichlet_bcs_dict": {
            pred_bottom: 0.0,
            pred_top: (_ux_top_rotated, _uy_top_rotated, _uz_top_rotated),
        },
        "neumann_bcs_dict": {},
        **SOLVER_CONFIG,
    }
    if use_mixed_precision is not None:
        overrides["use_mixed_precision"] = use_mixed_precision
    if tangent_f64 is not None:
        overrides["tangent_f64"] = tangent_f64
    if cg_backend_assembled is not None:
        overrides["cg_backend_assembled"] = cg_backend_assembled
    if ri is not None:
        overrides["quad_refinement_depth_interior"] = ri

    solver = NitscheNeoHookean(
        "config",
        override_params=overrides,
    )

    t0 = time.perf_counter()
    solver.forward()
    t_solve = time.perf_counter() - t0
    solver.save()

    print(f"  Newton iterations: {solver.newton_iterations}")
    print(f"  Final residual:    {solver.final_residual:.3e}")
    print(f"  Solve time:        {t_solve:.2f}s")

    diagnose_grad_phi(solver)
    return solver, t_solve, epsilon


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def diagnose_grad_phi(solver):
    """Compare numerical |grad phi| at boundary QPs with analytical values."""
    import warp as wp
    eps = solver._epsilon
    # Gather QP data from chunks if available, else from flat arrays
    if hasattr(solver, "_nitsche_chunks") and solver._nitsche_chunks:
        phi_t = torch.cat([wp.to_torch(c.phi_qp) for c in solver._nitsche_chunks])
        gpm_t = torch.cat(
            [wp.to_torch(c.grad_phi_mag_qp) for c in solver._nitsche_chunks]
        )
        active_t = torch.cat(
            [wp.to_torch(c.qp_active_wp) for c in solver._nitsche_chunks]
        )
    else:
        phi_t = wp.to_torch(solver._phi_qp)
        gpm_t = wp.to_torch(solver._grad_phi_mag_qp)
        active_t = wp.to_torch(solver._qp_active_wp)

    mask = (active_t > 0) & (phi_t > 0.01) & (phi_t < 0.99)
    phi_b = phi_t[mask]
    gpm_b = gpm_t[mask]
    gpm_analytical = 2.0 * phi_b * (1.0 - phi_b) / eps

    ratio = gpm_b / gpm_analytical.clamp(min=1e-30)
    err_rel = (gpm_b - gpm_analytical).abs() / gpm_analytical.clamp(min=1e-30)

    print(f"\n  --- |grad phi| diagnostic ({mask.sum().item()} boundary QPs) ---")
    print(f"  epsilon = {eps:.6f}")
    print(
        f"  |grad phi| numerical:  [{gpm_b.min():.2f}, {gpm_b.max():.2f}], mean={gpm_b.mean():.2f}"
    )
    print(
        f"  |grad phi| analytical: [{gpm_analytical.min():.2f}, {gpm_analytical.max():.2f}], mean={gpm_analytical.mean():.2f}"
    )
    print(f"  ratio: mean={ratio.mean():.4f}, median={ratio.median():.4f}")
    print(f"  relative error: mean={err_rel.mean():.4f}, max={err_rel.max():.4f}")


def compare_displacement(standard_solver, nitsche_solver):
    """Compare displacement between standard and Nitsche via nearest-neighbor matching.

    Uses _extract_solution to get (pts, u) in a common frame, then matches
    Nitsche nodes to their nearest standard node via KDTree.  Reports per
    z-station mean |u| ratios and overall L2 relative error.
    """
    from scipy.spatial import cKDTree

    hs = HALF_SIDE
    pts_ref, u_ref = _extract_solution(standard_solver, "standard")
    pts_nit, u_nit = _extract_solution(nitsche_solver, "nitsche")

    tree = cKDTree(pts_ref)
    dists, idx = tree.query(pts_nit)

    print(f"\n{'='*60}")
    print("DISPLACEMENT COMPARISON: standard vs Nitsche")
    print(f"{'='*60}")

    print("\n  z_frac  |u|_std     |u|_nit     ratio    (matched nodes)")
    for z_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_val = (z_frac - 0.5) * 2 * hs  # map to [-hs, hs]
        h_ref = 2 * hs / max(round(len(pts_ref) ** (1.0 / 3.0)) - 1, 1)
        near_z = np.abs(pts_nit[:, 2] - z_val) < max(0.04, 2 * h_ref)
        good = near_z & (dists < 2 * h_ref)
        if good.sum() == 0:
            print(f"    {z_frac:.2f}   insufficient nodes")
            continue
        mag_ref = np.linalg.norm(u_ref[idx[good]], axis=1).mean()
        mag_nit = np.linalg.norm(u_nit[good], axis=1).mean()
        ratio = mag_nit / mag_ref if mag_ref > 1e-15 else float("nan")
        print(
            f"    {z_frac:.2f}    {mag_ref:.4e}  {mag_nit:.4e}  {ratio:.4f}   ({good.sum()})"
        )

    # Overall L2 error (interior nodes only)
    z_nit = pts_nit[:, 2]
    interior = (np.abs(z_nit) > 0.1 * hs) & (np.abs(z_nit) < 0.9 * hs)
    h_ref = 2 * hs / max(round(len(pts_ref) ** (1.0 / 3.0)) - 1, 1)
    good = interior & (dists < 2 * h_ref)
    if good.sum() > 0:
        err = np.linalg.norm(u_nit[good] - u_ref[idx[good]], axis=1)
        ref_norm = np.linalg.norm(u_ref[idx[good]], axis=1)
        l2_rel = np.sqrt((err ** 2).sum() / (ref_norm ** 2).sum())
        print(f"\n  L2 relative error (interior): {l2_rel:.6f}  ({good.sum()} nodes)")


def compare_displacement_rotated(nitsche_solver, rotated_solver):
    """Compare axis-aligned Nitsche vs rotated Nitsche via nearest-neighbor matching.

    Both solutions are extracted into the aligned [-hs,hs]^3 frame via
    _extract_solution (which rotates the rotated solver's data back).
    """
    from scipy.spatial import cKDTree

    hs = HALF_SIDE
    pts_a, u_a = _extract_solution(nitsche_solver, "nitsche")
    pts_r, u_r = _extract_solution(rotated_solver, "rotated")

    # Match rotated nodes to nearest aligned node
    tree = cKDTree(pts_a)
    dists, idx = tree.query(pts_r)

    print(f"\n{'='*60}")
    print("DISPLACEMENT COMPARISON: axis-aligned vs rotated Nitsche")
    print(f"{'='*60}")

    # Infer spacing from aligned mesh (finer near boundary due to octree)
    h_ref = 2 * hs / max(round(len(pts_a) ** (1.0 / 3.0)) - 1, 1)
    dist_tol = max(0.04, 3 * h_ref)

    print(f"\n  z_frac  |u|_aligned  |u|_rotated  ratio    (matched)")
    for z_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_val = (z_frac - 0.5) * 2 * hs
        near_z = np.abs(pts_r[:, 2] - z_val) < max(0.04, 2 * h_ref)
        good = near_z & (dists < dist_tol)
        if good.sum() == 0:
            print(f"    {z_frac:.2f}   insufficient nodes")
            continue
        mag_a = np.linalg.norm(u_a[idx[good]], axis=1).mean()
        mag_r = np.linalg.norm(u_r[good], axis=1).mean()
        ratio = mag_r / mag_a if mag_a > 1e-15 else float("nan")
        print(
            f"    {z_frac:.2f}    {mag_a:.4e}    {mag_r:.4e}   {ratio:.4f}   ({good.sum()})"
        )

    # Overall L2 error (interior)
    z_r = pts_r[:, 2]
    interior = (np.abs(z_r) > 0.1 * hs) & (np.abs(z_r) < 0.9 * hs)
    good = interior & (dists < dist_tol)
    if good.sum() > 0:
        err = np.linalg.norm(u_r[good] - u_a[idx[good]], axis=1)
        ref_norm = np.linalg.norm(u_a[idx[good]], axis=1)
        l2_rel = np.sqrt((err ** 2).sum() / (ref_norm ** 2).sum())
        print(f"\n  L2 relative error (interior): {l2_rel:.6f}  ({good.sum()} nodes)")


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------


def _extract_solution(solver, kind):
    """Extract (positions, displacements) arrays from a solver.

    kind: "standard", "nitsche", or "rotated"
    Returns (pts, u) as numpy arrays — pts in the *aligned* [-hs, hs]^3 frame.

    For the "rotated" kind, both positions and displacements are rotated back
    to the aligned frame via R^T so that spatial matching against the standard
    solver works correctly.
    """
    hs = HALF_SIDE
    u = solver.u.dof_values.numpy().reshape(-1, 3)
    if kind == "standard":
        pts = solver._cached_node_positions - hs  # [0, 2*hs] → [-hs, hs]
        mask = np.ones(len(pts), dtype=bool)
    else:
        pts = solver._prev_octree_nodes.cpu().numpy()
        phi = solver._phi_nodal.cpu().numpy()
        mask = phi > 0.9
    pts, u = pts[mask], u[mask]
    if kind == "rotated":
        # Rotate back to aligned frame: x_local = R^T @ x_global
        # For row vectors: x_local_row = x_global_row @ R
        pts = pts @ ROT_MAT
        u = u @ ROT_MAT
    return pts, u


def _local_z(pts, kind):
    """Return the local z-coordinate (twist axis) for each node.

    Since _extract_solution already rotates "rotated" positions back to
    the aligned frame, local z = pts[:, 2] for all kinds.
    """
    return pts[:, 2]


def displacement_error(ref_data, test_solver, test_kind):
    """Relative L2 displacement error of test vs reference.

    ref_data: (pts_ref, u_ref) numpy arrays from _extract_solution.
    Matches test nodes to closest reference nodes, then computes
        ||u_test - u_ref||_2 / ||u_ref||_2
    over nodes away from BC faces (0.1*hs < |z_local| < 0.9*hs).
    """
    from scipy.spatial import cKDTree

    hs = HALF_SIDE
    pts_ref, u_ref = ref_data
    pts_test, u_test = _extract_solution(test_solver, test_kind)

    # Restrict test nodes to interior (away from BC faces) using local z
    z_test = _local_z(pts_test, test_kind)
    interior = (np.abs(z_test) > 0.1 * hs) & (np.abs(z_test) < 0.9 * hs)
    if interior.sum() < 10:
        return float("nan")
    pts_test = pts_test[interior]
    u_test = u_test[interior]

    tree = cKDTree(pts_ref)
    dists, idx = tree.query(pts_test)

    # Infer mesh spacing from reference
    h_ref = 2 * hs / max(round(len(pts_ref) ** (1.0 / 3.0)) - 1, 1)
    good = dists < 1.5 * h_ref
    if good.sum() < 10:
        good = dists < 5.0 * h_ref
    if good.sum() < 10:
        return float("nan")

    u_ref_m = u_ref[idx[good]]
    u_test_m = u_test[good]

    err = np.linalg.norm(u_test_m - u_ref_m, axis=1)
    ref_norm = np.linalg.norm(u_ref_m, axis=1)

    return np.sqrt((err ** 2).sum() / (ref_norm ** 2).sum())


def _run_variant(
    runner, key, label, octree_depth, kwargs, output_prefix, ref_data, test_kind, rows_d
):
    """Run a single Nitsche variant and record results."""
    suffix = key.replace("+", "p")
    print(f"\n--- [{label} d={octree_depth}] ---")
    try:
        s, t, eps = runner(
            output_dir=f"output/convergence/{output_prefix}_{suffix}",
            octree_depth=octree_depth,
            **kwargs,
        )
        err = displacement_error(ref_data, s, test_kind)
        rows_d[key] = (err, t, eps)
        print(f"  -> L2 rel error = {err:.6f}, time = {t:.1f}s, eps={eps:.6f}")
        del s
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  -> FAILED: {e}")
        import traceback

        traceback.print_exc()
        rows_d[key] = (float("nan"), 0.0, float("nan"))


def run_convergence_study(octree_depths):
    """Automatic comparison at each octree depth.

    For each depth d, runs:
      A.  Standard FEM at n = 2^d               (reference)
      B-F. Axis-aligned Nitsche variants         (partially conforming)
      G-K. Rotated (45 deg) Nitsche variants     (fully non-conforming)

    All variants are compared against A (body-fitted mesh).
    """
    print(f"\n{'#'*60}")
    print("CONVERGENCE STUDY")
    print(f"{'#'*60}")

    # Refinement settings shared by both axis-aligned and rotated families
    SETTINGS = [
        ("r0", "r=0", dict(quad_refinement_depth=0)),
        ("r1", "r=1", dict(quad_refinement_depth=1)),
        (
            "r2_ri0",
            "r2,ri0",
            dict(quad_refinement_depth=2, quad_refinement_depth_interior=0),
        ),
        (
            "r2_ri1",
            "r2,ri1",
            dict(quad_refinement_depth=2, quad_refinement_depth_interior=1),
        ),
        ("r0_2eps", "r0,2eps", dict(quad_refinement_depth=0, _eps_mult=2)),
        ("r0_4eps", "r0,4eps", dict(quad_refinement_depth=0, _eps_mult=4)),
    ]

    NIT_VARIANTS = [(f"nit_{s}", f"Nit({lb})", kw) for s, lb, kw in SETTINGS]
    ROT_VARIANTS = [(f"rot_{s}", f"Rot({lb})", kw) for s, lb, kw in SETTINGS]
    ALL_VARIANTS = NIT_VARIANTS + ROT_VARIANTS

    # Single high-resolution reference: Standard FEM at finest depth
    d_ref = max(octree_depths)
    n_ref = 2 ** d_ref
    print(f"\n--- [Reference: Std {n_ref}^3] ---")
    ref_solver, t_ref = run_standard(
        n_element=n_ref, output_dir=f"output/convergence/std_{n_ref}"
    )
    # Extract reference data to CPU numpy and free GPU memory
    ref_data = _extract_solution(ref_solver, "standard")
    del ref_solver
    torch.cuda.empty_cache()

    rows = {}
    for d in octree_depths:
        rows[d] = {}
        rows[d]["std"] = (t_ref,)

        def _run_family(variants, settings, runner, prefix, test_kind):
            for (key, label, kwargs), (skey, _, _) in zip(variants, settings):
                kw = dict(kwargs)
                eps_mult = kw.pop("_eps_mult", None)
                if eps_mult is not None:
                    dl = DOMAIN_LENGTH_ROTATED if runner is run_nitsche_rotated else 2.0
                    h_coarse = dl / 2 ** d
                    kw["epsilon"] = eps_mult * h_coarse / 2.0
                _run_variant(
                    runner,
                    key,
                    label,
                    d,
                    kw,
                    f"{prefix}_d{d}",
                    ref_data,
                    test_kind,
                    rows[d],
                )

        _run_family(NIT_VARIANTS, SETTINGS, run_nitsche, "nit", "nitsche")
        _run_family(ROT_VARIANTS, SETTINGS, run_nitsche_rotated, "rot", "rotated")

    # --- Printing helpers ---
    def _fmt_err(r, key):
        val = r.get(key, (float("nan"), 0.0, float("nan")))
        if isinstance(val[0], str):
            return f"{val[0]:>14s}"
        return f"{'FAIL':>14s}" if np.isnan(val[0]) else f"{val[0]:>14.6f}"

    def _fmt_eps(r, key):
        val = r.get(key, (float("nan"), 0.0, float("nan")))
        eps = val[2] if len(val) > 2 else float("nan")
        return f"{'--':>14s}" if np.isnan(eps) else f"{eps:>14.6f}"

    def _fmt_time(r, key):
        val = r.get(key, (float("nan"), 0.0, float("nan")))
        return f"{val[1]:>14.1f}" if len(val) > 1 else f"{0.0:>14.1f}"

    def _print_table(title, variant_list, value_fn, subtitle=""):
        labels = [lb for _, lb, _ in variant_list]
        keys = [k for k, _, _ in variant_list]
        n = len(labels)
        sep = "=" * (20 + 16 * n)
        print(f"\n{sep}")
        print(title)
        if subtitle:
            print(f"  {subtitle}")
        print(sep)
        hdr = f"  {'depth':>5s}  {'h':>10s}"
        for lb in labels:
            hdr += f"  {lb:>14s}"
        print(hdr)
        print(f"  {'':-<5s}  {'':-<10s}" + f"  {'':-<14s}" * n)
        for d in octree_depths:
            h = 2.0 / 2 ** d
            line = f"  {d:>5d}  {h:>10.6f}"
            for k in keys:
                line += f"  {value_fn(rows[d], k)}"
            print(line)

    # Error tables
    _print_table(
        "AXIS-ALIGNED NITSCHE: L2 relative error vs Standard",
        NIT_VARIANTS,
        _fmt_err,
        "Error = ||u_nitsche - u_standard||_2 / ||u_standard||_2",
    )
    _print_table(
        "ROTATED NITSCHE (45 deg): L2 relative error vs Standard",
        ROT_VARIANTS,
        _fmt_err,
        "Error = ||u_rotated - u_standard||_2 / ||u_standard||_2",
    )

    # Epsilon tables
    _print_table("AXIS-ALIGNED NITSCHE: Epsilon", NIT_VARIANTS, _fmt_eps)
    _print_table("ROTATED NITSCHE: Epsilon", ROT_VARIANTS, _fmt_eps)

    # Combined time table
    all_labels = [lb for _, lb, _ in ALL_VARIANTS]
    all_keys = [k for k, _, _ in ALL_VARIANTS]
    n_all = len(all_labels)
    sep = "=" * (36 + 16 * n_all)
    print(f"\n{sep}")
    print("SOLVE TIMES (seconds)")
    print(sep)
    hdr = f"  {'depth':>5s}  {'h':>10s}  {'Std n^3':>14s}"
    for lb in all_labels:
        hdr += f"  {lb:>14s}"
    print(hdr)
    for d in octree_depths:
        h = 2.0 / 2 ** d
        r = rows[d]
        t_std = r["std"][0] if r["std"] else 0.0
        line = f"  {d:>5d}  {h:>10.6f}  {t_std:>14.1f}"
        for k in all_keys:
            line += f"  {_fmt_time(r, k)}"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare standard FEM vs Nitsche on a 1.5^3 box"
    )
    parser.add_argument("--skip-standard", action="store_true")
    parser.add_argument("--skip-nitsche", action="store_true")
    parser.add_argument(
        "--run-rotated",
        action="store_true",
        help="Also run 45°-rotated Nitsche (fully non-conforming)",
    )
    parser.add_argument("--n-standard", type=int, default=48)
    parser.add_argument(
        "--quad-refinement-depth", type=int, default=QUAD_REFINEMENT_DEPTH
    )
    parser.add_argument("--quad-refinement-depth-interior", type=int, default=0)
    parser.add_argument("--octree-depth", type=int, default=OCTREE_DEPTH)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument(
        "--convergence",
        action="store_true",
        help="Run three-way convergence study (standard, Nitsche r=1, Nitsche d+1 r=0)",
    )
    parser.add_argument(
        "--conv-depths",
        type=int,
        nargs="+",
        default=[4, 5, 6],
        help="Octree depths to sweep (default: 4 5 6)",
    )
    args = parser.parse_args()

    if args.convergence:
        run_convergence_study(args.conv_depths)
        return

    standard_solver, t_standard = None, 0
    if not args.skip_standard:
        standard_solver, t_standard = run_standard(n_element=args.n_standard)

    nitsche_solver, t_nitsche = None, 0
    if not args.skip_nitsche:
        nitsche_solver, t_nitsche, _ = run_nitsche(
            epsilon=args.epsilon,
            quad_refinement_depth=args.quad_refinement_depth,
            quad_refinement_depth_interior=args.quad_refinement_depth_interior,
            octree_depth=args.octree_depth,
        )

    rotated_solver, t_rotated = None, 0
    if args.run_rotated:
        rotated_solver, t_rotated, _ = run_nitsche_rotated(
            epsilon=args.epsilon,
            quad_refinement_depth=args.quad_refinement_depth,
            quad_refinement_depth_interior=args.quad_refinement_depth_interior,
            octree_depth=args.octree_depth,
        )

    if standard_solver is not None and nitsche_solver is not None:
        compare_displacement(standard_solver, nitsche_solver)

    if nitsche_solver is not None and rotated_solver is not None:
        compare_displacement_rotated(nitsche_solver, rotated_solver)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if standard_solver is not None:
        n = args.n_standard ** 3
        print(
            f"  Standard: {n:>8d} elem  |  {standard_solver.newton_iterations} iters  |  "
            f"{t_standard:.2f}s  |  res={standard_solver.final_residual:.3e}"
        )
    if nitsche_solver is not None:
        print(
            f"  Nitsche:  {'box':>8s} embed |  {nitsche_solver.newton_iterations} iters  |  "
            f"{t_nitsche:.2f}s  |  res={nitsche_solver.final_residual:.3e}"
        )
    if rotated_solver is not None:
        print(
            f"  Rotated:  {'rot45':>8s} embed |  {rotated_solver.newton_iterations} iters  |  "
            f"{t_rotated:.2f}s  |  res={rotated_solver.final_residual:.3e}"
        )


if __name__ == "__main__":
    main()
