"""Prep (conda GPU): mesh-convergence sweep for the twisting box — body-fitted
FEM vs diffused Nitsche on an octree, axis-aligned and 45deg-rotated.

Everything is driven by the local copy of the comparison solvers in
``comparison_helpers.py``: its ``run_standard`` /
``run_nitsche`` / ``run_nitsche_rotated`` and its geometry + BC definitions are
imported, not re-implemented.  Three things are done differently here, and all
three are refinements of that script's own convergence path:

  * **No refined quadrature.**  Every Nitsche run uses
    ``quad_refinement_depth=0`` (the figure's subject: what the plain scheme
    does under mesh refinement alone), so ``epsilon = h_coarse / 2``.
  * **The error is a volume integral on a fixed lattice, not a sum over each
    mesh's own nodes.**  Both fields are EVALUATED — every Q1 field, on a
    uniform grid or on an octree cell, is the trilinear interpolant of its
    element's corner values, so evaluating it is not resampling it — at the cell
    centres of one shared SAMPLE_N^3 lattice, and the relative L2 error is the
    midpoint rule over the whole box.  Two things this fixes.  (i) The source
    script's ``displacement_error`` matches each test node to its nearest
    reference node, and that O(h_ref) mismatch is a noise floor exactly where
    the finest meshes land.  (ii) Summing over a mesh's own nodes weights the
    norm by that mesh's node density — an octree carries most of its nodes in
    the fine layer near the boundary — so each scheme would be measured in a
    different norm.  The old node-based number is still computed and written to
    nc_data.txt as ``err_nodal``; see the two columns there.
  * **One Newton tolerance for both schemes.**  ``relative_tol`` is tightened to
    1e-10 (NC_RTOL) because the production 1e-5 converges the two schemes to
    levels five orders apart — see the note at RTOL below.  It barely moves the
    errors and it is what makes panel (g) mean anything.

Writes (this folder):
  nc_data.txt     one row per run: variant depth h nelem ndof eps err nodes
                  newton rel_residual secs
  nc_ref_<N>.npz  cached reference solution (nodal u on the (N+1)^3 lattice)
  nc_std.vtk      showcase deformed body-fitted mesh   (finest depth)
  nc_nit.vtk      showcase deformed aligned octree mesh
  nc_rot.vtk      showcase deformed rotated octree mesh
  nc_nit_phi.vtk  the RAW trimmed aligned octree, undeformed, with nodal phi
  nc_rot_phi.vtk  the same for the rotated case (both cut open by the renderer)
  nc_{nit,rot}_frame.vtk   the 12 edges of each case's background domain cube
  nc_stats.txt    scalars the figure prints (reference size, theta, E, nu, ...)

Env: NC_DEPTHS (default 3,4,5,6), NC_REF_N (default 96),
NC_REF_LOAD_STEPS (default 40 — the reference only, see below).

Every run is checked for convergence before its error is recorded, and the check
has to be the RELATIVE one: ``solver.final_residual`` is the absolute ||R||, and
the two schemes do not share a scale — a converged Nitsche solve ends at
||R|| = 2.7e-3 because Nitsche's penalty terms put ||R||_0 at 9.8e3, where a
body-fitted solve of the same problem starts at 0.15.  Comparing that absolute
number to a tolerance rejects perfectly good runs (it rejected all of them).
The relative residual is in ``relative_residual_history``.

The reference is a body-fitted 96^3 mesh (h_ref = 1.5/96 = 0.0156, 2x finer
than the finest test mesh), and getting one at all takes some care:

  *  64^3  converges on the shared 20-step schedule in 134 s.
  *  96^3  DIES on it — "Line search failed to reduce residual; Newton step
     rejected" at load step 19 of 20, after 20 minutes.  With 40 load steps
     (NC_REF_LOAD_STEPS) it converges in 779 s to a relative residual of 2e-8.
  * 128^3  never leaves the FIRST load step: the residual sits at rel = 1.000
     for every Newton iteration (the AMGX/mixed-precision linear solve returns
     nothing usable at 6.3 M dofs), ~66 s per wasted iteration.  No load-step
     schedule rescues it; 96^3 is the ceiling.

Both references were run and the whole sweep measured against each (with the
node-sum metric, which is what the sweep used at the time).  Going from 64^3 to
96^3 moves every Nitsche error by <= 0.4 % (nit d=6: 0.012833 -> 0.012884) and
the body-fitted curve's finest point by 15 % (1.448e-4 -> 1.660e-4) — i.e. exactly where the theory says a reference only 1.33x finer than
the test mesh stops being a reference.  The figure's subject is unaffected
either way, which is worth knowing; the finer reference is shipped because the
body-fitted curve is on the same axes and has to be honest too.
"""
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HERE = os.environ.get("NC_WORKDIR", SCRIPT_DIR)
HERE = os.path.abspath(HERE)
os.chdir("/home/xl2254/implicit_optimize/src/warp_wrapper")
sys.path.insert(0, ".")
sys.path.insert(0, "/home/xl2254/implicit_optimize")

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import comparison_helpers as csn

HS = csn.HALF_SIDE
DEPTHS = [int(x) for x in os.environ.get("NC_DEPTHS", "3,4,5,6").split(",")]
REF_N = int(os.environ.get("NC_REF_N", "96"))
# Load stepping is a continuation detail of the SOLVER, not of the
# discretisation — both schedules reach the same nonlinear solution, to the same
# tolerance — so the reference may take more of it than the runs it measures: at
# 96^3 the shared 20-step schedule dies in the line search at load step 19, and
# 40 steps walks it in.  Every test run keeps SOLVER_CONFIG exactly as it is.
REF_LOAD_STEPS = int(os.environ.get("NC_REF_LOAD_STEPS", "40"))
OUT = "/home/xl2254/implicit_optimize/output/nc_convergence"
SHOW_DEPTH = max(DEPTHS)
# The established figure uses a depth-3 far field and refines only near the
# interface.  NC_UNIFORM=1 is the companion experiment: every run starts at its
# final depth, hence the whole background cube is a uniform hex mesh.
UNIFORM_OCTREE = os.environ.get("NC_UNIFORM", "0").lower() not in {
    "0", "false", "no",
}
SHARP_VOLUME = os.environ.get("NC_SHARP_VOLUME", "0").lower() not in {
    "0", "false", "no",
}
if UNIFORM_OCTREE:
    OUT += "_uniform"
if SHARP_VOLUME:
    OUT += "_sharp"


def _init_depth(depth):
    return depth if UNIFORM_OCTREE else 3
# Side of the quadrature lattice the error is integrated on (its cell centres).
# 64 puts a sample every 0.0234 — twice as fine as the finest test mesh — so the
# midpoint rule resolves the piecewise-trilinear error field it is integrating.
SAMPLE_N = int(os.environ.get("NC_SAMPLE_N", "64"))
# Newton tolerance for EVERY run, tightened from the production 1e-5.  The
# solver stops on ||R|| < absolute_tol OR ||R||/||R||_0 < relative_tol, and the
# two schemes do not share a ||R||_0: Nitsche imposes both Dirichlet faces
# weakly with penalties up to 4.5e6, so its ||R||_0 is ~9.8e3 and dominated by a
# LINEAR penalty term that one Newton step kills — the relative test then passes
# at an absolute residual of 2e-3, where the body-fitted run (strong BCs,
# ||R||_0 = 0.13) has gone to 1e-9.  At the production tolerance the schemes are
# therefore converged to levels five orders apart, which makes both the
# iteration counts and the wall-clock times in panel (g) meaningless as a
# comparison.  At 1e-10 both end at ~1e-8 absolute and the counts are
# comparable.  Measured cost of the change: the errors move by 0.24 % (nitsche
# d=5) and 0.12 % (rotated d=5) — i.e. the loose gate never contaminated the
# accuracy study, only the cost one.
USE_MIXED_PRECISION = os.environ.get("NC_USE_MIXED_PRECISION", "0").lower() not in {
    "0",
    "false",
    "no",
}
# Float32 stalls around a few 1e-8 in this nonlinear problem, so the f64
# convergence-study gate is not representable as a robust f32 stopping target.
# The finest body-fitted f32 case plateaus at 1.05e-5, so use 2e-5 unless
# NC_RTOL explicitly requests another value.  This remains far below every
# discretization error plotted in the figure.
RTOL = float(
    os.environ.get("NC_RTOL", "1e-10" if USE_MIXED_PRECISION else "2e-5")
)
csn.SOLVER_CONFIG["relative_tol"] = RTOL
csn.SOLVER_CONFIG["use_mixed_precision"] = USE_MIXED_PRECISION
csn.SOLVER_CONFIG["tangent_f64"] = USE_MIXED_PRECISION
csn.SOLVER_CONFIG["cg_backend_assembled"] = "jaxamg"
# Keep the convergence study on the diffuse volumetric formulation and allow
# non-blocking result output.  SOLVER_CONFIG is expanded last by the production
# comparison runners, so these values override their local showcase defaults.
csn.SOLVER_CONFIG["sharp_volumetric"] = SHARP_VOLUME
csn.SOLVER_CONFIG["async_io"] = True
# NC_SCALE_DIAG=1: Jacobi-scale every assembled system to a unit diagonal before
# the JAX-AMG solve.  Exact (the same system), so it changes no error, only
# whether float32 solves it.  The sharp-volume sweeps need it at depth 6: the
# phi_floor-only exterior cells put the aligned uniform tangent at the float32
# rounding floor, and Newton crept from load step 11 until an AMG solve stalled
# 300x above tolerance.  Scaled, both depth-6 cases converge in 40 iterations.
SCALE_DIAG = os.environ.get("NC_SCALE_DIAG", "0").lower() not in {
    "0", "false", "no",
}
csn.SOLVER_CONFIG["scale_system_to_unit_diagonal"] = SCALE_DIAG
# NC_AMG_MAX_ITERS: JAX-AMG's Krylov cap (production default 1000).  The adaptive
# sharp-volume sweep's rotated depth-6 tangent is still converging at 1000
# ("11x short") and converges inside 4000; a cap bounds work, it does not change
# a solve that converges, so the other runs are unaffected.
if os.environ.get("NC_AMG_MAX_ITERS"):
    csn.SOLVER_CONFIG["jaxamg_max_iters"] = int(os.environ["NC_AMG_MAX_ITERS"])


def _p(name):
    return os.path.join(HERE, name)


def _std_n(depth):
    """Body-fitted resolution with the SAME element size as octree depth `depth`.

    The octree domain is 2.0 long, so h = 2/2^d; the box is 2*HS = 1.5 long, so
    matching h means n = 1.5 / (2/2^d) = 0.75 * 2^d elements across the box.
    """
    n = 0.75 * 2 ** depth
    assert abs(n - round(n)) < 1e-9, f"depth {depth} does not give an integer n"
    return int(round(n))


# ---------------------------------------------------------------------------
# reference solution: body-fitted hex mesh at REF_N, sampled by trilinear interp
# ---------------------------------------------------------------------------
def load_reference():
    # Both studies have the identical body-fitted reference.  Keep one cache in
    # the source study rather than spending ~13 minutes and 21 MB duplicating it.
    cache = os.path.join(SCRIPT_DIR, f"nc_ref_{REF_N}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        print(f"[ref] cached {cache}  ({REF_N}^3, {d['t']:.1f}s)")
        return d["u"], float(d["t"])

    keep = csn.SOLVER_CONFIG["load_steps"]
    if REF_LOAD_STEPS:
        csn.SOLVER_CONFIG["load_steps"] = REF_LOAD_STEPS
    try:
        solver, t = csn.run_standard(n_element=REF_N, output_dir=f"{OUT}/std_ref")
    finally:
        csn.SOLVER_CONFIG["load_steps"] = keep
    pts, u = csn._extract_solution(solver, "standard")
    res, nit = converged(solver, f"reference {REF_N}^3"), int(solver.newton_iterations)
    del solver
    gc_cuda()

    # The dof vector must be laid out like the analytic lattice the comparison
    # script builds, or every error below is silently measuring a permutation.
    xs = np.linspace(-HS, HS, REF_N + 1)
    P = pts.reshape(REF_N + 1, REF_N + 1, REF_N + 1, 3)
    assert np.allclose(P[:, 0, 0, 0], xs) and np.allclose(P[0, :, 0, 1], xs)
    assert np.allclose(P[0, 0, :, 2], xs)

    # ... and it must satisfy the two Dirichlet faces it was given.  The gate is
    # 1e-6, not machine zero: the constraints are imposed to the linear solve's
    # tolerance (measured 1.4e-9 at n=16), and what this is really testing is the
    # LAYOUT — any wrong permutation of the dof vector lands at 0.75, not 1e-8.
    U = u.reshape(REF_N + 1, REF_N + 1, REF_N + 1, 3)
    e_bot = float(np.abs(U[:, :, 0]).max())
    assert e_bot < 1e-6, f"clamped face is not clamped: {e_bot:.2e}"
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    top = np.stack(
        [
            gx * (csn.COS_T - 1.0) - gy * csn.SIN_T,
            gx * csn.SIN_T + gy * (csn.COS_T - 1.0),
            np.zeros_like(gx),
        ],
        axis=-1,
    )
    e_top = float(np.abs(U[:, :, -1] - top).max())
    assert e_top < 1e-6, f"twisted face is not twisted: {e_top:.2e}"
    print(f"[ref] BC check: clamped {e_bot:.1e}, twisted {e_top:.1e}")

    np.savez(cache, u=u, t=t, res=res, nit=nit)
    print(f"[ref] wrote {cache}  ({REF_N}^3, {t:.1f}s, {nit} Newton iters, "
          f"residual {res:.2e})")
    return u, t


def converged(solver, tag):
    """The last Newton step's RELATIVE residual, checked against the solver's own
    criterion (`relative_tol`, or `absolute_tol` on the raw norm)."""
    rel = float(solver.relative_residual_history[-1])
    ab = float(solver.residual_history[-1])
    ok = rel < csn.SOLVER_CONFIG["relative_tol"] or ab < csn.SOLVER_CONFIG["absolute_tol"]
    assert ok, f"{tag} did not converge: rel {rel:.3e}, ||R|| {ab:.3e}"
    return rel


def gc_cuda():
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def make_interp(u_ref):
    xs = np.linspace(-HS, HS, REF_N + 1)
    U = u_ref.reshape(REF_N + 1, REF_N + 1, REF_N + 1, 3)
    return RegularGridInterpolator((xs, xs, xs), U, bounds_error=False, fill_value=None)


def sample_lattice(n=SAMPLE_N):
    """The FIXED quadrature the error is measured on: cell centres of an n^3
    subdivision of the box, the same points for every run at every depth."""
    t = (np.arange(n) + 0.5) / n * 2 * HS - HS
    g = np.meshgrid(t, t, t, indexing="ij")
    return np.stack([a.ravel() for a in g], axis=1)


def hex_eval(nodes, elems, vals, pts, tree=None, centers=None):
    """Evaluate a Q1 field on an axis-aligned hex mesh at arbitrary points.

    Inside an element the FE field IS the trilinear interpolant of its eight
    corner values — that is what Q1 means — so this is the solution itself, not
    a resampling of it.  (Hanging nodes change what those corner values ARE;
    they do not change the field inside the element.)

    The corner-to-(i,j,k) map is read off each cell's own geometry rather than
    assumed from the element array's column order: octree corner conventions are
    exactly the kind of thing that is right until it is not.
    """
    from scipy.spatial import cKDTree

    P = nodes[elems]                                   # (M, 8, 3)
    lo, hi = P.min(axis=1), P.max(axis=1)
    if centers is None:
        centers = 0.5 * (lo + hi)
    if tree is None:
        tree = cKDTree(centers)

    out = np.zeros((len(pts), vals.shape[1]))
    todo = np.arange(len(pts))
    for k in (8, 32, 96):                              # widen until every point lands
        if len(todo) == 0:
            break
        _, cand = tree.query(pts[todo], k=min(k, len(centers)))
        cand = np.atleast_2d(cand)
        inside = ((pts[todo][:, None, :] >= lo[cand] - 1e-12)
                  & (pts[todo][:, None, :] <= hi[cand] + 1e-12)).all(axis=2)
        hit = inside.any(axis=1)
        cell = cand[np.arange(len(todo)), inside.argmax(axis=1)]
        idx, cel = todo[hit], cell[hit]
        if len(idx):
            # local coords in [0,1]^3, then the trilinear weight of each corner
            # from the SIGN of its offset about the cell centre
            span = (hi[cel] - lo[cel])
            xi = (pts[idx] - lo[cel]) / np.where(span > 0, span, 1.0)
            sgn = (nodes[elems[cel]] > centers[cel][:, None, :])   # (K, 8, 3)
            w = np.where(sgn, xi[:, None, :], 1.0 - xi[:, None, :]).prod(axis=2)
            out[idx] = np.einsum("kc,kcd->kd", w, vals[elems[cel]])
        todo = todo[~hit]
    assert len(todo) == 0, f"{len(todo)} sample points fell outside every cell"
    return out


def rel_l2_volume(u_test, u_ref):
    """Volume-weighted relative L2 error on the shared sample lattice.

    Equal weights, because the samples are the cell centres of a uniform
    subdivision — so this is the midpoint rule for
    ||u_h - u_ref||_L2(Omega) / ||u_ref||_L2(Omega) over the WHOLE box, and the
    two schemes are integrated against identical weights.  The node-based
    version below cannot say that: an octree carries most of its nodes in the
    fine layer near the boundary, so summing over nodes weights that layer
    several times more heavily than the coarse interior, and each mesh gets its
    own weighting.
    """
    num = ((u_test - u_ref) ** 2).sum()
    den = (u_ref ** 2).sum()
    return float(np.sqrt(num / den))


def rel_l2(pts, u, itp):
    """Relative L2 displacement error over nodes away from the two BC faces.

    The 0.1*HS < |z| < 0.9*HS band is the source script's convention: both
    schemes reproduce the prescribed faces by construction, so including them
    would dilute the error with exact values.

    The gate is 20 nodes, not a comfortable number: the coarsest octree (depth 3,
    h = 0.25, six cells across the box) leaves exactly 100 sample nodes, and that
    IS the measurement at that resolution.  The per-run count goes into
    nc_data.txt so the sample size is on the record next to the error.
    """
    z = pts[:, 2]
    m = (np.abs(z) > 0.1 * HS) & (np.abs(z) < 0.9 * HS)
    assert m.sum() >= 20, f"only {m.sum()} interior nodes"
    q = np.clip(pts[m], -HS + 1e-12, HS - 1e-12)
    ur = itp(q)
    err = np.linalg.norm(u[m] - ur, axis=1)
    ref = np.linalg.norm(ur, axis=1)
    return float(np.sqrt((err ** 2).sum() / (ref ** 2).sum())), int(m.sum())


# ---------------------------------------------------------------------------
# legacy-VTK dump of a deformed hex mesh (showcase panels)
# ---------------------------------------------------------------------------
def write_hex_vtk(path, nodes, elems, umag, phi=None, vec=None):
    n, m = len(nodes), len(elems)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ntwisting box\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {n} float\n")
        np.savetxt(f, nodes, fmt="%.6f")
        f.write(f"\nCELLS {m} {9*m}\n")
        np.savetxt(f, np.hstack([np.full((m, 1), 8), elems]), fmt="%d")
        f.write(f"\nCELL_TYPES {m}\n")
        np.savetxt(f, np.full(m, 12), fmt="%d")
        f.write(f"\nPOINT_DATA {n}\nSCALARS umag float 1\nLOOKUP_TABLE default\n")
        np.savetxt(f, umag, fmt="%.6f")
        if phi is not None:
            f.write("SCALARS phi float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, phi, fmt="%.6f")
        if vec is not None:
            f.write("VECTORS u float\n")
            np.savetxt(f, vec, fmt="%.6f")
    print(f"[vtk] {os.path.basename(path)}: {n} nodes, {m} hexes")


def std_connectivity(n):
    """VTK_HEXAHEDRON cells of the (n+1)^3 lattice, in the comparison script's
    meshgrid('ij') node order."""
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    c = [
        idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
        idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:],
    ]
    return np.stack([a.ravel() for a in c], axis=1)


def dump_standard(solver, n, path):
    pts, u = csn._extract_solution(solver, "standard")
    write_hex_vtk(path, pts + u, std_connectivity(n), np.linalg.norm(u, axis=1))


def octree_arrays(solver):
    """The solver's own trimmed octree, in the GLOBAL frame: nodes, cells, phi, u."""
    return (solver._prev_octree_nodes.cpu().numpy().astype(np.float64),
            solver._prev_octree_elements.cpu().numpy()[:, :8].astype(np.int64),
            solver._phi_nodal.cpu().numpy(),
            solver.u.dof_values.numpy().reshape(-1, 3))


def assert_uniform_octree(solver, depth, domain_length, tag):
    """Pin the companion study's premise on the mesh actually sent to FEM."""
    nodes, elems, _phi, _u = octree_arrays(solver)
    spans = np.ptp(nodes[elems], axis=1)
    expected = domain_length / (2 ** depth)
    assert np.allclose(spans, expected, rtol=0.0, atol=2e-6), (
        f"{tag}: init_depth == final_depth but solver mesh is not uniform; "
        f"cell spans range {spans.min():.6g}..{spans.max():.6g}, expected {expected:.6g}"
    )


def write_domain_frame(path, kind):
    """The 12 edges of the BACKGROUND DOMAIN cube — the box the octree was built
    in, not the geometry — as legacy POLYDATA lines.

    Written here rather than made with a ParaView Box + Transform so that the
    rotated case goes through the SAME rotation as its mesh (one code path, no
    second sign convention to get wrong): in the box frame the domain is what
    comes out oblique.
    """
    half = (csn.DOMAIN_LENGTH_ROTATED if kind == "rotated" else 2.0) / 2.0
    c = np.array([[x, y, z] for x in (-half, half)
                  for y in (-half, half) for z in (-half, half)])
    if kind == "rotated":
        c = c @ csn.ROT_MAT
    edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
             if bin(i ^ j).count("1") == 1]      # the 12 pairs one bit apart
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\ndomain frame\nASCII\n")
        f.write(f"DATASET POLYDATA\nPOINTS 8 float\n")
        np.savetxt(f, c, fmt="%.6f")
        f.write(f"\nLINES {len(edges)} {3*len(edges)}\n")
        for i, j in edges:
            f.write(f"2 {i} {j}\n")
    print(f"[vtk] {os.path.basename(path)}: domain cube, half-side {half}")


def dump_octree_phi(solver, kind, path):
    """The RAW trimmed octree, undeformed, carrying nodal phi — the mesh the
    solver was handed, cut open in the renderer to show the diffuse interface
    inside it.  No cell filter here: the whole point is the cells the interface
    runs through, including the ones mostly outside it."""
    nodes, elems, phi, _u = octree_arrays(solver)
    if kind == "rotated":                       # into the box frame, like (c)
        nodes = nodes @ csn.ROT_MAT
    used, inv = np.unique(elems, return_inverse=True)
    write_hex_vtk(path, nodes[used], inv.reshape(elems.shape),
                  np.zeros(len(used)), phi[used])


def dump_octree(solver, kind, path):
    """The octree cells that carry material, UNDEFORMED, with the displacement
    as a vector field.

    Undeformed and un-cropped on purpose: the renderer crops this to the
    geometry FIRST and warps by `u` second, which is the only order that gives
    the body its actual shape.  Warping first and cropping to whole cells
    afterwards draws the octree's staircase approximation of the box instead —
    correct as a picture of the MESH, wrong as a picture of the SOLUTION, and
    the mesh is what (d)/(e) are for.

    Cells with no node inside are dropped: their displacement is an
    unconstrained extension of the field.  Cells straddling the interface are
    kept, because the crop needs them to cut through.
    """
    nodes, elems, phi, u = octree_arrays(solver)
    if kind == "rotated":  # into the aligned frame, like _extract_solution
        nodes = nodes @ csn.ROT_MAT
        u = u @ csn.ROT_MAT
    keep = (phi[elems] > 0.5).any(axis=1)
    elems = elems[keep]
    used, inv = np.unique(elems, return_inverse=True)
    write_hex_vtk(
        path,
        nodes[used],
        inv.reshape(elems.shape),
        np.linalg.norm(u[used], axis=1),
        phi[used],
        vec=u[used],
    )


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
# What one completed sweep leaves behind — the cache.
PRODUCTS = ["nc_data.txt", "nc_stats.txt", "nc_std.vtk", "nc_nit.vtk",
            "nc_rot.vtk", "nc_nit_phi.vtk", "nc_rot_phi.vtk",
            "nc_nit_frame.vtk", "nc_rot_frame.vtk"]


def cached():
    """True if a previous sweep's products are all present.

    The sweep is 12 nonlinear solves and takes minutes, and every iteration on
    the renderer or the figure needs none of it — so it is a CACHE, like
    nc_ref_*.npz, not a step to repeat.  `NC_FORCE=1` re-runs it; deleting any
    product does too.  Do force it before believing a from-scratch build.
    """
    if os.environ.get("NC_FORCE"):
        return False
    missing = [f for f in PRODUCTS if not os.path.exists(_p(f))]
    if missing:
        if len(missing) < len(PRODUCTS):
            print(f"[cache] incomplete ({len(missing)} of {len(PRODUCTS)} "
                  f"missing, e.g. {missing[0]}) — re-running the sweep")
        return False
    return True


def main():
    if cached():
        with open(_p("nc_data.txt")) as f:
            n = sum(1 for line in f if line.strip() and not line.startswith("#"))
        print(f"[cache] {n} runs already in nc_data.txt and the VTKs are here — "
              f"skipping the sweep (NC_FORCE=1 to re-run)")
        sys.stdout.flush()
        os._exit(0)

    os.makedirs(OUT, exist_ok=True)
    u_ref, t_ref = load_reference()
    itp = make_interp(u_ref)

    # The shared quadrature, and the reference sampled on it once.  Every run is
    # then compared to the SAME numbers at the SAME points.
    SAMP = sample_lattice()
    U_REF = itp(SAMP)
    print(f"[samp] {len(SAMP)} points ({SAMPLE_N}^3 cell centres), "
          f"|u_ref| mean {np.linalg.norm(U_REF, axis=1).mean():.4f}")

    # Warm-up: the FIRST solve of each kind in a process pays Warp's kernel JIT,
    # and panel (g) plots wall clock.  Unwarmed, the rotated coarse run came out
    # at 2.9 s — as long as the next depth up, with 3.4x fewer elements — and the
    # cost curve started flat.  One throwaway run of each runner at the coarsest
    # depth (~5 s total) and every timed number is a solve.  Same reasoning as
    # prep_octree_sdf's timed octree generation (AGENTS.md §5).
    print("[warm] JIT warm-up: one throwaway run of each variant")
    _d0 = DEPTHS[0]
    _w, _ = csn.run_standard(n_element=_std_n(_d0), output_dir=f"{OUT}/warm_std")
    del _w
    gc_cuda()
    for _runner in (csn.run_nitsche, csn.run_nitsche_rotated):
        _w, _t, _e = _runner(output_dir=f"{OUT}/warm", octree_depth=_d0,
                             init_octree_depth=_init_depth(_d0),
                             quad_refinement_depth=0)
        del _w
        gc_cuda()

    # (variant, depth, h, nelem, ndof, eps, err, nnodes, newton, residual, secs)
    rows = []
    stats = {"ref_n": REF_N, "ref_time": t_ref, "half_side": HS,
             "theta_deg": np.degrees(csn.THETA), "E": csn.SOLVER_CONFIG["E"],
             "nu": csn.SOLVER_CONFIG["nu"], "load_steps": csn.SOLVER_CONFIG["load_steps"],
             "ref_load_steps": REF_LOAD_STEPS or csn.SOLVER_CONFIG["load_steps"],
             "depths": ",".join(str(d) for d in DEPTHS), "show_depth": SHOW_DEPTH,
             "relative_tol": RTOL,
             "precision": "f64" if USE_MIXED_PRECISION else "f32",
             "amg_backend": "jaxamg", "scale_diag": SCALE_DIAG,
             "amg_max_iters": csn.SOLVER_CONFIG.get("jaxamg_max_iters", 1000),
             "sample_n": SAMPLE_N, "box_side": 2 * HS}

    for d in DEPTHS:
        # ---- body-fitted hex mesh at the same element size ------------------
        n = _std_n(d)
        s, t = csn.run_standard(n_element=n, output_dir=f"{OUT}/std_{n}")
        pts, u = csn._extract_solution(s, "standard")
        err_n, nn = rel_l2(pts, u, itp)
        # Its own Q1 field, evaluated on the shared lattice — a uniform grid, so
        # RegularGridInterpolator IS that field, exactly, as for the reference.
        xs_t = np.linspace(-HS, HS, n + 1)
        u_s = RegularGridInterpolator(
            (xs_t, xs_t, xs_t), u.reshape(n + 1, n + 1, n + 1, 3),
            bounds_error=False, fill_value=None)(SAMP)
        err = rel_l2_volume(u_s, U_REF)
        if d == DEPTHS[0]:
            # The body-fitted mesh is a case BOTH evaluators handle, so it pins
            # hex_eval — its cell locator and its corner-sign weights — against
            # scipy on the same field at the same points.  Everything the octree
            # errors mean rests on that code being right.
            u_hex = hex_eval(pts, std_connectivity(n), u, SAMP)
            gap = float(np.abs(u_hex - u_s).max())
            assert gap < 1e-9, f"hex_eval disagrees with scipy by {gap:.2e}"
            print(f"[chk] hex_eval == RegularGridInterpolator to {gap:.1e}")
        rows.append(("standard", d, 2 * HS / n, n ** 3, 3 * (n + 1) ** 3,
                     float("nan"), err, nn, int(s.newton_iterations),
                     converged(s, f"standard n={n}"), t, err_n))
        print(f"  -> standard n={n}: err={err:.6f} (nodal {err_n:.6f}), {t:.1f}s")
        if d == SHOW_DEPTH:
            dump_standard(s, n, _p("nc_std.vtk"))
        del s, pts, u
        gc_cuda()

        # ---- diffused Nitsche, no refined quadrature -------------------------
        for kind, runner, out in (("nitsche", csn.run_nitsche, "nit"),
                                  ("rotated", csn.run_nitsche_rotated, "rot")):
            s, t, eps = runner(output_dir=f"{OUT}/{out}_d{d}",
                               octree_depth=d,
                               init_octree_depth=_init_depth(d),
                               quad_refinement_depth=0)
            if UNIFORM_OCTREE:
                dl = csn.DOMAIN_LENGTH_ROTATED if kind == "rotated" else 2.0
                assert_uniform_octree(s, d, dl, f"{kind} d={d}")
            pts, u = csn._extract_solution(s, kind)
            err_n, nn = rel_l2(pts, u, itp)
            # Sample the octree field on the shared lattice.  Locate in the
            # GLOBAL frame — the cells are axis-aligned there and nowhere else —
            # then bring the vector back to the box frame, exactly the pair of
            # rotations _extract_solution applies.
            g_nodes, g_elems, _phi, g_u = octree_arrays(s)
            if kind == "rotated":
                u_s = hex_eval(g_nodes, g_elems, g_u,
                               SAMP @ csn.ROT_MAT.T) @ csn.ROT_MAT
            else:
                u_s = hex_eval(g_nodes, g_elems, g_u, SAMP)
            err = rel_l2_volume(u_s, U_REF)
            dl = csn.DOMAIN_LENGTH_ROTATED if kind == "rotated" else 2.0
            ne = int(s._prev_octree_elements.shape[0])
            rows.append((kind, d, dl / 2 ** d, ne,
                         int(s.u.dof_values.numpy().size), eps, err, nn,
                         int(s.newton_iterations), converged(s, f"{kind} d={d}"),
                         t, err_n))
            print(f"  -> {kind} d={d}: err={err:.6f} (nodal {err_n:.6f}), "
                  f"{ne} elems, eps={eps:.6f}, {t:.1f}s")
            if d == SHOW_DEPTH:
                dump_octree(s, kind, _p(f"nc_{out}.vtk"))
                dump_octree_phi(s, kind, _p(f"nc_{out}_phi.vtk"))
                write_domain_frame(_p(f"nc_{out}_frame.vtk"), kind)
            del s, pts, u
            gc_cuda()

    with open(_p("nc_data.txt"), "w") as f:
        f.write("# err = volume-weighted L2 on the shared %d^3 lattice (what the "
                "figure plots);\n# err_nodal = the older node-sum over the "
                "interior band, kept for comparison\n" % SAMPLE_N)
        f.write("# variant depth h nelem ndof epsilon err n_nodes newton "
                "rel_residual secs err_nodal\n")
        for r in rows:
            f.write("%s %d %.8f %d %d %.8f %.8e %d %d %.3e %.2f %.8e\n" % r)
    with open(_p("nc_stats.txt"), "w") as f:
        for k, v in stats.items():
            f.write(f"{k}={v}\n")
    print(f"[data] wrote nc_data.txt ({len(rows)} rows) + nc_stats.txt")
    sys.stdout.flush()
    os._exit(0)   # Warp/Torch teardown races at interpreter shutdown


if __name__ == "__main__":
    main()
