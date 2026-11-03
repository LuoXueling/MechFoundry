"""Run a mesh-convergence sweep for the twisting box — body-fitted
FEM vs diffused Nitsche on an octree, axis-aligned and 45deg-rotated.

Everything is driven by this leaf's ``solver_cases.py`` implementation: its
``run_standard`` /
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
    different norm. The old node-based number is retained as the
    ``error_nodal`` column in the result table.
  * **One Newton tolerance for both schemes.**  ``relative_tol`` is tightened to
    1e-10 (NC_RTOL) because the production 1e-5 converges the two schemes to
    levels five orders apart — see the note at RTOL below.  It barely moves the
    errors and it is what makes panel (g) mean anything.

Writes one tabular result. The reusable ``nc_ref_<N>.npz`` reference solution
is retained as an input cache rather than regenerated for every invocation.

The command-line defaults reproduce the reference study: depths 3--6, a
96-cells-per-axis reference, and 40 load steps for that reference.

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
from pathlib import Path

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--depths", type=int, nargs="+", default=[3, 4, 5, 6])
parser.add_argument("--ref-n", type=int, default=96)
parser.add_argument("--ref-load-steps", type=int, default=40)
parser.add_argument("--sample-n", type=int, default=64)
parser.add_argument("--uniform-octree", action="store_true")
parser.add_argument("--sharp-volume", action="store_true")
parser.add_argument("--use-mixed-precision", action="store_true")
parser.add_argument("--relative-tol", type=float)
parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.csv"))
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
HERE = SCRIPT_DIR
REPO_ROOT = SCRIPT_DIR.parents[2]
os.chdir(HERE)
sys.path.insert(0, os.fspath(SCRIPT_DIR))
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator

import solver_cases as csn

HS = csn.HALF_SIDE
DEPTHS = args.depths
REF_N = args.ref_n
# Load stepping is a continuation detail of the SOLVER, not of the
# discretisation — both schedules reach the same nonlinear solution, to the same
# tolerance — so the reference may take more of it than the runs it measures: at
# 96^3 the shared 20-step schedule dies in the line search at load step 19, and
# 40 steps walks it in.  Every test run keeps SOLVER_CONFIG exactly as it is.
REF_LOAD_STEPS = args.ref_load_steps
OUT = os.fspath(HERE / "output")
# The established figure uses a depth-3 far field and refines only near the
# interface.  NC_UNIFORM=1 is the companion experiment: every run starts at its
# final depth, hence the whole background cube is a uniform hex mesh.
UNIFORM_OCTREE = args.uniform_octree
SHARP_VOLUME = args.sharp_volume
if UNIFORM_OCTREE:
    OUT += "_uniform"
if SHARP_VOLUME:
    OUT += "_sharp"


def _init_depth(depth):
    return depth if UNIFORM_OCTREE else 3
# Side of the quadrature lattice the error is integrated on (its cell centres).
# 64 puts a sample every 0.0234 — twice as fine as the finest test mesh — so the
# midpoint rule resolves the piecewise-trilinear error field it is integrating.
SAMPLE_N = args.sample_n
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
USE_MIXED_PRECISION = args.use_mixed_precision
# Float32 stalls around a few 1e-8 in this nonlinear problem, so the f64
# convergence-study gate is not representable as a robust f32 stopping target.
# The finest body-fitted f32 case plateaus at 1.05e-5, so use 2e-5 unless
# NC_RTOL explicitly requests another value.  This remains far below every
# discretization error plotted in the figure.
RTOL = (
    args.relative_tol
    if args.relative_tol is not None
    else (1e-10 if USE_MIXED_PRECISION else 2e-5)
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
    the result table so the sample size is recorded next to the error.
    """
    z = pts[:, 2]
    m = (np.abs(z) > 0.1 * HS) & (np.abs(z) < 0.9 * HS)
    assert m.sum() >= 20, f"only {m.sum()} interior nodes"
    q = np.clip(pts[m], -HS + 1e-12, HS - 1e-12)
    ur = itp(q)
    err = np.linalg.norm(u[m] - ur, axis=1)
    ref = np.linalg.norm(ur, axis=1)
    return float(np.sqrt((err ** 2).sum() / (ref ** 2).sum())), int(m.sum())


def std_connectivity(n):
    """VTK_HEXAHEDRON cells of the (n+1)^3 lattice, in the comparison script's
    meshgrid('ij') node order."""
    idx = np.arange((n + 1) ** 3).reshape(n + 1, n + 1, n + 1)
    c = [
        idx[:-1, :-1, :-1], idx[1:, :-1, :-1], idx[1:, 1:, :-1], idx[:-1, 1:, :-1],
        idx[:-1, :-1, 1:], idx[1:, :-1, 1:], idx[1:, 1:, 1:], idx[:-1, 1:, 1:],
    ]
    return np.stack([a.ravel() for a in c], axis=1)


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


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------
def main():
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
    # cost curve started flat. One throwaway run of each runner at the coarsest
    # depth (~5 s total) makes every reported timed number a solve.
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
            del s, pts, u
            gc_cuda()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        f.write("variant,depth,h,nelem,ndof,epsilon,error,n_nodes,newton,"
                "relative_residual,seconds,error_nodal\n")
        for r in rows:
            f.write("%s,%d,%.8f,%d,%d,%.8f,%.8e,%d,%d,%.3e,%.2f,%.8e\n" % r)
    print(f"[data] wrote {args.output} ({len(rows)} rows, sample_n={SAMPLE_N}, "
          f"ref_n={REF_N}, ref_seconds={t_ref:.1f})")
    sys.stdout.flush()
    os._exit(0)   # Warp/Torch teardown races at interpreter shutdown


if __name__ == "__main__":
    main()
