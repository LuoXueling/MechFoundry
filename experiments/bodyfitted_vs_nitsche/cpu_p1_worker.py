"""The P1 clamp-band solve on the CPU, for the meshes the GPU path cannot hold.

The GPU job (``compare_bodyfitted_nitsche.py --phase solve``) prepares the
problem: the mesh, the clamp band and the load, with exactly the GPU path's
discretization (``cpu_load_and_clamp``).  It writes ``cpu_jobs/<tag>.npz``
and ``cpu_jobs/<tag>.json`` (the entry's record without the solve), then moves on.
This worker, run by the driver's CPU queue without a CUDA context, solves it
and writes what the GPU path would have written: ``p1_solution_<tag>.npz`` and
``result_<tag>.json`` (``solve.solver = "cpu"``; its times are CPU times).

It imports numpy, scipy, pyamg and (if present) pypardiso only, so it can run in
a small CPU env of its own.

Method: modified Newton.  The u = 0 tangent of the neo-Hookean model is linear
elasticity (lambda = kappa - 2 mu / 3).  It is factored once, by MKL PARDISO
(SPD, nested dissection), or, if PARDISO is missing or fails, by pyamg's
smoothed-aggregation AMG with CG (3x3 blocks, rigid-body near-null space).  The
full neo-Hookean residual drives the steps; the strains are ~1e-4, so each step
cuts it by about that much.

Validation (pyamg path, on stored GPU solutions): elephant d8 surface (1.0 M
DOFs) rel L2 1.6e-8, arch d7 surface 4.1e-10.  The float64 residual floor is ~1e-8.

    python cpu_p1_worker.py OUT_DIR TAG
"""
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CHUNK = 500_000  # tets per vectorised chunk (~1 GB of temporaries)


def tet_gradients(X, T):
    """Volumes and P1 shape-function gradients (m, 4, 3) of every tet."""
    p = X[T]
    D = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)
    vol = np.abs(np.linalg.det(D)) / 6.0
    Dinv = np.linalg.inv(D)  # rows: gradients of the barycentrics 1..3
    return vol, np.concatenate([-Dinv.sum(axis=1, keepdims=True), Dinv], axis=1)


def nodal_sum(n, T, values):
    """Sum per-tet-vertex values (m, 4, 3) into (n, 3), via bincount."""
    idx = T.reshape(-1)
    v = values.reshape(-1, 3)
    return np.stack([np.bincount(idx, weights=v[:, c], minlength=n) for c in range(3)], axis=1)


def internal_force(X, T, vol, G, u, mu, kappa):
    """The neo-Hookean internal force of ``NeoHookean`` (same PK1), per vertex.

    P1 gradients are constant per tet, so the one-point rule is exact, as is the
    order-2 rule the GPU path uses."""
    out = np.zeros((len(X), 3))
    for s in range(0, len(T), CHUNK):
        Tc, g = T[s:s + CHUNK], G[s:s + CHUNK]
        F = np.eye(3)[None] + np.einsum("mac,mad->mcd", u[Tc], g)
        J = np.linalg.det(F)
        F_inv_T = np.transpose(np.linalg.inv(F), (0, 2, 1))
        I1 = (F * F).sum(axis=(1, 2))
        iso = mu * J ** (-2.0 / 3.0)
        P = iso[:, None, None] * F + (kappa * np.log(J) - iso * I1 / 3.0)[:, None, None] * F_inv_T
        out += nodal_sum(len(X), Tc, np.einsum("mcd,mad->mac", P, g) * vol[s:s + CHUNK, None, None])
    return out


def linear_stiffness(X, T, vol, G, mu, kappa):
    """The neo-Hookean tangent at u = 0 (linear elasticity), CSR."""
    lam = kappa - 2.0 * mu / 3.0
    Dm = np.zeros((6, 6))
    Dm[:3, :3] = lam
    Dm[np.arange(3), np.arange(3)] += 2 * mu
    Dm[3:, 3:] = np.eye(3) * mu
    n = len(X)
    K = None
    for s in range(0, len(T), CHUNK):
        g, Tc = G[s:s + CHUNK], T[s:s + CHUNK]
        m = len(g)
        B = np.zeros((m, 6, 12))
        for i in range(4):
            gx, gy, gz = g[:, i, 0], g[:, i, 1], g[:, i, 2]
            B[:, 0, 3 * i], B[:, 1, 3 * i + 1], B[:, 2, 3 * i + 2] = gx, gy, gz
            B[:, 3, 3 * i], B[:, 3, 3 * i + 1] = gy, gx
            B[:, 4, 3 * i + 1], B[:, 4, 3 * i + 2] = gz, gy
            B[:, 5, 3 * i], B[:, 5, 3 * i + 2] = gz, gx
        Ke = np.matmul(np.transpose(B, (0, 2, 1)), np.einsum("ab,mbj->maj", Dm, B)) * vol[s:s + m, None, None]
        dofs = (3 * Tc[:, :, None].astype(np.int32) + np.arange(3, dtype=np.int32)).reshape(m, 12)
        part = sp.csr_matrix((Ke.ravel(), (np.repeat(dofs, 12, axis=1).ravel(), np.tile(dofs, (1, 12)).ravel())),
                             shape=(3 * n, 3 * n))
        K = part if K is None else K + part
        del B, Ke, part
    return K


def material_tangent(F, mu, kappa):
    """dP/dF (m, 3, 3, 3, 3) of the ``NeoHookean`` PK1
    P = mu J^-2/3 (F - I1/3 F^-T) + kappa ln J F^-T, with a = F^-T:
    C_ijkl = mu J^-2/3 [d_ik d_jl - 2/3 a_kl (F_ij - I1/3 a_ij) - 2/3 F_kl a_ij
             + I1/3 a_il a_kj] + kappa [a_ij a_kl - ln J a_il a_kj].
    At F = I it is linear elasticity with lambda = kappa - 2 mu / 3."""
    J = np.linalg.det(F)
    a = np.transpose(np.linalg.inv(F), (0, 2, 1))
    I1 = (F * F).sum(axis=(1, 2))
    iso = mu * J ** (-2.0 / 3.0)
    eye = np.eye(3)
    dev = F - (I1 / 3.0)[:, None, None] * a
    a_il_a_kj = a[:, :, None, None, :] * np.transpose(a, (0, 2, 1))[:, None, :, :, None]
    C = iso[:, None, None, None, None] * (
        np.einsum("ik,jl->ijkl", eye, eye)[None]
        - (2.0 / 3.0) * np.einsum("mkl,mij->mijkl", a, dev)
        - (2.0 / 3.0) * np.einsum("mkl,mij->mijkl", F, a)
        + (I1 / 3.0)[:, None, None, None, None] * a_il_a_kj)
    C += kappa * np.einsum("mij,mkl->mijkl", a, a) - (kappa * np.log(J))[:, None, None, None, None] * a_il_a_kj
    return C


def tangent_stiffness(X, T, vol, G, u, mu, kappa):
    """The neo-Hookean tangent at u (CSR), assembled in chunks."""
    n = len(X)
    K = None
    for s in range(0, len(T), CHUNK // 2):
        Tc, g = T[s:s + CHUNK // 2], G[s:s + CHUNK // 2]
        m = len(g)
        F = np.eye(3)[None] + np.einsum("mac,mad->mcd", u[Tc], g)
        C = material_tangent(F, mu, kappa)
        Ke = np.einsum("maikl,mbl->maibk", np.einsum("maj,mijkl->maikl", g, C, optimize=True), g,
                       optimize=True) * vol[s:s + m, None, None, None, None]
        Ke = Ke.reshape(m, 12, 12)
        dofs = (3 * Tc[:, :, None].astype(np.int32) + np.arange(3, dtype=np.int32)).reshape(m, 12)
        part = sp.csr_matrix((Ke.ravel(), (np.repeat(dofs, 12, axis=1).ravel(), np.tile(dofs, (1, 12)).ravel())),
                             shape=(3 * n, 3 * n))
        K = part if K is None else K + part
        del C, Ke, part
    return K


def newton_full(X, T, f, fixed, u0, mu, kappa, rtol, linsolve, load_steps=5, max_iter=30):
    """Full Newton with load steps, as the GPU path runs it (``load_steps``
    equal increments; each step to ``rtol`` relative to its first residual,
    the last step's history is the reported one).  ``linsolve(A, b)`` solves
    with the Dirichlet-eliminated tangent A (CSR)."""
    t0 = time.perf_counter()
    X, T = np.asarray(X, np.float64), np.asarray(T, np.int64)
    n = len(X)
    vol, G = tet_gradients(X, T)
    free = (~np.asarray(fixed, bool)).reshape(-1).astype(np.float64)
    Df, Dc = sp.diags(free), sp.diags(1.0 - free)
    u = np.array(u0, np.float64)
    steps, stamps, t_asm, t_lin, alphas = [], [], 0.0, 0.0, []
    for step in range(1, load_steps + 1):
        scale = step / load_steps
        hist, pending = [], None
        for it in range(max_iter + 1):
            if pending is not None:  # the line search's accepted trial
                R, norm = pending
            else:
                R = (internal_force(X, T, vol, G, u, mu, kappa) - scale * f).reshape(-1) * free
                norm = float(np.linalg.norm(R))
            if not np.isfinite(norm):
                raise RuntimeError(f"Newton diverged (non-finite residual) in load step {step}")
            hist.append(norm)
            stamps.append(time.perf_counter() - t0)
            print(f"[full-newton] scale={scale:.3f} Iter {it}: ||R|| = {norm:.6e}, rel = {norm / hist[0]:.6e}",
                  flush=True)
            if (it > 0 and norm / hist[0] <= rtol) or it == max_iter:
                break
            ta = time.perf_counter()
            A = (Df @ tangent_stiffness(X, T, vol, G, u, mu, kappa) @ Df + Dc).tocsr()
            tb = time.perf_counter()
            du = (linsolve(A, -R) * free).reshape(n, 3)
            tc = time.perf_counter()
            t_asm, t_lin = t_asm + tb - ta, t_lin + tc - tb
            del A
            # Step length: alpha = 1, 1/2, ... 1/64, the first acceptable one.
            alpha = 1.0
            for _ in range(7):
                with np.errstate(invalid="ignore", divide="ignore"):
                    trial = (internal_force(X, T, vol, G, u + alpha * du, mu, kappa)
                             - scale * f).reshape(-1) * free
                tn = float(np.linalg.norm(trial))
                # Plain Newton, as the GPU path runs it (teddy bear d8 dofs: rel 287
                # after the first iteration, then quadratic); back off only from
                # a non-finite residual (J <= 0) or a 1e4x growth.
                if np.isfinite(tn) and tn < 1e4 * norm:
                    break
                alpha *= 0.5
            if alpha < 1.0:
                alphas.append((step, it, alpha))
            pending = (trial, tn) if alpha >= 1.0 / 64 and np.isfinite(tn) else None
            u = u + alpha * du
        steps.append([h / hist[0] for h in hist])
        if hist[-1] / hist[0] > rtol:
            raise RuntimeError(f"Newton did not reach rel {rtol:g} in load step {step}: {steps[-1]}")
    last = steps[-1]
    return u, {"method": f"full Newton, {load_steps} load steps, backtracking only from non-finite or 1e4x residuals",
               "load_step_histories": steps, "line_search_steps": alphas,
               "relative_residuals": last, "newton_iterations": sum(len(s) - 1 for s in steps),
               "final_relative_residual": last[-1], "dofs": 3 * n, "iteration_seconds": stamps,
               "seconds": {"tangent_assembly": t_asm, "linear_solves": t_lin, "total": time.perf_counter() - t0}}


# In-core PARDISO memory budget.  The hamburger d8 surface (6.7 M DOFs) passed
# 196 GB of 251 GB before it was stopped; the elephant (1.0 M) peaked at 16 GB,
# the Pantheon (2.2 M) at 33 GB.  Out-of-core does not fit the free disk.
PARDISO_BUDGET_GB = float(os.environ.get("CPU_P1_PARDISO_GB", "100"))


class PardisoTooLarge(RuntimeError):
    pass


class Pardiso:
    """MKL PARDISO, real SPD (mtype 2), upper triangle; factored once.

    The analysis (phase 11) predicts the in-core memory (iparm 15-17, KB); a
    factorization over ``PARDISO_BUDGET_GB`` is not attempted."""

    def __init__(self, A, mtype=2):
        import pypardiso

        self.A = (sp.triu(A, format="csr") if mtype in (2, -2) else A.tocsr()).copy()
        self.A.sort_indices()
        self.ps = pypardiso.PyPardisoSolver(mtype=mtype)
        self.ps._check_A(self.A)
        b = np.zeros((self.A.shape[0], 1))
        self.ps.set_phase(11)
        self.ps._call_pardiso(self.A, b)
        ip = self.ps.iparm
        self.predicted_gb = max(int(ip[14]), int(ip[15]) + int(ip[16])) / 1e6
        self.factor_nnz = int(ip[17])
        print(f"[cpu-p1] PARDISO analysis: predicted {self.predicted_gb:.1f} GB, factor nnz {self.factor_nnz:.3e}",
              flush=True)
        if self.predicted_gb > PARDISO_BUDGET_GB:
            self.ps.free_memory(everything=True)
            raise PardisoTooLarge(f"predicted {self.predicted_gb:.1f} GB > budget {PARDISO_BUDGET_GB:g} GB")
        # Memory guard: factor only while the machine keeps 40 GB beyond the
        # prediction (GPU jobs and other CPU jobs share the RAM); wait <= 1 h.
        for _ in range(360):
            if mem_available_gb() - self.predicted_gb >= 40.0:
                break
            time.sleep(10)
        else:
            self.ps.free_memory(everything=True)
            raise PardisoTooLarge(f"predicted {self.predicted_gb:.1f} GB; only {mem_available_gb():.0f} GB "
                                  "available for an hour")
        self.ps.set_phase(22)
        self.ps._call_pardiso(self.A, b)

    def solve(self, b):
        self.ps.set_phase(33)
        return self.ps._call_pardiso(self.A, np.asfortranarray(b[:, None])).ravel()

    def free(self):
        """Release MKL's factor (phase -1); without it every refactorization
        of full Newton kept its factor (the loaf d7 test reached 44 GB)."""
        self.ps.free_memory(everything=True)


def pardiso_solve_once(A, b):
    lin = Pardiso(A)
    try:
        return lin.solve(b)
    finally:
        lin.free()


def mem_available_gb():
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    return 0.0


class AMG:
    def __init__(self, A, X, free, tol):
        import pyamg

        n = len(X)
        xc = X - X.mean(axis=0)
        null = np.zeros((n, 3, 6))
        null[:, 0, 0] = null[:, 1, 1] = null[:, 2, 2] = 1.0
        null[:, 0, 4], null[:, 0, 5] = xc[:, 2], -xc[:, 1]
        null[:, 1, 3], null[:, 1, 5] = -xc[:, 2], xc[:, 0]
        null[:, 2, 3], null[:, 2, 4] = xc[:, 1], -xc[:, 0]
        self.ml = pyamg.smoothed_aggregation_solver(A.tobsr(blocksize=(3, 3)), B=null.reshape(3 * n, 6) * free[:, None],
                                                    max_coarse=3000)
        self.tol, self.iterations = tol, []

    def solve(self, b):
        res, tc = [], time.perf_counter()

        def progress(_x):
            if len(res) % 50 == 0:
                print(f"[cpu-p1]   CG {len(res)}: rel {res[-1] / res[0]:.2e}, "
                      f"{(time.perf_counter() - tc) / len(res):.2f} s/it", flush=True)

        x = self.ml.solve(b, tol=self.tol, accel="cg", maxiter=3000, residuals=res, callback=progress)
        self.iterations.append(len(res) - 1)
        return x


class Nonlinear(RuntimeError):
    """Modified Newton does not contract: the entry needs full Newton."""


def solve(X, T, f, fixed, u0, mu, kappa, rtol, max_iter=40, linear="pardiso", cg_tol=1e-4, depth=6,
          load_steps=5, refactor_limit=900.0, full_newton=False):
    """Modified Newton to ``rtol`` (relative to the first free residual, after at
    least one step).  ``fixed`` (n, 3) marks the clamped components and ``u0``
    carries their values.  Returns u and a stats dict.

    The fixed u = 0 tangent contracts the residual by about the strain per step:
    ~1e-4 on most entries (2-3 steps), but only 0.66 on the hamburger d8
    surface, where even the GPU's full Newton leaves rel 0.3 after the first
    iteration of each load step.  So once a step contracts by less than 20x,
    the steps are Anderson-accelerated (type II, ``depth`` past steps, reset on
    any increase); the inner AMG-CG then only needs ``cg_tol`` (1e-4), since
    every outer step re-evaluates the full float64 residual.

    When even the first step contracts by less than 20x (the loaf of bread:
    0.66; Anderson then stalled between 1e-5 and 1e-3 on the loaf d7 surface),
    the entry is nonlinear: full Newton with load steps (``newton_full``) with a
    PARDISO refactorization per iteration if one factorization took under
    ``refactor_limit`` seconds, else ``Nonlinear`` is raised for the GPU
    job's lean full Newton (``lean_gpu_p1`` in the compare script)."""
    if full_newton:  # a previous run found the entry nonlinear
        u_full, st = newton_full(X, T, f, fixed, u0, mu, kappa, rtol, pardiso_solve_once, load_steps=load_steps)
        st.update(solver="cpu", linear_solver="pardiso (refactored per iteration)",
                  linear_solver_note="full Newton directly (an earlier modified Newton did not contract)")
        st["seconds"].update(assembly=0.0, linear_setup=0.0)
        return u_full, st
    t0 = time.perf_counter()
    X, T = np.asarray(X, np.float64), np.asarray(T, np.int64)
    n = len(X)
    vol, G = tet_gradients(X, T)
    K = linear_stiffness(X, T, vol, G, mu, kappa)
    free = (~np.asarray(fixed, bool)).reshape(-1).astype(np.float64)
    Df = sp.diags(free)
    A = (Df @ K @ Df + sp.diags(1.0 - free)).tocsr()
    del K
    t1 = time.perf_counter()
    method, note = linear, None
    if linear == "pardiso":
        try:
            lin = Pardiso(A)
        except Exception as exc:  # noqa: BLE001 - missing pypardiso or a failed factorization
            note = f"PARDISO failed ({type(exc).__name__}: {exc}); pyamg"
            print(f"[cpu-p1] {note}", flush=True)
            method, lin = "amg", None
    if method == "amg":
        lin = AMG(A, X, free, cg_tol)
    del A
    t2 = time.perf_counter()
    print(f"[cpu-p1] {3 * n} dofs, assembly {t1 - t0:.1f} s, {method} setup {t2 - t1:.1f} s", flush=True)
    u = np.array(u0, np.float64).reshape(-1)
    history, stamps, r0 = [], [], None
    dX, dF, prev, accel_steps = [], [], None, 0
    best = (np.inf, u.copy())
    for it in range(max_iter + 1):
        with np.errstate(invalid="ignore", divide="ignore"):
            R = (internal_force(X, T, vol, G, u.reshape(n, 3), mu, kappa) - f).reshape(-1) * free
        norm = float(np.linalg.norm(R))
        if not np.isfinite(norm):
            if not np.isfinite(best[0]):
                raise RuntimeError("CPU P1: non-finite residual at the start")
            # an accelerated step left the physical range (J <= 0): back to the
            # best iterate, plain steps from there
            print(f"[cpu-p1] Iter {it}: non-finite residual; back to the best iterate", flush=True)
            u, dX, dF, prev = best[1].copy(), [], [], None
            continue
        r0 = r0 if r0 is not None else max(norm, 1e-300)
        history.append(norm / r0)
        stamps.append(time.perf_counter() - t2)
        print(f"[cpu-p1] Iter {it}: ||R|| = {norm:.6e}, rel = {norm / r0:.6e}", flush=True)
        if (len(history) > 1 and norm / r0 <= rtol) or it == max_iter:
            break
        if norm > best[0]:
            dX, dF = [], []  # an increase: restart the acceleration history
        if norm < best[0]:
            best = (norm, u.copy())
        if len(history) == 2 and history[1] > 0.05:
            if method == "pardiso" and t2 - t1 <= refactor_limit:
                print(f"[cpu-p1] first step contracted only to {history[1]:.2e}: full Newton, PARDISO", flush=True)
                lin.free()
                del lin
                u_full, st = newton_full(X, T, f, fixed, u0, mu, kappa, rtol, pardiso_solve_once,
                                         load_steps=load_steps)
                st.update(solver="cpu", linear_solver="pardiso (refactored per iteration)",
                          linear_solver_note=f"modified Newton contracted only to {history[1]:.2e}",
                          modified_newton_residuals=history)
                st["seconds"].update(assembly=t1 - t0, linear_setup=t2 - t1)
                return u_full, st
            raise Nonlinear(f"modified Newton contracted only to {history[1]:.2e} in its first step "
                            f"({method}, setup {t2 - t1:.0f} s): needs full Newton (lean GPU path)")
        d = lin.solve(-R) * free  # the modified-Newton step: the fixed-point residual
        slow = len(history) > 1 and history[-1] > history[-2] / 20.0
        if prev is not None:
            dX.append(u - prev[0])
            dF.append(d - prev[1])
            dX, dF = dX[-depth:], dF[-depth:]
        prev = (u.copy(), d.copy())
        if slow and dF:
            Fm = np.stack(dF, axis=1)
            gamma, *_ = np.linalg.lstsq(Fm, d, rcond=None)
            u = u + d - (np.stack(dX, axis=1) + Fm) @ gamma
            accel_steps += 1
        else:
            u = u + d
    u = u.reshape(n, 3)
    t3 = time.perf_counter()
    stats = {"solver": "cpu", "linear_solver": method,
             "method": f"modified Newton, u=0 tangent, {'MKL PARDISO' if method == 'pardiso' else 'pyamg SA + CG'}",
             "linear_solver_note": note, "relative_residuals": history, "newton_iterations": len(history) - 1,
             "anderson_steps": accel_steps, "inner_cg_tol": cg_tol if method == "amg" else None,
             "cg_iterations": getattr(lin, "iterations", None), "iteration_seconds": stamps,
             "final_relative_residual": history[-1], "dofs": 3 * n,
             "seconds": {"assembly": t1 - t0, "linear_setup": t2 - t1, "newton": t3 - t2, "total": t3 - t0}}
    if history[-1] > rtol:
        raise RuntimeError(f"CPU P1 Newton did not reach rel {rtol:g}: {history}")
    return u, stats


def run_job(out_dir: Path, tag: str):
    """Solve ``cpu_jobs/<tag>`` and write the entry's solution and record."""
    import resource

    jobs = out_dir / "cpu_jobs"
    z = np.load(jobs / f"{tag}.npz")
    rec = json.loads((jobs / f"{tag}.json").read_text())
    try:  # fields solve_phase added to the deferred record (surface source, mesh fallback)
        deferred = json.loads((out_dir / f"result_{tag}.json").read_text())
        rec["info"].update({k: deferred[k] for k in ("surface_source", "mesh_fallback") if k in deferred})
    except (OSError, ValueError):
        pass
    started = datetime.now().isoformat()
    t0 = time.perf_counter()
    try:
        u, stats = solve(z["nodes"], z["elements"], z["f"], z["fixed"], z["u0"], float(z["mu"]), float(z["kappa"]),
                         rtol=float(rec["rtol"]), linear=rec.get("linear", "pardiso"),
                         load_steps=int(rec.get("load_steps") or 5), full_newton=bool(rec.get("full_newton")))
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {str(exc)[:600]}"
        rec["info"].update(ok=False, error=f"CPU solve failed: {text}", cpu_job={"started": started, "error": text})
        (out_dir / f"result_{tag}.json").write_text(json.dumps(rec["info"], indent=2, default=str))
        (jobs / f"{tag}.failed").write_text(traceback.format_exc())
        print(f"[cpu-p1 {tag}] FAILED: {text}", flush=True)
        return 1
    np.savez_compressed(out_dir / f"p1_solution_{tag}.npz", nodes=z["nodes"], elements=z["elements"], u=u,
                        clamped=z["fixed"], side_traction=z["side_traction"], f_ext=z["f"], mu=z["mu"],
                        kappa=z["kappa"], band_radius=z["band_radius"])
    sec = stats["seconds"]
    hit = next((k for k, r in enumerate(stats["relative_residuals"]) if k > 0 and r <= rec["session_rtol"]), None)
    if "load_step_histories" in stats:  # full Newton: the first row of any load step, as the GPU log is read
        rows = [(k, r) for hist in stats["load_step_histories"] for k, r in enumerate(hist)]
        hit = next((i for i, (k, r) in enumerate(rows) if k > 0 and r <= rec["session_rtol"]), None)
    stats["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    info = rec["info"]
    info["solve"] = {**stats, "reason": rec["reason"], "unit_diagonal": False, "solver_extra": {"cpu": True},
                     "failed_attempts": info.get("solve", {}).get("failed_attempts", [])}
    phases = info["timing"]["phases"]
    phases["solve"] = time.perf_counter() - t0
    info["timing"].update(
        solve_device="cpu", solve_seconds=phases["solve"],
        solve_seconds_session_tol=None if hit is None else
        sec["assembly"] + sec["linear_setup"] + stats["iteration_seconds"][hit],
        newton_iterations_session_tol=hit)
    info["p1_dofs"] = stats["dofs"]
    info["ok"] = True
    info["cpu_job"] = {"started": started, "finished": datetime.now().isoformat()}
    (out_dir / f"result_{tag}.json").write_text(json.dumps(info, indent=2, default=str))
    (jobs / f"{tag}.done").write_text(datetime.now().isoformat())
    print(f"[cpu-p1 {tag}] done in {phases['solve']:.0f} s ({stats['linear_solver']})", flush=True)
    return 0


def direct_main(src: Path, dst: Path):
    """``--direct A.npz x.npy``: one sparse direct solve for the GPU job's direct
    attempt (Nitsche or P1).  Symmetric matrices (to 1e-10 of the largest
    entry) are factored symmetric-indefinite (mtype -2), others as general
    (mtype 11); the memory guard of ``Pardiso`` applies."""
    t0 = time.perf_counter()
    z = np.load(src)
    A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    asym = abs(A - A.T).max()
    mtype = -2 if asym <= 1e-10 * abs(A).max() else 11
    lin = Pardiso(A, mtype=mtype)
    x = lin.solve(z["rhs"])
    lin.free()
    res = float(np.linalg.norm(A @ x - z["rhs"]) / max(np.linalg.norm(z["rhs"]), 1e-300))
    np.save(dst, x)
    print(f"PARDISO mtype {mtype}, n {A.shape[0]}, rel residual {res:.2e}, {time.perf_counter() - t0:.1f} s")
    return 0 if np.isfinite(res) and res < 1e-6 else 3


if __name__ == "__main__":
    if sys.argv[1] == "--direct":
        # A direct attempt blocks a GPU job: give it most of the cores (read by
        # MKL when pypardiso loads it, after this point).
        os.environ["MKL_NUM_THREADS"] = os.environ.get("DIRECT_THREADS", "48")
        sys.exit(direct_main(Path(sys.argv[2]), Path(sys.argv[3])))
    sys.exit(run_job(Path(sys.argv[1]), sys.argv[2]))
