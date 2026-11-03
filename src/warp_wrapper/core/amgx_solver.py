"""AMGX PCG+AMG solver integration via NVIDIA AMGX / pyamgx.

Provides:
- AMGXSolver class: wraps pyamgx to run AMGX's internal PCG solver with
  AGGREGATION AMG as preconditioner.  The entire CG loop runs on GPU
  inside AMGX — no per-iteration Python overhead.
- Singleton solver pool: one solver per (block_dim, config_type) for process
  lifetime.
- AMGX config selection via ``select_amgx_config``.

AGGREGATION AMG supports native block_dim > 1, so BSR matrices are uploaded
directly without scalar expansion.

The BLOCK_JACOBI smoother with a conservative relaxation factor is used
instead of MULTICOLOR_GS: it has much lower per-V-cycle cost while still
providing effective preconditioning.  Separate configs for 2D and 3D use
tuned relaxation factors (0.8 for 2D, 0.5 for 3D) and selector sizes
(SIZE_4 / SIZE_8).

Object lifecycle follows the pyamgx demo pattern:
  initialize → create (cfg, rsrc, solver, matrix, vectors) → use → destroy
  (matrix, vectors, solver, rsrc, cfg) → finalize
"""

import atexit

import warp as wp
from warp.sparse import BsrMatrix


# ---------------------------------------------------------------------------
# AMGX initialisation and singleton pool
# ---------------------------------------------------------------------------

_amgx_initialized = False
_amgx_solvers = {}  # Singleton pool: (block_dim, config_type) -> AMGXSolver
_atexit_registered = False

# ---------------------------------------------------------------------------
# AMGX solver configs
#
# PCG+AMG: AMGX runs its internal PCG solver with AGGREGATION AMG as
# preconditioner.  The full CG loop runs on GPU — no per-iteration
# Python overhead.  One V-cycle is applied per PCG iteration.
# No determinism_flag (allows parallel atomics for speed).
# ---------------------------------------------------------------------------

# 2D PCG + AGGREGATION AMG (native block_dim=2).
#
# Tuned via sweep over selectors (SIZE_2/4/8), smoothers (BLOCK_JACOBI / GS),
# relaxation factors, and sweep counts on a 400×100 2D elasticity problem:
#   - SIZE_4 selector: best balance of coarse grid size and V-cycle cost.
#     SIZE_2 gives fewer PCG iterations but higher per-V-cycle cost (deeper
#     hierarchy, 2× more work/level), resulting in ~32% longer wall time.
#   - BLOCK_JACOBI with relaxation_factor=0.8: all tested factors (0.5-0.8)
#     converge in 2D; 0.8 gives fewest CG iterations at minimal per-V-cycle cost.
#   - presweeps/postsweeps=1: 2 sweeps reduces iterations but doubles V-cycle
#     cost, resulting in net slowdown. 1 sweep is the sweet spot.
_AMGX_PCG_AMG_2D_CFG = {
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "max_iters": 1000,
        "convergence": "RELATIVE_INI",
        "tolerance": 1e-6,
        "norm": "L2",
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "monitor_residual": 1,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "selector": "SIZE_4",
            "max_iters": 1,
            "cycle": "V",
            "smoother": {
                "solver": "BLOCK_JACOBI",
                "relaxation_factor": 0.8,
            },
            "presweeps": 1,
            "postsweeps": 1,
            "max_levels": 50,
            "coarse_solver": "DENSE_LU_SOLVER",
        },
    },
}

# 3D PCG + AGGREGATION AMG (native block_dim=3).
#
# Tuned via sweep over selectors, smoothers, and relaxation factors on a
# 38^3 3D elasticity problem:
#   - SIZE_8 selector: standard approach for 3D, aggregates 8 nodes per
#     coarse node (naturally matches hex element structure).
#   - BLOCK_JACOBI with relaxation_factor=0.5: higher factors (>=0.7) diverge
#     in 3D; 0.5 is the sweet spot for convergence + speed.
_AMGX_PCG_AMG_3D_CFG = {
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "max_iters": 1000,
        "convergence": "RELATIVE_INI",
        "tolerance": 1e-6,
        "norm": "L2",
        "print_solve_stats": 0,
        "obtain_timings": 0,
        "monitor_residual": 1,
        "preconditioner": {
            "solver": "AMG",
            "algorithm": "AGGREGATION",
            "selector": "SIZE_8",
            "max_iters": 1,
            "cycle": "V",
            "smoother": {
                "solver": "BLOCK_JACOBI",
                "relaxation_factor": 0.5,
            },
            "presweeps": 1,
            "postsweeps": 1,
            "max_levels": 50,
            "coarse_solver": "DENSE_LU_SOLVER",
        },
    },
}

_AMGX_CONFIG_MAP = {
    "amg_vcycle": _AMGX_PCG_AMG_2D_CFG,
    "amg_vcycle_3d": _AMGX_PCG_AMG_3D_CFG,
}


# ---------------------------------------------------------------------------
# AMGX initialisation helpers
# ---------------------------------------------------------------------------


def _ensure_amgx_init():
    """Initialize pyamgx exactly once per process."""
    global _amgx_initialized, _atexit_registered
    if not _amgx_initialized:
        import pyamgx

        pyamgx.initialize()
        _amgx_initialized = True
    if not _atexit_registered:
        atexit.register(_finalize_amgx_at_exit)
        _atexit_registered = True


def _finalize_amgx_at_exit():
    """Clean up AMGX references at process exit.

    Individual object destruction is skipped at atexit because CUDA
    context teardown racing with AMGX cleanup can cause SIGABRT.
    ``pyamgx.finalize()`` is also skipped because it asserts on
    non-empty device pools when objects haven't been freed.

    Note: destroy() IS safe during normal (non-atexit) execution.
    The AMGXSolver._needs_rebuild path calls destroy() mid-session
    to get fresh AMGX objects free of stale hierarchy state.

    The CUDA driver releases all device memory when the process exits,
    so skipping explicit cleanup is safe.
    """
    global _amgx_initialized, _amgx_solvers
    _amgx_solvers.clear()
    _amgx_initialized = False


def _get_amgx_solver(
    block_dim: int,
    config_type: str = "amg_vcycle",
    mode: str = "dDDI",
) -> "AMGXSolver":
    """Return a singleton AMGXSolver for the given block dimension, config, and mode.

    AMGX resources are expensive to create and destroy.  This function
    maintains one solver per ``(block_dim, config_type, mode)`` for the lifetime
    of the process.  The solver handles matrix structure changes internally.
    """
    key = (block_dim, config_type, mode)
    if key not in _amgx_solvers:
        _amgx_solvers[key] = AMGXSolver(
            block_dim=block_dim, config_type=config_type, mode=mode
        )
    return _amgx_solvers[key]


def _clear_amgx_pool():
    """Reset all pooled solvers for reuse with a different problem.

    Does NOT destroy AMGX handles.  Instead, resets solver state so the
    next ``setup()`` call does a full matrix upload + hierarchy build.
    Actual cleanup happens at process exit.
    """
    for s in _amgx_solvers.values():
        s.reset()


# ---------------------------------------------------------------------------
# AMGXSolver
# ---------------------------------------------------------------------------


class AMGXSolver:
    """AMGX wrapper — runs PCG with AMG or Block-Jacobi preconditioner.

    For PCG+AMG configs (``amg_vcycle``, ``amg_vcycle_3d``): AMGX runs
    its internal PCG solver with AGGREGATION AMG as preconditioner.
    The entire CG loop executes on GPU — no per-iteration Python
    overhead.  The BSR stiffness matrix is uploaded directly with
    native block_dim.

    The AMG hierarchy is built on the first ``setup()`` call and
    rebuilt adaptively when CG iteration counts indicate the
    preconditioner has degraded.

    All GPU data is transferred via raw pointers (no host-device copies).
    """

    REBUILD_THRESHOLD = 500  # AMGX PCG iters above this trigger hierarchy rebuild

    def __init__(self, block_dim=3, config_type="amg_vcycle", mode="dDDI"):
        import pyamgx

        self._pyamgx = pyamgx
        _ensure_amgx_init()
        self._config_type = config_type
        self._mode = mode
        cfg_dict = _AMGX_CONFIG_MAP[config_type]

        self._cfg = pyamgx.Config().create_from_dict(cfg_dict)
        self._rsrc = pyamgx.Resources().create_simple(self._cfg)
        self._solver = pyamgx.Solver().create(self._rsrc, self._cfg, mode=mode)
        self._M = pyamgx.Matrix().create(self._rsrc, mode=mode)
        self._b = pyamgx.Vector().create(self._rsrc, mode=mode)
        self._x = pyamgx.Vector().create(self._rsrc, mode=mode)

        self._block_dim = block_dim
        self._n = 0
        self._bsr_nrow = 0
        self._bsr_nnz = 0
        self._is_setup = False
        self._destroyed = False
        self._needs_rebuild = True  # Force full build on first call

    def setup(self, K: BsrMatrix):
        """Upload BSR matrix *K* and build/update the preconditioner.

        On the first call the full matrix is uploaded and the AMG
        hierarchy (or PCG preconditioner) is built.  Subsequent calls
        with the same structure update values only (lagged), unless
        ``_needs_rebuild`` is set (via ``report_cg_iterations``), in
        which case the hierarchy is rebuilt with fresh values.

        Parameters
        ----------
        K : BsrMatrix
            Assembled stiffness matrix (warp BSR, on GPU).
        """
        wp.synchronize()
        bd = self._block_dim
        # BsrMatrix.nnz is an upper bound that may exceed the actual
        # deduplicated block count (e.g. after bsr_axpy unions two
        # patterns).  Sync from device offsets[nrow] to get the real nnz
        # so AMGX doesn't read garbage past the valid columns/values.
        K.nnz_sync()
        n_new = K.nrow
        nnz_new = K.nnz

        if self._is_setup and (n_new != self._bsr_nrow or nnz_new != self._bsr_nnz):
            self._is_setup = False
            self._needs_rebuild = True

        # pyamgx Matrix.upload / replace_coefficients hardcode check_for_dtype="float64",
        # so we must always pass a float64 array.  AMGX reads the actual matrix
        # precision from the mode string (3rd char: F=float32, D=float64), so the
        # Python array dtype is irrelevant to what AMGX reads from the pointer.
        # shape=(K.nnz,) is sufficient — pyamgx only extracts the raw pointer.
        values_flat = wp.array(
            ptr=K.values.ptr,
            dtype=wp.float64,
            shape=(K.nnz,),
            device=K.values.device,
            copy=False,
        )

        if not self._is_setup:
            wp.synchronize()
            self._M.upload(
                K.offsets,
                K.columns,
                values_flat,
                block_dims=[bd, bd],
                shape=[K.nrow, K.ncol],
            )
            self._n = n_new
            self._bsr_nrow = n_new
            self._bsr_nnz = nnz_new
            self._solver.setup(self._M)
            self._is_setup = True
            self._needs_rebuild = False
        elif self._needs_rebuild:
            # Destroy and recreate the AMGX matrix and solver to guarantee a
            # clean AMGX internal state.  After the initial upload+setup and
            # several lagged replace_coefficients calls, AMGX 2.5.0's
            # solver_setup() on the reused matrix object does not properly
            # rebuild the AMG hierarchy, causing PCG to fail to converge (1000
            # iterations without progress).  A fresh matrix/solver pair avoids
            # this stale-state bug entirely.
            #
            # Destruction order: solver first (it holds a reference to the old
            # matrix), then matrix.  pyamgx tests confirm destroy() is safe
            # during a session (only risky at atexit when CUDA teardown races).
            # Vectors b and x are independent of M/solver and remain alive.
            self._solver.destroy()
            self._M.destroy()
            self._M = self._pyamgx.Matrix().create(self._rsrc, mode=self._mode)
            self._solver = self._pyamgx.Solver().create(
                self._rsrc, self._cfg, mode=self._mode
            )
            wp.synchronize()
            self._M.upload(
                K.offsets,
                K.columns,
                values_flat,
                block_dims=[bd, bd],
                shape=[K.nrow, K.ncol],
            )
            self._n = n_new
            self._bsr_nrow = n_new
            self._bsr_nnz = nnz_new
            self._solver.setup(self._M)
            self._is_setup = True
            self._needs_rebuild = False
        else:
            # Lagged hierarchy: update matrix values only.
            wp.synchronize()
            self._M.replace_coefficients(values_flat)

    def solve(self, b_wp, x_wp):
        """Run AMGX solver: x = solve(A, b).

        Runs AMGX's internal PCG to convergence (all iterations on GPU,
        no Python overhead per iteration).

        Parameters
        ----------
        b_wp : wp.array(dtype=vecNd)
            Right-hand side vector.
        x_wp : wp.array(dtype=vecNd)
            Solution (modified in-place).

        Returns
        -------
        int
            Number of solver iterations.
        """
        n = self._n
        bd = self._block_dim
        # Start from x=0 for deterministic results.
        x_wp.zero_()
        wp.synchronize()
        self._b.upload_raw(b_wp.ptr, n, block_dim=bd)
        self._x.upload_raw(x_wp.ptr, n, block_dim=bd)
        self._solver.solve(self._b, self._x)
        self._x.download_raw(x_wp.ptr)
        wp.synchronize()
        return self._solver.iterations_number

    def report_cg_iterations(self, niter: int):
        """Record CG iteration count; trigger rebuild if above threshold.

        Called after each AMGX solve with the PCG iteration count.
        If the count exceeds ``REBUILD_THRESHOLD``, the next ``setup()``
        call will rebuild the AMG hierarchy instead of just updating
        matrix values.
        """
        if niter > self.REBUILD_THRESHOLD:
            self._needs_rebuild = True

    def reset(self):
        """Reset solver state for reuse with a different problem.

        AMGX handles are kept alive.  The next ``setup()`` call will do
        a full matrix upload and ``solver.setup()``.
        """
        self._is_setup = False
        self._needs_rebuild = True
        self._n = 0
        self._bsr_nrow = 0
        self._bsr_nnz = 0

    def destroy(self):
        """Mark solver as destroyed.

        Actual AMGX cleanup is handled at process exit.  Individual
        object destruction is skipped to avoid SIGABRT.
        """
        self._destroyed = True


# ---------------------------------------------------------------------------
# AMGX config selection
# ---------------------------------------------------------------------------


def select_amgx_config(block_dim: int, mode: str = "dDDI") -> str:
    """Select AMGX config type based on block dimension and precision mode.

    Parameters
    ----------
    block_dim : int
        BSR block dimension (2 for 2D, 3 for 3D).
    mode : str
        AMGX precision mode string (e.g. ``"dDDI"``, ``"dFFI"``). Precision
        is intentionally independent of the tuned hierarchy configuration.

    Returns
    -------
    str
        Config key in ``_AMGX_CONFIG_MAP``.
    """
    if block_dim >= 3:
        return "amg_vcycle_3d"
    return "amg_vcycle"
