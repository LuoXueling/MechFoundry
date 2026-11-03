"""GPU-native GCMMA (Globally Convergent Method of Moving Asymptotes) solver.

All element-wise operations are warp kernels running on GPU with a configurable
floating-point precision (float32 or float64).  Reductions (sums) use
``wp.atomic_add``.  Dual problem solved via scipy.optimize.minimize (L-BFGS-B).

Implements the NLopt CCSA formulation with:
- Multi-constraint support
- Convexity penalty parameters (rho, rhoc) for global convergence
- Inner conservatism loop with eval_func callback
- Oscillation-adapted asymptotes (sigma tracking)

Reference: Svanberg (2002) — GCMMA; NLopt CCSA implementation.

Internal precision is controlled by the ``dtype`` argument to CCSAMMASolver:
- ``wp.float64`` (default / mixed-precision mode): high-precision internal math.
- ``wp.float32`` (single-precision mode): unified float32 pipeline, no boundary
  casts between MMA internals and the float32 design-variable arrays.
"""

from typing import Any

import numpy as np
import warp as wp


# ---------------------------------------------------------------------------
# Warp kernels (dtype-generic via warp Any)
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _mma_compute_sigma(
    x: wp.array(dtype=Any),
    x_old1: wp.array(dtype=Any),
    x_old2: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    xmin: Any,
    xmax: Any,
    sigma_init: Any,
    has_history: wp.int32,
):
    """Oscillation-adapted sigma (asymptote distance parameter)."""
    i = wp.tid()
    if has_history == 0:
        sigma[i] = sigma_init * (xmax - xmin)
    else:
        osc = (x[i] - x_old1[i]) * (x_old1[i] - x_old2[i])
        s = sigma[i]
        if osc < type(s)(0.0):
            s = s * type(s)(0.7)
        elif osc > type(s)(0.0):
            s = s * type(s)(1.2)
        s = wp.clamp(
            s, type(xmax)(0.01) * (xmax - xmin), type(xmax)(10.0) * (xmax - xmin)
        )
        sigma[i] = s


@wp.kernel(enable_backward=False)
def _mma_compute_bounds(
    x: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    xmin: Any,
    xmax: Any,
    move: Any,
    L: wp.array(dtype=Any),
    U: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    beta: wp.array(dtype=Any),
):
    """Compute asymptotes L, U and move-limited bounds alpha, beta."""
    i = wp.tid()
    s = sigma[i]
    xi = x[i]
    L[i] = xi - s
    U[i] = xi + s
    lo = wp.max(xmin, xi - move * (xmax - xmin))
    hi = wp.min(xmax, xi + move * (xmax - xmin))
    alpha[i] = wp.max(lo, L[i] + type(s)(0.1) * s)
    beta[i] = wp.min(hi, U[i] - type(s)(0.1) * s)


@wp.kernel(enable_backward=False)
def _gcmma_compute_PQ(
    grad: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    rho: Any,
    P: wp.array(dtype=Any),
    Q: wp.array(dtype=Any),
    v: wp.array(dtype=Any),
):
    """NLopt P/Q formulation with convexity penalty rho.

    P[i] = max(0, g)*s^2 + 0.25*rho*s
    Q[i] = max(0, -g)*s^2 + 0.25*rho*s
    v[i] = |g|*s + 0.5*rho
    """
    i = wp.tid()
    g = grad[i]
    s = sigma[i]
    s2 = s * s
    rho_term = type(rho)(0.25) * rho * s
    P[i] = wp.max(type(s)(0.0), g) * s2 + rho_term
    Q[i] = wp.max(type(s)(0.0), -g) * s2 + rho_term
    v[i] = wp.abs(g) * s + type(rho)(0.5) * rho


@wp.kernel(enable_backward=False)
def _gcmma_reduce_sum(
    arr: wp.array(dtype=Any),
    result: wp.array(dtype=Any),
):
    """Reduce: result[0] += sum(arr)."""
    i = wp.tid()
    wp.atomic_add(result, 0, arr[i])


@wp.kernel(enable_backward=False)
def _gcmma_accumulate_PQ(
    Pi: wp.array(dtype=Any),
    Qi: wp.array(dtype=Any),
    lam_j: Any,
    P_lam: wp.array(dtype=Any),
    Q_lam: wp.array(dtype=Any),
):
    """P_lam += lam_j * Pi, Q_lam += lam_j * Qi."""
    i = wp.tid()
    P_lam[i] = P_lam[i] + lam_j * Pi[i]
    Q_lam[i] = Q_lam[i] + lam_j * Qi[i]


@wp.kernel(enable_backward=False)
def _gcmma_x_from_PQ(
    P_lam: wp.array(dtype=Any),
    Q_lam: wp.array(dtype=Any),
    L: wp.array(dtype=Any),
    U: wp.array(dtype=Any),
    alpha: wp.array(dtype=Any),
    beta: wp.array(dtype=Any),
    x_out: wp.array(dtype=Any),
):
    """x = (L + sqrt(Q/P)*U) / (1 + sqrt(Q/P)), clamped to [alpha, beta]."""
    i = wp.tid()
    p = P_lam[i]
    q = Q_lam[i]
    ratio = wp.sqrt(q / p)
    xi = (L[i] + ratio * U[i]) / (type(ratio)(1.0) + ratio)
    x_out[i] = wp.clamp(xi, alpha[i], beta[i])


@wp.kernel(enable_backward=False)
def _gcmma_dual_component(
    P: wp.array(dtype=Any),
    Q: wp.array(dtype=Any),
    x: wp.array(dtype=Any),
    L: wp.array(dtype=Any),
    U: wp.array(dtype=Any),
    result: wp.array(dtype=Any),
):
    """Reduction: result[0] += sum(P/(U-x) + Q/(x-L))."""
    i = wp.tid()
    xi = x[i]
    ux = U[i] - xi
    xl = xi - L[i]
    val = P[i] / ux + Q[i] / xl
    wp.atomic_add(result, 0, val)


@wp.kernel(enable_backward=False)
def _gcmma_approx_value(
    x_new: wp.array(dtype=Any),
    x_cur: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    P: wp.array(dtype=Any),
    Q: wp.array(dtype=Any),
    result: wp.array(dtype=Any),
):
    """Reduction: result[0] += sum(P/(sigma-dx) + Q/(sigma+dx))."""
    i = wp.tid()
    dx = x_new[i] - x_cur[i]
    s = sigma[i]
    sp = s + dx
    sm = s - dx
    val = P[i] / sm + Q[i] / sp
    wp.atomic_add(result, 0, val)


@wp.kernel(enable_backward=False)
def _gcmma_wval(
    x_new: wp.array(dtype=Any),
    x_cur: wp.array(dtype=Any),
    sigma: wp.array(dtype=Any),
    result: wp.array(dtype=Any),
):
    """Reduction: result[0] += sum(0.5 * dx^2 / (sigma^2 - dx^2))."""
    i = wp.tid()
    dx = x_new[i] - x_cur[i]
    s = sigma[i]
    s2 = s * s
    dx2 = dx * dx
    denom = s2 - dx2
    val = type(dx2)(0.5) * dx2 / denom
    wp.atomic_add(result, 0, val)


@wp.kernel(enable_backward=False)
def _mma_copy(src: wp.array(dtype=Any), dst: wp.array(dtype=Any)):
    i = wp.tid()
    dst[i] = src[i]


# ---------------------------------------------------------------------------
# CCSAMMASolver
# ---------------------------------------------------------------------------


class CCSAMMASolver:
    """GPU-native GCMMA solver for constrained optimization.

    Implements the Globally Convergent Method of Moving Asymptotes (GCMMA)
    following the NLopt CCSA formulation. All element-wise operations run on
    GPU via warp kernels. The multi-constraint dual problem is solved via
    scipy.optimize.minimize (L-BFGS-B).

    When ``eval_func`` is provided to :meth:`step`, an inner conservatism
    loop ensures the MMA approximation is a global upper bound of the true
    functions, guaranteeing global convergence.

    Parameters
    ----------
    n_vars : int
        Number of design variables.
    n_constraints : int
        Number of inequality constraints (default 0).
    xmin, xmax : float
        Global lower/upper bounds on design variables.
    move_limit : float
        Fractional move limit (relative to xmax - xmin).
    rel_tolerance : float
        Relative tolerance for conservatism check initialization.
    max_inner : int
        Maximum inner conservatism iterations per step.
    device : str
        Warp device string.
    dtype : warp scalar type
        Internal floating-point dtype.  ``wp.float64`` (default) for
        mixed-precision mode; ``wp.float32`` for single-precision mode.
    gpu_dual : bool
        If True, keep all n-dimensional arrays on GPU during the dual solve.
        Only the small lambda vector (size m) and scalar dual objective /
        gradient transfer to CPU for scipy.optimize.minimize.  Default False
        (bulk-download to numpy).
    """

    def __init__(
        self,
        n_vars,
        n_constraints=0,
        xmin=0.001,
        xmax=1.0,
        move_limit=0.2,
        rel_tolerance=1e-5,
        max_inner=15,
        device="cuda",
        dtype=wp.float64,
        gpu_dual=False,
    ):
        self.n = n_vars
        self.m = n_constraints
        self.gpu_dual = gpu_dual
        self.xmin = float(xmin)
        self.xmax = float(xmax)
        self.move = float(move_limit)
        self.rel_tolerance = rel_tolerance
        self.max_inner = max_inner
        self.device = device
        self._iter = 0
        self._dtype = dtype
        # Scalar type constructor: wp.float32(x) or wp.float64(x)
        self._T = dtype

        # GCMMA convexity penalty parameters
        self.rho = 1e-5
        self.rho_min = 1e-12
        self.rhoc = [1e-5] * n_constraints
        self.rhoc_min = [1e-12] * n_constraints

        # Conservatism tolerances (set on first call from function values)
        self.tol_f0 = None
        self.tol_fc = None

        # Persistent GPU buffers (dtype-parameterised)
        self.sigma = wp.zeros(n_vars, dtype=dtype, device=device)
        self.L = wp.zeros(n_vars, dtype=dtype, device=device)
        self.U = wp.zeros(n_vars, dtype=dtype, device=device)
        self.alpha = wp.zeros(n_vars, dtype=dtype, device=device)
        self.beta = wp.zeros(n_vars, dtype=dtype, device=device)
        self.P0 = wp.zeros(n_vars, dtype=dtype, device=device)
        self.Q0 = wp.zeros(n_vars, dtype=dtype, device=device)
        self.v0 = wp.zeros(n_vars, dtype=dtype, device=device)
        self.P_lam = wp.zeros(n_vars, dtype=dtype, device=device)
        self.Q_lam = wp.zeros(n_vars, dtype=dtype, device=device)
        self.x_trial = wp.zeros(n_vars, dtype=dtype, device=device)
        self.x_cur = wp.zeros(n_vars, dtype=dtype, device=device)
        self.x_old1 = wp.zeros(n_vars, dtype=dtype, device=device)
        self.x_old2 = wp.zeros(n_vars, dtype=dtype, device=device)

        # Per-constraint buffers (list of warp arrays)
        self.Pi = [
            wp.zeros(n_vars, dtype=dtype, device=device) for _ in range(n_constraints)
        ]
        self.Qi = [
            wp.zeros(n_vars, dtype=dtype, device=device) for _ in range(n_constraints)
        ]
        self.vi = [
            wp.zeros(n_vars, dtype=dtype, device=device) for _ in range(n_constraints)
        ]

        # Scalar reduction buffers
        self._scalar_buf = wp.zeros(1, dtype=dtype, device=device)
        self._scalar_buf2 = wp.zeros(1, dtype=dtype, device=device)

        # Per-constraint scalar buffers for GPU dual path
        self._dual_scalar_bufs = [
            wp.zeros(1, dtype=dtype, device=device) for _ in range(n_constraints)
        ]

        # Boundary cast buffers: only needed when internal dtype != wp.float32.
        # When internal dtype IS float32, caller arrays are used directly (no copy).
        self._needs_cast = dtype != wp.float32
        if self._needs_cast:
            self._x_internal = wp.zeros(n_vars, dtype=dtype, device=device)
            self._grad_f0_internal = wp.zeros(n_vars, dtype=dtype, device=device)
            self._grad_fi_internal: list = []
            self._x_trial_f32 = wp.zeros(n_vars, dtype=wp.float32, device=device)

    # -----------------------------------------------------------------
    # Main step
    # -----------------------------------------------------------------

    def step(self, x, f0, grad_f0, f_i=None, grad_f_i=None, eval_func=None):
        """Perform one GCMMA update.

        Parameters
        ----------
        x : wp.array(dtype=wp.float32)
            Current design variables (n,).
        f0 : float
            Current objective value.
        grad_f0 : wp.array(dtype=wp.float32)
            Objective gradient (n,).
        f_i : list[float] or None
            Constraint values (length m). Constraints satisfy f_i <= 0.
        grad_f_i : list[wp.array] or None
            Constraint gradients, each (n,).
        eval_func : callable or None
            If provided, called as ``eval_func(x_trial)`` returning
            ``(f0_new, f_i_new)`` for the conservatism check.
            When None, the inner loop is skipped (pure MMA mode).

        Returns
        -------
        wp.array(dtype=wp.float32)
            Updated design variables (n,).
        """
        n = self.n
        dev = self.device

        # Determine number of active constraints
        if f_i is not None and grad_f_i is not None:
            f_i = list(f_i)
            grad_f_i = list(grad_f_i)
            m = len(f_i)
        else:
            f_i = []
            grad_f_i = []
            m = 0

        # Ensure per-constraint buffers exist for m constraints
        self._ensure_constraint_buffers(m)

        # --- Boundary cast: float32 inputs → internal dtype (skip if same) ---
        if self._needs_cast:
            wp.utils.array_cast(in_array=x, out_array=self._x_internal)
            wp.utils.array_cast(in_array=grad_f0, out_array=self._grad_f0_internal)
            while len(self._grad_fi_internal) < m:
                self._grad_fi_internal.append(
                    wp.zeros(n, dtype=self._dtype, device=dev)
                )
            for j in range(m):
                wp.utils.array_cast(
                    in_array=grad_f_i[j], out_array=self._grad_fi_internal[j]
                )
            x_in = self._x_internal
            grad_f0_in = self._grad_f0_internal
            grad_fi_in = self._grad_fi_internal
        else:
            x_in = x
            grad_f0_in = grad_f0
            grad_fi_in = list(grad_f_i)

        # Store for use in _conservatism_loop
        self._x_in = x_in
        self._grad_f0_in = grad_f0_in
        self._grad_fi_in = grad_fi_in

        # 1. Initialize tolerances on first call
        if self._iter == 0:
            self.tol_f0 = max(self.rel_tolerance * abs(f0), 1e-8)
            if m > 0:
                self.tol_fc = [
                    max(self.rel_tolerance * abs(f_i[j]), 1e-8) for j in range(m)
                ]
            else:
                self.tol_fc = []

        # 2. Update sigma (oscillation tracking)
        sigma_init = self._T(self.move / 0.9)
        has_history = wp.int32(1 if self._iter >= 2 else 0)
        wp.launch(
            _mma_compute_sigma,
            dim=n,
            inputs=[
                x_in,
                self.x_old1,
                self.x_old2,
                self.sigma,
                self._T(self.xmin),
                self._T(self.xmax),
                sigma_init,
                has_history,
            ],
            device=dev,
        )

        # 3. Shift history and save x_cur for conservatism check
        wp.launch(_mma_copy, dim=n, inputs=[self.x_old1, self.x_old2], device=dev)
        wp.launch(_mma_copy, dim=n, inputs=[x_in, self.x_old1], device=dev)
        wp.launch(_mma_copy, dim=n, inputs=[x_in, self.x_cur], device=dev)

        # 4. Compute asymptotes + bounds
        wp.launch(
            _mma_compute_bounds,
            dim=n,
            inputs=[
                x_in,
                self.sigma,
                self._T(self.xmin),
                self._T(self.xmax),
                self._T(self.move),
                self.L,
                self.U,
                self.alpha,
                self.beta,
            ],
            device=dev,
        )

        # 5. Update penalty parameters (init or decay)
        self._update_penalties(f0, f_i, grad_f0, grad_f_i, m)

        # 6. Compute P0/Q0/v0 for objective
        wp.launch(
            _gcmma_compute_PQ,
            dim=n,
            inputs=[
                grad_f0_in,
                self.sigma,
                self._T(self.rho),
                self.P0,
                self.Q0,
                self.v0,
            ],
            device=dev,
        )

        # 7. Compute Pi/Qi/vi for each constraint
        for j in range(m):
            wp.launch(
                _gcmma_compute_PQ,
                dim=n,
                inputs=[
                    grad_fi_in[j],
                    self.sigma,
                    self._T(self.rhoc[j]),
                    self.Pi[j],
                    self.Qi[j],
                    self.vi[j],
                ],
                device=dev,
            )

        # 8–9. Solve dual problem and compute x_trial (sets self.x_trial).
        self._solve_dual(f0, f_i, m)

        # 11. Inner conservatism loop
        if eval_func is not None:
            self._conservatism_loop(f0, f_i, m, eval_func)

        self._iter += 1

        # Return x_trial in the internal dtype.  When _needs_cast=True the
        # internal dtype is float64; returning _x_trial_f32 (float32) here
        # causes wp.copy(float32, float64_dest) to reinterpret bytes rather
        # than convert, corrupting the design variables.  Callers must use
        # wp.utils.array_cast if they need a specific output dtype.
        return self.x_trial

    # -----------------------------------------------------------------
    # Penalty parameter management
    # -----------------------------------------------------------------

    def _update_penalties(self, f0, f_i, grad_f0, grad_f_i, m):
        """Initialize or decay convexity penalty parameters."""
        if self._iter == 0:
            # Initialize from max gradient magnitude
            g0_np = grad_f0.numpy()
            max_g0 = float(np.max(np.abs(g0_np)))
            self.rho_min = max(1e-5 * max_g0, 1e-12)
            self.rho = self.rho_min

            self.rhoc_min = [1e-12] * m
            self.rhoc = [1e-12] * m
            for j in range(m):
                gi_np = grad_f_i[j].numpy()
                max_gi = float(np.max(np.abs(gi_np)))
                self.rhoc_min[j] = max(1e-5 * max_gi, 1e-12)
                self.rhoc[j] = self.rhoc_min[j]
        else:
            # Decay penalties
            self.rho = max(0.1 * self.rho, self.rho_min)
            for j in range(m):
                self.rhoc[j] = max(0.1 * self.rhoc[j], self.rhoc_min[j])

    # -----------------------------------------------------------------
    # Dual problem
    # -----------------------------------------------------------------

    def _compute_intercept(self, f_val, v_arr_np):
        """Compute r = f_val - sum(v).  v_arr_np is a numpy array (already on CPU)."""
        return f_val - float(v_arr_np.sum())

    def _compute_x_from_lambda(self, lam, m):
        """Compute x_trial from P/Q aggregated by lambda vector (GPU path)."""
        n = self.n
        dev = self.device
        wp.launch(_mma_copy, dim=n, inputs=[self.P0, self.P_lam], device=dev)
        wp.launch(_mma_copy, dim=n, inputs=[self.Q0, self.Q_lam], device=dev)
        for j in range(m):
            lj = float(lam[j]) if hasattr(lam, "__len__") else float(lam)
            if lj > 0.0:
                wp.launch(
                    _gcmma_accumulate_PQ,
                    dim=n,
                    inputs=[
                        self.Pi[j],
                        self.Qi[j],
                        self._T(lj),
                        self.P_lam,
                        self.Q_lam,
                    ],
                    device=dev,
                )
        wp.launch(
            _gcmma_x_from_PQ,
            dim=n,
            inputs=[
                self.P_lam,
                self.Q_lam,
                self.L,
                self.U,
                self.alpha,
                self.beta,
                self.x_trial,
            ],
            device=dev,
        )

    def _solve_dual(self, f0, f_i, m):
        """Dispatch to GPU or CPU dual solve based on ``gpu_dual`` flag."""
        if self.gpu_dual:
            return self._solve_dual_gpu(f0, f_i, m)
        return self._solve_dual_cpu(f0, f_i, m)

    def _solve_dual_cpu(self, f0, f_i, m):
        """Solve the m-dimensional dual problem on CPU.

        Downloads P/Q and geometry arrays from GPU once, then runs the entire
        dual optimisation in numpy — eliminating O(n_evals) GPU round-trips.
        Sets self.x_trial on the GPU from the numpy result before returning.
        """
        from scipy.optimize import minimize

        # --- One-time bulk download (single wp.synchronize()) ---
        wp.synchronize()
        P0_np = self.P0.numpy().copy()
        Q0_np = self.Q0.numpy().copy()
        L_np = self.L.numpy().copy()
        U_np = self.U.numpy().copy()
        a_np = self.alpha.numpy().copy()
        b_np = self.beta.numpy().copy()
        Pi_np = [self.Pi[j].numpy().copy() for j in range(m)]
        Qi_np = [self.Qi[j].numpy().copy() for j in range(m)]
        v0_np = self.v0.numpy().copy()
        vi_np = [self.vi[j].numpy().copy() for j in range(m)]

        # Compute r0 / R[j] = f - sum(v)  in numpy (no GPU reductions needed)
        r0_np = float(f0) - float(v0_np.sum())
        R_np = np.array([float(f_i[j]) - float(vi_np[j].sum()) for j in range(m)])

        # --- Pure-numpy dual evaluation ---
        def _x_from_PQ(P, Q):
            P = np.maximum(P, 1e-300)
            Q = np.maximum(Q, 1e-300)
            ratio = np.sqrt(Q / P)
            xi = (L_np + ratio * U_np) / (1.0 + ratio)
            return np.clip(xi, a_np, b_np)

        def _dual_component(P, Q, x):
            ux = np.maximum(U_np - x, 1e-300)
            xl = np.maximum(x - L_np, 1e-300)
            return np.sum(P / ux + Q / xl)

        def _eval(lam_vec):
            P_lam = P0_np.copy()
            Q_lam = Q0_np.copy()
            for j, lj in enumerate(lam_vec):
                if lj > 0.0:
                    P_lam += lj * Pi_np[j]
                    Q_lam += lj * Qi_np[j]
            x = _x_from_PQ(P_lam, Q_lam)
            obj_part = _dual_component(P0_np, Q0_np, x)
            grad = np.array(
                [R_np[j] + _dual_component(Pi_np[j], Qi_np[j], x) for j in range(m)]
            )
            W = r0_np + obj_part + np.dot(lam_vec, grad)
            return -W, -grad

        if m == 0:
            x_trial_np = _x_from_PQ(P0_np, Q0_np)
            lam_opt = np.zeros(0)
        else:
            result = minimize(
                fun=_eval,
                x0=np.zeros(m),
                jac=True,
                method="L-BFGS-B",
                bounds=[(0, None)] * m,
                options={"ftol": 1e-6, "maxls": 20, "maxiter": 100, "disp": False},
            )
            lam_opt = np.maximum(0.0, result.x)

            P_lam = P0_np.copy()
            Q_lam = Q0_np.copy()
            for j, lj in enumerate(lam_opt):
                if lj > 0.0:
                    P_lam += lj * Pi_np[j]
                    Q_lam += lj * Qi_np[j]
            x_trial_np = _x_from_PQ(P_lam, Q_lam)

        # --- Upload x_trial back to GPU (internal dtype) ---
        wp.copy(
            src=wp.array(x_trial_np, dtype=self._dtype, device=self.device),
            dest=self.x_trial,
        )

        return lam_opt

    def _solve_dual_gpu(self, f0, f_i, m):
        """Solve the m-dimensional dual problem with data on GPU.

        Only the small lambda vector (size m) and scalar dual objective /
        gradient transfer between GPU and CPU for the scipy.optimize.minimize
        call.  All n-dimensional arrays stay on GPU.
        """
        from scipy.optimize import minimize

        n = self.n
        dev = self.device

        # --- Compute intercepts r0, R[j] via GPU reductions ---
        self._scalar_buf.zero_()
        wp.launch(
            _gcmma_reduce_sum,
            dim=n,
            inputs=[self.v0, self._scalar_buf],
            device=dev,
        )
        for j in range(m):
            self._dual_scalar_bufs[j].zero_()
            wp.launch(
                _gcmma_reduce_sum,
                dim=n,
                inputs=[self.vi[j], self._dual_scalar_bufs[j]],
                device=dev,
            )
        wp.synchronize()

        r0 = float(f0) - float(self._scalar_buf.numpy()[0])
        R = np.array(
            [
                float(f_i[j]) - float(self._dual_scalar_bufs[j].numpy()[0])
                for j in range(m)
            ]
        )

        if m == 0:
            # No dual to solve — just compute x from P0/Q0 on GPU
            self._compute_x_from_lambda(np.zeros(0), 0)
            wp.synchronize()
            return np.zeros(0)

        # --- GPU-native dual evaluation for scipy ---
        def _eval_gpu(lam_vec):
            # Aggregate P_lam/Q_lam and compute x_trial on GPU
            self._compute_x_from_lambda(lam_vec, m)

            # Objective dual component: sum(P0/(U-x) + Q0/(x-L))
            self._scalar_buf.zero_()
            wp.launch(
                _gcmma_dual_component,
                dim=n,
                inputs=[
                    self.P0,
                    self.Q0,
                    self.x_trial,
                    self.L,
                    self.U,
                    self._scalar_buf,
                ],
                device=dev,
            )

            # Constraint dual components: sum(Pi/(U-x) + Qi/(x-L))
            for j in range(m):
                self._dual_scalar_bufs[j].zero_()
                wp.launch(
                    _gcmma_dual_component,
                    dim=n,
                    inputs=[
                        self.Pi[j],
                        self.Qi[j],
                        self.x_trial,
                        self.L,
                        self.U,
                        self._dual_scalar_bufs[j],
                    ],
                    device=dev,
                )

            # Single sync to read all scalar results
            wp.synchronize()
            obj_part = float(self._scalar_buf.numpy()[0])
            grad = np.array(
                [R[j] + float(self._dual_scalar_bufs[j].numpy()[0]) for j in range(m)]
            )

            W = r0 + obj_part + np.dot(lam_vec, grad)
            return -W, -grad

        result = minimize(
            fun=_eval_gpu,
            x0=np.zeros(m),
            jac=True,
            method="L-BFGS-B",
            bounds=[(0, None)] * m,
            options={"ftol": 1e-6, "maxls": 20, "maxiter": 100, "disp": False},
        )
        lam_opt = np.maximum(0.0, result.x)

        # Final x_trial from optimal lambda (already on GPU)
        self._compute_x_from_lambda(lam_opt, m)
        wp.synchronize()

        return lam_opt

    # -----------------------------------------------------------------
    # Conservatism loop
    # -----------------------------------------------------------------

    def _eval_approximation(self, P, Q, f_val, v_arr):
        """Evaluate approximation at x_trial: r + sum(P/(sigma-dx) + Q/(sigma+dx))."""
        if self.gpu_dual:
            # GPU path: compute sum(v) and approx_value in parallel, single sync
            self._scalar_buf.zero_()
            wp.launch(
                _gcmma_reduce_sum,
                dim=self.n,
                inputs=[v_arr, self._scalar_buf],
                device=self.device,
            )
            self._scalar_buf2.zero_()
            wp.launch(
                _gcmma_approx_value,
                dim=self.n,
                inputs=[self.x_trial, self.x_cur, self.sigma, P, Q, self._scalar_buf2],
                device=self.device,
            )
            wp.synchronize()
            r = float(f_val) - float(self._scalar_buf.numpy()[0])
            return r + float(self._scalar_buf2.numpy()[0])
        else:
            wp.synchronize()
            r = self._compute_intercept(f_val, v_arr.numpy())
            self._scalar_buf.zero_()
            wp.launch(
                _gcmma_approx_value,
                dim=self.n,
                inputs=[self.x_trial, self.x_cur, self.sigma, P, Q, self._scalar_buf],
                device=self.device,
            )
            return r + float(self._scalar_buf.numpy()[0])

    def _eval_wval(self):
        """Compute w-value: sum(0.5 * dx^2 / (sigma^2 - dx^2))."""
        self._scalar_buf2.zero_()
        wp.launch(
            _gcmma_wval,
            dim=self.n,
            inputs=[self.x_trial, self.x_cur, self.sigma, self._scalar_buf2],
            device=self.device,
        )
        return max(float(self._scalar_buf2.numpy()[0]), 1e-15)

    def _conservatism_loop(self, f0, f_i, m, eval_func):
        """Inner GCMMA conservatism loop.

        Verifies the MMA approximation is a conservative (upper) bound of the
        true functions at x_trial. If not, increases convexity penalties and
        re-solves until conservative or max_inner iterations reached.
        """
        n = self.n
        dev = self.device

        for inner_it in range(self.max_inner):
            # Evaluate true objective/constraints at x_trial (internal dtype).
            eval_res = eval_func(self.x_trial)
            if isinstance(eval_res, tuple):
                f0_new, f_i_new = eval_res
            else:
                f0_new = eval_res
                f_i_new = []
            f_i_new = list(f_i_new) if f_i_new else []

            # Compute approximation values at x_trial
            gval = self._eval_approximation(self.P0, self.Q0, f0, self.v0)

            gcvals = []
            for j in range(m):
                gcval_j = self._eval_approximation(
                    self.Pi[j],
                    self.Qi[j],
                    f_i[j],
                    self.vi[j],
                )
                gcvals.append(gcval_j)

            # Compute w-value (penalty denominator)
            wval = self._eval_wval()

            # Check conservatism
            obj_ok = (gval + self.tol_f0) >= f0_new
            con_ok = (
                all((gcvals[j] + self.tol_fc[j]) >= f_i_new[j] for j in range(m))
                if m > 0
                else True
            )

            if obj_ok and con_ok:
                break  # Accept x_trial

            # Increase penalties where not conservative
            wval_safe = max(wval, 1e-20)
            if f0_new > gval:
                self.rho = min(
                    10.0 * self.rho,
                    1.1 * (self.rho + (f0_new - gval) / wval_safe),
                )
            for j in range(m):
                if len(f_i_new) > j and f_i_new[j] > gcvals[j]:
                    self.rhoc[j] = min(
                        10.0 * self.rhoc[j],
                        1.1 * (self.rhoc[j] + (f_i_new[j] - gcvals[j]) / wval_safe),
                    )

            # Recompute P/Q with updated penalties (using stored internal-dtype gradients)
            wp.launch(
                _gcmma_compute_PQ,
                dim=n,
                inputs=[
                    self._grad_f0_in,
                    self.sigma,
                    self._T(self.rho),
                    self.P0,
                    self.Q0,
                    self.v0,
                ],
                device=dev,
            )
            for j in range(m):
                wp.launch(
                    _gcmma_compute_PQ,
                    dim=n,
                    inputs=[
                        self._grad_fi_in[j],
                        self.sigma,
                        self._T(self.rhoc[j]),
                        self.Pi[j],
                        self.Qi[j],
                        self.vi[j],
                    ],
                    device=dev,
                )

            # Re-solve dual with new P/Q (downloads fresh arrays, sets x_trial)
            self._solve_dual(f0, f_i, m)

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def _ensure_constraint_buffers(self, m):
        """Ensure per-constraint GPU buffers exist for m constraints."""
        while len(self.Pi) < m:
            self.Pi.append(wp.zeros(self.n, dtype=self._dtype, device=self.device))
            self.Qi.append(wp.zeros(self.n, dtype=self._dtype, device=self.device))
            self.vi.append(wp.zeros(self.n, dtype=self._dtype, device=self.device))
            self.rhoc.append(1e-12)
            self.rhoc_min.append(1e-12)
            if self._needs_cast:
                while len(self._grad_fi_internal) < m:
                    self._grad_fi_internal.append(
                        wp.zeros(self.n, dtype=self._dtype, device=self.device)
                    )
        while len(self._dual_scalar_bufs) < m:
            self._dual_scalar_bufs.append(
                wp.zeros(1, dtype=self._dtype, device=self.device)
            )
        self.m = max(self.m, m)
