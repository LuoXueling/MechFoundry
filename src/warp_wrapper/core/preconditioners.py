"""Preconditioners and condition number estimation for CG solvers.

Provides:
- Jacobi (diagonal) preconditioner from BSR matrix
- Diagonal preconditioner extraction from matvec (for matrix-free CG)
- Power-iteration-based condition number estimation
- Helper warp kernels for vector operations

Condition number estimation adapted from torch-sla by walker chi
(MIT License, Copyright 2024). Reference: https://github.com/walkerchi/torch-sla
"""

import math

import numpy as np
import warp as wp
from warp.sparse import BsrMatrix, bsr_get_diag, bsr_mv
from warp.optim.linear import (
    LinearOperator,
    aslinearoperator,
)
from warp._src.types import Any


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _extract_block_diag_vec(
    diag_blocks: wp.array(dtype=Any),
    diag_vec: wp.array(dtype=Any),
):
    """Extract diagonal of each block: ``diag_vec[i] = get_diag(diag_blocks[i])``."""
    i = wp.tid()
    diag_vec[i] = wp.get_diag(diag_blocks[i])


@wp.kernel(enable_backward=False)
def _safe_inverse_vec3d(
    d: wp.array(dtype=wp.vec3d),
    d_inv: wp.array(dtype=wp.vec3d),
    eps: wp.float64,
):
    """Component-wise safe inverse for vec3d."""
    i = wp.tid()
    v = d[i]
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    rx = zero
    ry = zero
    rz = zero
    if wp.abs(v[0]) > eps:
        rx = one / v[0]
    if wp.abs(v[1]) > eps:
        ry = one / v[1]
    if wp.abs(v[2]) > eps:
        rz = one / v[2]
    d_inv[i] = wp.vec3d(rx, ry, rz)


@wp.kernel(enable_backward=False)
def _safe_inverse_vec2d(
    d: wp.array(dtype=wp.vec2d),
    d_inv: wp.array(dtype=wp.vec2d),
    eps: wp.float64,
):
    """Component-wise safe inverse for vec2d."""
    i = wp.tid()
    v = d[i]
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    rx = zero
    ry = zero
    if wp.abs(v[0]) > eps:
        rx = one / v[0]
    if wp.abs(v[1]) > eps:
        ry = one / v[1]
    d_inv[i] = wp.vec2d(rx, ry)


@wp.kernel(enable_backward=False)
def _safe_inverse_vec2f(
    d: wp.array(dtype=wp.vec2),
    d_inv: wp.array(dtype=wp.vec2),
    eps: wp.float32,
):
    """Component-wise safe inverse for vec2 (float32)."""
    i = wp.tid()
    v = d[i]
    one = wp.float32(1.0)
    zero = wp.float32(0.0)
    rx = zero
    ry = zero
    if wp.abs(v[0]) > eps:
        rx = one / v[0]
    if wp.abs(v[1]) > eps:
        ry = one / v[1]
    d_inv[i] = wp.vec2(rx, ry)


@wp.kernel(enable_backward=False)
def _safe_inverse_vec3f(
    d: wp.array(dtype=wp.vec3),
    d_inv: wp.array(dtype=wp.vec3),
    eps: wp.float32,
):
    """Component-wise safe inverse for vec3 (float32)."""
    i = wp.tid()
    v = d[i]
    one = wp.float32(1.0)
    zero = wp.float32(0.0)
    rx = zero
    ry = zero
    rz = zero
    if wp.abs(v[0]) > eps:
        rx = one / v[0]
    if wp.abs(v[1]) > eps:
        ry = one / v[1]
    if wp.abs(v[2]) > eps:
        rz = one / v[2]
    d_inv[i] = wp.vec3(rx, ry, rz)


@wp.kernel(enable_backward=False)
def _safe_inverse_f64(
    diag: wp.array(dtype=wp.float64),
    inv_diag: wp.array(dtype=wp.float64),
    eps: wp.float64,
):
    """Safe scalar inverse for float64 diagonal (block_dim=1 Jacobi)."""
    i = wp.tid()
    d = diag[i]
    inv_diag[i] = wp.float64(1.0) / wp.max(d, eps)


@wp.kernel(enable_backward=False)
def _vec_dot_accumulate(
    a: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    result: wp.array(dtype=wp.float64),
):
    """Atomic accumulate: ``result[0] += dot(a[i], b[i])``."""
    i = wp.tid()
    wp.atomic_add(result, 0, wp.float64(wp.dot(a[i], b[i])))


@wp.kernel(enable_backward=False)
def _vec_cw_mul(
    a: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    out: wp.array(dtype=Any),
):
    """Component-wise multiply: ``out[i] = cw_mul(a[i], b[i])``."""
    i = wp.tid()
    out[i] = wp.cw_mul(a[i], b[i])


@wp.kernel(enable_backward=False)
def _vec_scale(
    v: wp.array(dtype=Any),
    s: Any,
    out: wp.array(dtype=Any),
):
    """Scale: ``out[i] = s * v[i]``."""
    i = wp.tid()
    out[i] = s * v[i]


# ---------------------------------------------------------------------------
# Helper: extract block diagonal as vec array
# ---------------------------------------------------------------------------


def _get_block_scalar(K: BsrMatrix):
    """Return the scalar warp type of K's block values (float32 or float64)."""
    vdtype = K.values.dtype
    if hasattr(vdtype, "_wp_scalar_type_"):
        return vdtype._wp_scalar_type_
    return vdtype  # already a scalar type


def get_block_diagonal_vec(K: BsrMatrix, device: str):
    """Extract per-node diagonal from BSR as ``vecNd`` array (float64).

    Returns
    -------
    diag_vec : wp.array(dtype=vecNd)
        Diagonal entries.
    inv_diag_vec : wp.array(dtype=vecNd)
        Component-wise safe inverse of diagonal.
    """
    diag_blocks = bsr_get_diag(K)
    block_dim = K.block_shape[0]
    n = K.nrow

    if block_dim not in (2, 3):
        raise ValueError(f"Unsupported block dimension: {block_dim}")

    vec_type = wp.vec3d if block_dim == 3 else wp.vec2d
    diag_vec = wp.empty(n, dtype=vec_type, device=device)
    inv_diag_vec = wp.empty(n, dtype=vec_type, device=device)

    wp.launch(
        _extract_block_diag_vec,
        dim=n,
        inputs=[diag_blocks, diag_vec],
        device=device,
    )

    eps = wp.float64(1e-30)
    inv_kern = _safe_inverse_vec3d if block_dim == 3 else _safe_inverse_vec2d
    wp.launch(inv_kern, dim=n, inputs=[diag_vec, inv_diag_vec, eps], device=device)

    return diag_vec, inv_diag_vec


# ---------------------------------------------------------------------------
# Jacobi preconditioner
# ---------------------------------------------------------------------------


def build_jacobi_preconditioner(
    K: BsrMatrix, device: str, _cache: dict = None, solver_scalar=None
):
    """Build a Jacobi (diagonal) preconditioner with optional caching.

    When *_cache* is provided, internal buffers are stored on first call and
    reused on subsequent calls.  Only the diagonal values are updated from the
    new *K* -- the ``LinearOperator`` object and its GPU arrays stay at the
    same memory addresses, enabling CUDA-graph reuse.

    Supports block_dim = 1 (scalar float64/float32), 2 (vec2/vec2d), and 3 (vec3/vec3d).

    Parameters
    ----------
    solver_scalar : wp.float32 or wp.float64 or None
        Scalar dtype that matches the CG solver's vectors (b, x, du).
        If None, defaults to float64 (legacy behavior).

    Returns ``(M, info_dict)``.
    """
    if solver_scalar is None:
        solver_scalar = wp.float64  # legacy default
    n = K.nrow
    block_dim = K.block_shape[0]
    k_scalar = _get_block_scalar(K)

    # --- Scalar (block_dim=1) path ---
    if block_dim == 1:
        # Use solver_scalar for the inv_diag (must match CG vectors)
        s_eps = wp.float64(1e-30) if solver_scalar == wp.float64 else wp.float32(1e-30)
        s_kern = _safe_inverse_f64  # only float64 scalar kernel defined; cast if needed
        s_dtype = wp.float64  # always use float64 for scalar diagonal inversion

        if _cache is not None and "jacobi_M" in _cache:
            block_diag_buf = _cache["jacobi_block_diag"]
            inv_diag = _cache["jacobi_inv_diag"]
            bsr_get_diag(K, out=block_diag_buf)
            wp.launch(
                s_kern,
                dim=n,
                inputs=[block_diag_buf, inv_diag, wp.float64(1e-30)],
                device=device,
            )
            return _cache["jacobi_M"], {"type": "jacobi"}

        block_diag_buf = wp.zeros(n, dtype=s_dtype, device=device)
        bsr_get_diag(K, out=block_diag_buf)
        inv_diag = wp.empty(n, dtype=s_dtype, device=device)
        wp.launch(
            s_kern,
            dim=n,
            inputs=[block_diag_buf, inv_diag, wp.float64(1e-30)],
            device=device,
        )
        M = aslinearoperator(inv_diag)
        if _cache is not None:
            _cache["jacobi_M"] = M
            _cache["jacobi_block_diag"] = block_diag_buf
            _cache["jacobi_inv_diag"] = inv_diag
        return M, {"type": "jacobi"}

    # --- Vector (block_dim=2 or 3) path ---
    # inv_diag must match solver_scalar (the CG vector type)
    solver_vec_type = wp.types.vector(length=block_dim, dtype=solver_scalar)
    k_vec_type = wp.types.vector(length=block_dim, dtype=k_scalar)  # K's native diagonal type

    if solver_scalar == wp.float64:
        inv_kern = _safe_inverse_vec3d if block_dim == 3 else _safe_inverse_vec2d
        eps = wp.float64(1e-30)
    else:
        inv_kern = _safe_inverse_vec3f if block_dim == 3 else _safe_inverse_vec2f
        eps = wp.float32(1e-30)

    if _cache is not None and "jacobi_M" in _cache:
        # --- Fast path: reuse existing buffers, just update values ---
        block_diag_buf = _cache["jacobi_block_diag"]
        k_diag_vec = _cache["jacobi_k_diag_vec"]  # K's native precision
        inv_diag = _cache["jacobi_inv_diag"]
        M = _cache["jacobi_M"]

        bsr_get_diag(K, out=block_diag_buf)
        wp.launch(
            _extract_block_diag_vec,
            dim=n,
            inputs=[block_diag_buf, k_diag_vec],
            device=device,
        )
        # Cast K-precision diagonal to solver precision if needed
        if solver_vec_type != k_vec_type:
            solver_diag_vec = _cache["jacobi_solver_diag_vec"]
            wp.utils.array_cast(in_array=k_diag_vec, out_array=solver_diag_vec)
            wp.launch(
                inv_kern, dim=n, inputs=[solver_diag_vec, inv_diag, eps], device=device
            )
        else:
            wp.launch(
                inv_kern, dim=n, inputs=[k_diag_vec, inv_diag, eps], device=device
            )
        return M, {"type": "jacobi"}

    # --- First call: allocate buffers and build LinearOperator ---
    block_diag_buf = wp.zeros(n, dtype=K.values.dtype, device=device)
    bsr_get_diag(K, out=block_diag_buf)

    k_diag_vec = wp.empty(n, dtype=k_vec_type, device=device)
    inv_diag = wp.empty(n, dtype=solver_vec_type, device=device)

    wp.launch(
        _extract_block_diag_vec,
        dim=n,
        inputs=[block_diag_buf, k_diag_vec],
        device=device,
    )

    # Cast K-precision diagonal to solver precision if needed
    if solver_vec_type != k_vec_type:
        solver_diag_vec = wp.empty(n, dtype=solver_vec_type, device=device)
        wp.utils.array_cast(in_array=k_diag_vec, out_array=solver_diag_vec)
        wp.launch(
            inv_kern, dim=n, inputs=[solver_diag_vec, inv_diag, eps], device=device
        )
    else:
        solver_diag_vec = k_diag_vec  # same type, no cast needed
        wp.launch(inv_kern, dim=n, inputs=[k_diag_vec, inv_diag, eps], device=device)

    M = aslinearoperator(inv_diag)

    if _cache is not None:
        _cache["jacobi_M"] = M
        _cache["jacobi_block_diag"] = block_diag_buf
        _cache["jacobi_k_diag_vec"] = k_diag_vec
        _cache["jacobi_solver_diag_vec"] = solver_diag_vec
        _cache["jacobi_inv_diag"] = inv_diag

    return M, {"type": "jacobi"}


# ---------------------------------------------------------------------------
# Condition number estimation
# ---------------------------------------------------------------------------


def estimate_condition_number(
    K: BsrMatrix,
    num_iters: int = 20,
    device: str = "cuda",
) -> float:
    """Estimate condition number ``kappa = |lambda_max / lambda_min|``.

    Uses power iteration for the largest eigenvalue and Jacobi-preconditioned
    approximate inverse iteration for the smallest eigenvalue.

    Adapted from ``estimate_condition_number`` in torch-sla.
    (MIT License, Copyright 2024 walker chi).

    Parameters
    ----------
    K : BsrMatrix
        Symmetric positive-definite warp BSR matrix.
    num_iters : int
        Number of power / inverse iterations.
    device : str
        Warp device string.

    Returns
    -------
    float
        Estimated condition number (>= 1).
    """
    n = K.nrow
    block_dim = K.block_shape[0]
    vec_type = wp.vec3d if block_dim == 3 else wp.vec2d

    _, inv_diag_vec = get_block_diagonal_vec(K, device)

    # Temporary buffers
    w = wp.empty(n, dtype=vec_type, device=device)
    result_buf = wp.zeros(1, dtype=wp.float64, device=device)

    def _rand_unit_vec():
        """Return a random unit block-vector on GPU."""
        v_np = rng.standard_normal((n, block_dim)).astype(np.float64)
        nrm = np.linalg.norm(v_np)
        if nrm > 1e-30:
            v_np /= nrm
        return wp.array(np.ascontiguousarray(v_np), dtype=vec_type, device=device)

    rng = np.random.default_rng(42)

    # --- Power iteration for lambda_max ---
    v = _rand_unit_vec()

    lambda_max = 0.0
    for _ in range(num_iters):
        # w = K * v
        w.zero_()
        bsr_mv(K, v, w, alpha=1.0, beta=0.0)

        # lambda_max = v^T w
        result_buf.zero_()
        wp.launch(_vec_dot_accumulate, dim=n, inputs=[v, w, result_buf], device=device)
        lambda_max = float(result_buf.numpy()[0])

        # v = w / ||w||
        result_buf.zero_()
        wp.launch(_vec_dot_accumulate, dim=n, inputs=[w, w, result_buf], device=device)
        w_norm = math.sqrt(max(float(result_buf.numpy()[0]), 0.0))
        if w_norm < 1e-30:
            break
        wp.launch(
            _vec_scale, dim=n, inputs=[w, wp.float64(1.0 / w_norm), v], device=device
        )

    # --- Approximate inverse iteration for lambda_min ---
    v = _rand_unit_vec()

    lambda_min = float("inf")
    for _ in range(num_iters):
        # w = D_inv * v
        wp.launch(_vec_cw_mul, dim=n, inputs=[inv_diag_vec, v, w], device=device)

        # Normalise w -> v
        result_buf.zero_()
        wp.launch(_vec_dot_accumulate, dim=n, inputs=[w, w, result_buf], device=device)
        w_norm = math.sqrt(max(float(result_buf.numpy()[0]), 0.0))
        if w_norm < 1e-30:
            break
        wp.launch(
            _vec_scale, dim=n, inputs=[w, wp.float64(1.0 / w_norm), v], device=device
        )

        # Rayleigh quotient: lambda_est = v^T K v
        w.zero_()
        bsr_mv(K, v, w, alpha=1.0, beta=0.0)
        result_buf.zero_()
        wp.launch(_vec_dot_accumulate, dim=n, inputs=[v, w, result_buf], device=device)
        lambda_est = float(result_buf.numpy()[0])
        if lambda_est > 0:
            lambda_min = min(lambda_min, lambda_est)

    if lambda_min <= 0 or lambda_min == float("inf"):
        lambda_min = abs(lambda_max) * 1e-10

    kappa = abs(lambda_max / lambda_min) if lambda_min != 0 else 1e15
    return max(kappa, 1.0)


# ---------------------------------------------------------------------------
# Diagonal preconditioner from matvec (for matrix-free CG)
# ---------------------------------------------------------------------------


def build_diagonal_preconditioner_from_matvec(
    matvec_fn,
    n_nodes: int,
    dim: int,
    device: str = "cuda",
    _cache: dict = None,
    dtype=None,
):
    """Build a diagonal preconditioner by probing the matvec operator.

    Extracts the scalar diagonal of the operator via ``dim`` matvec calls
    (one per component direction), then returns the component-wise inverse
    as a ``LinearOperator``.  No full tangent matrix assembly is required.

    Parameters
    ----------
    matvec_fn : callable
        ``matvec(x, y, z, alpha, beta)`` implementing ``z = alpha*A*x + beta*y``.
    n_nodes : int
        Number of block rows (nodes).
    dim : int
        Block dimension (2 or 3).
    device : str
        Warp device string.
    _cache : dict or None
        Optional cache for buffer reuse across Newton iterations.

    Returns
    -------
    M : LinearOperator
        Diagonal preconditioner.
    info : dict
        ``{"type": "diagonal_from_matvec"}``.
    """
    if dtype is None:
        dtype = wp.float64
    k_scalar = dtype
    vec_type = wp.types.vector(length=dim, dtype=k_scalar)
    np_scalar = np.float32 if k_scalar == wp.float32 else np.float64
    wp_device = wp.get_device(device)

    # Reuse cached buffers when possible
    if _cache is not None and "diag_mv_out" in _cache:
        out_buf = _cache["diag_mv_out"]
        zero_buf = _cache["diag_mv_zero"]
    else:
        out_buf = wp.zeros(n_nodes, dtype=vec_type, device=device)
        zero_buf = wp.zeros(n_nodes, dtype=vec_type, device=device)
        if _cache is not None:
            _cache["diag_mv_out"] = out_buf
            _cache["diag_mv_zero"] = zero_buf

    # Extract diagonal via dim matvec calls using unit basis vectors
    diag_np = np.zeros((n_nodes, dim), dtype=np_scalar)
    for d in range(dim):
        basis_np = np.zeros((n_nodes, dim), dtype=np_scalar)
        basis_np[:, d] = 1.0
        e_d = wp.array(np.ascontiguousarray(basis_np), dtype=vec_type, device=device)
        out_buf.zero_()
        zero_buf.zero_()
        matvec_fn(e_d, zero_buf, out_buf, 1.0, 0.0)
        col = out_buf.numpy()
        diag_np[:, d] = col[:, d]

    # Safe inverse
    eps = 1e-30
    inv_np = np.where(np.abs(diag_np) > eps, 1.0 / diag_np, 0.0).astype(np_scalar)
    inv_diag = wp.array(np.ascontiguousarray(inv_np), dtype=vec_type, device=device)
    M = aslinearoperator(inv_diag)

    return M, {"type": "diagonal_from_matvec"}
