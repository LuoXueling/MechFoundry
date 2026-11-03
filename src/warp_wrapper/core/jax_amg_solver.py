"""JAX-AMG integration for Warp block sparse matrices.

The finite-element code stores matrices in Warp BSR form.  JAX-AMG accepts a
scalar CSR matrix plus ``block_dim``, so this module expands the sparse pattern
once per structure and gathers updated coefficients on the GPU for each solve.
Matrix values, right-hand sides, and solutions cross the Warp/JAX boundary via
DLPack and therefore never stage through host memory.
"""

from __future__ import annotations

import math
from copy import deepcopy

import warp as wp
from warp.sparse import BsrMatrix


# A reused hierarchy that needs this many Krylov iterations has drifted too far
# from the current tangent.  Rebuild before the following solve.
REBUILD_THRESHOLD = 500


# This is intentionally independent from ``amgx_solver``'s pyamgx settings.
# In f64, JAX-AMG converts our scalar CSR input back to BSR internally.  In f32
# we deliberately retain scalar CSR: JAX-AMG's block conversion otherwise
# copies, hashes, sorts, and collapses nine scalar entries per 3-D FEM block on
# every changed sparsity pattern.  Scalar aggregation avoids that cold-remesh
# bottleneck while retaining the empirically stable SIZE_8 / 0.5 setup.
_JAXAMG_BLOCK_PCG_CONFIG = {
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "max_iters": 1000,
        "convergence": "RELATIVE_INI",
        "tolerance": 1.0e-6,
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
            "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.5},
            "presweeps": 1,
            "postsweeps": 1,
            "max_levels": 100,
            "min_coarse_rows": 32,
            "dense_lu_num_rows": 64,
            "coarse_solver": "DENSE_LU_SOLVER",
        },
    },
}


def jaxamg_config(
    block_dim: int, *, use_mixed_precision: bool = True, max_iters: int | None = None
) -> dict:
    """Return a per-solve JAX-AMG block-system configuration.

    Float32 has a practical linear-residual floor well above the double
    precision configuration.  Stopping at ``5e-6`` avoids spending many PCG
    iterations below the accuracy available to the f32 FEM path, while the
    f64 configuration retains the historical ``1e-6`` tolerance. A deep copy
    prevents one solve's precision choice from mutating another's config.
    ``max_iters`` replaces the Krylov iteration cap (1000); solvers pass their
    ``jaxamg_max_iters`` param here.
    """
    if block_dim <= 1:
        raise ValueError("JAX-AMG's Warp adapter expects a block matrix")
    config = deepcopy(_JAXAMG_BLOCK_PCG_CONFIG)
    if block_dim == 2:
        config["solver"]["solver"] = "PBICGSTAB"
        preconditioner = config["solver"]["preconditioner"]
        preconditioner["selector"] = "SIZE_4"
        preconditioner["smoother"]["relaxation_factor"] = 0.8
    if not use_mixed_precision:
        config["solver"]["tolerance"] = 5.0e-6
        if block_dim == 2:
            config["solver"]["tolerance"] = 2.0e-4
    if max_iters is not None:
        # A solve reported "still converging ... ran out of iterations" needs
        # more of them, not a different setup: the rotated sharp-volume twist
        # cube at depth 6 (adaptive octree, float32) stops 11x short at 1000
        # and converges well inside 4000.
        config["solver"]["max_iters"] = int(max_iters)
    return config


# How far above the tolerance a stalled residual sits.  A Krylov method in
# float32 cannot push the residual below roughly the condition number times the
# machine epsilon, and stops moving once it arrives there; the same matrix then
# plateaus at a different level from one run to the next.  That is a different
# failure from a solve still making progress when it ran out of iterations, and
# the difference is the difference between raising max_iters and fixing the
# conditioning -- so the message says which one it is looking at.
_STALL_FACTOR = 100.0


def config_for_load(config: dict, rhs, block_dim: int) -> dict:
    """The config to iterate this right-hand side against: the given one, unless
    some of its block components carry no load.

    AMGX keeps a residual norm per block component and stops only when every
    component is ``tolerance`` below its own initial value.  A component the
    load leaves at rounding noise -- x and z under a vertical load, where a
    stool (7cf4533ae018) had ||b_x|| / ||b|| = 5e-11 -- is then held to machine
    zero, and AMGX iterates to ``max_iters`` over a solution that converged
    hundreds of iterations earlier.  For such a right-hand side AMGX judges the
    whole residual instead (``use_scalar_norm``), with the tolerance tightened
    to the share of the least loaded real component, so every component that
    carries load still meets ``tolerance`` of its own.  A component is load when
    its share of ||b|| is above ``tolerance``, and noise when it is above zero
    but not above that: its own relative target is then under ``tolerance**2``
    of the whole.  Every other right-hand side -- all components loaded, or
    the rest exactly zero -- keeps AMGX's per-component test, which is the
    stricter one where it can be met (the 2-D affine patch test depends on it).

    The tightened tolerance is rounded down to a power of ten: jaxamg caches a
    solver and its hierarchy per config string, so a tolerance for every solve
    would pay a cold setup on each.  A load along one axis keeps the tolerance
    itself.
    """
    import jax.numpy as jnp

    if block_dim <= 1:
        return config
    tolerance = float(config["solver"]["tolerance"])
    norms = [float(value) for value in
             jnp.linalg.norm(jnp.reshape(rhs, (-1, block_dim)), axis=0)]
    total = math.sqrt(sum(value * value for value in norms))
    if total == 0.0:
        return config
    loaded = [value / total for value in norms if value > tolerance * total]
    # A component with no load at all is already fine: AMGX counts a zero
    # initial residual as met.  Switching it anyway only moved the first use of
    # the default config to a later Newton iteration, and with it the hierarchy
    # a marginal float32 solve (two_d_right_boundary_force) started from.
    noise = [value for value in norms if 0.0 < value <= tolerance * total]
    if not loaded or not noise:
        return config
    tuned = deepcopy(config)
    tuned["solver"]["use_scalar_norm"] = 1
    tuned["solver"]["tolerance"] = tolerance * 10.0 ** math.floor(
        math.log10(min(loaded)) + 1.0e-12)
    return tuned


def _relative_residual(matrix, solution, rhs) -> float:
    """``||b - K x|| / ||b||`` over the whole vector: what the tolerance is meant to bound."""
    import jax.numpy as jnp

    rhs_norm = float(jnp.linalg.norm(rhs))
    residual = float(jnp.linalg.norm(matrix @ solution - rhs))
    return residual / rhs_norm if rhs_norm > 0.0 else residual


def _solve_failure_message(info, config, *, use_mixed_precision: bool,
                           scaled: bool, relative: float | None = None) -> str:
    """What failed, and -- when the shape of the failure says so -- what to try.

    ``relative`` is the whole-vector ``||b - K x|| / ||b||`` of the solution
    AMGX returned, which is what the relative tolerance bounds.  AMGX's own
    ``info["residual"]`` is an absolute norm of the first block component only
    (``solve_jaxamg`` says why), and comparing it with the tolerance gave advice
    at random.  Without ``relative`` the reported residual is taken as it is.
    """
    solver = config["solver"]
    tolerance = max(float(solver["tolerance"]), 1.0e-300)
    iterations = int(info["iterations"])
    if relative is None:
        residual = float(info["residual"])
        headline = (
            f"JAX-AMG solve failed with status {info['status']} after "
            f"{iterations} iterations (residual={residual:.3e})"
        )
    else:
        residual = float(relative)
        headline = (
            f"JAX-AMG solve failed with status {info['status']} after "
            f"{iterations} iterations (||b - Kx||/||b||={residual:.3e})"
        )
    if iterations < int(solver["max_iters"]) or residual <= tolerance:
        return headline
    if residual < tolerance * _STALL_FACTOR:
        return (
            f"{headline}; it was still converging and ran out of iterations "
            f"{residual / tolerance:.0f}x short of the {tolerance:.1e} "
            f"tolerance, so raise the solver's max_iters"
        )
    remedies = []
    if not scaled:
        remedies.append(
            "scale_system_to_unit_diagonal=True, an equivalent system that "
            "costs almost nothing (the tolerance then applies to the scaled "
            "residual)"
        )
    remedies.append(
        "beta_max_over_e, to cap a Nitsche penalty that sliver boundary "
        "elements drive far above E"
    )
    if not use_mixed_precision:
        remedies.append("use_mixed_precision=True, to assemble in float64")
    return (
        f"{headline}; it stalled {residual / tolerance:.0e}x above the "
        f"{tolerance:.1e} tolerance, which is the rounding floor of an "
        f"ill-conditioned system"
        + (" in float32" if not use_mixed_precision else "")
        + ", not a tolerance set too tight -- try " + ", then ".join(remedies)
    )


@wp.kernel(enable_backward=False)
def _gather_values_f32(
    source: wp.array(dtype=wp.float32),
    mapping: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    result[i] = source[mapping[i]]


@wp.kernel(enable_backward=False)
def _gather_values_f64(
    source: wp.array(dtype=wp.float64),
    mapping: wp.array(dtype=wp.int32),
    result: wp.array(dtype=wp.float64),
):
    i = wp.tid()
    result[i] = source[mapping[i]]


@wp.kernel(enable_backward=False)
def _expand_bsr_pattern(
    block_offsets: wp.array(dtype=wp.int32),
    block_columns: wp.array(dtype=wp.int32),
    scalar_offsets: wp.array(dtype=wp.int32),
    scalar_columns: wp.array(dtype=wp.int32),
    value_mapping: wp.array(dtype=wp.int32),
    n_block_rows: int,
    n_block_entries: int,
    block_dim: int,
):
    """Expand dense BSR blocks to scalar CSR entirely on the GPU."""
    block_row = wp.tid()
    if block_row == n_block_rows:
        scalar_offsets[n_block_rows * block_dim] = (
            n_block_entries * block_dim * block_dim
        )
        return

    first_entry = block_offsets[block_row]
    entries_in_row = block_offsets[block_row + 1] - first_entry
    scalar_base = first_entry * block_dim * block_dim
    for local_row in range(block_dim):
        scalar_row = block_row * block_dim + local_row
        row_base = scalar_base + local_row * entries_in_row * block_dim
        scalar_offsets[scalar_row] = row_base
        for entry_offset in range(entries_in_row):
            block_entry = first_entry + entry_offset
            for local_column in range(block_dim):
                output = row_base + entry_offset * block_dim + local_column
                scalar_columns[output] = (
                    block_columns[block_entry] * block_dim + local_column
                )
                value_mapping[output] = (
                    block_entry * block_dim * block_dim
                    + local_row * block_dim
                    + local_column
                )


def jaxamg_availability() -> tuple[bool, str | None]:
    """Return whether the JAX-AMG CUDA backend can be loaded."""
    try:
        import jax
        import jaxamg  # noqa: F401

        if not jax.devices("gpu"):
            return False, "JAX has no GPU device"
        # jaxamg itself imports lazily; loading this module validates that the
        # native extension and its AMGX/CUDA shared libraries are discoverable.
        from jaxamg._ext import _amgx  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def require_jaxamg() -> None:
    available, reason = jaxamg_availability()
    if not available:
        raise ImportError(
            "The 'jaxamg' backend was requested but JAX-AMG is unavailable"
            + (f": {reason}" if reason else ".")
        )


def resolve_amgx_backend(requested: str) -> str:
    """Resolve an assembled backend name without masking solve failures."""
    backend = str(requested).lower()
    if backend == "amgx":
        available, _ = jaxamg_availability()
        return "jaxamg" if available else "pyamgx"
    if backend in {"native", "jaxamg", "pyamgx"}:
        return backend
    raise ValueError(
        f"unknown assembled CG backend {backend!r} "
        "(expected 'native', 'amgx', 'jaxamg', or 'pyamgx')"
    )


def _scalar_type(dtype):
    return getattr(dtype, "_wp_scalar_type_", dtype)


def _vector_type(block_dim: int, scalar_type):
    if block_dim == 1:
        return scalar_type
    aliases = {
        (2, wp.float32): wp.vec2,
        (2, wp.float64): wp.vec2d,
        (3, wp.float32): wp.vec3,
        (3, wp.float64): wp.vec3d,
        (4, wp.float32): wp.vec4,
        (4, wp.float64): wp.vec4d,
    }
    alias = aliases.get((block_dim, scalar_type))
    if alias is not None:
        return alias
    return wp.types.vector(length=block_dim, dtype=scalar_type)


def _prepare_csr(K: BsrMatrix, scalar_type, cache: dict, device):
    K.nnz_sync()
    source_key = (
        K.offsets.ptr,
        K.columns.ptr,
        K.nrow,
        K.ncol,
        K.nnz,
        K.block_shape,
    )
    if cache.get("jaxamg_source_pattern_key") != source_key:
        cache["jaxamg_source_pattern_key"] = source_key
        # Row indices for unit-diagonal scaling follow the pattern, not the values.
        cache.pop("unit_diagonal_rows", None)
        block_dim = K.block_shape[0]
        scalar_nnz = K.nnz * block_dim * block_dim
        if (
            cache.get("jaxamg_pattern_shape")
            != (K.nrow, K.ncol, K.nnz, block_dim)
        ):
            cache["jaxamg_pattern_shape"] = (K.nrow, K.ncol, K.nnz, block_dim)
            cache["jaxamg_offsets"] = wp.empty(
                K.nrow * block_dim + 1, dtype=wp.int32, device=device
            )
            cache["jaxamg_columns"] = wp.empty(
                scalar_nnz, dtype=wp.int32, device=device
            )
            cache["jaxamg_mapping"] = wp.empty(
                scalar_nnz, dtype=wp.int32, device=device
            )
        wp.launch(
            _expand_bsr_pattern,
            dim=K.nrow + 1,
            inputs=[
                K.offsets,
                K.columns,
                cache["jaxamg_offsets"],
                cache["jaxamg_columns"],
                cache["jaxamg_mapping"],
                K.nrow,
                K.nnz,
                block_dim,
            ],
            device=device,
        )

    mapping_wp = cache["jaxamg_mapping"]
    source_scalar = _scalar_type(K.values.dtype)
    source = wp.array(
        ptr=K.values.ptr,
        dtype=source_scalar,
        shape=(K.nnz * K.block_shape[0] * K.block_shape[1],),
        device=K.values.device,
        copy=False,
    )
    native_key = f"jaxamg_values_native_{source_scalar.__name__}"
    native = cache.get(native_key)
    if native is None or len(native) != len(mapping_wp):
        native = wp.empty(len(mapping_wp), dtype=source_scalar, device=device)
        cache[native_key] = native
    kernel = _gather_values_f32 if source_scalar == wp.float32 else _gather_values_f64
    wp.launch(
        kernel,
        dim=len(mapping_wp),
        inputs=[source, mapping_wp, native],
        device=device,
    )

    if source_scalar == scalar_type:
        values = native
    else:
        value_key = f"jaxamg_values_{scalar_type.__name__}"
        values = cache.get(value_key)
        if values is None or len(values) != len(mapping_wp):
            values = wp.empty(len(mapping_wp), dtype=scalar_type, device=device)
            cache[value_key] = values
        wp.utils.array_cast(native, values)

    return cache["jaxamg_offsets"], cache["jaxamg_columns"], values


def _cast_rhs(b, block_dim: int, scalar_type, cache: dict, device):
    target_dtype = _vector_type(block_dim, scalar_type)
    if b.dtype == target_dtype:
        return b
    key = f"jaxamg_rhs_{scalar_type.__name__}_{block_dim}"
    result = cache.get(key)
    if result is None or len(result) != len(b):
        result = wp.empty(len(b), dtype=target_dtype, device=device)
        cache[key] = result
    wp.utils.array_cast(b, result)
    return result


def _scale_to_unit_diagonal(values_jax, columns_jax, rhs_jax, rows):
    """Symmetric Jacobi scaling: solve ``(S K S) u_hat = S b`` with ``S=diag(K)^-1/2``.

    Nitsche penalties make row magnitudes span several thousand times the elastic
    terms, so in single precision the elastic contribution is lost and the Krylov
    solve stalls.  Scaling to a unit diagonal removes that spread and makes the
    scaled unknown order one -- the same effect as non-dimensionalising the
    unknown, which a change of physical units cannot achieve because it rescales
    every term equally.  The scaling is symmetric, so the system stays symmetric
    positive definite and ``u = S u_hat`` solves the original system exactly
    when ``u_hat`` solves the scaled one.  The solve is still iterative: the
    stopping tolerance applies to the scaled residual, not the original one.
    """
    import jax.numpy as jnp

    diagonal = jnp.zeros(rhs_jax.shape[0], dtype=values_jax.dtype).at[rows].add(
        jnp.where(rows == columns_jax, values_jax, 0.0)
    )
    # A zero or negative diagonal cannot be scaled; leave those rows untouched.
    scale = jnp.where(diagonal > 0.0, 1.0 / jnp.sqrt(jnp.abs(diagonal)), 1.0)
    scaled_values = values_jax * scale[rows] * scale[columns_jax]
    return scaled_values, rhs_jax * scale, scale


def solve_jaxamg(
    K: BsrMatrix,
    b,
    *,
    use_mixed_precision: bool,
    cache: dict | None,
    device,
    rebuild_every_iter: bool = False,
    scale_to_unit_diagonal: bool = False,
    max_iters: int | None = None,
):
    """Solve a Warp BSR system through JAX-AMG and return a Warp array."""
    require_jaxamg()
    import jax
    import jax.experimental.sparse as jsp
    import jaxamg

    if use_mixed_precision:
        jax.config.update("jax_enable_x64", True)
    scalar_type = wp.float64 if use_mixed_precision else wp.float32
    block_dim = K.block_shape[0]
    work_cache = cache if cache is not None else {}
    wp_device = wp.get_device(device)

    offsets, columns, values = _prepare_csr(K, scalar_type, work_cache, wp_device)
    rhs = _cast_rhs(b, block_dim, scalar_type, work_cache, wp_device)
    values_jax = wp.to_jax(values)
    columns_jax = wp.to_jax(columns)
    offsets_jax = wp.to_jax(offsets)
    rhs_scalar = wp.array(
        ptr=rhs.ptr,
        dtype=scalar_type,
        shape=(len(rhs) * block_dim,),
        device=rhs.device,
        copy=False,
    )
    rhs_jax = wp.to_jax(rhs_scalar)
    scale = None
    if scale_to_unit_diagonal:
        import jax.numpy as jnp

        # Building row indices is the one non-trivial cost here, and it depends
        # only on the sparsity pattern, so it is reused across Newton iterations
        # and load steps until the pattern itself changes.
        rows = work_cache.get("unit_diagonal_rows")
        if rows is None or rows.shape[0] != values_jax.shape[0]:
            rows = jnp.repeat(
                jnp.arange(offsets_jax.shape[0] - 1, dtype=columns_jax.dtype),
                jnp.diff(offsets_jax),
                total_repeat_length=values_jax.shape[0],
            )
            work_cache["unit_diagonal_rows"] = rows
        values_jax, rhs_jax, scale = _scale_to_unit_diagonal(
            values_jax, columns_jax, rhs_jax, rows
        )
    matrix = jsp.BCSR(
        (values_jax, columns_jax, offsets_jax),
        shape=(K.nrow * block_dim, K.ncol * block_dim),
    )

    # A fresh field/mesh cache requests one resetup so an identical sparsity
    # pattern cannot inherit coefficients from an unrelated problem through
    # JAX-AMG's process-global native cache.  A genuine pattern miss is already
    # a cold setup, so reuse_setup has no effect in that case.
    force_rebuild = bool(work_cache.pop("jaxamg_force_rebuild", False))
    reuse_setup = not (rebuild_every_iter or force_rebuild)
    config = jaxamg_config(
        block_dim, use_mixed_precision=use_mixed_precision, max_iters=max_iters
    )
    # JAX-AMG's native block path reconstructs BSR from the expanded scalar
    # CSR on every new topology.  At block_dim=3 this means a GPU-to-host copy,
    # hash, and CPU sort of nine times the block nnz before AMGX can set up.
    # Its scalar aggregation path consumes the existing CSR directly and is
    # substantially faster in f32 3-D.  Keep f64 and 2-D on native blocks;
    # scalar aggregation is not robust for the small 2-D traction systems.
    amgx_block_dim = 1 if not use_mixed_precision and block_dim >= 3 else block_dim
    # What AMGX iterates against: the base config, or -- for a load that
    # leaves some components at noise -- one that judges the whole residual.
    iterated = config_for_load(config, rhs_jax, amgx_block_dim)
    solution, info = jaxamg.solve(
        matrix,
        rhs_jax,
        config=iterated,
        block_dim=amgx_block_dim,
        reuse_setup=reuse_setup,
    )
    solution.block_until_ready()
    # Keep the hierarchy across Newton iterations and load/time steps whenever
    # the mesh pattern is unchanged.  Iteration count alone is a poor rebuild
    # signal near an f32 residual floor and previously caused the last solve of
    # one step to make the first solve of every following step pay setup cost.
    # If a lagged hierarchy genuinely stops converging, rebuild once and retry.
    #
    # AMGX's verdict is checked against the solution before it is believed.
    # ``config_for_load`` keeps a component the load leaves at noise out of the
    # per-component test, but a lightly loaded one can still ask for its own
    # residual below the rounding floor: AMGX then reports NOT_CONVERGED over
    # a solution that meets the tolerance.  A rejected solution is kept when
    # its whole-vector residual does -- one SpMV, paid only then -- and judged
    # against the base tolerance, which is what the caller asked for.
    tolerance = float(config["solver"]["tolerance"])
    relative = None
    if info["status"] != jaxamg.AMGXStatus.SUCCESS:
        relative = _relative_residual(matrix, solution, rhs_jax)
    if relative is not None and relative > tolerance and reuse_setup:
        solution, info = jaxamg.solve(
            matrix,
            rhs_jax,
            config=iterated,
            block_dim=amgx_block_dim,
            reuse_setup=False,
        )
        solution.block_until_ready()
        relative = (
            None if info["status"] == jaxamg.AMGXStatus.SUCCESS
            else _relative_residual(matrix, solution, rhs_jax)
        )
    if relative is not None and relative > tolerance:
        raise RuntimeError(
            _solve_failure_message(
                info,
                config,
                use_mixed_precision=use_mixed_precision,
                scaled=scale is not None,
                relative=relative,
            )
        )
    if info["iterations"] > REBUILD_THRESHOLD:
        work_cache["jaxamg_force_rebuild"] = True
    if scale is not None:
        solution = solution * scale
    target_dtype = _vector_type(block_dim, scalar_type)
    shaped_solution = (
        solution if block_dim == 1 else solution.reshape((-1, block_dim))
    )
    solution_view = wp.from_jax(shaped_solution, dtype=target_dtype)
    key = f"jaxamg_solution_{scalar_type.__name__}_{block_dim}"
    result = work_cache.get(key)
    if result is None or len(result) != len(b):
        result = wp.empty(len(b), dtype=target_dtype, device=wp_device)
        work_cache[key] = result
    wp.copy(result, solution_view)
    return result, int(info["iterations"]), float(info["residual"])
