"""Focused tests for JAX-AMG selection and Warp BSR interoperability."""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from scipy.sparse import csr_matrix

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core import jax_amg_solver
from core.amgx_solver import _AMGX_PCG_AMG_3D_CFG
from core.jax_amg_solver import _prepare_csr, resolve_amgx_backend, solve_jaxamg
from tests.test_preconditioners import _build_3d_laplacian_bsr

wp.init()
DEVICE = "cuda"


def _require_jaxamg():
    available, reason = jax_amg_solver.jaxamg_availability()
    if not available:
        pytest.skip(f"JAX-AMG unavailable: {reason}")


def test_backend_resolution_prefers_jaxamg(monkeypatch):
    monkeypatch.setattr(jax_amg_solver, "jaxamg_availability", lambda: (True, None))
    assert resolve_amgx_backend("amgx") == "jaxamg"
    assert resolve_amgx_backend("JAXAMG") == "jaxamg"
    assert resolve_amgx_backend("pyamgx") == "pyamgx"


def test_backend_resolution_falls_back_only_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        jax_amg_solver,
        "jaxamg_availability",
        lambda: (False, "native extension missing"),
    )
    assert resolve_amgx_backend("amgx") == "pyamgx"
    with pytest.raises(ValueError, match="unknown assembled CG backend"):
        resolve_amgx_backend("invalid")


def test_jaxamg_block_config_is_independent_and_has_safe_coarse_floor():
    """JAX-AMG tuning must not mutate or alias the legacy pyamgx config."""
    config = jax_amg_solver.jaxamg_config(3)
    preconditioner = config["solver"]["preconditioner"]
    assert config is not _AMGX_PCG_AMG_3D_CFG
    assert preconditioner["algorithm"] == "AGGREGATION"
    assert preconditioner["selector"] == "SIZE_8"
    assert preconditioner["min_coarse_rows"] == 32
    assert preconditioner["dense_lu_num_rows"] == 64

    config_2d = jax_amg_solver.jaxamg_config(2)
    assert config_2d["solver"]["solver"] == "PBICGSTAB"
    preconditioner_2d = config_2d["solver"]["preconditioner"]
    assert preconditioner_2d["selector"] == "SIZE_4"
    assert preconditioner_2d["smoother"]["relaxation_factor"] == 0.8
    assert jax_amg_solver.jaxamg_config(2, use_mixed_precision=False)["solver"][
        "tolerance"
    ] == 2.0e-4


def test_jaxamg_f32_config_uses_its_precision_appropriate_tolerance():
    """f32 must not inherit the unnecessarily strict legacy dDDI tolerance."""
    f32 = jax_amg_solver.jaxamg_config(3, use_mixed_precision=False)
    f64 = jax_amg_solver.jaxamg_config(3, use_mixed_precision=True)
    assert f32 is not f64
    assert f32["solver"]["tolerance"] == 5.0e-6
    assert f64["solver"]["tolerance"] == 1.0e-6
    assert f32["solver"]["preconditioner"] == f64["solver"]["preconditioner"]


def _y_loaded_system(n=32):
    """A 3-D block system loaded along y, its x and z loads at rounding noise."""
    K = _build_3d_laplacian_bsr(n, device=DEVICE)
    rhs = np.zeros((n, 3))
    rhs[:, 1] = 1.0
    rhs[:, 0] = rhs[:, 2] = 1.0e-9 * np.random.default_rng(0).standard_normal(n)
    return K, rhs, wp.array(rhs, dtype=wp.vec3d, device=DEVICE)


def _verdict(monkeypatch, status, *, zero_solution=False):
    """Have AMGX return ``status`` whatever it found (optionally with no answer)."""
    import jax.numpy as jnp
    import jaxamg

    solve = jaxamg.solve

    def overridden(*args, **kwargs):
        x, info = solve(*args, **kwargs)
        return (jnp.zeros_like(x) if zero_solution else x), {**info, "status": status}

    monkeypatch.setattr(jaxamg, "solve", overridden)


def test_an_unconverged_verdict_is_checked_against_the_solution(monkeypatch):
    """AMGX judges each block component against its own initial residual.

    A vertically loaded stool (7cf4533ae018) had ||b|| = 31 and ||b_x|| = 1.6e-9,
    so the x residual was held to machine zero and AMGX said NOT_CONVERGED of
    an answer accurate to 2.6e-11.  The answer is what is checked now.
    """
    _require_jaxamg()
    import jaxamg

    _verdict(monkeypatch, jaxamg.AMGXStatus.NOT_CONVERGED)
    K, rhs, b = _y_loaded_system()
    x, _, _ = solve_jaxamg(K, b, use_mixed_precision=True, cache={}, device=DEVICE)
    solved = x.numpy()
    # The Laplacian blocks are 6I on the diagonal, -I beside it: check K x = b.
    applied = 6.0 * solved
    applied[1:] -= solved[:-1]
    applied[:-1] -= solved[1:]
    assert np.linalg.norm(applied - rhs) / np.linalg.norm(rhs) < 1.0e-6


def test_an_unconverged_verdict_on_a_wrong_solution_still_fails(monkeypatch):
    _require_jaxamg()
    import jaxamg

    _verdict(monkeypatch, jaxamg.AMGXStatus.NOT_CONVERGED, zero_solution=True)
    K, _, b = _y_loaded_system()
    with pytest.raises(RuntimeError, match=r"\|\|b - Kx\|\|/\|\|b\|\|=1\.000e\+00"):
        solve_jaxamg(K, b, use_mixed_precision=True, cache={}, device=DEVICE)


def test_a_load_that_leaves_components_at_noise_is_iterated_on_the_whole_residual():
    """AMGX stops when the solution is done, not when a noise component reaches zero."""
    _require_jaxamg()
    import jax.numpy as jnp

    base = jax_amg_solver.jaxamg_config(3)

    def config(*shares):
        rhs = np.outer(np.linspace(1.0, 2.0, 50), shares).reshape(-1)
        return jax_amg_solver.config_for_load(base, jnp.asarray(rhs), 3)

    vertical = config(1.0e-9, 1.0, 1.0e-9)
    assert vertical["solver"]["use_scalar_norm"] == 1
    assert vertical["solver"]["tolerance"] == 1.0e-6
    # x carries 29 % of ||b||: the whole residual is held to 1e-7 so x still gets 1e-6.
    assert config(0.3, 1.0, 1.0e-12)["solver"]["tolerance"] == pytest.approx(1.0e-7)
    # Every component loaded, none, or the rest exactly zero: AMGX's own test,
    # which counts a zero initial residual as met.
    assert config(1.0, 1.0, 1.0) is base and config(0.0, 0.0, 0.0) is base
    assert config(0.0, 1.0, 0.0) is base
    assert "use_scalar_norm" not in base["solver"]
    assert jax_amg_solver.config_for_load(base, jnp.ones(30), 1) is base


def test_a_vertical_load_stops_on_its_whole_residual(monkeypatch):
    _require_jaxamg()
    import jaxamg

    recorded = []
    solve = jaxamg.solve

    def recording(*args, **kwargs):
        x, info = solve(*args, **kwargs)
        recorded.append(info)
        return x, info

    monkeypatch.setattr(jaxamg, "solve", recording)
    K, rhs, b = _y_loaded_system()
    solve_jaxamg(K, b, use_mixed_precision=True, cache={}, device=DEVICE)
    # The initial residual AMGX divides by is ||b||, not the noise in b_x.
    assert float(recorded[-1]["residual_history"][0]) == pytest.approx(
        np.linalg.norm(rhs), rel=1.0e-6)


def test_a_failure_is_described_by_its_whole_relative_residual():
    """AMGX's reported residual is one component's absolute norm; the tolerance is relative."""
    config = jax_amg_solver.jaxamg_config(3, use_mixed_precision=True)
    max_iters = int(config["solver"]["max_iters"])

    def message(relative):
        return jax_amg_solver._solve_failure_message(
            {"status": "AMGXStatus.NOT_CONVERGED", "iterations": max_iters,
             "residual": 1.0e-14},
            config, use_mixed_precision=True, scaled=False, relative=relative,
        )

    short = message(3.0e-6)
    assert "||b - Kx||/||b||=3.000e-06" in short and "3x short" in short
    assert "stalled 9e+05x above" in message(0.9)


@pytest.mark.parametrize("target_type", [wp.float32, wp.float64])
def test_bsr_expansion_matches_scipy(target_type):
    _require_jaxamg()
    K = _build_3d_laplacian_bsr(8, device=DEVICE)
    offsets, columns, values = _prepare_csr(K, target_type, {}, wp.get_device(DEVICE))
    expanded = csr_matrix(
        (values.numpy(), columns.numpy(), offsets.numpy()),
        shape=(K.nrow * 3, K.ncol * 3),
    ).toarray()

    blocks = K.values.numpy()[: K.nnz]
    expected = np.zeros_like(expanded, dtype=blocks.dtype)
    block_offsets = K.offsets.numpy()
    block_columns = K.columns.numpy()
    for row in range(K.nrow):
        for entry in range(block_offsets[row], block_offsets[row + 1]):
            col = block_columns[entry]
            expected[row * 3 : (row + 1) * 3, col * 3 : (col + 1) * 3] = blocks[
                entry
            ]
    np.testing.assert_allclose(expanded, expected, rtol=0.0, atol=0.0)
    assert values.dtype == target_type


@pytest.mark.parametrize(
    ("mixed", "expected_dtype", "expected_block_dim", "rtol"),
    [(False, wp.vec3, 1, 2.0e-4), (True, wp.vec3d, 3, 1.0e-10)],
)
def test_jaxamg_solve_precision_and_residual(
    mixed, expected_dtype, expected_block_dim, rtol
):
    _require_jaxamg()
    import jaxamg

    n = 32
    K = _build_3d_laplacian_bsr(n, device=DEVICE)
    b = wp.array(
        np.ones((n, 3), dtype=np.float64), dtype=wp.vec3d, device=DEVICE
    )
    x, iterations, _ = solve_jaxamg(
        K,
        b,
        use_mixed_precision=mixed,
        cache={},
        device=DEVICE,
    )
    assert x.dtype == expected_dtype
    assert 0 < iterations < 1000
    expected_mode = "float64" if mixed else "float32"
    cache_entries = jaxamg.get_solver_cache_info()["single_gpu"]["entries"]
    assert any(
        entry["mode"] == expected_mode
        and entry["block_dim"] == expected_block_dim
        for entry in cache_entries
    )

    x_np = x.numpy().reshape(-1).astype(np.float64)
    b_np = b.numpy().reshape(-1)
    block_offsets = K.offsets.numpy()
    block_columns = K.columns.numpy()
    blocks = K.values.numpy()[: K.nnz]
    data = []
    indices = []
    indptr = [0]
    for row in range(K.nrow):
        for local_row in range(3):
            for entry in range(block_offsets[row], block_offsets[row + 1]):
                for local_col in range(3):
                    data.append(blocks[entry, local_row, local_col])
                    indices.append(block_columns[entry] * 3 + local_col)
            indptr.append(len(data))
    A = csr_matrix((data, indices, indptr), shape=(n * 3, n * 3))
    relative_residual = np.linalg.norm(A @ x_np - b_np) / np.linalg.norm(b_np)
    assert relative_residual < rtol


def test_jaxamg_cache_handles_coefficient_and_pattern_changes():
    _require_jaxamg()
    cache = {}
    b = wp.array(np.ones((20, 3)), dtype=wp.vec3d, device=DEVICE)
    K1 = _build_3d_laplacian_bsr(20, diag_value=6.0, device=DEVICE)
    K2 = _build_3d_laplacian_bsr(20, diag_value=12.0, device=DEVICE)
    x1, _, _ = solve_jaxamg(
        K1, b, use_mixed_precision=False, cache=cache, device=DEVICE
    )
    x1_np = x1.numpy().copy()
    pattern_shape1 = cache["jaxamg_pattern_shape"]
    x2, _, _ = solve_jaxamg(
        K2, b, use_mixed_precision=False, cache=cache, device=DEVICE
    )
    assert cache["jaxamg_pattern_shape"] == pattern_shape1
    assert not np.allclose(x1_np, x2.numpy())

    K3 = _build_3d_laplacian_bsr(21, device=DEVICE)
    b3 = wp.array(np.ones((21, 3)), dtype=wp.vec3d, device=DEVICE)
    solve_jaxamg(K3, b3, use_mixed_precision=False, cache=cache, device=DEVICE)
    assert cache["jaxamg_pattern_shape"] != pattern_shape1


def test_unit_diagonal_scaling_is_exact_and_collapses_the_spread():
    """Symmetric Jacobi scaling: exact, and it collapses a beta-like spread."""
    import numpy as np
    import pytest

    jnp = pytest.importorskip("jax.numpy")
    from core.jax_amg_solver import _scale_to_unit_diagonal

    rng = np.random.default_rng(0)
    size = 120
    dense = rng.normal(size=(size, size))
    matrix = dense @ dense.T + size * np.eye(size)
    # A handful of rows carrying a Nitsche-sized penalty.
    penalty = np.ones(size)
    penalty[:12] = 5000.0
    matrix = matrix * penalty[:, None] * penalty[None, :]
    expected = rng.normal(size=size)
    rhs = matrix @ expected

    rows = np.repeat(np.arange(size), size)
    columns = np.tile(np.arange(size), size)
    scaled_values, scaled_rhs, scale = _scale_to_unit_diagonal(
        jnp.asarray(matrix.reshape(-1)), jnp.asarray(columns),
        jnp.asarray(rhs), jnp.asarray(rows),
    )
    scaled = np.asarray(scaled_values, dtype=np.float64).reshape(size, size)

    # Unit diagonal, and conditioning improved by orders of magnitude.
    np.testing.assert_allclose(scaled.diagonal(), 1.0, rtol=1e-6)
    assert np.linalg.cond(scaled) < np.linalg.cond(matrix) / 1.0e5
    # S K S is an exact congruence, so a direct solve of it recovers u exactly.
    recovered = np.linalg.solve(scaled, np.asarray(scaled_rhs, dtype=np.float64))
    recovered = recovered * np.asarray(scale, dtype=np.float64)
    np.testing.assert_allclose(recovered, expected, rtol=1e-6)
    # Symmetry is preserved, so the system stays solvable by CG.
    np.testing.assert_allclose(scaled, scaled.T, rtol=1e-6, atol=1e-12)


def test_unit_diagonal_scaling_leaves_nonpositive_diagonal_rows_alone():
    """A zero diagonal cannot be scaled; those rows must pass through unchanged."""
    import numpy as np
    import pytest

    jnp = pytest.importorskip("jax.numpy")
    from core.jax_amg_solver import _scale_to_unit_diagonal

    matrix = np.array([[4.0, 1.0], [1.0, 0.0]])
    rows = np.repeat(np.arange(2), 2)
    columns = np.tile(np.arange(2), 2)
    _, _, scale = _scale_to_unit_diagonal(
        jnp.asarray(matrix.reshape(-1)), jnp.asarray(columns),
        jnp.asarray(np.ones(2)), jnp.asarray(rows),
    )
    np.testing.assert_allclose(np.asarray(scale), [0.5, 1.0], rtol=1e-6)
