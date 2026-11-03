"""Tests verifying that Newton residuals reach < 1e-10 in mixed precision mode.

Precision layout summary
========================
The modes (`use_mixed_precision=True/False`) deliberately use different data
types for key quantities.  In particular, the JAX-AMG non-mixed path is f32
end to end, whereas pyamgx must retain its supported double-precision path:

  Quantity              | non-mixed               | mixed
  ----------------------|-------------------------|----------------------------
  Geometry              | Grid2D / Grid3D (f32)   | Grid2D / Grid3D (f64)
  Displacement dtype    | vec2 / vec3 (f32)       | vec2d / vec3d (f64)
  Design (e) dtype      | float32                 | float64
  Integrand scalar      | wp.float32              | wp.float64
  Residual R output     | vec2d / vec3d (f64)     | vec2d / vec3d (f64)
  R accumulate_dtype    | wp.float64              | wp.float64
  R assembly            | dispatch (default)      | dispatch (default)
  K block type          | mat22 / mat33 (f32)     | f32, or f64 with tangent_f64
  K accumulate_dtype    | wp.float32              | f32, or f64 with tangent_f64
  K assembly            | dispatch (default)      | dispatch (default)
  BC projector          | mat22 / mat33 (f32)     | mat22 / mat33 (f32)
  b = -R dtype          | vec2d / vec3d (f64)     | vec2d / vec3d (f64)
  JAX-AMG conversion    | all f32                 | all f64 (upcast K if needed)
  JAX-AMG mode          | dFFI                     | dDDI
  pyamgx fallback       | dDDI (upcast K and b)    | dDDI
  Native CG (b cast)    | b cast f64 → f32        | b cast f64 → f32

JAX-AMG supports the desired all-float32 dFFI solve when mixed precision is
disabled. It receives float32 CSR coefficients, RHS, solver vectors, and emits
a float32 update. Mixed mode uses dDDI and promotes a float32 tangent when
``tangent_f64`` is disabled. pyamgx remains on dDDI in both modes: dDFI is not
implemented in AMGX 2.5.0, and its dFFI path is not reliable for these
nonlinear systems. Warp 1.17 provides the FEM float32/float64 types natively.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.neoHookean import NeoHookean


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TIGHT_OVERRIDES = dict(
    relative_tol=1e-11,
    absolute_tol=1e-12,
    max_iter=30,
    linear_solver="cg",
    cg_backend_assembled="amgx",
    ew_enabled=False,
    line_search_enabled=False,
    update_damping_limit=0.0,
    load_steps=1,
)

# Non-mixed precision uses float32 geometry, displacement, residual assembly,
# and JAX-AMG vectors. Its 3D JAX-AMG tolerance is 5e-6 (pyamgx does not
# provide an f32 path), so Newton tolerances must respect the f32 residual floor.
_NONMIXED_OVERRIDES = {
    **_TIGHT_OVERRIDES,
    "relative_tol": 1e-10,
    "absolute_tol": 1e-10,
}


def _run_mixed(config_path, tmp_path):
    overrides = {
        "output_root": str(tmp_path),
        "use_mixed_precision": True,
        **_TIGHT_OVERRIDES,
    }
    solver = NeoHookean(config_path, override_params=overrides)
    solver.forward()
    return solver


def _check_k_dtype(solver, expected_block_type, label):
    K = solver._mesh_cache.get("K") or solver._mesh_cache.get("K_extended")
    assert K is not None, f"[{label}] K not found in mesh_cache"
    actual = K.values.dtype
    actual_params = getattr(actual, "_wp_type_params_", None)
    expected_params = getattr(expected_block_type, "_wp_type_params_", None)
    assert actual_params == expected_params, (
        f"[{label}] K block dtype mismatch: got {actual}, "
        f"expected {expected_block_type}"
    )


def _check_r_dtype(solver, expected_vec_type, label):
    # R is transient, but u.dof_values reflects the displacement dtype:
    # vec2d / vec3d in mixed, vec2 / vec3 in non-mixed.
    actual = solver.u.dof_values.dtype
    assert (
        actual == expected_vec_type
    ), f"[{label}] displacement field dtype mismatch: got {actual}, expected {expected_vec_type}"


# ---------------------------------------------------------------------------
# 2D: mixed precision, force-driven Neumann BC
# ---------------------------------------------------------------------------


def test_2d_mixed_precision_residual_below_1e10(tmp_path):
    """2D mixed precision: Newton relative residual < 1e-10.

    Uses the two_d_right_boundary_force example (small mesh, Neumann load)
    with use_mixed_precision=True and tight tolerances.
    """
    config = "tests.examples.two_d_right_boundary_force.config"
    # Use a smaller mesh to keep the test fast
    overrides = {
        "output_root": str(tmp_path),
        "use_mixed_precision": True,
        "nx": 16,
        "ny": 6,
        **_TIGHT_OVERRIDES,
    }
    solver = NeoHookean(config, override_params=overrides)
    solver.forward()

    rel = solver.final_relative_residual
    assert (
        rel < 1e-10
    ), f"2D mixed precision: relative residual {rel:.3e} did not reach < 1e-10"

    # K blocks must be float32 (mat22)
    _check_k_dtype(solver, wp.mat22, "2D mixed")

    # Displacement field must be vec2d (float64) in mixed mode
    _check_r_dtype(solver, wp.vec2d, "2D mixed")


# ---------------------------------------------------------------------------
# 2D: non-mixed precision, same problem
# ---------------------------------------------------------------------------


def test_2d_nonmixed_precision_residual_below_1e9(tmp_path):
    """2D non-mixed precision: Newton absolute residual < 1e-9.

    Uses the two_d_affine_tension example (full Dirichlet BCs, exact affine
    solution) with a 2×2 mesh (1 interior node, 2 free DOFs).  Float32
    displacement arithmetic limits the Newton residual floor to ~5e-10, which
    is consistent with float32 eps × displacement magnitude.  The test checks
    < 1e-9 to stay within the float32 precision budget.

    For comparison, the mixed-precision test (float64 u) achieves < 1e-10 on a
    larger Neumann problem; the key difference here is that K is still float32
    (mat22 blocks) in non-mixed mode — same assembly path as in mixed mode.
    """
    config = "tests.examples.two_d_affine_tension.config"
    overrides = {
        "output_root": str(tmp_path),
        "use_mixed_precision": False,
        "nx": 2,
        "ny": 2,
        **_NONMIXED_OVERRIDES,
        # Float32 displacement floor: relax absolute tolerance to 1e-9.
        "absolute_tol": 1e-9,
    }
    solver = NeoHookean(config, override_params=overrides)
    solver.forward()

    res = solver.final_residual
    assert res < 1e-9, f"2D non-mixed: absolute residual {res:.3e} did not reach < 1e-9"

    # K blocks are float32 (mat22) in non-mixed mode — same as mixed, since
    # dispatch assembly ignores accumulate_dtype for local_result buffers.
    _check_k_dtype(solver, wp.mat22, "2D non-mixed")

    # Displacement field must be vec2 (float32) in non-mixed mode
    _check_r_dtype(solver, wp.vec2, "2D non-mixed")


# ---------------------------------------------------------------------------
# 3D: mixed precision, affine deformation (exact solution applied via BCs)
# ---------------------------------------------------------------------------


def test_3d_mixed_precision_residual_below_1e10(tmp_path):
    """3D mixed precision: Newton absolute residual < 1e-10.

    Uses the three_d_affine_tension example (full Dirichlet BCs, exact affine
    solution) so Newton converges in ~1 iteration with negligible residual.
    """
    config = "tests.examples.three_d_affine_tension.config"
    overrides = {
        "output_root": str(tmp_path),
        "use_mixed_precision": True,
        **_TIGHT_OVERRIDES,
    }
    solver = NeoHookean(config, override_params=overrides)
    solver.forward()

    res = solver.final_residual
    assert (
        res < 1e-10
    ), f"3D mixed precision: absolute residual {res:.3e} did not reach < 1e-10"

    # K blocks must be float32 (mat33)
    _check_k_dtype(solver, wp.mat33, "3D mixed")

    # Displacement field must be vec3d (float64) in mixed mode
    _check_r_dtype(solver, wp.vec3d, "3D mixed")


# ---------------------------------------------------------------------------
# 3D: non-mixed precision, same problem
# ---------------------------------------------------------------------------


def test_3d_nonmixed_precision_residual_below_1e9(tmp_path):
    """3D native-f32 residual assembly reaches its expected 1e-9 floor."""
    config = "tests.examples.three_d_affine_tension.config"
    overrides = {
        "output_root": str(tmp_path),
        "use_mixed_precision": False,
        **_NONMIXED_OVERRIDES,
        "absolute_tol": 1e-9,
    }
    solver = NeoHookean(config, override_params=overrides)
    solver.forward()

    res = solver.final_residual
    assert (
        res < 1e-9
    ), f"3D non-mixed: absolute residual {res:.3e} did not reach < 1e-9"

    # K blocks are float32 (mat33) in non-mixed mode — same as mixed.
    _check_k_dtype(solver, wp.mat33, "3D non-mixed")
    _check_r_dtype(solver, wp.vec3, "3D non-mixed")
    assert solver._mesh_cache["R_restrict_buf"].dtype._wp_scalar_type_ == wp.float32
    assert solver.free_mask_wp.dtype._wp_scalar_type_ == wp.float32


# ---------------------------------------------------------------------------
# AMGX mode verification
# ---------------------------------------------------------------------------


def test_amgx_precision_configurations():
    """JAX-AMG has f32 configs while pyamgx's production path stays dDDI."""
    from core.amgx_solver import select_amgx_config

    assert select_amgx_config(2, mode="dFFI") == "amg_vcycle"
    assert select_amgx_config(3, mode="dFFI") == "amg_vcycle_3d"
    assert select_amgx_config(2, mode="dDDI") == "amg_vcycle"
    assert select_amgx_config(3, mode="dDDI") == "amg_vcycle_3d"
