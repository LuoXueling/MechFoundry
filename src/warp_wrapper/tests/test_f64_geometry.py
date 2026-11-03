"""Float64 warp.fem integration via native ``fem.Quadmesh2D``.

Verifies that:
  - Quadmesh2D (float64 vertex positions) + f64 field + dispatch assembly
    achieves ≥13 significant decimal places for ∫(div u)·v dΩ.
  - Float32 Grid2D baseline gives ~7-8 significant decimal places.

Warp infers FEM precision from the geometry position dtype.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
import warp.fem as fem


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

wp.init()
DEVICE = "cuda"


def _make_quad_mesh_f64(nx, ny, wx, wy, device):
    xs = np.linspace(0.0, wx, nx + 1, dtype=np.float64)
    ys = np.linspace(0.0, wy, ny + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    positions = np.column_stack((xx.ravel(), yy.ravel()))
    quads = []
    for i in range(nx):
        for j in range(ny):
            n00 = i * (ny + 1) + j
            n10 = (i + 1) * (ny + 1) + j
            quads.append((n00, n10, n10 + 1, n00 + 1))
    return (
        wp.array(positions, dtype=wp.vec2d, device=device),
        wp.array(np.asarray(quads, dtype=np.int32), ndim=2, device=device),
        positions,
    )


@fem.integrand
def _linear_div(s: fem.Sample, u: fem.Field, v: fem.Field):
    """∫ (∂u_x/∂x + ∂u_y/∂y) * v_x dΩ  — analytic = 2.0 for u=(x,y)."""
    g = fem.grad(u, s)
    w = fem.inner(v, s)
    return (g[0, 0] + g[1, 1]) * w[0]


def _sig_digits(result: float, analytic: float) -> float:
    rel_err = abs(result - analytic) / (abs(analytic) + 1e-300)
    return -math.log10(rel_err) if rel_err > 0 else 16.0


def test_f64_quadmesh_precision():
    """Native float64 Quadmesh2D assembly achieves at least 13 digits."""
    nx, ny = 8, 8
    pos_wp, quad_wp, pos_np = _make_quad_mesh_f64(nx, ny, 1.0, 1.0, DEVICE)

    geo = fem.Quadmesh2D(quad_wp, pos_wp)
    space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2d)
    domain = fem.Cells(geo)

    u = space.make_field()
    u.dof_values = wp.array(
        np.ascontiguousarray(pos_np, dtype=np.float64), dtype=wp.vec2d, device=DEVICE
    )
    test_fn = fem.make_test(space=space, domain=domain)

    R = fem.integrate(
        _linear_div,
        fields={"u": u, "v": test_fn},
        accumulate_dtype=wp.float64,
        output_dtype=wp.vec2d,
    )
    result = float(R.numpy().sum())
    sd = _sig_digits(result, 2.0)
    print(f"\n  [f64+Quadmesh2D] result={result:.15f}, sig_digits≈{sd:.1f}")
    assert (
        sd >= 13
    ), f"Expected ≥13 sig digits with native float64 Quadmesh2D, got {sd:.1f}"
    return sd


def test_f32_comparison():
    """Float32 Grid2D baseline — should be ~7-8 sig digits."""
    nx, ny = 8, 8
    geo32 = fem.Grid2D(
        res=wp.vec2i(nx, ny),
        bounds_lo=wp.vec2(0.0, 0.0),
        bounds_hi=wp.vec2(1.0, 1.0),
    )
    space32 = fem.make_polynomial_space(geo32, degree=1, dtype=wp.vec2)
    domain32 = fem.Cells(geo32)

    u32 = space32.make_field()
    node_pos32 = space32.node_positions().numpy()
    u32.dof_values = wp.array(
        np.ascontiguousarray(node_pos32, dtype=np.float32), dtype=wp.vec2, device=DEVICE
    )
    test32 = fem.make_test(space=space32, domain=domain32)

    R32 = fem.integrate(
        _linear_div,
        fields={"u": u32, "v": test32},
        output_dtype=wp.vec2,
    )
    result32 = float(R32.numpy().sum())
    sd32 = _sig_digits(result32, 2.0)
    print(f"\n  [f32+Grid2D] result={result32:.15f}, sig_digits≈{sd32:.1f}")
    return sd32
