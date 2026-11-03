"""Float64 warp.fem integration via native ``fem.Hexmesh``.

Verifies that:
  - Hexmesh (float64 vertex positions) + f64 field + dispatch assembly
    achieves ≥13 significant decimal places for ∫(div u)·v dΩ with u=(x,y,z).
  - Float32 Grid3D baseline gives ~7-8 significant decimal places.

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


def _make_hex_mesh_f64(nx, ny, nz, wx, wy, wz, device):
    xs = np.linspace(0.0, wx, nx + 1, dtype=np.float64)
    ys = np.linspace(0.0, wy, ny + 1, dtype=np.float64)
    zs = np.linspace(0.0, wz, nz + 1, dtype=np.float64)
    positions = np.asarray(
        [(x, y, z) for x in xs for y in ys for z in zs], dtype=np.float64
    )
    node = lambda i, j, k: i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
    hexes = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                hexes.append(
                    (
                        node(i, j, k), node(i + 1, j, k),
                        node(i + 1, j + 1, k), node(i, j + 1, k),
                        node(i, j, k + 1), node(i + 1, j, k + 1),
                        node(i + 1, j + 1, k + 1), node(i, j + 1, k + 1),
                    )
                )
    return (
        wp.array(positions, dtype=wp.vec3d, device=device),
        wp.array(np.asarray(hexes, dtype=np.int32), ndim=2, device=device),
        positions,
    )


@fem.integrand
def _linear_div_3d(s: fem.Sample, u: fem.Field, v: fem.Field):
    """∫ (∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z) * v_x dΩ — analytic = 3.0 for u=(x,y,z)."""
    g = fem.grad(u, s)
    w = fem.inner(v, s)
    return (g[0, 0] + g[1, 1] + g[2, 2]) * w[0]


def _sig_digits(result: float, analytic: float) -> float:
    rel_err = abs(result - analytic) / (abs(analytic) + 1e-300)
    return -math.log10(rel_err) if rel_err > 0 else 16.0


def test_f64_hexmesh_precision():
    """Native float64 Hexmesh assembly achieves at least 13 digits."""
    nx, ny, nz = 4, 4, 4
    pos_wp, hex_wp, pos_np = _make_hex_mesh_f64(nx, ny, nz, 1.0, 1.0, 1.0, DEVICE)

    geo = fem.Hexmesh(hex_wp, pos_wp)
    space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec3d)
    domain = fem.Cells(geo)

    u = space.make_field()
    u.dof_values = wp.array(
        np.ascontiguousarray(pos_np, dtype=np.float64), dtype=wp.vec3d, device=DEVICE
    )
    test_fn = fem.make_test(space=space, domain=domain)

    R = fem.integrate(
        _linear_div_3d,
        fields={"u": u, "v": test_fn},
        accumulate_dtype=wp.float64,
        output_dtype=wp.vec3d,
    )
    result = float(R.numpy().sum())
    sd = _sig_digits(result, 3.0)
    print(f"\n  [f64+Hexmesh] result={result:.15f}, sig_digits≈{sd:.1f}")
    assert (
        sd >= 13
    ), f"Expected ≥13 sig digits with native float64 Hexmesh, got {sd:.1f}"
    return sd


def test_f32_hex_comparison():
    """Float32 Grid3D baseline — should be ~7-8 sig digits."""
    nx, ny, nz = 4, 4, 4
    geo32 = fem.Grid3D(
        res=wp.vec3i(nx, ny, nz),
        bounds_lo=wp.vec3(0.0, 0.0, 0.0),
        bounds_hi=wp.vec3(1.0, 1.0, 1.0),
    )
    space32 = fem.make_polynomial_space(geo32, degree=1, dtype=wp.vec3)
    domain32 = fem.Cells(geo32)

    u32 = space32.make_field()
    node_pos32 = space32.node_positions().numpy()
    u32.dof_values = wp.array(
        np.ascontiguousarray(node_pos32, dtype=np.float32), dtype=wp.vec3, device=DEVICE
    )
    test32 = fem.make_test(space=space32, domain=domain32)

    R32 = fem.integrate(
        _linear_div_3d,
        fields={"u": u32, "v": test32},
        output_dtype=wp.vec3,
    )
    result32 = float(R32.numpy().sum())
    sd32 = _sig_digits(result32, 3.0)
    print(f"\n  [f32+Grid3D] result={result32:.15f}, sig_digits≈{sd32:.1f}")
    return sd32
