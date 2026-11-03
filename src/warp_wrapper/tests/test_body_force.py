"""The body-force term, which nothing exercised until it was asked for.

`has_body` assembles this term only when the body force is nonzero, and every
run in the workflow so far had it at zero -- so the integrand was never compiled
and a defect in it waited for the first user who asked for gravity.  It failed
at Warp codegen with "Invalid external reference type: <class 'tuple'>", because
the integrand closed over a Python tuple.  Fixed, and covered here only in
float32, it then failed again for the first user who ran in double precision --
"Input types must be exactly the same, got ['vec3d', 'vec3f']" -- because the
constant was spelled `wp.vec3`.  These tests compile it and check it does the
work it claims, in both dimensions and both precisions.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import warp as wp
import warp.fem as fem

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.abstract_field import AbstractField

wp.init()


# Both precisions the solver can run in.  `use_mixed_precision` selects float64
# -- see AbstractField.__init__ -- and testing only the float32 side is what let
# a float32-pinned constant reach a user: it compiled here and failed there with
# "Input types must be exactly the same, got ['vec3d', 'vec3f']".
PRECISIONS = (wp.float32, wp.float64)


def _integrate(dim, body_force, scalar_dtype=wp.float32):
    """Assemble b . v over a unit grid, in the dimension and precision asked."""
    integrand = AbstractField._make_body_force_integrand(
        dim, body_force, scalar_dtype=scalar_dtype
    )
    # The grid carries the precision too, and via `scalar_type` rather than via
    # its bounds, exactly as AbstractField._build_mesh_state passes it: a double
    # space over a float32 grid still samples to vec3f, so the mismatch under
    # test would be hidden behind the grid rather than exercised.
    vec_type = wp.types.vector(length=dim, dtype=scalar_dtype)
    zero = vec_type(*([0.0] * dim))
    one = vec_type(*([1.0] * dim))
    if dim == 3:
        geometry = fem.Grid3D(
            res=wp.vec3i(2, 2, 2), bounds_lo=zero, bounds_hi=one,
            scalar_type=scalar_dtype,
        )
    else:
        geometry = fem.Grid2D(
            res=wp.vec2i(2, 2), bounds_lo=zero, bounds_hi=one,
            scalar_type=scalar_dtype,
        )
    space = fem.make_polynomial_space(geometry, degree=1, dtype=vec_type)
    test = fem.make_test(space=space, domain=fem.Cells(geometry))
    return fem.integrate(
        integrand, fields={"v": test}, domain=fem.Cells(geometry),
        output_dtype=wp.float64,
    ).numpy()


@pytest.mark.parametrize("scalar_dtype", PRECISIONS)
@pytest.mark.parametrize("dim", (2, 3))
def test_a_nonzero_body_force_compiles_and_loads_the_domain(dim, scalar_dtype):
    """The compile is the regression; the sum is the physics.

    Over a unit domain, the assembled load sums to the body force itself --
    every test function's share of b, added back up.
    """
    b = (0.0, -490.0, 0.0)[:dim]
    loaded = _integrate(dim, b, scalar_dtype)
    assert np.isfinite(loaded).all()
    total = loaded.reshape(-1, dim).sum(axis=0)
    assert total == pytest.approx(np.asarray(b), rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("scalar_dtype", PRECISIONS)
@pytest.mark.parametrize("dim", (2, 3))
def test_the_load_is_linear_in_the_body_force(dim, scalar_dtype):
    """Doubling the body force doubles the load, and zero loads nothing."""
    b = (1.0, -2.0, 3.0)[:dim]
    once = _integrate(dim, b, scalar_dtype)
    twice = _integrate(dim, tuple(2.0 * value for value in b), scalar_dtype)
    assert twice == pytest.approx(2.0 * once, rel=1e-6, abs=1e-12)
    assert _integrate(dim, (0.0,) * dim, scalar_dtype) == pytest.approx(
        np.zeros_like(once), abs=1e-12
    )


def test_a_short_body_force_is_padded_not_rejected():
    """A two-component value in three dimensions leaves z unloaded.

    The default config carries a 2D body force while the workflow runs in 3D,
    so this padding is on the live path rather than a curiosity.
    """
    loaded = _integrate(3, (1.0, -2.0))
    assert np.isfinite(loaded).all()
    total = loaded.reshape(-1, 3).sum(axis=0)
    assert total == pytest.approx(np.asarray([1.0, -2.0, 0.0]), rel=1e-6, abs=1e-9)
