"""The shape-optimization volume penalty is a CAP at f V_0 for every target fraction f.

``ShapeOpt._objective_function`` adds ``w_vol * softplus_k((V/V_0 - f) / |1 - f|)``.
The denominator used to be the SIGNED iteration-0 value of ``V/V_0 - f`` (``1 - f``),
which for f > 1 is negative and turns the one-sided hinge into a lower bound: the
volume was pushed up to f V_0 and never capped.  These tests evaluate the real
objective on a small embedded beam and read the penalty off it,
``penalty = (J - SE/SE_0) / w_vol`` with SE/SE_0 = 1 on an unchanged design.  A
different volume is presented to the SAME solve by rescaling the stored 1/V_0
(``_initial_objective_values["volume_raw"]``), which is exactly how the objective
forms V/V_0, so no remesh is needed.

Run:
    conda activate mechfoundry-warp117
    cd src/warp_wrapper && python -m pytest tests/test_volume_cap.py -v
"""

import math
import sys
from pathlib import Path

import pytest
import warp as wp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_phase_field_design import (  # noqa: E402
    CONFIG, _common_overrides, _make_problem_inputs, _octree_mesh_dict,
)

wp.init()

W_VOL = 10.0
K = 500.0


def _softplus(x):
    return math.log1p(math.exp(-abs(K * x))) / K + max(x, 0.0)


@pytest.fixture(scope="module")
def penalty_at(tmp_path_factory):
    """penalty_at(f, ratio): the objective's volume penalty for target fraction f
    when the design's volume is ``ratio`` x its starting volume."""
    from core.shape_opt import ShapeOptTrimmed

    nodes, elements, sdf, epsilon = _make_problem_inputs()
    solvers = {}

    def solver(f):
        if f not in solvers:
            ov = _common_overrides(str(tmp_path_factory.mktemp(f"f{f}")),
                                   target_volume_fraction=f, shape_opt_w_vol=W_VOL)
            ov["phase_field_epsilon"] = epsilon
            ov["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)
            s = ShapeOptTrimmed(CONFIG, override_params=ov)
            tape = wp.Tape()
            s.e.dof_values.requires_grad = True
            s.step(tape=tape)
            with tape:
                s._objective_function(tape)       # iteration 0: fixes SE_0, V_0
            inner = s._inner
            solvers[f] = (s, float(inner._initial_objective_values["volume_raw"].numpy()[0]))
        return solvers[f]

    def at(f, ratio):
        s, inv_v0 = solver(f)
        vals = s._inner._initial_objective_values
        vals["volume_raw"] = wp.array([inv_v0 * ratio], dtype=vals["volume_raw"].dtype,
                                      device=vals["volume_raw"].device)
        try:
            total = float(s._objective_function(None).numpy()[0])
        finally:
            vals["volume_raw"] = wp.array([inv_v0], dtype=vals["volume_raw"].dtype,
                                          device=vals["volume_raw"].device)
        return (total - 1.0) / W_VOL

    return at


def test_target_above_one_is_a_cap(penalty_at):
    f = 1.5
    # at the start and anywhere below f V_0: free to grow
    assert penalty_at(f, 1.0) < 1e-6
    assert penalty_at(f, 1.4) < 1e-6
    # above f V_0: penalised, linearly in (V/V_0 - f)/|1 - f|
    p = penalty_at(f, 1.6)
    assert p == pytest.approx(_softplus((1.6 - f) / abs(1.0 - f)), rel=1e-4)
    assert p > 0.1
    assert penalty_at(f, 1.8) > p


def test_target_below_one_unchanged(penalty_at):
    f = 0.5
    # the signed and the absolute normalisation coincide for f < 1
    for ratio in (1.0, 0.7, 0.45):
        assert penalty_at(f, ratio) == pytest.approx(
            _softplus((ratio - f) / (1.0 - f)), rel=1e-4, abs=1e-6)
    assert penalty_at(f, 1.0) > 0.9          # the start is above the cap
    assert penalty_at(f, 0.45) < 1e-6        # below it, free
