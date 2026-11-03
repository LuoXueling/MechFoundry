"""gd_lr_auto: the gradient-descent learning rate sized from the first gradient.

With ``gd_lr_auto`` on, every update steps at gd_lr = fraction * epsilon /
max|dJ/ds| over the free design nodes at that iteration (the geometry clean-up's
iteration-0 scaling, experiments/geometry_cleanup/run.py, with epsilon for h,
applied at each update).  Off -- the default -- the loop is the one it always was.

1. The choice itself, without FEM: the rate, the zero-gradient guard, the
   reference length (epsilon, or gd_lr_auto_length) and the setting checks.
2. A three-iteration ShapeOptTrimmed run on the beam: every update's largest
   nodal move is fraction * epsilon at a rate sized from that iteration's own
   gradient, and a flag-off run at the first rate takes the same first step.

Run:
    conda activate mechfoundry-warp117
    cd src/warp_wrapper && pytest tests/test_gd_lr_auto.py -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.abstract_field import AbstractField  # noqa: E402
from core.default_config import params as default_config  # noqa: E402
from tests.test_phase_field_design import (  # noqa: E402
    CONFIG, _common_overrides, _make_problem_inputs, _octree_mesh_dict,
)

wp.init()


class _Params(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def _fake(params, epsilon=None):
    made = SimpleNamespace(params=_Params(params), logged=[], _gd_lr_auto=None)
    if epsilon is not None:
        made._epsilon = epsilon
    made.log = made.logged.append
    return made


def test_defaults_are_off():
    assert default_config["gd_lr_auto"] is False
    assert default_config["gd_lr_auto_fraction"] == 0.5
    assert "gd_lr_auto_length" not in default_config
    assert AbstractField._gd_lr_auto_setup(_fake({}, epsilon=0.01)) is None


def test_setup_reads_epsilon_or_the_given_length_and_checks_them():
    on = {"gd_lr_auto": True}
    assert AbstractField._gd_lr_auto_setup(_fake(on, epsilon=0.01)) == (0.5, 0.01, "epsilon")
    given = _fake({**on, "gd_lr_auto_fraction": 0.25, "gd_lr_auto_length": 0.2}, epsilon=0.01)
    assert AbstractField._gd_lr_auto_setup(given) == (0.25, 0.2, "gd_lr_auto_length")
    with pytest.raises(ValueError, match="gd_lr_auto_length"):
        AbstractField._gd_lr_auto_setup(_fake(on))           # no epsilon, no length
    for bad in ({"gd_lr_auto_fraction": 0.0}, {"gd_lr_auto_fraction": float("nan")},
                {"gd_lr_auto_length": -1.0}):
        with pytest.raises(ValueError):
            AbstractField._gd_lr_auto_setup(_fake({**on, **bad}, epsilon=0.01))


def test_the_rate_is_sized_from_each_iteration_s_masked_gradient_max():
    made = _fake({})
    setup = (0.5, 0.01, "epsilon")
    first = AbstractField._gd_lr_auto_choose(made, 0, 1e-3, 4.0, setup)
    assert first == pytest.approx(0.5 * 0.01 / 4.0)
    # Sized again from the next iteration's gradient, not kept.
    second = AbstractField._gd_lr_auto_choose(made, 1, first, 2.0, setup)
    assert second == pytest.approx(0.5 * 0.01 / 2.0)
    record = made._gd_lr_auto
    assert record["gd_lr_configured"] == 1e-3
    assert [(s["iteration"], s["gd_lr"], s["applied"]) for s in record["steps"]] == [
        (0, pytest.approx(first), True), (1, pytest.approx(second), True)]
    assert record["gd_lr_effective"] == pytest.approx(second) and record["iteration"] == 1
    assert sum("gd_lr_auto: iteration" in line for line in made.logged) == 2


@pytest.mark.parametrize("gmax", [0.0, float("nan"), float("inf")])
def test_a_zero_or_non_finite_gradient_keeps_the_previous_rate(gmax):
    made = _fake({})
    setup = (0.5, 0.01, "epsilon")
    # At the first update the previous rate is the configured gd_lr ...
    assert AbstractField._gd_lr_auto_choose(made, 0, 1e-3, gmax, setup) == 1e-3
    assert made._gd_lr_auto["applied"] is False
    # ... later, the last rate chosen.
    rate = AbstractField._gd_lr_auto_choose(made, 1, 1e-3, 5.0, setup)
    assert AbstractField._gd_lr_auto_choose(made, 2, rate, gmax, setup) == rate
    assert [s["applied"] for s in made._gd_lr_auto["steps"]] == [False, True, False]
    assert sum("keeping gd_lr" in line for line in made.logged) == 2


def _run(tmp, sub, iters=2, **extra):
    """A short GD run on the beam, recording each update's nodal move."""
    from core.shape_opt import ShapeOptTrimmed

    nodes, elements, sdf, epsilon = _make_problem_inputs()

    class Recorded(ShapeOptTrimmed):
        moves = []

        def _pre_optimizer_step(self, iteration):
            super()._pre_optimizer_step(iteration)
            self._before = wp.to_torch(self.e.dof_values).detach().clone()

        def _post_optimizer_step(self, iteration):
            after = wp.to_torch(self.e.dof_values).detach()
            self.moves.append((after - self._before).abs().max().item())
            super()._post_optimizer_step(iteration)

    Recorded.moves = []
    ov = _common_overrides(tmp + sub, max_optimize_iters=iters, output_per_optimize_step=0,
                           **extra)
    ov["phase_field_epsilon"] = epsilon
    ov["octree_mesh"] = _octree_mesh_dict(nodes, elements, sdf)
    problem = Recorded(CONFIG, override_params=ov)
    problem.optimize()
    return problem, epsilon, list(Recorded.moves)


def test_a_short_run_moves_fraction_epsilon_at_every_step(tmp_path):
    tmp = str(tmp_path)
    auto, epsilon, moves = _run(tmp, "/auto", iters=3, gd_lr_auto=True,
                                gd_lr_auto_fraction=0.5)
    chosen = auto._gd_lr_auto
    steps = auto._optimize_steps
    assert chosen["length"] == pytest.approx(epsilon) and len(chosen["steps"]) == 3
    for step, picked, move in zip(steps, chosen["steps"], moves):
        # Each rate from that iteration's own masked gradient max ...
        assert picked["applied"]
        assert step["gd_lr"] == pytest.approx(0.5 * epsilon / step["max_abs_grad"])
        assert picked["gd_lr"] == pytest.approx(step["gd_lr"])
        # ... so every step's largest move is exactly half epsilon.
        assert step["max_move"] == pytest.approx(0.5 * epsilon)
        assert move == pytest.approx(0.5 * epsilon, rel=1e-6)
    assert chosen["gd_lr_effective"] == pytest.approx(steps[-1]["gd_lr"])

    # Off, at the first rate: the same first step (the option only picks the
    # rate).  Not bit-equal: two runs of the same code already differ by about
    # 1e-7 relative in |dJ/ds| (the GPU AMG solve is not deterministic).
    off, _, off_moves = _run(tmp, "/off", iters=1, gd_lr=steps[0]["gd_lr"])
    assert off._gd_lr_auto is None and "gd_lr" not in off._optimize_steps[0]
    assert off_moves[0] == pytest.approx(moves[0], rel=1e-5)
    assert off._optimize_steps[0]["max_abs_grad"] == pytest.approx(
        steps[0]["max_abs_grad"], rel=1e-5)
