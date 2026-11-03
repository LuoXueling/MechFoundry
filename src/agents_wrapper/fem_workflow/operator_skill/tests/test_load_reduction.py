"""The runner's load reduction after a partly converged load-stepped solve: the converged fraction
read from the failed solve's log, offered in the failure's next with its scale, copied into
plan.load_reduction (checked against the offers), applied to every load the values card sends,
its first run not counted, and named in the verdict as its own marked simplification (not a load
cut after the body).  No GPU, no API, no graph run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402

NOT_CONVERGED = ("ToolFailed: run_forward failed: RuntimeError: Newton solver did not converge in 20 iterations. "
                 "Final residual: 6.435e-04")
REJECTED = "ToolFailed: run_forward failed: RuntimeError: Line search failed to reduce residual; Newton step rejected."


def _log(steps: int, started: int, optimization: bool = False) -> str:
    """A solve log in run-forward's format whose load stepping started ``started`` of ``steps``."""
    lines = ["[10-13-2026,19:12:36.265] Workflow effective config: relative_tol=1.000e-02, load_steps="
             f"{steps}, run_mode=forward"]
    if optimization:
        lines.append("[10-13-2026,19:12:36.265] ===== Optimization Iteration 0 =====")
    for k in range(1, started + 1):
        lines.append(f"[10-13-2026,19:12:36.265] ===== Load step {k}/{steps}: scale={k / steps:.6f} =====")
        for i in range(3):
            lines.append(f"[10-13-2026,19:12:37.000]   scale={k / steps:.4f} Iter {i}: ||R|| = 1.0e-02, rel = 1.0e-01")
    return "\n".join(lines) + "\n"


def _session(tmp_path: Path, steps: int = 10, started: int = 9, run: str = "api_fem_forward-1") -> Path:
    session = tmp_path / "abc"
    (session / "solve" / run).mkdir(parents=True)
    (session / "solve" / run / "log.txt").write_text(_log(steps, started))
    lib.write_json(session / "solve" / "failed.json", {"log": f"/elsewhere/abc/solve/{run}/log.txt"})
    lib.write_json(lib.operator_dir(session) / "plan.json", {"regions": "Fix the base.", "settings": {"load_steps": steps}})
    return session


def test_the_log_gives_the_last_converged_load_scale(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text(_log(10, 9))
    assert lib.load_step_progress(path) == {"steps": 10, "converged": 8, "last_converged_scale": 0.8,
                                            "failed_at_scale": 0.9}
    path.write_text(_log(5, 2))
    assert lib.load_step_progress(path)["last_converged_scale"] == 0.2
    for text in (_log(5, 1), _log(1, 0), _log(10, 9, optimization=True)):
        path.write_text(text)                  # no step converged, no load steps, an optimization
        assert lib.load_step_progress(path) is None


def test_a_partly_converged_solve_names_the_offered_scale_in_its_next(tmp_path):
    session = _session(tmp_path)
    nxt = drive.solve_failure_next(session, {}, NOT_CONVERGED, {"solve": 1})
    assert "converged 8 of 10 steps, through load scale 0.8" in nxt and "failed at 0.9" in nxt
    assert "plan.load_reduction: 0.8 (copy it)" in nxt and "the Use table rows stay as they are" in nxt
    assert "not counted against the solve limit" in nxt
    record = lib.read_json(lib.operator_dir(session) / drive.LOAD_REDUCTION)
    assert [o["scale"] for o in record["offers"]] == [0.8]
    # a singular linear solve takes its supports row alone; a step that never converged offers nothing
    singular = ("RuntimeError: Linear solver failed at Newton iteration 0: JAX-AMG solve failed (||b - Kx||/||b||"
                "=2.4e+01)")
    assert "load_reduction" not in drive.solve_failure_next(session, {}, singular)
    first = _session(tmp_path / "other", steps=5, started=1)
    assert "load_reduction" not in drive.solve_failure_next(first, {}, REJECTED)


def test_the_plan_takes_only_an_offered_scale_and_the_values_card_scales_every_load(tmp_path):
    session = _session(tmp_path)
    assert drive.load_reduction_problem(session, 0.8) is not None          # nothing offered yet
    drive.solve_failure_next(session, {}, NOT_CONVERGED)
    assert drive.load_reduction_problem(session, 0.8) is None
    assert "offered: 0.8" in drive.load_reduction_problem(session, 0.7)    # a number of its own is refused
    assert drive.load_reduction_problem(session, True) is not None
    assert drive.load_reduction_problem(session, None) is None
    card = {"kind": "bc_values", "dirichlet": [{"name": "base"}], "neumann": [{"name": "top"}]}
    plan = {"values": {"dirichlet_values": {"base": {"components": [1], "value": 0.0}},
                       "neumann_values": {"top": {"force": [0.0, -25.0, 0.0]}}, "acceleration": [0, -9.80665, 0]},
            "contacts": {"rest": {"regions": ["base"]}}}
    full, _, _ = drive.plan_answer("bc_values", card, plan)
    reduced, _, _ = drive.plan_answer("bc_values", card, {**plan, "load_reduction": 0.8})
    assert full["neumann_values"]["top"] == {"force": [0.0, -25.0, 0.0]}
    assert reduced["neumann_values"]["top"]["force"][1] == -20.0
    assert abs(reduced["acceleration"][1] + 0.8 * 9.80665) < 1e-9
    assert reduced["dirichlet_values"] == full["dirichlet_values"]
    # the values card's section: copying the reduction after it was sent rewinds there
    assert ("load_reduction", "bc_values") in drive.SECTION_CARDS and "load_reduction" in drive.PLAN_KEYS


def test_the_first_run_at_the_offer_is_free_once_and_only_with_the_case_unchanged(tmp_path):
    session = _session(tmp_path)
    drive.solve_failure_next(session, {}, NOT_CONVERGED)
    plan = lib.read_json(lib.operator_dir(session) / "plan.json")
    assert not drive.reduction_retry(session, plan)                        # no reduction set
    lib.write_json(lib.operator_dir(session) / "plan.json", {**plan, "regions": "Fix the top."})
    assert not drive.reduction_retry(session, {**plan, "regions": "Fix the top.", "load_reduction": 0.8})
    lib.write_json(lib.operator_dir(session) / "plan.json", {**plan, "load_reduction": 0.8,
                                                             "settings": {"load_steps": 20}})
    assert not drive.reduction_retry(session, {**plan, "load_reduction": 0.8})          # other settings: counted
    lib.write_json(lib.operator_dir(session) / "plan.json", {**plan, "load_reduction": 0.8})
    assert drive.reduction_retry(session, {**plan, "load_reduction": 0.8})
    assert not drive.reduction_retry(session, {**plan, "load_reduction": 0.8})          # once
    assert "not counted" not in drive.solve_failure_next(session, {}, NOT_CONVERGED)


def test_a_failed_run_at_a_reduced_load_offers_the_fraction_of_that_load(tmp_path):
    session = _session(tmp_path)
    drive.solve_failure_next(session, {}, NOT_CONVERGED)
    drive.set_applied_load_reduction(session, 0.8)
    (session / "solve" / "api_fem_forward-2").mkdir()
    (session / "solve" / "api_fem_forward-2" / "log.txt").write_text(_log(10, 8))
    lib.write_json(session / "solve" / "failed.json", {"log": "/x/solve/api_fem_forward-2/log.txt"})
    nxt = drive.solve_failure_next(session, {}, REJECTED)
    assert "through load scale 0.7 of the 0.8 load it ran at" in nxt and "plan.load_reduction: 0.56 " in nxt
    # only the newest offer takes the free run
    plan = lib.read_json(lib.operator_dir(session) / "plan.json")
    assert not drive.reduction_retry(session, {**plan, "load_reduction": 0.8})
    assert drive.reduction_retry(session, {**plan, "load_reduction": 0.56})


def test_no_offer_under_half_the_brief_load_or_from_a_log_older_than_the_load_sent(tmp_path):
    session = _session(tmp_path, steps=10, started=5)          # converged through 0.4
    assert "load_reduction" not in drive.solve_failure_next(session, {}, NOT_CONVERGED)
    session = _session(tmp_path / "b")
    record = lib.operator_dir(session) / drive.LOAD_REDUCTION
    import time
    lib.write_json(record, {"applied": 0.9, "applied_time": time.time() + 100})     # sent after this log
    assert "load_reduction" not in drive.solve_failure_next(session, {}, NOT_CONVERGED)


def test_no_offer_beside_a_fix_the_failure_names(tmp_path, monkeypatch):
    session = _session(tmp_path)
    remedy = NOT_CONVERGED + " -- try scale_system_to_unit_diagonal=True, then use_mixed_precision=True"
    nxt = drive.solve_failure_next(session, {"material": {"mode": "uniform"}}, remedy)
    assert "plan.settings.use_mixed_precision: true" in nxt and "load_reduction" not in nxt
    scaled = NOT_CONVERGED + " (the residual of the Jacobi-scaled system)"
    assert "load_reduction" not in drive.solve_failure_next(session, {}, scaled)
    monkeypatch.setattr(drive, "solve_triage", lambda session, plan: [
        "solve triage: the holds only stop rigid motion. The one change: hold the loaded part"])
    nxt = drive.solve_failure_next(session, {}, NOT_CONVERGED)
    assert "hold the loaded part" in nxt and "load_reduction" not in nxt
    assert not (lib.operator_dir(session) / drive.LOAD_REDUCTION).exists()


def test_an_offer_is_stale_once_the_case_changed_and_a_clone_drops_it(tmp_path):
    session = _session(tmp_path)
    drive.solve_failure_next(session, {}, NOT_CONVERGED)
    plan = lib.read_json(lib.operator_dir(session) / "plan.json")
    assert drive.load_reduction_problem(session, 0.8, {**plan, "load_reduction": 0.8}) is None
    assert drive.load_reduction_problem(session, 0.8, {**plan, "load_reduction": 0.8,
                                                       "settings": {"load_steps": 20}}) is None
    stale = drive.load_reduction_problem(session, 0.8, {**plan, "regions": "Fix the top.", "load_reduction": 0.8})
    assert stale and "for this case" in stale
    assert "operator/load_reduction.json" in drive.CLONE_DROP


BRIEF = """# Case brief: a stack

## Use

Use table: {"actions": {"A1": {"load": "press on top", "state": "working", "kind": "applied", "source": "given", "size": 25, "unit": "N", "direction": [0, -1, 0], "enters": "top", "held": ["base"]}}, "run": "working", "uncovered": {}}

## Size and material

Real mass: 0.25 kg
"""


def test_the_reduction_is_its_own_marked_simplification_and_no_load_cut(tmp_path, monkeypatch):
    session = _session(tmp_path)
    operator = lib.operator_dir(session)
    drive.solve_failure_next(session, {}, NOT_CONVERGED)
    drive.set_applied_load_reduction(session, 0.8)
    (operator / "brief.md").write_text(BRIEF)
    (session / "brief_snapshots").mkdir()
    (session / "brief_snapshots" / "brief.first.md").write_text(BRIEF)
    lib.write_json(operator / "prediction_history.json",
                   [{"run": "r", "solved": {}, "load_reduction": drive.applied_load_reduction(session)}])
    said = drive.load_reduction_simplification(session)
    assert said.startswith("load reduced to 0.8 of the brief's (every load, the self-weight with it) because the "
                           "full load did not converge")
    assert "converged 8 of 10 steps, through 0.8, and failed at 0.9" in said
    # the Use table is as briefed: the lowered-rows check finds no cut, and the runner lists the reduction
    assert drive.run_row_sizes(operator) == {"A1": [25.0, "N"]}
    assert drive.rows_lowered_since_body(operator) == []
    (operator / "brief.md").write_text(BRIEF.replace('"size": 25', '"size": 20'))     # a cut by hand is one
    assert drive.rows_lowered_since_body(operator) == ["Use table row A1 25 N -> 20 N"]
    (operator / "brief.md").write_text(BRIEF)
    listed = drive.runner_simplifications(session, {}, [])
    assert said in listed and not any(s.startswith("load lowered after the body") for s in listed)
    # judged: the brief's case claimed gets the runner's simplified line; "Deviations: none" is refused
    (operator / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    (session / "checkpoints.sqlite").write_text("")
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))

    def judge(text):
        (operator / "j.md").write_text(text)
        drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                           file=str(operator / "j.md"), text=None))
        return printed[-1]
    out = judge("The brief's case was run.\n## Deviations\nnone\n")
    assert not out["ok"] and "'Deviations: none', but the solve ran with the load reduced to 0.8" in str(out)
    out = judge("The brief's case was run.\n## Deviations\nthe load at 0.8 of the brief's: the full load did not "
                "converge\n")
    assert out["ok"], out
    report = (operator / "report.md").read_text()
    assert "A simplified case was run: " in report and "load reduced to 0.8 of the brief's" in report


def test_the_prediction_is_compared_at_the_reduced_load():
    prediction = {"max_displacement": 0.01, "max_stress": 1e5, "source": "the runner's"}
    body = {"sha": "s", "size": 0.1}
    history = [{"run": "r1", "body": body, "ends": "e", "load_N": 25.0, "prediction": prediction,
                "table_loads": True, "solved": {}},
               {"run": "r2", "body": body, "ends": "e", "load_N": 20.0, "prediction": prediction,
                "table_loads": True, "load_reduction": 0.8, "solved": {"max_displacement": 0.008, "max_stress": 8e4}}]
    agreed = drive.agreement(history)
    assert agreed["load_scale"] == 0.8 and agreed["max_displacement"]["ratio"] == 1.0
    # without a Use table: a reference solve itself reduced had its prediction made for the full load
    for entry in history:
        entry["table_loads"] = False
    history[0].update({"load_reduction": 0.8, "load_N": 20.0})
    history[1].update({"load_reduction": None, "load_N": 25.0,
                       "solved": {"max_displacement": 0.01, "max_stress": 1e5}})
    agreed = drive.agreement(history)
    assert agreed["load_scale"] == 1.0 and agreed["max_displacement"]["ratio"] == 1.0
