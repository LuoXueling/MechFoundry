"""The runner's solve triage: the causes of a failed linear solve no solver setting removes (a
nearly incompressible part, a part-stiffness contrast, holds that leave a near rigid-body mode),
read at a stop before the first solve; a singular solve (‖b − Kx‖/‖b‖ of 1 or more) routed to the
supports-and-components row with no fallback; and the one change a failed solve takes next.
No GPU, no API, no graph run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402

STALL = ("RuntimeError: Linear solver failed at Newton iteration 0: JAX-AMG solve failed with status "
         "AMGXStatus.NOT_CONVERGED after 1000 iterations (||b - Kx||/||b||={r}); it stalled 5e+04x above "
         "the 1.0e-06 tolerance, which is the rounding floor of an ill-conditioned system{f}, not a "
         "tolerance set too tight -- try scale_system_to_unit_diagonal=True, then beta_max_over_e")


def _stall(r: str, float32: bool = False) -> str:
    return STALL.format(r=r, f=" in float32" if float32 else "")


# ------------------------------------------------ a singular solve --

def test_a_residual_of_one_or_more_is_singular_and_a_sure_one_takes_no_fallback(tmp_path):
    assert drive.linear_residual(_stall("4.711e+01")) == 47.11
    assert drive.singular_solve(_stall("4.711e+01")) and drive.singular_solve(_stall("1.000e+00"))
    # 10 or more: no backend converged on any logged case, so no fallback
    assert drive.backend_fallback(_stall("4.711e+01")) is None
    # between 1 and 10 a fallback has converged (1.05): it runs, unless the triage finds a near
    # rigid-body mode
    assert drive.backend_fallback(_stall("1.050e+00")) == ("pyamgx", None)
    assert not drive.fallback_pointless(tmp_path, {}, _stall("1.050e+00"))
    session, plan = _two_blocks_on_a_bar(tmp_path, 0.009, "bar")
    assert drive.fallback_pointless(session, plan, _stall("2.375e+00"))
    assert not drive.fallback_pointless(session, plan, _stall("4.1e-02"))
    # an ill-conditioned stall keeps its fallback
    assert not drive.singular_solve(_stall("4.138e-02"))
    assert drive.backend_fallback(_stall("4.138e-02")) == ("pyamgx", None)
    # a float32 solve under the strain floor stalls above 1 too: its remedy is float64, not supports
    assert not drive.singular_solve(_stall("3.7e+01", float32=True))
    assert drive.backend_fallback(_stall("3.7e+01", float32=True)) == ("pyamgx", None)
    assert not drive.singular_solve("Newton solver did not converge in 20 iterations")


def test_a_singular_solve_reads_the_supports_row_not_the_settings_ladder():
    value = {"ok": False, "stop": "solve", "solve_error": _stall("2.375e+00")}
    read = {(e["file"], e.get("section")) for e in drive.read_for("solve", drive._texts(value), value)}
    assert ("references/troubleshooting.md", "Supports, components and parts") in read
    assert ("references/troubleshooting.md", "Solve") not in read
    assert not any(f == "references/solver_settings.md" for f, _ in read)
    # an ill-conditioned stall still reads the Solve table and the ladder
    value["solve_error"] = _stall("4.1e-02")
    read = {(e["file"], e.get("section")) for e in drive.read_for("solve", drive._texts(value), value)}
    assert ("references/troubleshooting.md", "Solve") in read
    assert any(f == "references/solver_settings.md" for f, _ in read)


def test_the_supports_row_names_one_change_and_no_solver_setting(tmp_path):
    text = (SCRIPTS.parent / "references" / "troubleshooting.md").read_text(encoding="utf-8")
    section = text.split("### Supports, components and parts", 1)[1].split("\n## ", 1)[0]
    for cause in ("near rigid-body mode", "ν 0.48 or more", "contrast of 1e3 or more", "some components only"):
        assert cause in section
    assert "scale_system" not in section and "beta_max_over_e" not in section
    # thresholds and remedies only: no held-out case's measurements
    assert "held-out" not in section and "Newton iteration" not in section
    # and the scaling is the ladder's last rung, after the β cap, never earlier
    settings = (SCRIPTS.parent / "references" / "solver_settings.md").read_text(encoding="utf-8")
    ladder = settings.split("- **Linear solver fails**", 1)[1].split("- **Large deformation**", 1)[0]
    assert "2. `beta_max_over_e: 500`" in ladder and "3. last resort" in ladder
    assert ladder.index("2. `beta_max_over_e") < ladder.index("`scale_system_to_unit_diagonal: true`")
    assert "`scale_system_to_unit_diagonal: true`" not in ladder.split("3. last resort", 1)[0]
    # the ladder row says beta_max_over_e does nothing with parts
    assert "a uniform material only" in text
    settings = (SCRIPTS.parent / "references" / "solver_settings.md").read_text(encoding="utf-8")
    assert "1 or more" in settings and "not applied" in settings
    nxt = drive.solve_failure_next(tmp_path, {}, _stall("5.137e+01"), {"solve": 4})
    assert "no fallback ran" in nxt and "Supports, components and parts" in nxt
    assert "Not scale_system_to_unit_diagonal or beta_max_over_e" in nxt and "4 solve attempts left" in nxt
    # after a fallback that ran (1 to 10) and failed, the same row, said as such
    nxt = drive.solve_failure_next(tmp_path, {}, _stall("2.375e+00"), {"solve": 3}, fallback_failed=True)
    assert "the fallback did not converge either" in nxt and "Supports, components and parts" in nxt


def test_a_stall_names_the_solvers_own_remedy_first_and_its_retry_is_free(tmp_path):
    (tmp_path / "operator").mkdir()
    plan = {"material": {"mode": "uniform"}}
    nxt = drive.solve_failure_next(tmp_path, plan, _stall("2.0e-02"), {"solve": 2})
    assert "plan.settings.beta_max_over_e: 500" in nxt and "not counted" in nxt
    assert "scale_system_to_unit_diagonal" not in nxt          # measured to rescue none
    # the plan sets it: that retry is the uncounted one, once
    assert not drive.remedy_retry(tmp_path, plan)
    tried = {**plan, "settings": {"beta_max_over_e": 500}}
    # a retry that changes anything else as well is counted
    assert not drive.remedy_retry(tmp_path, {**tried, "settings": {"beta_max_over_e": 500, "load_steps": 5}})
    assert not drive.remedy_retry(tmp_path, {**tried, "regions": "Fix the other end."})
    assert drive.remedy_retry(tmp_path, tried) and not drive.remedy_retry(tmp_path, tried)
    # named once: a later stall goes on to the ladder
    assert "not counted" not in drive.solve_failure_next(tmp_path, plan, _stall("2.0e-02"))


def test_unit_diagonal_is_the_last_rung_once_the_others_are_spent(tmp_path):
    (tmp_path / "operator").mkdir()
    plan = {"material": {"mode": "uniform"}, "settings": {"use_mixed_precision": True}}
    # the first ladder stop names the ladder, not the scaling (the β rung is still open)
    nxt = drive.solve_failure_next(tmp_path, plan, _stall("2.0e-02").split(" -- try ")[0], {"solve": 3},
                                   fallback_failed=True)
    assert "one setting of solver_settings.md §2" in nxt and "scale_system_to_unit_diagonal" not in nxt
    # with no solve left, or a float32 run (mixed precision still open), no last rung
    beta = {**plan, "settings": {"use_mixed_precision": True, "beta_max_over_e": 500}}
    assert not drive.last_resort_due(tmp_path, beta, _stall("2.0e-02"), 0)
    assert not drive.last_resort_due(tmp_path, beta, _stall("2.0e-02", float32=True), 2)
    assert not drive.last_resort_due(tmp_path, beta, "Newton solver did not converge in 20 iterations", 2)
    # the β cap tried and failed, a solve left: the last resort, once
    nxt = drive.solve_failure_next(tmp_path, beta, _stall("2.0e-02").split(" -- try ")[0], {"solve": 2},
                                   fallback_failed=True)
    assert "last resort" in nxt and "plan.settings.scale_system_to_unit_diagonal: true" in nxt
    assert "unscaled residual" in nxt and "report says it ran scaled" in nxt
    assert not drive.last_resort_due(tmp_path, beta, _stall("2.0e-02"), 1)       # once
    # it failed too: remove it and report the failure, no further rung
    scaled = _stall("5.876e+00").replace("scale_system_to_unit_diagonal=True, then ", "")
    nxt = drive.solve_failure_next(tmp_path, {**beta, "settings": {**beta["settings"],
                                   "scale_system_to_unit_diagonal": True}}, scaled, {"solve": 1})
    assert "last-resort solve with unit-diagonal scaling failed" in nxt
    assert "scale_system_to_unit_diagonal: null" in nxt and "report the solve as failed" in nxt
    # β skipped (members too thin for it): one ladder change taken in its place opens the last rung
    other = tmp_path / "thin"
    (other / "operator").mkdir(parents=True)
    stall = _stall("6.9e-03").split(" -- try ")[0]
    assert "last resort" not in drive.solve_failure_next(other, plan, stall, {"solve": 3})
    assert "last resort" in drive.solve_failure_next(other, plan, stall, {"solve": 2})


def test_a_failed_fallback_names_the_triage_change_before_the_ladder(tmp_path, monkeypatch, capsys):
    session = _parts_session(tmp_path, [(0, 5e6, 0.48), (1, 1e10, 0.35)])

    def fake(args):
        print(json.dumps({"ok": False, "backend": args.backend, "error": ["Newton solver did not converge"]}))
    monkeypatch.setattr(drive, "cmd_solve_backend", fake)
    args = argparse.Namespace(output_root=str(tmp_path), session_id="abc", gpu="auto", allow_busy_gpu=False)
    value = drive.solve_fallback(args, session, _stall("4.1e-02"), {})
    assert "did not converge either" in value["next"] and "The one change: nu 0.45" in value["next"]
    assert "thin-member row" not in value["next"]
    # with no triage cause the setting the solver's message names is next, then the ladder: with the
    # β cap spent and solves left, its last rung, the unit-diagonal scaling
    plain = _parts_session(tmp_path / "plain", [(0, 7e10, 0.33)], mode="uniform")
    value = drive.solve_fallback(args, plain, _stall("4.1e-02"), {})
    assert "plan.settings.beta_max_over_e: 500" in value["next"]
    value = drive.solve_fallback(args, plain, _stall("4.1e-02"), {"settings": {"beta_max_over_e": 500}})
    assert "last resort" in value["next"] and "scale_system_to_unit_diagonal: true" in value["next"]
    # once: the next failure is back on the ladder's text
    value = drive.solve_fallback(args, plain, _stall("4.1e-02"), {"settings": {"beta_max_over_e": 500}})
    assert "thin-member row" in value["next"]


# ------------------------------------------------ the materials --

def _parts_session(root: Path, parts, mode: str = "parts") -> Path:
    session = root / "abc"
    (session / "operator").mkdir(parents=True, exist_ok=True)
    rows = [{"id": i, "youngs_modulus": E, "poisson_ratio": nu, "density": 1000.0} for i, E, nu in parts]
    material = ({"mode": "parts"} if mode == "parts" else
                {"mode": "uniform", "youngs_modulus": rows[0]["youngs_modulus"],
                 "poisson_ratio": rows[0]["poisson_ratio"]})
    lib.write_json(session / "session.json", {"material": material,
                                              "material_parts": rows if mode == "parts" else None})
    return session


def test_a_near_incompressible_part_and_a_stiffness_contrast_are_warned_with_one_change(tmp_path):
    session = _parts_session(tmp_path, [(0, 5e6, 0.48), (1, 1e10, 0.35), (2, 5e6, 0.48)])
    said = drive.material_triage(session, {})
    assert len(said) == 2
    nu, contrast = said
    assert "parts 0, 2 at nu 0.48" in nu and "The one change: nu 0.45" in nu
    assert "the soft parts' E raised to the stiffest / 100" in nu and "marked simplification" in nu
    assert "2000x" in contrast and "The one change" in contrast and "marked simplification" in contrast
    # under the thresholds: nothing
    calm = _parts_session(tmp_path / "calm", [(0, 2e8, 0.45), (1, 1e10, 0.35)])
    assert drive.material_triage(calm, {}) == []


def test_a_uniform_near_incompressible_material_is_warned_and_a_parts_penalty_cap_is_a_no_op(tmp_path):
    uniform = _parts_session(tmp_path / "u", [(0, 5e6, 0.49)], mode="uniform")
    said = drive.material_triage(uniform, {})
    assert len(said) == 1 and "the material at nu 0.49" in said[0] and "plan.material" in said[0]
    assert "contrast" not in said[0]
    parts = _parts_session(tmp_path / "p", [(0, 2e9, 0.35), (1, 1e10, 0.35)])
    said = drive.material_triage(parts, {"settings": {"beta_max_over_e": 500}})
    assert len(said) == 1 and "not applied" in said[0]
    assert drive.material_triage(uniform.parent.parent / "u" / "abc", {"settings": {"beta_max_over_e": 500}}) \
        == drive.material_triage(uniform, {})


# ------------------------------------------------ the holds --

def _two_blocks_on_a_bar(tmp_path, bar: float, held: str, pair: bool = True) -> tuple[Path, dict]:
    """Two 0.06 m blocks joined by a bar of square section ``bar`` m, 0.3 m long; held on the bar's
    middle ("bar") or on one block's end ("block"), loaded on the blocks' outer ends."""
    import trimesh
    session = tmp_path / f"s{bar}{held}{pair}"
    for sub in ("boundary_conditions", "geometry", "operator"):
        (session / sub).mkdir(parents=True)
    parts = [trimesh.creation.box(extents=[0.06, 0.06, 0.06]).apply_translation([x, 0.0, 0.0])
             for x in (-0.18, 0.18)]
    parts.append(trimesh.creation.box(extents=[0.32, bar, bar]))
    body = trimesh.util.concatenate(parts)
    for _ in range(3):
        body = body.subdivide()
    body.export(session / "geometry" / "physical.glb")
    centres, normals = body.triangles_center, body.face_normals
    if held == "bar":
        hold = (np.abs(centres[:, 0]) < 0.02) & (np.abs(centres[:, 0]) < 0.15)
    else:
        hold = (centres[:, 0] < -0.209) & (normals[:, 0] < -0.9)
    right = (centres[:, 0] > 0.209) & (normals[:, 0] > 0.9)
    left = (centres[:, 0] < -0.209) & (normals[:, 0] < -0.9) & ~hold
    np.savez(session / "boundary_conditions" / "resolved.npz", dirichlet_0=hold, neumann_1=right,
             neumann_2=left)
    area = float(body.area_faces[right].sum())
    regions = [{"name": "middle", "key": "dirichlet_0", "category": "dirichlet"},
               {"name": "right_end", "key": "neumann_1", "category": "neumann", "loaded_area": area},
               {"name": "left_end", "key": "neumann_2", "category": "neumann", "loaded_area": area}]
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": regions})
    plan = {"settings": {"remesh_octree_depth": 8},
            "values": {"neumann_values": {"right_end": {"force": [3.0, 0, 0]},
                                          "left_end": {"force": [-3.0 if pair else -1.0, 0, 0]}}}}
    return session, plan


def test_holds_on_a_slender_member_under_balanced_loads_warn_of_a_near_rigid_mode(tmp_path):
    # leaf at depth 8: 1.92 x 0.42 / 256 = 3.15 mm; a 9 mm bar is under 3 leaves
    session, plan = _two_blocks_on_a_bar(tmp_path, 0.009, "bar")
    thick = drive.hold_thickness(session)
    assert set(thick) == {"middle"} and 0.006 < thick["middle"] < 0.013
    said = drive.near_rigid_triage(session, plan)
    assert said and "the loads balance each other" in said and "near rigid-body mode" in said
    assert "The one change: the members between the holds and the loads over 8 leaves thick" in said
    # the same body held on a block, or loads that do not balance: no warning
    block, plan_block = _two_blocks_on_a_bar(tmp_path, 0.009, "block")
    assert drive.near_rigid_triage(block, plan_block) is None
    unbalanced, plan_unbalanced = _two_blocks_on_a_bar(tmp_path, 0.009, "bar", pair=False)
    assert drive.near_rigid_triage(unbalanced, plan_unbalanced) is None
    # a member thick enough at the hold: no warning
    stout, plan_stout = _two_blocks_on_a_bar(tmp_path, 0.04, "bar")
    assert drive.near_rigid_triage(stout, plan_stout) is None


def test_the_triage_is_a_stop_once_for_the_same_warnings_and_again_for_new_ones(tmp_path):
    session = _parts_session(tmp_path, [(0, 5e6, 0.48), (1, 1e10, 0.35)])
    first = drive.pending_triage(session, {})
    assert first and drive.pending_triage(session, {}) == []          # shown once: the next run goes on
    lib.write_json(session / "session.json", {"material": {"mode": "parts"}, "material_parts": [
        {"id": 0, "youngs_modulus": 1e8, "poisson_ratio": 0.48}, {"id": 1, "youngs_modulus": 1e10,
                                                                  "poisson_ratio": 0.35}]})
    again = drive.pending_triage(session, {})
    assert again and again != first and "contrast" not in " ".join(again)
    # the stop is the runner's own: named, read before the solve, and not an attempt
    assert "triage" in drive.STOPS and drive.STOP_STAGES["triage"] == ("solve",)
    assert drive.READ["triage"][0][1] == "### Supports, components and parts"
    assert [e.get("section") for e in drive.read_for("triage", first)] == ["Supports, components and parts"]
    assert drive._stop_payload("triage", session, tmp_path, {}, {}) == {"why": drive.STOPS["triage"], "look_at": []}
    # a singular failure names the hold change before a material one
    rigid = "solve triage: x, so the holds only stop rigid motion. The one change: hold it better"
    original = drive.solve_triage
    try:
        drive.solve_triage = lambda s, p: drive.material_triage(s, p) + [rigid]
        nxt = drive.solve_failure_next(session, {}, _stall("2.0e+01"))
        assert "make one change: hold it better" in nxt
        nxt = drive.solve_failure_next(session, {}, _stall("2.0e-02"))
        assert "The one change: nu 0.45" in nxt
    finally:
        drive.solve_triage = original


# ------------------------------------------------ run-forward's own message --

def _workflow_tool():
    import importlib.util
    path = SCRIPTS.parents[1] / "scripts" / "workflow_tool.py"
    spec = importlib.util.spec_from_file_location("workflow_tool_for_triage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_forward_rewords_a_singular_linear_failure_and_keeps_the_rest():
    tool = _workflow_tool()
    singular = tool._singular_message(_stall("4.711e+01").removeprefix("RuntimeError: "))
    assert singular.startswith("Linear solver failed at Newton iteration 0: JAX-AMG")
    assert "(||b - Kx||/||b||=4.711e+01)" in singular and "nearly singular system" in singular
    assert "try scale_system" not in singular and "it stalled" in singular
    # the operator's runner reads the reworded message as singular too
    assert drive.singular_solve(singular) and drive.backend_fallback(singular) is None
    # an ill-conditioned stall and a float32 one keep the solver's own advice
    assert tool._singular_message(_stall("4.1e-02")) is None
    assert tool._singular_message(_stall("3.7e+01", float32=True)) is None
    assert tool._singular_message("Newton solver did not converge in 20 iterations") is None

    def failing():
        raise RuntimeError(_stall("2.0e+01").removeprefix("RuntimeError: "))
    try:
        tool._solve_with_diagnosis(failing)
    except RuntimeError as exc:
        assert "nearly singular system" in str(exc) and isinstance(exc.__cause__, RuntimeError)
    else:
        raise AssertionError("the failure was swallowed")
    tool._solve_with_diagnosis(lambda: None)


def test_a_reworded_failure_between_one_and_ten_keeps_its_fallback():
    tool = _workflow_tool()
    for value, fallback in (("1.050e+00", ("pyamgx", None)), ("2.400e+00", ("pyamgx", None)),
                            ("9.900e+00", ("pyamgx", None)), ("1.000e+01", None)):
        text = tool._singular_message(_stall(value).removeprefix("RuntimeError: "))
        assert text and "stalled" in text and "nearly singular system" in text
        assert drive.singular_solve(text) and drive.backend_fallback(text) == fallback, value


def test_a_scaled_residual_is_not_read_as_singular_and_takes_the_scaling_off_first(tmp_path):
    tool = _workflow_tool()
    scaled = tool._singular_message(_stall("2.072e+01").removeprefix("RuntimeError: "), scaled=True)
    assert "Jacobi-scaled system" in scaled and "remove the scaling" in scaled and "stalled" in scaled
    assert drive.scaled_solve(scaled) and not drive.singular_solve(scaled)
    assert drive.backend_fallback(scaled) == ("pyamgx", None)
    # the solver's own message when it ran scaled: it no longer names the scaling as a remedy
    old = _stall("5.876e+00").replace("scale_system_to_unit_diagonal=True, then ", "")
    assert drive.scaled_solve(old) and not drive.singular_solve(old)
    nxt = drive.solve_failure_next(tmp_path, {}, old, {"solve": 2})
    assert "remove the scaling" in nxt and "scale_system_to_unit_diagonal: null" in nxt
    assert "Supports, components and parts" not in nxt
    # read from the case when the message says nothing (a slow solve that ran out of iterations)
    session = tmp_path / "s"
    lib.write_json(session / "case" / "case.json", {"shape_opt": {"scale_system_to_unit_diagonal": True}})
    slow = ("Linear solver failed at Newton iteration 0: JAX-AMG solve failed (||b - Kx||/||b||=2.0e-05); "
            "it was still converging")
    assert drive.scaled_solve(slow, session) and not drive.scaled_solve(slow)
    assert "remove the scaling" in drive.solve_failure_next(session, {}, slow)
    # a scaled residual never skips the fallback, whatever the triage
    pair_body, plan = _two_blocks_on_a_bar(tmp_path, 0.009, "bar")
    assert not drive.fallback_pointless(pair_body, plan, scaled)
    assert drive.fallback_pointless(pair_body, plan, _stall("2.375e+00"))


def test_materials_changed_after_the_triage_are_recorded_and_bar_the_brief_verdict(tmp_path, monkeypatch):
    session = _parts_session(tmp_path, [(0, 5e6, 0.48), (1, 1e10, 0.35)])
    (session / "checkpoints.sqlite").write_text("")
    assert drive.pending_triage(session, {})
    assert drive.material_simplification(session) == []          # nothing changed yet
    lib.write_json(session / "session.json", {"material": {"mode": "parts"}, "material_parts": [
        {"id": 0, "youngs_modulus": 1e8, "poisson_ratio": 0.45}, {"id": 1, "youngs_modulus": 1e10,
                                                                  "poisson_ratio": 0.35}]})
    changes = drive.material_simplification(session)
    assert changes == ["part 0: nu 0.48 -> 0.45 (nearly incompressible, for the solve)",
                       "part 0: E 5e+06 -> 1e+08 Pa (the part contrast, for the solve)"]
    # a second triage (of the changed materials) does not move the record
    drive.pending_triage(session, {})
    assert drive.material_simplification(session) == changes
    (session / "operator" / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "prediction_history.json", [{"run": "r", "solved": {}}])   # a solve ran
    (session / "operator" / "brief.md").write_text("")
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))

    def judge(text):
        (session / "operator" / "j.md").write_text(text)
        drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                           file=str(session / "operator" / "j.md"), text=None))
        return printed[-1]
    out = judge("The brief's case was run.\n## Deviations\nnone\n")
    assert out["ok"] and "A simplified case was run: " in (session / "operator" / "report.md").read_text()
    out = judge("A simplified case was run: the soft parts stiffened and nu lowered for the solve.\n"
                "## Deviations\nnone\n")
    assert out["ok"], out
    report = (session / "operator" / "report.md").read_text()
    assert "### Materials changed for the solve (written by the runner)" in report
    assert "- part 0: nu 0.48 -> 0.45" in report
    # one material after the triage is a change too
    lib.write_json(session / "session.json", {"material": {"mode": "uniform", "youngs_modulus": 1e10,
                                                           "poisson_ratio": 0.35}})
    assert drive.material_simplification(session) == ["the parts made one material (uniform) after the solve triage"]


def test_the_pyamgx_note_leaves_the_runner_s_triage_as_it_was():
    """run_forward adds a pyamgx recommendation to a JAX-AMG failure; the runner reads the
    message the same with it, and a singular one, already beyond any backend, gets none."""
    tool = _workflow_tool()
    for residual in ("4.1e-02", "2.0e+00"):
        plain = _stall(residual).removeprefix("RuntimeError: ")
        reworded = tool._singular_message(plain) or plain
        advised = tool._pyamgx_advice(reworded, "jaxamg")
        if tool.SINGULAR_NOTE in reworded:
            assert advised is None
            continue
        assert advised == reworded + tool.PYAMGX_NOTE and "-- try scale_system" in advised
        assert drive.backend_fallback(advised) == drive.backend_fallback(reworded)
        assert drive.scaled_solve(advised) == drive.scaled_solve(reworded)
        assert drive.singular_solve(advised) == drive.singular_solve(reworded)
        assert drive.linear_residual(advised) == drive.linear_residual(reworded)
    out_of_iterations = ("Linear solver failed at Newton iteration 0: JAX-AMG solve failed with "
                         "status AMGXStatus.NOT_CONVERGED after 1000 iterations "
                         "(||b - Kx||/||b||=4.5e-04); it was still converging and ran out of "
                         "iterations 450x short of the 1.0e-06 tolerance, so raise the solver's "
                         "max_iters")
    advised = tool._pyamgx_advice(out_of_iterations, "jaxamg")
    assert advised.startswith(out_of_iterations) and drive.backend_fallback(advised) == ("jaxamg", 4000)
    assert tool._pyamgx_advice(out_of_iterations, "pyamgx") is None
    assert tool._pyamgx_advice("Newton solver: NaN residual", "jaxamg") is None
    assert tool._pyamgx_advice(advised, "jaxamg") is None       # never twice
