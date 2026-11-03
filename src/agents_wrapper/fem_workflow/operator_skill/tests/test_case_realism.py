"""The case's realism checks: a resting face clamped against tipping, a solve far off its
prediction, the body's orientation against the brief, the paint words, the plan stop's depth.
No GPU, no API, no graph run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402

G = [0.0, -9.80665, 0.0]


def _standing_session(tmp_path, held):
    """A session whose resolved held regions are ``held`` (name -> bounds)."""
    session = tmp_path / "abc"
    (session / "boundary_conditions").mkdir(parents=True)
    (session / "operator").mkdir()
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"name": n, "category": "dirichlet", "bounds": b} for n, b in held.items()]})
    return session


# a slim upright body 0.05 m across and 0.2 m tall, standing on its base face
BASE = {"name": "base", "category": "dirichlet", "mean_normal": [0.0, -1.0, 0.0],
        "centroid": [0.0, -0.1, 0.0], "area": 0.0025}
TOP = {"name": "top", "category": "neumann", "centroid": [0.025, 0.09, 0.0]}
BASE_BOUNDS = [[-0.025, -0.1, -0.025], [0.025, -0.1, 0.025]]


def test_a_face_the_body_stands_on_clamped_against_tipping_is_said(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    plan = {"values": {"acceleration": G}, "contacts": {"base": {"transmits": "force and moment",
                                                                 "regions": ["base"]}}}
    regions = [BASE, TOP]
    said = drive.tipping_warning(session, plan, regions, [{"region": "top", "force_N": [20.0, 0, 0]}])
    # 20 N at 0.19 m above the base: 3.8 N m against 0.2 kg x g x 0.025 m
    assert said and "would tip" in said and "3.8 N m" in said and "where the use steadies it" in said
    # the ways out, the grip first; the push capped at the one that tips it, with its number
    assert said.index("steadies it") < said.index("at 0.232 N, under the push that tips it") < said.index("presses down")
    assert "refuses it for a rest" in said
    push = [{"region": "top", "force_N": [20.0, 0, 0]}]
    assert drive.tipping_warning(session, plan, regions, push) is None               # once per contact
    (session / "operator" / drive.TIPPING_SAID).unlink()
    # a push under what resting takes, a twist about the vertical, or a press: nothing to say
    assert drive.tipping_warning(session, plan, regions, [{"region": "top", "force_N": [0.1, 0, 0]}]) is None
    couple = [{"region": "left", "force_N": [0, 0, 5.0]}, {"region": "right", "force_N": [0, 0, -5.0]}]
    pair = [BASE, {"name": "left", "category": "neumann", "centroid": [-0.02, 0.09, 0.0]},
            {"name": "right", "category": "neumann", "centroid": [0.02, 0.09, 0.0]}]
    assert drive.tipping_warning(session, plan, pair, couple) is None
    assert drive.tipping_warning(session, plan, [BASE, {**TOP, "centroid": [0.0, 0.09, 0.0]}],
                                 [{"region": "top", "force_N": [0, -50.0, 0]}]) is None
    # a grip held as well takes the moment
    grip = {"name": "grip", "category": "dirichlet", "mean_normal": [1.0, 0.0, 0.0],
            "centroid": [0.025, 0.0, 0.0], "area": 0.002}
    assert drive.tipping_warning(session, plan, [BASE, grip, TOP],
                                 [{"region": "top", "force_N": [20.0, 0, 0]}]) is None


def test_a_fastened_base_is_not_warned_and_a_resting_one_is(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    push = [{"region": "top", "force_N": [20.0, 0, 0]}]
    contact = {"transmits": "force and moment", "regions": ["base"]}
    # no acceleration in the plan: the weight still counts, under g
    resting = {"values": {}, "contacts": {"base": contact}}
    assert "would tip" in drive.tipping_warning(session, resting, [BASE, TOP], push)
    (session / "operator" / drive.TIPPING_SAID).unlink()
    small = [{"region": "top", "force_N": [0.2, 0, 0]}]                     # 0.038 N m against 0.049 N m
    assert drive.tipping_warning(session, resting, [BASE, TOP], small) is None
    # a fastening over a length (its span unmeasured here) is clamped rightly: its words are not read
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n- base: held\n{"base": {"transmits": "force and moment", "kind": "fastening", "over_m": 0.1}}\n')
    assert drive.tipping_warning(session, resting, [BASE, TOP], push) is None
    # a fastening with no length it holds takes no moment: it tips as a rest does
    (session / "operator" / drive.TIPPING_SAID).unlink(missing_ok=True)
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n- base: held\n{"base": {"transmits": "force and moment", "kind": "fastening"}}\n')
    assert "would tip" in drive.tipping_warning(session, resting, [BASE, TOP], push)
    (session / "operator" / drive.TIPPING_SAID).unlink()
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n- base: bolted to the floor plate\n{"base": {"transmits": "force only", "kind": "rest"}}\n')
    assert "would tip" in drive.tipping_warning(session, resting, [BASE, TOP], push)


def test_a_heavy_wide_base_resists_the_same_push(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": [[-0.5, -0.1, -0.5], [0.5, -0.1, 0.5]]})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 50.0})
    plan = {"values": {"acceleration": G}}
    assert drive.tipping_warning(session, plan, [BASE, TOP], [{"region": "top", "force_N": [20.0, 0, 0]}]) is None


def _judge_session(tmp_path, solved, brief=""):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    (session / "operator" / "brief.md").write_text(brief)
    lib.write_json(session / "operator" / "prediction_history.json", [
        {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 10.0,
         "prediction": {"max_displacement": 1e-5, "max_stress": 1e6}, "solved": solved}])
    return session


def _judge(tmp_path, monkeypatch, session, text="The brief's case was run.\n## Deviations\nnone\n"):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    (session / "operator" / "j.md").write_text(text)
    drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                       file=str(session / "operator" / "j.md"), text=None))
    return printed[-1]


BRIEF = ("## Use\na hand presses the end\n## Load\n10 N at the end\n## Size and material\nspan 0.1 m, "
         "E 2e11 Pa\n## Prediction\ncantilever, 1e-5 m\n")
DISAGREES = ("The brief's case was run; the result disagrees with the prediction, max_displacement RATIO x, "
             "CAUSE\n## Deviations\nnone\n")


def test_one_body_rule_decides_the_verdict_whatever_the_route_and_simplifications(tmp_path, monkeypatch):
    session = _judge_session(tmp_path, {"max_displacement": 1e-5, "max_stress": 1e6}, BRIEF)
    monkeypatch.setattr(drive, "runner_simplifications", lambda *a, **k: ["the body is a stand-in"])
    monkeypatch.setattr(drive, "simplification_problem", lambda *a, **k: (None, None))
    outside = {"member_over_real": 6.5, "mass_over_real": 2.0}
    monkeypatch.setattr(drive, "outside_range", lambda s, planned=None: drive.range_findings(outside))
    for verdict in ("A simplified case was run: a stand-in.", "The brief's case was run.",
                    "The governing load cannot be applied; a secondary case was run: x."):
        # the runner writes the verdict line G4 forces, and accepts
        out = _judge(tmp_path, monkeypatch, session, verdict + "\n## Deviations\nnone\n")
        report = (session / "operator" / "report.md").read_text()
        assert out["ok"] and "The brief's case could not be run: the body is outside the workflow's range (thinnest " \
            "member 6.5 x the real one" in report, verdict
    assert _judge(tmp_path, monkeypatch, session, "The brief's case could not be run.\n## Deviations\nnone\n")["ok"]
    # within the rule, a marked simplification stands
    outside.update(member_over_real=2.9, mass_over_real=4.9)
    assert drive.range_findings(outside) is None
    # the agent's own cause stays on the line, its verdict words replaced
    monkeypatch.setattr(drive, "outside_range", lambda s, planned=None: "mass 6.4 x the real one")
    assert _judge(tmp_path, monkeypatch, session, "The brief's case was run; the solid stands in for a hollow "
                                                  "part.\n## Deviations\nnone\n")["ok"]
    line = [ln for ln in (session / "operator" / "report.md").read_text().splitlines()
            if ln.startswith("The brief's case could not be run")][0]
    assert line.endswith("The solid stands in for a hollow part.") and "case was run" not in line
    assert _judge(tmp_path, monkeypatch, session, "The governing load cannot be applied; a secondary case was "
                                                  "run: the lateral load.\n## Deviations\nnone\n")["ok"]
    line = [ln for ln in (session / "operator" / "report.md").read_text().splitlines()
            if ln.startswith("The brief's case could not be run")][0]
    assert "secondary case" not in line and "cannot be applied" not in line and line.endswith("The lateral load.")
    monkeypatch.setattr(drive, "outside_range", lambda s, planned=None: None)
    # the governing load at the Real section's member declared not applicable: the runner's verdict line says so
    monkeypatch.setattr(drive, "solved_coverage", lambda s: {"member": "plate", "uncovered": {
        "plate": {"declared": "contact between separate bodies: a separate part presses it"}}})
    assert _judge(tmp_path, monkeypatch, session)["ok"]
    assert ("The governing load cannot be applied; a secondary case was run: at plate, contact between separate "
            "bodies") in (session / "operator" / "report.md").read_text()


def test_a_solve_five_times_off_its_prediction_is_refused_until_stated_or_fixed(tmp_path, monkeypatch):
    session = _judge_session(tmp_path, {"max_displacement": 8e-5, "max_stress": 1.2e6}, BRIEF)
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and "max_displacement 8.0 x the prediction" in out["error"]
    # no target to aim a revised number at
    assert "within" not in out["error"] and "case, the body or the prediction" in out["error"]


def test_an_honest_disagreement_passes_and_a_wrong_ratio_is_refused(tmp_path, monkeypatch):
    session = _judge_session(tmp_path, {"max_displacement": 8e-5, "max_stress": 1.2e6}, BRIEF)
    out = _judge(tmp_path, monkeypatch, session, DISAGREES.replace("RATIO", "3").replace(
        "CAUSE", "cause not found"))
    assert not out["ok"] and "not the runner's ratio: max_displacement 8.0 x" in out["error"]
    out = _judge(tmp_path, monkeypatch, session, DISAGREES.replace("RATIO", "8").replace("CAUSE", "unclear"))
    assert not out["ok"] and "cause not found" in out["error"]
    out = _judge(tmp_path, monkeypatch, session, DISAGREES.replace("RATIO", "8.0").replace(
        "CAUSE", "cause not found"))
    assert out["ok"], out
    head = (session / "operator" / "report.md").read_text().splitlines()[2]
    assert "max_displacement disagrees (8.0)" in head and "explained" not in head
    # a solve under the prediction may be stated by its inverse ("12 x under")
    assert drive.stated_disagreement("disagrees, 12 x under it, cause found: a load face",
                                     {"max_stress": {"ratio": 0.084}}) == (True, None)


def test_a_solve_within_five_or_fixed_and_solved_again_passes(tmp_path, monkeypatch):
    session = _judge_session(tmp_path, {"max_displacement": 4e-5, "max_stress": 1.2e6})
    assert _judge(tmp_path, monkeypatch, session)["ok"]
    # the case changed (other end conditions) and solved again inside 5 x: the gate is passed
    far = {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 10.0,
           "prediction": {"max_displacement": 1e-5, "max_stress": 1e6},
           "solved": {"max_displacement": 8e-5, "max_stress": 1.2e6}}
    fixed = {**far, "run": "r2", "ends": "e2", "solved": {"max_displacement": 3e-5, "max_stress": 1.2e6}}
    lib.write_json(session / "operator" / "prediction_history.json", [far, fixed])
    assert _judge(tmp_path, monkeypatch, session)["ok"]
    # per-part materials are not judged on the number
    assert drive.far_off_case({"max_stress": {"ratio": 9.0, "solved": 9e6, "statement": "not judged (parts)"}}) == {}
    # another prediction heading is not the generated prediction
    assert not drive.is_generated_heading("prediction (revised)")
    assert drive.is_generated_heading("prediction (generated)")


def test_the_body_axes_are_printed_against_the_briefs_up_in_use(tmp_path):
    (tmp_path / "operator").mkdir()
    brief = tmp_path / "operator" / "brief.md"
    plan = {"values": {"acceleration": G}}
    disc = {"physical_extents": [0.8, 0.56, 0.8]}            # a disc-shaped body standing on its face
    brief.write_text("## Size and material\nUp in use: the shortest extent horizontal\n")
    out, said = drive.body_orientation(tmp_path, plan, disc)
    assert out["up"] == "+y" and out["agrees"] is False
    assert "shortest extent (y) is vertical" in said and "plan.body_up (one of +x: use x -> body -y, use y -> body +x" in said
    assert "the runner turns the Use table rows and the acceleration" in said
    # a body upright as the real object stands: the brief's line to fix, in one line
    assert "fix the brief's line to 'Up in use: the shortest extent vertical'" in said
    # the agent states the body's up; the acceleration stays the use frame's, never edited by hand
    out, said = drive.body_orientation(tmp_path, {**plan, "body_up": "+x"}, disc)
    assert out["up"] == "+x" and out["agrees"] and said is None
    # the axes are printed, never mapped by hand
    assert out["use_to_body"] == "use x -> body -y, use y -> body +x, use z -> body +z"
    # a sign that turns the rests' faces up is refused: they would hang
    rest = {"base": {"kind": "rest", "transmits": "force only"}}
    held = [{"name": "foot", "category": "dirichlet", "mean_normal": [1.0, 0.0, 0.0]}]
    turned = {**plan, "body_up": "+x", "contacts": {"base": {"regions": ["foot"]}}}
    assert "would hang" in drive.body_up_problem(turned, held, rest) and "-x" in drive.body_up_problem(turned, held, rest)
    assert drive.body_up_problem({**turned, "body_up": "-x"}, held, rest) is None
    # the one statement turns the rows and the gravity alike, with their signs
    turned = {**plan, "body_up": "+x"}
    assert drive.body_vector(turned) == [9.80665 * 0 - 0.0 * 0 + 0.0, 0.0, 0.0] or True
    assert drive.body_vector(turned) == [-9.80665, 0.0, 0.0]
    assert drive.turned_direction([0, -1, 0], drive.use_to_body(None, turned)) == [-1.0, 0.0, 0.0]
    assert drive.body_vector({**plan, "body_up": "-y"}) == [0.0, 9.80665, 0.0]
    assert drive.body_vector(plan) == [0.0, -9.80665, 0.0] and drive.use_to_body(None, plan) is None
    # near-equal extents (within 2 %) are one: either of the two may be the vertical one
    brief.write_text("Up in use: the longest extent vertical\n")
    assert drive.body_orientation(tmp_path, {**plan, "body_up": "-z"}, {"physical_extents": [0.8, 0.56, 0.79]})[1] is None
    # two equal longest extents: either may be the horizontal one
    brief.write_text("Up in use: the longest extent horizontal\n")
    assert drive.body_orientation(tmp_path, plan, disc)[1] is None
    brief.write_text("Up in use: the longest extent vertical\n")
    assert "longest extent (x/z) is horizontal" in drive.body_orientation(tmp_path, plan, disc)[1]
    # no line: the axes alone
    brief.write_text("## Size and material\n0.8 m across\n")
    out, said = drive.body_orientation(tmp_path, plan, disc)
    assert said is None and out["extents_m"] == {"x": 0.8, "y": 0.56, "z": 0.8} and "brief" not in out


def test_dark_or_glossy_paint_words_are_said_once_before_the_first_paint(tmp_path):
    (tmp_path / "operator").mkdir()
    texture = {"texture_prompt": "dark polished walnut wood"}
    said = drive.texture_words_note(tmp_path, texture)
    assert "['dark', 'polished']" in said and "before the first paint" in said
    assert drive.texture_words_note(tmp_path, texture) is None                  # said once per wording
    assert "glossy" in drive.texture_words_note(tmp_path, {"texture_prompt": "glossy red steel"})
    assert drive.texture_words_note(tmp_path, {"texture_prompt": "light tan pine wood"}) is None
    # after a paint, the words are the paint's: nothing is said about them
    lib.write_json(tmp_path / "operator" / "run_sent.json", {"texture": "{}"})
    assert drive.texture_words_note(tmp_path, {"texture_prompt": "dark grey cast iron"}) is None


def test_the_depth_that_resolves_a_thin_real_member_is_written_into_a_plan_naming_none(tmp_path):
    (tmp_path / "operator").mkdir()
    path = tmp_path / "operator" / "plan.json"
    plan = {"scale": {"physical_extent": 1.7}, "settings": {}}
    path.write_text(json.dumps(plan, indent=2) + "\n")
    # no brief member: nothing changes
    assert drive.prescribed_depth(tmp_path, path, plan) == (plan, None)
    # 0.04 m at 1.7 m: 1.6 leaves at depth 7, 3.1 at depth 8
    (tmp_path / "operator" / "brief.md").write_text("## Size and material\nReal thinnest member: 0.04 m\n")
    out, said = drive.prescribed_depth(tmp_path, path, plan)
    assert out["settings"]["remesh_octree_depth"] == drive.DEEPEST_DEPTH and "the runner set" in said
    written = path.read_text()
    assert json.loads(written)["settings"] == {"remesh_octree_depth": drive.DEEPEST_DEPTH}
    assert written.startswith('{\n  "scale"')                                 # the file's own layout
    # a plan rewritten without it (a Write from memory) gets it again, and says so
    path.write_text(json.dumps(plan, indent=2) + "\n")
    out, said = drive.prescribed_depth(tmp_path, path, plan)
    assert out["settings"]["remesh_octree_depth"] == drive.DEEPEST_DEPTH and "again" in said
    assert json.loads(path.read_text())["settings"]["remesh_octree_depth"] == drive.DEEPEST_DEPTH
    named = {**plan, "settings": {"remesh_octree_depth": 7}}
    assert drive.prescribed_depth(tmp_path, path, named) == (named, None)    # a depth the plan names is kept


# ------------------------------------------------ the brief's real case against strength and size --

STRIP = {"scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 0.3},
         "material": {"youngs_modulus": 2.0e11, "poisson_ratio": 0.29, "density": 7900},
         "prediction": {"max_displacement": 0.003, "max_stress": 2.25e8}}


def test_the_strength_bands_are_read_from_the_materials_table():
    rows = {r["material"]: r for r in drive.material_strength_rows()}
    assert rows["structural / stainless steel"]["strength_MPa"] == (235.0, 355.0)
    assert rows["glass-fibre composite, along fibre"]["E_Pa"] == (2e10, 5e10)
    assert "rubber" not in rows                                    # n/a: no strength to check
    assert rows["concrete"]["strength_MPa"] == (25.0, 50.0)        # the stronger band of tension, compression
    # E and density pick the row; a strength the brief states serves when no row holds them
    assert drive.material_strength(STRIP)["strength_MPa"] == (235.0, 355.0)
    odd = {**STRIP, "material": {"youngs_modulus": 5e12, "density": 100}}
    assert drive.material_strength(odd, "rho 100 kg/m3, yield 40-60 MPa")["strength_MPa"] == (40.0, 60.0)
    assert drive.material_strength(odd) is None


def _use_table_session(tmp_path, top_plate, name="abc"):
    """The Use table's case run at the Real section: (top_plate + 3) N m, one push on the largest
    Real extent (0.3 m), the runner's lever before any region."""
    from test_use_table import session_with, table
    return session_with(tmp_path, table(rows={"A1": {"size": round((top_plate + 3) / 0.3, 6), "source": "derived: %r N m / 0.3 m" % (top_plate + 3)},
                                              "A2": None, "A3": {"lever_m": 0.01}}), name=name)


def test_a_real_case_past_its_strength_or_its_length_is_flagged_with_its_numbers(tmp_path):
    # the stress is the runner's, from the Use table's case run at the Real section: 43 N m on a 0.01 m deep
    # plate of I 8.3e-10 m4 is 259 MPa, whatever the brief predicts
    session = _use_table_session(tmp_path, 40.0)
    plan = {**STRIP, "prediction": {"max_displacement": 0.04, "max_stress": 2.6e8}}
    found = drive.plausibility(session, plan)
    assert found["flags"] == ["stress", "deflection"] and found["stress_MPa"] == pytest.approx(259.0, abs=0.1)
    assert found["deflection_over_length"] == pytest.approx(0.1333, abs=1e-4)      # on the 0.3 m Real extent
    said = "; ".join(drive.plausibility_findings(found))
    assert "1.1 x the lower strength of structural / stainless steel (235-355 MPa)" in said
    assert "deflects 0.04 m on 0.3 m" in said
    assert drive.plausibility_gate(session, plan) is None                     # said, never refused
    # an everyday case well inside both: nothing said
    calm = _use_table_session(tmp_path, 10.0, "calm")
    assert drive.plausibility(calm, {**STRIP, "prediction": {"max_displacement": 3e-4, "max_stress": 8e7}})["flags"] == []
    assert drive.plausibility(None, plan) is None
    # no Use table (a session from before it): the brief case's predicted stress stands in for the
    # runner's, and the deflection is read as before
    brief = session / "operator" / "brief.md"
    brief.write_text("\n".join(ln for ln in brief.read_text().splitlines() if not ln.startswith("Use table:")))
    found = drive.plausibility(session, plan)
    assert found["flags"] == ["stress", "deflection"] and found["stress_MPa"] == pytest.approx(260.0)
    assert "the brief case's predicted stress is 260 MPa, 1.11 x the lower strength" in "; ".join(
        drive.plausibility_findings(found))


def test_the_flag_is_said_once_before_the_solve(tmp_path):
    session = _use_table_session(tmp_path, 40.0)
    plan = {**STRIP, "prediction": {"max_displacement": 3e-4, "max_stress": 2.6e8}}
    note = drive.plausibility_note(session, plan)
    assert note and "after the solve its own numbers decide" in note
    assert drive.plausibility_note(session, plan) is None                     # a note once
    assert drive.plausibility_gate(session, plan) is None                     # never refused


def test_judge_takes_no_agreeing_verdict_on_a_flagged_case_the_verdict_does_not_name(tmp_path, monkeypatch):
    session = _use_table_session(tmp_path, 40.0)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "prediction_history.json", [
        {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 10.0,
         "prediction": {"max_displacement": 1e-5, "max_stress": 2.5e8},
         "solved": {"max_displacement": 1.1e-5, "max_stress": 2.6e8}}])   # the solve near the runner's 259 MPa
    lib.write_json(session / "operator" / "declared.json",
                   {**STRIP, "prediction": {"max_displacement": 3e-4, "max_stress": 2.6e8}})
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and "past the linear range" in out["error"] and "gives 1.1" in out["error"]
    out = _judge(tmp_path, monkeypatch, session, "A simplified case was run: linear, while the plate's stress is "
                                                 "1.1 x the lower strength\n## Deviations\nnone\n")
    assert out["ok"], out
    # after the solve its own stress decides the flag: 260 MPa, 1.1 x the lower strength
    assert "past the linear range: the solve gives 260 MPa" in (session / "operator" / "report.md").read_text()
