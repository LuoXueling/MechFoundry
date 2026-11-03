"""The four physics gates: G1 the use-state sweep, G2 load-path integrity, G3 plausibility and
agreement, G4 body fidelity.  Each reads JSON and numbers (the brief's JSON lines, the plan, the
meshes, the statics), never the agent's prose.  Neutral parts only: a handle, a base, an
arm, a bar.  No GPU, no API, no graph run."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402
from test_case_realism import BASE, BASE_BOUNDS, TOP, _judge, _judge_session, _standing_session  # noqa: E402
from test_load_path_and_holds import _rod  # noqa: E402
from test_use_table import CONTACTS, ROWS, SIZE, TABLE, gate, session_with, table  # noqa: E402

G = [0.0, -9.80665, 0.0]
STEEL = {"youngs_modulus": 2.0e11, "poisson_ratio": 0.29, "density": 7900}


# ------------------------------------------------------------------------- G1 --

def test_g1_a_human_row_outside_its_range_is_refused(tmp_path):
    said = gate(session_with(tmp_path, table(rows={"A2": {"source": "H3 top", "size": 20}})))
    assert any("20 N is outside H3's range, 50-250 N" in s for s in said)
    assert not any("range" in s for s in gate(session_with(tmp_path, table(rows={"A2": {"source": "H3 top",
                                                                                           "size": 250}}))))


def test_g1_a_row_of_the_surroundings_that_is_no_force_is_size_0_and_held_binds_the_run_only(tmp_path):
    heat = {"+heat": {"load": "heat", "size": 0, "state": "working", "kind": "environment", "enters": "top plate",
                      "source": "derived: none", "held": ["bottom plate"]}}
    assert gate(session_with(tmp_path, table(rows=heat))) == []
    said = gate(session_with(tmp_path, table(rows={"+heat": {**heat["+heat"], "kind": "applied"}})))
    assert any('"size" is a number over 0 (0 only for an action of the surroundings' in s for s in said)
    # a row of another state needs no held contact; a row of the run state does
    assert gate(session_with(tmp_path, table(rows={"A3": {"held": []}}))) == []
    said = gate(session_with(tmp_path, table(rows={"A2": {"held": []}})))
    assert any("Use table row A2: \"held\" names the contacts of the Contacts JSON that hold the object while the "
               "run state acts" in s for s in said)
    # an agent's effect, of any sign or size, is ignored: the runner's stands
    assert gate(session_with(tmp_path, table(rows={"+heat": {**heat["+heat"], "held": [], "effect": {"top plate": 5}},
                                                   "A1": {"effect": {"top plate": -4}}}))) == []


def test_g1_uncovered_names_a_workflow_limit_and_never_the_real_sections_member(tmp_path):
    over = {"A3": {"size": 400, "source": "40 kg x g x 1"}}
    said = gate(session_with(tmp_path, table(rows=over, uncovered={"side grip": "it is small"})))
    assert any('"uncovered" is {"<Load path member>": {"limit": "<one of pressure inside a closed cavity' in s
               for s in said)
    said = gate(session_with(tmp_path, table(rows=over, uncovered={"side grip": {"limit": "small", "why": "x"}})))
    assert any('"uncovered" is {' in s for s in said)
    contact = {"limit": "contact between separate bodies", "why": "the grip is a separate sleeve",
               "no_approximation": "its force is unknown"}
    assert gate(session_with(tmp_path, table(rows=over, uncovered={"side grip": contact, "bottom plate": contact}))) == []


def test_g1_another_state_over_the_run_at_any_member_is_refused_naming_it(tmp_path):
    said = gate(session_with(tmp_path, table(rows={"A3": {"enters": "top plate", "size": 400,
                                                          "source": "40 kg x g x 1"}})))
    assert len(said) == 1 and said[0].count("120 N m at") == 2 and "against 75" in said[0]
    assert "Run 'handling'" in said[0]


def test_g1_a_brief_with_a_table_runs_on_and_one_without_spends_nothing(tmp_path, monkeypatch):
    from test_operator_scripts import PLAN, _thin_run
    session_with(tmp_path, TABLE)
    monkeypatch.setattr(drive, "early_case_problem", lambda session, plan: None)   # G1 alone (its stress is G3's)
    out, sent = _thin_run(tmp_path, monkeypatch, "0.01", PLAN)
    assert "Use table" not in str(out.get("error")) and sent            # past the gate: a step was sent
    session_with(tmp_path, table(rows={"A2": {"unit": "kN"}}))
    out, sent = _thin_run(tmp_path, monkeypatch, "0.01", PLAN)
    assert out["stop"] == "error" and not sent and "Use table row A2" in out["error"]


# ------------------------------------------------------------------------- G2 --

def test_g2_a_held_span_under_half_the_length_held_is_refused(tmp_path):
    session = _rod(tmp_path)
    contacts = {"hand hold": {"transmits": "force and moment", "kind": "grip", "over_m": 0.2}}
    rows = lambda region: [{"name": region, "region": region, "contact": "hand hold"}]   # noqa: E731
    [said] = drive.grip_span_problems(session, {"settings": {}}, rows("end"), contacts)
    assert "hand hold (grip, over_m 0.2 m)" in said and "one flat face" in said
    assert drive.grip_span_problems(session, {"settings": {}}, rows("band"), contacts) == []
    long = {"hand hold": {**contacts["hand hold"], "over_m": 0.5}}
    assert "under 0.5 x the length it holds" in drive.grip_span_problems(session, {"settings": {}}, rows("band"),
                                                                         long)[0]


def _rest_brief(session: Path, extra: str = "") -> None:
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n- base: rests on a support surface\n{"base": {"transmits": "force only", "kind": "rest"}'
        + extra + "}\n")


def test_g2_a_rest_only_case_that_tips_is_refused_and_passes_with_a_grip_or_a_capped_load(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    _rest_brief(session)
    plan = {"values": {"acceleration": G},
            "contacts": {"base": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}}
    push = [{"region": "top", "force_N": [20.0, 0, 0]}]
    said = drive.hold_gate(session, plan, [], [BASE, TOP], push)
    assert "rests clamped against 3.8 N m of overturning" in said and "at 0.232 N, under the push that tips it" in said
    assert "beside base, which stays a support" in said
    # capped at the push that tips it
    assert drive.hold_gate(session, plan, [], [BASE, TOP], [{"region": "top", "force_N": [0.25, 0, 0]}]) is None
    # a grip beside the rest takes the moment
    _rest_brief(session, ', "hand": {"transmits": "force and moment", "kind": "grip", "over_m": 0.05}')
    grip = {"name": "grip", "category": "dirichlet", "mean_normal": [1.0, 0.0, 0.0],
            "centroid": [0.025, 0.0, 0.0], "area": 0.002}
    gripped = {**plan, "contacts": {**plan["contacts"], "hand": {"transmits": "force and moment", "regions": ["grip"]}}}
    assert drive.hold_gate(session, gripped, [], [BASE, grip, TOP], push) is None


def test_g2_the_tipping_cap_is_given_for_each_side_row(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    _rest_brief(session)
    plan = {"values": {"acceleration": G},
            "contacts": {"base": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}}
    side = {"name": "side", "category": "neumann", "centroid": [0.025, 0.09, 0.0]}
    monkeypatch.setattr(drive, "use_table", lambda operator, brief=None: ({"sizes": {"A1": (15.0, "N"),
                                                                                   "A2": (5.0, "N")}}, []))
    monkeypatch.setattr(drive, "load_bindings", lambda s, p, r: {"A1": {"regions": ["top"]}, "A2": {"regions": ["side"]}})
    push = [{"region": "top", "force_N": [15.0, 0, 0]}, {"region": "side", "force_N": [5.0, 0, 0]}]
    tips = drive.tipping_excess(session, plan, [BASE, TOP, side], push)
    scale = tips["tipping_force"] / tips["side_force"]
    # each row scaled alike, rounded down to 3 figures, under the push that tips it by the margin
    assert all(0.99 * size * scale <= tips["caps"][k] <= size * scale for k, size in (("A1", 15), ("A2", 5)))
    said = drive.tipping_exits(tips, plan)
    assert f"A1 {tips['caps']['A1']:.3g} N" in said and f"A2 {tips['caps']['A2']:.3g} N" in said
    # the capped loads clear the check in one step
    capped = [{"region": "top", "force_N": [tips["caps"]["A1"], 0, 0]},
              {"region": "side", "force_N": [tips["caps"]["A2"], 0, 0]}]
    assert drive.tipping_excess(session, plan, [BASE, TOP, side], capped) is None
    # a vertical load far off the base tips it alone: said, never a side cap
    far = {"name": "far", "category": "neumann", "centroid": [0.5, 0.09, 0.0]}
    tips = drive.tipping_excess(session, plan, [BASE, far], [{"region": "far", "force_N": [0, -50.0, 0]}])
    assert tips["vertical_tips"] and "the vertical loads alone tip it" in drive.tipping_exits(tips, plan)


def test_g2_a_fastening_takes_the_moment_only_over_a_spanned_length(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    push = [{"region": "top", "force_N": [20.0, 0, 0]}]
    plan = {"values": {"acceleration": G},
            "contacts": {"base": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}}
    # a fastening held on a spot, force only: it holds as a rest does, and the tipping refusal stands
    _rest_brief(session, ', "fastener": {"transmits": "force only", "kind": "fastening"}')
    spot = {"name": "spot", "category": "dirichlet", "mean_normal": [0.0, -1.0, 0.0],
            "centroid": [0.0, -0.05, 0.0], "area": 1e-4}
    fastened = {**plan, "contacts": {**plan["contacts"], "fastener": {"transmits": "force only", "regions": ["spot"]}}}
    monkeypatch.setattr(drive, "_physical_mesh", lambda s: None)
    assert drive.tipping_excess(session, fastened, [BASE, spot, TOP], push)
    # "force and moment" over a length its held faces do not span: no moment, and the span is refused
    _rest_brief(session, ', "fastener": {"transmits": "force and moment", "kind": "fastening", "over_m": 0.2}')
    short = {"span_m": 0.01, "across_m": 0.0, "one_facing": 1.0, "leaf_m": 0.001, "leaf7_m": 0.001}
    monkeypatch.setattr(drive, "held_shape", lambda s, regions, depth: short)
    said = drive.hold_gate(session, fastened, [{"region": "spot", "contact": "fastener"}], [BASE, spot, TOP], push)
    assert "fastener (fastening, over_m 0.2 m): its held faces span" in said and "rests clamped" in said
    # a span that passes takes the moment
    monkeypatch.setattr(drive, "held_shape", lambda s, regions, depth: {**short, "span_m": 0.15, "across_m": 0.1})
    assert drive.hold_gate(session, fastened, [{"region": "spot", "contact": "fastener"}], [BASE, spot, TOP], push) is None


def test_g2_a_hang_never_tips(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n{"base": {"transmits": "force only", "kind": "hang"}}\n')
    plan = {"values": {"acceleration": G},
            "contacts": {"base": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}}
    push = [{"region": "top", "force_N": [2.0, 0, 0]}]
    assert drive.tipping_excess(session, plan, [BASE, TOP], push) is None
    assert drive.hold_gate(session, plan, [], [BASE, TOP], push) is None


def test_g2_each_hold_carries_a_reaction_its_kind_can(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    plan = {"values": {"acceleration": G},
            "contacts": {"base": {"transmits": "force and moment", "regions": ["base"]}}}
    # a hang carries its weights x the dynamic factor: a 20 N side push on a 0.2 kg hung object is not
    (session / "operator" / "brief.md").write_text('## Contacts\n{"base": {"transmits": "force only", "kind": "hang"}}\n')
    said = drive.hold_gate(session, {**plan, "contacts": {"base": {"transmits": "force only", "regions": ["base"],
                                                                     "simplified_to": "clamped"}}},
                           [], [BASE, TOP], [{"region": "top", "force_N": [20.0, 0, 0]}])
    assert "the hangs ['base'] take 20.1 N, over 3 x the 1.961 N of weight they carry" in said
    # a person's grip takes up to the Human actions top: a 2000 N thrust is no hand's
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n{"base": {"transmits": "force and moment", "kind": "grip", "over_m": 0.05}}\n')
    said = drive.hold_gate(session, plan, [], [BASE, TOP], [{"region": "top", "force_N": [0, -2000.0, 0]}])
    assert "the holds ['base'] (grip, socket or worn) take 2002 N, over the 500 N a person holds" in said
    assert drive.hold_gate(session, plan, [], [BASE, TOP], [{"region": "top", "force_N": [0, -200.0, 0]}]) is None
    # a rest takes compression only: a load lifting the object off it is refused
    (session / "operator" / "brief.md").write_text('## Contacts\n{"base": {"transmits": "force only", "kind": "rest"}}\n')
    rest = {**plan, "contacts": {"base": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}}
    said = drive.hold_gate(session, rest, [], [BASE, TOP], [{"region": "top", "force_N": [0, 30.0, 0]}])
    assert "the rests ['base'] would pull the object down by 28.04 N" in said


def test_g2_an_absent_member_needs_no_number(tmp_path):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "operator" / "brief.md").write_text(
        '## Size and material\nReal mass: 1 kg\nLoad path: {"arm": "series", "base": "series"}\n')
    body = {"kept": "the arm", "left_out": "the trim"}
    said = drive.path_gate(session, {"body_simplification": {**body, "absent": {"base": "small"}}})
    assert "base: absent gives \"minor\" or \"major\" (or nothing), not 'small'" in said
    for absent in (["base"], {"base": "minor"}, {"base": "major"}, {"base": None}, {"base": 0.12}):
        assert drive.path_gate(session, {"body_simplification": {**body, "absent": absent}}) is None, absent
    assert drive.absent_members({"body_simplification": {"absent": ["base"]}}) == {"base": None}
    # every member not declared absent is kept: no free-text "kept" is read
    assert drive.path_gate(session, {"body_simplification": body}) is None
    assert drive.path_gate(session, {"body_simplification": {}}) is None


# ------------------------------------------------------------------------- G3 --

def _g3_session(tmp_path, moment=13.0):
    """The Use table's case run at the Real section: ``moment`` N m on a 0.01 m deep plate, one push
    on the largest Real extent (0.3 m), the runner's lever before any region."""
    session = session_with(tmp_path, table(rows={"A1": {"size": round(moment / 0.3, 6), "source": "derived: %r N m / 0.3 m" % moment},
                                                 "A2": None, "A3": {"lever_m": 0.01}}))
    assert gate(session) == []
    return session


def _plan(**prediction):
    return {"scale": {"unit_system": "SI (m-kg-s)", "physical_extent": 0.3}, "material": dict(STEEL),
            "prediction": {"max_displacement": 1e-4, "max_stress": 8e7, **prediction}}


def test_g3_a_case_past_the_linear_range_is_said_and_named_in_the_verdict_never_refused(tmp_path, monkeypatch):
    session = _g3_session(tmp_path, moment=43.0)             # 43 N m x 0.005 m / 8.3e-10 m4 = 259 MPa
    found = drive.plausibility(session, _plan(max_stress=2.6e8, max_displacement=0.04))
    assert found["flags"] == ["stress", "deflection"] and found["stress_over_strength"] == pytest.approx(1.103, abs=1e-3)
    assert drive.plausibility_gate(session, _plan(max_stress=2.6e8, max_displacement=0.04)) is None   # no refusal
    note = drive.plausibility_note(session, _plan(max_stress=2.6e8, max_displacement=0.04))
    assert "259 MPa at 'top plate', 1.1 x the lower strength of structural / stainless steel" in note
    assert "deflects 0.04 m on 0.3 m, 0.133 of its largest Real extent" in note
    # at judge: the verdict names what the solve or a prediction puts past the range (the agent's 0.04 m),
    # never the runner's conservative estimate alone (its 259 MPa, the solve reading no stress)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "declared.json", _plan(max_stress=2.6e8, max_displacement=0.04))
    lib.write_json(session / "operator" / "prediction_history.json", [{"run": "r", "solved": {}}])   # a solve ran
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and "past the linear range" in out["error"] and "gives 0.133" in out["error"]
    assert "259 MPa" not in out["error"]
    named = ("The brief's case was run; it disagrees with the linear range: deflection over length 0.133\n"
             "## Deviations\nnone\n")
    out = _judge(tmp_path, monkeypatch, session, named)
    assert out["ok"], out
    report = (session / "operator" / "report.md").read_text()
    assert "past the linear range: the case deflects 0.04 m" in report and "### Past the linear range" in report
    assert "259 MPa at" not in report.split("### Past the linear range")[1]
    # an everyday case well inside: nothing said
    calm = _g3_session(tmp_path, moment=13.0)
    assert drive.plausibility_findings(drive.plausibility(calm, _plan())) == []


def test_g3_a_prediction_is_optional_and_a_drifting_one_is_said_never_refused(tmp_path):
    session = _g3_session(tmp_path)                           # 13 N m: 78.3 MPa
    assert drive.plausibility_gate(session, _plan(max_stress=3e8)) is None
    note = drive.plausibility_note(session, _plan(max_stress=3e8))
    assert "predicted stress 300 MPa is more than 3 x off the runner's 78.31 MPa" in note and "a note" in note
    assert drive.prediction_required({}, session) is None
    # without a prediction the runner's own stands: its stress, and a deflection from the plan's E
    made = drive.runner_prediction(session, {**_plan(), "prediction": None})
    assert made["max_stress"] == pytest.approx(7.831e7, rel=1e-3) and "brief_case" not in made
    assert made["max_displacement"] == pytest.approx(43.33 * 0.3 ** 3 / (3 * 2e11 * 8.3e-10), rel=1e-2)
    plan = drive.with_runner_prediction(session, {k: v for k, v in _plan().items() if k != "prediction"})
    assert plan["prediction"]["source"].startswith("the runner's")
    own = _plan(max_stress=1e8)
    # numbers the agent writes are set aside: the prediction is the runner's
    assert drive.with_runner_prediction(session, own)["prediction"]["source"].startswith("the runner's")


def test_g3_a_solve_five_times_off_needs_the_ratio_and_a_cause_in_the_verdict_line(tmp_path, monkeypatch):
    session = _judge_session(tmp_path, {"max_displacement": 8e-5, "max_stress": 1.2e6})
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and "max_displacement 8.0 x the prediction" in out["error"]
    line = "The brief's case was run; the result disagrees with the prediction, max_displacement 8 x, {}\n## Deviations\nnone\n"
    assert not _judge(tmp_path, monkeypatch, session, line.format("cause unclear"))["ok"]
    assert _judge(tmp_path, monkeypatch, session, line.format("cause not found"))["ok"]


def test_g3_was_run_is_refused_for_each_simplification_the_runner_lists(tmp_path, monkeypatch):
    session = _judge_session(tmp_path, {"max_displacement": 1.1e-5, "max_stress": 1.1e6})
    monkeypatch.setattr(drive, "held_regions", lambda *a, **k: [])
    assert _judge(tmp_path, monkeypatch, session)["ok"]
    for declared, word in (({"body_simplification": {"kept": "the arm", "left_out": "the trim",
                                                     "absent": {"base": 0.1}}}, "member base absent"),
                           ({"contacts": {"base": {"transmits": "force only", "regions": ["r"],
                                                   "absent_in_stand_in": {"carried_by": "held_at_joint"}}}},
                            "contact base has no part on the stand-in")):
        lib.write_json(session / "operator" / "declared.json", declared)
        # a moved hold is a deviation (round 12): "none" is refused (a member absent too, when the
        # brief's Load path or Contacts name it: this brief names none)
        assert _judge(tmp_path, monkeypatch, session)["ok"] is ("contacts" not in declared)
        out = _judge(tmp_path, monkeypatch, session,
                     "The brief's case was run.\n## Deviations\nthe base: the stand-in leaves it out\n")
        report = (session / "operator" / "report.md").read_text()
        assert out["ok"] and "A simplified case was run: " in report and word in report    # written by the runner
    lib.write_json(session / "operator" / "declared.json", {})
    monkeypatch.setattr(drive, "held_regions", lambda *a, **k: [{"region": "r", "contact": "base",
                                                                 "simplified_to": "clamped"}])
    assert _judge(tmp_path, monkeypatch, session)["ok"]
    assert "contact base clamped" in (session / "operator" / "report.md").read_text()
    monkeypatch.setattr(drive, "held_regions", lambda *a, **k: [])
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["thick"]})
    assert _judge(tmp_path, monkeypatch, session)["ok"]
    assert "the section stand-in (thick)" in (session / "operator" / "report.md").read_text()
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "real", "reasons": []})
    monkeypatch.setattr(drive, "use_coverage", lambda operator, brief=None: {"uncovered": {"arm": {
        "case": "handling", "effect": 9.0, "run": 2.0, "declared": "a pressure the workflow cannot apply"}}})
    out = _judge(tmp_path, monkeypatch, session)
    assert out["ok"] and ("member arm not covered by the run state (state 'handling' gives 9 against 2; a pressure "
                          "the workflow cannot apply)") in (session / "operator" / "report.md").read_text()
    # the simplified verdict passes on it, and the report lists it
    out = _judge(tmp_path, monkeypatch, session, "a simplified case was run: the arm's carry is not covered\n"
                                                 "## Deviations\nnone\n")
    assert out["ok"], out
    assert "- member arm not covered by the run state" in (session / "operator" / "report.md").read_text()


# ------------------------------------------------------------------------- G4 --

def test_g4_a_built_body_is_real_within_one_and_a_half_and_section_past_it(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "body_provenance", lambda session: "built")
    route = lambda r: drive.body_route(tmp_path, {}, r)["route"]                  # noqa: E731
    assert route({"member_over_real": 1.89, "mass_over_real": 1.0}) == "real"
    assert route({"member_over_real": 2.0, "mass_over_real": 2.4}) == "section"
    assert route({"member_over_real": 2.8, "mass_over_real": 1.0}) == "section"
    assert route({"member_over_real": 1.2, "mass_over_real": 2.0}) == "real"
    # a thin real member forces no route: a generated body stays generated
    monkeypatch.setattr(drive, "member_unresolvable", lambda session, plan: True)
    monkeypatch.setattr(drive, "body_provenance", lambda session: "triposg")
    assert drive.body_route(tmp_path, {}, {})["route"] == "generated"
    monkeypatch.setattr(drive, "body_provenance", lambda session: None)
    assert drive.body_route(tmp_path, {}, {})["route"] is None


def test_g4_no_step_is_refused_for_a_thin_member(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "member_unresolvable", lambda session, plan: True)
    assert drive.body_gate(tmp_path, {}, free_form_step=True) is None
    assert drive.route_order_problem(tmp_path, "geometry_input", {"description": "x"}, plan={}) is None


def test_g4_a_body_off_the_real_extents_is_refused_unless_a_member_is_declared_absent(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "operator" / "brief.md").write_text("## Size and material\nReal mass: 1 kg\nReal thinnest member: 0.08 m\n"
                                                   "Real extents: 0.3 x 0.2 x 0.1 m\n")
    monkeypatch.setattr(drive, "body_source", lambda s: tmp_path / "body.glb")
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "real", "reasons": []})
    turned = {"physical_extents": [0.1, 0.3, 0.2]}                              # the same body, another frame
    assert drive.body_gate(session, {}, turned) is None
    flat = {"physical_extents": [0.3, 0.2, 0.02]}
    said = drive.body_gate(session, {}, flat)
    assert "the smallest is off by more than 1.5 x" in said
    # a body lying diagonally in a plane is compared along its own principal axes
    diagonal = {"physical_extents": [0.35, 0.35, 0.1], "principal_extents": [0.3, 0.2, 0.1]}
    assert drive.body_gate(session, {}, diagonal) is None
    # the middle extent is not judged (a pose sets it, round 12)
    assert drive.body_gate(session, {}, {"physical_extents": [0.35, 0.35, 0.1]}) is None
    # a uniformly reduced body passes (a marked simplification); one that also lost an extent does not
    assert drive.body_gate(session, {}, {"physical_extents": [0.15, 0.1, 0.05]}) is None
    assert "the smallest is off" in drive.body_gate(session, {}, {"physical_extents": [0.15, 0.1, 0.01]})
    absent = {"body_simplification": {"kept": "k", "left_out": "l", "absent": {"base": 0.1}}}
    monkeypatch.setattr(drive, "stand_in_problem", lambda s, p: None)
    assert drive.body_gate(session, absent, flat) is None
    # a stand-in keeps all three extents too (round 10): its smallest is held to the real one ...
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["x"]})
    monkeypatch.setattr(drive, "self_weight_problem", lambda s, p: None)
    assert "the smallest is off" in drive.body_gate(session, {}, {"physical_extents": [0.3, 0.2, 0.25]})
    assert drive.body_gate(session, {}, {"physical_extents": [0.3, 0.2, 0.14]}) is None
    # a slender rod's middle extent is its thickness too: the stand-in grows it
    (session / "operator" / "brief.md").write_text("## Size and material\nReal mass: 1 kg\n"
                                                   "Real thinnest member: 0.012 m\n"
                                                   "Real extents: 2.1 x 0.012 x 0.012 m\n")
    assert drive.body_gate(session, {}, {"physical_extents": [2.1, 0.08, 0.08]}) is None
    # the whole body at a reduced scale is no refusal: a marked simplification the verdict lists
    assert drive.body_gate(session, {}, {"physical_extents": [1.0, 0.08, 0.08]}) is None
    assert "at 0.48 x the brief's Real size" in drive.scale_deviation(session, {"physical_extents": [1.0, 0.08, 0.08]})
    assert "the largest is off" in drive.body_gate(session, {}, {"physical_extents": [4.0, 0.08, 0.08]})


def test_g4_no_cut_or_route_field_is_accepted(tmp_path):
    from test_operator_scripts import PLAN
    for key in ("route", "about", "load_path", "cut_across", "at_m", "piece", "absent_in_stand_in"):
        body = {"kept": "k", "left_out": "l", key: "x"}
        with pytest.raises(drive.PlanError, match=re.escape(f"plan.body_simplification.{key}")):
            drive.check_plan({**PLAN, "body_simplification": body})
    with pytest.raises(drive.PlanError, match="plan.free_form_body is gone"):
        drive.check_plan({**PLAN, "free_form_body": "a twisted blade"})
    # the gone field points to the form the references give, a list of members
    assert '"absent": ["<member>", ...]' in drive.FIELDS_GONE["body_simplification.absent_in_stand_in"]
    # a free-form description: a few words, no negation, no mesh words; a drawing's fits the drawing model
    short = "grey steel part, two flat faces, a round bore"
    assert drive.check_plan({**PLAN, "geometry": {"description": short, "draw": True}})
    for words, said in (("grey steel part, solid, watertight", "drop ['solid', 'watertight']"),
                        ("grey steel part with no handle", "drop ['no']"),
                        (", ".join(["a grey steel part with two flat faces"] * 6), "of the drawing model's 77 tokens")):
        with pytest.raises(drive.PlanError, match=re.escape(said)):
            drive.check_plan({**PLAN, "geometry": {"description": words, "draw": True}})
    # the tokens are counted as CLIP counts them (its local vocabulary), the style included: a
    # hyphen-heavy 15-word description is 66 of 77 and passes, a 17-word one at 79 is refused
    if drive.clip_merges() is not None:
        assert drive.clip_tokens(drive.drawing_style()) == (32, True)
        fits = "a grey steel part, member-to-member joints, a low-profile cross-section, two side-by-side faces, an off-centre bore"
        assert drive.clip_tokens(f"{fits}, {drive.drawing_style()}") == (66, True)
        assert drive.check_plan({**PLAN, "geometry": {"description": fits, "draw": True}})
        over = ("a grey steel part, member-to-member joints, low-profile cross-sections, side-by-side faces, "
                "off-centre bores, chamfered edges, re-entrant corners, thick-walled")
        with pytest.raises(drive.PlanError, match="is 79 of the drawing model's 77 tokens"):
            drive.check_plan({**PLAN, "geometry": {"description": over, "draw": True}})
    # without the vocabulary the estimate counts the style as CLIP does, and passes a short description
    with mock.patch.object(drive, "clip_merges", lambda: None):
        assert drive.clip_tokens(drive.drawing_style()) == (32, False)
        assert drive.check_plan({**PLAN, "geometry": {"description": short, "draw": True}})
    assert drive.check_plan({**PLAN, "geometry": {"description": ", ".join([short] * 3)}})   # TripoSG text: no token cap
    assert drive.check_plan({**PLAN, "geometry": {"description": "a part without fillets", "generator": "cadcoder"}})
    with pytest.raises(drive.PlanError, match="plan.body_up is the body axis"):
        drive.check_plan({**PLAN, "body_up": "z"})
    with pytest.raises(drive.PlanError, match="plan.contacts.<contact>.fastened is gone"):
        drive.check_plan({**PLAN, "contacts": {"base": {"transmits": "force only", "fastened": "bolted"}}})
    with pytest.raises(drive.PlanError, match="plan.loads.<load>.dropped is gone"):
        drive.check_plan({**PLAN, "loads": {"wind": {"dropped": "small"}}})
    assert drive.check_plan({**PLAN, "body_simplification": {"kept": "k", "left_out": "l", "absent": {"base": 0.2}}})


def test_the_contacts_are_frozen_at_the_body_for_every_check(tmp_path, monkeypatch):
    # the reviewer's probe: a rest renamed after the body used to escape tipping
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    first = '## Contacts\n- base: rests\n{"base": {"transmits": "force only", "kind": "rest"}}\n'
    (session / "operator" / "brief.md").write_text(first)
    (session / "brief_snapshots").mkdir()
    (session / "brief_snapshots" / "brief.first.md").write_text(first)
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n- feet: rests\n{"feet": {"transmits": "force only", "kind": "rest"}}\n')
    renamed = {"values": {"acceleration": G},
               "contacts": {"feet": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}}
    said = drive.hold_gate(session, renamed, [], [BASE, TOP], [{"region": "top", "force_N": [20.0, 0, 0]}])
    assert "plan.contacts.feet is not a contact of the brief as the body was made (['base'])" in said
    assert "frozen at the body" in said and "regenerate it, or rebuild it with `blank`" in said
    assert "contacts ['feet'] differ from the body's" in said
    # a body on disk keys the record: an over_m edited after it is refused, a new body takes the brief as it stands
    body = tmp_path / "body.glb"
    body.write_bytes(b"x" * 10)
    monkeypatch.setattr(drive, "body_source", lambda s: body)
    operator = session / "operator"
    (session / "brief_snapshots" / "brief.first.md").write_text(
        '## Contacts\n{"base": {"transmits": "force and moment", "kind": "grip", "over_m": 0.1}}\n')
    (operator / "brief.md").write_text(
        '## Contacts\n{"base": {"transmits": "force and moment", "kind": "grip", "over_m": 0.5}}\n')
    assert drive.brief_contacts_v2(operator)["base"]["over_m"] == 0.1
    [said] = drive.contacts_frozen_problems(operator)
    assert said.startswith("contacts ['base'] differ from the body's: the run state's contacts and their over_m are "
                           "frozen")
    body.write_bytes(b"y" * 20)                                   # a new body
    assert drive.brief_contacts_v2(operator)["base"]["over_m"] == 0.5
    assert drive.contacts_frozen_problems(operator) == []


def test_the_contacts_are_the_run_states_and_a_new_run_state_takes_its_own(tmp_path, monkeypatch):
    # a contact only another state holds is no contact of the checks, plan.contacts or the snapshot
    extra = CONTACTS.replace("}}", '}, "shelf": {"transmits": "force only", "kind": "rest"}}')
    session = session_with(tmp_path, table(rows={"A3": {"held": ["shelf"]}}), contacts=extra)
    operator = session / "operator"
    assert sorted(drive.brief_contacts_v2(operator)) == ["bottom plate", "side grip"]
    assert gate(session) == []
    # a body made with the working state run; the run moved to handling: its contacts, not frozen ones
    body = tmp_path / "body.glb"
    body.write_bytes(b"x" * 10)
    monkeypatch.setattr(drive, "body_source", lambda s: body)
    assert sorted(drive.brief_contacts_v2(operator)) == ["bottom plate", "side grip"]
    handled = session_with(tmp_path, table(rows={"A3": {"held": ["side grip"], "effect": {"side grip": 30}}},
                                           run="handling", uncovered={"bottom plate": {"limit": "temperature", "why": "x"}}))
    assert sorted(drive.brief_contacts_v2(handled / "operator")) == ["side grip"]
    assert drive.contacts_frozen_problems(handled / "operator") == []


def test_the_freeze_is_kept_per_run_state_and_an_invalid_run_never_snapshots(tmp_path, monkeypatch):
    session = session_with(tmp_path, TABLE)
    operator = session / "operator"
    body = tmp_path / "body.glb"
    body.write_bytes(b"x" * 10)
    monkeypatch.setattr(drive, "body_source", lambda s: body)
    assert sorted(drive.brief_contacts_v2(operator)) == ["bottom plate", "side grip"]       # working, frozen
    renamed = CONTACTS.replace('"bottom plate": {', '"floor": {').replace("- bottom plate:", "- floor:")
    # an invalid run reads every contact: no snapshot is taken on it
    session_with(tmp_path, table(run="nowhere", rows={"A1": {"held": ["side grip", "floor"]}}), contacts=renamed)
    drive.brief_contacts_v2(operator)
    record = lib.read_json(session / "brief_snapshots" / drive.CONTACTS_AT_BODY)
    assert list(record["runs"]) == ["working"]
    # back on the working state with a renamed contact: still the body's, and refused
    session_with(tmp_path, table(rows={"A1": {"held": ["side grip", "floor"]}}), contacts=renamed)
    assert sorted(drive.brief_contacts_v2(operator)) == ["bottom plate", "side grip"]
    assert any("frozen at the body" in s for s in gate(session))


def test_a_plural_member_is_two_bars_at_least(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "operator" / "brief.md").write_text(
        '## Size and material\nReal mass: 1 kg\nLoad path: {"top": "series", "legs": "parallel"}\n')
    monkeypatch.setattr(drive, "bars_count", lambda plan: 2)
    said = drive.path_gate(session, {"body_simplification": {"left_out": "x"}})
    assert "the --bars body has 2 bars and boxes for the 2 members kept (['top', 'legs']), at least 3 bars" in said
    monkeypatch.setattr(drive, "bars_count", lambda plan: 3)
    assert drive.path_gate(session, {"body_simplification": {"left_out": "x"}}) is None
    assert not drive._plural("chassis") and drive._plural("front legs") and not drive._plural("axis")


def test_a_run_of_the_wrong_form_and_a_gate_that_raises_are_refusals(tmp_path, monkeypatch):
    said = gate(session_with(tmp_path, table(run=["A1", "A2"])))
    assert any('the Use table\'s "run" is ["A1", "A2"]: the state the solve applies' in s for s in said)
    said = gate(session_with(tmp_path, table(run={"work": 1})))
    assert any('"run" is {"work": 1}' in s for s in said)
    monkeypatch.setattr(drive, "use_table", lambda op, brief=None: 1 / 0)
    [said] = gate(session_with(tmp_path, TABLE))
    assert said.startswith("use gate could not read the case (ZeroDivisionError")
    monkeypatch.setattr(drive, "brief_members", lambda op: 1 / 0)
    assert drive.path_gate(tmp_path, {"body_simplification": {"kept": "k"}}).startswith("path gate could not read")
    said, info = drive.verdict_gate("x", tmp_path / "missing")
    assert info == {} and (said is None or isinstance(said, str))


def test_a_given_size_is_a_number_of_the_instruction(tmp_path):
    said = gate(session_with(tmp_path, table(rows={"A1": {"source": "given", "size": 200}})))
    assert any('"source": "given", but the user\'s instruction states no 200' in s for s in said)
    session = session_with(tmp_path, table(rows={"A1": {"source": "given", "size": 196.1}}),
                           instruction="a test part that carries 20 kg")
    assert not any("given" in s for s in gate(session))           # 20 kg x g


def test_a_lever_longer_than_the_object_is_the_objects(tmp_path):
    # the agent's one number is bounded by the Real extents, never refused
    session = session_with(tmp_path, table(rows={"A1": {"lever_m": 0.5}}))
    assert gate(session) == []
    built, _ = drive.use_table(session / "operator")
    assert built["effects"]["A1"]["top plate"] == pytest.approx(200 * 0.3)


def test_the_solve_is_held_against_the_runners_stress_at_judge(tmp_path, monkeypatch):
    from test_case_realism import STRIP, _use_table_session
    session = _use_table_session(tmp_path, 40.0)                  # 259 MPa from the table
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "declared.json", {**STRIP, "prediction": {"max_displacement": 1e-5}})
    history = [{"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 10.0,
                "prediction": {"max_displacement": 1e-5}, "solved": {"max_displacement": 1.1e-5, "max_stress": 2e7}}]
    lib.write_json(session / "operator" / "prediction_history.json", history)
    checked = drive.fem_against_runner(session, STRIP)
    assert checked["fem_stress_MPa"] == 20 and checked["runner_stress_MPa"] == pytest.approx(259.04, abs=0.01)
    # the runner's stress is a conservative bound: a solve far under it is no wrong case
    out = _judge(tmp_path, monkeypatch, session)
    assert out["ok"], out
    report = (session / "operator" / "report.md").read_text()
    assert "the solve's peak stress 20 MPa against the runner's 259 MPa at 'top plate'" in report
    # far over it is a report line too (the solve's peak may sit in a thinner member), while the solve's
    # own stress past the strength is named in the verdict
    (session / "operator" / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    history[0]["solved"]["max_stress"] = 1.5e9
    lib.write_json(session / "operator" / "prediction_history.json", history)
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and "fem_stress_over_runner" not in out["error"]
    assert "past the linear range" in out["error"] and "6.38" in out["error"]
    out = _judge(tmp_path, monkeypatch, session, "The brief's case was run; it disagrees with the strength, "
                                                 "6.38 x, cause not found\n## Deviations\nnone\n")
    assert out["ok"], out
    assert "the solve's peak stress 1500 MPa against the runner's 259 MPa" in (session / "operator" / "report.md").read_text()


def test_the_verdict_reads_the_coverage_of_the_table_as_solved(tmp_path):
    limit = {"limit": "dynamic or impact", "why": "the carry is a jolt at the grip",
             "no_approximation": "the jolt has no stated factor"}
    session = session_with(tmp_path, table(rows={"A3": {"size": 400, "source": "40 kg x g x 1"}},
                                           uncovered={"side grip": limit, "bottom plate": limit}))
    snapshots = session / "brief_snapshots"
    (snapshots / "brief.solve.md").write_text((session / "operator" / "brief.md").read_text())
    # the table edited after the solve to clear the uncovered member
    session_with(tmp_path, TABLE)
    assert drive.use_coverage(session / "operator")["uncovered"] == {}
    assert list(drive.solved_coverage(session)["uncovered"]) == ["side grip", "bottom plate"]
    said = drive.runner_simplifications(session, {}, [])
    assert any(s.startswith("member side grip not covered by the run state") and "dynamic or impact: the carry is a jolt at the grip" in s for s in said)


def test_an_absent_member_forgives_one_shorter_extent_and_bodies_keep_their_outline(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "operator" / "brief.md").write_text("## Size and material\nReal mass: 1 kg\nReal thinnest member: 0.08 m\n"
                                                   "Real extents: 0.3 x 0.2 x 0.1 m\n")
    monkeypatch.setattr(drive, "body_source", lambda s: tmp_path / "body.glb")
    monkeypatch.setattr(drive, "stand_in_problem", lambda s, p: None)
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "real", "reasons": []})
    absent = {"body_simplification": {"kept": "k", "left_out": "l", "absent": {"base": 0.1}}}
    assert drive.body_gate(session, absent, {"physical_extents": [0.3, 0.2, 0.02]}) is None
    # an absent member that shortens the largest extent: the other extents match at scale 1, taken
    assert drive.body_gate(session, absent, {"physical_extents": [0.19, 0.18, 0.1]}) is None
    said = drive.body_gate(session, absent, {"physical_extents": [0.3, 0.6, 0.02]})
    assert "the largest is off" in said
    # a generated, CAD or uploaded body: the largest and a wall's smallest too (round 10: one 3.5 x off its
    # smallest passed); the middle, a pose, is not judged (round 12)
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "generated", "reasons": []})
    assert "the smallest is off" in drive.body_gate(session, {}, {"physical_extents": [0.3, 0.2, 0.02]})
    assert drive.body_gate(session, {}, {"physical_extents": [0.3, 0.2, 0.08]}) is None
    assert "the smallest is off" in drive.body_gate(session, {}, {"physical_extents": [0.3, 0.05, 0.08]})


def test_a_malformed_contacts_entry_is_named(tmp_path):
    bad = CONTACTS.replace('"force only"', '"force only (rests)"')
    said = gate(session_with(tmp_path, TABLE, contacts=bad))
    assert any("Contacts JSON entry 'bottom plate': \"transmits\" is \"force only (rests)\", not one of "
               "['force and moment', 'force only']" in s for s in said)
    assert not any("are not contacts of the brief's Contacts JSON ([])" in s for s in said)


def test_the_real_section_member_is_loaded_and_the_tiers_come_at_once(tmp_path):
    # every row enters after the Real section's member on the Load path: nothing loads it
    size = SIZE.replace('Load path: {"top plate": "series", "side grip": "series", "bottom plate": "series"}',
                        'Load path: {"bottom plate": "series", "top plate": "series", "side grip": "series"}')
    unloaded = size.replace("Real section (top plate)", "Real section (bottom plate)")
    session = session_with(tmp_path, TABLE, size=unloaded)
    assert gate(session) == []
    # each row's path runs from its entry to its holds, whatever the Load path's list order
    assert drive.use_coverage(session / "operator")["run_effect"] == pytest.approx(75.0)          # whatever the order
    assert drive.runner_case(session, {})["moment_Nm"] == pytest.approx(75.0)
    # a table problem and a contact problem in one answer
    no_length = CONTACTS.replace(', "over_m": 0.1', "")
    said = gate(session_with(tmp_path, table(rows={"A1": {"size": 0}}), contacts=no_length))
    assert any('"size" is a number over 0' in s for s in said) and any('"over_m"' in s for s in said)


# ------------------------------------------------------ the runner does the work --

def test_a_slender_body_is_remeshed_at_its_cross_section_under_a_face_cap():
    import trimesh
    needle = trimesh.creation.cylinder(radius=0.0035, height=1.07, sections=32)     # a rod over 100 x its thickness
    edge = drive.blank_edge(needle)
    assert edge >= 0.007 / 3 * 0.999                                            # a third of its thickness
    assert needle.area / (0.433 * edge ** 2) <= drive.SLENDER_FACES * 1.01
    # a flat strip keeps faces across its thin side: the edge is no more than its thickness
    strip = trimesh.creation.box(extents=[1.0, 0.03, 0.002])
    assert drive.blank_edge(strip) <= max(0.002, (strip.area / (0.433 * drive.SLENDER_FACES)) ** 0.5) * 1.001
    block = trimesh.creation.box(extents=[0.3, 0.2, 0.1])                         # not slender: as before
    assert drive.blank_edge(block) == pytest.approx(max(min(0.3 / 150, 0.1 / 6), (block.area / (0.433 * 80_000)) ** 0.5))


@pytest.mark.real_remesh
def test_the_remesher_past_its_time_leaves_the_subdivided_body(monkeypatch):
    import subprocess
    import trimesh
    monkeypatch.setattr("inspect_session.compute_interpreter", lambda env: sys.executable)

    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="remesh", timeout=drive.REMESH_SECONDS)
    monkeypatch.setattr(subprocess, "run", slow)
    out = drive.remesh_built(trimesh.creation.box(extents=[0.3, 0.2, 0.1]))
    assert out.metadata["remesh"] == f"stopped after {drive.REMESH_SECONDS} s" and len(out.faces) > 12


def test_the_stand_ins_self_weight_is_the_runners_to_scale_never_a_body_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(drive, "body_source", lambda s: tmp_path / "body.glb")
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["thin"]})
    monkeypatch.setattr(drive, "self_weight_problem", lambda s, p: "Scale the self-weight")
    monkeypatch.setattr(drive, "real_extents", lambda op: None)
    assert drive.body_gate(tmp_path, {}, {"physical_extents": [1, 1, 1]}) is None


def test_judge_ends_a_session_without_a_solve_on_could_not_be_run(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and "without a solve the verdict is 'the brief's case could not be run'" in out["error"]
    not_run = ("The brief's case could not be run: the stand-in cannot carry the member\n"
               "## Deviations\nthe member is under the resolution\n")
    # nothing recorded kept it from running: refused, and no report is left behind
    out = _judge(tmp_path, monkeypatch, session, not_run)
    assert not out["ok"] and "the runner recorded nothing" in out["error"]
    assert not (session / "operator" / "report.md").exists()
    # so "was run" still has no report to pass on, and no solve: refused
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and not (session / "operator" / "report.md").exists()
    # a body G4 refused is a blocker: the verdict passes, the report names it
    drive.record_blocker(session, "the body (G4)", "this body cannot be the stand-in: its member is 0.8 leaves")
    out = _judge(tmp_path, monkeypatch, session, not_run)
    assert out["ok"], out
    report = (session / "operator" / "report.md").read_text()
    assert "No solve was run" in report and "the stand-in cannot carry the member" in report
    assert "- the body (G4): this body cannot be the stand-in: its member is 0.8 leaves" in report
    # and with the report there, "was run" is still refused without a solve
    out = _judge(tmp_path, monkeypatch, session)
    assert not out["ok"] and "no solve was run in this session" in out["error"]


def test_a_body_refusal_and_a_run_limit_are_recorded_as_blockers(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    monkeypatch.setattr(drive, "body_source", lambda s: tmp_path / "body.glb")
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "generated", "reasons": []})
    monkeypatch.setattr(drive, "stand_in_problem", lambda s, p: "the body weighs 9 x the real object")
    monkeypatch.setattr(drive, "real_extents", lambda op: None)
    assert drive.body_gate(session, {"body_simplification": {"absent": {}}}, {"physical_extents": [1, 1, 1]})
    assert drive.blockers(session)[-1]["kind"] == "body gate"
    assert drive.blockers(session)[-1]["text"] == "the body weighs 9 x the real object"


def test_a_minor_run_row_is_printed_not_bound(tmp_path):
    from test_load_path_and_holds import REGIONS
    # A2 at 15 N beside the run's 200 N: under a tenth of it, by the sizes (the effects are the agent's)
    session = session_with(tmp_path, table(rows={"A2": {"size": 15, "source": "derived: 15 N",
                                                         "effect": {"top plate": 0.5, "side grip": 0.1}}}))
    plan = {"values": {"acceleration": G}, "loads": {"A1": {"regions": ["end_pull"]}}}
    bound = drive.load_bindings(session, plan, REGIONS)
    assert bound["A2"] == {"state": "minor"}
    assert drive.loads_binding_problem(session, plan, REGIONS) is None


def test_acceleration_use_alone_is_the_acceleration_to_start_from(tmp_path, monkeypatch):
    from test_frames_and_runner_writes import _heavy_stand_in
    path, plan, _ = _heavy_stand_in(tmp_path, monkeypatch)
    alone = {**plan, "values": {"load": {"force": [0, -20, 0]}, "acceleration_use": [0, -9.80665, 0]}}
    path.write_text(json.dumps(alone) + "\n")
    out, said = drive.scaled_self_weight(tmp_path, path, alone)
    assert out["values"]["acceleration"] == [0.0, -round(9.80665 * 30 / 600, 5), 0.0] and said


def test_the_prediction_for_the_use_table_is_compared_with_the_final_solve_unscaled():
    first = {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 10.0, "table_loads": True,
             "prediction": {"max_displacement": 1e-3}, "solved": {"max_displacement": 4e-4}}
    final = {**first, "run": "r2", "load_N": 25.0, "prediction": None, "solved": {"max_displacement": 8.2e-4}}
    agreed = drive.agreement([first, final])
    assert agreed["load_scale"] == 1.0 and agreed["max_displacement"]["ratio"] == 0.82
    legacy = [{**first, "table_loads": False}, {**final, "table_loads": False}]
    assert drive.agreement(legacy)["load_scale"] == 2.5


# --------------------------------------------------------------- round four --

def test_a_hand_that_drives_the_action_is_its_load_not_a_hold(tmp_path):
    # A1 is pushed by the hand (grip in the anchor) and reacted by the bottom plate: the row may leave
    # the grip out; with no other contact it may not
    said = gate(session_with(tmp_path, table(rows={"A1": {"held": ["bottom plate"]}})))
    assert not any("have no grip" in s for s in said)
    rest_only = '- bottom plate: rests\n{"bottom plate": {"transmits": "force only", "kind": "rest"}}'
    said = gate(session_with(tmp_path, table(rows={"A2": {"held": []}}), contacts=rest_only))
    assert any('"held" names the contacts' in s for s in said)


def test_a_contact_a_gate_names_as_wrong_may_change_once_after_the_body(tmp_path, monkeypatch):
    session = session_with(tmp_path, TABLE)
    operator = session / "operator"
    body = tmp_path / "body.glb"
    body.write_bytes(b"x" * 10)
    monkeypatch.setattr(drive, "body_source", lambda s: body)
    assert drive.brief_contacts_v2(operator)["side grip"]["over_m"] == 0.1                   # frozen
    # a span refusal frees over_m only, and only toward a shorter length
    span = "side grip (grip, over_m 0.1 m): its held faces span 0.03 m along the member"
    assert drive.named_contacts(span) == {"side grip": {"over_m"}}
    longer = CONTACTS.replace('"over_m": 0.1', '"over_m": 0.2')
    session_with(tmp_path, TABLE, contacts=longer)
    drive.unfreeze_contacts(operator, drive.named_contacts(span))
    assert drive.brief_contacts_v2(operator)["side grip"]["over_m"] == 0.1                   # longer: not taken
    shorter = CONTACTS.replace('"over_m": 0.1', '"over_m": 0.05')
    session_with(tmp_path, TABLE, contacts=shorter)
    assert drive.brief_contacts_v2(operator)["side grip"]["over_m"] == 0.05
    assert drive.contacts_frozen_problems(operator) == []
    record = lib.read_json(session / "brief_snapshots" / drive.CONTACTS_AT_BODY)
    assert record["edited"] == ["side grip.over_m"] and record["unfrozen"] == {}
    # a kind refusal frees transmits (and over_m where asked), never the kind; tipping and the cap nothing
    said = 'contact \'bottom plate\' (rest): transmits "force only" (its reaction from statics'
    assert drive.named_contacts(said) == {"bottom plate": {"transmits"}}
    assert drive.named_contacts("the rests ['base'] would pull the object down by 3 N; the holds ['grip'] "
                                "(grip, socket or worn) take 900 N") == {}
    renamed = shorter.replace('"kind": "rest"', '"kind": "grip"')
    session_with(tmp_path, TABLE, contacts=renamed)
    drive.unfreeze_contacts(operator, {"bottom plate": {"transmits"}})
    assert drive.brief_contacts_v2(operator)["bottom plate"]["kind"] == "rest"


def test_every_gate_refusal_is_a_blocker_could_not_be_run_rests_on(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    monkeypatch.setattr(drive, "brief_members", lambda op: None)
    said = drive.path_gate(session, {"body_simplification": {"absent": {}}})
    assert said and drive.blockers(session)[-1]["kind"] == "path gate"
    out = _judge(tmp_path, monkeypatch, session, "The brief's case could not be run: the load path is unknown\n"
                                                 "## Deviations\nnone\n")
    assert out["ok"], out


def test_an_uncovered_limit_matches_the_dominant_row_that_loads_the_member(tmp_path):
    # the dominant row at the side grip is a twist (A3); a small force beside it does not change that
    twist = {"A3": {"unit": "N m", "size": 5, "source": "H7 middle", "effect": {"side grip": 9}, "dynamic": False},
             "+nudge": {"load": "nudge", "size": 2, "unit": "N", "source": "derived: 10 N x 0.2", "state": "handling",
                        "kind": "applied", "enters": "side grip", "held": [], "effect": {"side grip": 0.1}}}
    said = gate(session_with(tmp_path, table(rows=twist, uncovered={"side grip": {
        "limit": "contact between separate bodies", "why": "x"}})))
    assert any("its dominant row A3 is a twist: a torque about an axis is applied as a couple" in s for s in said)
    said = gate(session_with(tmp_path, table(rows=twist, uncovered={"side grip": {"limit": "temperature", "why": "x"}})))
    assert any("its dominant row A3 is no action of the surroundings" in s for s in said)
    said = gate(session_with(tmp_path, table(rows=twist, uncovered={"side grip": {"limit": "dynamic or impact",
                                                                                 "why": "x"}})))
    assert any("its dominant row A3 is neither an action of the surroundings nor marked dynamic" in s for s in said)
    said = gate(session_with(tmp_path, table(rows=twist, uncovered={"side grip": {"limit": "large deflection",
                                                                                 "why": "x"}})))
    assert any("the runner's deflection over length is not over 0.1" in s for s in said)
    marked = {**twist, "A3": {**twist["A3"], "dynamic": True}}
    assert gate(session_with(tmp_path, table(rows=marked, uncovered={"side grip": {
        "limit": "dynamic or impact", "why": "a jolt", "no_approximation": "no factor"}}))) == []


def test_the_load_path_members_are_kept_or_declared_absent(tmp_path):
    session = session_with(tmp_path, TABLE)
    assert drive.path_gate(session, {"body_simplification": {"left_out": "x"}}) is None
    # a part off the load path, absent, carries no load: taken
    assert drive.path_gate(session, {"body_simplification": {"absent": {"spare part": 0.05}}}) is None
    assert drive.path_gate(session, {"body_simplification": {"absent": {"side grip": 0.05}}}) is None
    assert drive._members_match("two side plates", "plate") and not drive._members_match("arm", "base")


def test_command_output_lists_are_capped_and_a_scattered_hold_is_refused():
    out = drive.capped({"pieces": list(range(130)), "nested": {"rows": [{"a": list(range(25))}]}})
    assert len(out["pieces"]) == 21 and out["pieces"][-1].startswith("... 110 more")
    assert out["nested"]["rows"][0]["a"][-1].startswith("... 5 more")
    rows = [{"name": f"base#{k}", "region": "base", "contact": "base", "face": "-y", "member_axis": "x"}
            for k in range(60)]
    assert "held regions ['base'] resolve into 60 pieces: the selector picks scattered faces" in \
        drive.pieces_problem(rows, {"base": {"regions": ["base"]}})


def test_the_human_cap_is_a_persons_hold_only(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    plan = {"values": {"acceleration": G}, "contacts": {"base": {"transmits": "force and moment", "regions": ["base"]}}}
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n{"base": {"transmits": "force and moment", "kind": "socket", "over_m": 0.05}}\n')
    assert drive.hold_gate(session, plan, [], [BASE, TOP], [{"region": "top", "force_N": [0, -2000.0, 0]}]) is None


def test_the_light_touch_row_is_read():
    rows = drive.human_rows()
    assert rows["H11"]["low"] == 1 and rows["H11"]["high"] == 10 and rows["H11"]["unit"] == "N"


def test_a_persons_hold_is_capped_and_a_machines_socket_is_not(tmp_path, monkeypatch):
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    plan = {"values": {"acceleration": G}, "contacts": {"base": {"transmits": "force and moment", "regions": ["base"]}}}
    push = [{"region": "top", "force_N": [0, -2000.0, 0]}]
    table_ = {"cases": {"working": ["A1"]}, "run": "working", "rows": {"A1": {"source": "derived: x"}}}
    grip = {"base": {"kind": "grip", "transmits": "force and moment", "over_m": 0.05}}
    assert "a person holds" in drive.hold_reaction_problem(session, plan, [BASE, TOP], push, grip, table_)
    socket = {"base": {"kind": "socket", "transmits": "force and moment", "over_m": 0.05}}
    assert drive.hold_reaction_problem(session, plan, [BASE, TOP], push, socket, table_) is None


def test_a_gate_that_passes_clears_its_blockers_and_a_draft_brief_records_none(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    monkeypatch.setattr(drive, "brief_members", lambda op: None)
    assert drive.path_gate(session, {"body_simplification": {"absent": {}}})
    assert [b["kind"] for b in drive.blockers(session)] == ["path gate"]
    assert drive.path_gate(session, {}) is None                       # it passes now
    assert drive.blockers(session) == []
    # use_gate on a draft brief at blank records nothing
    draft = session_with(tmp_path, None, name="draft")
    assert drive.use_gate(draft / "operator", record=False)
    assert drive.blockers(draft) == []


# ------------------------------------------------------------ all at once --

def test_the_plan_check_names_every_problem_in_one_refusal(tmp_path):
    plan = {"geometry": {"drawing_crop": "all", "asset_path": "/abs/body.glb", "web_image": {"query": "https://x"}},
            "scale": {}, "material": {},
            "values": {"neumann_values": {"top": "down"}},
            "contacts": {"base": {"transmits": "pinned"}}, "geometry_routes": ["triposg_text"]}
    with pytest.raises(drive.PlanError) as refused:
        drive.check_plan(plan)
    said = str(refused.value)
    for part in ("plan.geometry.drawing_crop is", "plan.geometry.web_image is", "leave draw and asset_path out",
                 "not as an absolute path", "the plan needs 'regions'",
                 "plan.contacts.base is", "values.neumann_values.top must be a traction"):
        assert part in said, part


def test_the_contacts_and_the_body_name_every_problem_in_one_refusal(tmp_path, monkeypatch):
    said = drive.contacts_problem({"base": {"transmits": "force only", "regions": ["r"],
                                            "absent_in_stand_in": {"carried_by": "nowhere"}},
                                   "lid": {"transmits": "force only", "regions": ["s"]}},
                                  {"base": "force only"})
    assert "absent_in_stand_in.carried_by is one of" in said and "lid: not a contact of the brief" in said
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "operator" / "brief.md").write_text("## Size and material\nReal mass: 1 kg\nReal thinnest member: 0.08 m\n"
                                                   "Real extents: 0.3 x 0.2 x 0.1 m\n")
    monkeypatch.setattr(drive, "body_source", lambda s: tmp_path / "body.glb")
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "real", "reasons": []})
    monkeypatch.setattr(drive, "stand_in_problem", lambda s, p: "the body weighs 9 x the real object")
    said = drive.body_gate(session, {"body_simplification": {"left_out": "x"}}, {"physical_extents": [0.3, 0.2, 0.02]})
    assert "the body weighs 9 x the real object" in said and "the smallest is off by more than 1.5 x" in said


def test_the_verdict_gate_names_every_problem_in_one_refusal(tmp_path, monkeypatch):
    session = _judge_session(tmp_path, {"max_displacement": 8e-5, "max_stress": 1.2e6})
    monkeypatch.setattr(drive, "runner_simplifications", lambda *a, **k: ["member arm absent (its share 0.1)"])
    out = _judge(tmp_path, monkeypatch, session)
    # the solve 8 x its prediction, and a simplification the verdict does not claim: both said at once
    assert not out["ok"] and "max_displacement 8.0 x the prediction" in out["error"]


def test_the_judgement_form_names_every_problem_at_once():
    said = drive.judgement_problem("It was fine.\n", {"changes": "x"})
    assert "first line must be the verdict" in said and "'Deviations' heading" in said
    said = drive.judgement_problem("The brief's case was run.\n", {"changes": "x"})
    assert "'Brief changes' section, so the verdict" in said and "'Deviations' heading" in said
    assert drive.judgement_problem("The brief's case was run.\n## Deviations\nnone\n", {}) is None


# ---------------------------------------------------------------------- other --

SKILL = SCRIPTS.parent
TEXT_BUDGET = 157_338          # the skill's text at fd61387, before the object-category rules


def test_skill_text_within_budget():
    total = sum(p.stat().st_size for p in SKILL.rglob("*.md")
                if not any(part.startswith(".") for part in p.relative_to(SKILL).parts))   # caches aside
    assert total <= TEXT_BUDGET, f"operator_skill *.md is {total} bytes, over {TEXT_BUDGET}"


# The judging keyword patterns and the word lists the four gates replaced.  The gates read JSON and
# numbers; a name here reappearing means a check of the agent's prose came back.
DELETED = ("USE_ACTIONS", "DUTY", "NO_SOURCE", "SIZED_BY", "SIZE_ARGUMENT", "STATE_ARGUMENT", "IDLE_STATE",
           "GRIP_HOLD", "REST_HOLD", "USER_BODY", "GRIP_CONTACT", "REST_CONTACT", "ACTS_APART", "WEIGHT_LOAD",
           "ITEM_ABSENT", "FASTENED_WORDS", "BRIEF_FASTENED_WORDS", "HAND_WORDS", "HOLD_PHRASE", "STATED_LENGTH",
           "NEGATED", "INSERTED_WORDS", "FLAT_FASTENED_WORDS", "INERTIA_WORDS", "ONE_PATCH", "DISTRIBUTIVE",
           "FREE_FORM_SHAPE_WORDS", "FREE_FORM_SHAPE_PHRASES", "FREE_FORM_VAGUE", "NOT_REVOLUTION", "CURVATURE_WORDS",
           "TWIST_WORDS", "OUT_OF_PLANE_WORDS", "HEIGHT_GIVEN", "REVOLUTION_WORDS", "REVOLVE_NOUNS",
           "REVOLVE_QUALIFIED", "REVOLVE_NOT_A_BODY", "PLAUSIBILITY_LINE", "REVISED_HEADING", "REVISED_INPUTS",
           "NEGATIONS", "NEGATED_NOUN_FILLERS",
           "anchor_problems", "anchor_state_problems", "comparison_in", "brief_comparison", "use_actions_missing",
           "use_loads_note", "no_source_in", "rest_in", "grip_and_rest_contacts", "drop_share_problems",
           "dropped_apart", "drop_path_problems", "drop_members_warning", "loads_warning", "fastening_problem",
           "fastened_contacts", "held_over_length", "held_inside", "grip_length_check", "grip_length_problem",
           "inertia_relieved", "tipping_problem", "pieces_why_warning", "says_one_patch", "members_problem",
           "free_form_problem", "frame_generation_problem", "route_provenance_problem", "built_route_problem",
           "note_simplification", "removed_simplification", "dropped_simplification_problem",
           "section_cut_problem", "section_cut_warning", "cut_moved_note", "piece_lengthwise", "twist_plane_warning",
           "built_plane", "flattened_frame_warning", "member_height_given", "revolution_named", "revolve_plan_note",
           "plausibility_addressed", "plausibility_problem", "plausibility_said", "revised_prediction",
           "declared_gap_problem", "caveat_sentences", "judged_caveats", "names_declaration", "prediction_gap_problem")


def test_no_judging_pattern_of_the_old_checks_remains():
    source = (SCRIPTS / "drive.py").read_text()
    found = [name for name in DELETED if re.search(rf"\b{name}\b", source)]
    assert found == []
