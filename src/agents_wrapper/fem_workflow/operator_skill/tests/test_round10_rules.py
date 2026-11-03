"""The rules of the round 10 audit: a thin real member generated thicker is a marked stand-in, a sub-leaf
governing member takes an equivalent-stiffness section, the case's statics and plausibility are
checked while the brief is editable, and all three extents are compared."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import drive
import sessionlib as lib


def _session(tmp_path: Path, size: str) -> Path:
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text("## Use\nused\n## Size and material\n" + size)
    return session


THIN_WALL = "Real thinnest member: 0.004 m\nReal extents: 0.4 x 0.3 x 0.3 m\nReal mass: 2 kg\n"


def test_a_generated_body_thickening_a_thin_real_member_is_a_marked_stand_in(tmp_path, monkeypatch):
    session = _session(tmp_path, THIN_WALL)
    monkeypatch.setattr(drive, "body_provenance", lambda s: "triposg")
    filled = {"member_thickness": {"median_m": 0.12}, "physical_extents": [0.4, 0.3, 0.3],
              "mass_over_real": 9.0, "member_over_real": 20.0}
    assert drive.range_findings(filled)                     # over the 5 x / 3 x rule ...
    assert drive.thickened_stand_in(session, filled)["real_wall_m"] == 0.004
    assert drive.outside_range(session, filled) is None     # ... yet no other structure
    assert drive.mass_route_problem(session, {"geometry": {}}, filled) is None
    # thin sheets generated several times too thick, far from solid: the same one stand-in
    sheets = {**filled, "member_thickness": {"median_m": 0.02}, "member_over_real": 1.2}
    assert drive.thickened_stand_in(session, sheets)["measured_m"] == 0.02
    assert drive.outside_range(session, sheets) is None
    # or G4's member measure says so while the voxel median reads thin
    walls = {**filled, "member_thickness": {"median_m": 0.006}, "member_over_real": 4.0}
    assert drive.thickened_stand_in(session, walls)["measured_m"] == pytest.approx(0.016)
    # members near the real thickness, the mass excess from elsewhere: the rule stands
    near = {**filled, "member_thickness": {"median_m": 0.006}, "member_over_real": 1.5}
    assert drive.thickened_stand_in(session, near) is None and drive.outside_range(session, near)
    # a built body, or a real member that is no thin wall: the rule stands
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    assert drive.thickened_stand_in(session, filled) is None
    monkeypatch.setattr(drive, "body_provenance", lambda s: "hunyuan3d")
    (session / "operator" / "brief.md").write_text("## Size and material\n" + THIN_WALL.replace("0.004", "0.05"))
    assert drive.thickened_stand_in(session, filled) is None


def test_a_thickened_stand_in_verdict_is_simplified_and_says_so(tmp_path, monkeypatch):
    session = _session(tmp_path, THIN_WALL)
    (session / "operator" / "report.md").write_text("# Session abc\n\n## Engineer's judgement\n\n_(fill in)_\n")
    lib.write_json(session / "operator" / "prediction_history.json", [
        {"run": "r1", "body": {"sha": "b", "size": 1.0}, "ends": "e", "load_N": 10.0,
         "prediction": {"max_displacement": 1e-5, "max_stress": 1e6},
         "solved": {"max_displacement": 1e-5, "max_stress": 1e6}}])
    monkeypatch.setattr(drive, "runner_simplifications",
                        lambda *a, **k: [drive.THICKENED_SAID + " (the real 0.004 m member generated 0.02 m thick)"])
    monkeypatch.setattr(drive, "outside_range", lambda s, planned=None: None)
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    for verdict in ("The brief's case could not be run.", "The brief's case was run."):
        (session / "operator" / "j.md").write_text(verdict + "\n## Deviations\nnone\n")
        drive.cmd_judge(argparse.Namespace(output_root=str(tmp_path), session_id="abc",
                                           file=str(session / "operator" / "j.md"), text=None))
        report = (session / "operator" / "report.md").read_text()
        assert printed[-1]["ok"] and "A simplified case was run: thick-sheet stand-in: sheets or walls thicker than " \
            "the real ones; stiffer than the real object, displacements are a lower bound, stresses are not the " \
            "real wall's" in report, verdict


BLADE = ("Real thinnest member: 0.001 m\nReal section (blade): rectangle 0.05 x 0.001 m\n"
         "Governing material: steel, E 2e11 Pa\nReal extents: 1 x 0.05 x 0.05 m\nReal mass: 0.4 kg\n")


def test_a_sub_leaf_governing_member_takes_the_equivalent_stiffness_section(tmp_path, monkeypatch):
    session = _session(tmp_path, BLADE)
    plan = {"material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850.0}}
    eq = drive.equivalent_section(session, 1.0, plan)
    # 2 leaves at depth 8 under a 1 m body; the width kept, the depth raised, E scaled to keep EI
    assert eq["dims_m"] == [0.05, 0.015] and eq["thinnest_m"] == 0.015
    assert abs(eq["E_factor"] * 0.015 ** 3 / 0.001 ** 3 - 1) < 1e-3
    assert abs(eq["plan_material"]["youngs_modulus"] / (2e11 * eq["E_factor"]) - 1) < 1e-3
    assert "equivalent-stiffness section" in eq["said"] and "exempt from G4" in eq["said"]
    assert drive.equivalent_section(session, 0.05, plan) is None          # a small body resolves it
    # built at it: recorded, exempt from the 3 x / 5 x rule, its E the scaled one, marked
    body = session / "operator" / "blank_box_x.glb"
    assert drive.record_equivalent(session, body, 0.015, 1.0, plan)["built_at_it"]
    assert not drive.record_equivalent(session, session / "operator" / "other.glb", 0.05, 1.0, plan)["built_at_it"]
    lib.write_json(session / "session.json", {"input_asset": {"source_name": body.name}})
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    assert drive.equivalent_body(session)["section"] == eq["section"]
    assert drive.outside_range(session, {"member_over_real": 15.0, "mass_over_real": 9.0}) is None
    refused, _ = drive.governing_material_check(session, plan)
    assert refused and "x 0.000296" in refused
    scaled = {"material": {**plan["material"], "youngs_modulus": eq["plan_material"]["youngs_modulus"]}}
    assert drive.governing_material_check(session, scaled) == (None, None)
    assert drive.subleaf_problem(session, plan, {"physical_extents": [1.0, 0.05, 0.015]}) is None
    said = drive.runner_simplifications(session, {}, [])
    assert any(s.startswith("equivalent-stiffness section: the real rectangle 0.05 x 0.001 m") for s in said)


def test_a_body_with_a_sub_leaf_governing_member_is_refused_before_every_solve(tmp_path, monkeypatch):
    session = _session(tmp_path, BLADE)
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    monkeypatch.setattr(drive, "body_digest", lambda s: "real-section body")
    resolution = {"physical_extents": [1.0, 0.05, 0.001]}
    said = drive.subleaf_problem(session, {}, resolution)
    assert said and "equivalent-stiffness section" in said and "rectangle 0.05 x 0.015 m" in said
    assert drive.subleaf_problem(session, {}, resolution) == said          # binding: refused again (round 12)
    # ended there, the verdict is the runner's: the governing member is unresolved
    (session / "operator" / "j.md").write_text("The brief's case could not be run.\n## Deviations\nnone\n")
    _, info = drive.verdict_gate((session / "operator" / "j.md").read_text(), session)
    assert info["verdict_line"] == "The brief's case could not be run: the governing member is unresolved."
    # a body that resolves it clears the blocker
    drive.subleaf_problem(session, {}, {"physical_extents": [0.05, 0.05, 0.001]})
    assert not any(b["kind"] == drive.SUBLEAF_BLOCKER for b in drive.blockers(session))


LIFTED = ('## Use\nused\nUse table: {"actions": {"A1": {"load": "pull", "state": "working", "kind": "weight", '
          '"source": "50 kg x g x 1", "direction": [0, 1, 0], "enters": "body", "held": ["feet"]}}, "run": "working"}\n'
          '## Contacts\n{"feet": {"transmits": "force only", "kind": "rest"}}\n## Load\nup\n'
          '## Size and material\nReal mass: 10 kg\nReal section (body): rectangle 0.1 x 0.05 m\n'
          'Real extents: 1 x 0.5 x 0.5 m\nLoad path: {"body": "series"}\n')


def test_a_rest_lifted_off_is_refused_once_before_any_body(tmp_path, monkeypatch):
    session = _session(tmp_path, "")
    (session / "operator" / "brief.md").write_text(LIFTED)
    monkeypatch.setattr(drive, "body_source", lambda s: None)
    said = drive.early_case_problem(session, {})
    assert said and "off its rests ['feet']" in said and "equal-and-opposite pair" in said
    assert "Before any body the brief is editable" in said
    assert drive.early_case_problem(session, {}) is None                    # said once
    # the weight holds it down: nothing to say
    (session / "operator" / "brief.md").write_text(LIFTED.replace("Real mass: 10 kg", "Real mass: 80 kg"))
    assert drive.rest_lift_off(session) is None


def test_a_plausibility_refusal_after_the_body_unlocks_the_real_section_once(tmp_path, monkeypatch):
    session = _session(tmp_path, "Real section (body): rectangle 0.1 x 0.05 m\nReal mass: 10 kg\n")
    operator = session / "operator"
    snapshots = session / "brief_snapshots"
    snapshots.mkdir()
    (snapshots / "brief.first.md").write_text((operator / "brief.md").read_text())
    monkeypatch.setattr(drive, "body_source", lambda s: Path("body.glb"))
    monkeypatch.setattr(drive, "early_case_findings", lambda s, p: {"deflection": "0.5 of the largest extent"})
    said = drive.early_case_problem(session, {})
    assert said and "the Real section line may change once" in said
    assert drive.early_case_problem(session, {}) is None
    (operator / "brief.md").write_text("## Use\nused\n## Size and material\nReal section (body): rectangle 0.1 x "
                                       "0.08 m\nReal mass: 10 kg\n")
    assert drive.real_lines_frozen_problem(operator) is None                 # the one unlocked line, taken
    assert drive.real_section(operator)["depth_m"] == 0.08
    assert "0.08 m after the body, unlocked once" in drive.brief_status(operator)["real_line_unlocked"]
    (operator / "brief.md").write_text("## Use\nused\n## Size and material\nReal section (body): rectangle 0.1 x "
                                       "0.09 m\nReal mass: 10 kg\n")
    assert "frozen" in drive.real_lines_frozen_problem(operator)            # once only


def test_a_body_keeps_the_largest_extent_and_a_wall_s_smallest(tmp_path, monkeypatch):
    session = _session(tmp_path, "Real mass: 20 kg\nReal thinnest member: 0.08 m\nReal extents: 1.0 x 0.5 x 0.1 m\n")
    monkeypatch.setattr(drive, "body_source", lambda s: tmp_path / "body.glb")
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    for route in ("generated", "section", "real"):
        monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": route, "reasons": []})
        said = drive.body_gate(session, {}, {"principal_extents": [1.0, 0.5, 0.35]}, record=False)
        assert said and "the smallest is off" in said, route
        assert drive.body_gate(session, {}, {"principal_extents": [1.0, 0.5, 0.12]}, record=False) is None
        assert drive.body_gate(session, {}, {"principal_extents": [1.0, 0.9, 0.12]}, record=False) is None
        assert "largest" in drive.body_gate(session, {}, {"principal_extents": [1.8, 0.5, 0.12]}, record=False)
    # a smallest extent no wall or section sets is a pose: not judged
    (session / "operator" / "brief.md").write_text("## Size and material\nReal mass: 20 kg\nReal thinnest member: "
                                                   "0.01 m\nReal extents: 1.0 x 0.5 x 0.1 m\n")
    assert drive.body_gate(session, {}, {"principal_extents": [1.0, 0.5, 0.35]}, record=False) is None
