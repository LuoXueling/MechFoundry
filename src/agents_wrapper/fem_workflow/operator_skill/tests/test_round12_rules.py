"""The rules of round 12: the equivalent-stiffness section by its EI, the lever check of a built
stand-in, and the audit's binding refusals, deviations, zero fields and twist rows."""

from __future__ import annotations

import math
from pathlib import Path

import drive
import sessionlib as lib


def _session(tmp_path: Path, size: str) -> Path:
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text("## Use\nused\n## Size and material\n" + size)
    return session


BLADE = ("Real thinnest member: 0.001 m\nReal section (blade): rectangle 0.05 x 0.001 m\n"
         "Governing material: steel, E 2e11 Pa\nReal extents: 1 x 0.05 x 0.05 m\nReal mass: 0.4 kg\n")


def test_any_built_section_of_the_printed_ei_is_the_equivalent_section(tmp_path, monkeypatch):
    session = _session(tmp_path, BLADE)
    plan = {"material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850.0}}
    eq = drive.equivalent_section(session, 1.0, plan)
    i_printed = 0.05 * 0.015 ** 3 / 12
    d = (64 * i_printed / math.pi) ** 0.25           # a solid round of the printed section's I
    body = session / "operator" / "blank_bars_x.glb"
    taken = drive.record_equivalent(session, body, d, 1.0, plan, [("round", [d])])
    assert taken["built_at_it"] and taken["section"].startswith("round") and taken["E_factor"] == eq["E_factor"]
    # a round 30 % stiffer is no equivalent
    other = session / "operator" / "blank_bars_y.glb"
    assert not drive.record_equivalent(session, other, 1.15 * d, 1.0, plan, [("round", [1.15 * d])])["built_at_it"]
    # the material check takes E x the factor for the plan's built body before it is uploaded
    scaled = {"geometry": {"asset_path": str(body)},
              "material": {**plan["material"], "youngs_modulus": eq["plan_material"]["youngs_modulus"]}}
    assert drive.governing_material_check(session, scaled) == (None, None)
    assert drive.governing_material_check(session, {**scaled, "material": plan["material"]})[0]
    # once uploaded, G4 and the thin-member checks compare the equivalent section, not the Real member
    lib.write_json(session / "session.json", {"input_asset": {"source_name": body.name}})
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    assert drive.compared_member(session) == taken["thinnest_m"]
    assert drive.thin_member_limit(session, {}, {"physical_extents": [1.0, 0.05, 0.05]})["real_member_m"] == \
        taken["thinnest_m"]


def test_an_edit_after_the_early_check_is_rechecked_and_listed(tmp_path, monkeypatch):
    session = _session(tmp_path, "Real section (arm): rectangle 0.1 x 0.05 m\nGoverning material: pine, E 1e10 Pa\n")
    monkeypatch.setattr(drive, "body_source", lambda s: None)
    found = {"deflection": "0.5 of the largest extent"}
    monkeypatch.setattr(drive, "early_case_findings", lambda s, p: dict(found))
    assert drive.early_case_problem(session, {})
    assert drive.early_case_problem(session, {}) is None                     # kept: said once
    brief = session / "operator" / "brief.md"
    brief.write_text(brief.read_text().replace("pine, E 1e10", "steel, E 2e11"))
    said = drive.early_case_problem(session, {})
    assert said and "re-checked after the brief changed" in said and "Governing material: steel" in said
    assert drive.early_case_problem(session, {}) is None
    # an edit that clears the finding passes, and the verdict lists it
    brief.write_text(brief.read_text().replace("0.1 x 0.05", "0.1 x 0.2"))
    found.clear()
    assert drive.early_case_problem(session, {}) is None
    listed = [s for s in drive.runner_simplifications(session, {}, []) if s.startswith("brief changed after the check")]
    assert len(listed) == 2 and any("0.1 x 0.2" in s for s in listed)


TABLE = ('## Use\nused\nUse table: {"actions": {"A1": {"load": "pull", "state": "working", "kind": "weight", '
         '"source": "50 kg x g x 1", "direction": [0, -1, 0], "enters": "arm", "held": ["base"]}}, "run": "working"}\n'
         '## Contacts\n{"base": {"transmits": "force and moment", "kind": "fastening"}}\n## Load\ndown\n'
         '## Size and material\nReal mass: 10 kg\nReal section (arm): rectangle 0.05 x 0.05 m\n'
         'Real extents: 1 x 0.2 x 0.2 m\nLoad path: {"arm": "series", "base": "series"}\n')


def test_a_dropped_load_path_member_or_a_moved_hold_is_a_deviation(tmp_path):
    session = _session(tmp_path, "")
    (session / "operator" / "brief.md").write_text(TABLE)
    none, named = "Verdict.\n## Deviations\nnone\n", "Verdict.\n## Deviations\nthe base is left out: stiffer\n"
    dropped = {"body_simplification": {"absent": ["base"]}}
    said = drive.dropped_member_problem(session, none, dropped)
    assert said and "member base declared absent" in said
    assert drive.dropped_member_problem(session, named, dropped) is None
    assert drive.dropped_member_problem(session, none, {"body_simplification": {"absent": ["lid"]}}) is None
    moved = {"contacts": {"base": {"regions": ["r"], "absent_in_stand_in": {"carried_by": "held_at_joint"}}}}
    assert "contact base held off its member's surface" in drive.dropped_member_problem(session, none, moved)


def test_a_zero_field_under_a_load_is_a_failed_solve(tmp_path, monkeypatch):
    loaded = {"regions": [{"category": "neumann", "applied_force": [0.0, -10.0, 0.0]}]}
    zero = {**loaded, "solve": {"run": "r1", "checks": {"max_displacement": 0.0, "max_von_mises_stress": 0.0}}}
    said = drive.zero_field(zero)
    assert said and said.startswith(drive.ZERO_FIELD_SAID) and "under 10 N" in said
    assert drive.zero_field({**zero, "regions": []}) is None                      # no load: nothing to say
    assert drive.zero_field({**loaded, "solve": {"checks": {"max_displacement": 1e-6}}}) is None
    session = _session(tmp_path, "")
    lib.write_json(session / "operator" / "prediction_history.json", [{"run": "r1", "solved": {}}])
    lib.write_json(session / "operator" / drive.ZERO_FIELD, {"run": "r1", "said": said})
    monkeypatch.setattr(drive, "runner_simplifications", lambda *a, **k: [])
    monkeypatch.setattr(drive, "held_regions", lambda *a, **k: [])
    _, info = drive.verdict_gate("The brief's case was run.\n## Deviations\nnone\n", session)
    assert info["verdict_line"] == f"The brief's case could not be run: {said}."
    # a later solve of its own is judged as any other
    lib.write_json(session / "operator" / "prediction_history.json", [{"run": "r1"}, {"run": "r2"}])
    assert drive.zero_field_said(session) is None


def test_a_hand_twist_far_over_a_fingertip_at_its_member_is_warned(tmp_path):
    session = _session(tmp_path, "")
    twist = TABLE.replace('"kind": "weight", "source": "50 kg x g x 1", "direction": [0, -1, 0]',
                          '"kind": "twist", "source": "H6 top", "direction": [0, 1, 0]')
    table_said = []
    for section, warned in (("round 0.004 m", True), ("round 0.03 m", True), ("round 0.1 m", False)):
        (session / "operator" / "brief.md").write_text(twist.replace("rectangle 0.05 x 0.05 m", section))
        table, _ = drive.use_table(session / "operator")
        table_said.append(table["sizes"]["A1"])
        said = drive.load_scale_warning(session)
        assert bool(said and "check the twist against the hand" in said) is warned, (section, said, table_said)


def test_a_built_body_whose_load_sits_by_its_hold_is_marked(tmp_path, monkeypatch):
    session = _session(tmp_path, "")
    (session / "operator" / "brief.md").write_text(TABLE)
    table, _ = drive.use_table(session / "operator")
    brief = drive.row_lever(table, "A1")
    assert brief > 0.5
    plan = {"contacts": {"base": {"transmits": "force and moment", "regions": ["foot"]}},
            "loads": {"A1": {"regions": ["tip"]}}}
    (session / "boundary_conditions").mkdir()

    def regions(at):
        lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": [
            {"name": "foot", "category": "dirichlet", "centroid": [0.0, 0.0, 0.0]},
            {"name": "tip", "category": "neumann", "centroid": [at, 0.0, 0.0], "area": 0.01}]})

    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    regions(0.1 * brief)
    said = drive.lever_shortfall(session, plan)
    assert said and said.startswith(drive.LEVER_SAID)
    assert any(s.startswith(drive.LEVER_SAID) for s in drive.runner_simplifications(session, plan, []))
    regions(0.9 * brief)
    assert drive.lever_shortfall(session, plan) is None
    # a generated body is not judged so (its outline is the object's)
    regions(0.1 * brief)
    monkeypatch.setattr(drive, "body_provenance", lambda s: "triposg")
    assert drive.lever_shortfall(session, plan) is None
