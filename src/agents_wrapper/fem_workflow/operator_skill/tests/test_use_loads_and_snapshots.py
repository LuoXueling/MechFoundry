"""The use's loads and the brief's snapshots: the case run covering the other cases member by
member, a resting clamp refused against tipping without its grip, the brief's snapshot taken at
the first body built, and the member warning said once.  No GPU, no API, no graph run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import trimesh

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402
from test_frames_and_runner_writes import BRIEF, _round  # noqa: E402

COMPARED = "A press and a side pull at the arm."


def _brief(session: Path, use: str, size: str = "") -> Path:
    (session / "operator").mkdir(parents=True, exist_ok=True)
    path = session / "operator" / "brief.md"
    path.write_text(f"## Use\n{use}\n## Contacts\nc\n## Size and material\nReal mass: 1 kg\n{size}")
    return path


# ------------------------------------------------------------- the case run's coverage --

def test_a_load_acting_with_the_governing_one_is_dropped_only_under_a_quarter_of_it(tmp_path):
    # no load is dropped in words: a state is run, and it must reach 75 % of every other state at every
    # Load path member, or the table declares that member uncovered with the workflow limit
    from test_use_table import gate, session_with, table
    # the runner's effects on the 0.3 m Real extent: the working state is 75 N m at every member
    within = table(rows={"A3": {"enters": "top plate", "size": 330, "source": "33 kg x g x 1"}})   # 99 <= 75 / 0.75
    assert gate(session_with(tmp_path, within)) == []
    over = table(rows={"A3": {"enters": "top plate", "size": 340, "source": "34 kg x g x 1"}})     # 102 > 100
    said = gate(session_with(tmp_path, over))
    assert any("102 N m at 'top plate' against 75" in s for s in said)
    # the same at another member: refused alike, unless declared
    later = {"A3": {"size": 340, "source": "34 kg x g x 1"}}
    said = gate(session_with(tmp_path, table(rows=later)))
    assert any("102 N m at 'side grip' against 75" in s and "'top plate'" not in s for s in said)
    limit = {"limit": "dynamic or impact", "why": "the carry is a jolt at the grip",
             "no_approximation": "the jolt has no stated factor"}
    session = session_with(tmp_path, table(rows=later, uncovered={"side grip": limit, "bottom plate": limit}))
    assert gate(session) == []
    assert drive.use_coverage(session / "operator")["uncovered"]["side grip"]["effect"] == 102


# ------------------------------------------------------------ the load path --

def test_a_member_named_after_the_body_joins_the_load_path_and_none_leaves_it(tmp_path):
    session = tmp_path / "abc"
    _brief(session, COMPARED, size='Load path: {"arm": "series", "post": "series"}\n')
    drive.brief_status(session / "operator")                                  # the body's snapshot
    _brief(session, COMPARED, size='Load path: {"arm": "series", "brace": "parallel"}\n')
    assert drive.brief_members(session / "operator") == {"arm": "series", "post": "series", "brace": "parallel"}
    body = {"left_out": "l"}
    assert drive.path_gate(session, {"body_simplification": body}) is None        # every member kept
    assert "brace: absent gives \"minor\"" in drive.path_gate(session, {"body_simplification": {
        "absent": {"brace": "some"}}})


def test_the_geometry_stop_asks_for_each_part_between_the_load_and_the_hold():
    said = drive.STOPS["geometry"]
    assert "between where the use's load enters and where the object is held" in said and "Load path" in said


# ------------------------------------------------------------------ tipping --

REST = '## Contacts\n- base: rests on its shelf\n{"base": {"transmits": "force only", "kind": "rest"}}\n'


def test_a_resting_clamp_warned_against_tipping_is_refused_without_its_grip_or_a_fastening(tmp_path, monkeypatch):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "operator" / "brief.md").write_text(REST)
    tips = {"held": ["base_ring"], "contacts": ["base"], "overturning": 5.6, "resisting": 1.0, "half_base": 0.03}
    monkeypatch.setattr(drive, "tipping_excess", lambda *a: dict(tips))
    resting = {"contacts": {"base": {"transmits": "force only", "regions": ["base_ring"], "simplified_to": "clamped"}}}
    said = drive.tipping_warning(session, resting, [], [])
    assert "would tip" in said and "a contact of kind grip or socket, or a fastening" in said and "refuses it" in said
    refused = drive.hold_gate(session, resting, [], [], [])
    assert "5.6 N m" in refused and "where the use steadies it" in refused
    # a contact of another kind is not this refusal's (a grip, a socket or a fastening takes the moment)
    (session / "operator" / "brief.md").write_text(REST.replace('"kind": "rest"', '"kind": "grip", "over_m": 0.1'))
    assert drive.hold_gate(session, resting, [], [], []) is None
    # the grip held, or the side load capped: tipping_excess finds nothing, and nothing is refused
    (session / "operator" / "brief.md").write_text(REST)
    monkeypatch.setattr(drive, "tipping_excess", lambda *a: None)
    assert drive.hold_gate(session, resting, [], [], []) is None


def test_a_held_grip_beside_the_resting_face_clears_the_tipping_refusal(tmp_path, monkeypatch):
    from test_case_realism import BASE, BASE_BOUNDS, TOP, _standing_session
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    (session / "operator" / "brief.md").write_text(REST.replace(
        "}}", '}, "hand": {"transmits": "force and moment", "kind": "grip", "over_m": 0.05}}'))
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    plan = {"values": {"acceleration": [0.0, -9.80665, 0.0]},
            "contacts": {"base": {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}}}
    push = [{"region": "top", "force_N": [20.0, 0, 0]}]
    assert drive.tipping_warning(session, plan, [BASE, TOP], push)
    assert "3.8 N m" in drive.hold_gate(session, plan, [], [BASE, TOP], push)
    grip = {"name": "grip", "category": "dirichlet", "mean_normal": [1.0, 0.0, 0.0],
            "centroid": [0.025, 0.0, 0.0], "area": 0.002}
    gripped = {**plan, "contacts": {**plan["contacts"], "hand": {"transmits": "force and moment", "regions": ["grip"]}}}
    assert drive.hold_gate(session, gripped, [], [BASE, grip, TOP], push) is None


def test_a_body_hanging_from_its_hold_cannot_tip(tmp_path, monkeypatch):
    from test_case_realism import _standing_session
    # the hold is the inside of a top bend: its face looks down, into the opening, and the body hangs below
    session = _standing_session(tmp_path, {"hook": [[-0.01, 0.1, -0.01], [0.01, 0.1, 0.01]]})
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.05})
    hook = {"name": "hook", "category": "dirichlet", "mean_normal": [0.0, -1.0, 0.0],
            "centroid": [0.0, 0.1, 0.0], "area": 0.0002}
    pull = {"name": "pull", "category": "neumann", "centroid": [0.02, -0.1, 0.0]}
    plan = {"values": {"acceleration": [0.0, -9.80665, 0.0]},
            "contacts": {"anchor": {"transmits": "force only", "regions": ["hook"], "simplified_to": "clamped"}}}
    loads = [{"region": "pull", "force_N": [5.0, -500.0, 0.0]}]
    assert drive.tipping_excess(session, plan, [hook, pull], loads) is None    # no mesh: the loads hang below
    (session / "geometry").mkdir()
    trimesh.creation.box(extents=[0.04, 0.2, 0.01]).export(session / "geometry" / "physical.glb")  # centred at 0
    assert drive.tipping_excess(session, plan, [hook, pull], loads) is None
    assert drive.hold_gate(session, plan, [], [hook, pull], loads) is None
    # the same face at the body's foot, the body above it: a resting support, which can tip
    foot = {**hook, "centroid": [0.0, -0.1, 0.0]}
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": [
        {"name": "hook", "category": "dirichlet", "bounds": [[-0.01, -0.1, -0.01], [0.01, -0.1, 0.01]]}]})
    side = [{"region": "pull", "force_N": [5.0, 0.0, 0.0]}]
    assert drive.tipping_excess(session, plan, [foot, {**pull, "centroid": [0.02, 0.1, 0.0]}], side)
    # a hold of kind hang never tips, wherever it is
    (session / "operator" / "brief.md").write_text('## Contacts\n{"anchor": {"transmits": "force only", "kind": "hang"}}\n')
    assert drive.tipping_excess(session, plan, [foot, {**pull, "centroid": [0.02, 0.1, 0.0]}], side) is None


def test_the_tipping_refusal_offers_the_capped_push_and_the_co_acting_weight(tmp_path, monkeypatch):
    from test_case_realism import BASE, BASE_BOUNDS, TOP, _standing_session
    session = _standing_session(tmp_path, {"base": BASE_BOUNDS})
    (session / "operator" / "brief.md").write_text(REST)
    monkeypatch.setattr(drive, "planned_resolution", lambda session, plan: {"planned_mass": 0.2})
    contact = {"transmits": "force only", "regions": ["base"], "simplified_to": "clamped"}
    plan = {"values": {"acceleration": [0.0, -9.80665, 0.0]}, "contacts": {"base": contact}}
    push = [{"region": "top", "force_N": [20.0, 0, 0]}]
    said = drive.hold_gate(session, plan, [], [BASE, TOP], push)
    assert "cap the side loads (20 N) at 0.232 N, under the push that tips it with a 10% margin" in said
    assert "presses down at the same time" in said and "inertia" not in said
    # capped at the push that tips it: nothing left to refuse
    assert drive.hold_gate(session, plan, [], [BASE, TOP], [{"region": "top", "force_N": [0.25, 0, 0]}]) is None


# --------------------------------------------------------------- the snapshots --

def test_a_refused_build_takes_no_snapshot_and_the_body_takes_the_brief_as_edited(tmp_path, monkeypatch):
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    brief = BRIEF.replace("MEMBER", "0.05").replace("two supports on the ground, held fully",
                                                    '- pads: held fully\n{"pads": {"transmits": "force and moment", '
                                                    '"kind": "fastening"}}').replace('"held": ["base"]', '"held": ["pads"]')
    (session / "operator" / "brief.md").write_text(brief)
    bars = tmp_path / "frame.json"
    bars.write_text(json.dumps([{"from": [0, 0, 0], "to": [1, 0, 0], "section": _round(0.1)}]))
    args = argparse.Namespace(output_root=str(tmp_path), session_id="abc", shape=None, size=None, radius=None,
                              length=None, axis=None, profile=None, hole=None, circle=None, bars=str(bars))
    monkeypatch.setattr(drive, "build_frame", lambda spec: (_ for _ in ()).throw(drive.PlanError("apart")))
    drive.cmd_blank(args)
    snapshots = session / "brief_snapshots"
    assert printed[-1]["ok"] is False and not (snapshots / "brief.first.md").exists()
    assert (snapshots / "brief_initial.md").exists()                         # the first run's brief, as ever
    # the brief corrected before any body: its contacts and mass are the body's
    (session / "operator" / "brief.md").write_text(brief.replace("pads", "rails").replace("30 kg", "40 kg"))
    monkeypatch.setattr(drive, "build_frame", lambda spec: (
        trimesh.creation.box(extents=[1.0, 0.1, 0.1]),
        {"bars": 1, "boxes": 0, "joints": 0, "smallest_section_m": 0.1}))
    drive.cmd_blank(args)
    assert printed[-1]["ok"] and "rails" in (snapshots / "brief.first.md").read_text()
    operator = session / "operator"
    assert drive.brief_contacts(operator) == {"rails": "force and moment"}
    assert drive.real_mass(operator) == 40.0
    [edit] = drive.brief_status(operator)["real_lines_edited_before_body"]
    assert "Real mass: 30 -> 40 before the body (allowed, stated)" == edit
    # after the body the snapshot holds: a later edit is an edit after the body
    (operator / "brief.md").write_text(brief.replace("pads", "rails").replace("30 kg", "50 kg"))
    assert drive.real_mass(operator) == 40.0 and drive.brief_status(operator)["real_lines_edited"] == ["Real mass"]
    # a revision record that fails to read says nothing, never a traceback
    monkeypatch.setattr(drive, "extents_revised_said", lambda s: 1 / 0)
    assert drive.brief_status(operator)["extents_revised"] is None


def test_blank_runs_the_one_photograph_search_itself_with_the_compute_env(tmp_path, monkeypatch):
    # a held-out run (2026-10-12) died here on a NameError: cmd_blank dropped the compute env it
    # needs for the search it runs when no photograph was tried yet
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.05"))
    bars = tmp_path / "frame.json"
    bars.write_text(json.dumps([{"from": [0, 0, 0], "to": [1, 0, 0], "section": _round(0.1)}]))
    args = argparse.Namespace(output_root=str(tmp_path), session_id="abc", shape=None, size=None, radius=None,
                              length=None, axis=None, profile=None, hole=None, circle=None, bars=str(bars))
    seen = []
    monkeypatch.setattr(drive, "web_photo_problem", lambda s: "no photograph tried")
    monkeypatch.setattr(drive, "blank_web_search",
                        lambda a, s, root, env: seen.append(env) or {"ok": True, "web_image": "shown"})
    drive.cmd_blank(args)
    assert seen and printed[-1] == {"ok": True, "web_image": "shown"}


def test_the_member_warning_is_said_once_for_the_same_real_lines(tmp_path):
    operator = tmp_path / "operator"
    operator.mkdir()
    lines = "## Size and material\nReal thinnest member: 0.008 m\nReal section: 0.02 m deep, I 6.0e-8 m4, A 1.8e-3 m2\n"
    (operator / "brief.md").write_text(lines)
    assert drive.brief_member_warning(operator)
    assert drive.member_warning_new(operator) is True
    assert drive.member_warning_new(operator) is False                        # the next stop: not again
    (operator / "brief.md").write_text(lines.replace("0.008", "0.006"))
    assert drive.member_warning_new(operator) is True                         # other lines: said for them
