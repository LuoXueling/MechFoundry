"""Built frames of bars (``blank --bars``) and what the runner writes into the plan itself.

A held-out batch (2026-10-10): oblique joints of round bars were refused as "not one watertight
body" without naming a joint, frame builds took 1.5-2.5 min, stand-ins sized for depth 8 were
checked at depth 7, the self-weight scaling the runner printed was retyped by every agent, and a
region preview whose every clause began "Fix" was refused for having no support.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import drive  # noqa: E402
from conftest import tabled  # noqa: E402
import sessionlib as lib  # noqa: E402

BRIEF = ("## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", "
         "\"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
         "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\n"
         "Real thinnest member: MEMBER m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n"
         "0.9 m tall, wood 1.2e10\n## Prediction\nmembers in compression, 0.1 MPa, microns")
SECTION = {"kept": "the load path", "left_out": "the trim"}
G = 9.80665
BRIEF = tabled(BRIEF)


def _round(d):
    return {"d": d}


def _coil(turns_of_eight=2, radius=0.012, drop=0.0015, d=0.005):
    """A polyline coil: 8 straight round bars a turn, every joint oblique and out of plane."""
    n = 8 * turns_of_eight
    pts = [[radius * math.cos(k * math.pi / 4), 0.064 - drop * k, radius * math.sin(k * math.pi / 4)]
           for k in range(n + 1)]
    return [{"from": pts[k], "to": pts[k + 1], "section": _round(d)} for k in range(n)]


# ------------------------------------------------------------------ oblique round joints --

def test_oblique_joints_of_round_bars_union_into_one_body(monkeypatch):
    # The union is read back unmerged: two round bars meeting at 45 degrees in one plane leave
    # vertices a few nanometres apart where their sides touch, and merging them pinched the body
    # (under the agents environment the union runs in the compute one, and came back that way).
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    in_plane = [{"from": [0, 0.064, 0], "to": [0.012, 0.064, 0], "section": _round(0.005)},
                {"from": [0.012, 0.064, 0], "to": [0.024, 0.052, 0], "section": _round(0.005)}]
    for spec in (in_plane, _coil()):
        mesh, built = drive.build_frame(spec)
        assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1
        assert built["joints"] == len(spec) - 1 and "squared_at_joints" not in built


def test_a_bar_s_prism_is_cut_into_rings_so_the_remesh_splits_no_sliver():
    bar = {"kind": "round", "size": [0.036]}
    prism = drive._bar_prism(bar, 0.9, 0.025)
    assert prism.is_watertight and prism.is_volume
    # the 32-gon's area, the length kept
    area = 0.5 * drive.BAR_SECTIONS * (0.018 ** 2) * math.sin(2 * math.pi / drive.BAR_SECTIONS)
    assert prism.volume == pytest.approx(area * 0.9, rel=1e-6)
    assert np.allclose(prism.bounds[:, 2], [-0.45, 0.45])
    # no side edge longer than a ring cell's diagonal (the slivers midpoint subdivision multiplied
    # by 4^5 ran the bar's whole length)
    chord = 2 * 0.018 * math.sin(math.pi / drive.BAR_SECTIONS)
    assert prism.edges_unique_length.max() <= math.hypot(0.025, chord) + 1e-9
    square = drive._bar_prism({"kind": "rect", "size": [0.05, 0.03]}, 1.0, 0.1)
    assert square.is_volume and square.volume == pytest.approx(0.05 * 0.03 * 1.0)
    assert np.allclose(square.extents, [0.05, 0.03, 1.0])


def _tagged_prisms(monkeypatch):
    """Each bar's prism carries whether it is round, for the failing union below to read."""
    real = drive._bar_prism

    def prism(bar, length, spacing):
        mesh = real(bar, length, spacing)
        mesh.metadata["round"] = bar["kind"] == "round"
        return mesh
    monkeypatch.setattr(drive, "_bar_prism", prism)


def _failing_union(fail_round_only: bool):
    """A union that fails for any group holding a round bar (or for every group): the whole
    frame comes back as two pieces, a joint's group as a body with a hole."""
    import trimesh
    real = drive._union_groups

    def union(meshes, groups):
        out = real(meshes, groups)
        broken = []
        for group, mesh in zip(groups, out):
            rounds = any(meshes[k].metadata.get("round") for k in group)
            if (rounds or not fail_round_only) and len(group) > 1:
                torn = mesh.copy()
                torn.update_faces(np.arange(len(torn.faces)) != 0)
                piece = trimesh.creation.box(extents=[0.001] * 3)
                piece.apply_translation(mesh.bounds[1] + 1.0)
                broken.append(trimesh.util.concatenate([torn, piece]))
            else:
                broken.append(mesh)
        return broken
    return union


def test_a_round_joint_that_does_not_union_is_built_square_and_said(monkeypatch):
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    _tagged_prisms(monkeypatch)
    monkeypatch.setattr(drive, "_union_groups", _failing_union(fail_round_only=True))
    spec = [{"from": [0, 0, 0], "to": [0.5, 0, 0], "section": _round(0.04)},
            {"from": [0.5, 0, 0], "to": [0.9, 0.4, 0], "section": _round(0.04)}]
    mesh, built = drive.build_frame(spec)
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1
    assert "bar 0, bar 1" in built["squared_at_joints"] and "square, side d" in built["squared_at_joints"]
    assert built["smallest_section_m"] == 0.04


def test_a_joint_that_does_not_union_is_named_with_what_to_change(monkeypatch):
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    monkeypatch.setattr(drive, "_union_groups", _failing_union(fail_round_only=False))
    spec = [{"from": [0, 0, 0], "to": [0.5, 0, 0], "section": [0.04, 0.04]},
            {"from": [0.5, 0, 0], "to": [0.9, 0.4, 0], "section": [0.04, 0.04]}]
    with pytest.raises(drive.PlanError) as caught:
        drive.build_frame(spec)
    said = str(caught.value)
    assert "the joint at [0.5, 0.0, 0.0] of bar 0, bar 1 (axes 45 degrees apart)" in said
    assert "square sections" in said and "a box over it" in said
    # a frame whose joints all union alone: the pieces are said, with the cause to look for
    message = drive._frame_fault_message([], _failing_union(False)(
        [drive._bar_prism({"kind": "rect", "size": [0.1, 0.1]}, 1.0, 1.0)] * 2, [[0, 1]])[0], [])
    assert "away from a joint" in message


def test_blank_bars_builds_bars_under_two_leaves_and_warns_with_their_numbers(tmp_path, monkeypatch):
    import trimesh
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.002"))
    # a post with four arms, 1.9 m tall: 2 leaves at depth 8 is 2 x 1.92 x 1.97 / 256 = 0.0296 m
    spec = [{"from": [0, 0, 0], "to": [0, 1.9, 0], "section": _round(0.07)}] + [
        {"from": [0, 1.9, 0], "to": to, "section": _round(0.02)}
        for to in ([0.9, 1.9, 0], [-0.9, 1.9, 0], [0, 1.9, 0.9], [0, 1.9, -0.9])]
    needed = drive.frame_sections_needed(spec, {}, session)
    assert needed["depth"] == 7 and needed["thin_bars"] == [1, 2, 3, 4]       # no depth to write: 7
    assert needed["section_needed_m"] == pytest.approx(2 * 1.92 * (1.9 + 0.07) / 128, rel=0.01)
    # at a depth the plan names, that depth's leaf
    deep = drive.frame_sections_needed(spec, {"settings": {"remesh_octree_depth": 8}})
    assert deep["depth"] == 8 and deep["thin_bars"] == [1, 2, 3, 4]
    bars = tmp_path / "frame.json"
    bars.write_text(json.dumps(spec))
    monkeypatch.setattr(drive, "build_frame",
                        lambda spec: (trimesh.creation.box(extents=[1.8, 1.97, 1.8]), {"smallest_section_m": 0.02}))
    args = dict(output_root=str(tmp_path), session_id="abc", shape=None, size=None, radius=None, length=None,
                axis=None, profile=None, hole=None, circle=None, bars=str(bars), revolve=None, wall=None)
    drive.cmd_blank(argparse.Namespace(**args))
    said = printed[-1]
    # built, never refused: the thin bars are a warning with their numbers
    assert said["ok"] is True and "thin member: bar 1, bar 2, bar 3, bar 4 (0.02 m thinnest)" in said["warning"]
    assert f"under 2 leaves ({needed['section_needed_m']:g} m) at depth 7" in said["warning"]


def test_the_frame_timing_is_spent_in_the_remesh_not_the_joint_search(monkeypatch):
    # 30 bars: the pair search and the joint search stay well under a second
    import time
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    spec = _coil(turns_of_eight=4)[:30]
    start = time.time()
    bars, boxes = drive.frame_members(spec)
    assert len(drive._frame_groups(bars, boxes)) == 1
    assert time.time() - start < 2.0


@pytest.mark.slow
@pytest.mark.real_remesh
def test_a_frame_of_round_bars_builds_in_seconds_with_the_real_remesh():
    import time
    from inspect_session import compute_interpreter
    if compute_interpreter(lib.DEFAULT_COMPUTE_ENV) is None:
        pytest.skip("no compute environment to remesh in")
    spec = [{"from": [0, 0, 0], "to": [0, 1.9, 0], "section": _round(0.07)}] + [
        {"from": [0, 1.9, 0], "to": to, "section": _round(0.065)}
        for to in ([0.9, 1.9, 0], [-0.9, 1.9, 0], [0, 1.9, 0.9], [0, 1.9, -0.9])]
    start = time.time()
    mesh, built = drive.build_frame(spec)
    # 150 s in a held-out run (2026-10-10), 203k faces into MeshLab; about 7 s now
    assert mesh.is_watertight and time.time() - start < 40
    assert len(mesh.faces) < 80_000


# ------------------------------------------------------------- the depth a stand-in needs --

def test_depth_8_is_written_only_where_it_brings_the_real_member_to_two_leaves(tmp_path):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.02"))
    path = tmp_path / "operator" / "plan.json"
    plan = {"scale": {"physical_extent": 1.0}, "settings": {}, "body_simplification": SECTION}
    path.write_text(json.dumps(plan, indent=2) + "\n")
    # 0.02 m at 1.0 m: 1.3 leaves at depth 7, 2.7 at depth 8: written, with its cause
    out, said = drive.prescribed_depth(tmp_path, path, plan)
    assert out["settings"]["remesh_octree_depth"] == drive.DEEPEST_DEPTH and "the runner set" in said
    assert json.loads(path.read_text())["settings"]["remesh_octree_depth"] == drive.DEEPEST_DEPTH
    # a member no depth brings to 2 leaves (0.005 m: 0.7 at depth 8): no depth, it is reported instead
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.005"))
    plain = {"scale": {"physical_extent": 1.0}, "settings": {}}
    path.write_text(json.dumps(plain, indent=2) + "\n")
    assert drive.prescribed_depth(tmp_path, path, plain) == (plain, None)
    # the member resolves at depth 7 (0.05 m at 1.0 m: 3.3 leaves): nothing written
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.05"))
    assert drive.prescribed_depth(tmp_path, path, plain) == (plain, None)
    # a depth the plan names is the agent's
    named = {**plan, "settings": {"remesh_octree_depth": 7}}
    assert drive.prescribed_depth(tmp_path, path, named) == (named, None)


# ------------------------------------------------------- the self-weight the runner scales --

def _heavy_stand_in(tmp_path, monkeypatch, planned=600.0):
    (tmp_path / "operator").mkdir(exist_ok=True)
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.002"))   # Real mass 30 kg
    mass = {"planned_mass": planned}
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: dict(mass))
    # a stand-in built thicker than the real member (the route is the body's: body_route)
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["built thicker"]})
    path = tmp_path / "operator" / "plan.json"
    plan = {"scale": {"physical_extent": 1.0}, "body_simplification": SECTION,
            "values": {"load": {"force": [0, -20, 0]}, "acceleration": [0, -G, 0]}}
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path, plan, mass


def test_the_runner_writes_the_scaled_self_weight_of_a_heavy_stand_in(tmp_path, monkeypatch):
    path, plan, mass = _heavy_stand_in(tmp_path, monkeypatch)
    want = [0.0, -round(G * 30 / 600, 5), 0.0]
    out, said = drive.scaled_self_weight(tmp_path, path, plan)
    assert out["values"]["acceleration"] == want and out["values"]["acceleration_use"] == [0.0, -G, 0.0]
    assert "self-weight scaled by the runner" in said and "20 x the brief's Real mass 30 kg" in said
    written = json.loads(path.read_text())["values"]
    assert written["acceleration"] == want and written["load"] == {"force": [0, -20, 0]}
    assert drive.self_weight_problem(tmp_path, out) is None
    # said once: the scaled plan is left as it is
    assert drive.scaled_self_weight(tmp_path, path, out) == (out, None)
    # the report names the runner's write
    assert "written by the runner" in drive.thin_declaration(tmp_path, out, summary={})["self_weight_scaled"]
    # a new, lighter body: the runner's own acceleration is scaled again
    mass["planned_mass"] = 300.0
    again, said = drive.scaled_self_weight(tmp_path, path, out)
    assert again["values"]["acceleration"] == [0.0, -round(G * 30 / 300, 5), 0.0] and said
    # a body within 2 x the Real mass: the use's acceleration is put back
    mass["planned_mass"] = 45.0
    back, said = drive.scaled_self_weight(tmp_path, path, again)
    assert back["values"]["acceleration"] == [0.0, -G, 0.0] and "put back" in said


def test_the_use_s_own_acceleration_is_scaled_and_a_contradicting_one_refused(tmp_path, monkeypatch):
    path, plan, _ = _heavy_stand_in(tmp_path, monkeypatch)
    use = [0.0, -3 * G, 3 * G]
    swing = {**plan, "values": {**plan["values"], "acceleration": use, "acceleration_use": use}}
    out, said = drive.scaled_self_weight(tmp_path, path, swing)
    assert out["values"]["acceleration"] == [round(c * 30 / 600, 5) for c in use] and said
    # an acceleration that is neither the use's nor the runner's: never read as the use; refused, asking
    odd = {**plan, "values": {**plan["values"], "acceleration": [0, -2.0, 0], "acceleration_use": [0, -G, 0]}}
    assert drive.scaled_self_weight(tmp_path, path, odd) == (odd, None)
    assert "was edited while plan.values.acceleration_use is" in drive.self_weight_problem(tmp_path, odd)
    assert "which is the use?" in drive.self_weight_problem(tmp_path, odd)
    plain = {**plan, "values": {**plan["values"], "acceleration": [0, -2 * G, 0]}}
    plain["values"].pop("acceleration_use", None)
    assert drive.scaled_self_weight(tmp_path, path, plain) == (plain, None)
    assert "put the use's own (a swing of a few g" in drive.self_weight_problem(tmp_path, plain)
    # the agent's own correct scaling is kept as written
    typed = {**plan, "values": {**plan["values"], "acceleration": [0.0, -round(G * 30 / 600, 5), 0.0]}}
    assert drive.scaled_self_weight(tmp_path, path, typed) == (typed, None)
    # within 2 x, or no section stand-in (the route derived from the body): nothing written
    assert drive.scaled_self_weight(tmp_path, path, plan)[1] is not None
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": None, "reasons": []})
    assert drive.scaled_self_weight(tmp_path, path, plan)[1] is None


# --------------------------------------------- a preview with no support, the proposal said --

def test_a_preview_with_no_support_says_what_the_synthesis_proposed(tmp_path):
    proposal = {"dirichlet": [], "neumann": [
        {"name": "held_face", "selector": {"type": "surface_segment", "params": {"segment_ids": [0]}},
         "traction": [0, 0, 1]},
        {"name": "tip", "selector": {"type": "highest", "params": {"axis": "y", "fraction": 0.03}},
         "traction": None}]}
    request = ("Fix all components of the middle part: its face toward +z, between 0.4 m and 0.6 m. "
               "Apply a traction toward +z on the tip.")
    rows = [{"timestamp": 1.0, "purpose": "bc_synthesis", "prompt": json.dumps({"request": "old"}),
             "response": json.dumps(proposal), "error": "ValidationError: old"},
            {"timestamp": 10.0, "purpose": "bc_synthesis",
             "prompt": json.dumps({"request": request, "correction": "..."}),
             "response": json.dumps(proposal, indent=1), "error": "ValidationError: <root>: Value error, "
                                                                  "at least one Dirichlet BC is required"}]
    (tmp_path / "api_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    said = drive.synthesis_said(tmp_path, since=5.0)
    assert "supports (dirichlet): none" in said and "held_face (traction [0, 0, 1]), tip" in said
    assert "'Fix all components of the middle part: its face toward +z, between 0.4 m and 0.6 m.'" in said
    assert "asked again with them named" in said
    # a clipped transcript still names the regions by the list they stand in
    rows[1]["response"] = json.dumps(proposal, indent=1)[:200] + "… [900 chars]"
    (tmp_path / "api_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert "loads (neumann): held_face" in drive.synthesis_said(tmp_path, since=5.0)
    # nothing failed in this run: nothing said
    assert drive.synthesis_said(tmp_path, since=20.0) is None


# ------------------------------------------------------------- review fixes (round 10) --

def test_a_runner_scaled_self_weight_is_put_back_when_the_route_leaves_the_stand_in(tmp_path, monkeypatch):
    path, plan, _ = _heavy_stand_in(tmp_path, monkeypatch)
    scaled, said = drive.scaled_self_weight(tmp_path, path, plan)
    assert said and scaled["values"]["acceleration"] != [0.0, -G, 0.0]
    # the body is no longer the stand-in (its route is derived): no scaling applies, the runner's own
    # value is undone
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": None, "reasons": []})
    moved = dict(scaled)
    back, said = drive.scaled_self_weight(tmp_path, path, moved)
    assert back["values"]["acceleration"] == [0.0, -G, 0.0] and "which the body no longer is" in said
    assert json.loads(path.read_text())["values"]["acceleration"] == [0.0, -G, 0.0]
    record = lib.read_json(tmp_path / "operator" / drive.SELF_WEIGHT_SET)
    assert record["scaled"] is False and "put back" in record["restored"]
    # said once, and in the report
    assert drive.scaled_self_weight(tmp_path, path, back) == (back, None)
    extra = drive.thin_declaration(tmp_path, back, summary={})
    assert "put back" in extra["self_weight_restored"]
    lines = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, extra)
    assert "\n- self-weight scaling withdrawn by the runner: the use's acceleration" in lines
    # an acceleration the agent typed itself is never touched when no scaling applies
    own = {**moved, "values": {**moved["values"], "acceleration": [0, -2.0, 0]}}
    assert drive.scaled_self_weight(tmp_path, path, own) == (own, None)


def test_a_runner_scaled_self_weight_is_kept_while_the_masses_are_unread(tmp_path, monkeypatch):
    path, plan, mass = _heavy_stand_in(tmp_path, monkeypatch)
    scaled, _ = drive.scaled_self_weight(tmp_path, path, plan)
    # still the stand-in, its planned mass unread: nothing says it is within 2 x, so the value stays
    mass.pop("planned_mass")
    kept, said = drive.scaled_self_weight(tmp_path, path, scaled)
    assert kept == scaled and "kept its scaled acceleration" in said and "could not be checked" in said
    assert json.loads(path.read_text())["values"]["acceleration"] == scaled["values"]["acceleration"]
    # nor the brief's Real mass
    mass["planned_mass"] = 600.0
    (tmp_path / "operator" / "brief.md").write_text("## Size and material\nReal thinnest member: 0.002 m\n")
    kept, said = drive.scaled_self_weight(tmp_path, path, scaled)
    assert kept == scaled and "kept its scaled acceleration" in said


def test_joints_built_square_are_recorded_with_the_body_and_marked_in_the_report(tmp_path, monkeypatch):
    import trimesh
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.05"))
    squared = "bar 0, bar 1: their round joint did not union into one closed body"
    monkeypatch.setattr(drive, "build_frame", lambda spec: (
        trimesh.creation.box(extents=[1.0, 0.1, 0.1]),
        {"bars": 2, "boxes": 0, "joints": 1, "smallest_section_m": 0.1, "squared_at_joints": squared}))
    bars = tmp_path / "frame.json"
    bars.write_text(json.dumps([{"from": [0, 0, 0], "to": [0.5, 0, 0], "section": _round(0.1)},
                                {"from": [0.5, 0, 0], "to": [0.9, 0.4, 0], "section": _round(0.1)}]))
    args = dict(output_root=str(tmp_path), session_id="abc", shape=None, size=None, radius=None, length=None,
                axis=None, profile=None, hole=None, circle=None, bars=str(bars))
    drive.cmd_blank(argparse.Namespace(**args))
    out = printed[-1]
    assert out["ok"] and out["built"]["squared_at_joints"] == squared
    record = lib.read_json(Path(out["path"]).with_suffix(".built.json"))
    assert record["squared_at_joints"] == squared
    plan = {"geometry": {"asset_path": out["path"], "asset_kind": "surface"},
            "body_simplification": {"kept": "k", "left_out": "l"}}
    assert drive.squared_joints(plan) == squared
    assert drive.built_at_real(session, plan) is False
    extra = drive.thin_declaration(session, plan, summary={})
    lines = drive.history_section([{"run": "r", "body": {}, "ends": "", "load_N": 1.0}], None, extra)
    assert "\n- simplification (by the runner): round bars built square at joints, bar 0, bar 1" in lines
    # the solve stop and judge read declared.json, not the plan: the squared bars are seen there too
    drive.write_declared(session, plan)
    declared = lib.read_json(session / "operator" / "declared.json")
    assert drive.squared_joints(declared) == squared and drive.built_at_real(session, declared) is False
    assert drive.thin_declaration(session, declared, summary={})["joints_squared"] == squared
    # a body built without squaring records none
    Path(out["path"]).with_suffix(".built.json").write_text(json.dumps({"bars": 2}))
    assert drive.squared_joints(plan) is None


def test_the_depth_note_names_its_cause(tmp_path):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.05"))
    path = tmp_path / "operator" / "plan.json"
    # nothing declared moves the depth: the member at 0.5 m resolves at depth 7
    section = {"scale": {"physical_extent": 0.5}, "settings": {}, "body_simplification": SECTION}
    path.write_text(json.dumps(section))
    assert drive.prescribed_depth(tmp_path, path, section)[1] is None
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.02"))
    thin = {"scale": {"physical_extent": 1.0}, "settings": {}}
    path.write_text(json.dumps(thin))
    said = drive.prescribed_depth(tmp_path, path, thin)[1]
    assert said.startswith("the brief's Real thinnest member is under 2 leaves at depth 7 and resolves at depth 8")


def test_a_frame_s_sections_are_checked_at_the_depth_the_case_runs_at(tmp_path):
    (tmp_path / "operator").mkdir()
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.05"))
    spec = [{"from": [0, 0, 0], "to": [0, 0.5, 0], "section": _round(0.05)},
            {"from": [0, 0.5, 0], "to": [0.4, 0.5, 0], "section": _round(0.05)}]
    # the real member resolves at depth 7 under the frame's extent, no stand-in declared: depth 7
    assert drive.frame_sections_needed(spec, {}, tmp_path)["depth"] == 7
    # a body_simplification declares no depth; a named depth is kept
    assert drive.frame_sections_needed(spec, {"body_simplification": SECTION}, tmp_path)["depth"] == 7
    assert drive.frame_sections_needed(spec, {"settings": {"remesh_octree_depth": 8},
                                              "body_simplification": SECTION}, tmp_path)["depth"] == 8
    # a real member under 2 leaves at depth 7 that depth 8 resolves (0.012 m at 0.55 m: 1.5, 2.9): depth 8
    (tmp_path / "operator" / "brief.md").write_text(BRIEF.replace("MEMBER", "0.012"))
    assert drive.frame_sections_needed(spec, {}, tmp_path)["depth"] == 8


def test_union_blank_reads_each_member_once_for_all_its_groups(tmp_path, monkeypatch):
    import trimesh
    import union_blank
    sources = []
    for k in range(3):
        sources.append(str(tmp_path / f"m{k}.ply"))
        trimesh.creation.box(extents=[1, 1, 1]).export(sources[-1])
    jobs = [{"target": str(tmp_path / f"out{n}.ply"), "sources": group}
            for n, group in enumerate([sources[:2], sources[1:], sources])]
    (tmp_path / "groups.json").write_text(json.dumps(jobs))
    real_load, reads = trimesh.load, []
    monkeypatch.setattr(trimesh, "load", lambda s, **k: reads.append(s) or real_load(s, **k))
    monkeypatch.setattr(union_blank, "union", lambda meshes: trimesh.util.concatenate(meshes))
    union_blank.main(["--groups", str(tmp_path / "groups.json")])
    assert sorted(reads) == sorted(sources)
    assert all(Path(job["target"]).exists() for job in jobs)


# ------------------------------------------- per-part materials: the planned mass has a density --

def test_a_per_part_stand_in_is_weighed_and_its_self_weight_scaled(tmp_path, monkeypatch):
    """Per-part materials carry no density of their own: a held-out run (2026-10-10) had its
    planned mass None, so a stand-in at 2.9 x the real weight was neither scaled nor refused."""
    import trimesh
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "operator" / "brief.md").write_text(
        BRIEF.replace("MEMBER", "0.002").replace("Real mass: 30 kg", "Real mass: 0.5 kg")
        .replace("wood 1.2e10", "shell: alloy, rho 2700 kg/m3; handle: polymer, rho 1400 kg/m3"))
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])                 # 1e-3 m3
    monkeypatch.setattr(drive, "resolved_plan", lambda s, p: p)
    monkeypatch.setattr(drive, "generated_extents", lambda s: [0.1, 0.1, 0.1])
    monkeypatch.setattr(drive, "fill_fraction", lambda s: 1.0)
    monkeypatch.setattr(drive, "_generated_mesh", lambda s: box)
    monkeypatch.setattr(drive, "body_source", lambda s: None)
    monkeypatch.setattr(drive, "member_thickness", lambda m: None)
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["built thicker"]})
    plan = {"scale": {"unit_system": "SI (m-kg-s)", "up_axis": "y", "physical_extent": 0.1},
            "material": {"mode": "parts", "description": "alloy shell, polymer handle",
                         "parts": [{"id": 2, "youngs_modulus": 8e9, "density": 1400.0}]},
            "body_simplification": SECTION,
            "values": {"load": {"force": [0, -5, 0]}, "acceleration": [0, -G, 0]}}
    # before the parts are valued: the mean of the densities the brief writes
    resolution = drive.planned_resolution(session, plan)
    assert resolution["planned_mass"] == pytest.approx(2.05, rel=1e-3)
    assert "the brief writes" in resolution["planned_mass_density"]
    # the scaling weighs the body the workflow solves (its lattice-snapped scale), not the planned one
    solved = 2.05 * (drive.workflow_scale(box.bounds, 1, 0.1, 7) / 1.0) ** 3
    scaling = drive.self_weight_scaling(session, plan)
    assert scaling["ratio"] == pytest.approx(solved / 0.5, rel=1e-3) and not scaling["scaled"]
    out, said = drive.scaled_self_weight(session, None, plan)
    assert out["values"]["acceleration"][1] == pytest.approx(-G * 0.5 / solved, rel=1e-3)
    assert f"{solved / 0.5:.3g} x" in said
    # once the parts are valued: the part field's volume-weighted mean
    field = session / "materials" / "parts" / "field"
    field.mkdir(parents=True)
    grid = np.zeros((4, 4, 4, 3))
    grid[..., 2] = 1000.0
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[:2] = True
    np.save(field / "material_grid.npy", grid)
    np.save(field / "valid_mask.npy", mask)
    (field / "manifest.json").write_text(json.dumps({"channels": ["youngs_modulus", "poisson_ratio", "density"]}))
    assert drive.plan_density(session, plan) == (1000.0, "the part field's mean density")
    assert drive.planned_resolution(session, plan)["planned_mass"] == pytest.approx(1.0)
    # a uniform material keeps its own density
    uniform = {**plan, "material": {"youngs_modulus": 7e10, "poisson_ratio": 0.33, "density": 2700.0}}
    assert drive.plan_density(session, uniform) == (2700.0, "plan")


def test_the_scaled_self_weight_is_the_real_weight_on_the_body_the_workflow_solves(tmp_path, monkeypatch):
    """The scaling took the planned mass (the generated mesh x extent / its extent), but the workflow
    snaps the bounds to its octree lattice before scaling: physical.glb came out 3 % heavier, and the
    applied body load (physical.glb's volume x density x acceleration) 1.03 x the Real weight."""
    import importlib.util
    import trimesh
    spec = importlib.util.spec_from_file_location(
        "workflow_tool_for_test", lib.REPO_ROOT / "src" / "agents_wrapper" / "fem_workflow" / "scripts" / "workflow_tool.py")
    workflow_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow_tool)
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "geometry").mkdir()
    (session / "operator" / "brief.md").write_text(
        BRIEF.replace("MEMBER", "0.002").replace("Real mass: 30 kg", "Real mass: 0.5 kg"))
    lib.write_json(session / "session.json", {"octree_depth": 7})
    # a body off the lattice: its bounds are not multiples of the leaf
    body = trimesh.creation.box(extents=[0.37, 0.53, 0.21])
    body.apply_translation([0.013, -0.071, 0.029])
    body.export(session / "geometry" / "generated.glb")
    for name in ("member_thickness", "thin_walls", "thin_gaps", "planned_sections", "load_path_numbers"):
        monkeypatch.setattr(drive, name, lambda *a, **k: None)
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: {"route": "section", "reasons": ["built thicker"]})
    plan = {"scale": {"unit_system": "SI (m-kg-s)", "up_axis": "y", "physical_extent": 0.53},
            "material": {"youngs_modulus": 7e10, "poisson_ratio": 0.33, "density": 2700.0},
            "body_simplification": SECTION, "values": {"acceleration": [0, -G, 0]}}
    out, said = drive.scaled_self_weight(session, None, plan)
    assert said and "self-weight scaled" in said
    workflow_tool.scale_surface(argparse.Namespace(
        surface=session / "geometry" / "generated.glb", output=session / "geometry" / "physical.glb",
        axis="y", extent=0.53, units="SI (m-kg-s)", octree_depth=7))
    solved = trimesh.load(session / "geometry" / "physical.glb", force="mesh", process=False)
    assert drive.planned_resolution(session, plan)["solved_volume"] == pytest.approx(abs(solved.volume), rel=1e-4)
    applied = abs(solved.volume) * 2700.0 * math.hypot(*out["values"]["acceleration"])
    assert applied == pytest.approx(0.5 * G, rel=1e-3)                  # the Real weight
    planned = drive.planned_resolution(session, plan)["planned_mass"]
    assert abs(planned * G / (abs(solved.volume) * 2700.0 * G) - 1) > 0.01   # the old mass was off
