"""The case's load path and holds: every member of the brief's load path kept or declared absent
with its share, each row of the case run bound to its regions, a hold over a length spanning it,
a stand-in's section cut on the load's way to a support, and the route derived from the body.
No GPU, no API, no graph run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402
from test_operator_scripts import PLAN, THIN_BRIEF, _thin_run  # noqa: E402

G = [0.0, -9.80665, 0.0]
PATH_LINE = 'Load path: {"upper fitting": "series", "middle member": "series", "lower fitting": "series"}\n'


def _brief(session: Path, size: str = "", contacts: str = "", use: str = "") -> None:
    (session / "operator").mkdir(parents=True, exist_ok=True)
    (session / "operator" / "brief.md").write_text(
        f"## Use\n{use}\n## Contacts\n{contacts}\n## Size and material\nReal mass: 1 kg\n{size}")


# ------------------------------------------------------------ the load path --

def test_the_brief_lists_its_load_path_members_in_series_or_parallel():
    assert drive.load_path_in("x\n" + PATH_LINE) == {"upper fitting": "series", "middle member": "series",
                                                     "lower fitting": "series"}
    assert drive.load_path_in('Load path: {"a": "both"}') is None          # a role is series or parallel
    assert drive.load_path_in("no line here") is None


def test_a_simplified_body_keeps_every_member_or_declares_it_absent(tmp_path):
    session = tmp_path / "abc"
    _brief(session)
    body = {"kept": "the middle member", "left_out": "the end fittings"}
    plan = {"body_simplification": body}
    assert drive.path_gate(session, {}) is None                              # the brief's own body: by eye
    # no load path in the brief: a simplified body has nothing to be measured against
    assert "Load path: {" in drive.path_gate(session, plan)
    _brief(session, PATH_LINE)
    # every member not declared absent is kept: no free-text "kept" is read
    assert drive.path_gate(session, plan) is None
    assert drive.path_gate(session, {"body_simplification": {}}) is None
    # declared absent with a share that is no number, or a name the brief does not list
    words = {**body, "absent": {"upper fitting": "same pull", "lower fitting": 0.1}}
    assert "not 'same pull'" in drive.path_gate(session, {"body_simplification": words})
    stray = {**body, "absent": {"top part": 0.1}}
    assert drive.path_gate(session, {"body_simplification": stray}) is None      # off the load path: taken
    absent = {**body, "absent": {"upper fitting": 0.05, "lower fitting": 0.05}}
    assert drive.path_gate(session, {"body_simplification": absent}) is None


def test_a_member_left_out_makes_the_brief_case_the_whole_path(tmp_path):
    plan = {"prediction": {"max_displacement": 1e-5, "max_stress": 1e7},
            "body_simplification": {"kept": "the middle member", "left_out": "fittings",
                                    "absent": {"upper fitting": 0.1}}}
    assert not drive.built_at_real(tmp_path, plan)                             # not the brief's body
    # no prediction is demanded: the runner's own, at the Real section, is the brief's case
    assert drive.prediction_required(plan, tmp_path) is None


# ------------------------------------------------------------------ loads --

REGIONS = [{"name": "end_pull", "category": "neumann"}, {"name": "side_push", "category": "neumann"},
           {"name": "base", "category": "dirichlet"}]


def _table_session(tmp_path, **changes):
    from test_use_table import session_with, table
    return session_with(tmp_path, table(**changes))


def test_each_listed_load_is_bound_to_its_regions_or_dropped_with_its_comparison(tmp_path):
    # the case run's rows (A1, A2) are each bound, by label or by their load's name; nothing is dropped
    session = _table_session(tmp_path)
    plan = {"values": {"acceleration": G}, "loads": {"A1": {"regions": ["end_pull"]}}}
    bound = drive.load_bindings(session, plan, REGIONS)
    assert bound == {"A1": {"state": "applied", "regions": ["end_pull"]}, "A2": {"state": "unbound"}}
    assert "['A2'] of the case run are bound to no load region" in drive.loads_binding_problem(session, plan, REGIONS)
    both = {**plan, "loads": {"A1": {"regions": ["end_pull"]}, "side pull": {"regions": ["side_push"]}}}
    assert drive.loads_binding_problem(session, both, REGIONS) is None
    # names the case run does not have, or regions this preview does not have, are refused
    wrong = {"loads": {"tug": {"regions": ["end_pull"]}, "A1": {"regions": ["nowhere"]}}}
    said = drive.loads_binding_problem(session, wrong, REGIONS)
    assert "tug: not a row of the case run 'working'" in said and "['nowhere'] are not load regions" in said
    # a row outside the case run is the coverage's, not plan.loads'
    assert "A3: not a row of the case run" in drive.loads_binding_problem(
        session, {**both, "loads": {**both["loads"], "A3": {"regions": ["side_push"]}}}, REGIONS)


def test_a_briefs_only_other_load_takes_every_load_region(tmp_path):
    session = _table_session(tmp_path, rows={"A2": None})          # a run state of one row
    bound = drive.load_bindings(session, {"values": {}}, REGIONS)
    assert bound == {"A1": {"state": "applied", "regions": ["end_pull", "side_push"]}}
    # a case run of several rows writes plan.loads
    session = _table_session(tmp_path)
    assert "writes plan.loads" in drive.loads_binding_problem(session, {"values": {}}, REGIONS)


# ------------------------------------------------------------------ holds --

def _rod(tmp_path):
    """A 1 m round member along x, held either on its end cap or on a 0.2 m band at that end."""
    import trimesh
    session = tmp_path / "abc"
    (session / "boundary_conditions").mkdir(parents=True)
    (session / "geometry").mkdir()
    rod = trimesh.creation.cylinder(radius=0.02, height=1.0, sections=32)
    rod.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    rod.apply_translation([0.5, 0.0, 0.0])
    for _ in range(4):
        rod = rod.subdivide()
    rod.export(session / "geometry" / "physical.glb")
    centres, normals = rod.triangles_center, rod.face_normals
    end = normals[:, 0] < -0.9
    band = (centres[:, 0] < 0.2) & (np.abs(normals[:, 0]) < 0.1)
    np.savez(session / "boundary_conditions" / "resolved.npz", dirichlet_0=end, dirichlet_1=band)
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": [
        {"name": "end", "key": "dirichlet_0", "category": "dirichlet"},
        {"name": "band", "key": "dirichlet_1", "category": "dirichlet"}]})
    return session


def test_a_flat_end_face_under_a_stated_grip_length_is_refused_and_said_without_one(tmp_path):
    session = _rod(tmp_path)
    plan = {"contacts": {"hand hold": {"transmits": "force and moment"}}, "settings": {}}
    rows = lambda region: [{"name": region, "region": region, "contact": "hand hold"}]   # noqa: E731
    shape = drive.held_shape(session, ["band"], 7)
    assert abs(shape["span_m"] - 0.2) < 0.02 and shape["one_facing"] < 0.5          # along the member
    grip = {"hand hold": {"transmits": "force and moment", "kind": "grip", "over_m": 0.2}}
    [said] = drive.grip_span_problems(session, plan, rows("end"), grip)
    assert "over_m 0.2 m" in said and "one flat face" in said and "never one end face" in said
    assert drive.grip_span_problems(session, plan, rows("band"), grip) == []
    # a socket, a worn hold and a fastening that takes a moment are held over their length too; a rest,
    # or a fastening of force only, is not read
    for kind in ("socket", "worn", "fastening"):
        assert drive.grip_span_problems(session, plan, rows("end"), {"hand hold": {**grip["hand hold"], "kind": kind}})
    for entry in ({"kind": "rest"}, {"kind": "fastening", "transmits": "force only"}):
        assert drive.grip_span_problems(session, plan, rows("end"), {"hand hold": {**grip["hand hold"], **entry}}) == []
    # the length is the Contacts JSON's, never read from the bullet's words
    (session / "operator").mkdir(exist_ok=True)
    (session / "operator" / "brief.md").write_text(
        '## Contacts\n- hand hold: held at the end\n'
        '{"hand hold": {"transmits": "force and moment", "kind": "grip", "over_m": 0.2}}\n')
    assert "one flat face" in drive.hold_gate(session, plan, rows("end"), [], [])
    assert drive.hold_gate(session, plan, rows("band"), [], []) is None


# ------------------------------------------------------ the section cut --

def _ends_held_as_one(tmp_path):
    """A 2.6 m bar held at both ends by one region, loaded over its middle (a physical body on disk
    and its region masks)."""
    import trimesh
    session = tmp_path / "abc"
    (session / "boundary_conditions").mkdir(parents=True)
    (session / "geometry").mkdir()
    (session / "operator").mkdir()
    bar = trimesh.creation.box(extents=[2.6, 0.9, 0.37])
    bar.apply_translation([1.3, 0.45, 0.185])
    for _ in range(5):
        bar = bar.subdivide()
    bar.export(session / "geometry" / "physical.glb")
    centres = bar.triangles_center
    held = (centres[:, 0] < 0.2) | (centres[:, 0] > 2.4)
    loaded = (~held) & (bar.face_normals[:, 2] > 0.9)
    np.savez(session / "boundary_conditions" / "resolved.npz", dirichlet_0=held, neumann_1=loaded)

    def box(mask):
        v = bar.vertices[bar.faces[mask].reshape(-1)]
        return [v.min(axis=0).tolist(), v.max(axis=0).tolist()]
    regions = [{"name": "ends", "key": "dirichlet_0", "category": "dirichlet", "bounds": box(held)},
               {"name": "front", "key": "neumann_1", "category": "neumann", "bounds": box(loaded),
                "centroid": centres[loaded].mean(axis=0).tolist()}]
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": regions})
    return session, regions


def test_the_cut_counts_a_support_by_its_pieces_and_lies_on_the_loads_way(tmp_path, monkeypatch):
    session, regions = _ends_held_as_one(tmp_path)
    supports, loads = drive._cut_regions(session, {}, regions)
    assert len(supports) == 2 and len(loads) == 1                        # the two ends, apart
    at = drive.default_cut_at(session, {}, "x")
    assert 0.6 < at < 0.9                                                  # halfway from 0.2 to the middle


def test_no_default_cut_from_regions_older_than_the_body(tmp_path):
    import os
    session, _ = _ends_held_as_one(tmp_path)
    old = (session / "geometry" / "physical.glb").stat().st_mtime - 60
    os.utime(session / "boundary_conditions" / "resolved.json", (old, old))
    assert drive.default_cut_at(session, {}, "x") is None


# ------------------------------------------------------------ the route --

def test_a_thickened_body_is_no_built_body(tmp_path, monkeypatch):
    # the route is derived from the body: a built body past the real member and mass is the stand-in
    monkeypatch.setattr(drive, "body_provenance", lambda session: "built")
    # a thin real member alone makes no stand-in: the body built at it is the real one
    monkeypatch.setattr(drive, "member_unresolvable", lambda s, p: True)
    assert drive.body_route(tmp_path, {}, {})["route"] == "real"
    thick = drive.body_route(tmp_path, {}, {"member_over_real": 2.0, "mass_over_real": 2.4})
    assert thick["route"] == "section" and "2.0 x the real one and its mass 2.4 x" in thick["reasons"][0]
    # the voxel member estimate alone decides nothing: a real-section body at its real mass reads 1.89
    for r in ({"member_over_real": 1.89, "mass_over_real": 1.0}, {"member_over_real": 1.2, "mass_over_real": 2.0},
              {"member_over_real": 2.4, "mass_over_real": 1.4}):
        assert drive.body_route(tmp_path, {}, r)["route"] == "real"
    alone = drive.body_route(tmp_path, {}, {"member_over_real": 2.8, "mass_over_real": 1.0})
    assert alone["route"] == "section" and "2.8 x the real one (over 2.5)" in alone["reasons"][0]
    assert drive.body_route(tmp_path, {}, {"member_over_real": 2.8})["route"] == "section"


# ------------------------------------------- the declarations the report reads --

def test_the_report_reads_the_loads_the_run_declared(tmp_path):
    # the solve-backend stop and judge rebuild the report from declared.json: the plan's loads are in it
    session = _table_session(tmp_path)
    plan = {**PLAN, "values": {"acceleration": G},
            "loads": {"A1": {"regions": ["end_pull"]}, "A2": {"regions": ["side_push"]}}}
    drive.write_declared(session, plan)
    declared = lib.read_json(session / "operator" / "declared.json")
    assert declared["loads"] == plan["loads"]
    summary = {"regions": [{**r, "applied_force": [0.0, -1.0, 0.0]} for r in REGIONS if r["category"] == "neumann"]}
    assert drive.thin_declaration(session, declared, summary).get("unbound_loads", []) == []


def test_a_load_applied_as_an_acceleration_is_bound_to_it(tmp_path):
    session = _table_session(tmp_path, rows={"A2": None})          # a run state of one row
    plan = {"values": {"acceleration": [0.0, -40.0, 0.0]}, "loads": {"A1": {"acceleration": True}}}
    assert drive.loads_binding_problem(session, plan, REGIONS) is None
    assert drive.load_bindings(session, plan, REGIONS)["A1"] == {"state": "applied", "regions": ["acceleration"]}
    still = {"values": {}, "loads": {"A1": {"acceleration": True}}}
    assert "values.acceleration and body_force are zero" in drive.loads_binding_problem(session, still, REGIONS)


def test_each_bound_row_is_applied_with_its_size_along_the_plans_direction(tmp_path):
    session = _table_session(tmp_path)                             # A1 200 N, A2 50 N, the working state
    plan = {"values": {"neumann_values": {"end_pull": [0.0, -3.0, 0.0]}, "load": {"force": [5.0, 0.0, 0.0]}},
            "loads": {"A1": {"regions": ["end_pull"]}, "A2": {"regions": ["end_pull", "side_push"]}}}
    out, note, problem = drive.use_table_values(session, plan, REGIONS)
    assert problem is None and "A1 200 N on ['end_pull']" in note and "A2 50 N on ['end_pull', 'side_push']" in note
    # the size is the table's, the direction the plan's (a row bound to two regions shares its size)
    assert out["values"]["neumann_values"] == {"end_pull": {"force": [0.0, -225.0, 0.0]},
                                               "side_push": {"force": [25.0, 0.0, 0.0]}}
    # a row's own direction wins
    turned = _table_session(tmp_path, rows={"A2": {"direction": [0, 0, 2]}})
    out, _, _ = drive.use_table_values(turned, plan, REGIONS)
    assert out["values"]["neumann_values"]["side_push"] == {"force": [0.0, 0.0, 25.0]}
    # no direction anywhere: said, by region
    session = _table_session(tmp_path)
    _, _, problem = drive.use_table_values(session, {**plan, "values": {}}, REGIONS)
    assert "load regions ['end_pull', 'side_push'] carry Use table rows but no direction" in problem


def test_a_rows_size_is_the_total_on_its_regions_and_a_loose_value_is_refused(tmp_path):
    session = _table_session(tmp_path, rows={"A2": None})          # a run state of one row: A1, 200 N
    # a pressure on two opposite walls: the total shared, never doubled
    walls = {"values": {"neumann_values": {"end_pull": [0.0, -1.0, 0.0], "side_push": [0.0, 1.0, 0.0]}},
             "loads": {"A1": {"regions": ["end_pull", "side_push"]}}}
    out, _, problem = drive.use_table_values(session, walls, REGIONS)
    assert problem is None and out["values"]["neumann_values"] == {"end_pull": {"force": [0.0, -100.0, 0.0]},
                                                                   "side_push": {"force": [0.0, 100.0, 0.0]}}
    # the two faces of a squeeze are a row each, the second "as" the first: F on each
    pair = _table_session(tmp_path, rows={"A2": {"source": "as A1", "size": None, "held": ["side grip"]}})
    squeeze = {**walls, "loads": {"A1": {"regions": ["end_pull"]}, "A2": {"regions": ["side_push"]}}}
    out, _, problem = drive.use_table_values(pair, squeeze, REGIONS)
    assert problem is None and out["values"]["neumann_values"] == {"end_pull": {"force": [0.0, -200.0, 0.0]},
                                                                   "side_push": {"force": [0.0, 200.0, 0.0]}}
    session = _table_session(tmp_path, rows={"A2": None})
    # a load region with a value that no row binds: refused
    loose = {"values": {"neumann_values": {"end_pull": [0.0, -1.0, 0.0], "side_push": [1.0, 0.0, 0.0]}},
             "loads": {"A1": {"regions": ["end_pull"]}}}
    _, _, problem = drive.use_table_values(session, loose, REGIONS)
    assert "load regions ['side_push'] have a value in the plan that no row of the run state binds" in problem


def test_a_twist_row_is_held_to_the_plans_couple(tmp_path):
    session = _table_session(tmp_path, rows={"A2": None, "A1": {"unit": "N m", "size": 4.0, "source": "derived: 4 N m",
                                                               "effect": {"top plate": 4, "bottom plate": 4}}})
    regions = [{"name": "end_pull", "category": "neumann", "centroid": [0.1, 0, 0], "loaded_area": 1.0},
               {"name": "side_push", "category": "neumann", "centroid": [-0.1, 0, 0], "loaded_area": 1.0},
               {"name": "base", "category": "dirichlet"}]
    couple = {"values": {"neumann_values": {"end_pull": {"force": [0.0, 20.0, 0.0]},
                                            "side_push": {"force": [0.0, -20.0, 0.0]}}},
              "loads": {"A1": {"regions": ["end_pull", "side_push"]}}}
    assert drive.use_table_values(session, couple, regions)[2] is None         # 20 N x 0.2 m = 4 N m
    couple["values"]["neumann_values"]["end_pull"] = {"force": [0.0, 30.0, 0.0]}
    couple["values"]["neumann_values"]["side_push"] = {"force": [0.0, -30.0, 0.0]}
    assert "A1 is 4 N m, and the plan's forces on ['end_pull', 'side_push'] make a couple of 6 N m" in \
        drive.use_table_values(session, couple, regions)[2]
    # with its axis as the row's "direction", the runner makes the couple itself
    axial = _table_session(tmp_path, rows={"A2": None, "A1": {"unit": "N m", "size": 4.0, "source": "derived: 4 N m",
                                                              "direction": [0, 0, 1]}})
    out, said, problem = drive.use_table_values(axial, {**couple, "values": {}}, regions)
    assert problem is None and "4 N m about [0.0, 0.0, 1.0] as a couple" in said
    forces = out["values"]["neumann_values"]
    assert forces["end_pull"]["force"] == pytest.approx([0.0, 20.0, 0.0])
    assert forces["side_push"]["force"] == pytest.approx([0.0, -20.0, 0.0])


def test_a_table_edited_after_the_values_card_is_a_change_there_and_an_unchanged_one_is_not(tmp_path):
    session = _table_session(tmp_path)
    plan = {"values": {"neumann_values": {"end_pull": [0.0, -3.0, 0.0]}, "load": {"force": [5.0, 0.0, 0.0]}},
            "loads": {"A1": {"regions": ["end_pull"]}, "A2": {"regions": ["end_pull", "side_push"]}}}
    sent = {"use_table": drive._normalized(drive.use_table_digest(session, plan))}
    with_digest = lambda: {**plan, "use_table": drive.use_table_digest(session, plan)}    # noqa: E731
    assert drive.changed_sections(with_digest(), sent, "material_input") == []        # unchanged: no rewind
    _table_session(tmp_path, rows={"A1": {"size": 250, "source": "H5 top"}})           # the brief edited
    assert drive.changed_sections(with_digest(), sent, "material_input") == [("use_table", "bc_values")]
    # a binding changed is one too
    rebound = {**plan, "loads": {"A1": {"regions": ["side_push"]}, "A2": {"regions": ["end_pull"]}}}
    sent = {"use_table": drive._normalized(drive.use_table_digest(session, plan))}
    assert drive.changed_sections({**rebound, "use_table": drive.use_table_digest(session, rebound)}, sent,
                                  "material_input") == [("use_table", "bc_values")]


def test_the_tables_forces_are_the_runners_own_on_the_next_run(tmp_path):
    # the values sent carry the table's forces, the plan does not: no plan change, so no rewind
    session = _table_session(tmp_path)
    plan = {"values": {"neumann_values": {"end_pull": [0.0, -3.0, 0.0]}, "load": {"force": [5.0, 0.0, 0.0]}},
            "loads": {"A1": {"regions": ["end_pull"]}, "A2": {"regions": ["end_pull", "side_push"]}}}
    for _ in range(2):                                             # two runs at the values card
        sent_plan, _, problem = drive.use_table_values(session, plan, REGIONS)
        assert problem is None
        sent = {"values": drive._normalized(sent_plan["values"])}
        assert drive.changed_sections(plan, sent, "material_input", *drive.statics_filled(session)) == []
    # a real change of the plan's values is still one
    edited = {**plan, "values": {**plan["values"], "acceleration": [0.0, -9.8, 0.0]}}
    assert drive.changed_sections(edited, sent, "material_input", *drive.statics_filled(session)) == [
        ("values", "bc_values")]


def test_a_session_without_a_use_table_binds_nothing(tmp_path):
    # a session written before the Use table: nothing to bind, nothing refused
    session = tmp_path / "abc"
    _brief(session)
    assert drive.load_bindings(session, {"values": {"acceleration": G}}, REGIONS) == {}
    assert drive.loads_binding_problem(session, {"loads": {"x": {"regions": ["end_pull"]}}}, REGIONS) is None


def test_member_names_match_whatever_their_case_and_plural(tmp_path):
    session = tmp_path / "abc"
    _brief(session, 'Load path: {"Legs": "parallel", "seat brace": "series"}\n')
    body = {"kept": "the four leg and the Seat Braces", "left_out": "the trim"}
    assert drive.path_gate(session, {"body_simplification": body}) is None
    absent = {**body, "kept": "the legs", "absent": {"seat braces": 0.1}}
    assert drive.path_gate(session, {"body_simplification": absent}) is None


def test_a_bars_body_has_a_bar_for_each_member_it_keeps(tmp_path):
    session = tmp_path / "abc"
    _brief(session, 'Load path: {"post": "series", "arm": "series", "strut": "parallel"}\n')
    asset = tmp_path / "frame.glb"
    lib.write_json(asset.with_suffix(".bars.json"), {"bars": [{"from": [0, 0, 0], "to": [0, 1, 0]},
                                                              {"from": [0, 1, 0], "to": [1, 1, 0]}]})
    body = {"kept": "the post, the arm and the strut", "left_out": "l"}
    plan = {"geometry": {"asset_path": str(asset)}, "body_simplification": body}
    assert "2 bars and boxes for the 3 members kept" in drive.path_gate(session, plan)


def test_a_declared_simplification_without_a_load_path_is_refused_at_the_first_run(tmp_path, monkeypatch):
    plan = {**PLAN, "body_simplification": {"kept": "k", "left_out": "l"}}
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    import re
    (session / "operator" / "brief.md").write_text(re.sub(r"Load path: \{[^\n]*\n?", "",
                                                          THIN_BRIEF.replace("MEMBER", "0.05")))
    out, sent = _thin_run(tmp_path, monkeypatch, "0.05", plan)
    assert out["stop"] == "error" and not sent and "Load path: {" in out["error"]
