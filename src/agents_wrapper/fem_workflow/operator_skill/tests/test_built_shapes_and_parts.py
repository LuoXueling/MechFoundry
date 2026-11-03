"""Built bodies beyond one section (a revolved profile, several separately held pieces), the
stand-in's section cut where asked, the paint budget and per-part colours, and the parts table read
from the segmentation the parts were valued on.  No GPU, no API, no graph run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
from conftest import tabled  # noqa: E402
import sessionlib as lib  # noqa: E402
from test_operator_scripts import PLAN, _built_session  # noqa: E402


# ------------------------------------------------------------- parts table --

def _segments(path: Path, method: str, rows: list[tuple[int, float, list[float]]], surface: Path | None = None) -> None:
    """A segmentation's json, its npz and the stamp of the surface it was computed on."""
    from src.agents_wrapper.fem_workflow.monitor import _digest
    path.parent.mkdir(parents=True, exist_ok=True)
    lib.write_json(path, {"method": method, "segments": [
        {"id": i, "area_fraction": a, "centroid": c} for i, a, c in rows]})
    np.savez(path.with_suffix(".npz"), segment_ids=np.zeros(1, dtype=np.int64))
    lib.write_json(path.with_name("segments.source.json"),
                   {"surface": str(surface), "sha256": _digest(surface) if surface else ""})


def test_the_parts_table_reads_the_segmentation_the_parts_were_valued_on(tmp_path):
    """A held-out run (2026-10-10): the regions were segmented by shape diameter and the material by
    the material step's own PartField; the table took the regions' segments by id.  The choice is
    the graph's: segments whose stamp records the current surface."""
    session = tmp_path / "s"
    surface = session / "geometry" / "physical.glb"
    surface.parent.mkdir(parents=True)
    surface.write_bytes(b"the body as it is now")
    lib.write_json(session / "session.json", {"segmentation_method": "shape_diameter", "surface_path": str(surface)})
    _segments(session / "boundary_conditions" / "segments.json", "shape_diameter",
              [(k, 0.05, [0.0, 0.1 * k, 0.0]) for k in range(8)], surface)
    own = [(0, 0.5, [0.0, 0.3, 0.0]), (1, 0.3, [0.0, 0.1, 0.0]), (2, 0.2, [0.0, 0.0, 0.0])]
    _segments(session / "materials" / "parts" / "segments.json", "partfield", own, surface)
    table = drive._segment_table(session)
    assert sorted(table) == [0, 1, 2] and abs(sum(r["area_fraction"] for r in table.values()) - 1.0) < 1e-6
    assert table[0]["centroid"] == [0.0, 0.3, 0.0]
    # the regions' segments are PartField's and current: the material step takes them, so does the table
    _segments(session / "boundary_conditions" / "segments.json", "partfield",
              [(0, 0.6, [1.0, 0.0, 0.0]), (1, 0.4, [2.0, 0.0, 0.0])], surface)
    lib.write_json(session / "session.json", {"segmentation_method": "partfield", "surface_path": str(surface)})
    assert sorted(drive._segment_table(session)) == [0, 1]
    # computed on an earlier body (whatever the files' times): the material step's own, current ones
    _segments(session / "boundary_conditions" / "segments.json", "partfield",
              [(0, 0.6, [1.0, 0.0, 0.0]), (1, 0.4, [2.0, 0.0, 0.0])], None)
    assert sorted(drive._segment_table(session)) == [0, 1, 2]
    surface.write_bytes(b"a rebuilt body")                 # neither matches: no table rather than a wrong one
    assert drive._segment_table(session) == {}


# ------------------------------------------------------- section cut where asked --

def test_the_stand_ins_section_is_cut_at_the_asked_station_not_the_nearest_precomputed_one(tmp_path):
    """A floor slab under a hollow body: the valid cut range (between the held underside and the
    loaded floor top) lies below the first precomputed station (a tenth of the height), so a cut
    snapped to a station never separated them, whatever at_m said (a held-out run, 2026-10-10)."""
    profile = "0,0 0.58,0 0.58,1.07 0.54,1.07 0.54,0.04 0.04,0.04 0.04,1.07 0,1.07"
    tub = drive.build_blank("extrude", length=0.3, axis="z", profile=profile)
    tub.apply_translation(-tub.bounds[0])
    session = _built_session(tmp_path, tub, 0.003, "0.04 m deep, I 4e-7 m4, A 1.2e-3 m2")
    tub.export(session / "geometry" / "physical.glb")
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": [
        {"name": "ground", "category": "dirichlet", "bounds": [[0, 0, 0], [0.58, 0.0, 0.3]]},
        {"name": "floor_load", "category": "neumann", "bounds": [[0.04, 0.04, 0], [0.54, 0.04, 0.3]],
         "centroid": [0.29, 0.04, 0.15]}]})
    plan = {**PLAN, "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "longest", "physical_extent": 1.07},
            "material": {"youngs_modulus": 1e9, "poisson_ratio": 0.4, "density": 950},
            "settings": {"remesh_octree_depth": 8}, "body_simplification": {"kept": "k", "left_out": "l"}}
    # cut exactly where asked, not at the nearest precomputed station: the whole floor slab
    exact = drive.section_at(session, plan, "y", 0.02)
    assert exact["at_m"] == 0.02 and abs(max(p["area_m2"] for p in exact["pieces"]) - 0.58 * 0.3) < 0.01
    # the runner's own cut, from the regions: no field of the plan places it, and nothing is refused
    eq = drive.section_equivalence(session, plan)
    assert "error" not in eq and "'floor_load'" in eq["cut_across_chosen"]


# ------------------------------------------------------------- revolved bodies --

BRIEF = ("## Use\nA load rests on it. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n"
         "## Contacts\nits base on the ground, held fully\n## Load\n20 N on the top, 2 kg x g x 1\n"
         "## Size and material\nReal mass: 0.3 kg\nReal thinnest member: 0.008 m\n"
         "Real section: 0.1 m deep, I 2.0e-6 m4, A 2.4e-3 m2\n0.1 m tall\n"
         "## Prediction\nwalls in compression, 0.1 MPa, microns")
BRIEF = tabled(BRIEF)


def _blank_session(tmp_path):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(BRIEF)
    return session


def _blank(tmp_path, monkeypatch, **kw):
    import argparse
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    args = {"output_root": str(tmp_path), "session_id": "abc", "shape": None, "size": None, "radius": None,
            "length": None, "axis": None, "profile": None, "hole": None, "circle": None, "bars": None,
            "revolve": None, "wall": None, **kw}
    drive.cmd_blank(argparse.Namespace(**args))
    return printed[-1]


@pytest.mark.parametrize("profile, wall, closed_bottom", [
    ("0,0 0.04,0 0.04,0.1", 0.008, True),                                   # an open shell: a wall line from the axis
    ("0,0 0.03,0.002 0.055,0.015 0.07,0.04", 0.008, True),                  # a flaring open shell
    ("0,0 0.15,0.03 0.15,0.045 0,0.015", None, True),                       # a shallow disc: a closed outline
    ("0,0 0.012,0 0.012,0.1 0.08,0.13 0.08,0.15 0,0.15", None, True),       # a disc on a stem
    ("0.05,0 0.06,0 0.06,0.1 0.05,0.1", None, False),                       # a tube, open along the axis
])
def test_a_revolved_profile_makes_one_watertight_body(profile, wall, closed_bottom):
    """Hollow and open shells had no route in a held-out run (2026-10-10): a profile in (r, z)
    revolved about the axis builds them as one body, closed where the outline runs along it."""
    outline = drive.revolve_outline(profile, wall)
    mesh = drive.build_revolved(outline, "y")
    assert mesh.is_watertight and mesh.volume > 0 and len(mesh.split(only_watertight=False)) == 1
    summary = drive.revolve_summary(outline, "y")
    assert (summary["solid_on_axis"] is not None) == closed_bottom
    assert abs(mesh.extents[0] - 2 * summary["outer_radius_m"]) < 1e-3          # the axis on y: x and z round it
    if wall:
        assert abs(drive.revolve_walls(outline) - wall) < 1e-6


def test_revolve_refuses_an_outline_it_cannot_turn_into_one_body():
    with pytest.raises(drive.PlanError, match="r >= 0"):
        drive.revolve_outline("-0.01,0 0.04,0 0.04,0.1", 0.008)
    with pytest.raises(drive.PlanError, match="crosses itself"):
        drive.revolve_outline("0.01,0 0.06,0.05 0.06,0 0.01,0.08")
    with pytest.raises(drive.PlanError, match="pinch"):
        drive.revolve_outline("0,0.05 0.05,0 0.05,0.1")                  # touches the axis at one point
    # a post's thickness is its diameter: the section is mirrored across the axis
    assert abs(drive.revolve_walls(drive.revolve_outline("0,0 0.01,0 0.01,0.3 0,0.3")) - 0.02) < 1e-6


def test_blank_revolve_prints_what_was_built_and_warns_of_a_wall_under_two_leaves(tmp_path, monkeypatch):
    session = _blank_session(tmp_path)
    out = _blank(tmp_path, monkeypatch, revolve="0,0 0.04,0 0.04,0.1", wall=0.008, axis="y")
    assert out["ok"], out
    revolved = out["built"]["revolved"]
    assert "body of revolution about y" in revolved["what"] and "closed bottom" in revolved["what"]
    assert revolved["thinnest_wall_m"] == 0.008 and revolved["walls_needed"]["thinnest_wall_leaves"] > 2
    assert Path(out["path"]).name.startswith("blank_revolve_") and out["plan"]["scale"]["up_axis"] == "longest"
    assert "thin member" not in str(out.get("warning"))
    import trimesh
    built = trimesh.load(out["path"], force="mesh")
    assert built.is_watertight and abs(built.bounds[0]).max() < 1e-9            # its lowest corner at the origin
    # a wall under 2 leaves at the plan's depth is built, and warned with its numbers
    lib.write_json(session / "operator" / "plan.json", {"settings": {"remesh_octree_depth": 7}})
    thin = _blank(tmp_path, monkeypatch, revolve="0,0 0.04,0 0.04,0.1", wall=0.0015, axis="y")
    assert thin["ok"] and "thin member" in thin["warning"] and "thinnest wall 0.0015 m" in thin["warning"]
    assert "a warning" in thin["warning"]
    assert "--wall belongs to --revolve" in _blank(tmp_path, monkeypatch, shape="box", size=[1, 1, 1], wall=0.01)["error"]


def test_a_revolved_body_takes_bars_beside_it_as_one_body(tmp_path, monkeypatch):
    """Members attached to a shell of revolution: bars in the revolve's frame (the axis through the origin), each end that meets
    the revolved body run into its wall; the pieces are joined into one body.  The union's curved
    faces are no planar facets for the test's remesh stand-in (conftest), so the union is kept as it
    comes; MeshLab's remesh of such a body is checked by hand (watertight, one piece)."""
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    session = _blank_session(tmp_path)
    arm = {"bars": [{"from": [0.04, 0.08, 0.0], "to": [0.09, 0.08, 0.0], "section": [0.012, 0.012]},
                       {"from": [0.09, 0.08, 0.0], "to": [0.09, 0.02, 0.0], "section": [0.012, 0.012]},
                       {"from": [0.09, 0.02, 0.0], "to": [0.04, 0.02, 0.0], "section": [0.012, 0.012]}]}
    (session / "operator" / "arm.json").write_text(json.dumps(arm))
    out = _blank(tmp_path, monkeypatch, revolve="0,0 0.04,0 0.04,0.1", wall=0.008, axis="y",
                 bars=str(session / "operator" / "arm.json"))
    assert out["ok"], out
    assert Path(out["path"]).name.startswith("blank_revolve_bars_") and "pieces" not in out["built"]
    record = lib.read_json(Path(out["path"]).with_suffix(".bars.json"))
    assert record["revolve"]["wall"] == 0.008
    assert drive.bars_count({"geometry": {"asset_path": out["path"]}}) == 4      # three bars and the shell
    # a bar that stops short of the wall leaves a second piece, which is named
    apart = {"bars": [{"from": [0.07, 0.08, 0.0], "to": [0.12, 0.08, 0.0], "section": [0.012, 0.012]}]}
    (session / "operator" / "apart.json").write_text(json.dumps(apart))
    out = _blank(tmp_path, monkeypatch, revolve="0,0 0.04,0 0.04,0.1", wall=0.008, axis="y",
                 bars=str(session / "operator" / "apart.json"))
    assert not out["ok"] and "run each bar that meets the revolved body into its wall" in out["error"]


# --------------------------------------------------------------- separate pieces --

PORTALS = [  # two frames standing apart, each on its own two feet
    {"from": [0.0, 0.0, 0.0], "to": [0.0, 0.3, 0.0], "section": [0.04, 0.04]},
    {"from": [0.0, 0.3, 0.0], "to": [0.0, 0.3, 0.8], "section": [0.04, 0.04]},
    {"from": [0.0, 0.3, 0.8], "to": [0.0, 0.0, 0.8], "section": [0.04, 0.04]},
    {"from": [0.6, 0.0, 0.0], "to": [0.6, 0.3, 0.0], "section": [0.04, 0.04]},
    {"from": [0.6, 0.3, 0.0], "to": [0.6, 0.3, 0.8], "section": [0.04, 0.04]},
    {"from": [0.6, 0.3, 0.8], "to": [0.6, 0.0, 0.8], "section": [0.04, 0.04]},
]


def test_separate_pieces_are_built_when_declared_each_held(monkeypatch):
    """In a held-out run (2026-10-10) two frames that each stand on their own feet were refused as
    "not one body", and a member joining them was invented.  (The test's facet remesh stand-in joins
    the coplanar feet of two pieces, so the union is kept as it comes; MeshLab's remesh keeps the two
    pieces watertight: test_separate_pieces_and_a_handle_survive_the_real_remesh.)"""
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    with pytest.raises(drive.PlanError, match="pieces_supported") as refused:
        drive.build_frame({"bars": PORTALS})
    assert "never an invented one" in str(refused.value)
    mesh, built = drive.build_frame({"bars": PORTALS, "pieces_supported": True})
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 2
    assert built["pieces"] == 2 and "every piece needs a held region" in built["pieces_note"]


@pytest.mark.slow
@pytest.mark.real_remesh
def test_separate_pieces_and_a_handle_survive_the_real_remesh():
    mesh, built = drive.build_frame({"bars": PORTALS, "pieces_supported": True})
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 2 and built["pieces"] == 2
    shell = drive.build_revolved(drive.revolve_outline("0,0 0.04,0 0.04,0.1", 0.008), "y")
    arm = [{"from": [0.04, 0.08, 0.0], "to": [0.09, 0.08, 0.0], "section": [0.012, 0.012]},
              {"from": [0.09, 0.08, 0.0], "to": [0.09, 0.02, 0.0], "section": [0.012, 0.012]},
              {"from": [0.09, 0.02, 0.0], "to": [0.04, 0.02, 0.0], "section": [0.012, 0.012]}]
    mesh, _ = drive.build_frame({"bars": arm}, extra=shell)
    assert mesh.is_watertight and len(mesh.split(only_watertight=False)) == 1


def test_a_piece_no_support_holds_is_refused_at_the_regions(tmp_path, monkeypatch):
    import trimesh
    monkeypatch.setattr(drive, "remesh_built", lambda mesh, edge=None: mesh)
    mesh, _ = drive.build_frame({"bars": PORTALS, "pieces_supported": True})
    session = tmp_path / "abc"
    (session / "geometry").mkdir(parents=True)
    (session / "boundary_conditions").mkdir()
    mesh.export(session / "geometry" / "physical.glb")
    mesh = trimesh.load(session / "geometry" / "physical.glb", force="mesh", process=False)
    monkeypatch.setattr(drive, "regions_on_body", lambda s: True)
    centres = mesh.triangles_center
    feet = centres[:, 1] < 1e-6                                     # the four feet's undersides
    first = feet & (centres[:, 0] < 0.3)
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"name": "feet", "key": "feet", "category": "dirichlet"}]})
    np.savez(session / "boundary_conditions" / "resolved.npz", feet=first)
    said = drive.body_pieces_problem(session)
    assert said and "2 separate solid pieces" in said and "floats" in said
    np.savez(session / "boundary_conditions" / "resolved.npz", feet=feet)
    assert drive.body_pieces_problem(session) is None
    # each piece's support is a contact the brief declared: a region no brief contact binds is refused
    (session / "operator").mkdir(exist_ok=True)
    (session / "operator" / "brief.md").write_text('## Contacts\n{"floor contact": "force and moment"}\n')
    second = feet & (centres[:, 0] > 0.3)
    lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": [
        {"name": "feet", "key": "feet", "category": "dirichlet"},
        {"name": "extra_clamp", "key": "extra_clamp", "category": "dirichlet"}]})
    np.savez(session / "boundary_conditions" / "resolved.npz", feet=first, extra_clamp=second)
    plan = {"contacts": {"floor contact": {"transmits": "force and moment", "regions": ["feet"]}}}
    said = drive.body_pieces_problem(session, plan)
    assert said and "['extra_clamp']" in said and "no contact of the brief's Contacts" in said
    plan = {"contacts": {"floor contact": {"transmits": "force and moment", "regions": ["feet", "extra_clamp"]}}}
    assert drive.body_pieces_problem(session, plan) is None


def test_a_sealed_void_is_no_piece_and_a_revolve_of_separate_parts_is_refused(tmp_path, monkeypatch):
    """A sealed hollow body has two surface components and one solid: counted by surface, its void's
    inner skin read as an unheld piece (a review, 2026-10-10)."""
    sealed = drive.build_revolved(drive.revolve_outline("0,0 0.05,0 0.05,0.1 0,0.1", 0.008), "y")
    assert drive.face_pieces(sealed)[1] == 2 and drive.solid_pieces(sealed)[1] == 1
    session = tmp_path / "abc"
    (session / "geometry").mkdir(parents=True)
    (session / "boundary_conditions").mkdir()
    sealed.export(session / "geometry" / "physical.glb")
    import trimesh
    mesh = trimesh.load(session / "geometry" / "physical.glb", force="mesh", process=False)
    monkeypatch.setattr(drive, "regions_on_body", lambda s: True)
    base = mesh.triangles_center[:, 1] < mesh.bounds[0][1] + 1e-6
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"name": "base", "key": "base", "category": "dirichlet"}]})
    np.savez(session / "boundary_conditions" / "resolved.npz", base=base)
    assert drive.body_pieces_problem(session) is None
    # two parts meeting only along the axis: refused before anything is built, the axis stretch named
    _blank_session(tmp_path / "b")
    out = _blank(tmp_path / "b", monkeypatch, axis="y",
                 revolve="0,0 0.05,0 0.05,0.01 0,0.04 0,0.06 0.05,0.09 0.05,0.1 0,0.1")
    assert not out["ok"] and "2 separate pieces, joined only along the axis" in out["error"]
    assert "(0, 0.04)-(0, 0.06)" in out["error"]


def test_the_summary_counts_loose_pieces_by_solid_not_by_surface(tmp_path):
    """A closed hollow body's inner skin is a second surface component, not a loose piece: a held-out
    run (2026-10-10) was warned of one at three stops."""
    import trimesh
    import inspect_session

    def summary(name, mesh, components):
        session = tmp_path / name
        (session / "geometry").mkdir(parents=True)
        mesh.export(session / "geometry" / "physical.glb")
        lib.write_json(session / "geometry" / "inspection.json", {"watertight": True, "components": components})
        out = inspect_session.summarize(session, solve=False)
        return out["geometry"], [w for w in out.get("warnings") or [] if "loose pieces" in w]

    sealed = drive.build_revolved(drive.revolve_outline("0,0 0.05,0 0.05,0.1 0,0.1", 0.008), "y")
    geometry, said = summary("sealed", sealed, 2)
    assert said == [] and geometry["solids"] == 1 and geometry["components"] == 2
    apart = trimesh.util.concatenate([trimesh.creation.box(extents=[0.1, 0.1, 0.1]),
                                      trimesh.creation.box(extents=[0.1, 0.1, 0.1]).apply_translation([0.5, 0, 0])])
    geometry, said = summary("apart", apart, 2)
    assert len(said) == 1 and "2 separate solid pieces" in said[0] and "solids" not in geometry
    # a closed spike whose sides nearly meet: the turn is named
    note = drive.revolve_fold_note(drive.revolve_outline("0,0 0.05,0 0.05,0.01 0.001,0.012 0.05,0.013 0.05,0.03 0,0.03"))
    assert "at (0.001, 0.012)" in note and "segment 2" in note and "4 degrees wide" in note


# ------------------------------------------------------- paint budget, part colours --

PARTS = {"mode": "parts", "description": "a stiff upper part and a soft lower part"}


def test_repaints_are_counted_after_the_first_painting_with_a_limit_of_two(tmp_path):
    """A held-out run (2026-10-10) painted one body 7 times (about 172 s) chasing colours by part."""
    plan = {**PLAN, "texture": {"texture_prompt": "light grey steel", "texture_seed": 0}}
    bare = {"kind": "material_input", "surface_has_colour": False, "textured_surface": None}
    answer, stage, stop = drive.plan_answer("material_input", bare, plan)
    assert answer["action"] == "texture_surface" and stage is None and stop == "texture"   # the first: free
    painted = {**bare, "textured_surface": "t.glb", "texture_prompt": "light grey steel", "texture_seed": 0}
    assert drive.plan_answer("material_input", painted, plan)[0].get("action") != "texture_surface"   # adopted
    again = {**plan, "texture": {"texture_prompt": "light grey steel", "texture_seed": 1}}
    answer, stage, _ = drive.plan_answer("material_input", painted, again)
    assert answer["action"] == "texture_surface" and stage == drive.REPAINTS
    assert drive.attempt_stage("material_input", answer, painted) == drive.REPAINTS        # sent by hand too
    assert drive.attempt_stage("material_input", answer, bare) is None
    (tmp_path / "operator").mkdir()
    for n in range(drive.RUN_LIMITS[drive.REPAINTS]):
        assert not drive.charge_attempt(tmp_path, drive.REPAINTS, {**answer, "texture_seed": n})["over"]
    assert drive.charge_attempt(tmp_path, drive.REPAINTS, {**answer, "texture_seed": 9})["over"]
    said = drive.limit_message(drive.REPAINTS, {})
    assert "per_part_colours" in said and "no colour by part" in said
    assert drive.attempts_left(tmp_path, {}, plan)[drive.REPAINTS] == 2
    assert drive.REPAINTS not in drive.attempts_left(tmp_path, {}, PLAN)            # nothing to paint


def test_per_part_colours_colour_the_parts_flat_without_the_painter():
    plan = {**PLAN, "material": PARTS, "texture": {"per_part_colours": True}}
    drive._check_texture(plan["texture"], True)
    drive._check_texture({"per_part_colours": {"1": "light brown", "0": "red"}}, True)
    for wrong, parts, said in (({"per_part_colours": True}, False, 'mode "parts"'),
                               ({"per_part_colours": True, "texture_prompt": "red"}, True, "beside"),
                               ({"per_part_colours": {"lower": "red"}}, True, "no part id"),
                               ({"per_part_colours": {"1": "red and green"}}, True, "one colour")):
        with pytest.raises(drive.PlanError, match=said):
            drive._check_texture(wrong, parts)
    bare = {"kind": "material_input", "surface_has_colour": False, "textured_surface": None, "material_parts": []}
    answer, stage, stop = drive.plan_answer("material_input", bare, plan)
    assert answer == {"action": "texture_surface", "per_part_colours": True} and stage is None and stop == "texture"
    coloured = {**bare, "textured_surface": "t.glb",
                "texture_fallback": {"kind": "part_colours", "asked": True}}
    answer, stage, _ = drive.plan_answer("material_input", coloured, plan)
    assert answer["action"] == "infer_parts"                                   # adopted: the parts are valued
    # no painter: neither warmed nor painted ahead
    assert not drive.plan_paints(plan) and drive.plan_paints({**plan, "texture": {"texture_prompt": "red"}})
    # a flat colouring is not counted, even after a painting
    painted = {**bare, "textured_surface": "t.glb", "texture_prompt": "red rubber"}
    answer, stage, _ = drive.plan_answer("material_input", painted, plan)
    assert answer["per_part_colours"] is True and stage is None
    assert drive.attempt_stage("material_input", answer, painted) is None
