"""The body's route derived from where it came from, the flat colouring of parts by material, and
the stand-in's cut chosen by the runner across its member.  No GPU, no API, no graph run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402
from test_operator_scripts import PLAN  # noqa: E402


# ------------------------------------------------------------------ route provenance --

def _session_body(session: Path, mode: str | None, source_name: str = "", generator: str = "triposg") -> None:
    """A session whose body is on disk: an upload (``mode`` surface, by its file name) or a generation,
    each with the record that carries its digest (the staged asset, the generation's receipt)."""
    import hashlib
    (session / "operator").mkdir(parents=True, exist_ok=True)
    (session / "geometry").mkdir(parents=True, exist_ok=True)
    body = session / "geometry" / ("original.glb" if mode == "surface" else "generated.glb")
    body.write_bytes(b"x")
    digest = hashlib.sha256(b"x").hexdigest()
    tool = "generate_hunyuan3d_image_surface" if generator == "hunyuan3d" else "generate_image_surface"
    lib.write_json(session / "session.json", {
        "geometry_mode": mode, "generation": {"generator": generator},
        "input_asset": {"source_name": source_name, "sha256": digest} if source_name else None,
        "tool_receipts": [] if mode == "surface" else [{"tool": tool, "artifact_digests": {str(body): digest}}]})


def test_the_route_is_where_the_body_came_from(tmp_path):
    # the plan's geometry entry, before the body exists
    assert drive.geometry_provenance({"asset_path": "x/operator/blank_revolve_0a1b2c3d.glb",
                                      "asset_kind": "surface"}) == "built"
    assert drive.geometry_provenance({"asset_path": "data/part.stl", "asset_kind": "surface"}) == "upload"
    assert drive.geometry_provenance({"description": "d", "generator": "cadcoder"}) == "cadcoder"
    assert drive.geometry_provenance({"description": "d", "draw": True}) == "triposg"
    assert drive.geometry_provenance({"description": "d", "generator": "hunyuan3d"}) == "hunyuan3d"
    # the session's body: a blank_* upload is built (the generation record left in surface mode is not read)
    built = tmp_path / "built"
    _session_body(built, "surface", "blank_revolve_0a1b2c3d.glb")
    assert drive.body_provenance(built) == "built"
    upload = tmp_path / "upload"
    _session_body(upload, "surface", "part.stl")
    assert drive.body_provenance(upload) == "upload"
    generated = tmp_path / "generated"
    _session_body(generated, "image", generator="hunyuan3d")
    assert drive.body_provenance(generated) == "hunyuan3d"
    assert drive.body_provenance(tmp_path / "none") is None
    # a new answer in flight names its generator before its body exists: the body on disk keeps its own
    state = lib.read_json(generated / "session.json")
    lib.write_json(generated / "session.json", {**state, "generation": {"generator": "cadcoder"}})
    assert drive.body_provenance(generated) == "hunyuan3d"
    # a body on disk no record carries (written since, not yet received): no provenance claimed
    (generated / "geometry" / "generated.glb").write_bytes(b"y")
    assert drive.body_provenance(generated) is None
    (upload / "geometry" / "original.glb").write_bytes(b"y")
    assert drive.body_provenance(upload) is None
    # the route follows from the source, never from a plan field: a blank body is the real one, a
    # generation "generated", a user's surface "upload"
    from unittest import mock
    with mock.patch.object(drive, "member_unresolvable", lambda s, p: False):
        assert drive.body_route(built, PLAN, {"member_over_real": 1.0, "mass_over_real": 1.0})["route"] == "real"
        assert drive.body_route(generated, PLAN, {})["route"] is None          # its body changed since
        _session_body(generated, "image")
        assert drive.body_route(generated, PLAN, {})["route"] == "generated"
        pending = {**PLAN, "geometry": {"asset_path": "o/blank_bars_0a1b2c3d.glb", "asset_kind": "surface"}}
        assert drive.body_route(tmp_path / "none", pending, pending=True)["route"] == "real"


def test_a_built_body_after_the_listed_routes_fell_short_is_route_built(tmp_path, monkeypatch):
    (tmp_path / "operator").mkdir()
    monkeypatch.setattr(drive, "member_unresolvable", lambda s, p: False)
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["drawing"])
    plan = {**PLAN, "geometry": {"asset_path": "o/blank_bars_0a1b2c3d.glb", "asset_kind": "surface"},
            "body_simplification": {"kept": "k", "left_out": "l"}}
    said = drive.body_simplification_problem(tmp_path, plan)
    assert said and "['drawing'] untried" in said
    lib.write_json(tmp_path / "operator" / "run_counts.json", {"drawing": 3})
    assert drive.body_simplification_problem(tmp_path, plan) is None
    # "built" listed first needs the brief's argument, not the list alone; a user's surface is never held to it
    lib.write_json(tmp_path / "operator" / "run_counts.json", {})
    lib.write_json(tmp_path / "operator" / "geometry_routes.json", ["built", "drawing"])
    assert drive.body_simplification_problem(tmp_path, plan)
    upload = {**plan, "geometry": {"asset_path": "data/part.stl", "asset_kind": "surface"}}
    assert drive.body_simplification_problem(tmp_path, upload) is None
    assert "the simplified body" in drive.route_next("drawing", ["drawing"])


# ----------------------------------------------------------- part colours by material --

RUBBER = {"youngs_modulus": 5e6, "poisson_ratio": 0.48, "density": 1100}
WOOD = {"youngs_modulus": 1e10, "poisson_ratio": 0.35, "density": 550}
STEEL = {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850}


def test_parts_of_one_material_share_its_colour(tmp_path):
    """A held-out run (2026-10-10): true took the montage's debug colours, one handle of one
    material in two colours; two runs spent a turn naming every part's colour by hand."""
    parts = [{"id": 0, **RUBBER}, {"id": 1, **WOOD}, {"id": 2, **RUBBER}, {"id": 3, **WOOD}, {"id": 4, **STEEL}]
    colours = drive.part_material_colours(parts, "wooden handle, red rubber cup, steel ferrule")
    assert colours["1"] == colours["3"] == "wooden" and colours["0"] == colours["2"] == "red"
    assert colours["4"] == "steel"
    # without the description's words: the table's colour for the values
    bare = drive.part_material_colours(parts)
    assert bare["1"] == "wood" and bare["0"] == "rubber" and bare["4"] == "steel"
    # a plastic in its words' colour; one with none, and a material the table does not know, in the palette
    plastic = {"youngs_modulus": 2.4e9, "poisson_ratio": 0.37, "density": 1200}
    odd = {"youngs_modulus": 3e12, "poisson_ratio": 0.2, "density": 20000}
    assert drive.part_material_colours([{"id": 0, **plastic}], "light green plastic shell") == {"0": "light green"}
    plain = drive.part_material_colours([{"id": 0, **plastic}, {"id": 1, **odd}], "a plastic shell")
    assert plain["0"] in drive.PART_PALETTE and plain["1"] in drive.PART_PALETTE and plain["0"] != plain["1"]
    # two materials of one kind (other values) are told apart by a shade
    soft = {**RUBBER, "youngs_modulus": 2e6}
    two = drive.part_material_colours([{"id": 0, **RUBBER}, {"id": 1, **soft}])
    assert two["0"] != two["1"] and two["1"].endswith("rubber")
    # densities keep the families apart where the moduli overlap; "metal" and "alloy" name metals
    one = lambda E, rho, words="": drive.part_material_colours(
        [{"id": 0, "youngs_modulus": E, "density": rho, "poisson_ratio": 0.3}], words)["0"]
    assert one(6.9e10, 2700) == one(7.2e10, 2810, "7075 alloy arms") == "aluminium"
    assert one(6.9e10, 2700, "a metal frame") == "aluminium"
    assert one(7e10, 2500, "a glass pane") == "glass" and one(7e10, 2500) == "light cyan"
    assert one(4.5e10, 1800) == "silver" and one(7e10, 1600, "carbon-fibre arms") == "charcoal"
    # one material whose valued values scatter by a few per cent is one colour
    steels = drive.part_material_colours([{"id": 0, **STEEL}, {"id": 1, **STEEL, "youngs_modulus": 2.1e11}],
                                         "steel frame, steel axle")
    assert steels == {"0": "steel", "1": "steel"}
    # material words the colour reader does not know take their family's colour, never mid grey
    assert one(1.1e10, 500, "timber beam") == one(1e10, 600, "plywood shelf") == "wood"
    assert one(1.95e11, 7900, "stainless shaft") == "steel" and one(2e6, 1100, "silicone pad") == "rubber"
    assert one(2.5e9, 1150, "nylon gear") in drive.PART_PALETTE
    # the colours are read by the workflow's own reader as the material's
    from src.hunyuan3d_wrapper.surface import plain_colour
    assert plain_colour(colours["1"])[1] == "wooden" and plain_colour(bare["0"])[1] == "rubber"


def test_true_is_sent_as_the_material_colours_once_the_parts_are_valued():
    plan = {**PLAN, "material": {"mode": "parts", "description": "wooden handle, rubber cup", "parts": []},
            "texture": {"per_part_colours": True}}
    assert drive.resolved_part_colours(plan, {}) is True                 # no values yet: the montage's
    card = {"material_parts": [{"id": 0, **RUBBER}, {"id": 1, **WOOD}, {"id": 2, **RUBBER}],
            "surface_has_colour": False, "textured_surface": "t.glb",
            "texture_fallback": {"kind": "part_colours", "asked": True}}
    resolved = drive.resolved_part_colours(plan, card)
    assert resolved == {"0": "rubber", "1": "wooden", "2": "rubber"}
    # coloured before the values: coloured again, by material, and the stop is the texture's
    assert not drive.texture_matches(card, plan["texture"], plan)
    answer, stage, stop = drive.plan_answer("material_input", card, plan, None)
    assert answer == {"action": "texture_surface", "per_part_colours": resolved} and stage is None and stop == "texture"
    assert drive.texture_matches({**card, "texture_fallback": {"kind": "part_colours", "asked": resolved}},
                                 plan["texture"], plan)
    # a correction in the plan moves a part to the other material
    corrected = {**plan, "material": {**plan["material"], "parts": [{"id": 2, **WOOD}]}}
    assert drive.resolved_part_colours(corrected, card)["2"] == "wooden"
    # a named mapping is sent as written
    named = {**plan, "texture": {"per_part_colours": {"0": "red"}}}
    assert drive.resolved_part_colours(named, card) == {"0": "red"}


# ------------------------------------------------------------- the cut across the member --

def _cut_session(tmp_path: Path, parts, regions, monkeypatch) -> tuple[Path, dict]:
    """A body of boxes (each (extents, centre)) at its low corner, with its resolved regions."""
    import trimesh
    from test_operator_scripts import THIN_BRIEF
    session = tmp_path
    (session / "operator").mkdir(parents=True, exist_ok=True)
    (session / "geometry").mkdir(exist_ok=True)
    (session / "operator" / "brief.md").write_text(THIN_BRIEF.replace("MEMBER", "0.002"))
    mesh = trimesh.util.concatenate([trimesh.creation.box(extents=e).apply_translation(c) for e, c in parts])
    mesh.apply_translation(-mesh.bounds[0])
    mesh.export(session / "geometry" / "generated.glb")
    mesh.export(session / "geometry" / "physical.glb")
    if regions:
        (session / "boundary_conditions").mkdir(exist_ok=True)
        lib.write_json(session / "boundary_conditions" / "resolved.json", {"regions": regions})
    resolution = {"physical_extents": [float(e) for e in mesh.extents]}
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: resolution)
    return session, resolution


def _cut(session, resolution, load=(0.0, -20.0, 0.0)):
    plan = {**PLAN, "values": {**PLAN["values"], "load": {"force": list(load)}},
            "body_simplification": {"kept": "k", "left_out": "l"}}
    return drive.section_equivalence(session, plan, resolution)


# a plate on four posts
POSTS = [([1.0, 0.03, 0.6], [0.5, 0.715, 0.3])] + [([0.04, 0.7, 0.04], [x, 0.35, z])
                                                   for x in (0.02, 0.98) for z in (0.02, 0.58)]
POSTS_REGIONS = [{"name": f"foot{i}", "category": "dirichlet", "bounds": [[x - 0.02, 0, z - 0.02], [x + 0.02, 0, z + 0.02]]}
                 for i, (x, z) in enumerate([(0.02, 0.02), (0.98, 0.02), (0.02, 0.58), (0.98, 0.58)])] + [
    {"name": "top_load", "category": "neumann", "bounds": [[0.4, 0.73, 0.2], [0.6, 0.73, 0.4]],
     "centroid": [0.5, 0.73, 0.3]}]


def test_the_cut_is_taken_on_the_member_the_load_lies_on(tmp_path, monkeypatch):
    """The cut is the runner's alone: across the member the load region lies on (a first version
    compared three planes through a point between the load and a support, and on a body of several
    members that point lay beside another member), bent about the axis the load is not along."""
    session, resolution = _cut_session(tmp_path / "posts", POSTS, POSTS_REGIONS, monkeypatch)
    chosen = _cut(session, resolution)
    assert chosen["cut_across"] == "x" and "'top_load'" in chosen["cut_across_chosen"]
    assert abs(chosen["stand_in"]["A_m2"] - 0.6 * 0.03) < 1e-3          # the plate's section, not a post's
    assert chosen["about"] == "z"                                       # the load down y bends it about z
    # an L-shaped part loaded at its arm's tip: the arm is chosen
    bracket = [([0.06, 0.4, 0.06], [0.03, 0.2, 0.03]), ([0.4, 0.01, 0.04], [0.26, 0.395, 0.03])]
    regions = [{"name": "base", "category": "dirichlet", "bounds": [[0, 0, 0], [0.06, 0, 0.06]]},
               {"name": "tip", "category": "neumann", "bounds": [[0.42, 0.4, 0.01], [0.46, 0.4, 0.05]],
                "centroid": [0.44, 0.4, 0.03]}]
    session, resolution = _cut_session(tmp_path / "bracket", bracket, regions, monkeypatch)
    auto = _cut(session, resolution)
    assert auto.get("cut_across") == "x", auto
    # a flat frame (two rails along x, two crossbars along z) loaded on a crossbar: cut across the crossbar
    frame = [([1.0, 0.04, 0.04], [0.5, 0.02, z]) for z in (0.02, 0.48)] + [
        ([0.04, 0.04, 0.5], [x, 0.02, 0.25]) for x in (0.3, 0.7)]
    regions = [{"name": f"end{i}", "category": "dirichlet", "bounds": [[x, 0, 0], [x + 0.04, 0, 0.5]]}
               for i, x in enumerate((0.0, 0.96))] + [
        {"name": "bar_load", "category": "neumann", "bounds": [[0.28, 0.04, 0.2], [0.32, 0.04, 0.3]],
         "centroid": [0.3, 0.04, 0.25]}]
    session, resolution = _cut_session(tmp_path / "frame", frame, regions, monkeypatch)
    auto = _cut(session, resolution)
    assert auto.get("cut_across") == "z" and auto["about"] == "x", auto


def test_single_plates_and_blocks_keep_their_cuts(tmp_path, monkeypatch):
    plate = [([1.0, 0.02, 0.5], [0.5, 0.01, 0.25])]
    regions = [{"name": "left", "category": "dirichlet", "bounds": [[0, 0, 0], [0.05, 0, 0.5]]},
               {"name": "right", "category": "dirichlet", "bounds": [[0.95, 0, 0], [1.0, 0, 0.5]]},
               {"name": "mid", "category": "neumann", "bounds": [[0.45, 0.02, 0.2], [0.55, 0.02, 0.3]],
                "centroid": [0.5, 0.02, 0.25]}]
    session, resolution = _cut_session(tmp_path / "plate", plate, regions, monkeypatch)
    chosen = _cut(session, resolution)
    assert chosen["cut_across"] == "x" and "the long axis of the member" in chosen["cut_across_chosen"]
    # before any regions: nothing is chosen (the sections' own axis), nothing printed as chosen
    bare, res = _cut_session(tmp_path / "bare", plate, None, monkeypatch)
    assert drive.member_cut_axis(bare, PLAN, res) is None and "cut_across_chosen" not in _cut(bare, res)
    # a stocky block: a cut, whatever the load's direction
    block, cube = _cut_session(tmp_path / "block", [([0.3, 0.25, 0.28], [0.15, 0.125, 0.14])], None, monkeypatch)
    for load in ((0, -20.0, 0), (20.0, 0, 0), (0, 0, 20.0)):
        assert "error" not in _cut(block, cube, load), load


def test_routes_say_the_thin_member_rule_where_they_are_read_at_the_start():
    routes = (SCRIPTS.parent / "references" / "objectives" / "routes.md").read_text(encoding="utf-8")
    gate = " ".join(routes[routes.index("**The body (G4).**"):routes.index("**A simplified body**")].split())
    assert "under 2 leaves" in gate and "never refused" in gate and "`Real extents`" in gate
    assert "resolution caveat" in gate and "4.2" not in routes
