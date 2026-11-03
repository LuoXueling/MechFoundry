"""The route rules of the review of the batch of 2026-10-11: a mass failure ends the ladder and is not
accepted while a route remains; a thin real member permits a built body first, never forces it."""

from __future__ import annotations

from pathlib import Path

import pytest

import drive
import sessionlib as lib


def _operator(session: Path) -> Path:
    (session / "operator").mkdir(parents=True, exist_ok=True)
    return session / "operator"


def _generated(monkeypatch, session: Path, source: str = "triposg", digest: str = "abc") -> None:
    monkeypatch.setattr(drive, "body_provenance", lambda s: source)
    monkeypatch.setattr(drive, "body_digest", lambda s: digest)
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)


@pytest.mark.web_rule
def test_a_heavy_generated_body_is_refused_while_the_drawing_is_untried(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    _generated(monkeypatch, tmp_path)
    lib.write_json(operator / "geometry_routes.json", ["triposg_text", "drawing", "hunyuan3d", "web_image", "built"])
    lib.write_json(operator / "run_counts.json", {"generation": 1})
    lib.write_json(operator / drive.ROUTE_SENT, ["textbody"])
    plan = {"geometry": {"description": "a body"}}
    said = drive.mass_route_problem(tmp_path, plan, {"mass_over_real": 11.5})
    assert said and "try the next route" in said and "SDXL drawing" in said
    # within the rule, or a built body: nothing to refuse
    assert drive.mass_route_problem(tmp_path, plan, {"mass_over_real": 4.0}) is None
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    assert drive.mass_route_problem(tmp_path, plan, {"mass_over_real": 11.5}) is None


@pytest.mark.web_rule
def test_a_mass_failure_skips_the_ladder_for_its_image(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    _generated(monkeypatch, tmp_path)
    lib.write_json(operator / "geometry_routes.json", ["triposg_text", "drawing", "hunyuan3d", "web_image", "built"])
    lib.write_json(operator / "run_counts.json", {"generation": 1, "drawing": 1})
    lib.write_json(operator / drive.ROUTE_SENT, ["textbody", "drawbody"])
    plan = {"geometry": {"description": "a body, two parts", "draw": True, "image_seed": 0}}
    drive.record_mass_failure(tmp_path, plan, 9.8)
    key = drive.plan_image_key(plan)
    assert key in drive.mass_failed_images(tmp_path)
    # the ladder's hint is the mass rule, naming Hunyuan3D while it is untried
    hint = drive.refine_hint(tmp_path, plan)
    assert hint.startswith(drive.MASS_NO_LADDER) and "Hunyuan3D" in hint
    refused = drive.refinement_problem(tmp_path, {"use_drawn_image": True}, plan, f"refine|{key}|x")
    assert refused and "no resolution problem" in refused
    # the acceptance names Hunyuan3D first, then the web photograph once Hunyuan3D was tried
    assert "Hunyuan3D" in drive.mass_route_problem(tmp_path, plan, {"mass_over_real": 9.8})
    lib.write_json(operator / "run_counts.json", {"generation": 1, "drawing": 1, "hunyuan": 1})
    assert "web photograph" in drive.mass_route_problem(tmp_path, plan, {"mass_over_real": 9.8})
    # past the photograph only `built` is left: the body is accepted, its verdict G4's
    monkeypatch.setattr(drive, "web_photo_problem", lambda s: None)
    assert drive.mass_route_problem(tmp_path, plan, {"mass_over_real": 9.8}) is None
    # a rescale brings the same image's body under 5 x: its ladder opens again
    drive.record_mass_failure(tmp_path, plan, 3.2)
    assert key not in drive.mass_blocked_images(tmp_path)
    assert not drive.refine_hint(tmp_path, plan).startswith(drive.MASS_NO_LADDER)
    drive.record_mass_failure(tmp_path, plan, 7.0)
    assert key in drive.mass_blocked_images(tmp_path)


def test_geometry_md_says_the_ladder_is_not_for_mass():
    text = (drive.SKILL_DIR / "references" / "geometry.md").read_text(encoding="utf-8")
    assert "never\n  mass" in text or "never mass" in text


def test_a_thin_real_member_permits_built_first_but_never_forces_it(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    (operator / "brief.md").write_text("## Size and material\nReal thinnest member: 0.004 m\n"
                                       "Real extents: 1.8 x 1.8 x 0.6 m\n")
    # the generated routes stay first and every generated step is sent (round 6 forced
    # [web_image, built] here and turned away 4 generated bodies of 10)
    routes = ["triposg_text", "drawing", "hunyuan3d", "built"]
    assert drive.canonical_routes(routes, tmp_path) == ["triposg_text", "drawing", "hunyuan3d", "web_image", "built"]
    assert drive.route_order_problem(tmp_path, "geometry_input", {"action": "draw_image", "description": "x"}) is None
    assert drive.route_order_problem(tmp_path, "geometry_input", {"generator": "cadcoder", "revise_prompt": True}) is None
    assert not drive.built_first_ok(tmp_path)                        # the line is what permits built first
    assert drive.routes_problem(["web_image", "built"], tmp_path)
    (operator / "brief.md").write_text("## Size and material\nReal thinnest member: 0.004 m\n"
                                       "Real extents: 1.8 x 1.8 x 0.6 m\nWhy not generated: a 4 mm wall\n")
    assert drive.built_first_ok(tmp_path) and drive.routes_problem(["web_image", "built"], tmp_path) is None
    assert drive.canonical_routes(["built", "drawing"], tmp_path) == ["web_image", "built", "triposg_text", "drawing"]


def test_a_thin_built_member_is_told_its_stand_in_and_planned_at_depth_8(tmp_path):
    operator = _operator(tmp_path)
    (operator / "brief.md").write_text("## Size and material\nReal thinnest member: 0.02 m\n"
                                       "Real extents: 1.8 x 1.8 x 0.6 m\n")
    # 0.02 m under a 1.8 m extent: 0.74 leaves at depth 7, 1.5 at depth 8; 2 leaves at depth 8 is
    # 0.027 m, 1.35 x the real member: within the 3 x rule, so the solve is planned at depth 8
    advice = drive.thin_build_advice(tmp_path, {}, 0.02, 1.8, 7)
    assert advice["solve_depth"] == 8 and advice["stand_in_at_depth_8_m"] == 0.027
    assert "within the 3 x G4 rule" in advice["said"] and "plans the solve at depth 8" in advice["said"]
    # depth 8 alone leaves the member as built under 2 leaves: the advice says so
    assert "as built is still 1.5 leaves, under 2" in advice["said"]
    built = tmp_path / "operator" / "blank_box_0a1b2c3d.glb"
    drive.record_built_depth(tmp_path, built, advice)
    plan = {"geometry": {"asset_path": str(built), "asset_kind": "surface"}, "scale": {"physical_extent": 1.8}}
    assert drive.built_needs_depth(tmp_path, plan)
    plan_path = operator / "plan.json"
    plan_path.write_text('{"geometry": {}, "settings": {}}')
    planned, note = drive.prescribed_depth(tmp_path, plan_path, plan)
    assert planned["settings"]["remesh_octree_depth"] == 8 and "built body" in note
    # a real member no stand-in within the rule resolves: the warning says the solve may be singular
    (operator / "brief.md").write_text("## Size and material\nReal thinnest member: 0.003 m\n"
                                       "Real extents: 1.8 x 1.8 x 0.6 m\n")
    far = drive.thin_build_advice(tmp_path, {}, 0.003, 1.8, 7)
    assert far["solve_depth"] == 7 and "may be singular" in far["said"]
    # a member that resolves at the planned depth: nothing to say
    assert drive.thin_build_advice(tmp_path, {}, 0.06, 1.8, 7) is None


def test_a_uniform_material_is_the_governing_members_own(tmp_path):
    operator = _operator(tmp_path)
    brief = ("## Size and material\nReal section (part b): round 0.012 m\nReal extents: 0.3 x 0.2 x 0.2 m\n"
             "Material (decided, uniform): ABS, E 2.2e9 Pa, nu 0.38, rho 1100 kg/m3 (a solid plastic body stands "
             "in for the plastic part a, steel part b and foam part c)\n")
    (operator / "brief.md").write_text(brief)
    plastic = {"material": {"youngs_modulus": 2.2e9, "poisson_ratio": 0.38, "density": 1100}}
    # a material word beside the member in the bulk's line, no E stated for it: uncertain, a warning
    assert drive.governing_material_problem(tmp_path, plastic) is None
    warned = drive.governing_material_check(tmp_path, plastic)[1]
    assert warned and "'part b'" in warned and "steel" in warned
    # the member's own line: certain
    governing = brief + "Governing material: steel, the part b tube\n"
    (operator / "brief.md").write_text(governing)
    said = drive.governing_material_problem(tmp_path, plastic)
    assert said and "'part b'" in said and "steel" in said and '"mode": "parts"' in said
    steel = {"material": {"youngs_modulus": 2.0e11, "poisson_ratio": 0.29, "density": 7850}}
    assert drive.governing_material_problem(tmp_path, steel) is None
    assert drive.governing_material_problem(tmp_path, {"material": {"mode": "parts"}}) is None
    # the E the brief states for the member is what is compared, not the band
    (operator / "brief.md").write_text(brief + "Governing material: steel, E 1.0e11 Pa\n")
    assert drive.governing_material_problem(tmp_path, {"material": {"youngs_modulus": 1.0e11}}) is None
    assert "E 1e+11 Pa as the brief states" in drive.governing_material_problem(tmp_path, steel)
    # phrases, not words: "carbon" alone is no carbon fibre; a filled polymer is a polymer
    assert list(drive.material_phrases("carbon steel")) == ["steel"]
    assert "carbon fibre" in drive.material_phrases("carbon-fibre tube")
    (operator / "brief.md").write_text(brief + "Governing material: glass-filled nylon\n")
    assert drive.governing_material_problem(tmp_path, {"material": {"youngs_modulus": 6e9}}) is None
    # a support's or payload's material in another clause is not read
    (operator / "brief.md").write_text(brief.replace("steel part b", "a part b") +
                                       "Load: a steel weight rests on part b\n")
    assert drive.governing_material_check(tmp_path, plastic) == (None, None)


ENTRY_BRIEF = """## Use
A person loads the object in use.
Use table: {"actions": {"A1": {"load": "top load", "state": "working", "kind": "weight", "source": "80 kg x g x 1", "direction": [0, -1, 0], "enters": "part b", "held": ["base"]}}, "run": "working"}
## Contacts
the base rests on the ground
{"base": {"transmits": "force and moment", "kind": "fastening", "member": "part a"}}
## Load
785 N down on part b
## Size and material
Real mass: 30 kg
Real thinnest member: 0.05 m
Real section (part a): rectangle 0.1 x 0.05 m
Real extents: 1 x 1 x 1 m
Load path: {"part b": "series", "part a": "series"}
"""


def test_the_member_a_load_enters_is_found_on_the_body_or_declared_absent(tmp_path):
    operator = _operator(tmp_path)
    (operator / "brief.md").write_text(ENTRY_BRIEF)
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part a; 1: part c"})
    regions = [{"name": "base_fix", "category": "dirichlet"}, {"name": "top_load", "category": "neumann"}]
    plan = {"loads": {"A1": {"regions": ["top_load"]}}, "regions": "Fix the base. Apply a traction on the top."}
    said = drive.entry_member_warning(tmp_path, plan, regions)
    assert said and "A1 enters 'part b'" in said and "absent" in said
    # named by the body's segments, or by the load's clause, it is found
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part a; 1: part b"})
    assert drive.entry_member_warning(tmp_path, plan, regions) is None
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part a; 1: part c"})
    assert drive.entry_member_warning(tmp_path, {**plan, "regions": "Apply a traction on part b."}, regions) is None
    # declared absent: nothing said, and the verdict's governing load is not applied as in use
    absent = {**plan, "body_simplification": {"absent": ["part b"]}}
    assert drive.entry_member_warning(tmp_path, absent, regions) is None
    assert drive.entry_absent_rows(tmp_path, absent) == {"A1": "part b"}


def test_the_entry_member_is_matched_spaced_compound_or_plural_and_by_its_load_regions_shape(tmp_path, monkeypatch):
    assert drive.entry_named("side member", "sidemember") and drive.entry_named("sidemembers", "the side member")
    assert drive.entry_named("end plate", "two endplates") and not drive.entry_named("rest", "forest")
    operator = _operator(tmp_path)
    (operator / "brief.md").write_text(ENTRY_BRIEF)
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part a; 1: partb"})
    plan = {"loads": {"A1": {"regions": ["top_load"]}}, "regions": "Fix the base. Apply a traction on the top."}
    regions = [{"name": "base_fix", "category": "dirichlet"},
               {"name": "top_load", "category": "neumann", "loaded_fraction": 0.95}]
    monkeypatch.setattr(drive, "load_region_shape", lambda s, names, *a: {n: {"segments": {1}, "patches": 1} for n in names})
    assert drive.entry_member_warning(tmp_path, plan, regions) is None
    # a label names it, but the load region is scattered: the member is likely absent (a warning)
    monkeypatch.setattr(drive, "load_region_shape", lambda s, names, *a: {n: {"segments": {1}, "patches": 3} for n in names})
    said = drive.entry_member_warning(tmp_path, plan, regions)
    assert said.startswith("member likely absent") and "3 patches" in said
    monkeypatch.setattr(drive, "load_region_shape", lambda s, names, *a: {n: {"segments": {1}, "patches": 1} for n in names})
    thin = [regions[0], {**regions[1], "loaded_fraction": 0.5}]
    assert "50% loaded" in drive.entry_member_warning(tmp_path, plan, thin)


def test_a_warned_member_may_be_answered_present_or_absent_and_is_never_refused(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    (operator / "brief.md").write_text(ENTRY_BRIEF)
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part a; 1: part c"})
    regions = [{"name": "base_fix", "category": "dirichlet"}, {"name": "top_load", "category": "neumann"}]
    plan = {"loads": {"A1": {"regions": ["top_load"]}}, "regions": "Fix the base. Apply a traction on the top."}
    said = drive.entry_member_warning(tmp_path, plan, regions)
    assert drive.ENTRY_FORM in said and "never a refusal" in said
    assert not hasattr(drive, "entry_member_problem")
    # answered: not warned again; "absent" is declared absent, "present" re-checked and listed
    absent = {**plan, "entry_members": {"part b": "absent"}}
    assert drive.entry_member_warning(tmp_path, absent, regions) is None
    assert drive.entry_absent_rows(tmp_path, absent) == {"A1": "part b"}
    (tmp_path / "boundary_conditions").mkdir(exist_ok=True)
    lib.write_json(tmp_path / "boundary_conditions" / "resolved.json", {"regions": regions})
    monkeypatch.setattr(drive, "load_region_shape", lambda s, names, *a: {n: {"segments": {1}, "patches": 2} for n in names})
    present = {**plan, "entry_members": {"part b": {"present": "top_load, the upper slab"}}}
    assert drive.entry_member_warning(tmp_path, present, regions) is None
    assert drive.entry_present_answers(tmp_path, present) == [
        "member part b answered present at 'top_load, the upper slab' (top_load: 2 patch(es), no segment names it)"]
    assert "unchecked" in drive.entry_present_answers(tmp_path, {"entry_members": {"part b": {"present": "on top"}}})[0]
    # a hold's member is read the same way: named by no segment or word, it is warned too
    held = {**present, "regions": "Fix the base. Load the top.",
            "contacts": {"base": {"transmits": "force and moment", "regions": ["base_fix"]}}}
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part c; 1: part d"})
    assert "hold base rests on 'part a'" in drive.entry_member_warning(tmp_path, held, regions)
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part a; 1: part d"})
    assert drive.entry_member_warning(tmp_path, held, regions) is None


def test_a_load_region_mostly_on_one_segment_is_one_patch(tmp_path):
    import numpy as np
    import trimesh
    (tmp_path / "geometry").mkdir()
    (tmp_path / "boundary_conditions").mkdir()
    mesh = trimesh.creation.box(extents=[1.0, 0.2, 0.5]).subdivide().subdivide()
    mesh.export(tmp_path / "geometry" / "physical.glb")
    c = mesh.triangles_center
    top = mesh.face_normals[:, 1] > 0.9
    split = top & (np.abs(c[:, 0]) > 0.2)                    # two patches on the top, apart
    np.savez(tmp_path / "boundary_conditions" / "resolved.npz", r0=split)
    lib.write_json(tmp_path / "boundary_conditions" / "resolved.json",
                   {"regions": [{"name": "load", "key": "r0", "category": "neumann"}]})
    np.savez(tmp_path / "boundary_conditions" / "segments.npz", segment_ids=np.where(top, 1, 0))
    lib.write_json(tmp_path / "session.json", {"segment_description": "0: part a; 1: part b"})
    assert drive.load_region_shape(tmp_path, ["load"], "part b")["load"]["patches"] == 1
    # the segment's label must name the member: another member's, or none asked, leaves the patches
    assert drive.load_region_shape(tmp_path, ["load"], "part c")["load"]["patches"] == 2
    assert drive.load_region_shape(tmp_path, ["load"])["load"]["patches"] == 2
    np.savez(tmp_path / "boundary_conditions" / "segments.npz", segment_ids=np.where(c[:, 0] > 0, 1, 0))
    assert drive.load_region_shape(tmp_path, ["load"], "part b")["load"]["patches"] == 2


def test_a_fine_extraction_is_capped_and_a_segmentation_out_of_memory_is_not_resent(tmp_path, monkeypatch):
    image = tmp_path / "picture.png"
    image.write_bytes(b"png")
    card = {"kind": "geometry_input"}
    fine = {"description": "a body", "asset_path": str(image), "asset_kind": "image",
            "generation_settings": {"init_octree_depth": 6, "octree_depth": 9}}
    prepared = drive.prepare_answer(card, {}, fine)
    assert not prepared.errors and prepared.answer["generation_settings"]["faces"] == drive.FINE_FACES
    coarse = drive.prepare_answer(card, {}, {**fine, "generation_settings": {"seed": 0}})
    assert "faces" not in coarse.answer["generation_settings"]
    # the segmentation of a body that ran out of GPU memory is refused when sent unchanged
    _operator(tmp_path)
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    monkeypatch.setattr(drive, "_generated_mesh", lambda s: None)
    answer = {"action": "preview_segmentation", "segmentation": {"method": "partfield", "num_clusters": 5}}
    assert drive.out_of_memory("torch.OutOfMemoryError: CUDA out of memory")
    drive.record_segmentation_oom(tmp_path, answer)
    assert "not sent again unchanged" in drive.segmentation_oom_problem(tmp_path, answer)
    assert drive.segmentation_oom_problem(tmp_path, {**answer, "segmentation": {"method": "partfield",
                                                                                "num_clusters": 3}}) is None
    monkeypatch.setattr(drive, "body_digest", lambda s: "body2")
    assert drive.segmentation_oom_problem(tmp_path, answer) is None


def test_regions_are_previewed_only_on_an_accepted_body(tmp_path, monkeypatch):
    _operator(tmp_path)
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    preview = {"action": "preview_regions", "description": "Fix the base."}
    assert "accepted" in drive.body_lock_problem(tmp_path, preview)
    assert drive.body_lock_problem(tmp_path, {"action": "approve_regions"}) is None
    drive.accept_body(tmp_path)
    assert drive.body_lock_problem(tmp_path, preview) is None
    monkeypatch.setattr(drive, "body_digest", lambda s: "body2")         # a new body: locked again
    assert drive.body_lock_problem(tmp_path, preview)


def test_small_fixes_of_the_review(tmp_path):
    # a view or pose phrase in the description is refused with the other words
    said = drive.description_problem({"description": "a steel body, two parts, seen from the side"})
    assert said and "seen from" in said
    assert drive.description_problem({"description": "a steel body, side view"})
    assert drive.description_problem({"description": "a steel body, two parts"}) is None
    # a view phrase that is part of the object's name in the user's instruction is kept
    assert drive.description_problem({"description": "a steel side view body"}, "Simulate a Side  View body") is None
    assert drive.description_problem({"description": "a steel side view body, not hollow"}, "a side view body")
    # only the exact phrase: one that shares its first words with the instruction's is refused
    assert drive.description_problem({"description": "a steel body seen from the side"},
                                     "a body seen from the shore")
    assert drive.description_problem({"description": "a steel body seen from the shore"},
                                     "a body seen from the shore") is None
    assert drive.description_problem({"description": "a steel side view body"}, "a side viewer body")
    # pieces at one station along the member are one contact split across it
    rows = [{"span_m": [0.40, 0.50]}, {"span_m": [0.40, 0.50]}]
    assert drive.held_stations(rows) == 1 and drive.held_stations([{"span_m": [0.0, 0.1]}, {"span_m": [2.0, 2.1]}]) == 2
    contacts = {"part a": {"transmits": "force only", "regions": ["r"]}}
    seen = [{"name": f"r#{k}", "region": "r", "contact": "part a", "face": "side face", "member_axis": "z",
             "span_m": [0.4, 0.5]} for k in (1, 2)]
    assert drive.pieces_problem(seen, contacts) is None
    apart = [{**seen[0]}, {**seen[1], "span_m": [2.0, 2.1]}]
    assert "the runner sees 2 held pieces" in drive.pieces_problem(apart, contacts)
    assert drive._plural("rear supports") and not drive._plural("sheet")
    # a plural contact's extra pieces are taken; on another member's segment they are warned, not refused
    plural = {"lower parts": {"transmits": "force only", "regions": ["r"]}}
    on = [{**p, "contact": "lower parts", "segment": "lower part"} for p in apart]
    assert drive.pieces_problem(on, plural) is None and drive.plural_pieces_warning(on, plural) is None
    off = [on[0], {**on[1], "segment": "upper plate"}]
    assert drive.pieces_problem(off, plural) is None
    assert "upper plate" in drive.plural_pieces_warning(off, plural)
    unknown = [{**p, "contact": "lower parts"} for p in apart]          # no segmentation: as before
    assert drive.pieces_problem(unknown, plural) is None and drive.plural_pieces_warning(unknown, plural) is None
    # a piece with no segment label is unknown and fails nothing
    partly = [on[0], {**on[1], "segment": None}]
    assert drive.on_contact_member("lower parts", partly) and drive.plural_pieces_warning(partly, plural) is None


def _rest_session(session: Path) -> None:
    """A slab held on its underside only: a rest that leaves the body free to slide and spin."""
    import numpy as np
    import trimesh
    _operator(session)
    (session / "geometry").mkdir(parents=True, exist_ok=True)
    (session / "boundary_conditions").mkdir(parents=True, exist_ok=True)
    mesh = trimesh.creation.box(extents=[1.0, 0.2, 0.5]).subdivide().subdivide()
    mesh.apply_translation(-mesh.bounds[0])
    mesh.export(session / "geometry" / "physical.glb")
    down = mesh.face_normals[:, 1] < -0.9
    np.savez(session / "boundary_conditions" / "resolved.npz", r0=down)
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"name": "underside", "key": "r0", "category": "dirichlet"}]})


def test_rests_that_leave_rigid_body_modes_free_get_the_fewest_components_added(tmp_path):
    _rest_session(tmp_path)
    plan = {"values": {"dirichlet_values": {"underside": {"components": [1], "value": 0.0}}}}
    found = drive.rigid_restraint(tmp_path, plan)
    assert set(found["free"]) == {"translation along x", "translation along z", "rotation about y"}
    assert found["added"] == {"underside": [0, 2]} and "still_free" not in found
    values = drive.with_restraint(plan["values"], found["added"])
    assert values["dirichlet_values"]["underside"]["components"] == [0, 1, 2]
    assert "rigid-body modes free" in drive.restraint_said(found)
    # held in every component (no entry, or all three): nothing is free
    assert drive.rigid_restraint(tmp_path, {"values": {}}) is None
    assert drive.rigid_restraint(tmp_path, {"values": {"dirichlet_values": {
        "underside": {"components": [0, 1, 2], "value": 0.0}}}}) is None
    # the verdict lists the runner's restraint
    lib.write_json(tmp_path / "operator" / drive.RESTRAINT_ADDED, found)
    listed = drive.runner_simplifications(tmp_path, {}, [])
    assert any(line.startswith("restraint added by the runner") for line in listed)


def test_restraint_is_refused_when_the_loads_would_slide_the_body(tmp_path):
    _rest_session(tmp_path)
    plan = {"values": {"dirichlet_values": {"underside": {"components": [1], "value": 0.0}}},
            "loads": {"A1": {"regions": ["top"]}}}
    regions = [{"name": "top", "category": "neumann", "centroid": [0.5, 0.2, 0.25]}]
    sideways = drive.rigid_restraint(tmp_path, plan, regions, [{"region": "top", "force_N": [100.0, -10.0, 0.0]}])
    assert sideways["unbalanced"]["row"] == "A1" and not sideways["added"]
    said = drive.sliding_said(sideways)
    assert said.startswith("the object slides under A1") and "over the 20%" in said
    assert "grip or socket or fastening" in said and "hold underside on every component" in said
    # pressed straight down on its rest: self-balanced along the free modes, the restraint is added
    down = drive.rigid_restraint(tmp_path, plan, regions, [{"region": "top", "force_N": [0.0, -100.0, 0.0]}])
    assert "unbalanced" not in down and down["added"] and down["carried_share"] < 1e-6
    lib.write_json(tmp_path / "operator" / drive.RESTRAINT_ADDED, down)
    assert any("by statics it carries" in line for line in drive.runner_simplifications(tmp_path, {}, []))


def test_a_small_push_along_the_rests_is_held_by_a_friction_stand_in(tmp_path):
    _rest_session(tmp_path)
    plan = {"values": {"dirichlet_values": {"underside": {"components": [1], "value": 0.0}}},
            "loads": {"A1": {"regions": ["top"]}}}
    regions = [{"name": "top", "category": "neumann", "centroid": [0.5, 0.2, 0.25]}]
    small = drive.rigid_restraint(tmp_path, plan, regions, [{"region": "top", "force_N": [10.0, -100.0, 0.0]}])
    assert "unbalanced" not in small and small["friction"] and small["added"] == {"underside": [0, 2]}
    assert "a friction stand-in" in drive.restraint_said(small)
    lib.write_json(tmp_path / "operator" / drive.RESTRAINT_ADDED, small)
    assert any("a friction stand-in" in line for line in drive.runner_simplifications(tmp_path, {}, []))


def test_a_push_no_rest_friction_resists_is_refused_and_takes_one_grip_on_the_body(tmp_path):
    import numpy as np
    import trimesh
    _rest_session(tmp_path)
    # a small rest under the middle, pressed down over it, twisted by a small couple about its normal:
    # the loaded mode spins the body about the rest, which barely moves its faces
    mesh = trimesh.load(tmp_path / "geometry" / "physical.glb", force="mesh", process=False).subdivide().subdivide()
    mesh.export(tmp_path / "geometry" / "physical.glb")
    c = mesh.triangles_center
    np.savez(tmp_path / "boundary_conditions" / "resolved.npz",
             r0=(mesh.face_normals[:, 1] < -0.9) & (np.abs(c[:, 0] - 0.5) < 0.04) & (np.abs(c[:, 2] - 0.25) < 0.04))
    plan = {"values": {"dirichlet_values": {"underside": {"components": [1], "value": 0.0}}},
            "loads": {"A1": {"regions": ["top", "left", "right"]}}}
    regions = [{"name": "top", "category": "neumann", "centroid": [0.5, 0.2, 0.25]},
               {"name": "left", "category": "neumann", "centroid": [0.1, 0.1, 0.25]},
               {"name": "right", "category": "neumann", "centroid": [0.9, 0.1, 0.25]}]
    planned = [{"region": "top", "force_N": [0.0, -100.0, 0.0]}, {"region": "left", "force_N": [0.0, 0.0, 5.0]},
               {"region": "right", "force_N": [0.0, 0.0, -5.0]}]
    found = drive.rigid_restraint(tmp_path, plan, regions, planned)
    assert found["unbalanced"]["friction_share"] <= drive.FRICTION_SHARE
    assert found["unbalanced"]["friction_arm"] < drive.FRICTION_ARM
    said = drive.sliding_problem(tmp_path, found, record=False)
    assert "no rest's friction resists it" in said
    # a push that lifts the body off its rests: no normal force, so no friction holds it
    lifted = drive.rigid_restraint(tmp_path, plan, regions, [{"region": "top", "force_N": [5.0, 100.0, 0.0]}])
    assert lifted["unbalanced"]["friction_share"] is None
    assert "not in compression" in drive.sliding_said(lifted)
    calls = []
    orig = drive.allow_contact_additions
    drive.allow_contact_additions = lambda op, kinds, cause="tipping": calls.append((tuple(kinds), cause))
    try:
        drive.sliding_problem(tmp_path, found, record=False)
    finally:
        drive.allow_contact_additions = orig
    assert calls == [(drive.ADD_FOR_SLIDING, "sliding")]
    # a fastening that only transmits force stops a slide, not a tip
    fastening = {"kind": "fastening", "transmits": "force only"}
    assert drive.addable(fastening, drive.ADD_FOR_SLIDING, "sliding")
    assert not drive.addable(fastening, drive.ADD_FOR_TIPPING, "tipping")


def test_the_real_extents_are_revised_once_at_the_geometry_stop_within_the_tolerance(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    brief = "## Size and material\nReal mass: 20 kg\nReal extents: 1.0 x 0.5 x 0.4 m\n"
    (operator / "brief.md").write_text(brief)
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    assert drive.real_extents(operator) == [1.0, 0.5, 0.4]
    body = [0.4, 1.0, 0.9]                       # the body's measured extents, in any order
    # off the body's own extents, or without a reason: not taken, and said why
    (operator / "brief.md").write_text(brief + "Real extents revised: 1.0 x 0.6 x 0.4 m, the upper part is taller\n")
    record, why = drive.extents_revision(tmp_path, take=True, body=body)
    assert record is None and "at most 1.1 x" in why
    (operator / "brief.md").write_text(brief + "Real extents revised: 1.0 x 0.9 x 0.4 m\n")
    assert "reason" in drive.extents_revision(tmp_path, take=True, body=body)[1]
    # a reader records nothing: only the refusal takes the line
    (operator / "brief.md").write_text(brief + "Real extents revised: 1.0 x 0.9 x 0.4 m, the first guess was low\n")
    assert drive.extents_revision(tmp_path) == (None, None) and drive.extents_revised_said(tmp_path) is None
    # at the body's extents (1.8 x the first guess: no cap against it), with a reason, on a body not
    # yet accepted: taken once, and the report marks it; every other reader keeps the first extents
    assert drive.extents_revision(tmp_path, take=True, body=body)[0]["extents"] == [1.0, 0.9, 0.4]
    assert drive.real_extents(operator) == [1.0, 0.5, 0.4]
    said = drive.extents_revised_said(tmp_path)
    assert said.startswith(drive.EXTENTS_TAKEN_SAID) and "1.0, 0.5, 0.4] -> [1.0, 0.9, 0.4]" in said
    (operator / "brief.md").write_text(brief + "Real extents revised: 1.0 x 0.7 x 0.4 m, and again, larger\n")
    assert "once already" in drive.extents_revision(tmp_path)[1]
    # an accepted body is past the geometry stop: nothing is taken there
    other = tmp_path / "other"
    _operator(other)
    (other / "operator" / "brief.md").write_text(brief + "Real extents revised: 1.0 x 0.6 x 0.4 m, the first guess was low\n")
    drive.accept_body(other)
    assert drive.extents_revision(other, take=True)[0] is None


def test_the_one_revision_may_carry_the_real_mass_when_it_is_the_outlier(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    brief = "## Size and material\nReal mass: 2 kg\nReal extents: 1.0 x 0.5 x 0.4 m\n"
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    near = {"planned_mass": 12.0}                # 6 x the Real mass, solid as generated
    assert drive.mass_outlier(tmp_path, near) is None
    (operator / "brief.md").write_text(brief)
    assert drive.mass_outlier(tmp_path, near) == 6.0
    assert "Real mass revised" in drive.revision_said(tmp_path, near)
    assert "Real mass revised" not in drive.revision_said(tmp_path, {"planned_mass": 3.0})
    (operator / "brief.md").write_text(brief + "Real mass revised: 30 kg, a catalogue sheet says so\n")
    assert "outside" in drive.extents_revision(tmp_path, take=True, resolution=near)[1]
    (operator / "brief.md").write_text(brief + "Real mass revised: 9 kg\n")
    assert "source" in drive.extents_revision(tmp_path, take=True, resolution=near)[1]
    (operator / "brief.md").write_text(brief + "Real mass revised: 9 kg, a catalogue sheet says so\n")
    assert "only when" in drive.extents_revision(tmp_path, take=True, resolution={"planned_mass": 3.0})[1]
    assert drive.extents_revision(tmp_path, take=True, resolution=near)[0]["mass"] == 9.0
    assert drive.real_mass(operator) == 9.0 and "2 -> 9 kg" in drive.extents_revised_said(tmp_path)
    # the line is not frozen with the Real lines
    assert drive._real_lines((operator / "brief.md").read_text()) == ["Real mass: 2 kg", "Real extents: 1.0 x 0.5 x 0.4 m"]


def test_only_a_generated_body_s_shown_extent_refusal_takes_the_revision(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    brief = "## Size and material\nReal mass: 20 kg\nReal extents: 1.0 x 0.5 x 0.4 m\n"
    (operator / "brief.md").write_text(brief + "Real extents revised: 1.6 x 0.5 x 0.4 m, the first guess was low\n")
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    monkeypatch.setattr(drive, "body_source", lambda s: tmp_path / "body.glb")
    route = {"route": "real"}
    monkeypatch.setattr(drive, "body_route", lambda *a, **k: dict(route))
    near, far = {"principal_extents": [1.6, 0.5, 0.4]}, {"principal_extents": [2.0, 0.5, 0.4]}
    # a built body: refused, no revision offered or taken
    said = drive.body_gate(tmp_path, {}, near, record=False)
    assert "largest" in said and "Real extents revised" not in said
    assert not (operator / drive.EXTENTS_REFUSED).exists()
    # a generated body (a web photograph's too): the first refusal shows the line and no re-generation,
    # the next takes it at the body's own extents
    route["route"] = "generated"
    said = drive.body_gate(tmp_path, {}, near, record=False)
    assert "Real extents revised" in said and drive.NO_REGENERATION in said
    assert drive.extents_revised_said(tmp_path) is None
    assert drive.body_gate(tmp_path, {}, near, record=False) is None
    assert "-> [1.6, 0.5, 0.4]" in drive.extents_revised_said(tmp_path)
    assert any(s.startswith(drive.EXTENTS_TAKEN_SAID) for s in drive.runner_simplifications(tmp_path, {}, []))
    # another body is then held to the revision within REVISED_FIT
    assert "largest" in drive.body_gate(tmp_path, {}, far, record=False)
    # the middle extent (a pose) is not judged
    assert drive.body_gate(tmp_path, {}, {"principal_extents": [1.6, 1.0, 0.4]}, record=False) is None
