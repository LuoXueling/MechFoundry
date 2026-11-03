"""Per-part materials from the brief (batch of 2026-10-12: all ten runs uniform, several bodies of
visibly different materials): the brief's 'Materials by part:' line, the uniform refusal it drives,
the runner's fill of each valued segment from it, and the per-part material line of the report."""

from __future__ import annotations

import pytest

import drive
import sessionlib as lib
from test_use_table import SIZE, session_with, table

PARTS = ("Governing material: steel, E 200 GPa\n"
         "Materials by part: top plate = steel, E 200 GPa; side grip = ABS plastic; bottom plate rubber\n")
UNIFORM = {"material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850}}
LABELS = "0: top plate upper face. 1: side grip and its edge. 2: bottom plate underside. 3: small boss"


def _session(tmp_path, size=SIZE + PARTS, name="abc", labels=LABELS):
    session = session_with(tmp_path, table(), size=size, name=name)
    lib.write_json(session / "session.json", {"segment_description": labels, "segmentation_method": "partfield"})
    return session


def _card(**extra):
    parts = [{"id": i, "youngs_modulus": 1e9, "poisson_ratio": 0.3, "density": 1000.0} for i in range(4)]
    return {"kind": "material_input", "surface_has_colour": True, "textured_surface": "t.glb",
            "material_description": "what each part is made of", "material_parts": parts, **extra}


def test_the_brief_lists_its_parts_and_their_materials(tmp_path):
    entries = drive.brief_part_materials(_session(tmp_path))
    assert [e["part"] for e in entries] == ["top plate", "side grip", "bottom plate"]
    assert entries[0]["E"] == 2e11 and entries[1]["material"] == "abs" and entries[2]["material"] == "rubber"
    assert 1.5e9 <= entries[1]["E"] <= 3.9e9 and 1050 <= entries[1]["rho"] <= 1250    # the row's middle
    stated = drive.part_material_entry("top plate steel, E = 70 GPa, nu 0.33, rho 2700 kg/m3")
    assert (stated["part"], stated["E"], stated["nu"], stated["rho"]) == ("top plate", 7e10, 0.33, 2700.0)
    assert drive.governing_entry(_session(tmp_path, name="g"))["E"] == 2e11


def test_a_uniform_material_across_parts_apart_in_e_is_refused_at_the_plan_check(tmp_path):
    session = _session(tmp_path)
    said = drive.uniform_materials_problem(session, UNIFORM)
    assert "Materials by part" in said and '"mode": "parts"' in said and "uniform_why" in said
    why = {"material": {**UNIFORM["material"], "uniform_why": "a shape optimization takes one material"}}
    assert drive.uniform_materials_problem(session, why) is None
    assert drive.uniform_materials_problem(session, {"material": {"mode": "parts", "description": "x"}}) is None
    one = _session(tmp_path, size=SIZE + "Materials by part: top plate = steel; side grip = steel\n", name="one")
    assert drive.uniform_materials_problem(one, UNIFORM) is None
    near = _session(tmp_path, size=SIZE + "Materials by part: top plate = steel; side grip = titanium\n", name="near")
    assert drive.uniform_materials_problem(near, UNIFORM) is None                 # within 2 x


def test_the_runner_fills_each_segment_from_the_brief_and_stops_once(tmp_path):
    session = _session(tmp_path)
    plan = {"scale": {"unit_system": "SI (m-kg-s)"},
            "material": {"mode": "parts", "description": "what each part is made of"}}
    fills, record = drive.brief_part_fills(session, plan, _card())
    by_id = {f["id"]: f for f in fills}
    assert by_id[0]["youngs_modulus"] == 2e11 and record["segments"]["0"]["part"] == "top plate"
    assert record["segments"]["1"]["material"] == "abs"
    assert by_id[3]["youngs_modulus"] == 2e11 and record["unmatched"] == [3]      # the Governing material
    assert by_id[2]["poisson_ratio"] == drive.NU_STAND_IN and by_id[2]["youngs_modulus"] == 2e9  # stand-ins
    assert any("contrast" in s for s in record["stand_ins"])
    # the valuation runs on; the brief's values are applied, then the one stop; then approved
    bare = _card(material_parts=None)
    filled = drive.with_brief_parts(session, plan, bare)
    assert drive.plan_answer("material_input", bare, filled)[1:] == ("materials", None)
    valued = _card(material_inferred=True)
    filled = drive.with_brief_parts(session, plan, valued)
    answer, stage, stop = drive.plan_answer("material_input", valued, filled)
    assert answer["action"] == "apply_parts" and stop == "materials" and len(answer["parts"]) == 4
    applied = _card(material_parts=[{**p, "density": p.get("density", 1000.0)} for p in
                                    drive.with_brief_parts(session, plan, valued)["material"]["parts"]],
                    material_inferred=False)
    assert drive.plan_answer("material_input", applied, drive.with_brief_parts(session, plan, applied))[0] == \
        {"material_mode": "parts"}
    # the plan's own correction by id wins over the brief's value
    own = {**plan, "material": {**plan["material"], "parts": [{"id": 1, "youngs_modulus": 3e9}]}}
    assert {p["id"]: p for p in drive.with_brief_parts(session, own, valued)["material"]["parts"]}[1][
        "youngs_modulus"] == 3e9
    said = drive.per_part_said(session)
    assert said.startswith("per-part: top plate steel, side grip abs, bottom plate rubber")
    assert "segments 3 unmatched, the Governing material" in said
    run = {"material_mode": "parts", "run": "r1", "body": {"sha": "b"}, "ends": "e", "load_N": 1.0}
    section = drive.history_section([run], [], {"per_part": said})
    assert f"- material: {said}" in section


def test_without_labels_the_valuation_stands_and_without_the_line_nothing_changes(tmp_path):
    session = _session(tmp_path, labels="")
    plan = {"material": {"mode": "parts", "description": "x"}}
    fills, record = drive.brief_part_fills(session, plan, _card())
    assert not fills and record["unmatched"] == [0, 1, 2, 3]
    drive.with_brief_parts(session, plan, _card())
    assert "the valuation's values" in drive.per_part_said(session)
    plain = _session(tmp_path, size=SIZE, name="plain")
    assert drive.with_brief_parts(plain, plan, _card()) is plan


@pytest.mark.parametrize("label, expected", [("side grip edge", "side grip"), ("upper plate", None)])
def test_a_segment_matches_the_part_its_label_names(label, expected):
    entries = [{"part": "top plate", "E": 2e11}, {"part": "side grip", "E": 2e9}, {"part": "bottom plate", "E": 1e8}]
    found = drive.match_part(label, entries)
    assert (found or {}).get("part") == expected


def test_g4_prices_the_body_at_the_parts_densities_not_the_governing_one(tmp_path, monkeypatch):
    import trimesh
    session = _session(tmp_path)
    entries = drive.brief_part_materials(session)
    plan = {"scale": {"unit_system": "SI (m-kg-s)", "up_axis": "y", "physical_extent": 0.1}, **UNIFORM}
    # no segments yet: the listed parts' plain mean, said so
    monkeypatch.setattr(drive, "_segment_table", lambda s: {})
    rho, how = drive.g4_density(session, plan)
    assert rho == pytest.approx(sum(e["rho"] for e in entries) / 3) and "no segments yet" in how
    # segments: each matched to its listed part (an unnamed one the Governing material), by area
    areas = {0: {"area_fraction": 0.4}, 1: {"area_fraction": 0.3}, 2: {"area_fraction": 0.2}, 3: {"area_fraction": 0.1}}
    monkeypatch.setattr(drive, "_segment_table", lambda s: areas)
    rho, how = drive.g4_density(session, plan)
    governing = drive.governing_entry(session)["rho"]
    weighted = 0.4 * entries[0]["rho"] + 0.3 * entries[1]["rho"] + 0.2 * entries[2]["rho"] + 0.1 * governing
    assert rho == pytest.approx(weighted) and "matched to the segments" in how
    # per-part materials before valuation take the same mix
    parts = {**plan, "material": {"mode": "parts", "description": "x"}}
    assert drive.plan_density(session, parts) == (pytest.approx(weighted), how)
    # the planned mass stays the solve's density; G4's ratio and its method are the parts'
    box = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    for name, value in (("resolved_plan", lambda s, p: p), ("generated_extents", lambda s: [0.1, 0.1, 0.1]),
                        ("fill_fraction", lambda s: 1.0), ("_generated_mesh", lambda s: box),
                        ("body_source", lambda s: None), ("member_thickness", lambda m: None)):
        monkeypatch.setattr(drive, name, value)
    res = drive.planned_resolution(session, plan)
    assert res["planned_mass"] == pytest.approx(7.85, rel=1e-3)
    assert res["g4_mass"] == pytest.approx(weighted * 1e-3, rel=1e-3)
    assert res["mass_over_real"] == pytest.approx(weighted * 1e-3, rel=1e-2) and "matched" in res["mass_over_real_from"]
    # parts of one density: the plan's own, said so
    same = _session(tmp_path, size=SIZE + "Materials by part: top plate = steel; side grip = steel\n", name="same")
    assert drive.g4_density(same, plan) == (7850, "the plan's density")
    # the geometry stop's record keeps the method beside the ratio
    monkeypatch.setattr(drive, "body_provenance", lambda s: "triposg")
    monkeypatch.setattr(drive, "plan_image_key", lambda p: "img")
    monkeypatch.setattr(drive, "body_digest", lambda s: "d1")
    drive.record_mass_failure(session, plan, 9.0, res["mass_over_real_from"])
    now = lib.read_json(lib.operator_dir(session) / drive.MASS_NOW, {})
    assert now["img"] == 9.0 and "matched" in now["density"]["img"]
    assert drive.mass_blocked_images(session) == {"img"}
    assert "matched" in lib.read_json(lib.operator_dir(session) / drive.MASS_FAILED, [])[0]["density"]
