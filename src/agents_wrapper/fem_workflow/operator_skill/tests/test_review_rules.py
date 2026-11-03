"""The rules of the review of the batch of 2026-10-12: built first honoured wherever `built` is listed,
with `blank` running the one photograph search itself; a move past a generated body or drawing that
passed the runner's checks says which check failed; the brief's Real lines frozen once a body exists;
several real materials on the load path; an uncovered governing state that an approximation could
carry; the applied load against the Use table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

import drive
import sessionlib as lib

THIN_BRIEF = ("## Size and material\nWhy not generated: the walls are too thin for a generated surface\n"
              "Real thinnest member: 0.005 m\nReal extents: 1 x 0.5 x 0.5 m\n")


def _operator(session: Path) -> Path:
    (session / "operator").mkdir(parents=True, exist_ok=True)
    return session / "operator"


def _found(searched, candidates=1):
    """A stand-in for find_web_images: ``candidates`` survivors of the pre-screen."""
    def find(args, session, output_root, compute_env, query):
        searched.append(query)
        folder = drive.web_folder(session, query)
        shown = []
        for k in range(candidates):
            (folder / f"c-{k}").mkdir(parents=True)
            (folder / f"c-{k}" / "cut.png").write_bytes(b"png")
            shown.append({"id": f"c-{k}", "image": str(folder / f"c-{k}" / "cut.png"),
                          "thumb": str(folder / f"c-{k}" / "cut.png"), "licence": "CC0 1.0"})
        out = {"query": query, "found": candidates, "sources": [], "candidates": shown}
        folder.mkdir(parents=True, exist_ok=True)
        lib.write_json(folder / "candidates.json", out)
        return out
    return find


def test_built_first_is_honoured_wherever_built_is_listed(tmp_path):
    operator = _operator(tmp_path)
    (operator / "brief.md").write_text(THIN_BRIEF)
    assert drive.built_first_ok(tmp_path)
    for routes in (["drawing", "built"], ["built", "drawing"], ["hunyuan3d", "built", "triposg_text"]):
        assert drive.canonical_routes(routes, tmp_path)[:2] == ["web_image", "built"]
    lib.write_json(operator / "geometry_routes.json", drive.canonical_routes(["drawing", "built"], tmp_path))
    # the photograph, first in the sorted list, is not refused as out of turn
    assert drive.route_order_problem(tmp_path, "geometry_input",
                                     {"asset_path": drive.WEB_DIR, "asset_kind": "image"}) is None


@pytest.mark.web_rule
def test_blank_runs_the_one_photograph_search_itself(tmp_path, monkeypatch):
    operator = _operator(tmp_path)
    (operator / "brief.md").write_text(THIN_BRIEF)
    lib.write_json(operator / "geometry_routes.json", drive.canonical_routes(["drawing", "built"], tmp_path))
    plan = {"geometry": {"description": "steel item frame, two posts and a bar"}}
    lib.write_json(operator / "plan.json", plan)
    args = argparse.Namespace(gpu="auto")
    searched = []
    monkeypatch.setattr(drive, "find_web_images", _found(searched, candidates=1))
    out = drive.blank_web_search(args, tmp_path, tmp_path, "env")
    # the description's first clause is searched, and the photographs are shown instead of a refusal
    assert searched == ["steel item frame"] and out["stop"] == "web_image"
    assert out["web_image"]["candidates"][0]["id"] == "c-0" and '"pick": "none"' in out["next"]
    # turned down in the plan: `blank` builds (None), with no second search
    plan["geometry"]["web_image"] = {"query": "steel item frame", "pick": "none",
                                     "why": "none shows the whole object alone"}
    lib.write_json(operator / "plan.json", plan)
    assert drive.blank_web_search(args, tmp_path, tmp_path, "env") is None and searched == ["steel item frame"]
    assert drive.web_photo_problem(tmp_path) is None
    # nothing found: built at once
    other = tmp_path / "other"
    _operator(other)
    (other / "operator" / "brief.md").write_text(THIN_BRIEF)
    lib.write_json(other / "operator" / "geometry_routes.json", drive.canonical_routes(["built"], other))
    lib.write_json(other / "operator" / "plan.json", {"geometry": {"description": "steel item frame"}})
    monkeypatch.setattr(drive, "find_web_images", _found(searched, candidates=0))
    assert drive.blank_web_search(args, other, other, "env") is None and drive.web_image_done(other)


# ------------------------------------------------- moving past a recognisable attempt --

def test_a_move_past_a_passed_drawing_or_body_names_the_check_it_failed(tmp_path, monkeypatch):
    from test_use_table import session_with, table
    session = session_with(tmp_path, table())
    # the Real section's member, the run's load entry, and each held contact's member
    assert drive.primary_members(session) == ["top plate", "side grip", "bottom plate"]
    drawn = {"description": "steel part, a top plate on a side grip", "image_seed": 0, "draw": True}
    value = {"ok": True, "look_at": ["d.png"], "drawing": {"objects": 1}}
    drive.record_standing(session, "drawing", value, drawn)
    # generating from it (any generator) is no move; other words are, and are refused silent
    assert drive.move_on_problem(session, {"geometry": {**drawn, "generator": "hunyuan3d"}}, "") is None
    other = {"geometry": {**drawn, "image_seed": 1}}
    assert "plan.geometry.reject" in drive.move_on_problem(session, other, "")
    # a sub-member that is not primary is accepted and declared, never a reason
    said = drive.move_on_problem(session, other, "the handle is missing")
    assert "names none of the checks" in said and "body_simplification" in said
    for line in ("the top plate is missing", "the wrong object: a shelf", "two objects drawn",
                 "only part of the object", "the cut-out lost an edge"):
        assert drive.move_on_problem(session, other, line) is None, line
    plan, line = drive.split_reject({"geometry": {**other["geometry"], "reject": "  the wrong\n object "}})
    assert "reject" not in plan["geometry"] and line == "the wrong object"
    drive.record_move_on(session, plan, line)
    assert not (session / "operator" / drive.STANDING).exists()
    assert lib.read_json(session / "operator" / drive.REJECTED)[-1]["reject"] == "the wrong object"
    # a drawing of several objects never stands: moving on needs no line
    drive.record_standing(session, "drawing", {**value, "drawing": {"objects": 2}}, drawn)
    assert drive.move_on_problem(session, other, "") is None
    # a free-form body the runner's checks passed stands; the same line again is the previous one's
    monkeypatch.setattr(drive, "generated_body", lambda s: True)
    monkeypatch.setattr(drive, "body_digest", lambda s: "abc")
    body = {"ok": True, "geometry": {"components": 1, "watertight": True},
            "planned_resolution": {"g4": "within the rule"}}
    drive.record_standing(session, "geometry", body, drawn)
    assert drive.move_on_problem(session, {"geometry": drawn}, "") is None
    built = {"geometry": {"asset_path": "o/blank_box_0a1b2c3d.glb", "asset_kind": "surface"}}
    assert "plan.geometry.reject" in drive.move_on_problem(session, built, "")
    assert "said of the previous attempt" in drive.move_on_problem(session, built, "the wrong object")
    # a body the runner refused (its error, G4 outside the rule) does not stand
    drive.record_standing(session, "geometry", {**body, "planned_resolution": {"g4": "'could not be run' (x)"}}, drawn)
    assert drive.move_on_problem(session, built, "") is None
    monkeypatch.setattr(drive, "generated_body", lambda s: False)       # a built or CAD body never stands
    drive.record_standing(session, "geometry", body, drawn)
    assert drive.move_on_problem(session, built, "") is None


def test_the_real_lines_freeze_once_a_body_exists(tmp_path):
    operator = _operator(tmp_path)
    size = "Real mass: 2 kg\nReal thinnest member: 0.004 m\nReal extents: 1 x 0.5 x 0.5 m\n"
    (operator / "brief.md").write_text("## Use\nu\n## Size and material\n" + size)
    assert drive.real_lines_frozen_problem(operator) is None                 # no body yet: edits are free
    drive.brief_status(operator)                                              # the body's snapshot
    (operator / "brief.md").write_text("## Use\nu, edited\n## Size and material\n" + size
                                       + "Real extents revised: 1.2 x 0.5 x 0.5 m, the body's legs are longer\n")
    assert drive.real_lines_frozen_problem(operator) is None                 # prose and the one revision
    (operator / "brief.md").write_text("## Size and material\n" + size.replace("0.004", "0.02"))
    said = drive.real_lines_frozen_problem(operator)
    assert "frozen" in said and "Real thinnest member: 0.004 m" in said and "0.02" in said


def test_a_uniform_material_across_different_real_materials_needs_its_reason(tmp_path):
    from test_use_table import SIZE, session_with, table
    mixed = SIZE + "Materials: top plate steel, E 200 GPa; side grip and bottom plate ABS plastic\n"
    session = session_with(tmp_path, table(), size=mixed)
    found = drive.load_path_materials(session)
    assert set(found) == {"top plate", "side grip", "bottom plate"} and found["top plate"][0] == "steel"
    uniform = {"material": {"youngs_modulus": 2e11, "poisson_ratio": 0.3, "density": 7850}}
    said = drive.uniform_materials_problem(session, uniform)
    assert '"mode": "parts"' in said and "uniform_why" in said
    why = {"material": {**uniform["material"], "uniform_why": "shape optimization takes one material"}}
    assert drive.uniform_materials_problem(session, why) is None
    assert drive.uniform_materials_problem(session, {"material": {"mode": "parts", "description": "x"}}) is None
    same = session_with(tmp_path, table(), size=SIZE + "Materials: top plate, side grip and bottom plate steel\n",
                        name="same")
    assert drive.uniform_materials_problem(same, uniform) is None
    answer = drive.plan_answer("material_input", {}, why)[0]
    assert "uniform_why" not in answer


def test_an_uncovered_state_an_approximation_could_carry_says_why_not(tmp_path):
    from test_use_table import gate, session_with, table
    later = {"A3": {"size": 340, "source": "34 kg x g x 1"}}
    bare = {"limit": "dynamic or impact", "why": "the carry is a jolt at the grip"}
    said = gate(session_with(tmp_path, table(rows=later, uncovered={"side grip": bare, "bottom plate": bare})))
    assert any("no_approximation" in s and "inertial approximation" in s for s in said)
    told = {**bare, "no_approximation": "the jolt's factor is unknown"}
    assert gate(session_with(tmp_path, table(rows=later, uncovered={"side grip": told, "bottom plate": told}),
                             name="told")) == []


def test_a_body_load_short_of_its_use_table_rows_is_refused_and_never_no_deviation(tmp_path, monkeypatch):
    from test_use_table import session_with, table
    session = session_with(tmp_path, table(run="handling"))
    size = drive.use_table(session / "operator")[0]["sizes"]["A3"][0]          # 10 kg x g
    plan = {"material": {"youngs_modulus": 2e9, "poisson_ratio": 0.35, "density": 1000.0},
            "values": {"acceleration": [0, -9.80665, 0]}, "loads": {"A3": {"acceleration": True}}}
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: {"planned_mass": 5.0})
    said = drive.load_shortfall_problem(session, plan, [])
    assert "['A3']" in said and "shortfall_why" in said and f"{size:g} N" in said
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: {"planned_mass": 9.0})
    assert drive.load_shortfall_problem(session, plan, []) is None              # within 20 %
    monkeypatch.setattr(drive, "planned_resolution", lambda s, p: {"planned_mass": 5.0})
    plan["loads"]["A3"]["shortfall_why"] = "the body is half the real mass"
    assert drive.load_shortfall_problem(session, plan, []) is None
    assert lib.read_json(session / "operator" / drive.SHORTFALL_RECORD)[0]["why"].startswith("the body")
    lines = drive.runner_simplifications(session, {}, [])
    assert any("applied at" in line for line in lines)
    for none in ("## Deviations\n\n- none\n", "Deviations: none."):
        assert "name that shortfall" in drive.shortfall_judged_problem(session, "v\n" + none)
    assert drive.shortfall_judged_problem(session, "v\n## Deviations\nThe body load is half the rows'.") is None


def test_the_skill_text_accepts_a_recognisable_attempt_without_one_sub_member():
    text = (drive.SKILL_DIR / "references" / "visual_checks.md").read_text(encoding="utf-8")
    assert "geometry.reject" in text and "primary member" in text and "every load-bearing member" not in text
    geometry = (drive.SKILL_DIR / "references" / "geometry.md").read_text(encoding="utf-8")
    assert "every load-bearing member" not in geometry
