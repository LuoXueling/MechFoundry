"""Gates whose way out was a plan edit (the review of 2026-10-12): the refinement ladder on the same image
needs no reject check, only the poor generation it fixes; a support's member is primary; the route list
never shrinks once an attempt was sent, and a route the runner refused counts as taken."""

from __future__ import annotations

import json
from pathlib import Path

import drive
import sessionlib as lib

DRAWN = {"description": "wooden toy cart, a box body on four wheels", "draw": True, "image_seed": 0,
         "generation_settings": {"seed": 1}}
BODY = {"ok": True, "geometry": {"components": 1, "watertight": True}, "planned_resolution": {"g4": "within the rule"}}


def _standing_body(session: Path, monkeypatch) -> None:
    (session / "operator").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(drive, "generated_body", lambda s: True)
    monkeypatch.setattr(drive, "body_digest", lambda s: "abc")
    drive.record_standing(session, "geometry", BODY, DRAWN)


def test_the_refinement_ladder_on_the_same_image_needs_only_the_poor_generation(tmp_path, monkeypatch):
    _standing_body(tmp_path, monkeypatch)
    fine = {"geometry": {**DRAWN, "generation_settings": {"seed": 1, "init_octree_depth": 6, "octree_depth": 9}}}
    hunyuan = {"geometry": {**DRAWN, "generator": "hunyuan3d", "generation_settings": {"seed": 0}}}
    poor = "the wheels came out fused into a blob and the body is shredded"
    for rung in (fine, hunyuan):
        assert drive.move_on_problem(tmp_path, rung, poor) is None
        assert "poor generation it fixes" in drive.move_on_problem(tmp_path, rung, "")
        assert "poor generation it fixes" in drive.move_on_problem(tmp_path, rung, "looks odd")
    # the same reason again on the next rung is no repeat (the fault often stays)
    drive.record_move_on(tmp_path, fine, poor)
    _standing_body(tmp_path, monkeypatch)
    assert drive.move_on_problem(tmp_path, hunyuan, poor) is None
    # a new image, a new seed at the same depth, or another depth pair is no rung: a check is needed
    for other in ({**DRAWN, "image_seed": 1},
                  {**DRAWN, "generation_settings": {"seed": 2}},
                  {**DRAWN, "generation_settings": {"seed": 1, "init_octree_depth": 4, "octree_depth": 8}}):
        said = drive.move_on_problem(tmp_path, {"geometry": other}, "the cab is fused")
        assert said and "names none of the checks" in said, other
    assert not drive.routes_refused(tmp_path) or set(drive.routes_refused(tmp_path)) == {"drawing"}


def test_a_support_member_is_primary(tmp_path, monkeypatch):
    from test_use_table import session_with, table
    contacts = ('- wheel rests: the wheels stand on the floor\n'
                '{"wheel rests": {"transmits": "force only", "kind": "rest", "member": "axle"}}')
    held = {"held": ["wheel rests"]}
    session = session_with(tmp_path, table(rows={"A1": held, "A2": held}), contacts=contacts)
    primary = drive.primary_members(session)
    assert "axle" in primary and primary[0] == "top plate" and "wheel" not in primary
    assert drive.reject_problem(session, "the axle is fused into one blob") is None
    assert "names none of the checks" in drive.reject_problem(session, "the chimney is missing")
    # a hold's surroundings are no member: "floor rest" resolves through the Load path, not its words
    floor = ('- floor rest: the bottom plate stands on the floor\n'
             '{"floor rest": {"transmits": "force only", "kind": "rest"}}')
    held = {"held": ["floor rest"]}
    other = session_with(tmp_path, table(rows={"A1": held, "A2": held}), contacts=floor, name="floor")
    assert "floor" not in drive.primary_members(other)


def test_the_route_list_never_shrinks_once_an_attempt_was_sent(tmp_path, monkeypatch):
    operator = tmp_path / "operator"
    operator.mkdir()
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    monkeypatch.setattr(drive, "member_unresolvable", lambda s, p: False)
    listed = ["triposg_text", "drawing", "hunyuan3d"]
    lib.write_json(operator / "geometry_routes.json", listed)
    # before any attempt the brief may still narrow its list
    assert drive.kept_routes(tmp_path, ["triposg_text", "drawing"]) == (["triposg_text", "drawing"], [])
    lib.write_json(operator / "run_counts.json", {"generation": 2, "drawing": 1})
    assert drive.kept_routes(tmp_path, ["triposg_text", "drawing"]) == (listed, ["hunyuan3d"])
    plan = {"geometry": {**DRAWN}, "geometry_routes": ["triposg_text", "drawing"],
            "body_simplification": {"absent": ["wheels"]}}
    said = drive.body_simplification_problem(tmp_path, plan)
    assert "['hunyuan3d'] not yet" in said and "take off the list" not in said
    # a route the runner refused (the standing body kept, or the budget spent) counts as taken
    _standing_body(tmp_path, monkeypatch)
    cad = {"geometry": {"description": "a cart", "generator": "cadcoder"}}
    lib.write_json(operator / "geometry_routes.json", listed + ["cadcoder"])
    assert "names none of the checks" in drive.move_on_problem(tmp_path, cad, "the cab is wrong")
    assert "cadcoder" in drive.routes_refused(tmp_path)
    drive.note_route_refused(tmp_path, "hunyuan3d", "the generated budget was spent")
    assert drive.body_simplification_problem(tmp_path, plan) is None
    assert json.dumps(drive.routes_refused(tmp_path))


def test_a_run_load_lowered_after_the_body_is_a_simplification(tmp_path):
    from test_use_table import session_with, table
    session = session_with(tmp_path, table())
    operator = session / "operator"
    snapshots = session / "brief_snapshots"
    snapshots.mkdir(exist_ok=True)
    (snapshots / "brief.first.md").write_text((operator / "brief.md").read_text())
    assert drive.rows_lowered_since_body(operator) == []
    # a push cut to an invented derivation after the body, to clear a later gate
    cut = table(rows={"A2": {"source": "derived: 10 N", "size": None}})
    session_with(tmp_path, cut)
    said = drive.rows_lowered_since_body(operator)
    assert said == ["Use table row A2 50 N -> 10 N"]
    assert any("load lowered after the body: Use table row A2" in s
               for s in drive.runner_simplifications(session, {}, []))
    # a raised row, or the row the runner's own warning flags set to another Human actions row, is none
    session_with(tmp_path, table(rows={"A2": {"source": "H3 middle", "size": 60}}))
    assert drive.rows_lowered_since_body(operator) == []
    # the early check's record: a row lowered after it is listed with the lines changed
    lib.write_json(operator / drive.EARLY_SAID, {"lines": drive.checked_lines(operator),
                                                 "rows": drive.run_row_sizes(operator)})
    session_with(tmp_path, cut)
    assert drive.early_check_changes(operator) == ["Use table row A2 60 N -> 10 N"]


def test_a_declared_piece_count_is_the_one_seen_and_faces_the_contact_s_way(tmp_path):
    from test_use_table import session_with, table
    contacts = ('- feet: the feet stand on the floor\n'
                '{"feet": {"transmits": "force only", "kind": "rest", "member": "bottom plate"}}')
    held = {"held": ["feet"]}
    session = session_with(tmp_path, table(rows={"A1": held, "A2": held}), contacts=contacts)
    down = {"region": "base", "contact": "feet", "face": "end face", "member_axis": "y", "span_m": [0, 0.1], "normal": [0, -1, 0], "area": 1.0}
    side = {**down, "face": "side face", "normal": [1, 0, 0], "area": 0.2}
    rows = [{**down, "name": "base#1"}, {**down, "name": "base#2"}, {**side, "name": "base#3"}]
    declared = {"feet": {"regions": ["base"], "transmits": "force only", "pieces": 3, "pieces_why": "three feet"}}
    said = drive.pieces_problem(rows, declared, session)
    assert said and "['base#3'] face another way" in said
    # a count other than the one seen is refused, also after a first refusal
    four = {"feet": {**declared["feet"], "pieces": 4}}
    assert "another count is never taken" in drive.pieces_problem(rows[:2], four, session)
    assert "another count is never taken" in drive.pieces_problem(rows[:2], four, session)
    # the pieces seen, all facing the rest's way: taken
    two = {"feet": {**declared["feet"], "pieces": 2}}
    assert drive.pieces_problem(rows[:2], two, session) is None
    assert drive.off_direction(rows[:2]) == []
    # the count seen is no simplification: nothing marked
    assert not (session / "operator" / drive.PIECES_ACCEPTED).exists()
    assert not any("pieces where the runner sees" in s for s in drive.runner_simplifications(session, {}, []))
    # a plural name does not take faces caught off the rest's direction either
    plural = {"feet": {"regions": ["base"], "transmits": "force only"}}
    assert "face another way" in drive.pieces_problem(rows, plural, session)
    # a V-cradle's two flanks (90 degrees apart) and a hook's inside are one rest's faces
    flank = [{**down, "name": "base#1", "normal": [0.707, 0.707, 0]}, {**down, "name": "base#2", "normal": [-0.707, 0.707, 0]}]
    assert drive.off_direction(flank) == [] and drive.pieces_problem(flank, plural, session) is None


def test_a_derivation_shows_its_formula(tmp_path):
    from test_use_table import session_with, table, gate
    for formula in ("derived: 25 kg x g x 0.05", "derived: 1000 kg/m3 x 9.80665 m/s2 x 2 m",
                    "derived: 2 kPa x 300 cm2", "derived: 120 N x 0.05 m"):
        assert drive.derivation_problem(formula) is None, formula
    for bare in ("derived: 60 N", "derived: 6 N m", "derived: 12"):
        assert "a bare number" in drive.derivation_problem(bare), bare
    said = gate(session_with(tmp_path, table(rows={"A2": {"size": None, "source": "derived: 60 N"}})))
    assert any("is a bare number, no derivation" in s for s in said)


def test_the_tipping_cap_is_sized_by_the_runner(tmp_path):
    from test_use_table import session_with, table, gate
    session = session_with(tmp_path, table(rows={"A2": {"size": None, "source": "tipping cap"}}))
    assert any("computed none for it" in s for s in gate(session))
    lib.write_json(session / "operator" / drive.TIPPING_CAPS, {"A2": {"cap": 21.5, "from": 50}})
    built, problems = drive.use_table(session / "operator")
    assert built["sizes"]["A2"][0] == 21.5 and not any("A2" in p for p in problems)
    assert "derived: <" not in drive.tipping_exits({"contacts": [], "caps": {"A2": 21.5}, "tipping_force": 21.5,
                                                    "side_force": 50}, {})


def test_a_thinnest_member_raised_after_the_warning_needs_its_reason(tmp_path):
    operator = tmp_path / "operator"
    operator.mkdir()
    snapshots = tmp_path / "brief_snapshots"
    snapshots.mkdir()
    size = "## Size and material\nReal mass: 1 kg\nReal thinnest member: {} m\n"
    (snapshots / "brief_initial.md").write_text(size.format(0.012))
    lib.write_json(snapshots / "member_warning.json", {"member_m": 0.012, "section": None})
    (snapshots / "brief.first.md").write_text(size.format(0.04))
    (operator / "brief.md").write_text(size.format(0.04))
    edit = drive.pre_body_edits(operator)[0]
    assert not edit["allowed"] and "not taken" in edit["text"]
    assert drive.real_member(operator) == 0.012                       # the caveat's member stays
    why = size.format(0.04) + "Thinnest member why: the boiler wall is the thinnest loaded member, 40 mm\n"
    (snapshots / "brief.first.md").write_text(why)
    (operator / "brief.md").write_text(why)
    edit = drive.pre_body_edits(operator)[0]
    assert edit["allowed"] and edit["warned"] and drive.real_member(operator) == 0.04
    assert any("raised after the runner's member warning" in s
               for s in drive.runner_simplifications(tmp_path, {}, []))


def test_a_clone_measures_its_rows_from_its_own_first_run(tmp_path, monkeypatch):
    from test_clone_and_seed import _clone, _source
    from test_use_table import session_with, table
    briefs = {k: (session_with(tmp_path / "briefs", t, name=k) / "operator" / "brief.md").read_text()
              for k, t in (("source", table()), ("variant", table(rows={"A2": {"size": None, "source": "derived: 4 N x 5"}})),
                           ("cut", table(rows={"A2": {"size": None, "source": "derived: 2 N x 5"}})))}
    source = _source(tmp_path)
    (source / "operator" / "brief.md").write_text(briefs["source"])
    (source / "brief_snapshots" / "brief.first.md").write_text(briefs["source"])
    lib.write_json(source / "operator" / drive.EARLY_SAID, {"lines": [], "rows": {"A2": [50.0, "N"]}})
    for variant in ("variant", "source"):          # a BC sweep's new rows, a material sweep's same rows
        root, out, _ = _clone(tmp_path, monkeypatch, source, session_id=f"{variant[0] * 12}")
        session = root / out["session_id"]
        operator = session / "operator"
        assert "rows" not in lib.read_json(operator / drive.EARLY_SAID)
        (operator / "brief.md").write_text(briefs[variant])
        drive.take_rows_baseline(session)
        assert lib.read_json(session / drive.ROWS_BASELINE)["rows"], variant
        assert drive.rows_lowered_since_body(operator) == [], variant
        assert not any("lowered" in s for s in drive.early_check_changes(operator)), variant
    # the variant's own cut after its first run is marked
    (operator / "brief.md").write_text(briefs["cut"])
    assert drive.rows_lowered_since_body(operator) == ["Use table row A2 50 N -> 10 N"]
    # the source itself still measures from its body
    (source / "operator" / "brief.md").write_text(briefs["variant"])
    assert drive.rows_lowered_since_body(source / "operator") == ["Use table row A2 50 N -> 20 N"]
