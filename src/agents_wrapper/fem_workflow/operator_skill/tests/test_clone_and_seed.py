"""`drive.py clone` (a session copied under a new id, its checkpoints re-keyed) and the batch seed
(FEM_OPERATOR_SEED): the batch tools behind experiments/agentic_batch's realization and sweep scripts."""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402


# ------------------------------------------------------------------ seeds --

def test_the_batch_seed_replaces_only_zero_or_unset_seeds():
    plan = {"geometry": {"description": "a chair", "generation_settings": {"seed": 0}},
            "texture": {"texture_prompt": "oak wood", "texture_seed": 0}}
    out, changed = drive.with_batch_seed(plan, 17)
    assert out["geometry"]["generation_settings"]["seed"] == 17
    assert out["texture"]["texture_seed"] == 17
    assert "image_seed" not in out["geometry"]          # not drawing: no image seed added
    assert set(changed) == {"geometry.generation_settings.seed", "texture.texture_seed"}
    assert plan["geometry"]["generation_settings"]["seed"] == 0      # the input is not changed
    again, changed_again = drive.with_batch_seed(out, 17)
    assert again == out and changed_again == []                     # idempotent


def test_the_batch_seed_keeps_the_agents_own_seed_and_sets_a_drawings():
    plan = {"geometry": {"description": "a chair", "draw": True, "generation_settings": {"seed": 3}},
            "texture": {"texture_prompt": "oak", "texture_seed": 2}}
    out, changed = drive.with_batch_seed(plan, 17)
    assert out["geometry"]["generation_settings"]["seed"] == 3
    assert out["texture"]["texture_seed"] == 2
    assert out["geometry"]["image_seed"] == 17 and changed == ["geometry.image_seed"]


def test_a_batch_seed_over_the_generators_limit_is_folded_for_the_geometry_only():
    # The first real realization batch (seeds 101, 102) was refused at the plan lint: the geometry
    # seed is 0-100 (models.py GeometryGenerationSettings), the image and texture seeds are not.
    from src.agents_wrapper.fem_workflow.models import GeometryGenerationSettings
    plan = {"geometry": {"description": "a chair", "draw": True, "generation_settings": {"seed": 0}},
            "texture": {"texture_prompt": "oak", "texture_seed": 0}}
    seeds = {}
    for batch in (100, 101, 102, 200, 201):
        out, _ = drive.with_batch_seed(plan, batch)
        seeds[batch] = out["geometry"]["generation_settings"]["seed"]
        GeometryGenerationSettings(**out["geometry"]["generation_settings"])      # passes the lint
        assert out["geometry"]["image_seed"] == batch and out["texture"]["texture_seed"] == batch
    assert seeds == {100: 100, 101: 1, 102: 2, 200: 100, 201: 1}


def test_no_batch_seed_and_an_uploaded_body_change_nothing(monkeypatch):
    plan = {"geometry": {"asset_path": "x.glb", "generation_settings": {"seed": 0}}}
    assert drive.with_batch_seed(plan, None) == (plan, [])
    assert drive.with_batch_seed(plan, 5) == (plan, [])
    monkeypatch.delenv(drive.SEED_ENV, raising=False)
    assert drive.batch_seed() is None
    monkeypatch.setenv(drive.SEED_ENV, "41")
    assert drive.batch_seed() == 41
    monkeypatch.setenv(drive.SEED_ENV, "zero")
    assert drive.batch_seed() is None


# ------------------------------------------------------------------ clone --

def _graph_session(folder: Path, thread: str):
    """A real LangGraph SqliteSaver database with one thread waiting on an interrupt."""
    from typing import TypedDict

    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph
    from langgraph.types import interrupt

    class S(TypedDict):
        n: int

    def ask(state):
        answer = interrupt({"kind": "shape_opt_settings"})
        return {"n": state["n"] + int(answer or 0)}

    builder = StateGraph(S)
    builder.add_node("ask", ask)
    builder.set_entry_point("ask")
    builder.add_edge("ask", END)
    folder.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(folder / "checkpoints.sqlite", check_same_thread=False)
    graph = builder.compile(checkpointer=SqliteSaver(connection))
    graph.invoke({"n": 1}, config={"configurable": {"thread_id": thread}})
    return graph, connection


def test_rethreading_moves_every_checkpoint_to_the_new_thread(tmp_path):
    graph, connection = _graph_session(tmp_path / "src", "aaaaaaaaaaaa")
    connection.close()
    db = tmp_path / "src" / "checkpoints.sqlite"
    assert drive.checkpoint_threads(db) == ["aaaaaaaaaaaa"]
    moved = drive.rethread_checkpoints(db, "aaaaaaaaaaaa", "bbbbbbbbbbbb")
    assert moved >= 1 and drive.checkpoint_threads(db) == ["bbbbbbbbbbbb"]
    graph, connection = _graph_session(tmp_path / "other", "unused")      # a builder to reopen with
    connection.close()
    from langgraph.checkpoint.sqlite import SqliteSaver
    connection = sqlite3.connect(db, check_same_thread=False)
    try:
        reopened = graph.builder.compile(checkpointer=SqliteSaver(connection))
        state = reopened.get_state({"configurable": {"thread_id": "bbbbbbbbbbbb"}})
        assert [i.value for t in state.tasks for i in t.interrupts] == [{"kind": "shape_opt_settings"}]
        assert not reopened.get_state({"configurable": {"thread_id": "aaaaaaaaaaaa"}}).values
    finally:
        connection.close()


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "batch" / "workflow" / "aaaaaaaaaaaa"
    _, connection = _graph_session(source, "aaaaaaaaaaaa")
    connection.close()
    (source / "operator").mkdir()
    (source / "brief_snapshots").mkdir()
    (source / "geometry").mkdir()
    (source / "geometry" / "physical.glb").write_bytes(b"glb")
    lib.write_json(source / "session.json", {"session_id": "aaaaaaaaaaaa",
                                              "session_dir": "/elsewhere/aaaaaaaaaaaa", "stage": "done"})
    for name in ("plan.json", "brief.md", "judgement.md", "verdict.json", "report.md", "driver.lock",
                 "prediction_history.json", "run_counts.json"):
        (source / "operator" / name).write_text("{}")
    (source / "brief_snapshots" / "brief.solve.md").write_text("brief")
    (source / "brief_snapshots" / "brief.first.md").write_text("brief")
    return source


def _clone(tmp_path, monkeypatch, source, **extra):
    root = tmp_path / "root"
    monkeypatch.setattr(lib, "DEFAULT_OUTPUT_ROOT", tmp_path / "default")
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    forked = []
    monkeypatch.setattr(drive, "fork_newest", lambda g, sid: forked.append(sid))
    monkeypatch.setattr(drive, "event_from", lambda g, sid, r, s: {"card": {"kind": "shape_opt_settings"},
                                                                   "workflow": {"stage": "settings"}})
    printed = []
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    args = argparse.Namespace(output_root=str(root), source=str(source), session_id=None, checkpoint=None,
                              fresh_solve=False, note="material: rubber legs")
    vars(args).update(extra)
    drive.cmd_clone(args)
    return root, printed[-1], forked


def test_clone_copies_the_session_under_a_new_id_and_leaves_the_source(tmp_path, monkeypatch):
    source = _source(tmp_path)
    before = sorted(p.relative_to(source).as_posix() for p in source.rglob("*"))
    root, out, forked = _clone(tmp_path, monkeypatch, source)
    assert out["ok"] and out["cloned_from"] == "aaaaaaaaaaaa" and out["kind"] == "shape_opt_settings"
    new = out["session_id"]
    assert new != "aaaaaaaaaaaa" and forked == [new]
    session = root / new
    assert drive.checkpoint_threads(session / "checkpoints.sqlite") == [new]
    assert drive.checkpoint_threads(source / "checkpoints.sqlite") == ["aaaaaaaaaaaa"]
    assert sorted(p.relative_to(source).as_posix() for p in source.rglob("*")) == before
    snapshot = json.loads((session / "session.json").read_text())
    assert snapshot["session_id"] == new and snapshot["session_dir"] == str(session)
    assert (session / "geometry" / "physical.glb").read_bytes() == b"glb"
    # the source's judgement is not the copy's; its solve records stay without --fresh-solve
    assert not (session / "operator" / "judgement.md").exists()
    assert not (session / "operator" / "verdict.json").exists()
    assert (session / "operator" / "report.md").exists()
    assert (session / "brief_snapshots" / "brief.solve.md").exists()
    record = json.loads((session / "operator" / drive.CLONE_RECORD).read_text())
    assert record["source_session_id"] == "aaaaaaaaaaaa" and record["note"] == "material: rubber legs"
    assert record["source_session_dir"] == "/elsewhere/aaaaaaaaaaaa"
    assert (tmp_path / "default" / new).resolve() == session.resolve()     # the path the skill names
    log = [json.loads(line) for line in (session / "operator" / "driver_log.jsonl").read_text().splitlines()]
    assert log[-1]["command"] == "clone"


def test_clone_with_a_fresh_solve_drops_the_solve_records(tmp_path, monkeypatch):
    source = _source(tmp_path)
    root, out, _ = _clone(tmp_path, monkeypatch, source, fresh_solve=True, session_id="cccccccccccc")
    session = root / "cccccccccccc"
    assert out["session_id"] == "cccccccccccc"
    for gone in drive.CLONE_FRESH_SOLVE:
        assert not (session / gone).exists(), gone
    assert (session / "brief_snapshots" / "brief.first.md").exists()       # the body's snapshot stays
    assert (session / "operator" / "run_counts.json").exists()              # the limits stay counted
    assert (source / "operator" / "report.md").exists()


def test_clone_refuses_an_existing_id_and_a_missing_source(tmp_path, monkeypatch):
    source = _source(tmp_path)
    _clone(tmp_path, monkeypatch, source, session_id="dddddddddddd")
    _, out, _ = _clone(tmp_path, monkeypatch, source, session_id="dddddddddddd")
    assert not out["ok"] and "already exists" in out["error"]
    _, out, _ = _clone(tmp_path, monkeypatch, tmp_path / "nothing")
    assert not out["ok"] and "no session" in out["error"]


def test_clone_with_an_instruction_stores_it_for_the_checks_and_keeps_the_sources(tmp_path, monkeypatch):
    # A BC sweep's sizes are the variant's: without this, "source": "given" was refused for them.
    source = _source(tmp_path)
    drive.store_instruction(source, "a sewing machine")
    root, out, _ = _clone(tmp_path, monkeypatch, source, session_id="eeeeeeeeeeee",
                          instruction="a sewing machine | bc: a 60 N pull on the head")
    session = root / "eeeeeeeeeeee"
    operator = session / "operator"
    assert drive.stored_instruction(operator) == "a sewing machine | bc: a 60 N pull on the head"
    assert drive.given_in(60.0, drive.stored_instruction(operator))
    record = json.loads((operator / drive.CLONE_RECORD).read_text())
    assert record["source_instruction"] == "a sewing machine"
    assert drive.stored_instruction(source / "operator") == "a sewing machine"          # the source's stays
    # without it the copy keeps the source's
    root, _, _ = _clone(tmp_path, monkeypatch, source, session_id="ffffffffffff")
    assert drive.stored_instruction(root / "ffffffffffff" / "operator") == "a sewing machine"


def test_a_run_with_the_batch_seed_writes_it_into_the_plan_and_sends_it(tmp_path, monkeypatch):
    from conftest import tabled
    from test_operator_scripts import PLAN
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    (session / "checkpoints.sqlite").write_text("")
    (session / "operator" / "brief.md").write_text(tabled(
        "## Use\nA load rests on its top for an hour. {\"loads\": {\"top load\": 1.0}, \"unit\": \"Pa\", \"governing\": \"top load\"}\n## Contacts\ntwo supports on the ground, held fully\n"
        "## Load\n785 N on the top, 80 kg x g x 1\n## Size and material\nReal mass: 30 kg\nReal thinnest member: 0.05 m\nReal section: 0.05 m deep, I 5.2e-7 m4, A 2.5e-3 m2\n0.9 m tall, wood 1.2e10\n"
        "## Prediction\nmembers in compression, 0.1 MPa, microns"))
    plan = {**PLAN, "geometry": {**PLAN["geometry"], "generation_settings": {"seed": 0}}}
    state = {"card": {"kind": "geometry_input", "description": ""}}
    printed, sent = [], []
    monkeypatch.setenv(drive.SEED_ENV, "23")
    monkeypatch.setattr(drive, "open_graph", lambda s: contextlib.nullcontext(None))
    monkeypatch.setattr(drive, "pending", lambda g, sid: {"card": state["card"], "workflow": {},
                                                         "checkpoint_id": "c", "next": []})
    monkeypatch.setattr(drive, "_newest_card", lambda g, sid, kind: "checkpoint")
    monkeypatch.setattr(drive, "fork", lambda g, sid, c: None)
    monkeypatch.setattr(drive, "join_paint_ahead", lambda session: None)
    monkeypatch.setattr(drive, "start_paint_ahead", lambda *a, **k: None)
    monkeypatch.setattr(drive, "preload", lambda *a, **k: None)
    monkeypatch.setattr(drive, "prepare_answer", lambda c, w, a, merge, force: drive.Prepared(answer=a))
    monkeypatch.setattr(drive, "_stop_payload", lambda *a, **k: {"why": "", "look_at": []})
    monkeypatch.setattr(drive, "emit", lambda value, session=None: printed.append(value))
    monkeypatch.setattr(drive, "resolved_plan", lambda session, plan: plan)

    def step(args, session, root, env, head, kind, prepared, send, show=True):
        sent.append(prepared.answer)
        state["card"] = {"kind": "bc_input", "description": ""}
        return {"ok": True, "card": state["card"]}
    monkeypatch.setattr(drive, "_run_step", step)
    path = session / "operator" / "plan.json"
    path.write_text(json.dumps(plan, indent=1) + "\n")
    drive.cmd_run(argparse.Namespace(output_root=str(tmp_path), session_id="abc", plan=str(path), full=True))
    assert json.loads(path.read_text())["geometry"]["generation_settings"]["seed"] == 23
    assert any((a.get("generation_settings") or {}).get("seed") == 23 for a in sent if isinstance(a, dict))
    assert "FEM_OPERATOR_SEED" in json.dumps(printed[-1], default=str)


def test_clone_is_a_parser_command():
    args = drive.parser().parse_args(["clone", "some/folder", "--fresh-solve", "--note", "x"])
    assert args.func is drive.cmd_clone and args.fresh_solve and args.note == "x" and args.instruction is None
    args = drive.parser().parse_args(["clone", "some/folder", "--instruction", "a 60 N pull"])
    assert args.instruction == "a 60 N pull"
