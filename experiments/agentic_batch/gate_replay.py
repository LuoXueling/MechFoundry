"""Replay the operator skill's four gates on COPIES of archived batch sessions (a test-only tool).

    python experiments/agentic_batch/gate_replay.py prepare RUN_DIR [RUN_DIR ...]
    python experiments/agentic_batch/gate_replay.py run FIXTURE.json [FIXTURE.json ...]

``prepare`` copies ``RUN_DIR/workflow/<sid>`` to ``$GATE_REPLAY_ROOT/<sid>`` (default
/tmp/gate_replay; the archive is never written) and writes ``drafts/<sid>.json``: the actions the
run's stored use anchor listed (read from its file, no API call: the skill makes none since
2026-10-11) beside the agent's own numbers (its contacts, Load path, Real lines, the plan's
loads, the verdict), from which a fixture is transcribed by hand.  A fixture holds only numbers
the agent wrote: an action the agent never sized gets no row.

``run`` writes a fixture's Use table, Contacts JSON v2, the Real section's member and the Real
extents into the copy's brief (and its first-run snapshot, which the contacts are read from),
migrates the final plan to the new fields (the deleted ones dropped; ``plan_patch`` merged on
top; each row's "state" and "kind" from the stored anchor's actions when the fixture gives none;
the runner computes the effects, a fixture's "effect" is ignored), and calls the gates: use_gate (G1), hold_gate and path_gate (G2), plausibility_gate and
verdict_gate on the archived judgement (G3), body_gate (G4).  The results go to
``results/<sid>.json`` and are printed one line per gate.  Nothing here is part of the skill.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "src/agents_wrapper/fem_workflow/operator_skill/scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402

ROOT = Path(os.environ.get("GATE_REPLAY_ROOT", "/tmp/gate_replay"))
ARCHIVE = REPO.parents[2] / "output" / "agentic_batch" if (REPO.parents[2] / "output").exists() else None


def _copy(run_dir: Path) -> Path:
    sessions = sorted((run_dir / "workflow").iterdir())
    src = sessions[-1]
    dest = ROOT / src.name
    if not dest.exists():
        shutil.copytree(src, dest, symlinks=True)
    lib.write_json(dest / "replay_source.json", {"run_dir": str(run_dir), "session": str(src)})
    return dest


def _json_after(text: str, label: str):
    return drive._json_after(text, label)


def batch_instruction(run_dir: Path) -> str | None:
    """The user's instruction for a run, from its batch's summary.json (runs from before the use
    anchor stored none in the session)."""
    for summary in sorted((run_dir.parent / "_batches").glob("*/summary.json")):
        data = json.loads(summary.read_text())
        rows = data if isinstance(data, list) else data.get("runs") or data.get("rows") or []
        for row in rows:
            if isinstance(row, dict) and row.get("run_name") == run_dir.name and row.get("instruction"):
                return row["instruction"]
    return None


USE_ANCHOR = "use_anchor.json"          # where a run before 2026-10-11 stored its anchor (data only)


def stored_actions(session: Path) -> dict[str, dict]:
    """The stored use anchor's actions by label, "A1", "A2", ... across its states (its file; no
    API call): each {"action", "acts_on", "state", "kind"}."""
    record = lib.read_json(session / "brief_snapshots" / USE_ANCHOR, {}) or {}
    out: dict[str, dict] = {}
    for state in ((record.get("anchor") or {}).get("states") or []):
        for action in state.get("actions") or []:
            out[f"A{len(out) + 1}"] = {"action": action.get("action"), "acts_on": action.get("acts_on"),
                                       "state": state.get("state"), "kind": action.get("kind")}
    return out


def prepare(run_dir: Path) -> dict:
    session = _copy(run_dir)
    operator = session / "operator"
    old = lib.read_json(session / "brief_snapshots" / USE_ANCHOR, {}) or {}
    instruction = batch_instruction(run_dir) or old.get("instruction")
    if not instruction:
        raise SystemExit(f"{run_dir.name}: no instruction in its batch summary or its anchor")
    drive.store_instruction(session, instruction)
    if not (operator / "brief.md").exists():
        draft = {"sid": session.name, "run": run_dir.name, "instruction": instruction,
                 "note": "the run wrote no brief: nothing to replay"}
        (ROOT / "drafts").mkdir(parents=True, exist_ok=True)
        lib.write_json(ROOT / "drafts" / f"{session.name}.json", draft)
        return {**draft, "actions": None}
    brief = (operator / "brief.md").read_text(encoding="utf-8")
    sections = drive.brief_sections(brief)
    plan = lib.read_json(operator / "plan.json", {}) or {}
    judgement = (operator / "judgement.md").read_text(encoding="utf-8") if (operator / "judgement.md").exists() else ""
    history = lib.read_json(operator / "prediction_history.json", []) or []
    draft = {
        "sid": session.name, "run": run_dir.name, "instruction": instruction,
        "actions": {k: f"{a['state']}: {a['action']} (on {a['acts_on']}) [{a['kind']}]"
                    for k, a in stored_actions(session).items()},
        "comparison": next((v for v in drive._json_objects(sections.get("use", "")) if "loads" in v), None),
        "contacts": drive.contacts_v2_in(sections.get("contacts", "")),
        "load_path": drive.load_path_in(sections.get("size and material", "")),
        "real_section": drive.real_section_in(sections.get("size and material", "")),
        "real_mass": drive.real_mass_in(sections.get("size and material", "")),
        "plan_loads": plan.get("loads"), "neumann_values": (plan.get("values") or {}).get("neumann_values"),
        "load": (plan.get("values") or {}).get("load"), "scale": plan.get("scale"),
        "material": plan.get("material"), "prediction": plan.get("prediction"),
        "body_simplification": plan.get("body_simplification"),
        "contacts_plan": {k: {kk: vv for kk, vv in (v or {}).items() if kk != "regions"}
                          for k, v in (plan.get("contacts") or {}).items()},
        "verdict": judgement.strip().splitlines()[0] if judgement.strip() else None,
        "agreement": drive.agreement(history),
        "size_text": sections.get("size and material", "")[:1500],
        "use_text": sections.get("use", "")[:2500],
    }
    (ROOT / "drafts").mkdir(parents=True, exist_ok=True)
    lib.write_json(ROOT / "drafts" / f"{session.name}.json", draft)
    return draft


# --------------------------------------------------------------------- the run --

def _replace_contacts_json(section: str, contacts: dict) -> str:
    """The Contacts section with its JSON object replaced by the v2 one (appended when none)."""
    for value in drive._json_objects(section):
        if drive.contacts_v2_in(json.dumps(value)):
            start = section.find("{")
            while start >= 0:
                try:
                    found, end = json.JSONDecoder().raw_decode(section, start)
                except ValueError:
                    start = section.find("{", start + 1)
                    continue
                if found == value:
                    return section[:start] + json.dumps(contacts) + section[end:]
                start = section.find("{", end)
    return section.rstrip("\n") + "\n" + json.dumps(contacts) + "\n"


def patch_brief(text: str, fixture: dict) -> str:
    """The brief with the fixture's JSON lines in place of the old ones."""
    out, heading = [], None
    blocks = re.split(r"(^#+[^\n]*\n)", text, flags=re.M)
    for block in blocks:
        if re.match(r"^#+", block):
            heading = block.strip("# \n").lower()
            out.append(block)
            continue
        if heading == "use":
            block = re.sub(r"(?m)^\s*Anchor:\s*\{.*$", "", block)
            block = block.rstrip("\n") + "\nUse table: " + json.dumps(fixture["use_table"]) + "\n\n"
        elif heading == "contacts" and fixture.get("contacts"):
            block = _replace_contacts_json(block, fixture["contacts"])
        elif heading == "size and material":
            member = fixture.get("real_section_member")
            if member:
                block = re.sub(r"(?i)real section(?:\s*\([^()]*\))?:", f"Real section ({member}):", block, count=1)
            if fixture.get("real_extents"):
                a, b, c = fixture["real_extents"]
                block = re.sub(r"(?im)^real extents:.*\n?", "", block)
                block = block.rstrip("\n") + f"\nReal extents: {a} x {b} x {c} m\n\n"
            if fixture.get("load_path"):
                block = re.sub(r"(?im)^load path:\s*\{.*\}\s*$", "", block)
                block = block.rstrip("\n") + "\nLoad path: " + json.dumps(fixture["load_path"]) + "\n\n"
        out.append(block)
    return "".join(out)


def states_table(table: dict, operator: Path) -> dict:
    """A fixture's Use table of cases as a table of states: each row's "state" and "kind" (the
    stored anchor's for its label when the row gives none), "run" the state of the run case's rows
    (every action of a state acts together), "cases" and the per-member "unit" dropped."""
    table = json.loads(json.dumps(table))
    cases = table.pop("cases", None) or {}
    table.pop("unit", None)
    actions = stored_actions(operator.parent)
    for label, row in (table.get("actions") or {}).items():
        if isinstance(row, dict) and label in actions:
            row.setdefault("state", actions[label]["state"])
            row.setdefault("kind", actions[label]["kind"])
    run_rows = cases.get(table.get("run")) or []
    states = [drive._row_state(a, table["actions"].get(a)) for a in run_rows]
    table["run"] = next((st for st in states if st), table.get("run") if table.get("run") in drive.USE_STATES
                        else "working")
    return table


GONE_BODY = ("route", "about", "load_path", "cut_across", "at_m", "piece", "absent_in_stand_in")
GONE_CONTACT = ("fastened", "inertia_relief", "beside_load")


def migrate_plan(plan: dict, fixture: dict) -> dict:
    """The final plan with the deleted fields dropped and the fixture's patch merged on top."""
    plan = json.loads(json.dumps(plan))
    plan.pop("free_form_body", None)
    body = plan.get("body_simplification")
    if isinstance(body, dict):
        for key in GONE_BODY:
            body.pop(key, None)
    for entry in (plan.get("contacts") or {}).values():
        for key in GONE_CONTACT:
            (entry or {}).pop(key, None)
        if isinstance((entry or {}).get("absent_in_stand_in"), dict):
            entry["absent_in_stand_in"].pop("why", None)
    loads = plan.get("loads")
    if isinstance(loads, dict):
        plan["loads"] = {k: v for k, v in loads.items() if isinstance(v, dict) and "dropped" not in v}
    return drive.merge_patch(plan, fixture.get("plan_patch") or {})


def _gate(name: str, fn, *args, **kwargs):
    try:
        out = fn(*args, **kwargs)
    except Exception as exc:  # a gate that crashed on old data is reported, never hidden
        return {"gate": name, "error": f"{type(exc).__name__}: {exc}"[:600]}
    if isinstance(out, list):
        return {"gate": name, "refused": "; ".join(out) if out else None}
    if isinstance(out, tuple):
        return {"gate": name, "refused": out[0], "info": out[1]}
    return {"gate": name, "refused": out}


def run(fixture_path: Path) -> dict:
    import inspect_session
    fixture = json.loads(fixture_path.read_text())
    session = ROOT / fixture["sid"]
    operator = session / "operator"
    # every run starts from the archived brief and plan, kept the first time
    for name in ("brief.md", "plan.json", "declared.json"):
        keep = session / "replay_original" / name
        if not keep.exists() and (operator / name).exists():
            keep.parent.mkdir(exist_ok=True)
            shutil.copy(operator / name, keep)
    keep_first = session / "replay_original" / "brief.first.md"
    if not keep_first.exists() and (session / "brief_snapshots" / "brief.first.md").exists():
        shutil.copy(session / "brief_snapshots" / "brief.first.md", keep_first)
    fixture = {**fixture, "use_table": states_table(fixture["use_table"], operator)}
    (operator / "brief.md").write_text(patch_brief((session / "replay_original" / "brief.md").read_text(), fixture))
    if keep_first.exists():
        (session / "brief_snapshots" / "brief.first.md").write_text(patch_brief(keep_first.read_text(), fixture))
    plan = migrate_plan(lib.read_json(session / "replay_original" / "plan.json", {}) or {}, fixture)
    lib.write_json(operator / "plan.json", plan)
    declared = lib.read_json(session / "replay_original" / "declared.json", {}) or {}
    declared = {**declared, **{k: plan.get(k) for k in ("contacts", "prediction", "body_simplification", "loads",
                                                        "geometry", "scale", "settings", "values", "material")}}
    lib.write_json(operator / "declared.json", declared)
    for stale in ("planned_resolution.json", drive.COVERAGE, drive.TIPPING_SAID, drive.PLAUSIBILITY_SAID):
        (operator / stale).unlink(missing_ok=True)
    summary = inspect_session.summarize(session, solve=False)
    regions = summary.get("regions") or []
    bound = drive.bind_by_name(plan, regions)
    rows = drive.held_regions(session, (bound.get("values") or {}).get("dirichlet_values") or {}, bound.get("contacts"))
    planned = drive.planned_load(bound, regions)
    judgement = (operator / "judgement.md").read_text() if (operator / "judgement.md").exists() else ""
    results = [
        _gate("G1 use_gate", drive.use_gate, operator),
        _gate("G2 hold_gate", drive.hold_gate, session, bound, rows, regions, planned),
        _gate("G2 path_gate", drive.path_gate, session, bound),
        _gate("G3 plausibility_gate", drive.plausibility_gate, session, bound),
        _gate("G4 body_gate", drive.body_gate, session, bound, None, final=True),
        _gate("G3 verdict_gate", drive.verdict_gate, judgement, session) if judgement else
        {"gate": "G3 verdict_gate", "refused": None, "note": "no judgement in the archive"},
    ]
    try:
        coverage = drive.use_coverage(operator)
        if coverage is None:
            # An archived brief answers only the states its agent considered, so G1 refuses the
            # rest as "no row" before comparing.  The comparison on the rows it does have says
            # whether coverage alone would also have refused the case run.
            real_table = drive.use_table
            drive.use_table = lambda op, brief=None: (lambda t, p: (t, [x for x in p if "has no row" not in x]))(
                *real_table(op, brief))
            try:
                coverage = drive.use_coverage(operator)
                if coverage is not None:
                    coverage = {**coverage, "on_given_rows": True,
                                "refuses": any(not u["declared"] for u in coverage["uncovered"].values())}
            finally:
                drive.use_table = real_table
    except Exception as exc:
        coverage = {"error": str(exc)}
    try:
        route = drive.body_route(session, bound)
    except Exception as exc:
        route = {"error": str(exc)}
    out = {"sid": fixture["sid"], "label": fixture.get("label"), "expect": fixture.get("expect"),
           "results": results, "coverage": coverage, "route": route,
           "plausibility": drive.plausibility(session, bound)}
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    lib.write_json(ROOT / "results" / f"{fixture['sid']}.json", out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("runs", nargs="+")
    sub.add_parser("run").add_argument("fixtures", nargs="+")
    args = parser.parse_args(argv)
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        for run_dir in args.runs:
            draft = prepare(Path(run_dir))
            print(draft["sid"], draft["run"], json.dumps(draft["actions"])[:300])
        return
    for path in args.fixtures:
        out = run(Path(path))
        print(f"== {out['label']} ({out['sid']}), expected: {out['expect']}")
        for r in out["results"]:
            said = r.get("error") or r.get("refused")
            print(f"   {r['gate']}: {'REFUSED: ' + str(said)[:400] if r.get('refused') else ('ERROR: ' + str(said)[:300] if r.get('error') else 'pass')}")


if __name__ == "__main__":
    main()
