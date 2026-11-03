# Driving the workflow: commands, cards and answers

Each command is one plain command from the repository root, typed in full (SKILL.md).

## 1. Driver commands

Global flags go **before** the subcommand, for example `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py --no-preload answer …`:
- `--output-root DIR`: default `output/fem_workflow_operator`; the UI's root is `output/fem_workflow`.
- `--gpu auto|N` and `--allow-busy-gpu`: the last two are user permission only.
- `--gpu-wait S` (default 1800), `--max-foreign-mib M` (1024), `--max-util P` (10).
- `--no-preload`: don't warm the next step's workers.
- `--prefetch last|all|none`: which field GLBs to export after a solve.

| Command | Does |
|---|---|
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py blank SID --shape box\|cylinder\|extrude [--size X Y Z] [--radius R --length L --axis x\|y\|z] [--profile 'x,y …' --length L --axis x\|y\|z [--hole 'x,y …'] [--circle 'cx,cy,r']]` (`--axis` required for a cylinder or an extrusion) | a watertight body in metres under `operator/`, remeshed into faces of about half a leaf; prints the plan's `geometry` and `scale` entries. `--hole` and `--circle` (repeatable): through-holes of the profile. Named by its dimensions, so a rebuilt body is a new upload |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py blank SID --bars FILE.json` | a frame of members in any direction as one watertight body: bars `{"from": [x,y,z], "to": [x,y,z], "section": [w, h] \| {"d": d}}` (a list, or `{"bars": [...], "boxes": [{"min": [x,y,z], "max": [x,y,z]}]}`), each end that meets another member a joint, filled by a hub. A bar under 2 leaves at the case's depth is built and warned (objectives/routes.md §3.4). Prints `built` (bars, boxes, joints, the smallest section, `sections_needed`; `squared_at_joints` when round bars were built square at a joint) and the plan entries; a joint that does not union is named with what to change. Separate pieces need `"pieces_supported": true`. About 5-15 s |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py blank SID --revolve 'r,z r,z …' [--wall T] [--axis x\|y\|z] [--bars FILE.json]` | a body of revolution: the (r, z) profile (r from the axis, z along `--axis`, default z) revolved, a closed outline or, with `--wall T`, an open line T thick (objectives/routes.md §3.4). Refuses separate pieces; a wall under 2 leaves is warned. Prints `built.revolved` and `moved_by_m`; `--bars` beside it adds members in the revolve's frame |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu [--stop-workers]` | per GPU: idle verdict, used, ours and foreign MiB, utilization, our workers. `--stop-workers` stops this root's resident workers |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py start --instruction "<the user's whole instruction, word for word>" [--provider deepseek\|groq\|gemini] [--model M] [--vision-provider P] [--vision-model M] [--compute-env ENV] [--octree-depth 7] [--session-id ID]` | probes the compute env (no API call), creates the session, stores the instruction, returns the first card. DeepSeek is the workflow's default |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py run SID --plan plan.json [--patch '<JSON>']` | answers cards from the plan (§6) until the next stop, printing `look_at`, the stop's block and `warnings`. Running again approves the stop; editing the plan first corrects it. `--patch` is an RFC 7396 merge patch (`null` removes a key) written into plan.json before the run; an invalid one spends nothing. Counts attempts against the limits; a failed step returns `stop: "error"` with the card restored |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py answer SID --json '<obj>' \| --file f.json [--dry-run] [--no-merge] [--force] [--no-lint]` | merges the answer with the card (`notes` shows what was kept), lints it, counts it against `run`'s limits (refusing past one), runs it under the GPU guard, and on failure reopens the card. `--no-merge` sends the answer as given: the workflow keeps the card's value for every omitted field, and the lint still checks the merged view |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py show SID` | the pending card, from the checkpoint, running nothing |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py status SID [--no-gpu]` | safe while a step runs: stage, running step, newest tool, solve progress, recent worker log lines, GPUs |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py history SID` | every card the session waited on, newest first, with checkpoint ids |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py rewind SID --kind <card> [--nth k] \| --checkpoint ID` | makes that earlier card current again (§3) |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py retry SID` | re-runs a pending step that has no card (a generation or segmentation that failed or was killed); never a solve the user stopped (§4) |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py solve-backend SID --backend pyamgx\|native\|jaxamg\|amgx [--max-iters N]` | re-runs `case/case.json` with another linear-solver backend (or JAX-AMG with `--max-iters 4000`), under the GPU guard, into `solve/`. Out of graph: end the session with `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py finish SID`. Prints `look_at`, `checks` and `region_displacement`, by which two backends are compared; trust pyamgx only if Newton converged and a `native` run agrees (known_issues.md #1) |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py judge SID --file j.md \| --text "…" [--finish]` | replaces the placeholder under "Engineer's judgement" in `operator/report.md`; `--finish` then ends the session from its settings card in the same command (as the last `run`, or `finish` after `solve-backend`), only when the judgement was accepted |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py finish SID` | ends the session from its settings card whatever the last graph solve did: the way to close a session whose accepted result came from `solve-backend` (`run` finishes only after a successful graph solve) |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py summary SID [--yield-strength S]` | every metric and warning, as JSON |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py render SID --what regions\|segments\|geometry\|field\|material\|design\|all [--field displacement\|stress\|strain --component magnitude\|x\|y\|z\|von_mises\|xx… --deform auto\|0\|k]` | PNGs in `<session>/operator/` |
| `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py report SID [--yield-strength S]` | `<session>/operator/report.md` |

**`blank --axis`** (required for a cylinder or an extrusion, no default): the axis its `--length` runs along;
the section lies on the other two axes in x, y, z order (`--axis z`: profile x,y; `y`: x,z; `x`:
y,z), with the body's lowest corner at the origin. A box takes `--size X Y Z` as given, with no
axis. The output's `axis` repeats the convention it was built with.

**What an answer prints:** one JSON object: `ok`, `session_id`, `checkpoint_id`, `stage`, `kind`
(the next card's), `card` (`solve_log` trimmed), `complete`, `notes` (merged fields), `warnings`,
`gpu` (the card's index, peak and foreign memory), `elapsed_s`, `field_assets` after a solve,
and on failure `error`, `restored`, `traceback_tail`. The full event is saved as
`<session>/operator/last_event.json`; every step is logged to `operator/driver_log.jsonl`, every
GPU step to `operator/gpu_log.jsonl`.

**Long steps:** run with the Bash tool's `run_in_background: true` (the result is saved in
`operator/last_run.json` or `last_event.json`) and poll `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py status SID`. Never start a
second `answer` on the same session; the lock refuses it.
