# Driving the workflow: rewinds, failures, budget (cli_protocol §3–§5)

## 3. Rewind and history

```bash
python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py history SID                       # cards, newest first
python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py rewind SID --kind bc_values       # the most recent bc_values card becomes current
python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py rewind SID --kind bc_input --nth 2
```
- Rewinding discards the *state* after that card (values, material, settings), not the files
  on disk. Re-answer everything after it; the material and settings cards come back empty when
  you rewind before them.

## 4. Failures and rollback

When a step raises (a lint miss, a tool failure, a graph validation error), the driver reopens
the same card clean from the checkpoint the answer was sent from (`graph.reopen_card`): fix the
answer and send it again.

**If the driver process is killed mid-step** (a timeout, or the user stops it), the step is left
pending with no card; `show` says so:
- a stopped generation or segmentation you still want: `retry`;
- a stopped solve (`rewind --kind shape_opt_settings`), or anything whose settings you no longer
  want: `rewind --kind <its card>`, then answer again. `retry` would rerun it as it was.

Failures on the card rather than as exceptions: `api_error` (an LLM call failed after its
retries), `preview_error` (a selector resolved empty), `solve_error` (the solve failed). The
step returns to the card; read the message and act.

## 5. Budget

**API calls:**
- A run needs about 4–6: the geometry overview, the segment description, and BC synthesis
  (vision plus text).
- Each `revise_prompt`, `infer_*` or repeat preview with new text costs one more.
- A failed call is not cached; a successful one is (identical inputs), so repeating it is free.

Model probes at `start` are not counted. The count is in `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py summary`
(`session.api_calls`).
