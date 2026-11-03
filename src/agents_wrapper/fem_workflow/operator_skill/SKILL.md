---
name: fem-workflow-operator
description: Run a finite-element simulation of any object end to end, autonomously, in about five minutes, with the API-assisted FEM workflow's command line (src/agents_wrapper/fem_workflow). From a name alone, a description, an image or a mesh it decides a meaningful case (supports, loads, size, material), builds or generates the geometry, checks every picture like a mechanical engineer, solves on an idle GPU, and reports the result with its evidence. Use it whenever asked to simulate, analyse, stress-test, run FEA on, or demo the workflow with an object. It is not the prompt pack under fem_workflow/skill/, which belongs to the workflow's own LLM specialists.
---

# FEM workflow operator

You turn an objective into a solved, checked, reported simulation in five to fifteen minutes. A
plan file holds your decisions; the runner `drive.py run` sends them to the workflow and stops
where you must look: the generated geometry, the boundary-condition regions, the solved fields.
There you decide as a mechanical engineer would.

Paths below are relative to this folder (`src/agents_wrapper/fem_workflow/operator_skill/`).
Every command is one plain command from the repository root, typed in full with literal
arguments as written here (no variables, `$(…)`, `;`, `&&`, pipes or redirects). Each runs
`python3 -u …` (any Python 3.8 or newer): the script hands itself to the `mechfoundry-agents`
environment's own interpreter, through conda only when it cannot find it; if `drive.py start`
says neither was found, stop and say so. Bash runs only these `drive.py` and `inspect_session.py`
commands; read and search files with the Read, Grep and Glob tools (any other shell command is refused).

**Reading.** This file is the normal path. `drive.py start` and every stop print a `read`
field: the reference sections that apply from there, each named once as `file:a-b why` (offset a,
limit b − a + 1; a file alone is read whole), and relisted as `file:a-b` at each stop of that
kind until you run on. Read each with the Read tool when named, before you decide; nothing else
unless a rule here names it, and nothing already in the conversation. Never cat, wc, head or tail
through Bash (refused); Edit, not Bash, changes the brief. After a context summary, re-read the
sections in `operator/shown.json` your next decision needs.

## Rules

1. **Reason, do not ask and do not look up.** Never ask the user anything the instruction did
   not tell you to ask. Reason whatever the user did not give from how the real object is used,
   and write it in the case brief (references/objectives/brief.md §2). Say in the report what
   you decided.
2. **Look at every picture the runner names** with the file reader, and decide from
   references/visual_checks.md. Never approve regions or report a result you have not seen.
3. **Stay inside two folders.** Read this skill folder, your session folder
   (`output/fem_workflow_operator/<session_id>/`) and any file the user named. Write only under
   `<session>/operator/`. Do not open the workflow or solver source, other sessions, or the UI's
   root; edit no code. Drive the workflow through `drive.py` only.
4. **Respect the limits, then stop.** The runner counts every attempt and refuses past the limit.
   At a limit, or an error no plan change can fix, write the failure report and stop. Do not
   investigate the workflow: quote the message and the receipt path it names.
5. **GPU steps run on an idle GPU.** The runner picks one and waits; a `waiting_for_idle_gpu`
   line on stderr needs nothing from you. `--gpu N` or `--allow-busy-gpu` only when the user
   allowed it.
6. **Keep the defaults** (PartField, depth 7, mixed precision, a forward solve); change one field
   at a time, for a reason the brief states.
7. **Report what happened**, with the pictures and the numbers: a decided value is called
   decided, a caveat is stated, a failure is a failure.

## The procedure

**1. Start the session and write the case brief.** `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py start --instruction "<the user's whole instruction, word for word>"` makes a
new session and prints its `session_id` and the `read` for the brief and the plan; use that id
only. Write `output/fem_workflow_operator/SID/operator/brief.md` and `plan.json` (paths from the
repository root) with Write, one each; change them with Edit, a whole-file Write or `run --patch`
(once used, keep to `--patch` or Read plan.json first), never a heredoc or `sed`. The brief has
four headings (objectives/brief.md §2): Use (its `Use table:`, a row per action of the use's
states), Contacts (its JSON with each contact's kind), Load, Size and material (its `Real …`,
`Load path` and `Materials by part` lines); a Prediction in words is optional. You state the loads and contacts; the
runner computes the effects, coverage, levers, reactions, stress and prediction. It gates them
before anything is spent: G1 the Use table and its coverage, G2 the load path and holds, G3 the
plausibility; G4 the body (objectives/routes.md §3.4). The question is the deflection and the
stress pattern unless the user asked otherwise; a safety factor only when asked.

**2. Write the plan from the brief** at `output/fem_workflow_operator/SID/operator/plan.json`
(every field: references/cli_protocol/plan.md §6). The body follows the routes the brief narrowed
to (`plan.geometry_routes`, objectives/routes.md §3.4). A generated body comes first: one TripoSG
text attempt (about 20 words), then an SDXL drawing (`"draw": true`), TripoSG from it, Hunyuan3D
from that drawing; joined prismatic parts → CAD-Coder with `"revise_prompt": true`; then a web
photograph (`"web_image"`). A thin member is warned, never refused. `blank` (built) is the fallback
after 2 generated bodies (or every listed generated route) and the photograph (`blank` searches
when none was tried): one section →
`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py blank SID --shape box|cylinder|extrude` with `--size X Y Z`, `--radius R --length L` or
`--profile 'x,y …' --length L` (`--axis` is required for a cylinder or an extrusion: the profile's depth,
references/cli_protocol/commands.md), which prints the `geometry` and `scale` entries; `blank` comes after
the brief, which it refuses without; members in several planes → `blank SID --bars FILE.json`;
a body of revolution → `blank SID --revolve 'r,z …' [--wall T] --axis y`.

The plan skeleton (objectives §4), the brief in JSON:

```json
{"geometry": {"description": "<from §3.4>", "generation_settings": {"seed": 0}},  // + generator, revise_prompt, draw, image_seed, drawing_crop, asset_path, asset_kind
 "geometry_routes": ["triposg_text", "drawing", "hunyuan3d"],  // generated routes first, built last (§3.4)
 "texture": {"texture_prompt": "<colour> <material>", "texture_seed": 0},  // optional: one mid or light colour, ≤ 8 words (plan.md)
 "segmentation": {"method": "partfield", "num_clusters": 5}, "region_revisions": {},  // when needed
 "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "<y or longest>", "physical_extent": <metres>},
 "regions": "Fix <each held contact, as §3.3>. Apply a traction in <direction> on <each load face, as §3.3>.",
 "values": {"dirichlet_values": {"<support region>": {"components": [<held>], "value": 0.0}},
            "acceleration": [0, -9.80665, 0]},   // the loads are the Use table's sizes along its directions
 "material": {"youngs_modulus": <Pa>, "poisson_ratio": <nu>, "density": <kg/m3>},
 "contacts": {"support A": {"transmits": "force only", "regions": ["<held region>"]},   // a rest: dirichlet_values on its normal component(s)
              "support B": {"transmits": "force and moment", "regions": ["<held region>"], "pieces": 2, "pieces_why": "<the brief's reason>"},  // "pieces": separate real patches
              "support D": {"transmits": "force only", "regions": ["<its region on the stand-in>"], "absent_in_stand_in": {"carried_by": "load_at_joint"}}},  // its member left out of a stand-in (routes.md §3.4)
 "loads": {"A1": {"regions": ["<its load region>"]}, "A2": {"acceleration": true}},  // each row of the run state, applied at its size
 "body_simplification": {"absent": ["<member>"]},  // others kept; the route is the runner's (routes.md §3.4)
 "entry_members": {"<member>": {"present": "<its region or segment>"}},  // optional, or "absent"
 "bonded_gaps": "<the parts joined across a gap under a leaf, and why>", "settings": {}}  // bonded_gaps only when thin_gaps stops
```

- `regions` states the contacts first, one clause per region, then each load face (§3.3:
  objectives/regions.md); the words are sent when you run on from the geometry stop.
- Region names exist only after the preview: write `contacts`' `regions` (keyed by the brief's
  contact names exactly), `dirichlet_values` and `loads` at the regions stop from the names it
  prints. Each bound row is applied at its size along its `direction`; a support without an
  entry is held in all components.

**3. Run to each stop, look, decide.** The same command each time:

```bash
python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py run SID --plan output/fem_workflow_operator/SID/operator/plan.json
```

At each stop read every `look_at` picture and every section its `read` names in one message
(parallel reads), then run with the one change as `--patch '<JSON merge patch>'` (written into
plan.json; refused, unchanged, when invalid; an apostrophe is `'`). The stop prints a compact
view (four significant figures; a `note`, `why` or `next` printed before reads "as before"; a key
left out is empty); `operator/last_run.json` has every number.

| Stop | Read | Decide (visual_checks.md: the lines its `read` names) |
|---|---|---|
| `revision` (CAD-Coder); `drawing` (`"draw": true`); `web_image`; `texture` (`plan.texture`) | `revised_description`; the drawing and `drawing`; the photographs and `web_image`; the painted views and `texture` | the revision: every part, dimension ratio and hole of the brief; a drawing or photograph: recognisably the whole object, only one, its primary members; the painting: the look the figure will show → run again (a photograph: `"pick"`). Otherwise edit `geometry.description` (or `image_seed`, `plan.texture`) with `geometry.reject` and run again |
| `geometry` | `geometry/views.png`; the geometry block; `planned_resolution` (extents, `thinnest_member_leaves`, `resolution_check`, `load_path`, `mass_over_real`, `member_over_real`, `g4`, `thin_gaps`, `real_member`; a repeat stop prints what changed); `cad_faces`; `brief` | recognisably the object asked for, one body, standing as in use, at the brief's Real extents, its primary members present (another missing: declared) → run again; a "thin member" warning (under 2 leaves) is reported, never refused: keep it or take depth 8; the stop's `error` is refused on leaving it. Otherwise edit `geometry` (with `geometry.reject`, visual_checks.md §1) or `scale.up_axis` and run again |
| `regions` | `operator/regions.png`, `operator/segments.png`; `planned_load`, `planned_supports`, `held_regions`, `synthesized_selectors`, `reactions_from_statics`, `coverage` | each support on its contact patch, held in the brief's components; each load on the brief's face with its total; no purple → write `dirichlet_values`, each contact's `regions` and `loads`, and run again (that approves); every problem is named at once. Otherwise `segmentation.num_clusters` first, then `regions`, `region_revisions` or `values`; run again |
| `materials` (`"mode": "parts"`, or an inferred uniform `material`) | `materials/parts/montage.png`; `parts`, `parts_caveats`, `soft_parts`, `segments_by_material`; or `material` | each part the material the brief names → run again; a wrong value: add the part to `material.parts` |
| `solve` | `operator/displacement_magnitude.png`, `operator/stress_von_mises.png`; the solve block, `report`, `coverage` | supports at zero, the mode and peak where predicted, Newton converged → write the judgement. `prediction_check.agreement` is the runner's: 2–5 off is printed "disagrees"; over 5, fixed and solved again or stated in the verdict line (objectives/judging.md §5). A `solve_error`: the one change troubleshooting.md gives for that message |
| `finished` | | |

Editing a plan section after its card was answered makes the next `run` rewind to that card and
re-send from there. `stop: "error"`: a step was refused or failed, the card restored; fix the value
it names. `stop: "limit"`: write the failure report. Every stop prints `attempts_left`. An
unchanged resend of a generation is served from the cache, uncounted. A `run` whose output you
lost is not re-run (running again approves the stop): read `operator/last_run.json`.

Anything the plan does not cover (a what-if, another load case, a shape optimization) goes through
`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py answer` and `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py rewind` (references/cli_protocol/cards.md, recovery.md), with the same looks
and limits; a shape optimization reads solver_settings.md §4 first, and visual_checks.md §4 and
judging_results.md §7 at its solve stop. A linear solve that "ran out of iterations" or "stalled":
the runner takes the fallback it names, as an ordinary solve stop; settings (troubleshooting.md)
only once no backend converged. A `solve-backend` result is closed by `judge --finish` or
`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py finish SID`, never `run`.

**4. Deliver.** The `solve` stop wrote `operator/report.md`. Write `operator/judgement.md`, then put
it in and finish the session in one command:
`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py judge SID --file output/fem_workflow_operator/SID/operator/judgement.md --finish`.
`judge` refuses a judgement without a verdict first line or a "Deviations" heading; a verdict the
runner's records decide otherwise (G4, its simplifications) it replaces with its own line. The runner writes into the
report the agreement, each contact as declared beside its hold, the coverage and every
simplification it finds. The judgement contains, in order:
- the verdict (objectives/judging.md §5): "the brief's case was run"; "a variant of the brief's case was
  run"; "a simplified case was run: …"; "the brief's case could not be run"; or "the governing
  load cannot be applied; a secondary case was run: …";
- "Deviations": "none", or each deviation and its effect (not caveats; a member left out or a hold moved is one);
- the case as run, every decided value marked decided: object, size, supports, loads (the applied
  totals from `loads`), material (per-part: the report's line), settings, and whether the body was built or generated;
- the results with units: the peak displacement and where; the stress away from the held and loaded
  patches and the peak as mesh-dependent; Newton's convergence; no safety factor unless asked;
- the runner's prediction against the result, and what a disagreement comes from;
  the mass ratio to the real object;
- each contact's held and released components and what the holding overstates;
- caveats: the resolution (a thin member the runner reports), the singular peak at held edges,
  known issues that touched the run;
- the session id, the report path, and the two field pictures.

## Limits and time

A limit is the total number of attempts of that stage in the session, first tries included,
whichever command sent them. A route's limit is a ceiling, not a target: move to the next listed
route when no change of words or seed addresses what the stop showed; the last geometry or preview
is never approved because no attempt is left.

| Stage | Limit | Past the limit |
|---|---|---|
| TripoSG text generations | 1, a second only with a new seed when it returned no watertight surface | the next listed route (the runner names it) |
| SDXL drawings | 3 | the next listed route; never a body from a drawing that failed its checks: a fresh drawing made for Hunyuan3D is Hunyuan3D's attempt |
| Hunyuan3D attempts: each drawing it is given (taken over from the drawing route, or drawn for it), rejected or not (geometry.md) | 2 | the next listed route |
| web photograph searches | 1 (2 when the first found nothing); 1 body | `blank` |
| CAD generations | 3 | build it if it reduces to one section; otherwise the next listed route |
| every listed route tried | | an untried route added to the list with the brief's reason, or the simplified body (objectives/routes.md §3.4); never an earlier attempt re-run unchanged |
| built or uploaded bodies | 3 | report: the body could not be built |
| region previews | 4 | report: the regions could not be placed, with `regions.png` |
| region revisions | 4 | as for previews |
| re-segmentations (the same `regions` words on a new `segmentation`; the first valuation of its parts is free) | 3 | as for previews |
| part valuations (`mode: parts`) | 3 | uniform, the load path's material, said so |
| solves, `solve-backend` included | 5 | report: the solve failed, with `solve_error` |
| per retry | one change | the words *or* the seed, the regions *or* the segmentation, one setting |

Warm, a run is about one minute of compute. Run the driver in the foreground; only a shape
optimization goes in the background, polled with `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py status SID`.

**Failure report** (at a limit, or an error the plan cannot fix): the session id, the stage,
each attempt's plan value and what the picture or message showed, the last error verbatim with
its receipt path, and what a person could try next. Then stop.

## References

Each is read when a `read` field or a rule names it, the section named only.

| File | Holds |
|---|---|
| references/objectives/ brief.md, routes.md, size.md, regions.md, judging.md | §1–§2 the brief; §3.4 routes; §3.1–§3.2 size; §3.3 regions; §5 caveat or deviation |
| references/cli_protocol/ plan.md, commands.md, cards.md, recovery.md | §6 every plan field; §1–§5 commands, card answers, rewinds, failures |
| references/visual_checks.md | what each picture must show, and the decision |
| references/boundary_conditions.md | segmentation, region wording, the region checks, values |
| references/troubleshooting.md | an error message or a preview symptom → the one change to make |
| references/geometry.md | what each route controls, generator mechanics, judging and scaling geometry |
| references/materials.md | the material table; per-part and VoMP materials |
| references/solver_settings.md | every setting; mesh convergence; shape optimization |
| references/judging_results.md | the result checks; a safety factor when asked; presenting |
| references/cae_practice.md | Human actions and engineering practice |
| references/concepts.md | what happens behind each card; units; resolution |
| references/gpu_usage.md | the GPU rule, VRAM per step, workers |
| references/known_issues.md | workflow limits to mention in the report |
