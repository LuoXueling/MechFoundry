# Illustrations

Paper figures. Each figure has its own subfolder of scripts and is built by
`build.sh <target>`; only the final `NAME.pdf` / `NAME.png` sit in this folder. The
toolchain, house style (Arial everywhere) and per-figure notes are in
[`AGENTS.md`](AGENTS.md). This README documents one family in full: the **agentic
results** figures, which show what an LLM agent produced when it drove the FEM workflow
from a one-line instruction.

**The batch folders are archived.** `output/agentic_batch50` (Sonnet), `…_haiku`,
`…_opus` and `output/showcase` are packed in `output/archives/<name>.tar.zst`. Every build
below that renders runs reads them at their original paths, so extract the ones it needs
first: `zstd -dc output/archives/<name>.tar.zst | tar -x -C output` (→ `output/<name>/`).

---

## 1. The agentic-results figure

`./build.sh agenticresults` → `agentic_results.{pdf,png}` (or `$AR_NAME.{pdf,png}`).
One row per agent run, 180 mm wide (Nature double column), as tall as its rows. Every
number on it is read from `agentic_results/ar_stats.json`, which the prep writes (§6).

**The main-paper figure** is built from `agentic_results/ar_paper_runs.json`, six Opus runs
of `output/agentic_batch50_opus`, in this order: electric toothbrush, park bench, car
suspension control arm with rubber bushings, truck leaf spring, claw hammer, human femur
(the user's set of 2026-10-16, replacing the earlier eight: computer mouse, sofa, pogo
stick, slice of watermelon, industrial robot arm, car brake disc, human femur, leaning
tower of Pisa; on 2026-10-19 row 3, the wooden rocking horse, became the car brake disc and
then the control arm, drawn at `draw_scale` 0.9). The control arm's
prompt would take three lines at the card's 7 pt and make every row taller (rows 26.9 mm,
page 190.4 mm); its entry gives `"prompt_max_lines": 2` instead (below), so the figure
keeps rows of 23.9 mm and a 172.4 mm page. Each row is its
run's primary session (`metrics.json` `session_id`), and every one completed a forward
solve. It has the four panel columns below, no segmentation column, and
passes the strict compose with every camera the chooser's own pick.

**Per-row prompt size (`prompt_max_lines`).** A runs-JSON entry's `prompt_max_lines` n
(1–3, default none) holds that row's prompt card to at most n lines: `ar_figure.prompt_fit`
sets it at the largest font size ≤ the card's 7 pt, in 0.1 pt steps, at which
`balanced_wrap` fits it into n lines, and never under 5 pt (the build fails instead). Its
line pitch scales with the size (3.0 mm × fs / 7), and the block's ink centre stays where
a 7 pt prompt of as many lines would sit, so the card and every row keep their heights
(the card height follows the figure's longest prompt in lines, after this cap). The user's
call, 2026-10-19, for the control arm ("use smaller font size for this box and don't
change the figure size"): n = 2, which gives 5.8 pt. The prep records each row's `prompt`
(`max_lines`, `fs`, `lines`, the rule) in `ar_stats.json`, and the compose asserts that it
draws exactly that.

**Per-object size override (`draw_scale`).** The control arm's entry gives `"draw_scale":
0.9` (the user, 2026-10-19: "not necessarily the same sizing, try 0.9x"). The override
was made for the car brake disc (2026-10-19, since replaced by the control arm): fitted
to its panel by the rule below, the flat disc was drawn too large beside its neighbours, so
the user had it drawn at 0.8, then 0.75 "further smaller but not moved downwards", with its
"SDXL → Hunyuan3D" label back in the lower right. A runs-JSON
entry's `draw_scale` f (0 < f ≤ 1, default 1) draws that row at f × the size the sizing
rule (`ar_figure.row_crop`: the row's union ink box fitted into the panel) gives it, in
every panel of the row. It is the only per-object size knob; the rule itself is unchanged
for every other row (f = 1 is the rule exactly). How it is applied: `render_ar.fit` widens
the view angle after the camera fit so the body projects exactly f × as large (tan of the
half angle ÷ f; same camera position, so the perspective is unchanged, and the frame keeps
the same room for the ground and its fade); `ar_figure.row_crop` divides the crop's px/mm
by f and shrinks the drawing about the centre of the row's ink box *as the rule places
it*, horizontally and vertically, so the body keeps its place in the row and does not
sink (an earlier version kept the ink bottom on the usual line: the shrunken disc sank,
its body then filled the lower-right label band and the route label was pushed to the
top right). Shrinking about the centre frees the bottom label band too: under it the brake disc's
route label passed the lower-right corner test (`pick_corner`) at every f ≤ 1 (1.65 mm of
air above the band at 0.8, 2.04 mm at 0.75), so the label did not set the value; 0.75 was
the user's "further smaller" step. `ar_prep.preview_print` calls the same
`row_crop`, so the camera chooser predicts the print checks (BC floors, corner labels) on
the drawing as it will be. The grid-cell label stays true, since the grid is drawn in
the body's own lengths and shrinks with it; ×N and the body load are physical and
unchanged. The value is recorded per row as `draw_scale` in `ar_stats.json`, and the
compose prints each row's object ink (mm) so the shrink can be checked:

```shell
cd illustrations && AR_RUNS_JSON=ar_paper_runs.json ./build.sh agenticresults
```

### Columns

| Column | Head | What it shows | Source |
|---|---|---|---|
| Instruction | INSTRUCTION | The instruction in a prompt card (italic, in quotes), then three cost bars | the run's `metrics.json` |
| Generated shape | AGENT · GENERATED SHAPE | The body the workflow analysed, in its own colours | session `geometry/physical.glb`, coloured from `geometry/texture/textured.glb` (or `materials/texture/textured.glb`), else the generator's own colours (`geometry/generated.glb`), else plain clay |
| Supports & loads | AGENT · SUPPORTS & LOADS | Supports teal, loaded faces orange, load arrows along each traction | `boundary_conditions/resolved.npz` + `resolved.json`, tractions from `case/case.json` |
| Young's modulus | AGENT · YOUNG'S MODULUS | E on the surface, on one log colour scale for the whole figure | the solve's own volume output (`solve/<run>/vtu/solution_t*.vtu`, field `youngs_modulus`), else `case/material.json` (uniform), else the per-part / VoMP material grid |
| Deformed shape | FEM SOLVER · DEFORMED SHAPE | The magnified displacement, coloured by \|u\| | the same VTU's `displacement` |

**Segmentation column** (`AR_SEGMENTATION=1`; the ten Opus per-category figures, §4). A
fifth column, AGENT · SEGMENTATION, between the generated shape and the supports and loads:
the parts the BC selectors were resolved against, on the same `physical.glb` triangles
under the row's camera. Source: the session's `boundary_conditions/segments.npz`
(`segment_ids`, one label per face), which `resolve-bcs` reads for every method — PartField
(cluster labels, every Opus run), shape diameter, or CAD B-rep faces (`segments.json`
`method`). It is used only when its face count matches `physical.glb` and the surface
sha256 in `segments.source.json` matches the file; otherwise the panel reads `not
available` and the prep records why (`segmentation.why`). No Opus run lacks one.

- **Colours: the PartField bunny's** (`partfield_segmentation`, `pfbunny/pf_prep.py`, also
  the overview's segmentation panel): the eight pastel `ov_style.SEGMENT_TINTS`, assigned
  by `overview/ov_palette_check.segment_tints` (every assignment tried, the closest touching
  pair of parts as far apart as possible), imported, not copied. A run with more parts
  than tints (the knee implant: 14) uses `ar_prep.greedy_segment_tints`, the same
  objective solved greedily, tints reused only between parts that do not touch. The
  record (`segmentation` in `ar_stats_<name>.json`) holds the method, parts, the tint per
  part, the assignment rule and the closest touching pair's ΔE00.
- **Key**: the eight tints as a strip, "One colour per part", and "Parts: PartField" (the
  methods that made the figure's parts).
- **Layout**: five 29.0 mm panels instead of four of 36.5 mm in the same 180 mm. The heads
  no longer fit tag-beside-title between the arrows ("FEM SOLVER DEFORMED SHAPE" is 32 mm),
  so the tag stands over its title (head band 7.9 mm instead of 6.2); the ×N key wraps to
  two lines (footer +3 mm). Every row's ink ends above the bottom label band (`INK_ROOM`
  3.0 mm instead of `SH_ROOM` 1.7 mm, showcase's rule): in a 29 mm panel no camera left a
  corner beside a centred body for "Body load …" (the femur, the cane). The grid-cell
  label moves to the segmentation panel's lower left (`CELL_PANEL`), since beside the route
  it ran into it ("□ 2 mmSDXL → Hunyuan3D"). The chooser predicts all of this through
  `print_layout()`. The four-column figures are unchanged.

The pills say who does the step: purple **AGENT** for the steps the LLM agent decides,
green **FEM SOLVER** for the GPU solve. Light arrows between the heads give the order.
All four panels of a row are the same triangles (`physical.glb`) under the same camera,
moved from the octree to the surface by the production `core.remeshing.octree_hex_interpolate`.

**Cost bars** (under the prompt card), one scale for all rows of a figure, each relative
to the largest run in it:

1. `X input (P % cached)`: total input tokens = `input_tokens` + `cache_read_input_tokens`
   + `cache_creation_input_tokens`; P = cache reads / total. The light purple part of the
   bar is the cached share, the saturated part the rest (tokens written to the cache were
   processed in full and count as not cached).
2. `X output tokens`: `output_tokens`, a grey bar.
3. The wall clock (`wall_clock_s`), split by `ar_time.py` into four segments: **Agent**
   (the LLM reading, reasoning and writing: all time outside its runner commands),
   **Workflow steps** (drawing, 3-D generation, scaling, segmentation, BC synthesis,
   values and material, painting), **Solver** (the FEM solves, and the solver fallbacks
   and cross-checks after them) and **Runner overhead** (process start, GPU choice and
   idle-GPU waits, stop pictures, bookkeeping commands). The sections are measured from
   the Claude transcript's timestamps and the runner's `operator/driver_log.jsonl` and add
   up to the wall clock exactly. A run whose `status` is not ok gets ` · <status>` after
   its time.

Token counts come from `metrics.json` first; a field it lacks is summed from the Claude
transcript(s) in `claude_session/`, each assistant message counted once (`ar_runs.py`
records which source every value came from).

### Labels in the panels

- **Grid cell** (generated shape, lower left): a square in the floor grid's colour and the
  side of one grid cell, e.g. `□ 0.2 m`. The grid is the UI viewer's floor grid
  (`src/ui/app.py` `placeGround`): the cell is the 1/2/5 × 10^k length nearest to a fifth of
  the body's largest extent, printed in m, or in mm under 0.1 m. It lies under all four
  panels of the row.
- **Route** (generated shape, lower right; upper right when the body fills the lower right): where the body came from, from `session.json`:
  `Text → TripoSG`, `SDXL → TripoSG` / `SDXL → Hunyuan3D` (the workflow drew an image
  first), `Image → …`, `Scribble → …`, `CAD-Coder`, `Built (bars)` / `Built (revolve)` /
  … (the agent built the body itself; the word is the builder's kind), `Uploaded mesh`.
- **Body load** (supports & loads, lower right; upper right when the body fills the lower right): the force the case's body load applies,
  `Body load 36 N`. A weight (`loads.acceleration`) is mass × |a|, the mass being the
  enclosed volume of `physical.glb` × the case density (a spatial material: the field's
  mean density; no density: the agent's stated real mass in `operator/brief.md`). A
  `loads.body_force` is a force per volume, × the volume. Two significant figures (whole newtons from
  100 N), kN from 1000 N.
- **×N** (deformed shape, top corner): the displacement is drawn N times magnified. N is
  the 1/2/3/5 × 10^k value nearest (in log ratio) to 0.12 × the largest extent / |u|max,
  never below 1 (`AR_FIG_FRACTION`); a body standing on a downward-facing support steps
  down that ladder until the magnified shape sinks at most 2 % of its extent below its
  base (`AR_DIP_MAX`), since the solve has no contact. The thin grey outline is the
  undeformed silhouette.
- `not available`: the run did not reach that panel's stage (no body, no supports and
  loads, no E, or no completed solve). The row stays in the figure.

Supports a camera cannot see (feet soles) are drawn as a teal outline just off the face;
a load on a face turned away from the camera is laid over the body as a see-through
orange tint, its arrows nearly opaque.

### Keys (footer)

Under the instruction column: the cost bars' colours (Cached input, Not cached; Agent,
Workflow steps, Solver, Runner overhead). Under supports & loads: Support, Load (swatch
and arrow), and "Load on a far face, seen through" when a row has one. Under Young's
modulus: the colour bar with its decade ticks (all labelled when they fit, otherwise every
2nd, 3rd … decade with its unit) and "Log". Under the deformed shape: the |u| ramp from 0
to each row's own max |u|, "×N: Displacement magnified N times", and the undeformed
outline.

The E scale spans the figure's smallest to largest modulus in Pa, widened to half-decade
ends and to at least one decade.

---

## 2. Pipeline (`agentic_results/`)

| Step | Script | Env | Does |
|---|---|---|---|
| 0 | `ar_style.py` | python3 | Gate: the look (colours, light kit, E ramp, solve recipe) equals Figure 1's source (`overview/ov_style.py`, `render_ov_solve.py`), except the key light's angles: `KEY_LIGHT` puts it on the view axis at elevation 30° (Figure 1: 35° to the camera's left at 45°), so the faces looking at the viewer are lit whichever side each row's camera stands on (2026-10-22). `render_ar.py` renders with `AR_SURF_VIEW` / `AR_SOLVE_VIEW`; the gallery keeps Figure 1's kit |
| 1 | `ar_runs.py` | (imported) | Loads each run: `metrics.json` > Claude transcript, the session folder under `workflow/`, and `ar_time.split_run` |
| 2 | `ar_prep.py` | `mechfoundry-warp117`, small GPU use | Per run: the surface, colours, BC masks and arrows, the camera (`choose_view`), the surface interpolation of u and E, the ×N factor, ground and grid; writes `_work/ar_<i>_*.vtk` and `_work/ar_stats.json`. Calls `ar_ui_scale.py` in `mechfoundry-agents` for the UI's own deformation scale (kept for reference) |
| 3 | `render_ar.py` | `pvbatch` (EGL GPU) | One PNG per panel (`ar_<i>_<panel>.png`), the ground alone (`…_g.png`), the BC id pass and the see-through layers |
| 4 | `ar_figure.py` | `mechfoundry-warp117` | Crops, composes and checks the page; writes `NAME.{pdf,png}`, `NAME_tokens.{csv,md}`, `NAME_time.csv` |
| – | `ar_time.py` | (imported) | The wall-clock split (sections and the four bar groups) |
| – | `ar_categories.py` | python3 | Picks runs and builds one figure per category (§4) |
| – | `ar_compare.py` | python3 / `mechfoundry-warp117` | The model-comparison pages (§5) |

The compose fails on a clipped render, text under 5 pt, overlapping text, a prompt over
three lines, a cost line that wraps, a corner label off its panel, and (unless
`AR_FLOOR=warn`) a BC region or arrow under its print floor (0.5 mm² for a region, 0.3 mm²
for an arrow, an arrow printing under 40 % of its length, a corner label on the object)
or a region whose rendered colour is more than ΔE00 10 from its legend chip.

---

## 3. `build.sh` targets and environment variables

### `./build.sh agenticresults`

| Variable | Default | Meaning |
|---|---|---|
| `AR_RUNS` | – | Space-separated run folders (`output/<batch>/<run>/` with `workflow/`, `claude_session/`, `metrics.json`; or bare session folders), relative to the repo root or absolute |
| `AR_RUNS_JSON` | `ar_dev_runs.json` | A runs JSON, used when `AR_RUNS` is unset (entries may give `instruction`, `transcript`, `model`, `label`, `view_dir`, `draw_scale`, `prompt_max_lines`) |
| `AR_NAME` | `agentic_results` | Output name |
| `AR_GPU` | `1` | GPU for the prep and pvbatch |
| `AR_GROUND` | `grid` | Ground under each object: `grid` (the UI's floor grid), `shadow` (key-light shadow) or `none` |
| `AR_FLOOR` | strict | `warn` lists print-floor and region-colour misses instead of failing (for looking at a batch) |
| `AR_SEGMENTATION` | `0` | `1` adds the segmentation column (§1); read by the prep (chooser layout, `ar_<i>_seg.vtk`), the renderer (`ar_<i>_seg.png`) and the compose |
| `AR_KEEP_WORK` | `0` | `1` keeps `_work/` (the renders), so a layout change reruns only `ar_figure.py` |
| `AR_VIEW_TABLE` | unset | `1` prints every candidate camera's filters and score, and each print check (§7) |
| `AR_FIG_FRACTION` | `0.12` | The ×N rule's target displayed peak / extent |
| `AR_DIP_MAX` | `0.02` | How far a standing body may sink below its base, × extent |
| `AR_UI_ENV` | `mechfoundry-agents` | The env `ar_ui_scale.py` runs in |

The target copies `_work/ar_stats.json` to `agentic_results/ar_stats.json`.

### `./build.sh agenticcompare`

| Variable | Default | Meaning |
|---|---|---|
| `ARC_HAIKU`, `ARC_SONNET`, `ARC_OPUS` | required | The three batch roots (absolute, or relative to the repo root) |
| `ARC_NAME` | `agentic_results_compare` | Output name: `NAME_p<k>.{pdf,png}` |
| `ARC_ONLY` | all | Category slugs to include, e.g. `"household_items furniture"` |
| `ARC_ROWS` | `17` | Rows per page at most |
| `ARC_GPU` | `1` | GPU for the prep and pvbatch |
| `ARC_GROUND` | `grid` | As `AR_GROUND` |
| `ARC_KEEP_WORK` | `0` | `1` keeps `agentic_results/_cwork/` |
| `AR_FLOOR` | `warn` | Passed to the prep |

---

## 4. Run picking and per-category builds

`experiments/agentic_batch/objects_50.txt` lists fifty instructions under ten
`# <Category>` headings of five each. A batch root (`output/agentic_batch50`, and one per
model, `…_haiku`, `…_opus`) has one folder per run and `_batches/<id>/summary.csv` per
batch launch, with the columns `instruction`, `path`, `status` and the run's metrics.

**Picking** (`ar_categories.latest_runs`): batches are read in id (timestamp) order and,
for each instruction, the last row whose run folder exists wins, so an object rerun in a
later batch replaces its earlier run. Runs of every status are kept. The run's
`metrics.json` names its primary workflow session (`session_id`); `ar_runs.py` opens that
session under the run's `workflow/`.

**Per-category figures**:

```shell
cd illustrations/agentic_results
python3 ar_categories.py --batch-root output/agentic_batch50 --dry-run    # the run lists
python3 ar_categories.py --batch-root output/agentic_batch50                # Sonnet
python3 ar_categories.py --batch-root output/agentic_batch50_haiku --suffix _haiku
python3 ar_categories.py --batch-root output/agentic_batch50_opus  --suffix _opus --segmentation
```

`--segmentation` (`AR_SEGMENTATION=1`) gives the figures the segmentation column (§1); the
Opus set is built with it, the Sonnet and Haiku sets without. (From a worktree, pass the
batch root as an absolute path: `output/` lives in the main checkout.)

Each complete category (all five runs present) is built with `./build.sh agenticresults`
as `illustrations/<prefix><slug><suffix>.{pdf,png}`, prefix `agentic_results_` by default
(`--prefix`), slug the heading in lower case (`household_items`, …,
`medical_and_anatomy`, `landmarks`). `--only SLUG …` builds some, `--watch` polls a running
batch every minute and builds each category as its runs appear, `--list` takes another
object list. `AR_GPU` (default 1) and `AR_FLOOR` (default `warn` here) pass through. After
each build the figure's record is kept as `agentic_results/ar_stats_<name>.json`, and the
shared `ar_stats.json` is restored. Logs go to `agentic_results/_logs/<name>.log`.

---

## 5. The model-comparison pages

`./build.sh agenticcompare` → `agentic_results_compare_p1.{pdf,png}`, `_p2`, … : every
object of the list, in list order, one row each, the three models side by side, on US
letter pages (8.5 × 11 in, portrait) for the supplementary information.

**Columns (13).** The prompt card, as tall as the row; then for each model (Claude Haiku,
Sonnet, Opus, left to right, the version read from the runs' `metrics.json`) four columns:
TOKENS & TIME (the three cost bars of §1), GENERATED SHAPE, SUPPORTS & LOADS and YOUNG'S
MODULUS, each panel head with its AGENT pill. A model-name header with a rule under it
spans each group. There is no deformed-shape column.

**Rows.** Each image panel (12.1 × 10.4 mm) is exactly as tall as the stack of three cost
bars beside it; the text sizes (prompt 5.4 pt, cost 4.6 pt, labels 4.2 pt, keys 5 pt) were
set for that row. The panel corner labels of §1 move to a label line under the images:
the route under the shape, the body load under the supports and loads (past 999 kN in MN
and GN), the grid-cell square under Young's modulus. A label wider than its column is set
smaller (down to 3.8 pt for `SDXL → Hunyuan3D`) and the build lists it.

**Scales.** One prep over all runs of all three models, so the E colours share one scale
on every page; the cost bars share one scale too (the largest input, output and wall
clock over all runs).

**Pages.** At most 17 rows per page (`ARC_ROWS`); the objects are split over the fewest
pages as evenly as possible, and of the orders of those page sizes the one whose smallest
piece of a category on a page is largest is used. For fifty objects: **3 pages of 17, 16
and 17 rows**, every category piece at least two rows. Each page carries all headers and
the keys (cost colours, Support / Load, the floor-grid square, the E bar), but not the
deformed-shape keys. Categories are marked by a rotated name in a strip at the left with a
bracket along their rows, and a darker rule above their first row.

**Cameras.** Each run keeps the camera `choose_view` picks for it (§7) and is cropped to
the union ink box of its three panels: the three models' bodies of one object are
different meshes, so one shared camera would suit one of them.

**Failed and missing runs.** A panel whose stage a run never reached is a `not reached`
cell (`failed` when the run's own status is not ok); its cost bars are drawn. An object
with no run of some model stops the build at the plan step, naming every missing object
per model.

**Steps** (`build_agenticcompare`): `ar_compare.py plan` (runs, order, model names →
`_cwork/compare_plan.json`, `runs.txt`) → `ar_prep.py` on all runs → `ar_compare.py
restrict` (render shape, BCs and E only) → `render_ar.py` → `ar_compare.py compose`. It
writes `agentic_results/ar_stats_<name>.json`, `<name>_tokens.{csv,md}` and
`<name>_time.csv` with one row / column per run.

**Testing without renders.** `compose --stub-panels crops --stub-from <illustrations>`
fills the image panels with crops of the built single-model category figures
(`agentic_results_<slug>.png`, corner labels blanked) and computes everything else from
the real runs on the CPU (`light_record`: route, body load, grid cell, E range, reached
stages). `--stub-panels synthetic` writes fake renders into the work folder and runs the
real compose path. `plan --allow-missing` (stub pages only) draws missing runs as
`no run yet`. Stub pages carry a red "stub panels" note.

```shell
cd illustrations/agentic_results
python3 ar_compare.py plan --haiku ROOT_H --sonnet ROOT_S --opus ROOT_O --work /tmp/cmp \
    --only household_items furniture
python3 ar_compare.py compose /tmp/cmp --out /tmp/cmp/test --stub-panels crops --stub-from ..
```

---

## 6. Stats files

| File | Written by | Holds |
|---|---|---|
| `agentic_results/ar_stats.json` | `agenticresults` | The record behind every number of the last main build: `E_log10_range_Pa`, and per run `draw_scale` (the per-object size override, §1; 1 = the rule's size), `prompt` (the card's lines and font size, `prompt_max_lines`, §1), `meta` (instruction, model, session, tokens, wall clock, status, `time_split`) with `source` (where each value came from), `notes`, `body_source`, `extent`, `dims`, `unit_system`, `painted`, `bc` (regions with area, normal, markers; arrows; `body_load` with its weight and method), `view` (the chosen camera and its scores), `solve` (run, interpolation checks, \|u\|max), `scale` (×N and its rule), `E_source`, `E_surface_Pa`, `ground` |
| `agentic_results/ar_stats_<name>.json` | `ar_categories.py`, `agenticcompare` | The same record for one named figure |
| `agentic_results/<name>_tokens.csv`, `.md` | `ar_figure.py`, `ar_compare.py` | Per run: instruction, session, model, input total / cached / not cached (written to cache, plain), output tokens, wall clock, status, ×N; a header note with the bars' full-scale values |
| `agentic_results/<name>_time.csv` | `ar_figure.py`, `ar_compare.py` | Every wall-clock section of `ar_time.SECTIONS` (its bar group, label, meaning) and each group's sum, in seconds, one column per run; the last row is the wall clock |

---

## 7. The camera (`ar_prep.choose_view`)

Candidates: 24 azimuths (every 15°) × 4 elevations (the paper view's ~20°, 30°, 40°,
50°), view-up vertical, never from below. Filters:

1. **Line of sight.** No view within 35° of the governing load (the traction with the
   largest resultant |t| × area, else the body load), either way along it, since its
   arrows would print as stubs.
2. **No pure front or side view.** The horizontal heading keeps 20° off every principal
   horizontal axis of the body.
3. **Loads seen.** For each load region, the share of its area that faces the camera and
   is not hidden by the body (ray casts); a view scores its worst load region relative to
   that region's best view, and views at ≥ 50 % of that pass (when none does, those within
   90 % of the best). Supports that some view shows at all must be seen at ≥ 25 % of their
   best where possible.

Score: the cube root of three ratios, each over its best passing view — the body's
silhouette area, its visible (front-facing, unoccluded) surface area, and the share of the
panel (1.4 : 1) the silhouette fills once fitted — times the governing load's printed
arrow length to the power 0.5, times the share of the body's height the view shows.

**Preference order.** The paper view (1.5, 1.25, 3.0) when it passes and scores ≥ 85 %
of the best; then the passing views within 95 % of the best, nearest the paper view
first; then every other passing view, best score first.

**Print check** (`preview_print`). The score cannot see what the strict compose fails
on, so the views are tried in that order and the first one the compose is *predicted to
pass* is taken. For each, the row is rendered again at 350 × 350 by Warp ray casts from
`render_ar`'s own perspective camera (the fit points' bounding sphere at the 25° view
angle, dolly 1.02), with the overlays the prep would add for that camera (load arrows,
support outline tubes, the seen-through layers), cropped by `row_crop`'s rule into the
figure's own panel box (`ar_figure.print_layout`: its height follows the longest
prompt), and checked as `ar_figure` checks it:

- every BC region over its print floor (0.5 mm², or the small-region rule) × 1.15, and
  every arrow over 0.3 mm² × 1.15 and 40 % of its length. A region predicted under
  2 mm² is measured again at the render's own resolution over its box, eroded by one
  pixel as the compose erodes it; that measure tracked the compose to about ±0.04 mm²
  on the paper runs, so it only has to reach the floor itself;
- every corner label (route, body load, grid cell, ×N) placed by the compose's own
  corner rule (`pick_corner` / `right_band`) with no ink, grown by one preview pixel,
  under its text.

When no candidate passes, the views are tried once more with that row's ink lifted above
the bottom label band (`ar_figure.LIFTED_ROOM`, 3.0 mm under the ink instead of `SH_ROOM`
1.7 mm): the body prints a little smaller, and the route and body-load labels get a free
bottom corner. The view records `ink_room` and the compose crops that row so; every other
row is drawn as before (the earlier paper figure's leaning tower of Pisa, a tall centred body
whose route label lay on the tower in every view; no row of the current six needs it). In the five-column layout every row already
has this room, so the retry does not run there.
When still no candidate passes, the first preferred view is kept and the strict compose fails
on that row, naming the label or region (Opus vehicle parts: the car wheel, whose
tyre-contact load only one view shows, and that view's body fills both right corners).
`ar_stats.json` records the check under `view.print_check` (the views rejected and why,
the predicted areas and corners). `AR_VIEW_TABLE=1` prints the table and each print
check; a runs-JSON `view_dir` overrides the choice (the paper runs pin none). The
comparison pages (`--no-print-check`) and the showcase keep the plain chooser: their
layouts differ.

---

## 8. The showcase (variation) figure

`./build.sh showcase` → `showcase.{pdf,png}` (or `$SC_NAME.{pdf,png}`), scripts in
`showcase/` (`sc_prep.py` → agentic_results' `render_ar.py` → `sc_figure.py`). It reuses
the agentic-results prep, renderer and look; every number is read from
`showcase/sc_stats.json`, the copy of the build's `ar_stats.json` (with a `rows` list).

Three rows, each one sweep of agentic runs:

| Row | Varied | Input panel | Runs from |
|---|---|---|---|
| Shapes | realizations of one instruction (`run_realizations.sh`) | BC (body, supports, loads) | `SC_SHAPE_RUNS` |
| Materials | per-part materials on one body (`run_material_sweep.sh`) | E map (this figure's cheese-lime-watermelon-herb map, below) with part seams | `SC_MATERIAL_RUNS` |
| Supports & loads | load cases on one body (`run_bc_sweep.sh`) | BC | `SC_LOADING_RUNS` |

Each `SC_*_RUNS` is a space-separated list of run folders or globs (absolute, or relative
to the checkout holding `output/`); skip a sweep's `_batches` folder with `[!_]*`. A row
not given this way comes from `SC_ROWS_JSON` / `showcase/sc_rows_placeholder.json` and is
tagged PLACEHOLDER. Other variables: `SC_GPU` (default 1), `SC_KEEP_WORK=1` keeps
`showcase/_work` so `python sc_figure.py _work --out NAME` recomposes without rendering,
`SC_GROUND`, `SC_ASPECT` (tile height / width, default 0.85).

- **Tiles.** Up to nine items per row (`MAX_ITEMS = 9`): the input grid and the deformed
  grid are each 3 across and up to 3 down, item k at the same place in both. A tile is
  0.85 as tall as it is wide (`SC_ASPECT`, 23.0 × 19.6 mm), with 0.5 mm between tile rows
  and 0.4 / 0.2 mm of row padding. At 1.0 most bodies had a band of air above and below.
  A row is as many tile rows deep as its items need, and never shorter than its label
  column, so a row of fewer items is a shorter row. With the short cards below, the label
  columns and the grids come out within about 2 mm of each other (rows 60.3, 62.4, 62.4 mm;
  the page went from 276.4 to 209.3 mm; the material row's E key since adds 3.8 mm, below).
- **Label column.** The row title, then one block per kind of headless Claude call that
  made the row, each a prompt card (quoted, italic) followed by that call's three cost
  bars (§1: input with its cached share, output tokens, the wall clock with its split).
  The sweep rows' cards are **short gists for illustration**, not the record (the whole
  prompts are the `prompt_source` files). Each is 4–5 card lines in the prompt's own words,
  and every cut is an ellipsis. The condensing is deterministic (regexes in
  `sc_prep.condense_authoring` / `condense_run_prompt`). `sc_figure.py` re-condenses both
  cards from their source files at compose time (`sc_prep.refresh_cards`), so a
  compose-only build of an older `_work` prints the current cards and writes them back
  to its `ar_stats.json`.
  - **Shapes**: one kind. The card is the instruction every run was given; the bars are
    **"Mean of N runs"**.
  - **Materials, Supports & loads**: two kinds, stacked (`sc_prep.sweep_calls`).
    1. **The authoring call** (no head over its bars): the one call that wrote all N
       cases. The card is `<runs>/_batches/<batch>/authoring/prompt.txt` condensed by
       `condense_authoring` to three pieces: the first sentence's head without its
       demonstration clause or parenthetical, the head of the rule that frees the cases,
       and that rule's purpose sentence. For example, "You are writing 9 material
       assignments … Assign materials freely … The point is variety, not realism." and
       "You are writing 9 load cases … Choose supports and loads freely … The point is
       variety.". The bars sit right under the card at agentic_results' card → first
       cost line distance (3.3 mm), so no gap is left where the former "1 call" head
       was. The cost is read from `authoring/claude_result_<k>.json`, summed over every
       try (the build prints the number of tries). Input is
       `input_tokens` + cache reads + cache writes, and cached is the cache reads, as
       for a run. The time is the CLI's own `duration_ms`. The batch log's span from
       "authoring … (try 1" to "authoring: ok" is recorded beside it as a check, and
       it agreed to the second on the paper sweeps (45 s material, 30 s BC). Nothing
       but the model runs inside this call, so its whole time bar is **Agent**.
    2. **"Mean of N runs"**: the N run agents. The card is their `prompt.txt` condensed
       by `condense_run_prompt`. The skill boilerplate (SKILL.md, the session files, the
       numbered steps, the reply) is cut, as is the simulated object's name (the tiles
       show it). The case itself, the only text that differs between the runs, is stood
       in for by **[one of the N cases]**. The brackets mark an editorial insertion, not
       the prompt's words. What is kept is "This run is a <kind> variant of a finished
       simulation", the case, and the head of the rule that has the case run as given.
       For example, "This run is a material variant of a finished simulation …
       [one of the 9 cases] … Apply them as given …" and "This run is a load-case
       variant of a finished simulation … [one of the 9 cases] … Run this case as
       given …". The build warns if the runs' condensed prompts differ.

  A mean is over the row's N distinct runs (a run listed twice counts once). Every run is
  a separate agent session, so the per-run mean is the cost, not the row's sum. The wall
  clock's split is averaged section by section. **All cost bars of the figure share one
  scale per metric**: the largest block (authoring call or row mean) fills the track.
  The build prints every block's numbers and the authoring files, and warns when a run
  lacks a value (that run counts 0 and the divisor stays N, so the mean is then a lower
  bound). The rows take the height of whichever is taller, their tiles or their label
  column. The tiles stay at the row's top, so a two-block row has a band free of tiles
  under its grid. The record is `rows[].calls` in `sc_stats.json`. When `sc_figure.py`
  is run on a `_work` from before that field, it reads the run and batch files itself
  and writes `calls` back to `_work/ar_stats.json`.
- **Keys (footer)** are agentic_results' (`ar_figure.cost_keys`, `bc_keys`, `e_key`,
  `u_keys`, shared with that figure): same items, wording, swatches, order and places.
  - Under the label column are the cost colours.
  - Centred under the input cell are Support, Load (swatch and arrow), and
    "Load on a far face, seen through".
  - Centred under the deformed cell are the |u| ramp from 0 to max |u|, "×N:
    Displacement magnified N times", and "Undeformed shape". The swatch is drawn in this
    figure's darker outline grey.
- **Young's modulus key** (not in the footer): at the bottom of the material row, centred
  under its E tiles, at the footer keys' width (half a cell): agentic_results' `e_key`
  bar with its decade ticks and "Log", and an italic *E* in front, because this figure
  has no Young's-modulus column head to name the ramp. The row reserves `E_KEY_BAND`
  (6.0 mm) under its tiles for it (material row 66.2 mm instead of 62.4; the page 213.2 mm).
- **The E colormap is this figure's own, "cheese, lime, watermelon, herb"**
  (`sc_prep.SC_E_ANCHORS`; user, 2026-10-16: nothing on the hamburger may look blue, the
  colours should look delicious, and "I need multiple colors to make the difference more
  visible, not blue or this deep brown"). It is the designer's proposal P4
  (`output/showcase_cmap_candidates/food_multi/designer_multi.md`, not committed): one
  fresh-produce hue per material class, each class on a nearly flat band, the hue
  changes only in the decades no layer uses. Eight anchors, straight CIELAB lines between
  consecutive ones in log10 E, held past the ends:

  | Class (layers) | Band | From | To |
  |---|---|---|---|
  | soft (gel, foam, rubber, foam core) | 100 kPa – 80 MPa | bun `#fcedc0` (94, −2, 24) | cheese `#fdd359` (86, 2, 64) |
  | polymers (HDPE, PP, PC, nylon, balsa) | 1 – 3 GPa | lime `#a7c83e` (76, −30, 62) | lettuce `#98c13b` (73, −33, 60) |
  | wood / stone (ice, oak, pine, bone, concrete, granite, aluminium, glass) | 9 – 70 GPa | watermelon `#e67f95` (65, 42, 6) | tomato pink `#e3738a` (62, 46, 8) |
  | metals (titanium, copper, steel) | 110 – 200 GPa | herb `#438632` (50, −38, 38) | pickle `#3c7e31` (47, −37, 35) |

  The blends between bands (lime → pink through a short khaki at 3–9 GPa, pink → green
  through a greyish band at 70–110 GPa) show only on the key's bar. Designer's rationale:
  the only proposal in which all four classes have their own colour at a glance, warm
  (cheese, watermelon) and green (lime, herb) families alternating; still an ordered light
  → deep progression; all fresh produce, nothing brown, burnt, blue or purple; the one
  red-family colour is a cool pink, not the |u| orange → red. L\* falls 94 → 86 → 76 → 73
  → 65 → 62 → 50 → 47. History: ice blue → steel (shared), gold → blue → indigo, + lilac,
  blue → petrol, "golden crust" browns (rejected as one deep-brown hue), then P4.

  `sc_prep.check_e_map` (run by the prep and the compose) asserts: in gamut; L\* monotone
  in normal vision and simulated protan / deutan / tritan; the polymer, wood/stone and
  metal bands ≥ 20 ΔE00 from the |u| orange → dark-red ramp in normal vision (21.6 at
  worst, the pink); neighbouring band centres ≥ 18 normal / ≥ 10 colour-blind (20.1 / 11.5
  soft → polymers, 60.3 / 18.9, 62.4 / 20.5); the support teal ≥ 15 normal over the whole
  ramp (22.3) and ≥ 6 colour-blind over the four bands (6.5). Changed with P4: the teal
  colour-blind floor applies to the bands only, since the unused lime → pink blend comes
  within 1.9 of the teal under protanopia and no layer has that colour. Not floored, as
  before: the soft band against the deformed body's cream (11.3 / 5.7) and the light half
  of |u| (10.9 / 3.5).

  **What colour-blind readers lose.** Under deuteranopia and protanopia the greens and the
  pink collapse to khaki and grey, so the four classes survive as four lightness steps
  (pale yellow → mustard → grey-beige → olive): order and separability stay, the "four
  different hues" are lost. No blue-free food palette keeps E apart from |u| point by
  point for these readers either (the metal greens map onto |u|'s olive-brown, 0.9; the
  pink near its mid tones, 3.4), so there is no colour-blind floor against |u|. For them
  E and |u| are told apart by layout (E only in the input column, |u| only in the
  deformed column), by their separate keys (the E bar under the material row, the |u|
  bar in the footer) and by field shape (E constant per layer with dark seams, |u| a
  smooth gradient over a cream body). Tritanopia keeps the four classes distinct. The
  prep writes the map into `ar_<i>_materials.vtk` through `ar_prep.write_run_files(…,
  e_rgb=)` and the key draws it through `ar_figure.e_key(…, stops=)`; both parameters
  default to the shared "steel" map, so every agentic-results figure keeps that.

  The former "Colour and ×N per item" note is gone. Each tile prints its own ×N, and
  "max |u|" is the tile's own peak, as it is the row's own in §1. When the loading row
  shares one scale, its tiles print the same ×N.

Each sweep row's authoring prompt (the one call that wrote its 9 cases), as sent, is kept in
`experiments/agentic_batch/showcase_authoring_prompt_{material,bc}.txt`; `build.sh showcase`
refreshes them after each build with `sc_record.py`.

## 9. The three-model evaluation

Moved to [`experiments/model_eval/`](../experiments/model_eval/README.md): the rubric, the evaluator
prompts, the three evaluators' ratings and the voted table (`evaluation_table.{md,xlsx}`).
