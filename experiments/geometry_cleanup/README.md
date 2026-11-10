# Geometry clean-up of the agentic-batch bodies

A short, **volume-as-is shape optimization** of each of the 50 Opus runs in
`output/agentic_batch50_opus/`, used as a geometry clean-up: small generation defects
(bumps, specks, thin artefacts) are smoothed out and very weak members (thin necks) thicken
a little, while the object stays recognisably the same. Every run stops early, as soon as
the surface has moved a fixed fraction of a leaf away from where it started.

Both the input batch and this study's outputs are archived:
`output/archives/agentic_batch50_opus.tar.zst` and `output/archives/geometry_cleanup.tar.zst`.
The scripts read and write the original paths, so to re-run, re-summarize or re-render,
extract them first: `zstd -dc output/archives/<name>.tar.zst | tar -x -C output`.

The runs go through the workflow's own `workflow_tool.run_forward` with
`run_mode: "shape_optimization"`, so case parsing, BC selectors, material, loads and the
`ShapeOptTrimmed` parameters are exactly those of the workflow's "Run FEA". The agent
sessions are never modified: each is copied first.

## Files

| file | runs in | what it does |
|---|---|---|
| `common.py` | – | paths, the run list (`metrics.json` `session_id` of each run), and the settings below |
| `setup.py` | `mechfoundry-warp117`, CPU | copies each session's inputs to `output/geometry_cleanup/<object>/session/`, writes `case.json` (+ `setup.json`), with a uniform stand-in material for per-part sessions |
| `run.py` | `mechfoundry-warp117`, GPU | one object's run with the step rule, the early stop and the outputs (`results.json`, `traj.tsv`, `surfaces/it_<k>.npz`, `solve/`) |
| `run_queue.sh` | bash | runs objects one after another on one GPU, each under `timeout`, never next to another compute process on that GPU |
| `summarize.py` | any, CPU | `output/geometry_cleanup/summary.{tsv,md}` |
| `render_prep.py` | `mechfoundry-warp117` (warp) | per run, the start and final surfaces as VTK with the signed change from the start (in h) |
| `render.py` | `pvbatch` (EGL) | start / final / change renders from two fixed cameras |
| `montage.py` | any (PIL) | contact sheets `output/geometry_cleanup/montage_<page>.png` |

```shell
cd experiments/geometry_cleanup
conda activate mechfoundry-warp117
python setup.py                                     # all 50 (or name objects)
nohup bash run_queue.sh 0 3600 "" <objects...> &    # one queue per GPU
python summarize.py
python render_prep.py && pvbatch render.py /home/xl2254/implicit_optimize/output/geometry_cleanup/*/ \
  && python montage.py
```

`run.py` options exist for trials only (`--step`, `--threshold`, `--cap`, `--reinit`,
`--extra`, `--tag`); a tagged run writes to `<object>/<tag>/` with its own case.
`GEOMETRY_CLEANUP_SAVE_EVERY=1` (opt-in, off for the batch) keeps the solver's own output
(VTU volume + surface with displacement and stress) at every iteration instead of the
first and last; the demo video's optimisation card uses it (`illustrations/demo_video/dv_cleanup.py`).

## The settings

| | value | where |
|---|---|---|
| volume | `volume_as_is: true` (penalty `w_vol softplus(V/V0 - 1)`, the session's `w_vol`, 5 everywhere) | `common.COMMON` |
| BC preservation | `sdf_bc_fix_method: "mask"` for all 50 | `common.COMMON` |
| SDF reinit / octree rebuild | **off** (`reinit_sdf: false`), see "Trials" | `common.REINIT` |
| step | first update's largest surface move = **0.5 h** | `common.STEP_FRACTION` |
| distance | symmetric, area-weighted **99th percentile** of nearest-point distances, current vs starting surface | `common.STOP_PERCENTILE`, `run.surface_distance` |
| stop | at the first shape whose p99 distance exceeds **1.5 h** or whose maximum distance exceeds **3 h**; that shape is rejected and the run ends on the last accepted one | `common.STOP_THRESHOLD`, `common.STOP_MAX` |
| cap | **40** accepted updates | `common.MAX_UPDATES` |
| Krylov cap | `jaxamg_max_iters: 4000` (a solver parameter, not a card setting) | `run.SOLVER_EXTRA` |
| everything else | the session's own: octree depths (the forward solve's `remesh_octree_depth`, 7 or 8), supports, loads, body force, precision, Newton tolerances | the session's `case.json` |

h is the analysis leaf, `h_target * 2^(octree_depth - remesh_octree_depth)`, as
`run_forward` derives it.

**Step.** `gd_lr` has units of length² and the bodies range from 3 cm to 200 m, so
`run.py` reads, at iteration 0, the largest |dJ/ds| over the free nodes and scales the
gradient by a constant so that `gd_lr_eff * max|dJ/ds| = 0.5 h`. This is the run with
`gd_lr = gd_lr_eff` (reported in `results.json`), taken without a separate probe run.

**Distance.** Every solved shape's marching-cubes surface is compared with iteration 0's:
for every vertex of each surface, the distance to the nearest point of the other (warp BVH),
weighted by the vertex's share of the area; the larger of the two directions' 99th
percentiles. The percentile lets a locally thickened neck (a small area that moves a few
h) through and stops on change that is spread over the body. It cannot see a small
feature being erased (a tusk is far less than 1 % of an elephant), so the maximum of the
same distances is bounded too, at 3 h: the necks and junctions the clean-up is meant to
thicken reached 2.3–2.8 h (chair, park bench, crocodile), while erased features show up
as jumps of the maximum to 4–7 h. None of the 50 starting surfaces has a speck (all are
one watertight component), so the maximum bound removes no speck clean-up. p95 and the
mean are recorded too.

**Stop.** A shape over the threshold is rejected: its update is not taken, the design SDF
is set back to the last accepted one (copied before every update; the octree is fixed, so
it is an exact copy) and one more iteration solves that shape, which ends the loop through
the solver's own convergence check. The final shape is therefore always within the
threshold and always solved (its SE is known). At the cap, the shape just solved is final.

**Per-part materials.** 17 sessions use `mode: "parts"`, which `run_forward` refuses for
shape optimization. `setup.py` replaces it with a uniform stand-in: the part-volume-weighted
mean E, ν and ρ, with part volumes from the rule the part field itself uses
(`material_parts.field.part_grid`: every lattice cell the surface crosses or encloses takes
the part of the nearest surface point) on a 160³ lattice. `setup.json` lists the parts and
their volume fractions; results carry `standin: true`.

## Trials (2026-10-14)

Shapes: school chair (thin necks, per-part), crocodile (organic, generated detail),
park bench (thin members, per-part), Himeji castle (large building, 0.70 m leaves,
per-part), USB stick (small, 0.86 mm leaves, per-part), giraffe (thin legs), bar stool.

1. **The solver's per-iteration reinit is not a clean-up.** With the workflow's default
   `reinit_sdf: true` (MC → exact distance → new octree every iteration) and a **zero**
   step (`--step 1e-6`), the chair's surface moved 0.22 h (p99) in one iteration and
   0.56 h in eight, and its legs (≈ 2 h thick) became beaded — the same trajectory as with
   a real step (0.22 / 0.45 h at iterations 1 / 6). The MC → distance → MC cycle is not
   idempotent on members a few leaves thick. `reinit_sdf_method: sussman` was worse
   (4 h in two iterations, then a failed linear solve); `remeshing_method: interpolate`
   identical. With reinit off the zero-step distance stays 0.000 h, and the optimization
   alone moves the shape: the chair keeps V/V0 0.997 (0.92 with reinit), SE/SE0 0.40 after
   40 updates, its legs unchanged and the seat/back necks thickened. So the clean-up runs
   with reinit off; `run.py` then extracts each iteration's surface itself (the solver only
   re-extracts it inside the reinit).
2. **Step 0.5 h vs 1 h** (max first move). Both converge to the same kind of shape; 1 h
   makes V/V0 alternate 0.98 / 1.00 on the chair (the volume penalty overshooting) and
   moves thin-featured bodies by a lot in one update (Himeji: p99 2.9 h after the first
   update at 1 h, 1.44 h at 0.5 h, 0.55 h at 0.25 h). 0.5 h is smooth everywhere.
3. **Threshold.** p99 distance against what the renders show (step 1 h, 40 updates):
   chair necks visibly thicker from p99 ≈ 1.0 h and still fine at 1.5 h (max 2.3 h at the
   necks); crocodile legs thicker, snout slightly thinner, same animal at 1.4–1.7 h;
   park bench junctions thicker at 1.1–1.5 h; giraffe lower legs thicker at 1.1–1.5 h.
   A p99 of 1 h stopped the chair before its necks had finished thickening (the user's
   requirement), while above ≈ 1.5 h the USB stick, a box only ≈ 8 leaves thick, visibly
   turns into a wedge. **1.5 h** was fixed; the local change it admits (the max) is
   2–3 h, the "a little thicker" the clean-up asks for.
4. **Final settings on the trial set** (step 0.5 h, 1.5 h, cap 40): chair stops at the
   threshold after 34 updates (SE/SE0 0.42, V/V0 0.997, necks thickened); crocodile,
   giraffe, park bench, USB stick reach the cap at p99 1.4 / 1.1 / 1.1 / 1.4 h;
   Himeji's second update (2.6 h: its sub-leaf roof eaves vanish under any step) is
   rejected and the run ends on the first (1.44 h, eaves eroded slightly).

5. **Batch, first pass with the p99 bound only** (kept as `<object>/_p99only/`): in the
   contact sheets the elephant lost its tusks and tail, the Taj Mahal's dome and
   minarets shrank, the road roller's canopy sheet opened, the Pantheon's drum and the
   stone arch changed visibly — all with p99 ≤ 1.5 h but a maximum of 4–7 h. The 11
   objects whose accepted shapes had passed 3 h were rerun with the maximum bound; the
   other 39 never reached it, so the bound does not change their runs.

Known limits: thin sheets below a leaf (roof eaves) cannot be cleaned up without being
eroded; a chunky body only a few leaves thick (USB stick) changes visibly relative to its
size within 1.5 h; without reinit, the design SDF drifts from a distance field, so
faceted patches and gap bridging appear where the change is largest: the chair's final
shape grows a short, slightly faceted brace (≈ 2 h) from the rear leg toward the back
hoop, exactly at the weak neck, plus the thickened hoop/seat junction.

Artefacts seen on the 50 (contact sheets `output/geometry_cleanup/montage_<k>.png`):

* **Mask BC patches.** `sdf_bc_fix_method: "mask"` freezes the SDF on the support and
  load regions, so the surface around them moves and they do not: a flat patch ends up
  standing proud or sunk by ~1–2 h with a sharp rim (computer mouse "palm"/"click",
  loaf "knife band", watermelon "bite patch", TV remote "button traction", hamburger
  faintly), and the rim of a frozen base can serrate (computer mouse bottom, igloo base).
* **Uniform shrink of unloaded bulk.** The volume-as-is penalty `softplus(V/V0 - 1)` has
  slope 0.5 at V = V0, so material with no load path is pulled in everywhere (jelly
  dessert V/V0 0.915, skyscraper 0.926, Pantheon 0.963).
* **Sub-leaf features.** Sheets and spikes thinner than a leaf vanish under any step
  (Himeji eaves, lighthouse gallery, Taj Mahal minarets, road-roller canopy); the
  maximum bound rejects that update, so these bodies end after 0–1 updates, unchanged
  (the leaning tower, whose starting surface is already fragmented at its 0.85 m
  leaves, is rejected by the p99 bound at its first update).
* **Gap bridging / new fragments** between members 1–2 leaves apart (crankshaft webs,
  a few specks under the lumbar vertebra), the reinit-free SDF's drift.
