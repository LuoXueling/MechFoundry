# Diffused Nitsche vs body-fitted P1, per workflow session

`compare_bodyfitted_nitsche.py` takes a finished API-workflow session and solves
its case two ways: with the workflow's own diffused-Nitsche solve on the octree,
and with body-fitted P1 tetrahedra posed the way the Nitsche solve poses the
problem. It then measures the Nitsche field against the P1 field directly. There
is no third, finer reference. `run_batch.py` runs it over a batch of sessions
and `summarize.py` tabulates the result.

It is a standalone experiment. It imports production code from `src/` and
changes none of it.

## Method

**Nitsche.** `workflow_tool.run_forward` runs on a copy of the session's
`case.json`, redirected into the output folder and forced to `run_mode=forward`:

- float64, the sessions' precision; diffuse volume;
- `remesh_octree_depth` 7 and 8 (`--depths`). `remesh_init_octree_depth` is
  lowered only if it is not below the depth (never for this batch; recorded);
- the current dev code, in which saved-segment and solver-resolved BC selectors
  both use `core.nitsche_field.bc_match_tolerance` = max(1.5 eps, 0.75 h_min)
  (dev 8c2f794).

Paths a batch session recorded under its slot folder are re-rooted on the
session folder. A per-part session gets, for both methods, the
part-volume-weighted mean E, nu and rho of its parts (`part_standin`, copied
from the geometry clean-up's `setup.py` and cross-checked against its
`output/geometry_cleanup/<object>/setup.json`).

**Body-fitted P1.** The surface the regions were resolved on
(`geometry/physical.glb`) is cleaned (GLB seam vertices merged) and
tetrahedralized with TetGen. Every boundary side is mapped back to the source
face it lies on, so the masks in `boundary_conditions/resolved.npz` carry over
face for face. The production `core.neoHookean.NeoHookean` Newton solver runs on
a Warp `Tetmesh` (`TetNeoHookean`), float64. It is posed as the Nitsche solve
poses the problem:

- **Clamp band.** Every node within `bc_match_tolerance(eps, h_min)` of a
  Dirichlet region's faces is held, in that region's components. eps is
  `phase_field_epsilon_factor` x h_min, and h_min is the leaf of the depth being
  compared (0.75 h_min at eps = h_min/2).
- **Nitsche load definition.** A Neumann region with traction t loads each
  boundary face with t max(0, n . n_r), where n_r is the region's area-weighted
  mean outward normal, so facets turned away get no load. The total force is
  t x `loaded_area` = t x sum A max(0, n . n_r): the area the Nitsche solve
  rescales its band weight to (`SurfaceSelector.loaded_area`, the value
  `resolve-bcs` reports). Overlapping regions are averaged per side. The
  region's face set is the drawn one; the Nitsche band's diffuse spill is not
  reproduced.

**Three P1 meshes, re-matched at each depth:**

- `dofs`: a uniform size s* whose vertex count is within 5 % of the Nitsche node
  count at that depth. The surface is brought to ~s* (kept, split exactly, or
  remeshed isotropically with pymeshlab) and the interior is bounded by s*. s* is
  found by a secant search on log(vertices).
- `surface`: the same family at s = h_min (surface triangles ~h_min).
- `graded`: the surface of the `surface` mesh. The interior is sized by TetGen
  `-m` from the depth's own Nitsche octree: the leaf size of the cell containing
  each node of a box background grid, times one factor found by the same secant
  search on the vertex count. If `-m` crashes at every factor it is retried once
  without `-Y`.

## Metrics

All metrics are evaluated on the P1 mesh's own order-2 quadrature, over the true
body. The P1 field is exact there. The Nitsche field is the trilinear Q1
interpolant of the octree cell that contains each point, read from its VTU, so
neither field is resampled. With e = u_Nitsche - u_P1:

- **headline:** `rel L2 w/o rigid` = ||e - r|| / ||u_P1||, where r is the
  best-fit infinitesimal rigid motion of e (six modes, weighted least squares at
  the quadrature points);
- `rel L2` = ||e|| / ||u_P1||;
- stress: each method's Cauchy stress, from its own displacement gradient and
  the same neo-Hookean material (sigma = P F^T / J). Reported as the rel L2 of
  the tensor ||sigma_N - sigma_P1|| / ||sigma_P1||, the rel L2 of von Mises, and
  the volume-weighted p95 and p99 von Mises ratios (Nitsche / P1). The tensor rel
  L2 is also given over the points at least 2 h_min from every Dirichlet and
  Neumann patch, because the clamp band and the loads distort stress locally.
  Stress needs no rigid-motion removal, since an infinitesimal rigid motion
  carries no strain.
- `compliance` = l(u_N) / l(u_P1), where l is the P1 load functional (traction
  and body force at the P1 nodes). Below 1 means Nitsche is stiffer.

## The pipeline: one command

```shell
nohup timeout 30h python experiments/bodyfitted_vs_nitsche/run_batch.py --gpus 0 1 --largest-first \
  > output/bodyfitted_vs_nitsche/_batch/full.log 2>&1 &
```

The command runs, unattended:
1. **Phase 1, solve.** One session job per idle GPU (bounded waits), largest
   first. The job is the session's **orchestrator** (`--phase solve`):
   - it meshes (TetGen in spawned children) and decides the ladders;
   - it runs **every solver rung in its own child process** (`--phase
     attempt`), so a crash, a NaN-poisoned CUDA context or a stuck rung costs
     that rung only, and pyamgx always initializes AMGX first in its process;
   - it never initializes AMGX itself;
   - each rung writes a record (`attempts/<kind>_<entry>_<rung>.json`: backend,
     unit-diagonal and other flags, error, residuals, device, timings), and a
     re-run resumes from the records.
2. **The CPU queue,** concurrently with phase 1 (`cpu_p1_worker.py`, no CUDA
   context, 2 jobs of 24 MKL threads, one job of 1.2 M nodes or more at a time).
3. **Repair rounds** for sessions without `solve_done`.
4. **The CPU queue drained,** then the **lean GPU full Newton** for every CPU
   job the worker found too nonlinear for its fixed tangent and too large for
   PARDISO. Then one more repair round for any other failed CPU job.
5. **Phase 2, evaluate:** every metric from the stored outputs only, chunked.
6. **The summary.**

**GPU scheduling.** A job starts on a GPU when no process holds it. Light,
idle foreign processes do not count: those that hold 2 GB or less together
and leave the GPU under 10 % utilization over 30 s, such as the fem_workflow
UI's warm `tool_server` or pvbatch renders. Anything larger or busier makes
the batch wait. This rule was adopted during the fresh run of 2026-10-16,
after 17 of 50 sessions, when the UI kept respawning its workers. It changes
scheduling only, not results or solver timings.

**Hang watchdog.** A job whose `run.log`, `status.json` and rung records do
not change for 20 min, while its process tree uses under 2 % of a core and its
GPU stays under 5 % utilization, is killed and retried once (recorded as
`hang`). Long PARDISO or CPU solves burn CPU and GPU rungs use the GPU, so they
do not trigger it. It was added during the same fresh run, after 23 sessions,
when the natural-stone-arch orchestrator hit "malloc(): unaligned tcache chunk
detected" and sat idle for 65 min. It changes scheduling only.

A job that ends without `solve_done` is run once more, and resumes from the
rung records. A job that ends with a non-zero code after `solve_done` (an AMGX
destructor abort at exit) is done. A rung whose process ends after writing its
record is judged by the record.

### The Nitsche ladder (`NITSCHE_LADDER`, per depth)

It follows the operator skill's order for a stalled JAX-AMG solve
(`operator_skill/references/troubleshooting.md`, `drive.py backend_fallback`).

| rung | what | accepted when |
|---|---|---|
| `jaxamg` | the workflow's run_forward, float64, rel 1e-6 | Newton converges |
| `jaxamg_4000` | 4000 Krylov iterations; only after "ran out of iterations" | same |
| `pyamgx` | `cg_backend_assembled=pyamgx`: it keeps its best iterate, Newton's residual check decides | same |
| `pyamgx_beta500` | + `beta_max_over_e: 500`, the setting the error names; uniform material only, skipped for a stand-in | same |
| `jaxamg_unit_diagonal` | the last GPU resort (a user exception to the skill's "never") | Newton's rel is the unscaled force residual (the scaling acts inside the linear solve only) |
| `direct` | MKL PARDISO on the CPU for each linear solve (`cpu_p1_worker.py --direct`), session load steps | Newton converges |
| `direct_continuation` | + 10 load steps, line search | same |
| `direct_continuation_session_tol` | the session's own tolerance | flagged "looser tolerance" in the summary, outside the medians |

A pyamgx rung with unit-diagonal is not a rung: pyamgx ignores the flag
(core.utils), so it would be the pyamgx rung again.

### The P1 ladder (`P1_LADDER`, per entry)

Standard GPU path, `TetNeoHookean`, the session's own `load_steps`:

| rung | what | accepted when |
|---|---|---|
| `jaxamg` | JAX-AMG, 1000 Krylov iterations | Newton converges |
| `jaxamg_4000` | only after "ran out of iterations" | same |
| `pyamgx` | fresh process, warm-up on pyamgx | same |
| `jaxamg_unit_diagonal` | the last GPU resort | the unscaled ‖f_int(u) − f‖/‖f‖ on the free components, recomputed from the solution, ≤ 1e-6 |

After that the entry goes to the CPU queue. An entry goes there directly when
its mesh has 450k nodes or more (the d8 surface meshes of 0.49–2.2 M nodes ran
out of GPU memory), or when a rung runs out of GPU memory. Unit-diagonal does
not change memory.

**P1 never waits on Nitsche unnecessarily.**
- The matched-surface mesh needs only h.
- The matched-DOF mesh needs the Nitsche DOF count, which the octree build
  of the first Nitsche rung reports, even when that rung fails.
- The graded mesh needs the octree's leaf sizes. Any Nitsche VTU carries them;
  the octree does not depend on the load or the solver.

So per depth the order is: the first Nitsche rung; P1 surface and dofs; the
rest of the Nitsche ladder; P1 graded.

### The CPU queue (`cpu_p1_worker.py`) and the lean GPU path

- **The same discrete problem.** The GPU job writes the mesh, the clamp band
  and the load in the GPU discretization (`cpu_jobs/<tag>.{npz,json}`). On the
  arch d7 surface, the load matches the GPU `f_ext` to 8e-15.
- **Modified Newton.** The u = 0 tangent is factored by MKL PARDISO. PARDISO's
  analysis predicts its memory (elephant: 15 GB predicted, 16 GB measured). It
  factors in core only up to 100 GB, and only while 40 GB stay free beyond the
  prediction; otherwise pyamg SA-CG is used.
- **Full Newton when it does not contract.** When the first modified-Newton
  step contracts by less than 20x (teddy bear: rel 1 to 354; hamburger: 0.66),
  the worker switches to full Newton:
  - the session's load steps, the re-assembled neo-Hookean tangent
    (finite-difference checked to 9e-11);
  - PARDISO refactorized per iteration, if a factorization takes < 900 s;
  - plain Newton steps, backing off only from a non-finite or 1e4x residual.
- **Else "needs full Newton".** The driver then runs the **lean GPU full
  Newton** (`torch_newton`):
  - the tangent is computed in torch on the GPU, into a fixed 3x3-block Warp
    BSR, solved by JAX-AMG;
  - before first use it validates itself on the GPU against two stored
    standard-path solutions (`--validation-root`: teddy bear d7 dofs and
    elephant d8 surface; rel L2 and compliance ≤ 1e-6, measured 1.7e-9 and
    1.6e-8).
- **Validation of the CPU path:** elephant d8 surface against the GPU, rel L2
  1.6e-8.
- **Timing.** CPU and lean-GPU solve times are labelled (`solve_device`) and
  are not comparable with the GPU times.

### Robustness

- **Wall-time cap per rung:** 30 min for the first, 15 min for the others, 90
  min for direct rungs. It is checked before every linear solve, and the rung's
  process is under `timeout` as well. A direct solve is capped at 30 min.
- **Records and markers:**
  - a rung record that exists is reused;
  - a rung process that ends without its record is recorded as a crash or
    timeout;
  - the CPU queue's `.running`/`.done`/`.failed` markers; the memory guard.
- **Evaluate** skips a session whose `comparison.json` is newer than all its
  inputs (`--recompute` redoes it). Its integrals and the rigid fit run in
  chunks (the dental crown d8 surface has 73 M quadrature points).

**TetGen.**

- **Spawned child.** TetGen runs in a `spawn`-ed child that imports only numpy
  and tetgen (`tetgen_child.py`), under a 30-minute timeout. Forking a process
  that already runs JAX threads can deadlock, and a TetGen crash costs only
  one attempt.
- **Background scaling (TetGen 1.6.0 bug).** `randomsample` compares a squared
  distance with the unsquared box diagonal D. With `-m` and D >~ 4.6 it rejects
  every sample and segfaults (pyvista/tetgen#65). The input surface, the
  background nodes and the sizes are scaled by an exact power of two
  k = 2^-ceil(log2(D/0.5)) and the output nodes divided by k. This is
  bit-identical where it already worked; k is recorded.
- **Remesh rule.**
  - Our isotropic remesh is checked for self-intersections: pymeshlab's
    per-face selection, confirmed by TetGen `-d` with exact predicates.
  - If it intersects, it is redone at `maxsurfdist` 0.05 s.
  - If that still intersects, the cleaned source is meshed with `-Y`, flagged
    "meshed at source resolution".
- **Variant chain.** The switches as given, with `-Y` toggled, and with a
  looser quality bound. A graded mesh whose `-m` run crashes at every scale is
  retried without `-Y`. Every attempt is logged in `tetgen_log`.
- **True failures.**
  - A source surface that already self-intersects is never repaired.
  - For it, or a mesh no variant meshes, `tetgen_evidence.json` is written
    automatically: TetGen -d intersection counts on the cleaned source, the
    pymeshlab count, and every variant on the source as-is.
  - The summary prints it.

**Timing.**

- Wall time per phase, GPU-synchronized at every phase boundary, after an
  untimed warm-up in the same rung process (JIT excluded).
- **Nitsche prep:** surface load, case parse, BC resolution, SDF and octree,
  solver build and trimming, and `build-case`.
- **P1 prep:** surface remesh, sizing, TetGen, boundary mapping, BC transfer and
  solver setup. Reported with the successful TetGen attempt only, and again
  including the failed ones.
- Solve times are reported to the tight tolerance and to rel 1e-2.

**Reproducibility.** Each phase records the git commit, the library versions
(tetgen, warp, torch, pymeshlab, numpy, trimesh, scipy), the GPU and all
options, in `solve.json` and `comparison.json`. The driver records the same in
`_summary/run_info.json`. NumPy and Torch seeds are fixed (0), and
`PYTHONHASHSEED=0`.

### Documented gaps, notes and verification

- **Coverage.** The summary counts P1 entries "solved" and "compared". A P1
  entry is compared only where its depth's Nitsche solve converged.
- **Documented gaps.** A Nitsche entry no rung converges is listed with every
  rung's outcome, the support rigid-mode ratio, the worst PARDISO linear
  residual, and its P1 side. A session's `notes.json` adds manual evidence, or
  a superseded result kept outside every table.
- **Jackhammer d8 is the one such gap.** No rung converges. Even PARDISO
  reaches only a linear residual of 7.5e-6. Newton wanders even at 10 % load.
  Its support rigid-mode ratio is 9.8e-5; every other session audited is above
  2e-3. This is a near rigid-body mode. An earlier direct run's "convergence"
  is a non-reproducible chance landing, recorded only in its notes.
- **`compare_runs.py NEW_ROOT`** compares two output roots entry by entry:
  status, path taken and metric change.

## Flags

Flags are evaluated at the session's own analysis leaf h:

- thin: area-median wall thickness < 4 h (inward rays);
- crevices: more than 1 % of the surface faces an exterior gap narrower than h
  (outward rays);
- a load patch within 3 h of a support;
- a support piece under 50 h^2, or a partial-component clamp on a section under
  4 h.

Also recorded, not a flag: the **support rigid-mode ratio**. It is
λ_min/λ_max of the area-weighted Gram matrix of the six rigid modes over the
clamped components of the supports, with rotations taken about the support
centroid in part-size units. Near 0 it means a near rigid-body mode.

The summary gives median, min and max over all sessions and over the unflagged
ones, with n per column.

## Reproduce

TetGen and pymeshlab are the extra dependencies:

```shell
conda activate mechfoundry-warp117
python -m pip install tetgen==0.8.4 pymeshlab
```

The CPU P1 queue runs in a throwaway venv (numpy, scipy, pyamg, MKL
PARDISO). No conda env is changed for it:

```shell
python -m venv output/bodyfitted_vs_nitsche/_cpu_env
output/bodyfitted_vs_nitsche/_cpu_env/bin/pip install numpy scipy pyamg pypardiso
```

The sessions come from the Opus batch, which is archived; extract it first with
`zstd -dc output/archives/agentic_batch50_opus.tar.zst | tar -x -C output`.

One session, on an idle GPU:

```shell
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/bodyfitted_vs_nitsche/compare_bodyfitted_nitsche.py \
  --session output/agentic_batch50_opus/<run>/workflow/<session_id> \
  --output output/bodyfitted_vs_nitsche/<object>
```

The whole run is the one command above. Phase by phase:

```shell
python experiments/bodyfitted_vs_nitsche/run_batch.py --phases solve
python experiments/bodyfitted_vs_nitsche/run_batch.py --phases evaluate summary
```

One session: `compare_bodyfitted_nitsche.py --session ... --output ... --phase solve`,
then `--phase evaluate`.

Outputs per session (`--output`):

- `comparison.json`: everything, plus the flat `row` the summary reads;
- `p1_solution_d<D>_<mode>.npz`: the stored P1 solution: nodes, tets, nodal u,
  the clamped-node mask, the per-side traction table, the nodal load vector, and
  the material and band radius;
- `nitsche_d<D>/`, `nitsche_d<D>.json`: the Nitsche runs;
- `meshes/`, `result_d<D>_<mode>.json`: cached meshes and results, reused by a
  re-run (`--recompute` ignores them).

**Every metric comes from stored solve outputs.** The Nitsche field is read
from `nitsche_d<D>/api_fem_forward-*/vtu/solution_t*.vtu`, and the P1 field from
`p1_solution_d<D>_<mode>.npz`, in phase 2 only.

`summarize.py` writes `<output>/_summary/summary.{md,csv,xlsx,json}`.

**The final run's summary is kept here, in [`summary/`](summary/).** It is the
`_summary/` of the fresh run of 2026-10-16 (`summary.{md,csv,xlsx,json}`,
`run_info.json`, `compare_vs_previous.md`). The run's full outputs, including
the CPU venv `_cpu_env`, are archived in
`output/archives/bodyfitted_vs_nitsche.tar.zst`, and `output/bodyfitted_vs_nitsche/`
was removed. Paths in these files still point at `output/bodyfitted_vs_nitsche/`,
and `output/bodyfitted_vs_nitsche_old`, named in the notes, no longer exists. To
re-evaluate or re-summarize, first extract the archive into `output/`:
`zstd -dc output/archives/bodyfitted_vs_nitsche.tar.zst | tar -x -C output`.

`python ... --help` lists every option.

## Compact tables and the H1 error

[`summary/tables/`](summary/tables/) holds the per-depth tables for the
Supplementary Information (50 rows each, depths 7 and 8): degrees of freedom of
the Nitsche octree and the three P1 meshes; accuracy ($\eta_{\mathrm{rig}}$,
$\eta^{H^1}_{\mathrm{rig}}$, compliance ratio, $s_{95}$/$s_{99}$ von Mises
percentile ratios); cost (solver and mesh retries, preparation and solve time
of the successful attempt with the all-attempts total in parentheses, solve
time to $10^{-2}$), with medians over all and unflagged objects.  Object names
and row order are those of the model-evaluation table
(`experiments/model_eval/evaluation_table.md`): the instruction verbatim from
`experiments/agentic_batch/objects_50.txt`, in its category order (categories
not shown).  The tables carry no notes: each caption defines the symbols, the
flags (T, G, L, S with thresholds), units, the parenthesized all-attempts
times, "—" (no body-fitted mesh) and the markers kept (C, CPU solve; R,
reduced GPU solver; †, converged only to 10⁻², excluded from medians), and
refers to the SI procedure section (`\ref{si:error-comparison}`).  Accuracy rows
with η_rig > 10 % against any mesh are gray (bold in the md).

- `tex/tab_si_bf_{dofs,accuracy,cost}_d{7,8}.tex`: fragments for `\input`, label
  `tab:si-bf-<kind>-d<depth>`.  Accuracy and cost: landscape (pdflscape)
  longtables with booktabs rules and a continued caption.  DOFs: one landscape
  page, the 50 rows in two halves side by side.  Packages pdflscape, longtable,
  booktabs, array, amsmath, xcolor (`\usepackage[table]{xcolor}`).  `tables.tex`
  holds all six; `tables_preview.tex` inputs them standalone (`pdflatex
  tables_preview.tex`, run twice).
- `tables.md` (same headers, names and captions, plus definitions and
  provenance) and `tables.xlsx` (raw values, successful and all attempts in
  separate columns, gray accuracy rows as in the PDF, a Captions sheet).
- `AUDIT.md`: an independent audit (2026-10-17) that re-extracted every printed
  number from the archived raw outputs; it passed, and its two caption points
  (the DOF match is a target, not a guarantee; the exact mesh-retry definition)
  are in the captions.
- `collected.json`: the records `tables.py` collected.  When the run's outputs are
  not extracted (they are archived), `tables.py` renders from it, so a
  presentation change needs no extraction.

The H1 error is an addition to `evaluate_p1` (`h1_errors`, `rigid_gradient`;
the existing metrics are unchanged): ||e - r*||_H1 / ||u_P1||_H1 with
||v||^2_H1 = int |v|^2 + int |grad v|^2, the same r* as the rigid-removed L2,
and the same quadrature points and per-method gradients as the stress metrics.
grad r* is the constant skew matrix of r*'s rotation, so the removal also acts on
the seminorm. The final run's `comparison.json` files predate it.

Recomputing needs the extracted run (`output/bodyfitted_vs_nitsche/`); `tables.py`
alone also runs without it, from `collected.json`:

```shell
# mechfoundry-warp117, one idle GPU (~20 min): re-evaluate every compared entry from
# the stored npz + VTU, add H1, check against comparison.json
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python experiments/bodyfitted_vs_nitsche/recompute_metrics.py
/home/xl2254/anaconda3/bin/python experiments/bodyfitted_vs_nitsche/tables.py
```

On 2026-10-17 all 282 compared entries recomputed bit-for-bit equal to the
stored metrics (`summary/tables/recomputed_metrics.json`).

## Notes kept from earlier rounds

- In float32, `F = I + grad u` keeps about three digits of a 1e-4 strain. The
  residual then floors near rel 1e-3, and the batch sessions (all float64) do
  not converge in float32, so both methods run in float64.
- The tet quadrature of the solve is pinned at order 2. Left to dispatch, Warp
  asks for tet rules it does not implement.
- A background mesh built by TetGen from the body's own surface made `-m` fail
  to locate points and segfault, so the graded sizing uses a box Kuhn grid.
