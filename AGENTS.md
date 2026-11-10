# MechFoundry — Agent Guide

Start here. This file covers what the repository (MechFoundry, the paper's name for
it) does, where things live, and how to run the FEM solver, the API-assisted
workflow and its Streamlit UI. The deep references are [`README.md`](README.md)
(installation of every environment, paper results reproduction),
[`src/agents_wrapper/fem_workflow/README.md`](src/agents_wrapper/fem_workflow/README.md)
(workflow CLI/JSON protocol), [`examples/README.md`](examples/README.md) (runnable
leaves) and [`illustrations/AGENTS.md`](illustrations/AGENTS.md) (paper figures).

**To run a simulation of an object end to end, follow the operator skill:
[`src/agents_wrapper/fem_workflow/operator_skill/SKILL.md`](src/agents_wrapper/fem_workflow/operator_skill/SKILL.md).**
It is the skill the *AI operator* (a coding agent such as Claude Code) follows to
drive the workflow CLI from a one-line objective to a solved, checked and reported
case, through its runner `operator_skill/scripts/drive.py` (§5). Do not confuse it
with `fem_workflow/skill/`, the prompt pack of the workflow's own LLM specialists.

---

## 1. What the repository does

It is a GPU framework for simulating and optimizing **implicit geometry**: a shape
defined by a signed distance field (SDF), analysed on a non-conforming octree mesh
without body-fitted meshing. The pipeline has five stages:

1. **Geometry → SDF → octree.** A text prompt, image, scribble, CAD part or surface
   mesh becomes a surface (TripoSG, Hunyuan3D-2.1, Text2CAD, CAD-Coder or an upload),
   then an SDF, then a 2:1-balanced adaptive octree hex mesh refined near the
   surface. The image can also be drawn from text by SDXL on the geometry card.
2. **GPU FEM + density topology optimization.** Linear or neo-Hookean elasticity
   in NVIDIA Warp, with a Helmholtz filter, tanh-SIMP and GCMMA (`TopoOpt`).
3. **Diffused Nitsche boundary conditions** on the embedded, non-conforming mesh.
   BC regions are picked on the surface by composable selectors or by automatic
   segmentation (shape diameter, PartField, or CAD B-rep faces).
4. **SDF shape optimization** (`ShapeOpt` / `ShapeOptTrimmed`): boundary
   evolution on an adaptive octree with MC/BVH re-initialization, an optional
   hidden phase field that opens interior voids, and "BC-zip", which restores exact
   BC patches onto the changed body each iteration.
5. **Latent-space shape optimization** (`LatentShapeOpt`): the FEM loss is
   backpropagated through the TripoSG decoder into its latent code, with a ladder
   of pull-back-metric preconditioners (`precond_mode`: none / levelset / rownorm
   / jacobi / cg).

On top of the solver sits an **API-assisted FEM workflow**: a deterministic
LangGraph state machine that uses a few compact LLM calls (DeepSeek, Groq or
Gemini) to turn a text, image or scribble into a reviewed forward solve (or
shape optimization). It runs from a CLI or a Streamlit chat UI. **No
LLM-generated code or command is ever executed.** LLM output is validated JSON
data, and only predefined tools run. The one exception is CAD-Coder's CadQuery
script, which runs only inside `src/cad_wrapper/cadcoder_wrapper/sandbox.py`.

On top of the workflow sits the **AI operator**: a coding agent that follows the
operator skill (`fem_workflow/operator_skill/`) and drives the workflow CLI with
no human in the loop, deciding the case, checking every picture and reporting the
result. The paper's 50-object batches (Haiku, Sonnet, Opus) are runs of this
skill, launched by `experiments/agentic_batch/run_batch.sh`.

---

## 2. Folder map

```
README.md                 installation of all conda environments, tests, paper results reproduction, experiments
AGENTS.md                 this file
.assets/                  images used by README.md
src/
  warp_wrapper/           THE SOLVER (Warp FEM, adjoints, optimizers)
    core/                 production code, imported as `core.*` (see §3)
    configs/              dotdict configs for the classic examples (cantilever 2D/3D, Nitsche, shape-opt)
    tests/                pytest suite; run_all.sh runs it in two processes (AMGX quirk)
  triposg_wrapper/        TripoSG image/text→latent→SDF pipeline, hierarchical octree extraction, trimesh SDF
  agents_wrapper/         LLM provider clients (deepseek.py, groq.py, gemini.py, wrapper.py, environment.py)
    fem_workflow/         the LangGraph workflow runtime (see §5)
      graph.py            nodes and routes of the state machine (build_graph)
      models.py           pydantic state: WorkflowState, Stage, BCProposal, MaterialParameters, ShapeOptSettings…
      specialists.py      the LLM calls (one per purpose), cached per session in api_cache/
      monitor.py          ToolMonitor: the ONLY thing that executes tools; receipts + caching
      workers.py          start/stop resident warm workers (geometry, tools, solver, CAD, Hunyuan, PartField, VoMP)
      scripts/            workflow_tool.py (deterministic tool subcommands), tool_server.py (warm worker), generate_surface.py
      skill/              SKILL.md + references/*.md: the prompts the LLM specialists are given
      operator_skill/     THE OPERATOR SKILL the AI operator follows (SKILL.md; see §5):
                          references/ (one file per decision: geometry, boundary_conditions,
                          materials, solver_settings, visual_checks, cli_protocol/, objectives/, …),
                          scripts/ (drive.py runner, inspect_session.py, sessionlib.py, …), tests/
      run_workflow.py     CLI entry point (start / resume)
  ui/                     Streamlit chat frontend
    app.py                the page (conversation cards, 3-D viewer, tool window)
    controller.py         WorkflowController: adapter between Streamlit and the checkpointed graph
    browse.py             session file browsing/preview/zip (Streamlit-free, testable)
    telemetry.py          machine-load sampler + solver-progress reader for the tool window
    agent_session.py      optional agent sessions: Claude Code (Agent SDK) drives the operator skill (Streamlit-free)
  material_parts/         per-part materials → 64³ material field (same artifact format as VoMP)
  partfield_wrapper/      optional PartField segmentation (own env `mechfoundry-partfield`), with a server/worker
  cad_wrapper/            optional text→CAD: text2cad_wrapper/, cadcoder_wrapper/ (sandboxed), cad_common/
  hunyuan3d_wrapper/      optional Hunyuan3D-2.1 image→3D generator + surface painter (own env)
  vomp_wrapper/           optional VoMP volumetric material inference (own env `mechfoundry-vomp`)
  dependency/             third-party checkouts (TripoSG, amgx, pyamgx, PartField, Text2CAD, CAD-Coder,
                          Hunyuan3D-2.1, VoMP). All git-ignored, each cloned at a pinned commit
                          by README's install steps; TripoSG by src/triposg_wrapper/fetch_triposg.sh,
                          which also applies its one patch. Do not edit them.
examples/                 one independently runnable leaf per capability (geometry/, analysis/,
                          optimization/, boundary_conditions/, benchmarks/). Each has run.py (+ prepare.py)
experiments/              research studies behind the paper, not production (each has a README)
  agentic_batch/          run_batch.sh: one headless Claude Code session per object following the
                          operator skill (objects_50.txt = the paper's 50); realization/material/BC
                          sweeps for the showcase; tests/ with a stubbed claude
  model_eval/             three blinded evaluators grading the Haiku/Sonnet/Opus batches: the Fable
                          round (top level: prompts, eval_<k>.json, evaluation_table.{md,xlsx}) and
                          the GPT-6-Astra round (astra_high/, with agreement.md between the two)
  bodyfitted_vs_nitsche/  TetGen body-fitted P1 vs each Opus session's diffused-Nitsche solve;
                          summary/ (summary.{md,csv,xlsx,json}) and summary/tables/ (SI tables, .tex, AUDIT.md)
  geometry_cleanup/       short volume-as-is shape optimization of the 50 Opus bodies (defect clean-up)
illustrations/            paper figures: one subfolder per figure + build.sh; read illustrations/AGENTS.md first.
                          Besides the method figures (octree, bczip, nitscheconv, shapeopt, …):
                          challenge/ (challenge_overview), intro_overview/, operator/ (operator_loop),
                          gallery/ (agentic_gallery, README's opening figure), agentic_results/,
                          showcase/, overview/
output/                   all run outputs (git-ignored): output/fem_workflow/<session_id>/, example runs, …
  archives/               finished runs packed as <name>.tar.zst (see below)
```

**Archived outputs.** Large finished runs no longer sit in `output/` as folders; each
is packed in `output/archives/<name>.tar.zst`, with `<name>/` as its top-level
directory, and the archives are released on Zenodo:
<https://zenodo.org/records/23511564>. They are `agentic_batch50` (Sonnet), `agentic_batch50_haiku`,
`agentic_batch50_opus`, `bodyfitted_vs_nitsche`, `geometry_cleanup` and `showcase`.
Scripts and figure builds that read one of these folders (`model_eval`,
`bodyfitted_vs_nitsche`, `geometry_cleanup`, and `illustrations/build.sh` targets such as
`agenticresults`, `agenticcompare`, `gallery`, `showcase` (its `SC_*_RUNS` sweeps) and
`shapeoptmerged` when it re-renders its cached clean-up panels) still expect it at its
original path, so extract it there first:

```shell
zstd -dc output/archives/<name>.tar.zst | tar -x -C output    # → output/<name>/
```

Each optional `*_wrapper` follows the same pattern. It has an `availability.py`
with a `*_status()` that reports `ready` (discovered through env vars such as
`PARTFIELD_CONDA_ENV`/`*_ROOT`/`*_CHECKPOINT`), a `generate.py`/`segment.py`/
`inference.py` that runs **inside its own conda env**, and a `server.py` resident
worker. The workflow never imports these models into its own process.

---

## 3. Environments and conventions

| Conda env | Runs | Notes |
|---|---|---|
| `mechfoundry-warp117` | solver, TripoSG, examples, workflow tools, figures | Warp 1.17, Torch cu124, JAX-AMG/AMGX, NumPy 1.26 |
| `mechfoundry-agents` | workflow CLI, Streamlit UI, agent tests | LangGraph, provider SDKs, Streamlit. Needs `DEEPSEEK_API_KEY` (or `GROQ_API_KEY` / `GEMINI_API_KEY`) |
| `mechfoundry-partfield`, `mechfoundry-text2cad`, `mechfoundry-cadcoder`, `mechfoundry-hunyuan3d`, `mechfoundry-vomp` | optional models | each is incompatible with the solver env; launched as subprocess workers |

- **Run commands from the repository root** unless a README says otherwise
  (example leaves are run from their own directory).
- **Import styles differ by area.** Solver code is imported as `from core.shape_opt import
  ShapeOptTrimmed` after putting `src/warp_wrapper` on `sys.path`, as examples and
  `workflow_tool.py` do. Everything else uses `from src.<package> import …` with
  the repo root on `sys.path`.
- Solver parameters are `dotdict`s. `src/warp_wrapper/core/default_config.py` holds
  the documented defaults, and a user config overrides them. The solver's precision is
  float32 by default, and `use_mixed_precision=True` selects float64; the fem_workflow's
  settings card (`ShapeOptSettings`) defaults to float64 and always passes its own value. `cg_backend_assembled`
  ∈ `amgx | jaxamg | pyamgx | native`.
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, because JAX shares the GPU with
  Warp and Torch. The first run of anything is slow because Warp JIT-compiles its kernels.
- Never commit API keys, model weights, `output/`, or anything under
  `src/dependency/`.
- Code style: match the surrounding file. Docstrings and comments here are
  explanatory prose that says *why* a choice was made, often with measurements.
  Keep that density when you edit.

---

## 4. The solver (`src/warp_wrapper/core`)

Class hierarchy (key files):

```
AbstractField ─┬─ NeoHookean ── TopoOpt                     density topology opt (topo_opt.py, mma_solver.py: CCSAMMASolver/GCMMA)
               └─ OctreeField ─┬─ OctreeNeoHookean
                               └─ NitscheNeoHookean          diffused Nitsche BCs (nitsche_field.py, nitsche_kernels.py)
                                    └─ ShapeOpt ── ShapeOptTrimmed ── LatentShapeOpt
                                       (shape_opt.py)                  (latent_shape_opt.py + preconditioners.py)
```

- `bc_surface_selectors.py`: `SurfaceSelector` family (BoundingBox, Sphere,
  Cylinder, HalfSpace, Slab, Lowest/Highest, Percentile, Normal, Curvature,
  FlatRegion, SharpFeature, FloodFill, ConnectedComponent, ShapeDiameterSegment,
  and the Union/Intersect/Subtract/Complement combinators). BC regions are always
  expressed with these.
- `bc_surface_segmentation.py`: shape-diameter segmentation (`SurfaceSegmentation`).
- `remeshing.py`, `sdf_reinit.py`, `analytical_sdf.py`: octree rebuild, SDF
  re-initialization, BC-zip patch restoration, analytic SDFs for benchmarks.
- `amgx_solver.py`, `jax_amg_solver.py`, `condensation.py`: linear solvers.
- `material_field.py`: `SampledMaterial`, the spatial E/ν/ρ fields from VoMP or per-part materials.
- `vtu_writer.py`: VTU/surface output that ParaView and the UI read.

Quick ways to exercise it (env `mechfoundry-warp117`), taken from `examples/README.md`:

```shell
cd examples/optimization/topology_cantilever_2d && python run.py --n-iters 1 --nx 80 --ny 25
cd examples/optimization/shape_opt_capture     && python run.py --n-iters 1 --depth 7
cd examples/optimization/latent_shape_opt      && python run.py --n-iters 1 --init-octree-depth 4 --octree-depth 6 --latent-lr 5e-2 --latent-optimizer adam --disable-pullback
```

A one-iteration run is a smoke test of the same problem and does not replace the full
trajectory. The leaf READMEs give the reference settings. Results are written as VTU
series under `output/<project_name>-<timestamp>/`.

---

## 5. The API-assisted FEM workflow

### Driving it as the AI operator: the operator skill

An agent asked to simulate, analyse or stress-test an object should not type the
JSON cards below by hand. It follows
[`src/agents_wrapper/fem_workflow/operator_skill/SKILL.md`](src/agents_wrapper/fem_workflow/operator_skill/SKILL.md),
which turns a one-line objective into a solved, checked and reported case in a few
minutes. The agent writes its decisions into a plan file, and the runner
`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py run …`
answers the workflow's cards from it and stops wherever the agent must look (the
generated geometry, the BC regions, the solved fields). Other `drive.py` commands:
`start`, `status`, `show`, `history`, `answer`, `retry`, `rewind`, `blank` (a
dimensioned body built from boxes, cylinders, profiles or bars), `clone`,
`solve-backend`, `finish`, `judge` (the report), `gpu`. `inspect_session.py` reads a session;
`sessionlib.py` is the shared library.

| Part | Where | What |
|---|---|---|
| The skill | `operator_skill/SKILL.md` | rules and the normal path; each stop prints the reference sections to read next |
| References | `operator_skill/references/` | read on demand, one section at a time: `geometry.md`, `boundary_conditions.md`, `materials.md`, `solver_settings.md`, `visual_checks.md`, `judging_results.md`, `troubleshooting.md`, `cli_protocol/`, `objectives/`, … |
| Scripts | `operator_skill/scripts/` | `drive.py`, `inspect_session.py`, `sessionlib.py`, helpers (`backend_solve.py`, `remesh_blank.py`, `union_blank.py`, `paint_ahead.py`, `web_images.py`) |
| Tests | `operator_skill/tests/` | pytest, env `mechfoundry-agents`, no API calls (§6) |

Operator sessions live in `output/fem_workflow_operator/<session_id>/`, and the
operator writes only under that session's `operator/`. Batches of operator runs are
`experiments/agentic_batch/run_batch.sh` (README "Paper results reproducibility").
The skill is distinct from `fem_workflow/skill/`, the prompts of the workflow's own
LLM specialists.

### Stages (graph nodes in `graph.py`)

```
geometry_input ─(refine_geometry | refine_cad_geometry | refine_cadcoder_geometry)─┐
      │                                                                              │
      └─> generate_geometry → texture_geometry → describe_geometry → geometry_physics (scale/units)
                                                                            │
bc_input ─(refine_bcs)─> segment → describe_segments → synthesize_bcs → bc_values
      → material_input (uniform | per-part "parts" | VoMP "vomp") → shape_opt_input → build_case → solve
```

Each node that needs a human waits on a LangGraph **interrupt**. The CLI prints it
as JSON, and the UI renders it as a card. The state (`WorkflowState`, `Stage` enum in
`models.py`) is checkpointed to SQLite after every step, so any session can be
resumed or rewound to an earlier checkpoint.

### Execution model

- `specialists.py` makes the LLM calls. Each purpose has its prompt in
  `skill/references/*.md`. Responses are cached in the session's `api_cache/` and
  logged in `api_log.jsonl`. Transient failures are retried twice, and invalid JSON
  is repaired once.
- `monitor.py` (`ToolMonitor`) is the **only** executor. It runs subcommands of
  `scripts/workflow_tool.py` (`stage-asset`, `inspect-surface`, `scale-surface`,
  `render-surface`, `segment-surface`, `render-segmentation`, `render-part-montage`,
  `build-part-field`, `plain-colour-surface`, `resolve-bcs`, `build-case`,
  `export-surface-field`, `run-forward`) in the compute env. It writes a receipt plus stdout/stderr per call
  to `tool_receipts/` and caches results by input digest (except
  `segment_surface`, `resolve_bcs` and `run_forward`).
- `workers.py` keeps resident warm processes alive: a tool server, the geometry
  generator, a spare solver process, and the CAD, Hunyuan3D, PartField and VoMP
  workers. These are purely an optimization. The monitor falls back to one-shot
  launches when no worker is serving.
- `run-forward` builds a `ShapeOptTrimmed` from the `case/` JSON and solves it
  (forward only, or `run_mode: "shape_optimization"`).

### Session folder: `output/fem_workflow/<session_id>/`

`session.json` (current state) · `checkpoints.sqlite` · `ui_history.json` ·
`api_cache/`, `api_log.jsonl` · `tool_receipts/` · `geometry/` · `previews/` ·
`boundary_conditions/` · `materials/` · `case/` · `solve/` (the
`api_fem_forward-<timestamp>/` VTU run, plus per-field `.glb` surfaces such as
`displacement_*`, `stress_von_mises` and `strain_von_mises`, each with `.json` range and `.disp.bin`
deformation sidecars) · `versions/` (every version of the files under `inputs/`,
`geometry/`, `boundary_conditions/`, `materials/` and `case/`, by content; each
checkpoint records which it had, and a rewind puts them back). Read these files to debug a session.

### CLI (env `mechfoundry-agents`)

```shell
python src/agents_wrapper/fem_workflow/run_workflow.py start --compute-env mechfoundry-warp117
python src/agents_wrapper/fem_workflow/run_workflow.py start --provider deepseek \
  --vision-provider groq --vision-model qwen/qwen3.8-27b --compute-env mechfoundry-warp117
python src/agents_wrapper/fem_workflow/run_workflow.py resume SESSION_ID
```

Options: `--output-root`, `--model`, `--vision-model`, `--octree-depth` (default
7), `--session-id`. At each interrupt you type one JSON object. The full set of
examples is in the workflow README. In brief:

```json
{"description":"a bracket with two mounting feet","asset_path":"sketch.png","asset_kind":"scribble"}
{"action":"draw_image","description":"a sturdy oak bench, a thick plank seat on two slab legs","image_seed":0}
{"description":"a sturdy oak bench, a thick plank seat on two slab legs","use_drawn_image":true}
{"up_axis":"z","physical_extent":0.25,"unit_system":"SI (m-kg-s)"}
{"action":"preview_regions","description":"Fix the feet and apply a downward traction to the top"}
{"action":"approve_regions"}
{"action":"infer_values","value_description":"Fully fix the feet, 1 MPa downward traction"}
{"material_mode":"uniform","youngs_modulus":7e10,"poisson_ratio":0.33,"density":2700.0}
{}
```

The last `{}` accepts the default `ShapeOptTrimmed` settings (forward solve).

Gotchas:
- Text2CAD and CAD-Coder **require** `"revise_prompt": true` first, because they
  do not know object names.
- BC value names must match the region names printed by the preview.
- `physical_extent` also sets resolution: leaf size = `1.92 · extent / 2^octree_depth`.
- Spatial material fields (`vomp`, `parts`) are forward-solve only.
- Settings-card defaults (`ShapeOptSettings` in `models.py`) differ from the solver
  core's: float64 (`use_mixed_precision: true`, because float32 stalls at a residual
  floor on stiff everyday cases), `gd_lr` 0.01 (length², calibrate it to the part)
  and `sdf_bc_fix_method` `"mask"`. The workflow offers only `"mask"` and `"zip"` for
  BC preservation; the core's `"graft"` is retired there, and a stored one loads as
  `"mask"`.
- A session runs one solve at a time, in every front end: the solve holds flocks in the
  session folder (`.solve_claim.lock`, `.solve_run.lock`; `runtime_files.py`), and a
  duplicate gets `SolveAlreadyRunning`, which is not a failed solve. Opening a session in
  the UI (`controller.open`) runs no step; `resume` runs the pending one. *Run FEA* turns into
  *Stop FEA* while one runs, which ends it (`runtime_files.stop_solve`), which is not a failure either: the
  written iterations stay and the settings card returns.
- `gd_lr_auto` (off by default) sizes the SDF step at every update to
  `gd_lr_auto_fraction · ε / max|dJ/ds|` (fraction 0.5), so `gd_lr` is then unused;
  such a run ends at `max_optimize_iters`, not at a relative tolerance.
- A failed optimization keeps the viewer on its last iteration that solved
  (`optimization_status.json`), and its error recommends a forward solve first, then
  `gd_lr = 0.5 ε / max|dJ/ds|` from the last step (`step_size` in that status file)
  or turning on `gd_lr_auto`.
- `linear_solver` on the settings card: `"jaxamg"` (default; the solver's
  `cg_backend_assembled: "amgx"`) or `"pyamgx"`, which a JAX-AMG failure recommends.
  Each solve runs in its own process, which pyamgx needs.

### Streamlit UI (env `mechfoundry-agents`)

```shell
streamlit run src/ui/app.py
# remote: bind to loopback and tunnel
streamlit run src/ui/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true
ssh -N -L 8501:127.0.0.1:8501 USER@REMOTE_HOST      # then open http://127.0.0.1:8501
# another sessions root (default output/fem_workflow), with its workers pinned to GPU 1
FEM_WORKFLOW_OUTPUT_ROOT=/abs/root CUDA_VISIBLE_DEVICES=1 streamlit run src/ui/app.py
```

- **Sidebar**: conversation list (new, resume, delete) and setup (provider/model
  probes, compute-env probe).
- **Main column**: one card per interrupt. There are upload and scribble canvases, an
  SDXL image generator beside the image upload, editable prompts with a *Revise prompt*
  button, region previews, BC value, material and texture cards, and an editable
  `ShapeOptTrimmed` settings card with *Run FEA*.
- **3-D viewer**: the generated surface, segmentation and solved fields as GLB, with a
  deformation scale.
- **Tool window** sections: *Resources* (GPU/CPU load), *Solver* (live progress
  of the solve log), *Backend* (tool/worker activity), *Files* (browse, preview
  and zip-download the session folder).
- The UI uses the same checkpointed graph as the CLI (`controller.py`). A session
  started in one can be resumed in the other.
- **Agent sessions** (`agent_session.py`, optional): *Start agent session* on the setup
  card runs Claude Code through the Claude Agent SDK (one loop thread per server, a client per
  conversation) on the operator skill, with the agentic batch's allow/deny lists and an
  Allow/Deny card for anything else. The setup card's choices reach `drive.py start` as
  `FEM_OPERATOR_PROVIDER`/`_MODEL`/`_VISION_PROVIDER`/`_VISION_MODEL`/`_COMPUTE_ENV`, and its
  sessions are written to the UI's own root (`FEM_OPERATOR_OUTPUT_ROOT`). The page shows them
  through `controller.peek` (reads the checkpoint, writes nothing, starts no worker), never
  `open`, which could re-offer a card under the agent's running step; sidebar clicks on an
  agent-driven session go to its agent view. Conversations are stored in
  `<root>_agent/<id>/` (`meta.json`, `chat.jsonl`, `uploads/`). Offered only when
  `agent_status()` is ready (the SDK importable and a `claude` CLI found).

The UI cannot be imported outside a Streamlit runtime. Put testable logic in
`controller.py`, `browse.py` or `telemetry.py`, not in `app.py`. To see a UI change working,
launch it headless and drive it with a browser. The test suite
(`src/agents_wrapper/tests/test_fem_workflow.py`) covers the graph and controller
paths without any API calls.

---

## 6. Tests

```shell
# mechfoundry-warp117
(cd src/warp_wrapper && bash tests/run_all.sh)        # two pytest processes: JAX-AMG and pyamgx cannot share one
python -m pytest src/triposg_wrapper/tests

# mechfoundry-agents
python -m pytest src/agents_wrapper/tests src/partfield_wrapper/tests \
  src/cad_wrapper/text2cad_wrapper/tests src/cad_wrapper/cadcoder_wrapper/tests src/hunyuan3d_wrapper/tests
python -m pytest src/agents_wrapper/fem_workflow/operator_skill/tests   # operator skill; -m slow adds the real MeshLab remesh tests

# each optional env runs its own wrapper tests, e.g. mechfoundry-vomp:
python -m pytest src/vomp_wrapper/tests
```

Run the relevant suite after changing solver or workflow code. For a solver change,
also run the affected example leaf with its bounded command.

---

## 7. Figures

Paper figures live in `illustrations/`, one subfolder per figure, and are built
with `illustrations/build.sh <target>` using Inkscape for SVG, matplotlib, and
headless ParaView `pvbatch`. Only the final `NAME.pdf/png` sit in `illustrations/`
itself. **Read `illustrations/AGENTS.md` before touching anything there.** It has
the toolchain choice, house style (Arial everywhere), the GPU env vars without
which ParaView silently falls back to software rendering, and per-figure lessons.
The figures built from the operator batches are documented in
[`illustrations/README.md`](illustrations/README.md) (agentic results, model comparison,
showcase) and in `illustrations/AGENTS.md` §5m (the 50-object gallery). Their batch
folders are archived (§2, "Archived outputs") and must be extracted before a rebuild
that re-renders them. The newer schematic figures (challenge §5k, `intro_overview`
§5k-bis, `operator_loop` §5l) have their own sections there.

---

## 8. Working rules for agents

- Do not modify `src/dependency/`. Wrap third-party code from a `*_wrapper`
  package instead. The one standing source edit, TripoSG's, is kept as
  `src/triposg_wrapper/patches/triposg.patch` (with the sha256 manifest
  `triposg.sha256` beside it) and applied by `fetch_triposg.sh`, because the checkout
  itself is not in git: a fix made in the checkout is lost on the next fresh clone.
  Should another edit ever be unavoidable, change the patch and regenerate the
  manifest in the same commit.
- Keep the workflow's safety boundary. LLM output stays data validated by
  pydantic models. Any new capability becomes a predefined `workflow_tool.py`
  subcommand run through `ToolMonitor`. Never add a code path that executes
  model text.
- New tool subcommands automatically work in the warm `tool_server.py`, because it
  reuses `workflow_tool`'s parser. Add a new tool to `DIRECT_LAUNCH` /
  `WORKER_TOOLS` / `NEVER_CACHED` in `monitor.py` as appropriate.
- Optional models must stay optional. Gate them on their `*_status().ready`, run
  them in their own env, and never import them in `mechfoundry-agents` or
  `mechfoundry-warp117`.
- Put scratch outputs under `output/` or a temp dir, never in the source tree.
- GPU memory is shared between Warp, Torch, JAX and any resident workers. If a
  run OOMs, check for workers left running from UI sessions (`workers.py` has
  `stop_*` helpers).
