# API-assisted FEM workflow

The deterministic LangGraph runtime, its predefined geometry and FEM scripts, and
the launcher all live in this package. A few compact API calls turn a
text/image/scribble concept into a reviewed `ShapeOptTrimmed` forward solve. No
LLM-generated code or command is executed.

Sessions are written to `output/fem_workflow/<session_id>` at the repository root;
pass `--output-root` to either subcommand to put them somewhere else.

## Install and configure

Follow **API-backed agents** in the [repository README](../../../README.md) to
create `mechfoundry-agents` and set an API key. Geometry and FEM tools run in the
compute environment (`mechfoundry-warp117`) through its own Python interpreter, so
the two environments do not need matching packages.

Default models per provider (override with `--model` and `--vision-model`):

| Provider | Text calls | Image calls |
|---|---|---|
| `deepseek` (default) | `deepseek-flash` | `deepseek-flash` |
| `groq` | `openai/gpt-oss-120b` | `qwen/qwen3.8-27b` |
| `gemini` | `gemini-2.5-flash` | `gemini-2.5-flash` |

## Command-line workflow

```shell
python src/agents_wrapper/fem_workflow/run_workflow.py start --compute-env mechfoundry-warp117
python src/agents_wrapper/fem_workflow/run_workflow.py start \
  --provider gemini --vision-provider groq --compute-env mechfoundry-warp117
python src/agents_wrapper/fem_workflow/run_workflow.py resume SESSION_ID
```

A new session first probes each selected model. If one is unavailable, the CLI asks
for a replacement such as `{"provider":"gemini","model":"gemini-2.5-flash"}`; it
never switches providers silently.

The CLI prints small JSON descriptions of requested input. Enter JSON for structured
questions, for example:

```json
{"description":"a bracket with two mounting feet","asset_path":"sketch.png","asset_kind":"scribble","condition_on_text":true}
{"description":"a hex nut","generator":"text2cad","revise_prompt":true}
{"description":"a red ceramic mug","asset_path":"photo.png","asset_kind":"image","generator":"hunyuan3d"}
{"action":"draw_image","description":"a sturdy oak bench, a thick plank seat on two slab legs","image_seed":0}
{"description":"a sturdy oak bench, a thick plank seat on two slab legs","use_drawn_image":true}
{"up_axis":"z","physical_extent":0.25,"unit_system":"SI (m-kg-s)"}
{"action":"preview_regions","description":"Fix the feet and apply a downward traction to the top","segmentation":{"num_clusters":5,"num_rays":25,"smoothing":0.25,"cone_angle_deg":120}}
{"action":"approve_regions"}
{"action":"infer_values","value_description":"Fully fix the feet, apply 1 MPa downward traction, and include Earth gravity"}
{"dirichlet_values":{"fixed_feet":{"components":[0,1,2],"value":0}},"neumann_values":{"top_load":[0,-1000000,0]},"body_force":[0,0,0],"acceleration":[0,-9.81,0]}
{"material_mode":"uniform","youngs_modulus":70000000000.0,"poisson_ratio":0.33,"density":2700.0}
{}
```

- **Text2CAD and CAD-Coder need a revised prompt.** Neither knows object names:
  "a hex nut" sent as is comes back a box. Send the description with
  `"revise_prompt":true` first (**Revise prompt** in the UI), review the rewrite
  the card returns, then send that to generate. The card warns until the text in
  it is that generator's revision, and the CLI prints a warning otherwise.
- `"generator":"text2cad"` builds a CAD part from the text alone with the optional
  [Text2CAD](../../../README.md) generator. TripoSG is the default and the only one
  that takes a scribble. Only one generator model stays loaded at a time.
  Text2CAD knows shapes, not object names ("a hex nut" alone gives a box), so its
  `revise_prompt` uses its own revision, separate from TripoSG's: it rewrites the
  name as a short geometric sentence ("A hexagonal prism with a circular hole
  through its center."). Review it, then send it to generate.
- `"generator":"cadcoder"` uses the optional CAD-Coder instead: it writes a CadQuery
  script, which runs only in a sandbox (see the repository README). Its
  `revise_prompt` has the API model write the expert-level description, with
  explicit sketch coordinates, that CAD-Coder follows; send that to generate.
- `"generator":"hunyuan3d"` uses the optional
  [Hunyuan3D-2.1](../../../README.md) generator for an image instead of TripoSG.
  It keeps the object's colour on the surface, so the renders the vision model
  reads are in colour too. `generation_settings` takes `inference_steps`
  (default 30), `hunyuan3d_octree_resolution` (256 or 384), `texture` (false for
  shape only), and three speed switches: `hunyuan3d_flashvdm` (default true, about
  half the shape time), `hunyuan3d_texture_size` (1024, 2048 or 4096; default 1024)
  and `hunyuan3d_upscale` (RealESRGAN, default false). The defaults take about
  55 s per image once the model is loaded, against about 110 s for upstream's.
- **An image drawn from text.** When SDXL is installed (the Hunyuan3D painter's
  reference model, in the `mechfoundry-hunyuan3d` environment; the card's `sdxl` says
  whether it is ready), `{"action":"draw_image","description":…,"image_seed":…}`
  draws one object from the description (the card's, when the answer leaves it
  out), cuts it out from its background with TripoSG's RMBG-1.4, and returns to
  the geometry card with it as `drawn_image` (the drawing as drawn is
  `drawn_image_raw`). Draw again with another seed or other words, or send
  `"use_drawn_image":true` (or its path as an `image` input) to any image
  generator: TripoSG's image mode or Hunyuan3D-2.1. TripoSG's text mode has no
  picture to follow, and its image mode follows one closely, so this is the
  reliable route when a text-only shape comes back missing or fusing parts. It
  takes about 13 s and 8–13 GB of GPU memory. **Generate with SDXL** beside
  the upload does the same in the UI.
- `generation_settings.erosion_radius` (TripoSG, default 0) is how many voxels its
  largest-component filter erodes before choosing the one body to keep. A positive
  radius cuts apart pieces joined only by something thinner than twice that many
  voxels, so thin members (slats, spindles, rails) and whatever hangs from them
  are dropped; 0 keeps every part joined to the body, and floating fragments
  still go.
- A Text2CAD or CAD-Coder part is segmented by its own B-rep faces by default (`"method":"cad_faces"`,
  one segment per face, each named by surface type such as `cylinder`). Set
  `"method"` to `partfield` or `shape_diameter` in `segmentation` to segment it
  another way.
- BC names in the value object must match the names printed by the region preview.
  A region the answer leaves out keeps the value the card shows. A load may be given
  as a total force, `"top_load":{"force":[0,-981,0]}`; it is divided by the region's
  *loaded* area (the solver loads a region along its average normal, weighting each
  face by the cosine to it), so the force that reaches the part is the one stated.
- `infer_values` is optional: it spends one API call and returns to the same step so
  you can review the inferred numbers before applying them.
- Every card keeps what it shows for any field an answer leaves out: the scaling
  axis, extent and units, the segmentation settings, the material mode and values,
  and the solver settings. `{"physical_extent":0.3}` changes the extent alone.
- A rejected answer does not cost the card. The CLI prints `answer_rejected` with
  the reason and shows the same card again; `resume` does the same for a session
  left on a rejected answer.
- The final `{}` accepts the default `ShapeOptTrimmed` settings in forward-only mode.
  Set `run_mode` to `shape_optimization` there to run the optimization loop
  (`load_steps` then ramps the first shape's solve). The card's defaults solve in
  float64 (`use_mixed_precision: true`), and an optimization steps with
  `gd_lr: 0.01` (units of length², so calibrate it to the part) and keeps BC
  patches with `sdf_bc_fix_method: "mask"` (`"zip"` is the alternative;
  the solver core's `"graft"` is not a workflow option, and a session or plan
  that stored it loads as `"mask"`). An optimization exports the displacement GLB as it runs, and a
  failed one keeps the viewer on the last iteration that solved: the failed
  iteration's outputs move to `failed_iteration/` in its run folder, and
  `optimization_status.json` beside its log records `failed_iteration` and
  `shown_iteration`; its error ends with a recommendation: run the same setup as a
  forward solve first, then retry with a smaller `gd_lr`, sized so the last step's
  largest nodal SDF move (`gd_lr · max|dJ/ds|` over the free design nodes, logged by
  the loop) stays under 0.5 ε: `gd_lr = 0.5 ε / max|dJ/ds|` (a tenth of the current
  rate when no step was taken). The numbers are under `step_size` in the status.
  `gd_lr_auto: true` (off by default) applies that rule by itself at every update:
  iteration k steps at `gd_lr = gd_lr_auto_fraction · ε / max|dJ/ds|_k` (fraction 0.5,
  at most 2), so the card's `gd_lr` is not used. Each rate is logged, the latest (and
  the range) is shown in the UI's Solver panel, and all are recorded under
  `gd_lr_auto.steps` in the status. This is normalized gradient descent: every step
  moves the surface by the fraction of ε at its largest node, however small the
  gradient, so the design oscillates near an optimum instead of settling and the run
  ends at `max_optimize_iters`; a lower fraction gives a finer final shape.
  `linear_solver` picks the AMG backend of each Newton step: `"jaxamg"` (the
  default, and what a session stored without the key runs; it is the solver's
  `cg_backend_assembled: "amgx"`, which uses pyamgx only when JAX-AMG cannot be
  imported) or `"pyamgx"`, which keeps its last iterate when AMGX stops short so
  Newton's residual decides (a JAX-AMG linear failure recommends it, except a
  singular one). Every solve runs in a process of its own, as pyamgx
  needs (it cannot follow JAX-AMG's AMGX in one process). After a solve the settings
  card stays open so the case can be run again; `{"action":"finish"}` ends the
  session.
- An STL upload is welded into a GLB when it is staged, since STL keeps no shared
  vertices; GLB, OBJ and PLY are staged as they are (the viewer draws a GLB preview
  of an OBJ or PLY).
- A session folder can be copied or moved and resumed where it now is: its recorded
  paths are rebased onto the folder it is opened from.

`physical_extent` also sets the mesh resolution. The octree root cube is `1.92 x`
the largest physical extent, so the leaf size is `1.92 * extent / 2**octree_depth`,
matching
[`shape_opt_capture`](../../../examples/optimization/shape_opt_capture) and
[`hidden_phase_field`](../../../examples/optimization/hidden_phase_field). Override
the ratio with `scale-surface --cube-pad`.

### API usage

A normal run makes the availability probes, two image calls (describing the
generated geometry and the segmented regions), and one call to synthesize BC
selectors. Optional extras cost one call each: revising the geometry or BC prompt
(`"revise_prompt":true`, or **Revise prompt** in the UI; required, not optional,
for Text2CAD and CAD-Coder) and inferring scale, BC
values, or material. Transient failures are retried twice and
invalid JSON is repaired once; after that the card that made the call reports the
failure under its button and stays editable, and clicking again retries. The
providers are fixed for the run; switching one means starting a new conversation.

## Chat frontend

```shell
conda activate mechfoundry-agents
streamlit run src/ui/app.py
```

The frontend replaces CLI JSON entry with chat history, upload controls,
visualization, and an editable `ShapeOptTrimmed` settings card. It uses the same
checkpointed workflow and compute environment as the CLI.

It lists the sessions under `output/fem_workflow`; set `FEM_WORKFLOW_OUTPUT_ROOT`
to open another root, such as a CLI run's `--output-root`. Resident workers are
shared by every front end on the same root and inherit the environment of the
process that starts them, so `CUDA_VISIBLE_DEVICES=1 streamlit run src/ui/app.py`
keeps this page's generators, tools and solves on GPU 1.

The two cards that offer to paint a surface (the texture card after generation
and the material card) start Hunyuan3D-2.1's paint stage and SDXL's reference
models when entered, without the shape stage: about 20 s and 17 GB, after which a
paint takes about 33 s. The worker stays while the card is open, including
beside a VoMP run when the GPU has room for both (42.5 GB at VoMP's peak), and
any other card stops it.

A painter that crashes on a surface (a fatal signal such as SIGSEGV, in its own run
or in the worker) is recorded in `tool_receipts/paint_crashes.jsonl` and never run
again on the same surface. One other seed is tried. After that, the
`plain-colour-surface` tool gives the surface one flat colour read from the
texture words, marked as `texture_fallback` on the card. A paint killed by
SIGKILL (out of memory, or a user) is reported as such and not counted.
