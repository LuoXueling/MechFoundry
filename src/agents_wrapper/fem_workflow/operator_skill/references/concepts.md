# What happens behind the cards

## The pipeline

```
text / image / scribble / mesh
  → generate (TripoSG | CAD-Coder | Hunyuan3D) → geometry/generated.glb
  → inspect (watertight? components?) → optional paint (Hunyuan3D paint stage)
  → describe (1 vision call: geometry_overview, used by later prompts)
  → scale (uniform scale to the physical extent, min corner at the origin) → geometry/physical.glb
  → segment (PartField | shape diameter | CAD faces) → boundary_conditions/segments.*
  → describe segments (1 vision call: "0: <part>; 1: <part>; ...")
  → synthesize BCs (1 text call → BCProposal: selector trees per region)
  → resolve (selectors → face masks on physical.glb) → boundary_conditions/resolved.*
  → values (tractions, prescribed displacements, body loads) → proposal.json
  → material (uniform | per-part field | VoMP field) → case/material.json
  → settings → case/case.json → run_forward (ShapeOptTrimmed) → solve/<run>/
  → field export (GLB coloured by displacement / stress / strain) → solve/*.glb (+ .json, .disp.bin)
```

- **The LLM only returns validated JSON data**: prompts, selector trees, numbers. A
  deterministic tool monitor (`monitor.py`) runs every computation; the one exception is
  CAD-Coder, whose CadQuery code runs in a sandbox.
- **Tool calls are cached by input digest**: a repeated step with unchanged inputs is free.
  `segment_surface`, `resolve_bcs` and `run_forward` are never cached.
- **Each tool writes a receipt** to `tool_receipts/<key>.json`, with its stdout and stderr logs.
  Judge success by `returncode`.

## Conventions

- **Up is +y and the object faces +z**: front +z, back −z, left and right ±x. Every LLM prompt
  and selector uses this. "Downward" means −y.
- **`up_axis` on the scale card is the *reference axis*** along which `physical_extent` is
  measured, not a claim about orientation. A body 2 m long along x: `up_axis: "x",
  physical_extent: 2.0`.
- **The only surface load is a traction: one fixed vector over the whole region**, in force per
  area. A downward push on a top face is `[0, −p, 0]`. There is no pressure load (one acting
  along each face's own normal); on a flat face the two coincide. On a curved region the traction
  keeps its one direction, each point weighted by the cosine between its normal and the region's
  average normal: a load per projected area, not a pressure. A pressure on a whole curved wall or
  in a closed cavity cannot be applied; the nearest case is the traction along the average
  normal, and the report says so.
- **Dirichlet `components` are 0/1/2 = x/y/z.** `value` is one displacement shared by those
  components; zero means held.
- **Gravity goes in `acceleration`**, e.g. `[0, −9.80665, 0]` in SI, and needs a density.
  `body_force` is a direct force per volume.

## Unit systems

The solver has no unit logic: every number must be in the session's system.

| System | Length | Force | Stress / E / traction | Density | Acceleration | g |
|---|---|---|---|---|---|---|
| `SI (m-kg-s)` | m | N | Pa | kg/m³ | m/s² | 9.80665 |
| `SI-mm (mm-tonne-s)` | mm | N | MPa | tonne/mm³ (1 kg/m³ = 1e−12) | mm/s² | 9806.65 |
| `US-inch (in-lbf-s)` | in | lbf | psi | lbf·s²/in⁴ (1 kg/m³ = 9.357e−8) | in/s² | 386.09 |

Dimensional settings: `absolute_tol` is in force units; `gd_lr` in length² (solver_settings.md
§4). Outputs below 1e−6 stress units are written as zero.

**Default to SI with the object's real size in metres.**

## Resolution

- **Leaf size** h = 1.92 × (largest physical extent) / 2^`remesh_octree_depth`: the part spans
  about 67 leaves at depth 7, 33 at depth 6. The card's `strain_floor` and the inspector report
  it.
- **Refinement.** The finest leaves hug the surface; the interior coarsens to
  `remesh_init_octree_depth` (4 by default).
- **Thin members.** A member under about 2 leaves is too stiff in bending (warned, and reported as
  a resolution caveat). Below about one leaf it vanishes from the marching-cubes surface.
- **Accuracy.** The displacement error is first order in h (about 0.2–0.4·h/L on a benchmark);
  at depth 7 a body measured 3 % stiffer than a body-fitted reference. Depth 7 is a sound default.
- **Cost.** Depth 8 has about 4× the nodes of depth 7, about 3× the VRAM, and about 3.5× the
  time per optimization iteration.

## Diffused Nitsche boundary conditions

The body is an SDF on a non-conforming octree, with φ = ½(1 − tanh(d/ε)) and ε =
`phase_field_epsilon_factor`·h.

- **Supports** are enforced weakly by a Nitsche penalty β on the constrained components, derived
  per element from its eigenvalues, floored at 0.2(λ+2μ)/h, and cappable with `beta_max_over_e`
  (uniform material only). A held region therefore shows a tiny non-zero displacement. That is normal,
  and should be under about 5 % of the peak.
- **A selected region** is widened to quadrature points within max(1.5ε, 0.75h) of its faces, so
  regions under about 3 leaves wide blur. The widening moves where a load lands, not how much.
- **Loads** are applied with one average normal n̂ per region, weighting each point by
  max(0, n·n̂); a total force is divided by that cosine-weighted `loaded_area`, and each load
  region's traction is scaled so its total is exactly traction × `loaded_area`. Keep load
  regions single-facing.
- **Trimming.** Exterior elements are trimmed, so only your supports hold the body; there is no
  hidden clamp on the bounding box.

## Material model and solve

- **Material.** Compressible neo-Hookean everywhere, μ = E/2(1+ν), κ = E/3(1−2ν). Under small
  loads it is linear elasticity and Newton converges in 1–2 iterations. No plasticity: yield is
  judged by you (judging_results.md).
- **Newton** stops when ‖R‖/‖R₀‖ < `relative_tol` or ‖R‖ < `absolute_tol`, from iteration 1 on.
  `load_steps`: solver_settings.md §1.
- **Linear solver.** AMG-preconditioned CG (JAX-AMG / AMGX), float64 by default
  (`use_mixed_precision: false` is float32). A linear solve that ends above
  its tolerance (checked as ‖b − Kx‖/‖b‖) raises, and the message reports how far off it was.
- **Float32 strain floor.** A run needs strain ≥ 10·1.19e−7/`relative_tol` (1.2e−4 at the
  default); below it rel stays flat after iteration 1: **stiff materials under light loads need
  mixed precision** (the default).

## Outputs of a solve

`solve/api_fem_forward-<timestamp>/`: `log.txt` (Newton history, node counts, config, errors),
`vtu/` (volume fields), `vtu_sdf/`, `vtu_surface/` (fields on the marching-cubes surface),
`glb/` and `*.pvd` (for ParaView).

The field GLBs `solve/<field>_<component>[_sNN].glb` are coloured blue→red over
`[minimum, maximum]` in their `.json` sidecar; `.disp.bin` holds the displacements for deformed
views. Surface stress is neo-Hookean Cauchy stress, with each node's own material in field
modes; strain is small-strain Voigt.

A failed solve leaves `solve/failed.json` and deletes the field GLBs (cli_protocol/cards.md §2).

## Shape optimization

- **Objective.** J = SE/SE₀ + w_vol·softplus₅₀₀((V/V₀ − f)/|1 − f|), with SE the strain energy,
  V₀ the initial volume and f `target_volume_fraction` (≤ 2). While V > f·V₀ the penalty removes
  material and SE rises; near the target the volume alternates with period 2 and SE falls as the
  layout improves. `volume_as_is: true` only stops growth above V₀.
- **Update.** Each iteration: a forward solve, an adjoint, a gradient step
  `d ← d − gd_lr·∂J/∂d` on the SDF, a marching-cubes re-initialisation, a remesh.
- **Frozen regions.** BC regions are frozen. The default `sdf_bc_fix_method: "mask"` holds the
  SDF there (BC faces may drift slightly). With `"zip"` the
  exact iteration-0 BC patches are stitched back into the new surface every remesh, so supports
  and load faces keep their shape. Zip needs BC patches that are clean disks attached to the body.
- **Outputs.** Per iteration the log has `SE=… Vol=… Total=…` and `Opt iter k: obj=…`, and one
  saved surface (`glb/surface_tNNNN.glb`).

## The session folder

`<output_root>/<session_id>/`: `session.json` and `checkpoints.sqlite` (state), `api_cache/` and
`api_log.jsonl` (LLM calls), `tool_receipts/`, `inputs/`, `geometry/`, `boundary_conditions/`,
`materials/`, `case/`, `solve/`, and `operator/` (the driver's logs, renders and report).
Workers and the first-call cost: gpu_usage.md.
