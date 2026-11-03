# Examples — Agent Guide

## Isolation contract

Each leaf may read only files in that leaf and files under repository `src/`.
Do not import a root runner, `illustrations/`, another example, or
`src/warp_wrapper/configs`. Every runner that calls `forward()`, `step()`, or
`optimize()` must own and load `config.py`. Resolve paths from `__file__` and
write outputs into the leaf.

The API-assisted FEM workflow is not an example leaf. Its launcher, `config.py`,
predefined scripts, state machine, schemas, API skill, and guarded monitor all
live under `src/agents_wrapper/fem_workflow`, and it writes only beneath
`output/fem_workflow/<session_id>`. No example may import it; original user
assets remain byte-for-byte unchanged.

`LatentShapeOpt` is reserved for
`optimization/latent_shape_opt/run.py`. If latents only supply
geometry for a forward solve, decode/filter the SDF, extract a surface, scale it
to physical coordinates, and pass it to `ShapeOptTrimmed` as
`remeshing_trimesh`.

Each leaf README distinguishes a bounded smoke command from a reference-aligned
command. A smoke command must still exercise the feature named by the example
and must preserve any mesh resolution required by its geometry or boundary
conditions. A reference-aligned optimization validation may limit iteration
count, but it must not coarsen the reference mesh or change the physical case.
Document the full reference schedule separately when it is needed to reproduce
the final trajectory or figure.

## Where to look before editing

| Task | Example | Production reference | Additional evidence |
|---|---|---|---|
| Image/text/scribble decoding | `geometry/image_to_octree`, `geometry/text_scribble_to_surface` | `src/triposg_wrapper/triposg_pipeline_wrapper.py` | vendored TripoSG inference scripts |
| Surface/latent conversion | `geometry/mesh_to_latent_octree` | `src/triposg_wrapper/utils.py` | `src/warp_wrapper/tests/test_latent_shape_opt.py` |
| Surface mesh to octree | `geometry/surface_mesh_to_octree` | `src/triposg_wrapper/trimesh_sdf.py` | `illustrations/sdfoctree/prep_octree_sdf.py` |
| Nitsche forward solve | `analysis/nitsche_comparison`, `analysis/bunny_nitsche` | `src/warp_wrapper/core/nitsche_field.py` | Nitsche tests and `illustrations/nitsche/` |
| Surface selectors | `boundary_conditions/manual_bunny_regions`, `automatic_bunny_ears` | `src/warp_wrapper/core/bc_surface_selectors.py` | selector tests and `illustrations/bcregion/` |
| Topology optimization | `optimization/topology_cantilever_*` | `src/warp_wrapper/core/topo_opt.py` | end-to-end examples tests |
| SDF shape optimization | `optimization/sdf_shape_cantilever`, `shape_opt_capture` | `src/warp_wrapper/core/shape_opt.py` | `illustrations/shapeopt/so_prep.py` |
| Latent optimization | `optimization/latent_shape_opt` | `src/warp_wrapper/core/latent_shape_opt.py` | latent-shape tests |
| Interior voids | `optimization/hidden_phase_field` | phase-field path in `shape_opt.py` | `illustrations/hiddenphi/hp_prep.py` |
| BC restoration/zipper | `boundary_conditions/bc_patch_zip*` | `src/warp_wrapper/core/remeshing.py` | remeshing tests and `illustrations/bczip*/` |
| Cost measurement | `benchmarks/overview_performance` | octree and shape-opt production paths | `illustrations/overview/ov_bench.py` |

Search the exact symbol with `rg` and read its tests before changing a call.
Illustration files are references, not runtime dependencies. Keep quick commands
bounded, but do not weaken the production defaults used by full examples.

The three explicit-surface shape-optimization examples
(`sdf_shape_cantilever`, `shape_opt_capture`, and `hidden_phase_field`) use an
illustration-aligned Newton absolute tolerance of `5e-7`.  With the default
depth-7 mesh and the physical `0.9 h_target` refinement band, remeshed solves
can plateau around `3e-7`; keep this example setting aligned with the three
optimizer captures while leaving the production solver default unchanged.

## API-assisted FEM workflow

`src/agents_wrapper/fem_workflow/run_workflow.py` uses the deterministic
LangGraph coordinator from `src/agents_wrapper`. A normal run makes only two
cached, JSON-only API calls. API output is validated data and must never become
arbitrary Python or shell. The production monitor may execute only the scripts
registered in that package, using argument arrays and paths contained in the
session. Keep API tests under `src/agents_wrapper/tests`, offline with injected
fake clients and mock GPU-heavy tools.
