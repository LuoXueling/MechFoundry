# Examples

The API-assisted workflow now lives with its runtime in
[`src/agents_wrapper/fem_workflow`](../src/agents_wrapper/fem_workflow), not as an
example leaf; see its README for the LangGraph installation and Groq/Gemini setup.

Each leaf is independently runnable and imports only its own files plus
production code under `src/`. Run commands from the listed leaf directory.
Every primary workflow is named `run.py`; `prepare.py` creates a separately
reusable input.

| Area | Leaf | Bounded command |
|---|---|---|
| Geometry | `geometry/image_to_octree` | `python run.py --init-octree-depth 4 --octree-depth 6` |
| Geometry | `geometry/mesh_to_latent_octree` | `python run.py --init-octree-depth 4 --octree-depth 6` |
| Geometry | `geometry/text_scribble_to_surface` | `python run.py --cases combined --num-inference-steps 16 --init-octree-depth 6 --octree-depth 7` |
| Geometry | `geometry/bunny_sdf` | `python run.py` |
| Geometry | `geometry/surface_mesh_to_octree` | `python prepare.py && python run.py --init-depth 3 --depth 6` |
| Analysis | `analysis/image_to_fem_coordinate_bcs` | `python run.py --init-octree-depth 4 --octree-depth 5` |
| Analysis | `analysis/image_to_fem_surface_bcs` | `python run.py --init-octree-depth 4 --octree-depth 5` |
| Analysis | `analysis/nitsche_comparison` | `python run.py --n-standard 12 --octree-depth 4 --run-rotated` |
| Analysis | `analysis/bunny_nitsche` | `python prepare.py && python run.py` |
| Analysis | `analysis/nitsche_convergence` | `python run.py --depths 3 4 --ref-n 64` |
| Optimization | `optimization/topology_cantilever_2d` | `python run.py --n-iters 1 --nx 80 --ny 25` |
| Optimization | `optimization/topology_cantilever_3d` | `python run.py --n-iters 1 --nx 16 --ny 25 --nz 4` |
| Optimization | `optimization/sdf_shape_cantilever` | `python run.py --remeshing --n-iters 1 --remesh_octree_depth 6 --remesh_init_octree_depth 4 --nx 60 --ny 23 --nz 23` |
| Optimization | `optimization/latent_shape_opt` | `python run.py --n-iters 1 --init-octree-depth 4 --octree-depth 6 --latent-lr 5e-2 --latent-optimizer adam --disable-pullback` |
| Optimization | `optimization/shape_opt_capture` | `python run.py --n-iters 1 --depth 7` |
| Optimization | `optimization/hidden_phase_field` | `python run.py --n-iters 2 --phi-start-iter 0 --depth 7` |
| BCs | `boundary_conditions/manual_bunny_regions` | `python prepare.py && python run.py` |
| BCs | `boundary_conditions/automatic_bunny_ears` | `python prepare.py && python run.py` |
| BCs | `boundary_conditions/bc_patch_zip` | `python run.py --case bracket` |
| BCs | `boundary_conditions/bc_patch_zip_special` | `python run.py --case nested` |
| Benchmark | `benchmarks/overview_performance` | `python prepare.py && python run.py --depth 6 --optimization-depth 7 --optimization-iters 1` |

The leaf READMEs describe full reference settings where they differ. Limiting
an optimizer to one iteration is a smoke validation of the same numerical
problem, not a replacement for its full trajectory.
