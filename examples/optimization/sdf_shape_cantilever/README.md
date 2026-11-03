# SDF shape optimization

- Smoke: `python run.py --remeshing --n-iters 1 --remesh_octree_depth 6 --remesh_init_octree_depth 4 --nx 60 --ny 23 --nz 23`
- Reference validation: `python run.py --remeshing --n-iters 1`
- Full reference trajectory: `python run.py --remeshing`

The nonlinear solve is not reliable on the formerly documented `30x12x12`,
depth-5 discretization. The smoke mesh comes from this runner's source docstring;
the reference command preserves the old root README's depth-7 parser defaults.
The original `1e-3` relative Newton tolerance is retained; a `5e-7` absolute
floor handles the observed native-float32 residual plateau on both meshes.
