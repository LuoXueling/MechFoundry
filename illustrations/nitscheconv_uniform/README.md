# Uniform-octree Nitsche convergence

This is the fixed-depth companion to `../nitscheconv`. Its build uses the same
prep, render, and compose sources, but sets `NC_UNIFORM=1`, which passes
`init_octree_depth=octree_depth` for every aligned and rotated Nitsche run.
The prep asserts that every cell in the actual FEM octree has the expected
single edge length before recording a result.

Build from `illustrations/` with `./build.sh nitscheconvuniform`.
