# Adaptive-octree, sharp-volume Nitsche convergence

This companion runs the shared `../nitscheconv` study on the original adaptive
octrees (depth-3 far field) with `sharp_volumetric=True`. Build it from
`illustrations/` with `./build.sh nitscheconvsharp`. It writes
`nitsche_convergence_sharp.{pdf,png}`.

The sweep is float32 at depths 3–6 with `NC_SCALE_DIAG=1 NC_AMG_MAX_ITERS=4000`.
The rotated depth-6 solve needs more than JAX-AMG's default 1000 iterations even
on the scaled system. As in the uniform companion, the scaling's ~1.4 s per new
mesh size puts a step at the coarse end of panel (g). See `../AGENTS.md` §5h.
