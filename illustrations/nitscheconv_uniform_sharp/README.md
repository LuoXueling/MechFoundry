# Uniform-octree, sharp-volume Nitsche convergence

This companion runs the shared `../nitscheconv` study with both
`init_octree_depth == octree_depth` and `sharp_volumetric=True`. Build it from
`illustrations/` with `./build.sh nitscheconvuniformsharp`.

The sweep is float32 at depths 3–6, like the other targets. It sets
`NC_SCALE_DIAG=1` (the assembled system is scaled to unit diagonal) and
`NC_AMG_MAX_ITERS=4000`. Without the scaling the depth-6 aligned run stalls:
the exterior cells carry only the φ-floor stiffness, which puts the tangent at
the float32 rounding floor, and an AMG solve ends 300× above tolerance. The
scaling costs ~1.4 s per new mesh size, which shows up as a step at the coarse
end of the timing panel (g). See `../AGENTS.md` §5h.
