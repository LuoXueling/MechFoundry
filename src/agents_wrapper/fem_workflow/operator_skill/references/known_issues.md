# Known issues

Open problems in workflow or solver code, or in the models, which the skill does not modify:
each with its evidence, its effect on a run, and the workaround. Name in the report every one
that touched the run.

## Solver

**1. Load-bearing members only a few leaves thick make the linear system ill-conditioned.**
- **Evidence** (a06a628ccc5c, members 1.5–2 leaves thick at depth 7, about 3 at depth 8, mixed
  precision): JAX-AMG reached only ‖b − Kx‖/‖b‖ = 6.9e-3 (depth 7) and 5.2e-3 (depth 8) after
  1000 iterations against its 1e-6 tolerance, and raised at Newton iteration 0.
  `scale_system_to_unit_diagonal` made it worse (3.9e-2: why it is last), a β cap of 500 much worse (2.0).
  `solve-backend` at depth 8: `pyamgx` 2 Newton iterations, `native` 6, both rel 8.8e-3,
  agreeing to 0.1 % in p99 surface stress and 4 % in peak displacement. Members 6 leaves thick
  solve in one Newton step on the default backend.
- **Recognise it:** ‖b − Kx‖/‖b‖ stays around 1e-2 to 1e-3 whatever the settings, on members
  under about 3 leaves.
- **Workaround:** question the resolution first (depth 8 if VRAM allows, or report the member
  as under-resolved). For a result anyway, `solve-backend --backend pyamgx`: pyamgx never reads
  AMGX's status, so it is inexact Newton, validated by Newton's residual check (SKILL.md).

**2. `gd_lr` has units of length², so its default 0.01 suits only some part sizes.**
- **Evidence:** the boundary moves per iteration by about gd_lr·w_vol·h²/((1−f)·V₀). At the
  former default of 8, a 0.4 m³ blank hit the volume target in 4 iterations, then oscillated.
- **Workaround:** the calibration of solver_settings.md §4, then adjust on the first 10–20
  iterations. Verified on two blanks (formula 0.68 and 0.79; the volume then fell 2–3 % of V₀
  per iteration).

## LLM stages

**3. The BC model sometimes builds a zero-thickness box or slab** (`min_pt[1] == max_pt[1]` at a
face's height; ce2b0eee1356, b07f73d49a61), which selects nothing. Saying in the description
that every box spans the whole thickness, with explicit bounds, fixed it on the next preview
both times.

**4. Segment names from the vision call are unreliable** (46fed29be766, c90e421c2909: a name
on an inside surface, the same name twice, a name on the wrong region). Verify every segment
you use by its render, centroid and bounds, or describe regions positionally.

**5. PartField parts are geometric, not material** (c90e421c2909: two identical members split
between two parts, which `infer_parts` valued 70 GPa and 11 GPa). Check the montage and give
physically identical members one material with `apply_parts`.

**6. VoMP ranges can be implausible** (c90e421c2909: E 0.35–96 GPa, density up to 5866 kg/m³
on a body of two materials). Use VoMP only as a comparison.

**7. CAD-Coder failure modes** (held-out): features in two planes cannot be placed (1 trial);
a part offset along the extrusion normal came out at the wrong height or was lost (2 trials);
a cut extruded away from the body removed nothing (2 generations); parts that only touch were
left unfused (2 generations); a part was dropped (1). The geometry stop prints `cad_parts`
(each part's position, height range and operation as revised) and `connected_solids`: check
them against the brief; a part that reduces to one extruded section is built instead.
(Text2CAD, not used: 0 of 9 held-out generations matched the brief; its revision drops
numbers.)

## Optimization output

**8. Zip-stitched optimization surfaces can be non-watertight** (b07f73d49a61: after 8
iterations a closed main body plus four open fragments of 8–86 faces near it; the log reported
`BC zip: … non-manifold edges`). The design (the SDF) is continuous: treat the GLB as a
visualisation, and re-mesh from `vtu_sdf/` outside the workflow for a watertight part. The
inspector classifies the pieces.

## Results

**9. No support reactions.** The solver writes displacement, strain and stress, not the
reactions at held regions. Integrating the surface stress over a support recovered only 7–53 %
of the applied force on three sessions, because the stress sampled inside the diffused Nitsche
band under-reports the traction there. Each support's share is argued from the brief's
statics; a support that moves a visible fraction of the peak is the only sign the result gives.
