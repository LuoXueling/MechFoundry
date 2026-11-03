# shape_opt_merged — content for the figure legend

The figure carries block titles, arrow labels and one key only; everything below
belongs in the legend.  Numbers come from `so_stats.txt`, `../hiddenphi/hp_stats.txt` and (row c)
`som_cleanup.json`.

- **a, the optimization loop**, two rows.  Each station is its render in a thin grey
  outline (dashed = optional).  Row 1 (forward, left to right, black arrows): the initial
  design (iteration 0 of b) → **Geometry + BC** — the design surface with its
  boundary-condition regions (supports, teal: the feet, Dirichlet; loads, orange: the top, Neumann, its downward traction drawn as arrows) → the signed distance d on the
  octree → the phase field φ = ½(1 − tanh(d/ε)) on the trimmed mesh → the **Nitsche
  solve**, i.e. the *diffused*-Nitsche finite-element solve on the trimmed octree mesh, for
  the displacement 𝐮 → the objective 𝒥 = SE/SE₀ + w·softplus₅₀₀((V/V₀ − f)/(1 − f)),
  w = 5, f = 0.5.  Row 2 (backward, right to left, magenta): ∂𝒥/∂𝐮, D𝒥/Dφ̄ (optional),
  D𝒥/Ds, and the update step.
- **Backward (magenta)**: 𝒥 → ∂𝒥/∂𝐮 by Warp automatic differentiation of the taped
  strain-energy integral; ∂𝒥/∂𝐮 → D𝒥/Dφ (the dot on the lane) by the adjoint state method
  (Kλ = −∂𝒥/∂𝐮, solved with the forward stiffness K) plus automatic differentiation of the
  residual; D𝒥/Dφ → D𝒥/Ds by automatic differentiation of the tanh map.  The optional
  D𝒥/Dφ̄ branches off at D𝒥/Dφ (dashed): it is D𝒥/Dφ taken at φ = 1, on interior nodes
  only, because the hidden field's interior weight is held at one in the solve.
- **∂𝒥/∂𝐮** is the adjoint right-hand side.  The run stores no gradient with respect to 𝐮,
  so it was recomputed on the CPU (`shapeopt/so_dJdu.py`) from the run's own output of the
  shown iteration (the trimmed solve mesh, its nodal displacement and phase field):
  ∂𝒥/∂𝐮 = (1/SE₀) ∂SE/∂𝐮, the volume term being independent of 𝐮, with ∂SE/∂𝐮 the
  internal-force vector Σ φ_w P(F)·∇N (P the neo-Hookean first Piola stress, E = 10,
  ν = 0.3, φ_w = (1 − 10⁻⁴)φ + 10⁻⁴), 2 × 2 × 2 Gauss points per hex, hanging-node entries
  passed to their masters, strong Dirichlet nodes zeroed as the adjoint does.  It is not
  the solver's own vector: the solver integrates boundary elements with an adaptive
  Nitsche quadrature that the port replaces by the standard rule, which leaves a faint
  band around d = 0.  Away from the boundary the internal forces balance (interior values
  ~2 decades below the loaded top), so the field is dominated by the load and supports.
- **Update (one station, left of row 2)**: a gradient step on d ("Step on d"; the dashed
  grey curve is d = 0 before the step, the solid one after it) and, optionally, on φ̄
  ("Step on φ̄", dashed half), then **Geometry + BC** again: the design
  surface is extracted by marching cubes, d is re-initialised from it on a new octree
  (remeshing), and the boundary-condition patches are restored exactly onto the new
  surface (BC-zip) — closing the loop.
- **Optional hidden phase field φ̄ (dashed).**  φ̄ does not enter the solve: the interior
  weight of the residual and of the strain-energy / volume integrands is held at one.
  D𝒥/Dφ̄ is D𝒥/Dφ (direct + adjoint) taken at φ = 1, on interior nodes only .  It steps φ̄
  off the tape (in the band, φ̄ is reset to ½(1 − tanh(d/ε))); "Step on φ̄, carve φ̄ < ½"
  shows the stepped φ̄ (blue ramp of φ; teal: φ̄ = ½; grey: outside the body, where φ̄ is
  not used); its closed interior φ̄ = ½ loop is the dip that is carved into d (φ̄ < ½) at
  the next remesh, with the d step, through the same return into Geometry + BC.  Only in this hidden-φ̄ mode
  is D𝒥/Ds zeroed outside the band |d| < 3ε; the D𝒥/Ds block shown comes from the
  shape-optimization run, where it is not.
- **Gradient blocks** (key "|D𝒥/D·|, log"): |D𝒥/D·| divided by each field's own 99th
  percentile (of its
  non-zero values over the free nodes; D𝒥/Ds and D𝒥/Dφ̄ normalised separately), clipped
  at 1, log scale over 4 decades (10⁻⁴ … ≥1); boundary-condition nodes (frozen d) at the
  floor colour; cell frames drawn at 40 % so the band reads; d = 0 in a white casing.
- **Which run each render comes from**: the loop's renders are the shape-optimization run
  at iteration 40 (the compliance turn; b's bold frame); the φ̄ row (D𝒥/Dφ̄ and Step on φ̄) is
  the hidden-φ̄ run at iteration 43 (b's right-hand run, iteration 43, its bold frame), the iteration
  whose update first carves an enclosed cavity.  Both runs solve the same problem
  (cylinder blank, four fixed feet, the whole top loaded, 50 % volume target, octree depth
  7, h = 0.03, ε = h/2).
- **Wall clock (b's two titles)**: the optimizer's cumulative compute time measured by the
  frames-mode re-runs of the same two problems (per-iteration output off, frame-dump I/O
  subtracted; NVIDIA RTX A6000): `shapeopt/so_fr_stats.txt` (101 iterations, 239.96 s)
  and `hiddenphi/hp_fr_stats.txt` (91 iterations, 232.29 s).
- **b, results**: one row, the two table runs side by side, three frames each: the
  design's marching-cubes surface at the given iterations (number above each shape), cut
  on y = 0 with the near half faint (opacity 0.12, was 0.34 before 2026-10-16); frames are
  brightened (gamma 0.72) at compose time.  Left, "Shape optimization": iterations 0, 40
  (bold, the loop's) and 100.  Right, "Topology optimization" (titled "Concurrent topology
  and shape optimization" before 2026-10-16; the run is the same: the SDF shape update plus
  the hidden phase field φ̄, switched on at iteration 35; before it both runs coincide):
  iterations 20, 43 (bold) and 90.  The bold "43" is the surface after iteration 43's
  update and carve, i.e. the design iteration 44 starts from.  **Voids are drawn dark
  grey**: each surface point is shaded by its openness, the fraction of 256 cosine-weighted
  hemisphere rays from it that leave the whole (uncut) design without hitting it — at most
  0.08 is full void grey, at least 0.22 the body colour (`_shared/void_open.py`,
  `_shared/evo_style.py`; one rule for every frame of both runs).  The carved voids appear
  inside the body from 43 on (an enclosed cavity, openness 0) and open to the surface by
  iteration 80 (tunnels, openness a few hundredths); the shape run has none (frames 0 / 40
  have no point below 0.3; at 100 a sub-pixel crevice under the top's rim, 0.2 % of the
  surface, falls below 0.08).
- **|𝐮|** uses the paper's displacement ramp (cream to brick, 0 to the solve's own max).
- Captured designs: shape run 90 374 nodes / 73 875 hexes (trimmed solve mesh 78 223 /
  61 804); hidden-φ̄ run 91 096 / 74 575 (79 153 / 62 690), 39 interior dips kept, a
  1 180-cell cavity.
- **c, refining generated shapes** (the geometry clean-up,
  `experiments/geometry_cleanup`, its README): 17 bodies of the agentic-workflow batch
  (`output/agentic_batch50_opus`), each run through the workflow's own shape
  optimization as a clean-up: volume as is (penalty w·softplus(V/V₀ − 1), w = 5), the
  session's own supports, loads, material and octree, BC regions frozen (`mask`), SDF
  re-initialisation off, the first update's largest move 0.5 h_min, and the run stopped at
  the first shape whose surface lies more than 1.5 h_min (99th percentile, area-weighted,
  both directions) or 3 h_min (maximum) from the starting surface (that shape rejected),
  or after 40 updates.  **h_min is the smallest octree leaf size of the clean-up's
  analysis** (the finest finite-element leaf of the session's octree, 1.92 × extent /
  2^depth; e.g. 0.15 mm on the 10 mm dental crown, 10 mm on the leaf spring).  Each body
  shows two panels from one camera, an arrow leading from the first to the second: left, its supports (teal; a support facing away from
  the camera drawn as a teal outline) and loads (orange regions, arrows along the
  traction) on the starting surface — the crocodile, elephant, rhinoceros and stone arch
  carry their self-weight only, so no arrow; right, "change in shape": the final body
  coloured by its signed distance to the starting surface in units of h_min (red:
  material added, blue: removed, white within ±0.25 h_min, saturated at ±3 h_min).
  "SE/SE₀ = x" after each name: the strain energy (compliance) of the final body over
  the starting one's.
  Updates run: crocodile, horse, femur, robot arm, park bench and dental crown 40 (the cap);
  hammer 32, school chair 34, rocking horse 30, cake 9, pogo stick 27, leaf spring 33
  (distance threshold); elephant 5, rhinoceros 17, control arm 37, stone arch 10,
  Pantheon 7 (maximum-distance stop).  The dental crown barely changes (SE/SE₀ 0.99, at
  most +1.2 h_min after 40 updates) and is drawn from azimuth 90° instead of 35° so its
  changed flank faces the camera.  Per-part materials (claw hammer, park bench, school
  chair, control arm, pogo stick; `som_cleanup.json`, `standin_material`) are replaced by
  their volume-weighted uniform mean for the clean-up.
