# shape_opt_loop — content for the figure legend

The figure carries station titles, arrow labels and one legend row only; everything below
belongs in the legend.  Numbers come from `so_stats.txt` and `so_fr_stats.txt`.  The
layout is the merged figure's (`../_shared/loop_layout.py`) without the hidden phase field.

- **a, one iteration of SDF shape optimization**, two rows.  Each station is its render in
  a thin grey outline.  Row 1 (forward, black arrows, left to right): the initial design
  (iteration 0) → **Geometry + BC** — the design surface with its boundary-condition
  regions (Dirichlet feet; Neumann top, its downward traction drawn as arrows) → the signed distance d on the octree → the phase
  field φ = ½(1 − tanh(d/ε)) on the trimmed mesh → the **Nitsche solve**, i.e. the
  *diffused*-Nitsche finite-element solve on the trimmed octree mesh, for the displacement
  𝐮 → the objective 𝒥 = SE/SE₀ + w·softplus₅₀₀((V/V₀ − f)/(1 − f)), w = 5, f = 0.5.
- **Row 2 (backward, magenta, right to left)**: 𝒥 → ∂𝒥/∂𝐮 by Warp automatic
  differentiation of the taped strain-energy integral; ∂𝒥/∂𝐮 → D𝒥/Dφ (the dot on the lane)
  by the adjoint state method (Kλ = −∂𝒥/∂𝐮, solved with the forward stiffness K) plus
  automatic differentiation of the residual; D𝒥/Dφ → D𝒥/Ds by automatic differentiation of
  the tanh map.
- **Update**: "Step on d" is a gradient step on d (the dashed grey curve is d = 0 before
  the step, the solid one after it); its arrow returns to **Geometry + BC**, where the
  design surface is extracted by marching cubes, d is re-initialised from it on a new
  octree (remeshing) and the boundary-condition patches are restored exactly (BC-zip).
- **∂𝒥/∂𝐮** is the adjoint right-hand side.  The run stores no gradient with respect to 𝐮,
  so it was recomputed on the CPU (`so_dJdu.py`) from the run's own output of the shown
  iteration: (1/SE₀)·∂SE/∂𝐮, the internal-force vector Σ φ_w P(F)·∇N (neo-Hookean, E = 10,
  ν = 0.3, φ_w = (1 − 10⁻⁴)φ + 10⁻⁴), 2 × 2 × 2 Gauss points, hanging-node entries passed
  to their masters, strong Dirichlet nodes zeroed.  Boundary elements use the standard
  rule rather than the solver's adaptive Nitsche quadrature, which leaves a faint band at
  d = 0; away from the boundary the internal forces balance (≈2 decades below the loaded
  top).
- **Gradient stations**: |D𝒥/D·| divided by each field's own 99th percentile (of its
  non-zero values on the free nodes), clipped at 1, log scale over 4 decades;
  boundary-condition nodes (frozen d) at the floor colour; d = 0 in a white casing.
- **|𝐮|** uses the paper's displacement ramp (cream to brick, 0 to the solve's own max).
- **b**: the design's marching-cubes surface at the given iterations (number above each
  shape), cut on y = 0 with the near half translucent, brightened (gamma 0.72); bold =
  the iteration panel a shows (40, the compliance turn).  Wall clock: the optimizer's
  cumulative compute in the frames-mode re-run of the same problem (per-iteration output
  off, frame-dump I/O subtracted; NVIDIA RTX A6000): 101 iterations, 239.96 s.
- Problem: cylinder blank, four fixed feet, the whole top loaded, 50 % volume target,
  octree depth 7 (h = 0.03), ε = h/2.  Captured design 90 374 nodes / 73 875 hexes,
  trimmed solve mesh 78 223 / 61 804; five restored BC patches.
