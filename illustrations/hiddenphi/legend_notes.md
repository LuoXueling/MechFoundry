# hidden_phi_loop — content for the figure legend

The figure carries station titles, arrow labels and one legend row only; everything below
belongs in the legend.  Numbers come from `hp_stats.txt` and `hp_fr_stats.txt`.  The
layout is the merged figure's (`../_shared/loop_layout.py`) with the hidden phase field φ̄
drawn as part of the method.

- **a, one iteration of concurrent topology and shape optimization**, two rows.  Row 1
  (forward, black, left to right): the initial design (iteration 0) → **Geometry + BC**
  (the restored surface with its Dirichlet feet and Neumann top, the downward traction drawn as arrows) → the signed distance d
  on the octree → the phase field φ = ½(1 − tanh(d/ε)) on the trimmed mesh → the
  **Nitsche solve** (the *diffused*-Nitsche finite-element solve on the trimmed octree
  mesh) for 𝐮 → 𝒥 = SE/SE₀ + w·softplus₅₀₀((V/V₀ − f)/(1 − f)), w = 5, f = 0.5.
- **φ̄ does not enter the solve.**  The interior weight of the residual and of the
  strain-energy / volume integrands is held at one, so the forward pass sees φ in the band
  and full material in the interior.
- **Row 2 (backward, magenta, right to left)**: 𝒥 → ∂𝒥/∂𝐮 (automatic differentiation of
  the taped integrals); ∂𝒥/∂𝐮 → D𝒥/Dφ (the dot) by the adjoint state method (Kλ = −∂𝒥/∂𝐮,
  solved with the forward stiffness K) plus automatic differentiation of the residual.
  From D𝒥/Dφ come both D𝒥/Ds (automatic differentiation of the tanh map; zeroed outside
  the band |d| < 3ε, the interior being handed to φ̄) and D𝒥/Dφ̄ = D𝒥/Dφ taken at φ = 1,
  on interior nodes only.
- **Update (one station)**: "Step on φ̄" steps φ̄ off the tape in the interior (in the band
  φ̄ is reset to ½(1 − tanh(d/ε))); the teal curve is φ̄ = ½, grey is outside the body
  where φ̄ is unused, and the closed interior loop is the dip that is carved into d
  (φ̄ < ½) at the next remesh.  "Step on d" steps d (dashed grey: d = 0 before the step).
  The station's arrow returns to **Geometry + BC**: marching cubes, d re-initialised on a
  new octree, BC patches restored exactly (BC-zip).
- **Which run each render is from**: every station is the hidden-phase-field run at
  iteration 43 (the iteration whose update first carves a cavity), except **∂𝒥/∂𝐮**.  The
  hidden-φ run's displacement vector was not kept, so that station shows the
  shape-optimization run's ∂𝒥/∂𝐮 at iteration 40 (same problem, recomputed on the CPU
  from its output as the internal-force vector; see `../shapeopt/legend_notes.md`).
- **Gradient stations**: |D𝒥/D·| divided by each field's own 99th percentile, clipped at
  1, log scale over 4 decades; boundary-condition nodes at the floor colour; d = 0 in a
  white casing.
- **b**: the design's marching-cubes surface at the given iterations, cut on y = 0 with
  the near half translucent, brightened (gamma 0.72).  The hidden field switches on at
  iteration 35; the bold "43" is the surface after iteration 43's update and carve, i.e.
  the design iteration 44 starts from.  The carved voids appear inside the body from 43
  on, seen through the translucent near half, and open to the surface by iteration 80.
  Wall clock: the frames-mode re-run of the same problem (per-iteration output off, I/O
  subtracted; NVIDIA RTX A6000): 91 iterations, 232.29 s.
- Captured iteration 43: 39 interior dips kept, a 1 180-cell cavity; design octree 91 096
  nodes / 74 575 hexes, trimmed solve mesh 79 153 / 62 690; ε = h/2 = 0.015.
