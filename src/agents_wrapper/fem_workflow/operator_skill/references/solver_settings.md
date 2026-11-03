# Solver settings (the `shape_opt_settings` card)

The card's `defaults` are the session's current settings. Send only the fields you change:
`{}` re-runs with the current settings; `{"use_mixed_precision": true}` changes one field and
keeps the rest (the driver's `notes` name them). The card also shows
`strain_estimate` and `strain_floor`.

## 1. Every field

| Field | Default | What it does | When to change it |
|---|---|---|---|
| `run_mode` | `forward` | one solve, or `shape_optimization` (gradient descent on the SDF) | optimization only with a uniform material and clean BC patches (§4) |
| `use_mixed_precision` | true | float64 assembly and solve | keep it on (needed when `strain_estimate < strain_floor`); 1.3–2× float32's time, VRAM |
| `relative_tol` | 1e−2 | Newton stop on ‖R‖/‖R₀‖ (from iteration 1) | 1e−3 for publication numbers or prescribed displacements (a Nitsche penalty term one step removes can dominate ‖R₀‖). Also sets the float32 floor 1.19e−6/rtol |
| `absolute_tol` | 1e−5 | Newton stop on ‖R‖, **in force units** | rarely; scale it with your units |
| `max_newton_iters` | 20 | Newton cap | 40 for large deformation; more do not fix a stall |
| `load_steps` | 1 | ramps loads linearly in N steps; a forward run saves each step; an optimization ramps only the *first* solve; later ones warm-start at full load | 3–10 for large deformation or when Newton diverges at full load; 2–3 when an optimization's first solve struggles |
| `remesh_octree_depth` | 7 | analysis leaf size h = 1.92·extent/2^depth | 8 for thin members, or a convergence check after a failed solve (about 4× the nodes, 10–20 GB); 6 only for a quick look |
| `remesh_init_octree_depth` | 4 | interior coarseness | leave it (5–6 only to test a uniform interior) |
| `phase_field_epsilon_factor` | 0.5 | diffuse-boundary width ε = f·h | leave it; 1.0 smooths slivers at a cost in accuracy |
| `line_search_enabled` | false | Armijo backtracking in Newton | when Newton oscillates or diverges (rel > 1) under large loads |
| `scale_system_to_unit_diagonal` | false | Jacobi-scales the linear system; the reported ‖b − Kx‖/‖b‖ is then the scaled system's | §2's last rung only (it once raised the residual); failed with it on earlier: remove it first |
| `beta_max_over_e` | null | caps the Nitsche penalty β at this multiple of E (uniform only) | the same stall message; try 100–1000. It weakens support enforcement: check support motion. Keep it only if ‖b − Kx‖/‖b‖ drops |
| `sdf_bc_fix_method` | `mask` | how BC patches survive optimization remeshes: the SDF is frozen there (faces may drift slightly) | keep it; `zip` (iteration-0 patch stitched back) needs clean BC patches (§4) |
| `gd_lr` | 0.01 | the gradient step on the SDF (units of length²) | **calibrate for every optimization** (§4) |
| `gd_lr_auto` | false | sizes `gd_lr` at each update so the largest SDF move is a·ε (a: `gd_lr_auto_fraction`, 0.5); steps never shrink: runs end at `max_optimize_iters` | a calibrated or recommended `gd_lr` fails; then lower a |
| `max_optimize_iters` | 101 | optimization iterations | 60–150 |
| `output_per_optimize_step` | 1 | save every Nth iteration; 0 saves nothing | 1 (renders and the design check need them) |
| `optimize_relative_tol` | 1e−9 | convergence test (effectively off) | leave it |
| `target_volume_fraction` | 0.5 | volume cap as a fraction of the *initial* volume (≤ 2) | 0.3–0.7; lower is more dramatic, and riskier for connectivity |
| `volume_as_is` | false | only penalise growth above V₀ (stiffness at fixed volume); `target_volume_fraction` unused | reshaping without removing material |
| `shape_opt_w_vol` | 5.0 | volume-penalty weight | 2–10; higher tracks the target tightly but oscillates more |
| `reinit_sdf` | true | marching-cubes re-initialisation and remesh each iteration | **never false**: false disables remeshing and BC restoration |
| `reinit_every` | 1 | remesh every N iterations | leave it |

`reinit_n_iters` is gone; a plan's is dropped.

**Fixed by the workflow:** gradient descent on the SDF, marching-cubes re-initialisation, the
AMG backend, neo-Hookean material, no output of deformed GLBs.

## 2. Forward-solve recipes

- **Default:** the class defaults (float64, also in the runner's plan), depth 7, rtol 1e−2. In
  `parts` / `vomp` modes the card's `strain_estimate` is null.
- **Before the first solve** the runner's `triage` stop (shown once, not an attempt) names the
  causes no setting removes, each with its one change (troubleshooting.md, Supports, components
  and parts); a material changed so is a marked simplification the report states.
- **‖b − Kx‖/‖b‖ of 1 or more on the unscaled system** is a singular or nearly singular system: no
  setting below helps; the stop's `next` names the change in the supports, components or parts.
- **Linear solver fails** (`JAX-AMG solve failed … NOT_CONVERGED after 1000 iterations
  (||b - Kx||/||b||=…)`, under 1): SKILL.md's order (procedure step 3): `solve-backend` first
  (the runner runs it); then a triage cause the stop's `next` names; only then one change at a
  time, comparing ‖b − Kx‖/‖b‖ after each:
  1. `use_mixed_precision: true`, if the run was float32;
  2. `beta_max_over_e: 500` (uniform material only: with `parts` or `vomp` it is not applied; on
     members under about 4 leaves it makes the residual worse: skip it);
  3. last resort, all above failed and a solve left (`next` names it):
     `scale_system_to_unit_diagonal: true`, once. Its residual is the scaled system's: only
     Newton's unscaled check validates it; the report says so. Failed too: remove it, report it.

  No scaling earlier (§1). Do not lower the depth to escape it. A `solve-backend` result
  is closed as SKILL.md step 3 says, not by `run` (the graph's last solve stays failed).
- **Large deformation** (strain above 1e−2, soft materials):
  `{"load_steps": 5, "line_search_enabled": true, "max_newton_iters": 40}`.
- **Mesh-convergence check** (only after a solve failed; otherwise one solve at the plan's depth
  is the result): re-run at `remesh_octree_depth` 8, or `run --two-depths`. Peak displacement
  should change by 10 % or less. Peak stress at a clamped edge or re-entrant corner keeps
  rising (singular): quote stresses away from it (judging_results.md).

**Measured** (scales with node count): depth 7, 45–83k nodes, float64: one Newton
step, 7–12 s, +4.5–6.1 GB; depth 8, 210–264k nodes: 18–19 s, about 19–20 GB.

## 3. Reading a failed solve

The card comes back with `solve_error`; `solve_log` holds the tail of the log, and the reported
‖b − Kx‖/‖b‖ says how far off a linear solve was. Each message's cause and the one change:
troubleshooting.md, Solve; `Linear solver failed … NOT_CONVERGED`: §2. Every graph solve and
fallback counts against the solve limit (5). The failure leaves `solve/failed.json` and deletes earlier runs'
field GLBs, so nothing stale is shown. A failed optimization shows its last good iteration; the failed
one moves to `failed_iteration/`; `failed.json`'s `optimization` names both.

## 4. Shape optimization

**Preconditions:**
- a uniform material;
- a converged forward solve of the same case, judged sound;
- BC regions that are clean disks attached to the body, with no small holes or thin rims inside;
- the load on a whole face (a small central load patch tends to pinch off);
- a bulky starting design with room to remove material (a built blank); a generated body
  near its final form has little to give.

**Calibrate `gd_lr`.** Per iteration the boundary moves by about
gd_lr·w_vol·h²/((1−f)·V₀), with h the leaf size, f the target fraction and V₀ the initial
volume (the inspector's `physical_volume`). Aim for about half a leaf per iteration:

    gd_lr ≈ (1 − f) · V₀ / (2 · w_vol · h)

For example V₀ = 0.41 m³, h = 0.03 m, f = 0.5, w_vol = 5 gives 0.68.

Run 10–20 iterations first (`"max_optimize_iters": 15`) and read the trajectory (`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py summary`
optimization block, or the `SE=… Vol=…` log lines):
- volume falls 2–5 % of V₀ per iteration and reaches the target in 15–30 iterations: good;
- it reaches the target within about 8 iterations, then alternates widely: halve `gd_lr`;
- under 1 % per iteration, or "the design barely changed": double `gd_lr`.

**Then the full run:** `{"run_mode":"shape_optimization","gd_lr":<calibrated>,"max_optimize_iters":100}`,
about 1.5–4 s per iteration at depth 7 in float64. Poll `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py status`.

**A good run**, besides the volume above:
- SE (normalized compliance) rises while material is removed, then falls at constant volume.
- The volume alternating within a few % of the target, mostly with period 2, is normal
  with the one-sided penalty; judge SE where the penalty is active.
- One body with a readable load path. A junction thinning below a leaf late in the run makes the
  last surfaces flicker between one and two bodies: quote the latest design in one body (next list).

**Warnings to act on:**
- "detached solid piece(s)": material cut off from the load path. Lower `gd_lr`, raise the
  target, or check the BCs.
- "solid piece(s) within one leaf of the main body": a neck thinner than the resolution. Quote
  the inspector's `latest_whole_design`, or lower `gd_lr`.
- "the objective did not decrease": a wrong lr or broken BCs.
- "volume still above the target": more iterations or a higher lr.
- BC-zip errors (`BC zip …`, `BC region is being isolated`) under `zip`: make the regions meet
  the preconditions above and lower `gd_lr`, or return to `mask` (say BC faces may drift).
- Open surface fragments near the body in the final GLB are zip seams, not physics: the design
  is the SDF (`vtu_sdf/`).

**Settings to keep:** the defaults: depth (8 takes about 3.5× as long per iteration), `reinit_sdf`,
`sdf_bc_fix_method`, `shape_opt_w_vol` and `target_volume_fraction`.
