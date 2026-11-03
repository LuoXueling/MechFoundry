# Judging a result

A run is credible when it **finished**, **converged**, is **physically sensible**, and **agrees
with the brief's prediction**. A colour plot alone proves none of these. Work through the list
in order.

## 1. Did it finish?

- The card's `message` is "Simulation complete…" and `solve_error` is null; in `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py summary SID`,
  `solve.succeeded` is true (the newest run folder has "Simulation finalized", its `*.pvd`
  files and a surface VTU).
- **Staleness.** If the newest run failed, the field GLBs belong to an earlier run: the summary
  warns "predates the newest run" and names `last_successful_run`. Never quote numbers from a
  stale sidecar.

## 2. Did it converge?

`solve.newton`: a small-strain elastic case converges in 1–2 Newton iterations, `final_rel`
under `relative_tol`. `rel_above_one: true` in the final solve: not converged, do not use it.
Many iterations with rel flat from iteration 1 is the float32 floor: re-run in mixed precision.

## 3. Is it physically sensible?

`solve.checks` and `solve.region_displacement`:

| Check | Expect | If not |
|---|---|---|
| supports' `mean` displacement | under about 5 % of the peak (weak enforcement leaves a small residual) | a support held in all components that moves did not hold in the solve: check the region first (objectives/judging.md §5, the solve); a partly held one may move by a rigid motion its released components leave free |
| where the peak displacement is (render) | where the brief's mode puts it, never at a support | a wrong load place or sign, or a missing support |
| `max_displacement_over_extent` | within 0.5–2 of the brief's generated prediction (the runner's agreement) | a units slip, the load's size or face, a contact |
| `max_von_mises_strain` | the brief's predicted stress over E, within 0.5–2; above the material's elastic range the linear result does not hold | a units slip or a load of the wrong size |
| `stress_concentration_vs_traction` | the ratio the brief's mechanism implies (predicted stress over applied traction); meaningless when self-weight dominates | orders of magnitude off: check the units |
| `applied_surface_force` (loads block) | the force you intended | the region faces many ways: narrow it and re-apply |
| linearity (a second load level) | displacement scales with the load | nonlinearity, or a solver problem |

**Look at the renders** (`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py render SID --what all`): the deformation (scaled to 5 % of the
size) and colours tell the mechanical story the brief predicted: the mode, the peak where the
load path narrows or turns, zero at free tips. A different story is a finding about the
contacts or the load.

## 4. Does it agree with the prediction?

The prediction is the runner's (a conservative bound from the Use table at the Real section);
the runner states whether the solve agrees (under 2 × its bound), and the solve decides. The usual real
causes of a gap, each checkable: a held patch resisting rotations the formula's end condition
leaves free (stiffer); a load spread over a patch rather than a point (stiffer); a member under
2 leaves thick (stiffer: the first-order limit from two depths, `checks.depths`, measures it
when a failed solve was re-run that way); a unit slip (orders of magnitude). A gap no such check
explains is reported as unexplained; it is no verdict.

## 5. A safety factor, only when the user asks for one

The default report gives the deflection and the stress pattern (where the stress concentrates,
the stress there away from the held and loaded patches, the peak as mesh-dependent), no safety
factor. When asked, the model is elastic with no plasticity: compare the stress with the
material's yield or strength (`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py summary SID --yield-strength <σy>` gives
`safety_factor_vs_yield`), and:
- **Several materials** (`mode: parts`): `checks.stress_by_material` gives p99 and max von
  Mises per E; compare each material's stress with its own strength. A part-wide peak mixes
  materials that fail at different stresses.
- **Singular peaks** at clamped edges, re-entrant corners and load-patch edges grow with
  refinement: compare the stress away from them, and state the peak as "mesh-dependent, at
  <location>".
- **Brittle materials:** the largest principal (tensile) stress against the tensile strength.
- **Wood:** the strength along the grain for the mechanism (bending, compression).
- Say what factor the use requires and why, rather than quoting a habitual one.

## 6. Resolution caveats

Depth-7 numbers are indicative, and the report says so; the depth-8 change is shown only when
a failed solve was re-run at two depths (`checks.depths`, solver_settings.md).

## 7. Shape-optimization results

The `solve.optimization` block reports:
- `volume_last_over_first` near the target fraction;
- `se_first` → `se_last`: SE rises while material is removed, then falls at constant volume;
  compliance per volume (se × volume) should drop;
- `objective_increases`: many are normal once the volume alternates around the target; judge
  the trend on the penalty-on iterations;
- `final_design_pieces`: one main body. A "detached solid" is a real problem. A "solid behind a
  sub-leaf neck" hangs on a junction thinner than one leaf: quote `latest_whole_design` and
  note the neck. An "open fragment" is a zip seam in the output surface.

`operator/design_initial.png` and `design_final.png` should show a load path from the loads to
the supports, material where the stress flows, and no dangling pieces.

## 8. Presenting it

- One story per image: the regions (what holds, what loads), the deformed displacement (how it
  moves), the stress (where it concentrates). For optimization, the initial and final designs
  side by side, with the objective and volume history.
- Label with units and the load case: "<load and where>, <material>, depth <d>: peak
  <displacement> at <place>, <stress> at <place> away from the supports".
- State the deformation scale (the inspector writes it in the title).
- The GLBs in `solve/` open in any glTF viewer; the `*.pvd` files in ParaView.
