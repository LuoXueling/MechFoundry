# Working as a CAE engineer

## Before the run: define the problem

- **Name the question.** Stiffness (how much it deflects), the stress pattern (where the
  stress concentrates and how large it is there), the load path, a lighter design at equal
  stiffness (shape optimization), or a safety factor, only when asked. The question decides
  what you report.
- **Model the real use, not a convenient one.** Each contact restrains only what the real
  contact can transmit (objectives/brief.md §2, Contacts); together the contacts remove all six
  rigid-body motions. Holding a face that is really free makes the part too stiff and puts
  stress peaks at the edges of the hold.
- **Loads are distributed** over the area through which the real force enters: the contact
  patch of the pushing body, as large as that body's own contact with the part. Never load a
  point or a sliver; the result there is meaningless.
- **Magnitudes are derived, never recalled** (objectives/brief.md §2, Load): a mass times g, a density
  times a volume, a pressure times an area, or a design-code clause you name and quote. State
  the derivation in the brief; a load out-ranking the use's own action needs a stated source.
- **Human actions.** A force or torque a person applies, with no measurement, standard or
  stated value in the instruction, takes its size from this table, its source in the Use table
  the row's id: `"H<k> top"` for a use done at full effort (the largest the stated use puts on the
  part), `"H<k> middle"` for a light or precise use (a tool worked by feel, a fine adjustment). The
  runner refuses a size outside the row's range.

  | Id | Action | Range, adult | Source |
  |---|---|---|---|
  | H1 | hand grip, squeeze (whole hand round a handle) | 250–500 N | grip-strength norms; EN 1005-3 |
  | H2 | fingertip or thumb press | 10–40 N | EN 1005-3; ergonomics handbooks |
  | H3 | push or pull, one hand, standing | 50–250 N | ISO 11228-2; DIN 33411 |
  | H4 | push or pull, both hands or body weight behind it | 200–400 N | ISO 11228-2; DIN 33411 |
  | H5 | press down with the arm, standing over it | 100–300 N | DIN 33411 |
  | H6 | two-finger twist (a cap, a dial or a stem up to 30 mm) | 0.3–1.5 N m | ergonomics handbooks |
  | H7 | wrist twist (whole hand round a lid, handle or grip of 40–100 mm) | 2–8 N m | DIN 33411 |
  | H8 | foot press (stepping or pushing down with one foot) | 250–800 N, x a stated dynamic factor | EN 1005-3; body weight |
  | H9 | carry or lift, one hand (5–15 kg x g) | 49–147 N, x a stated dynamic factor | ergonomics handbooks |
  | H10 | lift or carry, both hands (15–25 kg x g) | 147–245 N, x a stated dynamic factor | ISO 11228-1 |
  | H11 | light touch, fine manipulation (a switch, a key, a page) | 1–10 N | ergonomics handbooks (finger forces) |

  A combined action (a press with a twist, a pull with a drag) takes each part from its own row,
  acting together.
- **Units.** One system throughout; SI with real sizes is the default. Convert every input
  before sending (MPa to Pa, kg to N via 9.80665, mm to m).
- **Material.** State the grade, and the direction when it matters (wood along or across the
  grain). For a composite or a foam, say it is an equivalent isotropic approximation.

## During the run

- **Change one thing at a time**, and note why: a run that changes three things teaches
  nothing.
- **Keep the defaults** until evidence (a warning, a failed check) says otherwise.
- **Look at every intermediate artifact:** the views, the segment montage, the regions, the
  loads summary. The LLM steps make plausible mistakes: a mislabelled segment, a zero-thickness
  selector box, a load wrapped round a curved part.

## After the run: verify

- **Sanity:** the supports barely move; the deformation follows the load; the stress
  concentrates where the bending moment or the load path concentrates.
- **The prediction is the runner's.** `prediction_check` sets the solve against the runner's
  own stress and deflection at the Real section; a gap is explained from the case
  (judging_results.md §4), never re-derived by hand. Deep beams (L/d < 5) add shear deflection;
  diffused boundaries and coarse meshes make thin members stiffer.
- **Linearity.** Doubling the load doubles the response, less any unchanged self-weight.
- **Mesh convergence.** Only after a solve failed, re-run one depth finer (solver_settings.md):
  displacements should move by 10 % or less. Otherwise the plan's depth is the result, stated
  as such. Peak stresses at singular points never converge; quote them as mesh-dependent.
- **Model limits.** Elastic and small-strain in practice: no contact, no plasticity, no
  buckling check (a slender compression member is a stated caveat).

## Reporting

- **The case, completely:** geometry and size, supports, loads with the applied totals,
  material with its source, the settings changed.
- **Results with units and location:** the peak displacement and where; the stress away from
  the held and loaded patches and the peak, each with its place; a safety factor only when
  asked.
- **The verification:** the runner's prediction against the solve, linearity, convergence.
- **Assumptions and limits:** an inferred material, an idealised support, the resolution, known
  issues that touched the run, a generated rather than measured geometry.
- **Reproducibility:** the session id, generator and seed, settings that differ from the
  defaults, the report path.
- **Never present a failed or unverified run as a result.**
