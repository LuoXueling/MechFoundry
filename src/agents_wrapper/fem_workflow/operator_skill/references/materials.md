# Materials

**The brief's `Materials by part` decides: E differing by more than 2 x → `parts`, the default for a
forward solve; else uniform. Uniform across such parts needs `"uniform_why"`: a shape optimization
(uniform only) or the differing parts all under 1 leaf, listed as a simplification. VoMP is a
comparison only, never in a plan.**

## Uniform

Send `{"material_mode":"uniform","youngs_modulus":E,"poisson_ratio":nu,"density":rho}` in the
session's units (SI: Pa and kg/m³; SI-mm: MPa and tonne/mm³, 1 kg/m³ = 1e−12 tonne/mm³;
US-inch: psi and lbf·s²/in⁴). Density is required when `acceleration` is nonzero.
`infer_material` (one API call) fills the card from a description; check its values against the
table before applying, because the driver sends the card's values.

Handbook bands (SI): take a value *inside* the band for the grade the brief names, and state it.
A material the table omits takes its handbook value, with the source named.

| Material | E (Pa) | ν | ρ (kg/m³) | Yield / strength for a safety factor |
|---|---|---|---|---|
| structural / stainless steel | 1.9e11–2.1e11 | 0.27–0.30 | 7850–8000 | 235–355 MPa yield |
| cast iron | 8e10–1.7e11 | 0.21–0.29 | 7200 | 150–250 MPa (brittle: tensile) |
| aluminium alloy | 6.8e10–7.3e10 | 0.32–0.35 | 2700 | 150–300 MPa |
| titanium alloy | 1.0e11–1.2e11 | 0.32–0.34 | 4400 | 800 MPa |
| bronze / brass / copper | 9.6e10–1.3e11 | 0.33–0.36 | 8500–8900 | 150–400 MPa |
| glass | 6.5e10–7.5e10 | 0.21–0.25 | 2500 | 30–50 MPa (brittle: tensile) |
| ceramic, porcelain | 6e10–1e11 | 0.20–0.25 | 2300–2500 | 20–40 MPa (brittle: tensile) |
| stone, marble | 5e10–7e10 | 0.25–0.30 | 2600–2700 | 5–15 MPa tension |
| concrete | 2e10–4e10 | 0.15–0.22 | 2400 | 2–5 MPa tension, 25–50 compression |
| glass-fibre composite, along fibre | 2e10–5e10 | 0.25–0.30 | 1800–2100 | 400–1000 MPa |
| carbon-fibre composite, along fibre | 7e10–1.5e11 | 0.25–0.30 | 1500–1650 | 600–1500 MPa |
| hardwood, along grain | 1.0e10–1.5e10 | 0.30–0.40 | 600–750 | 40–100 MPa (bending) |
| softwood / glulam | 7e9–1.3e10 | 0.30–0.40 | 450–700 | 20–40 MPa |
| plywood | 5e9–1e10 | 0.25–0.35 | 550–700 | 20–30 MPa |
| MDF, particle board | 2.5e9–4.5e9 | 0.25–0.30 | 600–750 | 10–20 MPa |
| acrylic / polycarbonate / PET | 2.0e9–3.4e9 | 0.35–0.40 | 1200–1400 | 50–70 MPa |
| ABS / PLA / nylon | 1.5e9–3.9e9 | 0.33–0.41 | 1050–1250 | 30–60 MPa |
| HDPE / PTFE | 4e8–1.4e9 | 0.40–0.47 | 950–2200 | 20–30 MPa |
| leather, textile | 5e7–5e8 | 0.30–0.45 | 900–1000 | n/a |
| rubber | 1e6–2e8 | 0.47–0.49 | 1100 | n/a |
| silicone / PU foam | 2e4–2e6 | 0.20–0.45 | 30–1100 | n/a |

**Plausibility.** The driver rejects obvious unit slips (after converting to SI: E outside
1e3–1.2e12 Pa, ρ outside 1–2.5e4 kg/m³, or E/ρ outside 50–4e8) and warns when ν > 0.49, which
conditions badly. `--force` only for a genuinely exotic material. A solve stalls from ν 0.48 and
from a part contrast of 1e3 or more: a rubber-like part is given
ν 0.45, and parts that much softer than the rest an E of at least the stiffest / 100, both marked.

**Stiff materials under light loads** give strains far below the float32 floor; the runner
always solves in mixed precision (the default).

## Per part (`parts`)

A single E across parts of different materials misstates how the load divides between them, so
the part field transcribes the brief's `Materials by part`, judged on the assignment, not the numbers.

**In the plan** it is one section:

```json
"material": {"mode": "parts", "description": "<what each part is made of>",
             "parts": [{"id": <id>, "youngs_modulus": <Pa>, "density": <kg/m3>}]}
```

- `description` says what each part is made of, in the brief's words for its parts.
- A bare surface is coloured flat by part (seconds), or painted from `texture_prompt` when given.
- One vision call values the BC stage's PartField or CAD-face segments (else the material
  step's own 6: `parts_segmentation`). The runner then fills each segment from `Materials by part`,
  matching part names to the workflow's segment labels; a segment it cannot match takes the
  Governing material (`parts_caveats`). It also lowers ν from 0.48 to 0.45 and raises a part
  1e3 x softer than the stiffest to the stiffest / 100, both simplifications. One stop then shows
  `materials/parts/montage.png`, the table (id, label, material, E, ν, ρ, area fraction,
  centroid), `soft_parts` and `segments_by_material`. A material with no segment of its own:
  re-segment (boundary_conditions.md §1).
- A wrong value is corrected with `parts` (id, fields to change; it wins over the brief): the
  next run applies it and shows the table again; the one after approves. Valuing counts against
  the `materials` limit (3); colouring and corrections do not.
- The report's `material:` line reads "per-part: <part> <material>, …"; the verdict's material says it.
- The solve stop lists the parts under `material`, the mass as the mesh volume times the field's
  mean density (`mass_note`), and `checks.stress_by_material`.
- **What is judged:** each part has a material the real part is plausibly made of, with values
  of the right order, and no part has a neighbour's material by mistake. Not the number: the
  runner's agreement statement reads "not judged" for a per-part run.

By hand: `{"action":"infer_parts","material_description":"…"}` (needs a coloured surface:
`texture_surface` then `use_texture` first), `{"action":"apply_parts","parts":[…]}` for
corrections, `{"material_mode":"parts"}` to approve (forward-only).

**Caveats:**
- Values the runner did not fill are the vision model's, which calls stiff parts soft more
  often than the reverse.
- PartField parts are geometric, not material (known_issues.md #5): two physically identical
  members can get different values; correct them to one.
- The field is a 64³ grid: a part thinner than a grid cell takes a neighbour's value.
- Measured, apart from the grid: an interior point takes the material of the nearest surface
  part, so a soft part bonded to a stiff one is stiff beyond the plane midway between their far
  faces (slabs under layers a fifth and a half their thickness: stiff above that plane in each),
  whatever the segments. The report states it as a caveat.
- Judge each material's stress against its own strength (`stress_by_material`).

## VoMP (`vomp`), for comparison only

`{"action":"infer_vomp"}` needs a coloured surface and runs a volumetric property predictor in
its own environment; approve with `{"material_mode":"vomp"}` (forward-only). Its training
distribution and renderer differ from the workflow's generated surfaces, and its ranges can be
implausible (known_issues.md #6): report it only as a comparison, with its ranges.
