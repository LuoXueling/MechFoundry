# Post-run evaluation of agent-built simulations

You are an independent evaluator. Coding agents each received a one-line instruction naming an
everyday object ("a hamburger", "a car brake disc", ...) and, by driving an FEM workflow, produced
one linear-elastic simulation of it: a 3-D body, supports and loads, and materials. Three agents
(labelled **A**, **B**, **C**; the labels are deliberately anonymous) each ran the same 50
instructions, so there are 150 simulations. Rate each one on three categories:
**geometry**, **boundary conditions (BC)** and **material**, each **High**, **Moderate** or
**Low**, with the rubric below.

This is a post-run check of whether each simulation *as set up* makes sense. Do not ask whether a
different setting would have been better. Judge the setting the agent chose against a sensible
reading of the instruction. Agents may legitimately pick different readings: a figurine or a
living animal, a hand press or a sitting person, a fingertip press instead of a palm press. Any
coherent, physically real reading is acceptable. You judge whether the geometry, BC and materials
are realistic *for that reading* and consistent with each other.

## Your packet: `{PACKET}`

- `index.json`: one entry per simulation: `label`, `category`, `object` (the instruction),
  `figure` (path), `row` (1-5, top to bottom), `reply` (path), `session` (path or null), and
  `status` (the runner's verdict on whether the agent session ended normally).
- `figures/<category>_<label>.png`: one figure per category and agent, with five rows in list
  order. Columns, left to right:
  - **Instruction**, with token and time bars. Ignore the bars: cost is not part of this
    evaluation.
  - **Generated shape**, with a scale bar (the square is the stated length) and the geometry
    route.
  - **Supports & loads**: teal = supports, orange = load regions and arrows (a pale orange patch
    is a load on a far face, seen through the body); the total load is printed.
  - **Young's modulus**: a log colour map from 100 MPa to 100 GPa (lighter is softer).
    Multi-coloured bodies have per-part materials.
  - **Deformed shape**: the displacement magnitude, magnified ×N, with the undeformed outline.
  A row can also show a failed or partly converged run.
- `reply`: the agent's final report: the session, the verdict line, and the deviations it
  declared.
- `session/` (when the run made a session) holds the files to check facts against. Read them on
  demand, and only as far as you need:
  - `operator/brief.md`: the scenario the agent wrote (reading, use case, contacts, load rows,
    materials).
  - `operator/judgement.md`: its self-check.
  - `boundary_conditions/resolved.json`: the regions, values, held components, directions and
    newtons.
  - `boundary_conditions/synthesized.json`: the BC proposal.
  - `session.json`: state, including the material parameters and the per-part table.
  - `materials/`: the per-part material files, when parts are used.
  - `previews/*.png` and `operator/*.png`: renders.
  - `geometry/`: the body.

  When the reply and the files disagree, trust the files.

**Blinding.** Use only the packet and the session folders it links. Do not open `metrics.json`,
`claude_*`, `settings.json`, `prompt.txt` or other files in a run folder, and do not try to work
out which model an agent is. Rate each simulation on the absolute rubric. Never rate it relative to
the other agents' runs of the same object.

## Rubric

Score every criterion **met**, **partial** or **failed** using the anchors. Then derive the
category rating mechanically:

- **High**: every criterion met.
- **Moderate**: no criterion failed, and one or two partial.
- **Low**: any criterion failed, or three or more partial.
- **n/a**: only when the run never produced that setting. For example, no body was ever accepted,
  so there are no BCs and no materials. A setting that exists but whose solve failed is still
  rated.

Judge sizes, loads and stiffnesses by order of magnitude against typical real values, using your
own engineering knowledge. The anchors below are guides, not an exhaustive table.

### Geometry: is the simulated body recognizable, realistic, physical and complete?

Judge the body that was simulated (the Supports & loads and Deformed columns), not an earlier
attempt.

| | met | partial | failed |
|---|---|---|---|
| **G1 Recognizable** | identifiable as the object without reading the caption | identifiable only once you know the caption | not identifiable even with the caption |
| **G2 Realistic form** | proportions and the defining features look like a typical real one (a wrench's jaws and handle, a femur's head, shaft and condyles) | one defining feature simplified or missing, or proportions clearly off; a hand-built proxy that keeps the load-bearing form | a primitive stand-in (box, bar, blob) or a different object |
| **G3 Physical** | one connected solid with plausible member thicknesses; real-world size within 2× of typical (check the scale bar and the brief) | a visible generation artefact (lumps, spurious tabs or slabs, fused gaps, holes) or size off 2-5× | floating or disconnected pieces carrying load, a non-solid sheet, or size off more than 5× |
| **G4 Complete** | every major part present | a minor part missing or truncated | a major part missing (legs, wheels, a whole section) |

### BC: are the supports and loads where they would really be, with realistic magnitude and direction?

First identify the scenario from the brief and reply: what holds the object, and what loads it.

| | met | partial | failed |
|---|---|---|---|
| **B1 Support regions** | where the object rests, is held, mounted or embedded in that scenario | the right kind of place but the wrong extent or position, or a support missing or added | on a surface that is free in reality, or a scenario that does not exist (the object held by a part nothing touches) |
| **B2 Load regions** | where the load acts in that scenario (the seat, the striking face, the top bun) | the right kind of place but the wrong extent or position | on a surface nothing loads in that scenario |
| **B3 Load magnitude and direction** | within 3× of a typical value for the scenario, and the direction is how it acts. Self-weight, if used, should match the body's mass | 3-30× off a typical value, or direction off by more than ~30° | more than 30× off, or a physically impossible direction (a floor pulling, a press pulling away) |
| **B4 Held components** | held components fit the contact type: bonded, bolted, embedded or gripped → all components; a resting contact → at least the normal component (tangential holds standing in for friction or rigid-body removal are fine) | over-restraint that materially changes this load case's response (a clamped end where free sliding or rotation governs), or under-restraint patched by an unphysical hold | holds that contradict the contact, such as a support that must pull to resist the load where the real contact can only push |

A brief change the agent declared (for example, a lighter load row chosen for convergence) is
judged as the scenario it became. Do not penalize the fact that the agent changed the brief.

### Material: does each part get a material that matches what the object is?

| | met | partial | failed |
|---|---|---|---|
| **M1 Part mapping** | parts that are clearly different real materials get different materials, and the map follows the part boundaries; an object that is really one material is uniform | a multi-material object treated as uniform where the other materials sit only in parts off the main load path; or a minor part misassigned; or a boundary visibly in the wrong place | uniform where the main load path crosses clearly different materials (a rubber tyre given the rim's steel), or a major part assigned another part's material |
| **M2 Material class** | each material is the right class for its part under the chosen reading (metal, wood, rigid polymer, elastomer, foam, glass, stone, bone, food, soft tissue) | a secondary part has the wrong class | the main body has the wrong class |
| **M3 Stiffness value** | Young's modulus within 3× of typical for the named material | 3-10× off | more than 10× off |

Typical E, for calibration only:

| Material | Typical E |
|---|---|
| Steels | ~200 GPa |
| Cast iron | 100-170 GPa |
| Titanium | 110 GPa |
| Aluminium | 70 GPa |
| CoCr | ~220 GPa |
| Glass | ~70 GPa |
| Zirconia | ~200 GPa |
| Enamel | ~80 GPa |
| Dentin | ~18 GPa |
| Cortical bone | 15-20 GPa |
| Trabecular bone | 0.1-1 GPa |
| Concrete | 20-40 GPa |
| Stone / marble | 20-70 GPa |
| Brick masonry | 2-15 GPa |
| Timber along the grain | 8-15 GPa |
| Plywood / MDF | 3-8 GPa |
| Ice | ~9 GPa |
| Packed snow | 1-100 MPa |
| Rigid plastics (ABS, PP, PC, nylon) | 1-3 GPa |
| UHMWPE | ~1 GPa |
| Rubber and elastomers | 1-20 MPa |
| Foams and upholstery | 0.01-1 MPa |
| Thatch | ~1-50 MPa |
| Soft tissue | 1 kPa-1 MPa |
| Bread, cake | 5-500 kPa |
| Jelly | 1-20 kPa |
| Watermelon flesh | ~0.1-1 MPa |
| Watermelon rind | ~5-20 MPa |

## Procedure

1. Read `index.json`. Work one figure at a time: open the figure image, then for each of its five
   rows read the reply, and open session files only where the image and reply leave a criterion
   undecided. Typical checks:
   - the load in newtons and the held components (`resolved.json`);
   - the E per part (`session.json` or `materials/`);
   - the brief's scenario.
2. Score the criteria, derive the three ratings, and write a short note per category (at most 30
   words). The note names each partial or failed criterion and its evidence, e.g.
   "B3 partial: 2 kN on a mug handle, ~50x a hand lift".
3. **Write results to `{OUT}` after every figure**, rewriting the whole file each time so that
   progress survives an interruption. The file is a JSON list with one object per simulation:
   ```json
   {"label": "A", "category": "food", "object": "a hamburger",
    "geometry": "High|Moderate|Low|n/a", "bc": "...", "material": "...",
    "criteria": {"G1": "met", "G2": "partial", "...": "...", "M3": "met"},
    "geometry_note": "...", "bc_note": "...", "material_note": "..."}
   ```
   Use `object` exactly as written in `index.json`.
4. Finish with all 150 entries in `{OUT}`. Your final reply should be short: the count of entries,
   anything you could not judge and why, and any rubric case you found ambiguous and how you
   resolved it.
