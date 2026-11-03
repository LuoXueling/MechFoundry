# Looking at the pictures

Every stop names the PNGs to read. Read each with the file reader and answer its checklist;
each list ends with the decision. Do not decide from the numbers alone: a region can have the
right face count on the wrong side of the object.

Every panel shows six views, labelled by the direction the camera looks *from*: `+x`, `−x`,
`+y` (from above), `−y` (from below), `+z` (front), `−z` (back). Up is +y in every side view;
in the `+z` view right is +x, in the `+x` view right is −z. A part the `+z` view hides behind
the body, and the `−z` view shows in front, lies at −z.

## 1. `geometry/views.png` (after generation)

1. **Is it the object?** Name what you see in the `+z` and `+x` views: shown them, anyone names
   the object asked for, as a whole, by its shape, proportions and features. A blob, another
   object, a cropped part (never rescaled to stand in for the whole) or a body thickened into
   another object fails. A declared `body_simplification` or section stand-in is judged by its
   own declaration.
2. **The primary members:** the Real section's member, the member the run load enters and each held
   member, present and attached (one held that never came out: objectives/brief.md §2's substitution).
   Any other missing or merged is no failure: declare it absent or a stand-in
   (`body_simplification`, objectives/routes.md §3.4); the verdict lists it.
3. **One body.** `components: 1`; in the `−y` view the base or every point of contact is on one
   plane, nothing floating.
4. **Standing as in use.** In the side views the object stands as in service, +y up. One that
   came out in another orientation with the brief's parts in place keeps the brief's contacts
   and load face: rewrite their words for where the parts now are, and turn gravity and the
   load directions with the body (objectives/size.md §3.1). Moving a contact to another
   part because the words are easier is a change of brief, not a rewording.
5. **Not fused.** Members that should be separate are separate in the `+y` or `−y` view; fused
   members make the part far too stiff. A plate fused under the base is the support patch,
   unless the brief stands the body on several supports.
6. **The right size and dimensions.** `planned_resolution` gives the extents under the plan's
   scale and the leaf size h. The longest extent must be the dimension you meant; it may lie
   along x or z (`scale.up_axis: "longest"` takes either, and costs no generation). For a built
   or CAD body, every dimension and feature the brief must keep is checked here in numbers
   (objectives/routes.md §3.4).
7. **Thick enough.** `thinnest_member_leaves` (a wall measured straight through) is the one
   measure every check uses; `resolution_check` says what it gives.

**Decision.** Pass → run again. Reject only for the wrong object, several objects, not the whole
object, a broken cut-out or a missing primary member (checks 1–2; a ladder rung names only the fault,
geometry.md): change the words (geometry.md, "What the generator does"), seed, route or built
dimensions, with one line `geometry.reject: "<that check>"`; a silent move past a body or drawing
the runner's checks passed is refused. Fail 1 is
never accepted for want of attempts. A drawing or photograph of the whole object plus extra
structure has not failed: `"drawing_crop": "largest"`, or widen the brief's body. After the text
limit, `"draw": true` (the `drawing` stop: judge it the same way).

7 under 2 leaves is a warning, never a refusal (objectives/size.md §3.1): keep the body (the
report carries the caveat), or take depth 8.

## 2. `operator/segments.png` and `operator/regions.png` (after the preview)

**Segments** (one colour per part, a legend with the vision model's names): the parts you will
name are separate segments; the names are hints only and can sit on the wrong part.

**Regions** (cool = held, warm = loaded, purple = both, grey = free):
1. **Each support is on its contact patch**, as the brief's contact line places it, seen from
   the view that shows that patch. No blue up the sides of a part, on a loaded face, or on an
   inside surface. A tiny `area_fraction` for a contact on a flat underside means something
   protruding a leaf below it was caught (troubleshooting.md, "Region words").
2. **Each load is on the brief's face**, seen from the view that shows it. Red wrapping round a
   part, on its underside, or inside a cavity fails.
3. **Both present, not overlapping.** At least one blue and one red region, no purple. A case
   loaded by self-weight alone rightly has no red region (`planned_load` empty). Purple, or a
   support whose `selector` is `highest` (a load's `lowest`), means the words were read
   upside-down: add the facing.
4. **Delivery and size.** A load region is judged on where it is, the total it delivers
   (`planned_load`, `loaded_fraction` ≥ 0.8) and its share. `normal_coherence` below the
   threshold is a prompt to look for a caught neighbour of another facing, not a verdict; a
   curved face that delivers the stated force along the stated direction is right. A region
   under 50 faces or 3 leaves wide is too small to carry a load. No region covers more than
   40 % of the surface unless the case says so. A support's coherence does not matter.
5. **The physics.** Each blue patch restrains what the brief's contact line says and nothing it
   did not accept (`planned_supports`); each red patch is the face the use loads, with the
   direction in `values`. A region on a neighbouring face is a deviation, not a variant.

**Decision.** Pass all → run again (that approves). An empty region (`preview_error` "selected
no faces") → read `segments.png` and `segment_description`, name the part's segment with its
facing, or a position alone. Fail 1–3 → segments that do not separate the face: re-segment
first (`{"num_clusters": 8}`, then `{"method": "shape_diameter"}`; boundary_conditions.md §1);
otherwise reword with boundary_conditions.md §2. Fail 4 only → narrow the region when the
picture shows a caught neighbour; otherwise continue. At the preview limit with a check still
failing, the session ends as "the brief's case could not be run".

## 2b. `materials/parts/montage.png` (`mode: parts` only)

The painted surface with each segment outlined and numbered; the stop prints each id's E, ν,
ρ, area fraction and centroid (height is the second coordinate).
1. **Each id is its part**: place each number by its centroid and area fraction. A part fused
   with its neighbour in the segmentation is fused here too: then the case is uniform.
2. **Stiffness and density are the material's** (materials.md table). `soft_parts` lists ids
   under 1e9 Pa: each must really be a soft material.
3. **Thin parts** take a neighbour's value in the 64³ field: a caveat, not a correction.

Pass: every part within a factor of 3 of the named material → run again. Fail: list the part
in `material.parts` and run again. Three valuations that stay wrong → uniform, said so.

## 3. The field renders (after the solve)

`operator/displacement_magnitude.png` (deformation exaggerated to 5 % of the size; the title
says by how much) and `operator/stress_von_mises.png`, in the session's units.
1. **Supports at zero.** Blue at every fully held region, in every view.
2. **The brief's mode.** The exaggerated shape deforms as the brief predicted, in the direction
   the load pushes; the peak is where the brief said, never at a support.
3. **Smoothly graded.** Colour grows away from the supports, without isolated islands of red.
4. **Stress where predicted.** Highest where the load path narrows or turns, low at free tips
   and in unloaded bulk. A hot spot at a random place on a smooth surface is an artefact: look
   at the depth and the region edges.
5. **Numbers.** `checks.max_displacement_over_extent` and `max_von_mises_strain` within a factor
   of the prediction (the runner states agreement); fully held supports' mean displacement
   under 5 % of the peak; `newton.final_rel` below `relative_tol` (more iterations still below
   the tolerance are converged too, and go with a large deflection, a caveat of its own). A
   `strain_estimate` below `float32_floor` is not a problem in mixed precision.

**Decision.** Pass all → write the judgement. Fail 1, or a fully held support that moves (the
runner warns) → the region did not hold in the solve: back to the regions (edit `regions`),
before any solver setting. Fail 2 → the load's sign or face. Fail 5 by more than 2 → a units
slip in E, the size or the load, or a wrong face or contact: find which, fix the plan, solve
again. A failed solve has no pictures: read `solve_error` and apply the one change
troubleshooting.md gives for it.

## 4. Shape optimization (`operator/design_initial.png`, `design_final.png`)

As judging_results.md §7.
