# Boundary conditions: segment, describe, check, value

You never write selectors yourself. You write a **description**, an LLM turns it into selector
trees, and the tools resolve them into face masks you can inspect. Your leverage is the wording,
the segmentation, and your judgement of the preview.

## 1. Segmentation (`preview_segmentation`)

| Method | What you get | Settings |
|---|---|---|
| `partfield` (default when ready) | exactly `num_clusters` parts, which need not be connected (separate members of one kind can be one part), about 10 s warm | `num_clusters` (2–20, default 5) |
| `shape_diameter` | *connected* pieces of each thickness cluster (5 clusters gave 10–21 segments, slivers included), each with its thickness (`thickness_leaves` in `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py summary`, warned under 3 leaves) | defaults are fine |
| `cad_faces` (default for a CAD-generated body) | one segment per B-rep face, typed `plane` or `cylinder`: each hole is its own segment | none |

- **The vision call names the segments** (`segment_description`); the names are often wrong
  (known_issues.md #4). Check each segment you rely on by its centroid, bounds and dominant
  facing (`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/inspect_session.py summary`) and the montage `boundary_conditions/segments.png`.
- Above 20 segments the montage colours repeat: identify segments by id and bounds.
- **Re-segment first** when the segments do not separate the face a load or support needs, or
  the parts the brief gives different materials: change `num_clusters`, then the method, until
  they do, before rewording regions or correcting values part by part (its own limit, SKILL.md).

## 2. Writing the region description (`preview_regions`)

The description becomes `{"dirichlet":[…],"neumann":[…]}` with one selector per region.
- **Supports first, one clause per region**, each with its role: "Fix all components of …",
  "Apply a traction in −y on …".
- **Positional language that maps to one selector:** "the bottom 2 % of the height" becomes
  `lowest`; "the upward-facing top face" becomes `and(segment, aligned_normal +y)`; "x below
  0.12 m and x above 1.88 m" becomes slabs; "the rear-most 20 % along z" becomes `lowest z 0.2`.
  Positional words survive re-segmentation; segment names do not.
- **Revising what came back.** Ask for the regions in words; the regions stop prints the
  synthesized selectors (`synthesized_selectors`). Where one does not select what the words
  meant (a bound exactly at the body's extreme, a unit read wrongly), put the corrected selector
  in `plan.region_revisions` under that region's name and run again: it is re-resolved and
  recorded (`region_revision`), with its own limit (4); the other regions stay as they are.
  Selectors are never written without a synthesis.
- **Guard "lowest" and "highest".** The synthesizer sometimes reads a band upside-down. Add the
  facing ("faces whose normal points down (−y)"), or "(the lowest faces, not the highest)" for a
  region without one. A `selector` that prints `highest` for lowest words is that flip.
- **Unions** work (the workflow nests a many-way `or` into pairs), but one positional selector
  covering several contacts at one height is more robust and keeps up with the shape during
  optimization.
- **Segment ids** are fine once verified ("segment 9, the top face"); several ids go in one
  selector (`surface_segment` with `segment_ids`).
- **Spell out the thickness of boxes and slabs** (known_issues.md #3): "every box must span y
  from <a> to <b> m".
- **Restrict a load to one facing:** "…the faces whose normal points up (+y), not the end or
  side faces". A load region that wraps round a part is applied along its *average* normal and
  loses the tilted and back faces.
- **`revise_prompt: true`** on this card (one API call) rewrites the text to bind regions to
  segment names; check the rewrite (`description`) before previewing.

**Compositions** (part ∩ position ∩ facing − exclusions; objectives/regions.md §3.3), placeholders to
fill, not sentences to paste:
- "Fix all components of the <contact>: the <facing> faces of the <part> within <position>."
- "Apply a traction in <direction> on the <facing> faces of the <part> within <position>, not
  the <neighbour>."

## 3. Checking the preview (always, before approving)

1. `operator/regions.png`: every region exactly where the brief's contact or load is, and
   nowhere else (visual_checks.md §2).
2. The region rows: `faces` at least about 50 and `bbox_size` at least 3 leaves wide; for each
   load, `normal_coherence` at least the threshold and `loaded_fraction` at least 0.8 (supports
   may face many ways); `area_fraction` over 40 % is rarely meant; no overlap; at least one
   support.
3. The physics: do the supports remove all six rigid-body motions and no more than the brief's
   contact lines allow? A region held in every component removes all six by itself; a released
   component needs another contact to take that motion. Does the load act where the real load
   acts? Is any region an artefact (a spike tip, a sliver, an inside surface)?

**When the preview is wrong**, in order, within the 4 previews: reword (more positional, the
facing, a verified segment id); `revise_prompt` once; change the segmentation; report.
`preview_error` "BC 'x' selected no faces" means a selector was empty: reword. A synthesis
`ValidationError` means the model's JSON was invalid: retry once, then reword.

## 4. Values (`bc_values`)

The card lists `dirichlet` regions (`components`, `value`) and `neumann` regions (`traction`,
`area`), plus `body_force` and `acceleration`, in the session's units.
- **Supports.** `components [0,1,2]` and `value 0` hold a region fully. Release components only
  as the brief's contact line says. A nonzero `value` is a prescribed displacement.
- **Loads, in order of preference:**
  - the Use table's rows, bound in `plan.loads`: the runner writes each region's total
    `{"neumann_values":{"<region>":{"force":[fx,fy,fz]}}}` from its rows' sizes, divided by the
    region's `loaded_area` (the cosine-weighted area), so the force that reaches the part is the
    row's; the driver names the resulting traction in `notes`;
  - a row's `direction` gives the line; on a curved region a fixed direction acts weighted by
    the cosine to the average normal (concepts.md, Conventions).
- **Self-weight:** `"acceleration":[0,-9.80665,0]` in SI (−9806.65 mm/s² in SI-mm, −386.09
  in/s² in US-inch); the material then needs a density.
- **Magnitudes and dynamic factors** are derived in the brief (objectives/brief.md §2, Load).
- **Check after applying:** `loads.applied_surface_force` equals the rows' totals.
