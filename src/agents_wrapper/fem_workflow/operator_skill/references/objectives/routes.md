# From the brief to the plan (objectives §3.4): getting the body

### 3.4 Getting the body

**Narrow the routes first.** From the brief's kind of body and what each route controls
(geometry.md), write in the brief which routes can make this body, one reason each.
`plan.geometry_routes` names them in any order, at least one ready generated route; the runner
takes the generated ones first (`triposg_text`, `drawing` i.e. SDXL then TripoSG, `hunyuan3d` from
the drawing, `cadcoder` for joined prismatic parts), at most 2 generated bodies, then a web
photograph (`web_image`), then `built` (`blank` runs the one photograph search itself when none
was tried). `built` comes first (after that search), wherever it is listed, only with the brief's line `Why not generated: <the member and its real
size>`, for a real thinnest member under 1/50 of the largest Real extent (a thin sheet, shell or
lattice, which the generators fill solid), or a one-member Load path whose Real section is a shape;
the runner checks both on the Real lines. The route is a property of the shape, never of the
object's name. A route the list leaves out (an untried one may be added), one ahead of its turn
and an attempt re-run unchanged are refused; after the first attempt none comes off it. Within a route, retry while one change (the words or
the seed) addresses what the stop showed; else move on.

**Web photograph** (its own one body, outside the 2): `"web_image": {"query": "<the object's noun
phrase>"}` in `plan.geometry`, words, never a URL (no `draw`, `asset_path`), its head noun a word of
the description or the instruction. The runner searches open collections for licensed
photographs, cuts the object out and shows up to 3 at the `web_image` stop. Judge each as a
drawing: one whole object, the object itself (not a print or a scene), a complete cut-out, no
person; `"pick": "<id>"` generates from it, `"pick": "none"` with `"why"` (one reason for all, or
per id) moves on. One search; a second, other words, only when the first returned nothing; none
usable: `blank` follows. Untried, `blank` searches the description's first clause (or
`web_image.query`) and shows the photographs; a `"pick": "none"` then lets `blank` build. A poor body from a picked photograph is refined first (geometry.md).

**The body (G4).** The runner reads where the body came from (`blank`, a generator, CAD or an
upload) and whether it is the brief's body:
- a built body whose thinnest member and mass are within 1.5× of the real ones (or the member alone
  within 2.5×) is the real body;
- a built body with thicker members is a section stand-in: its stiffness, area and mass ratios are
  printed; past the rule below it is "could not be run", so build at the Real dimensions.

One rule for every body: a load-path wall over 3 × its real thickness (`member_over_real`) or a
mass over 5 × the real (`mass_over_real`, at the parts' own densities) is another structure, "could not be run" (a generated one
is refused while a route remains, geometry.md). Except a recognisable generated body whose thin
real sheets, walls or tubes came out thicker or solid (`g4` says so): a marked thickened stand-in,
its self-weight scaled; the verdict is "simplified". Wrong extents are still refused.

**Thin members.** A member under 2 leaves (`thinnest_member_leaves`, the real member at depth 8, a
bar or wall to build) is warned and reported as a resolution caveat (too stiff in bending), never
refused: keep the body, or take depth 8. `blank` prints the smallest stand-in that resolves and
whether it keeps within 3×. A governing member under 1 leaf at depth 8: `blank` prints the
equivalent-stiffness section (raised to resolve, E scaled to keep EI); built at it (or any section of
its I within 5 %) with that E, it is exempt from 3×, compared as that section, and marked; any other body is refused before every solve.

Every body keeps the `Real extents` (sorted: the largest within 1.5×, the smallest too when a wall
or the Real section sets it, not a pose; an absent member forgives the one it shortens; a `--bars`
body also has a bar per member, two for a plural) and the offsets of where the loads and holds sit;
at the geometry stop it is recognisably the object. Once that stop refused a generated or web-photo
body's extents or mass, the brief's `Real extents revised: <a> x <b> x <c> m, <reason>` at the body's
own extents (within 1.1×), with `Real mass revised: <n> kg, <source>` when offered, is taken once
and marked; a new seed or depth never answers it. Any body over 2× the Real mass takes its self-weight scaled to the Real mass: the plan
keeps the use's acceleration (`values.acceleration_use`, else its own) and the runner writes
`values.acceleration` × Real mass / planned mass into plan.json; E and ρ stay the real material's.
A member left out of the stand-in that carries load to a support moves that support onto the
stand-in: its contact lists its own region there with
`"absent_in_stand_in": {"carried_by": "held_at_joint" | "load_at_joint" | "bar_kept"}` (the joint
held; the joint loaded with the support's statics reaction; a kept bar's end, with `"direction"`),
never another contact's region.

**A simplified body** declares `plan.body_simplification` `{"absent": ["<member>", …]}` (every other member is kept). It is taken once the listed routes were
tried (one the runner refused counts), or up front with `built` listed and the brief's reason.

**One section** (a box, a cylinder, or one profile extruded along one axis, with through-holes):
`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py blank SID --shape box|cylinder|extrude …` writes a
watertight body in metres under `operator/` and prints the plan's `geometry` and `scale` entries.
`blank` refuses until the brief has its headings, Real lines and Use table. A cylinder or an
extrusion names `--axis`, the axis its length runs along; the profile's two coordinates lie on the
other two axes in x, y, z order (`--axis z`: profile in x,y). Holes: `--hole 'x,y …'` or
`--circle 'cx,cy,r'`.

**Members in several planes**: `blank SID --bars FILE.json`, the file `{"bars": [{"from": [x, y, z],
"to": [x, y, z], "section": [w, h] | {"d": d}}], "boxes": [{"min": [x, y, z], "max": [x, y, z]}]}`:
each bar point to point with its section, plates and blocks as boxes, joined into one body. Every member
that carries load to a support is built; a bar under 2 leaves is warned. Separate pieces, each on
a support of its own, are built with `"pieces_supported": true`. Never invent a member or a clamp
to join or hold them.

**A body of revolution:** `blank SID --revolve 'r,z r,z …' --axis y`, r from the axis and z along
it: a closed outline (solid where it runs along r = 0), or an open line with `--wall T`. `--bars`
beside it adds members in the revolve's own frame; the body is then moved by `moved_by_m`.

**Joined prismatic parts:** CAD-Coder (`"generator": "cadcoder"`, `"revise_prompt": true`, a
description of geometry). At the `revision` stop check each part, dimension ratio and hole against
the brief; at the geometry stop the extents' ratios, thickness, mass and `cad_faces` (a hole is one
or two cylinder faces); a dimension or feature the brief keeps that came out otherwise is a wrong
geometry, not a caveat. `"segmentation": {"method": "cad_faces"}` names faces exactly.

**Free-form:** list `triposg_text`, `drawing` and `hunyuan3d`. One text attempt goes first (the
runner adds it; a new seed only when it returned no watertight surface), then the drawing. Hunyuan3D takes the same SDXL drawing: a good drawing TripoSG made poorly, once refined
(geometry.md), is given to it as it is (`"generator": "hunyuan3d"`, the same description and `image_seed`). The description,
at most about 20 words (geometry.md, Writing the description):

> "<material> <object>, <2-3 defining shape features>"

No negation, no view, pose or base words, no mesh words. A retry names the one missing part or
changes the seed. A body refused for identity goes to the next generator; a failed
free-form shape is not given to CAD-Coder.
