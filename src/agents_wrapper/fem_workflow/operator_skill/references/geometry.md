# Geometry: build, generate or upload; judge; scale

The brief narrows the routes to the body to those that can make it (`plan.geometry_routes`,
generated routes first, then a web photograph, built last, objectives/routes.md §3.4). This file says what each route controls and costs,
and how to judge and scale what comes out.

## What each route controls

| Route | Input | Controls | Does not control | Cost |
|---|---|---|---|---|
| built (`python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py blank`) | dimensions | every dimension: a box, a cylinder, one extruded profile with through-holes, a frame of bars (`--bars`), a body of revolution (`--revolve`) | a section that changes along a member other than by revolution, a blind hole | seconds; the `upload` limit |
| `triposg` (default generator) | text, image, scribble (+text) | the overall shape it is told or shown, recognisably | exact dimensions (the scale card sets the size); a wall or member under about 1 % of the largest extent, which comes out thicker; features it was not told about | 5–20 s warm, about 7 GB |
| `cadcoder` | text, revised into coordinates | parts sketched in one plane and extruded along its normal, through-holes as inner loops; the ratios of the dimensions given | features in a plane other than the sketch plane; fillets, threads, revolved or lofted surfaces; the absolute size (the scale card sets it) | about 18 GB |
| `hunyuan3d` | an image: a drawing (`"draw": true`) or the user's (`asset_path`, `"asset_kind": "image"`) | shape from the picture (it arrives grey) | as TripoSG | cold about 71 s, warm about 36 s, 7–13 GB; its own limit (2) |
| upload (`asset_kind: "surface"`) | a watertight GLB/OBJ/PLY/STL | everything the file holds | | seconds |

One generator model stays loaded at a time (a switch is a cold start).

## Writing the description

- **TripoSG text and a drawing:** at most about 20 words, `"<material> <object>, <2-3 defining
  shape features>"` (objectives/routes.md §3.4). No negation (a negated part gets drawn), no view,
  pose or orientation words (the workflow's style frames it), no base, mount or stand unless the
  object has one, no mesh words (solid, watertight, hollow, empty inside: they change nothing).
  The runner refuses a negation, mesh or view words, or a drawing's words that overflow the drawing model.
  A retry changes one thing: names the one missing part, or the seed; replace words, never add.
- **Image:** `description` is still required: what the object is.
- **Scribble:** `asset_kind: "scribble"`, `condition_on_text: true` and a description (TripoSG).
- **CAD-Coder:** the part as it is made, in geometry, not names: every part, feature and hole,
  each outline, extrusion, position and diameter, in real proportions. Send `revise_prompt:
  true`, read the revision at the `revision` stop, and run again to generate from it. Revisions
  are counted: 2 before each CAD generation, so fix every departure in one edit of the words.
- **Keep `erosion_radius` at 0** (TripoSG): a positive radius cuts off thin members.
- **TripoSG's extraction depths** (`generation_settings`): text stays at `init_octree_depth` 4,
  `octree_depth` 7. A body made poorly (fragments, a member missing or broken, a thin part lost)
  from an image that passed its stop is first refined on the same image and seed (`geometry.reject`
  names the fault: fused, missing), never a new image or route: (a) 6/9 (5–20 s, about 10 GB free); (b) if that collapsed, failed or is still
  poor, 5/8; (c) Hunyuan3D from the same image (first when the members are thin). (a) and (b)
  (at most 2 per image) and (c) (once) are no new body; init is always depth − 3. No watertight surface: one
  new seed at the same depth. The runner refuses other depths. The ladder is for poor shape, never
  mass: for a body over 5× the Real mass (G4) the runner skips (a)–(b), takes (c) while Hunyuan3D
  is untried, else the photograph, then built; thin real members generated thicker are accepted
  (objectives/routes.md §3.4).

## Judging the generated surface

`geometry/views.png` (six views, +y up, the object facing +z) and the stop's geometry block:
- `watertight: true` and `components: 1` are required; a generation failing them is rejected.
- `discarded_components`: TripoSG drops small fragments; check the views for a missing part.
- The thinnest load-bearing member: under 2 leaves after scaling is warned
  (`planned_resolution`, visual_checks.md §1).

**Retries, within a listed route.** Retry while one change addresses what the stop showed;
otherwise take the next listed route:
- A drawing: change the words (name the missing part, say how it stands, "one single …, alone")
  or `image_seed`; up to 3 drawings. A good drawing TripoSG made poorly takes the ladder (Writing
  the description; Hunyuan3D: `"generator": "hunyuan3d"`, the same words and seed); up to 2
  Hunyuan3D attempts, each one drawing, rejected or not. TripoSG text:
  once, first; a new seed only after no watertight surface.
- CAD-Coder: change the description where the revision or the part departed from the brief; up
  to 3 CAD generations. A part that reduces to one extruded section is built instead.
- Built bodies: rebuild with corrected dimensions; up to 3.

## SDXL drawing (after one TripoSG text attempt)

TripoSG's text mode has no picture to follow and fails outright on some wordings; its image mode
follows a picture closely. When `sdxl.ready` is true, the geometry card draws one:
1. `{"action":"draw_image","description":"<the object>","image_seed":0}` (about 13 s, 8–13 GB).
   **Keep the description to about 20 words** (Writing the description): SDXL reads 77 tokens,
   and the workflow adds its own single-object style after your words.
2. The `drawing` stop shows it: recognisably the whole object, only one, its primary members
   (visual_checks.md §1), a little above and to one side; a drawing that fails a check, or reads 2 or more `drawing.objects`
   without `"drawing_crop": "largest"` (its `largest_object`, when that is the whole object),
   is never generated from by any route (the runner refuses it): new words or `image_seed` draw
   again. With the drawing limit spent, a
   fresh drawing made for Hunyuan3D (new words or `image_seed`) is Hunyuan3D's own attempt, when
   the routes list it; else the next listed route (`drawing.last_drawing` says so).
3. Run on: it generates from the drawing (TripoSG, or Hunyuan3D from the same drawing).
4. TripoSG sets the orientation from the drawing's viewpoint: read the views before `up_axis`.

In a plan, `"draw": true` does steps 1–3.

## Texture (painted once the geometry is final)

`plan.texture` `{"texture_prompt":"<look>"}` paints every body in the background once the regions
are previewed, joined at the material card. Without it a uniform-material body stays grey, and
`mode: parts` paints from its own words. `generation_settings.texture: true` (Hunyuan3D) asks for a
coloured body, which is not repainted. The `texture` stop shows the painted views; new words or
seed paint again. Only vertex colours.

Write the look as the material and one mid or light colour in a few words (`"<colour>
<material>"`, a finish word at most). A body generated from an image (drawing, web photograph)
is painted from that image (the runner adds it). From words alone the painter draws one
reference picture and spreads its body colour over every face, so lighting, gloss, wear and
colours tied to places come back as a darker, mottled or two-tone body; dark, polished or glossy
words paint near-black. A paint that came back
right is kept whatever its words. Repaints are counted: 2 after the first painting (about 30 s
each). **The painter places no colour by part.** When the parts must differ in colour (`mode:
parts`), give `{"per_part_colours": true}`: each part one flat colour, its material's (wood brown,
steel grey, rubber dark, a plastic in its words' colour), deterministic and uncounted;
`{"per_part_colours": {"<part id>": "<colour words>"}}` names one, with ids from the part montage.
A face that comes back flat grey while the rest is coloured is a failed view: change
`texture_seed`. The stop measures the paint and warns (never refuses) when over 5 % is near-black
(L* under 15, C* under 12) on a colour that is not dark, when its facings differ by over 30 in
lightness, or when under half of the surface is the hue the words name; faces looking straight
down or on a held region are listed apart. The views picture then hatches the measured vertices:
repaint only when the body itself shows the defect, never on the numbers alone.

## Scaling (`geometry_physics` card)

`{"unit_system":"SI (m-kg-s)","up_axis":"y","physical_extent":<m>,"scale_description":"<what the size is>"}`
- **`up_axis` is the axis the extent is measured along**, not an orientation. For an upload the
  card pre-fills z and the file's own z extent: accept it only if the file is at its real size.
- The scale is uniform, puts the minimum corner at the origin, and snaps to the octree lattice
  (about 1 %). CAD-Coder's output size is set here; the geometry stop's extents are the check.
- Real sizes only, reasoned in the brief; `infer_scale` is a check, never the source.

## What the generator does (keyed by what the geometry stop shows)

| You see in `views.png` | What happened | Change the words to |
|---|---|---|
| a different object | the name alone was ambiguous | describe the working part's shape instead of the name |
| the body on a thin stalk or rod | a hanging or suspended wording | describe it standing on the ground; the brief states the substitution |
| only part of the object, or a close-up of one end | a long slender object with a small working end | "the whole <object> from end to end, small in the frame" |
| a hollow interior full of clutter | an opening left in the description | describe the openings as closed, unless the object is open by its use |
| an open object filled to the brim, or closed over | "solid" in the words of an open object | its opening and its empty interior, no "solid" |
| a primary member missing (another: declared, visual_checks.md §1) | a thin part the generator dropped | from a passed image, refine first (Writing the description); else name it: its shape and where it joins |
| members far thinner than described | a part thin by nature; adjectives do not thicken it | keep it (a resolution caveat) or depth 8 (objectives/routes.md §3.4) |
| lying on its side or upside-down | no orientation words | say how it stands or lies (no view words) |
| the outline, but not recognisably the object | the words named parts, not how it looks | add its characteristic features and proportions |
| the body smaller than the size you gave | a thin appendage is the longest extent | keep it; give heights in metres in the region words |
