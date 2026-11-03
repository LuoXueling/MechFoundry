# Driving the workflow: the plan file (cli_protocol §6)

## 6. The plan file (`run`)

One JSON object, one entry per card, written from the case brief (`run` refuses to start until
it exists with its four headings): its skeleton and the stop's compact view are SKILL.md's (steps
2–3). A stop's `step_warnings` are what a step reported while succeeding (a preview that kept the
previous regions, a solve that fell back); a failure carries `error` (`errors` for a lint miss,
`solve_error` for a solve). The rules for each field:

- `geometry`, `geometry_routes` (unless the user gave the file), `scale`, `regions`, `values`
  and `material` are required; the rest may be left out. `geometry_routes` lists, from
  `triposg_text`, `drawing`, `hunyuan3d`, `cadcoder`, `web_image`, `built` (the runner orders them), the
  routes the brief judged able to make the body (objectives/routes.md §3.4). Extra keys inside
  `geometry` go to the card as they are (§2).
- `scale.up_axis` is `x`, `y`, `z`, or `"longest"` (the generated mesh's longest axis). The
  geometry stop prints `planned_resolution`: the extents and the leaf size under the plan's scale.
  `body_up` (top level, `+y` when left out) is the body axis that points up in use, for a body
  not upright (objectives/size.md §3.1).
- Each route has its own attempt counter: `generation` (TripoSG text), `drawing`, `web_image`,
  `hunyuan` (`generator: hunyuan3d`), `cad` (`generator: cadcoder`), `upload` (`asset_kind:
  surface`, a built blank or a user's mesh).
- `revise_prompt: true` revises the description first; edited words are revised again. For
  CAD-Coder it stops at `revision` before generating; its revisions are counted, 2 before each CAD
  generation (`cad_revisions`).
- `draw: true` draws (`image_seed`), stops at `drawing`, then generates; `"drawing_crop": "largest"`
  beside it generates from the drawing's largest object alone; `web_image`: routes.md §3.4;
  `reject`: the check a drawing or body that passed the runner's checks failed, when the plan
  moves past it (visual_checks.md §1).
- `texture`: `{"texture_prompt": "<look>"}` paints every body in the background once the regions are
  previewed, joined at the material card; without it a uniform body stays grey in the figure. The
  words are the material and one mid or light colour, `"<colour> <material>"`, a finish word at
  most: the painter spreads one body colour over every face, and dark, polished or glossy words
  paint near-black. A body generated from an image (drawing, web photograph) is painted from that
  image (the runner adds it); the words then only name the material and a fallback colour. With
  per-part materials, `{"per_part_colours": true}` (alone) colours each part flat, deterministic
  and uncounted; `{"per_part_colours": {"<part id>": "<colour words>"}}` names
  one; else one colour per part, `"<colour> <material> <part>, …"`. `run` refuses more than 8
  words or a second colour (per part). `"texture_seed": <n>` paints the same words again
  differently. Repaints are counted: 2 after the first painting (`attempts_left.repaints`).
- `segmentation`: `{"method": "partfield", "num_clusters": <n>}` (boundary_conditions.md §1);
  `region_revisions`: `{"<synthesized region>": <its selector, corrected>}`. A reword re-synthesises every region; `--patch
  '{"region_revisions": {"<name>": null}}'` removes a revision.
- `regions`: the supports and loads in words. Words the card already previewed are approved; new
  words preview again.
- `values.load` goes to every load region: a traction `[tx, ty, tz]` (a fixed direction, stress
  units) or a total `{"force": [fx, fy, fz]}`; `neumann_values` by name wins over it. Every support
  region is held in all components unless `dirichlet_values` names it. `reference_hold`
  `{"regions": [<a held region>]}` names the hold beside rests applied as reactions; optional.
- `loads`, written at the regions stop, binds each row of the Use table's run state (by its label
  or its `"load"`): `{"regions": [<its load regions>]}` or `{"acceleration": true}` (applied as
  `values.acceleration` or `body_force`). A run state of one row takes every load region when
  `loads` is left out. The runner applies each bound row with its size, the total shared among
  its regions (the two faces of a squeeze are a row each), along the row's `"direction"` or the
  plan's value there, which then gives the direction only; a twist about its `"direction"` as a
  couple on its regions. A row whose runner effect is under a tenth of the run's largest is "not
  applied (minor)". A body load (mass × acceleration) over 20 % short of its rows' sizes is refused
  without `"shortfall_why"` on the row's entry, and is a deviation in the judgement.
- `material`: uniform, in the session's units. `{"action": "infer_material",
  "material_description": "…"}` infers once. `{"mode": "parts", "description": "<what each part is
  made of>", "parts": [{"id": 2, "youngs_modulus": 2e11}]}` is the per-part field (materials.md):
  the runner colours a bare surface by part, values the parts, fills them from the brief's
  `Materials by part`, applies the listed corrections (the `materials` stop), then approves. `mode: "vomp"` is refused.
- `settings`: every field is sent, the class defaults with `use_mixed_precision` on, then the
  plan's entries (unknown keys dropped). After a finished solve, `run` with unchanged settings finishes the session, with
  changed settings solves again. After a failed solve, unchanged settings are refused; `run
  --two-depths` solves that case one depth coarser and again, once, for one solve attempt.
  `solve-backend` re-runs `case/case.json` with the last graph solve's settings.
