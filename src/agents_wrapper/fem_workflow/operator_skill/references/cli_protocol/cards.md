# Driving the workflow: cards and answers (cli_protocol §2)

## 2. Cards and answers

Values in `<>` are yours; *(merged)* marks what the driver merges from the card.

### `geometry_input`

**Card:** `description` (after a revision, the revised text); `cad_revision {generator, prompt}`;
`generator`; readiness of `cadcoder`, `hunyuan3d` and `sdxl` (which draws an image input);
`drawn_image` (the newest SDXL drawing, cut out; `drawn_image_raw` as drawn), the
`drawn_image_prompt` it was drawn from, and `image_seed`, and after a drawing what was sent with
it (`asset_kind`, `generation_settings`); `api_error`.

**Answer:**
```json
{"description": "<text>",                // required unless asset_kind is "surface"; (merged)
 "generator": "triposg|cadcoder|hunyuan3d",
 "asset_path": "<abs path>", "asset_kind": "image|scribble|surface",
 "condition_on_text": true,               // scribble only
 "revise_prompt": false,                  // true = one API call, back to this card with the rewrite
 "use_drawn_image": false,                // true = the newest drawing as the image input
 "generation_settings": {"seed": 42, "faces": -1,
                         "init_octree_depth": 4, "octree_depth": 7, "guidance_scale": 7.0,
                         "erosion_radius": 0,
                         "texture": true, "hunyuan3d_octree_resolution": 384}}
{"action": "draw_image", "description": "<text>", "image_seed": 0}
                                          // SDXL draws the description; back to this card
```
- Leave out `inference_steps` (or the whole block) for the route's default: 16 steps for text,
  50 for an image, 30 for Hunyuan. `seed` ranges over 0..100.
- A surface upload skips generation.
- An image comes from the user, `draw_image` or a web photograph, copied into
  `<session>/inputs/original.*`; an upload is cut out by the generator, the others arrive cut out.
- `draw_image` needs `sdxl.ready`; an omitted description is the card's. Each drawing is
  `inputs/sdxl/<description hash>_s<seed>.png`, served from the cache for the same words and
  seed. `use_drawn_image` takes the newest drawing; leave `asset_path` and `asset_kind` out.

### `geometry_texture`

Only for a colourless surface when the painter is ready.

```json
{"action": "skip_texture"}                                       // default; the runner always skips it
{"action": "texture_surface", "texture_prompt": "<look>", "texture_seed": 0, "texture_image": "<abs path>"}
{"action": "use_texture"}                                        // adopt the painted copy
```

### `geometry_physics`

**Card:** `up_axis`, `physical_extent`, `unit_system`, `measured_extents` (uploads only),
`unit_details`.

**Answer:**
```json
{"unit_system": "SI (m-kg-s)|SI-mm (mm-tonne-s)|US-inch (in-lbf-s)",   // every answer (merged)
 "up_axis": "x|y|z", "physical_extent": <number>,                    // the axis the size is measured along (merged)
 "scale_description": "<text>", "action": "apply_scaling|infer_scale"}
```
- `infer_scale` makes one API call and returns to the card with the values filled in.
- A field the answer leaves out keeps the card's value: `{"physical_extent":0.3}` changes the
  extent alone. An upload's pre-filled pair: geometry.md, Scaling.

### `bc_input`

**Card:** `description`; `segmentation {method, num_clusters, num_rays, smoothing,
cone_angle_deg}`; `has_segmentation`, `has_regions`; `segment_description`, `preview_error`,
`partfield.ready`, `cad_faces`.

**Answer:**
```json
{"action": "preview_segmentation|preview_regions|revise_prompt|approve_regions",
 "description": "<supports and loads>",
 "segmentation": {"method": "partfield|shape_diameter|cad_faces", "num_clusters": 5}}   // partial is fine (merged)
```
- `preview_regions` re-segments first if the settings changed; `approve_regions` needs
  `has_regions`.
- The regions are in `boundary_conditions/proposal.json` and `resolved.json`, not on the card.

### `bc_values`

**Card:** `dirichlet [{name, components, value}]`, `neumann [{name, traction, area}]`,
`body_force`, `acceleration`, `units`, `value_description`.

**Answer:**
```json
{"action": "apply_values",                               // default
 "dirichlet_values": {"<name>": {"components": [0,1,2], "value": 0.0}},   // unlisted regions keep the card's values (merged)
 "neumann_values": {"<name>": [0, -5000, 0], "<name2>": {"force": [0, -500, 0]}},  // force: divided by the loaded area
 "body_force": [0,0,0], "acceleration": [0,-9.80665,0], "value_description": "<text>"}
{"action": "infer_values", "value_description": "<text>"}   // one API call; back to this card
```
- Region values go **inside** `dirichlet_values` / `neumann_values`. The workflow reads only the
  keys above and ignores others, so `{"<region>": {"force": …}}` at the top level would leave
  the card's traction in place; the driver rejects it.
- Every Neumann region needs a traction before the solve. A zero traction is accepted but makes
  the region a free surface; the driver warns.
- The `neumann` rows carry `loaded_area`, the cosine-weighted area the solver loads: a
  `{"force": …}` is divided by it, so the stated force reaches the part (`notes` show the
  resulting traction); a typed traction is applied as the solver applies it, totalling
  traction × `loaded_area`.

### `material_input`

**Card:** `material_mode`, `youngs_modulus`, `poisson_ratio`, `density`; `material_parts`,
`parts_montage`, `parts_summary`; `material_artifact`, `material_summary`;
`surface_has_colour`, `textured_surface`; `units`, `api_error`.

**Answer:**
```json
{"material_mode": "uniform", "youngs_modulus": <E>, "poisson_ratio": <nu>, "density": <rho>}   // apply (merged from card)
{"action": "infer_material", "material_description": "<text>"}          // one API call; fills the card
{"action": "texture_surface", "texture_prompt": "<look>"}               // paint for parts/VoMP
{"action": "infer_parts", "material_description": "<what parts are made of>"}
{"action": "apply_parts", "parts": [{"id": 5, "youngs_modulus": 1.1e10}]}   // unlisted parts kept (merged)
{"action": "infer_vomp"}
{"material_mode": "parts"}  /  {"material_mode": "vomp"}                // approve a field
```

### `shape_opt_settings`

**Card:** `defaults` (the current settings, solver_settings.md); `strain_estimate`,
`strain_floor`; `solve_error`, `solve_log`, `solve_log_path`, `solve_complete`.

**Answer:**
```json
{}                                             // run with the card's settings
{"use_mixed_precision": true}                  // change fields; the rest are kept (merged)
{"run_mode": "shape_optimization", "gd_lr": 0.7, "max_optimize_iters": 100}
{"action": "finish"}                           // the only way to end the session
```
- Every settings answer is a new solve in a new `solve/api_fem_forward-<ts>/` folder.
- The card carries `finish_answer: {"action":"finish"}` and stays open after a solve so the case
  can be run again.
- A failed solve deletes the coloured field GLBs at the top of `solve/` and writes
  `solve/failed.json` (`started`, `error`, `log`); surfaces older than `started` are then not
  offered as results. The next successful solve removes the marker.
- After a successful solve the driver exports the field GLBs (`field_assets`).
