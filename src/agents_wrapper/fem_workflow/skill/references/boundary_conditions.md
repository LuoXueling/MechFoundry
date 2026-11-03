# Boundary-condition selection

Read by an agent working the skill directly. The API call never sees this file:
it is sent `boundary_conditions_api.md`, the same task written to a word budget
because it goes out as a system prompt on every call. Two files, two readers --
keep their rules in step.

Return only `{"dirichlet":[...],"neumann":[...]}`. A Dirichlet item is
`{name,selector,value,components}`; a Neumann item is
`{name,selector,traction}`. Each selector is `{type,params,children}`. Use only
selector types supplied in the request. Keep names short. Supports may constrain all
components or an explicit subset. Loads are constant 3-vectors; use `null` when the
user omitted magnitude. Select at least one support and one load. Never invent a
magnitude. Do not emit prose, code, repeated metadata, or unused optional fields.

## Selector representation and use

Every selector is recursive JSON:

```json
{"type":"highest","params":{"axis":"z","fraction":0.05},"children":[]}
```

All coordinates and distances use the supplied **physical** coordinate system and
units. Spatial selectors test triangle centroids, not vertices. Use `x`, `y`, or `z`
for axes. Fractions are in `(0,1]`; percentiles are in `[0,100]`; angles are degrees.
Use a small positive tolerance around intended planes because face centroids rarely
lie exactly on a bounding-box boundary.

World-direction language has a fixed convention: **upward is `+y` and downward is
`-y`**, and the object faces `+z`, so **forward is `+z` and backward is `-z`**.
Therefore, an upward traction has a positive y component and a downward
traction has a negative y component, for example `[0, magnitude, 0]` and
`[0, -magnitude, 0]`; a traction pressing into the front of the object points
`-z`, for example `[0, 0, -magnitude]`. Apply the same convention to positional
descriptions: "top/upper" means the high-y side and "bottom/lower" means the
low-y side, "front" means the high-z side and "back/rear/behind" means the low-z
side, unless the user explicitly names another axis or direction. Do not
reinterpret upward, downward, front, or back from the geometry's shape or camera
orientation.

Leaf selectors have no children:

- `bounding_box`: `min_pt`, `max_pt`; select centroids inside an axis-aligned box.
  Best for a known local patch. Derive both corners from geometry bounds.
- `sphere`: `center`, `radius`; select a compact neighborhood. Use as a seed, not
  for a broad flat support.
- `cylinder`: `p1`, `p2`, `radius`; select near the finite axis segment (distance is
  measured to the closest point on the segment). Use for holes, shafts, or legs.
- `half_space`: `point`, `normal`; select the side satisfying
  `(centroid-point) dot normal >= 0`. Reverse `normal` to select the other side.
- `slab`: `axis`, `min_val`, `max_val`; select an absolute coordinate interval.
  It is a band, not a plane: faces are tested by their centre, so `min_val` and
  `max_val` must differ by enough to contain one. To catch the surface at a
  coordinate, span the feature you mean -- the bounds the geometry reports for
  that face -- rather than repeating one number twice.
- `lowest` / `highest`: optional `axis` (default `y`) and `fraction` (default `0.1`);
  select the bottom/top fraction of the mesh centroid range. Prefer these for ends.
- `percentile`: `axis`, optional `lo_pct`, `hi_pct`; select a percentile band when
  absolute dimensions are uncertain.
- `furthest_from_center`: optional `fraction`; select the radially outermost faces.
- `normal`: `target_vector`, `max_angle_deg`; orientation-insensitive, so both
  parallel and antiparallel normals match.
- `aligned_normal`: `target_vector`, `max_angle_deg`, optional `oriented` (default
  `true`); use for only outward, upward, or downward facing surfaces.
- `curvature`: `min_kappa`, `max_kappa`, optional `curvature_type` (`mean` or
  `gaussian`); use only when meaningful thresholds can be inferred from diagnostics.
- `flat_region`: optional `max_curvature`; select low mean-curvature faces.
- `sharp_feature`: optional `min_curvature`; select high mean-curvature faces.
- `dihedral_angle`: optional `min_angle_deg`, `max_angle_deg`; select faces by their
  mean angle to adjacent faces.

Composition and topology selectors wrap children:

- `and`, `or`, `subtract` require two children. They mean intersection, union, and
  `children[0]` minus `children[1]`, respectively.
- `not` requires one child and selects its complement. Avoid an untrimmed complement
  because it commonly produces an excessively large BC.
- `flood_fill` requires one seed child and optional `max_feature_angle_deg` (usually
  20--45). It grows across connected faces but stops at sufficiently sharp turns.
- `grow_selection` requires one child and optional `num_rings`; it adds that many
  face-adjacency rings.
- `boundary_edge` requires one child and keeps the perimeter faces of its region.
- `connected_component` accepts zero or one child, plus optional `component_index`
  and `sort_by`: `size`, `area`, or `centroid_x|y|z`. Index zero is the largest for
  size/area; use it to discard disconnected leakage after a broad selector.
- `surface_segment` takes reviewed `segment_ids` and no children.
- `shape_diameter_segment` takes one seed child for on-demand shape-diameter expansion.
- `preview_segments`: a deprecated spelling of `shape_diameter_segment` with
  `segment_ids`; accepted for saved sessions, not for new ones
  from the saved segmentation preview. It has no children. Use it whenever the user
  explicitly names displayed segment numbers.

Prefer the simplest robust selector: position first, position plus oriented normal
second, then topology or segmentation. For example, an upward-facing patch near the
top is:

```json
{"type":"and","params":{},"children":[
  {"type":"highest","params":{"axis":"y","fraction":0.08},"children":[]},
  {"type":"aligned_normal","params":{"target_vector":[0,1,0],"max_angle_deg":25},"children":[]}
]}
```

Always choose parameters from supplied bounds, segment metadata, and the user's
words. Never invent an unseen coordinate. The local resolved preview is the final
check; if it is empty, broad, disconnected, or on the wrong side, revise the
selector rather than changing the requested BC.

## Using Surface Segmentation

The montage and `geometry.segments` describe the same reviewed segmentation, whether
it was produced by shape diameter, PartField, or the faces of a generated CAD model.
Each segment has a stable numeric diagnostic ID, color, bounding box, centroid, and
face/area fraction. Shape-diameter statistics are also present for that backend; a
CAD-face segment is exactly one face of the part and names its `cad_surface` type
(`plane`, `cylinder`, ...), so a bore or bolt hole is its own cylinder segment. Use the available evidence together to
identify the part described by the user. The locally rendered and resolved preview
is authoritative.

The user may refer directly to the displayed diagnostic number, for example
"segment 2 is fixed" or "apply the load to segments 4 and 5." Resolve each such
reference against the supplied `geometry.segments[].id`; never substitute a nearby
number or infer a missing ID. Put all IDs for the same BC region in one selector's
`params.segment_ids`, for example:

```json
{"type":"surface_segment","params":{"segment_ids":[2,4,5]},"children":[]}
```

Given ids, the selector reads the reviewed segmentation directly and never recomputes
it. Prefer this whenever the wanted part is one or more of the displayed segments,
whether or not the user typed the number: it costs nothing and selects exactly the
partition shown in the preview.

`surface_segment` does **not** accept `segment_id`, `seed`, or children. Only when no
displayed segment matches what the user described, construct the smallest robust
seed that clearly falls inside the desired part and wrap it in the legacy
`shape_diameter_segment` expansion selector:

```json
{
  "type": "shape_diameter_segment",
  "params": {
    "num_clusters": 5,
    "num_rays": 25,
    "smoothing": 0.25,
    "cone_angle_deg": 120,
    "adjacent_segments": 0
  },
  "children": [{
    "type": "highest",
    "params": {"axis": "z", "fraction": 0.05},
    "children": []
  }]
}
```

Choose seeds in this order:

1. Prefer `highest` or `lowest` when the segment is separated along a world axis, and
   a `slab` or tight `bounding_box` when a specific interval is meant. Read bounds off
   the supplied physical metadata rather than composing them, and give a slab two
   distinct numbers. `percentile` is seldom what is wanted: it is a quantile of face
   centres, so what it covers moves with the tessellation.
2. Add `aligned_normal` with `and` when position alone would seed multiple parts.
3. Use a small `sphere` around the supplied segment centroid only when that centroid
   lies on or close to the desired surface patch.
4. Use `connected_component`, `flood_fill`, or `grow_selection` only when the simpler
   seed-and-expand construction cannot express the user's region.

Keep `num_clusters`, `num_rays`, `smoothing`, and `cone_angle_deg` exactly equal to
the segmentation metadata settings so the preview and selector recompute the same
segmentation. Do not guess or alter these values. Leave `adjacent_segments` at zero
unless the user explicitly wants a
neighboring segment included. If inclusion is required, use the smallest count and
set `adjacent_direction` when the intended direction is clear. Use boolean
intersection after expansion to trim an overly broad segment; use union only when the
user named multiple regions.

Do not infer a region solely from color, because palette colors can repeat when there
are many segments. Confirm it with its ID-linked bounds, centroid, and orientation.
