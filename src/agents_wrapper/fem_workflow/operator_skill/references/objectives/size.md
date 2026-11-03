# From the brief to the plan (objectives §3.1–§3.2): size, resolution, material

### 3.1 Size and resolution

`physical_extent` is the brief's size along `up_axis`: `"y"` with the height for anything that
stands taller than it is long; `"longest"` with the length for anything longest along the
ground. When an appendage is the largest extent, the body comes out smaller than you think: give
heights in region words in metres from the stop's extents.

The geometry stop prints `planned_resolution`: the leaf size (1.92 × the largest extent /
2^depth, about 1.5 % of the object at depth 7) and the thinnest member in metres and leaves. A
member under 2 leaves is too stiff in bending: it is warned with its numbers, never refused, and
the report carries it as a resolution caveat. A real size that leaves a member thin is still the
right size: keep it, or take depth 8 (the runner writes it when that brings the real member to 2
leaves); never shrink the object or simulate a part of it (a thickened body:
objectives/routes.md §3.4).

The orientation the body came out in never changes the case, and no tool turns a body: each
contact and load stays on the part the brief named. Rows' directions and `values.acceleration`
stay in the use frame (+y up), never edited by hand; when views.png shows the body not upright,
state the body axis that points up in use as `"body_up": "+x" | "-x" | "+z" | "-z" | "-y"`,
and the runner turns every row direction and the acceleration with it
(`planned_resolution.orientation` prints the axes beside the brief's `Up in use`). A change of rule (another contact, load face, direction or
division of the load) goes under "Brief changes", derived from the unchanged Use sentence, and
the verdict is then "a variant of the brief's case was run".

### 3.2 Material

materials.md's table is the one reference (a value inside the band for the grade, stated);
judge brittle materials on tension and wood along the grain.
