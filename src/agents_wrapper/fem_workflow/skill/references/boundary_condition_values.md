# Boundary-condition values

Return only one JSON object:
`{"dirichlet_values":[...],"neumann_values":[...],"body_force":[x,y,z],"acceleration":[x,y,z]}`.
Return one entry per supplied region, as an object keyed by that region's exact `name`
from `region_statistics`:
`{"seat_traction":[0,-228.5,0],"backrest_plus_z":[0,0,-121.9]}`.

Writing the name beside the value is what keeps a value on the region it was worked out
for. Left to position, a description that mentions its regions in a different order than
`region_statistics` lists them produces loads on the wrong regions -- right magnitudes,
wrong places, and nothing in the result says so. Work each region out by name, and emit
it under that name.

A positional array is accepted only where a provider cannot emit an object: Dirichlet
entries in the order the `dirichlet` regions appear in `region_statistics`, Neumann
entries in the order the `neumann` ones do. A Dirichlet value
is `{"components":[...],"value":number}` where components are zero-based `0=x`,
`1=y`, and `2=z`; `value` is the shared prescribed displacement on those components.
A Neumann value is a three-component traction vector, or `{"force":[x,y,z]}` for a total force (below). Body force is a three-component
force-per-volume vector. Acceleration is a three-component length-per-time-squared
vector. Put gravity and self-weight in `acceleration`, not `body_force`; put a directly
specified force per volume in `body_force`. Both may be nonzero when independently
requested. Use `[0,0,0]` for either quantity when it was not requested.

Combine the prior region description with the new value description, giving the new
description precedence. Interpret quantities in the supplied unit system and convert
explicitly stated compatible units into the requested units. In this workflow,
upward is `+y` and
downward is `-y`, and the object faces `+z`, so forward is `+z` and backward is
`-z`: a load pressing into the front of the object has a negative z
component. Derive every value from the description. A region the description
says nothing about takes a zero value, which the controller replaces with whatever is
already set.
**A total force is returned as a force, not divided by you.** When the description
gives a Neumann region a total force, a weight or a mass to carry, return that region
as `{"force":[x,y,z]}` in `units.force` -- the controller divides it by the region's
measured area exactly. A mass becomes its weight: mass times standard gravity
(9.80665 m/s^2), converted into `units.force` (10 kg in SI is `98.0665` N). Return a
plain `[x,y,z]` traction only when the description gives a traction, a pressure or a
stress, which is already per area -- never divide that by area.
`{"top_load":{"force":[0,-98.0665,0]},"side_pressure":[-2000,0,0]}` is a 10 kg load
on the top and a 2 kPa pressure on the side.
Never omit an entry, and never return an empty list where regions of that kind were supplied -- a case with no `neumann` region in `region_statistics` returns `"neumann_values":{}`, which is the one empty answer there is. Never invent a nonzero magnitude. Return no prose, units, Markdown, or extra fields.
