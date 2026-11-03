# Physical scale

Return only one JSON object:
`{"reference_axis":"x"|"y"|"z","physical_extent":number}`.

`reference_axis` is the axis the size below is measured along. It is not a claim
about which way is up. When `request` or `description` names a direction for the
dimension -- "1 m in z", "40 cm along x" -- return that axis, whatever the object's
orientation. When a dimension is named without an axis, map it to the axis it
describes, in a workflow where up-down is `y` and the object usually stands along
it, and front-back is `z` because the object faces `+z`: height and tall are `y`,
depth and deep are `z`, width and wide are `x`. With nothing stated either way,
return `y`.

`physical_extent` is the object's real-world size along `reference_axis`, expressed
in `length_unit`. Use a dimension stated in `request` or `description`, converting it
to `length_unit`. Otherwise estimate the typical real size of the named object along
that axis. Return one positive number, never zero, a range, or a unit string.

An image, when supplied, is low resolution and shows only what the object is and
how it is oriented; read no dimensions from it. Return no prose, Markdown, or
extra fields.
