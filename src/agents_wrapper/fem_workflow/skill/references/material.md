# Material parameters

Return one JSON object. For the normal uniform case return
`{"mode":"uniform","youngs_modulus":number,"poisson_ratio":number,"density":number}`.
If and only if the request explicitly asks for VoMP, adaptive material assignment,
or a spatially varying/heterogeneous material field, return `{"mode":"vomp"}`.
Never invent an artifact path or spatial values; a deterministic local worker creates
and validates those after this decision. Return no material name and no other field.

Decide what the object is most likely made of. When `request` names a material, or
names the part being analysed -- "a soft cushion", "the steel bracket" -- that is the
answer: take it, even when it is the padding rather than the frame. The request is the
one thing here that is not a guess.

Only when the request is silent, fall back on `description` and the image. The value
feeds a stiffness analysis, so default to whatever carries the load -- the frame,
shell, or structural member -- over a coating, a fastener, or the padding and
upholstery wrapped around it. Never ask a question; when several materials are
plausible, take the commonest for that object.

Return no prose, units, Markdown, or extra fields.

Do not infer yield strength. The user supplies it in the safety-factor viewer.
