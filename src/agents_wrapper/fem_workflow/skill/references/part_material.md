# Part materials

Return one JSON object:
`{"parts":[{"id":0,"youngs_modulus":number,"poisson_ratio":number,"density":number}, ...]}`,
with every id in `parts` exactly once. Never invent an id or a spatial value; a
deterministic local worker fills each part's volume with its values after this
decision. Return no material name and no other field.

The image has two halves showing the same six views. The top half is the object in
its own colours. The bottom half paints each part one flat colour, and `parts` names
that colour for every id, with the part's size, position and facing. It is given
because it is known: find the part in the bottom half, then look at the same place
in the panel directly above it to see its colour, texture and sheen.

`segments`, when present, is an earlier reading of the parts that says what each one
is -- "part 0 is the seat cushion, part 3 the legs" -- and `object` is an earlier
reading of the whole object. Their credibility is intermediate: a separate vision stage
wrote them from one rendering, and they can misname a part or the object itself. Use them to tell which part is which where the picture leaves it
open, and weigh them against the picture: where a part plainly looks like one
material -- chrome, glass, wood grain, upholstery -- that settles it over a name in
either reading. They never override `request`.

Decide what each part is most likely made of. When `request` names a material for a
part, or for the whole object -- "a soft cushion", "the steel frame", "an oak chair" --
that is the answer for the parts it names: take it. The request is the one thing here
that is not a guess.

Only where the request is silent, fall back on the image, `object` (what the object
was read as) and `requested` (what it was asked to be); use their words for the parts
-- seat, leg, frame, top, cushion. Unlike a single value for the whole object, a part
is not replaced by whatever carries the load: the padding and the frame each take
their own material. Never ask a question; when several materials are plausible, take
the commonest for that part. Two parts that look the same take the same values, and a
part that mixes materials takes the one covering most of it.

Reading the picture:

- An upholstered seat, back or cushion takes its cover: leather where it is smooth
  and has a sheen, woven fabric where it is matte and textured, and felt where it is
  thick, fuzzy and uniform. Choose polyurethane foam only for a bare foam block or a
  soft pillow with no distinct cover.
- Mirror-bright or chrome metal is stainless steel. Painted or dark metal frames,
  legs and bases are structural steel. Light, brushed metal is aluminium alloy.
- Clear, tinted or frosted see-through panels are glass.
- Visible grain is wood; see its species rows. Layered edges are plywood. A flat
  painted or laminated board without grain is MDF.
- Moulded plastic parts are ABS unless they are clear (polycarbonate, acrylic) or
  waxy (HDPE). Small black feet, caps and bumpers are hard rubber unless they are
  plainly plastic or metal.

Return no prose, units, Markdown, or extra fields.
