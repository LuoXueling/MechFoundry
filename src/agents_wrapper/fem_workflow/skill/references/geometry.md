# Geometry refinement

Return one JSON object: `{"refined_prompt":"..."}`. Edit the user's description into a concise,
generation-ready description of one watertight 3D object. This is an edit, not a
summary: retain every explicit geometric requirement. In particular, never drop or
alter part counts, dimensions, units, proportions, topology, holes, cutouts, openings,
symmetry, orientation, relative placement, connections, thicknesses, cross-sections,
or shape-defining exclusions.

Adapt the amount of detail to the source:

- If it is only a short or generic label, expand it into a useful generation prompt.
  Add conservative shape descriptors consistent with the named object and any attached
  image, such as the dominant form, solid/watertight construction, broad proportions,
  edge or surface character, symmetry, and how already-mentioned parts connect. Do not
  invent numeric dimensions, part counts, holes, cutouts, or new functional features.
- If it is already detailed or long, compact it. Merge repeated requirements, remove
  filler and non-geometric information, and organize the remaining constraints by main
  form, features, and spatial relationships. Every unique geometric fact must survive.

Remove only content that cannot affect the generated surface, such as conversational
lead-in, analysis purpose, loads, supports, material properties, scenery, rendering,
and presentation requests. A negative clause may be removed only when it is not a
geometric constraint. Return 5--15 words in one compact phrase. Within that limit,
prioritize explicit user constraints over decorative detail and do not repeat
synonymous phrases.

Return only `refined_prompt`. Never ask a question. If the source contains unresolved
ambiguity, preserve the ambiguous wording rather than guessing, deleting it, or
requesting clarification.

An attached image or scribble remains unchanged. Treat the text and visible geometry
as complementary evidence; do not discard an explicit textual constraint merely
because it is unclear in the image. For an image-only checkpoint, the text is intent
metadata and must not be described as model conditioning.
