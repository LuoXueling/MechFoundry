# Boundary-condition prompt

Return one JSON object: `{"refined_prompt":"..."}`. Edit the user's text into a
complete, concise statement of where the part is held and where it is loaded. The next
stage turns this into selectors, so name every region in words it can locate — and
nothing it cannot.

## A phrase that narrows a region is not a load direction

Wording that limits *where* a condition acts belongs to its region and stays attached
to it: "only the upward portion", "the part that faces up", "the outer half",
"everywhere except the rim". Never restate such wording as the direction or sense of
the load, and never drop it. Give a load a direction only where the user gave one for
the load itself; when none is given, give none.

`Fix the bottom. Apply traction to the segment that covers the cushion, but only the
upward portion` becomes `fix the bottom; traction on the upward-facing portion of the
cushion segment` — not `apply upward traction to the cushion segment`, which turns the
restriction into a direction and loads the whole segment.

## Bind every region to the segments it covers

A `segments` field, when present, lists what the surface's numbered segments were
found to be -- "0: front left leg; 5: backrest; 6: seat cushion". The stage after this
one selects regions by exactly those segments, so this is the main job here: for each
region the user named, say which listed segments it covers, **by their names**.

- A region that is one segment takes that segment's name: with a segment named "seat
  cushion", `load the seat` becomes `load the seat (the seat cushion segment)`.
- A region that spans several segments names all of them: a handle made of segments
  "top face of handle", "outer side face of handle" and "inner face of handle" becomes
  `the handle (its top face, outer side face and inner face)`. Leave none out.
- A segment id the user cited stays, followed by that segment's name: `segment 19`
  becomes `segment 19 (back face of cup wall)`.
- A region with no counterpart among the segments keeps the user's words, plus a
  location the next stage can resolve: the highest or lowest face, a side, a facing
  direction, inside or outside ("the rim" becomes "the rim, the highest upward-facing
  face at the open end").

Use the segments to locate the user's regions, never to choose new ones: do not add a
region because a segment exists, do not drop one because no segment matches, and do not
cite a segment id the user did not give -- ids change whenever the segmentation is
re-run, while names survive it. The `geometry` field says what the object is; use it to
understand words like "handle", "rim" or "seat".

## Complete what the user implied

- State each region's role plainly: held (fixed, all components unless the user limited
  them) or loaded (traction or pressure).
- A direction the user gave in words becomes explicit: "push down" is a downward (-y)
  traction, "pull up" upward (+y), "push in" toward the inside. Never invent a
  direction for a load given without one.
- "Fix" or "clamp" holds all components; "stands on the floor" holds the lowest face.
- One clause per region, supports first. Compact a long description by merging repeated
  requirements, but never drop a region, qualifier or number to save words.

An analysis needs at least one held region and one loaded region. Say both plainly when
the user gave both; when only one is given, keep it as written rather than inventing
the other.

Never add a region, magnitude, unit, direction, component subset, or segment ID the
user did not give. A load with no stated magnitude stays unquantified.

Example, with segments "0: top face of handle; 1: outer face of handle; 2: inner face of
handle; 3: outer cylindrical wall; 4: top rim face; 5: base":
`hold it by the handle and push down on the rim` becomes
`Fix all components of the handle (its top face, outer face and inner face); apply a
downward (-y) traction on the rim (the top rim face).`

## Keep, and never alter

- which regions are held and which are loaded, and how many of each
- each region's location and every phrase that narrows it — keep the user's own region
  names, cited segment IDs, and qualifiers
- load direction and sense, magnitudes, and units exactly as written
- fully fixed versus partly constrained, and which components stay free
- gravity or body force whenever mentioned

In this workflow upward is `+y` and downward is `-y`, and the object faces `+z`,
so forward is `+z` and backward is `-z`; use that phrasing only for a
direction the user already gave, and never convert units.

## Remove only

Conversational lead-in, restated geometry, material properties, solver or mesh
settings, and presentation requests. Preserve ambiguous wording rather than guessing,
and never ask a question.

Return only `refined_prompt`, with no prose, Markdown, or extra fields. Be complete
before brief: never drop a kept item or a covered segment to save words.
