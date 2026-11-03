# From the brief to the plan (objectives §3.3): writing a region

### 3.3 Writing a region

The region synthesizer turns a clause into one selector. It resolves a small vocabulary, and
any support or load is composed from it.

**The vocabulary.**
- *A band of height:* "the lowest N % of the height", "between A % and B % of the height", or
  in metres, "y below 0.2 m".
- *An end of the length:* "the outermost N % of the length along z at the −z end"; the axes are
  the workflow's (y up, z front, x to the right in the +z view).
- *A facing:* "faces whose normal points down (−y)", "upward-facing (normal within 45° of +y)";
  the cone is the tolerance you intend.
- *A part:* a segment name the stop's `segment_description` lists, or a plain part noun.
- *A box:* always with all three spans ("spanning the whole thickness in y").
- *An exclusion:* "not the <neighbouring part>".
- *A segment override:* "selected by position alone with no segment", for a part the
  segmentation did not separate, or when a part noun could match the wrong segment.

**The composition.** A region is *part ∩ position ∩ facing − exclusions*: "the <role>: the
<facing> faces of the <part> within the <position>, not the <other part>". Lead with the role
("Fix all components of …", "Apply a traction in −y on …"). Two of the three narrow most
regions enough; add the third when the picture shows a neighbour caught. A support's words
always carry its facing. A grip, socket or worn contact is the faces along its `over_m`
(objectives/brief.md §2, Contacts). A region that resolves to something other than the words is a
symptom with a fix in troubleshooting.md ("Region words"), never approved.

**Direction and sign.** The faces named are the ones the force enters through, so their facing
is *opposite* to the force: a −z force on faces whose normal points +z. A force total
`{"force":[…]}` is spread over the region's loaded area; a traction `[tx,ty,tz]` is per area. A
support is held in all components unless the brief released some (`dirichlet_values`).

**Composing a load.** A traction may point in any direction on any face, along the face as well
as into it (a shear load). From that:
- a **moment or torque** about an axis is a couple: two load regions on opposite sides of the
  axis, bound to one Use table row in N m whose `direction` is the axis; the runner computes the
  couple's forces;
- a **varying pressure** on a flat face is a few bands along the direction it varies, all bound to
  one row whose `"varies": {"along": [x, y, z], "from": <a>, "to": <b>}` states the profile (the
  relative intensity at each end); the runner computes each band's value (three or four bands
  reproduce a linear variation's total and moment to a few per cent); a single mean keeps the
  total and misstates the moment, said so;
- a load **shared by several faces** is one region per face with its share.

Built bodies have well-shaped faces of about half a leaf, so a band can be selected; when a band
selects a whole face of an uploaded body, its faces are too coarse.

**The preview's numbers.** `normal_coherence` is how single-facing the region is;
`loaded_fraction` how much of the force reaches the part along the direction given. Below the
coherence that applies (0.9, or the brief's derived value), look for a caught neighbour.
