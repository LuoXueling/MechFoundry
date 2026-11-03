# From the objective to a case (objectives §1–§2): the case brief

The user's words decide how much you decide. Decide the rest by reasoning about the real object
and how it is used, and never ask: the only question you may put to the user is one the
instruction told you to ask. Nothing here is looked up by the object's name.

## 1. How much did the user give?

| The user gave | You decide |
|---|---|
| a name only | everything: the use, the contacts, the loads and their sizes, the size, the material, the question |
| a name plus one thing (a load, a material, a size, a question) | the rest; keep what was given |
| a full case (supports, loads, material, size) | nothing; convert the units and run it |
| an image or a mesh, with or without words | as above, from what it shows; the description still says what the object is |
| "ask me if …" | ask exactly that, once, and decide everything else |

What the user gave comes out as given. A full case the workflow cannot run as stated (a contact,
a moving load, a pressure normal to a curved surface, plasticity) runs as the nearest static
elastic case, and the report says what was substituted.

## 2. The case brief

The brief is the reasoning, the plan its transcription. Mark each value given or decided. The
runner reads its JSON lines and its `Real …` lines, never its prose, and computes the numbers
itself: each action's effect at each member, the coverage of the states, the levers, the
reactions, the stress at the Real section and the prediction. Write no hand calculation.

### Use

One sentence: who uses the object, doing what. The section ends with one line:

`Use table: {"actions": {"A1": {"load": "<name>", "state": "working" | "handling" | "assembly" | "storage", "kind": "applied" | "weight" | "environment", "source": "H<k> top" | "<m> kg x g x <factor>" | "derived: <formula>" | "given" | "as A<k>", "size": <n>, "unit": "N" | "N m", "direction": [x, y, z], "enters": "<Load path member>", "held": ["<contact>", …]}}, "run": "<state>", "uncovered": {"<member>": {"limit": "<limit>", "why": "<…>"}}}`

- The table covers the use's working, handling and assembly states where they load the object
  (storage when something rests on it), one row per action, none omitted: a small one is a row
  with its small number. Every action of a state acts together; `run` names the state the solve
  applies.
- A row's source: a human action is its row of cae_practice.md (Human actions), `"H<k> top"` for
  full effort, `middle` for a light or precise use (`"H<k> top with factor 1.5"` for a row that
  takes a dynamic factor); a weight is `"<m> kg x g x <factor>"`. The runner computes both sizes:
  leave `size` and `unit` out. `"given"` only for a number the instruction states; else
  `"derived: <formula>"`, two or more numbers with units (`g` for gravity);
  the runner evaluates it: give `unit`, leave `size` out. `"as A1"` takes A1's size. An action of the
  surroundings that is no force (heat) is size 0.
- `direction` is the action's line in the use frame (a twist: its axis); `enters` the Load path
  member it acts on (one the body lacks is declared absent, and the governing load is then not
  applied as in use); `held` the contacts holding the object meanwhile (for the run state,
  contacts of the Contacts JSON). A hand that drives the action is its load, not a hold.
- The runner's effect of a row is its force times its lever (a twist as it is) at every Load path
  member between its entry and its holds: the regions' measured arm once they exist, before them
  the hold's distance along the Load path. A row's `"lever_m"` is rarely needed (`0`: at its
  hold; under the default on another state's row it needs `"lever_why"`). Where another state
  exceeds the run by more than a third at a member, run that state, or declare the member `uncovered` with its
  `limit`, one of: pressure inside a closed cavity, contact between separate bodies, large
  deflection, dynamic or impact, temperature (the Real section's member under contact only when a separate body
  touches it: a rest or hang holding its row, or a contact on it); with dynamic or contact, also
  `"no_approximation": "<why a dynamic factor or a reaction load does not carry it>"`: printed, and the
  verdict is "the governing load cannot be applied; a secondary case was run". Moving the Real
  section or declaring members the refusal did not name clears nothing.

### Contacts

Statics with prescribed supports (no lift-off or friction). One line per support: where it is,
which components it fixes (0/1/2 = x/y/z), and why. The section ends with one JSON object of the run state's contacts, which `plan.contacts` repeats:
`{"<contact>": {"transmits": "force and moment" | "force only", "kind": "grip" | "rest" | "hang" | "fastening" | "socket" | "worn", "over_m": <m>}}`.
A contact not named after the Load path member it holds adds `"member"`.

- A grip, socket or worn hold transmits force and moment, and `over_m` is the length it holds:
  its region spans at least half of it along the member, never one small flat face (an end face).
  A fastening takes a moment (against tipping) only as "force and moment" with an `over_m` its
  region spans the same way. A rest or a hang transmits force only.
- A rest is held on its normal component(s) (`dirichlet_values` components), or its reaction is
  applied as a load (`reactions_from_statics`, with a hold elsewhere); held on every component it
  is clamped, a marked simplification. A set of holds statics cannot split is
  reported, never refused. Holds that leave the body free to slide or spin get the
  fewest components that stop it, marked: a push along it of at most 20 % of the rests' normal
  force, a friction stand-in; rests not pressed, a larger push, or a pivot or spin no rest's
  friction resists, is refused (the refusal gives the exits). A rest's held faces are its
  support's lowest faces, where the body touches the ground; a missing or swallowed support is
  rejected through the refine ladder, never replaced by another member's faces.
- A rest holds against tipping only by weight: a run that tips the object takes the grip that
  steadies it in use, or side loads capped (the refusal gives each cap).
- The holds' reaction stays within their kind: a person's hold the Human actions top, a hang its
  weights × the dynamic factor, a rest compression only. An internal press or clamp force is an
  equal-and-opposite pair of rows on the two parts, never one external load.
- Contacts and `over_m` freeze when a body is made; a tipping, sliding or reaction refusal takes one contact
  of the kind it names (a grip beside a rest; never a rest or hang) without a new body, until the
  refusal clears or a solve is sent. Every Load path member, from each action's
  entry to its hold, is in the body or declared absent.
- A support resolving into several pieces (a contact named in the plural needs none) declares `"pieces": <n>` with `"pieces_why"` naming the
  separate parts of the real object that touch there (never what the surface or regions show; one
  patch split so is revised): the count the runner sees.

### Load

- The face where the force enters and its direction in the use frame (+y up); the runner turns
  the directions and the gravity into the body's frame. Several faces at once: one load region each. Self-weight always.
- The size from the mechanism that sets it: a mass times g, a pressure times an area, a code
  clause you quote. No remembered figure. A load built up slowly is not amplified; a shock takes a
  stated dynamic factor.
- A load no composition of tractions and body loads reproduces (a pressure normal to a curved
  face, contact, a moving load) is declared `governing_load_not_applied`, with the secondary case.

### Size and material

The real object at its real size (a model, toy or miniature only when the instruction asks for one);
a body more than 1.5 x under the Real size is taken as a simplified case, its extents still
compared with the Real ones at that scale.
Each on its own line, before any body (the runner checks the case's statics and plausibility first; the
`Real` lines freeze at the first body, but for one a later refusal names; routes.md §3.4 has the revision):
- `Real mass: <n> kg`; `Real thinnest member: <n> m`, the thinnest member the load passes through;
- `Real section (<Load path member>): <shape>`, the member the run loads most, as
  `rectangle <width> x <depth> m`, `round <d> m`, `tube <d> x <wall> m` or `box <width> x <depth> x <wall> m`
  (the depth across the axis it bends about); the runner computes I and A;
- `Governing material: <material>, E <n> Pa`, that member's own, not a skin or bulk around it;
- `Materials by part: <part> = <material>, E <n> Pa; …`, every visibly distinct part, named as one
  describes the body (one entry for one material);
- `Real extents: <a> x <b> x <c> m`, the real object's, in any order;
- `Load path: {"<member>": "series" | "parallel", …}`, every member each load passes from where it
  enters to where the object is held, joints and fittings included;
- `Up in use: the <longest|shortest> extent <vertical|horizontal>` when it stands one way.

A member under 2 leaves at the solved depth is warned as a resolution caveat, never refused. Values
come from materials.md. Parts whose E differs by more than 2 x take `"mode": "parts"`; uniform is for
one material, a shape optimization, or differing parts all under 1 leaf (a caveat). A uniform
material off the Governing E is refused.

### Prediction (optional)

The deformation mode and where the peak stress will be, in words. The runner's own stress and
deflection at the Real section are the prediction; write no numbers of your own. A mismatch
after the solve is a finding to explain.
