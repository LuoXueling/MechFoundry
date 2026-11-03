# Judging the stops against the brief (objectives §5)

At each stop the question is whether what the workflow made is the brief's case. Two kinds of
difference, kept apart in every report:
- a **caveat** leaves the result an answer to the brief's question (a thickening below the
  limit, a mesh-dependent peak, a dynamic stand-in);
- a **deviation** means what ran is not the brief's case: another object, a contact on another
  part or holding other components, a load on another face, direction or share, or a dimension
  the user gave that came out otherwise. Agreement with the prediction is neither.

The five verdicts are the only first lines `judge` accepts: **"the brief's case was run"**;
**"a variant of the brief's case was run"** (a "Brief changes" section, objectives/size.md §3.1);
**"a simplified case was run: …"**; **"the brief's case could not be run"**;
**"the governing load cannot be applied; a secondary case was run: …"**. A deviation is removed by
changing the words or the plan within the limits, never by accepting it.

"The brief's case was run" is accepted only when the runner sees no simplification: the body is
the real one, no contact is simplified, no member is absent, every member is covered, and the
materials are unchanged. Otherwise the verdict is "a simplified case was run: …"; the runner names
each item in the report's head.

- **geometry:** is it the real object, with a real one's proportions, standing as the use says,
  with the dimensions and features the brief must keep?
- **regions:** does each blue patch realise a contact of the brief, held in its components
  (`planned_supports`; a rest on its normal components or as its reaction), and the red the
  brief's load face, in the stated direction, with its share (`planned_load`, `loaded_fraction`)?
  Changing one region's words re-synthesises all of them: re-check every region. Never approve a
  region because no attempts are left.
- **solve:** compare the fields with the runner's prediction: the mode, the place
  of the peak, the numbers. The runner states the agreement: 0.5–2 agrees; 2–5 is printed "disagrees", run down
  (units, the load's size or face, a contact, the resolution) and explained; over 5 (or under a
  fifth) is a wrong case until shown otherwise: fixed and solved again, or the verdict line gives
  the ratio with "cause found: …" or "cause not found". A case past the linear range gives the
  runner's strength ratio and deflection over length in the verdict line. A
  support that moves a visible fraction of the peak did not hold: check its region before any
  setting; partly held, quote deflections relative to it. The workflow reports no support
  reactions (known_issues.md #9). The stress compared is `surface_von_mises.max_away_from_regions`
  at `max_away_at`; one that keeps rising with depth is a local peak, reported as mesh-dependent.
  `surface_max_principal` for brittle materials.
