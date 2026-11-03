> Independent re-extraction of every printed number from the archived raw outputs (`output/archives/bodyfitted_vs_nitsche.tar.zst`), 2026-10-17; the auditor's scripts and raw json are not in the repository.

# Audit of the SI body-fitted vs Nitsche tables (tab_si_bf_{dofs,accuracy,cost}_d{7,8}.tex)

Date: 2026-10-17. Repo: /home/xl2254/implicit_optimize (dev, 4ba8066). Auditor scripts are in this folder.

**Verdict: PASS.** Every printed number traces to raw run outputs. I found no hallucinated,
mistyped or mis-mapped value. Of 9,154 checks, 9,151 agree. The 3 that disagree are my own
independent recomputation landing on the other side of a rounding boundary. In those 3 cases the
table matches the stored record (details below). There are also three caption/definition points
for the authors (section 4). They are wording or interpretation issues, not data errors.

## 1. Method (independent of the generator)

I did not use `tables.py`, `collected.json`, `tables.xlsx` or `recomputed_metrics.json`.
The archive was extracted to `output/bodyfitted_vs_nitsche/` and deleted again afterwards.

- `audit.py` builds `raw_records.json` from per-entry raw files:
  - Nitsche: `attempts/nitsche_d<D>_<rung>.json` (every rung that ran), the run's `log.txt`
    (`inner N nodes`) and the VTU header (`NumberOfPoints`).
  - P1: `result_d<D>_<mode>.json`, `attempts/p1_*.json`, the P1 npz (`nodes` row count),
    `meshes/<tag>.json` (`tetgen_attempts`, `n_nodes`), `cpu_jobs/<tag>.log` (wall time of each
    CPU run, from `/usr/bin/time`) and `cpu_jobs/<tag>.lean.json`.
  - Metrics: `comparison.json`. Flags: `comparison.json["flags"]`.
  - Names: each session's own `prompt.txt` in `output/agentic_batch50_opus/<run>/` (read only).
- `my_metrics.py` is my own GPU re-evaluation of all **282 compared entries**, built from the
  stored P1 npz and the Nitsche VTU of the successful rung. It has its own VTU parser and
  octree-cell locator (per-level hashed lookup, nearest-centre fallback), trilinear Q1 by corner
  position, and the 4-point order-2 tet rule. It computes the weighted least-squares rigid fit,
  η_rig, η^H1_rig (gradient of r* = skew(θ)), the compliance ratio from f_ext, and s95/s99 from
  neo-Hookean Cauchy stress (P = μJ^{-2/3}(F − I1/3 F^{-T}) + κ lnJ F^{-T}, checked against
  `core/neoHookean.py`). This goes well beyond the 3-entry spot check that was asked for.
- `compare.py` parses the .tex cells (gray `\rowcolor`, "a (b)", C/R/†, "---") and compares
  each cell after rounding. Results are in `compare_results.json`.
- η^H1 is **not in `comparison.json`**: those files predate the H1 addition. The only other
  source is `recomputed_metrics.json`, which I was told not to use. So the H1 column is verified
  against my own recomputation only.

### Rounding rules I inferred (every cell is consistent with them)

| Quantity | Rule |
|---|---|
| η_rig, η^H1_rig (%) | 2 significant figures |
| ℓ_N/ℓ_P1, s95, s99 | 3 decimals |
| DOFs (thousands) | 3 significant figures |
| Times | 3 significant figures; values ≥ 1000 s as integers (e.g. 1991, 8765) |
| Retries, n | integers |

Medians are taken of the raw values (the mean of the two middle values when n is even), then rounded.

### Definitions I reconstructed, which reproduce every cell

- **Solver retries:** attempts that ran minus 1. For Nitsche, that is the rung records. For P1,
  it is the GPU rung records plus the CPU run plus the lean-GPU run. The CPU worker's internal
  modified-to-full Newton switch is not counted.
- **Mesh retries:** failed TetGen-stage calls plus the rejected remeshes (`remesh_retry`,
  `remesh_fallback_to_source`) of the surface the kept mesh was built from. A rejection is
  counted once (not again for its `selfint_check` entry).
- **Nitsche all-attempts:**
  - prep: Σ prep over rungs, with build_case counted once;
  - solve: Σ solve over rungs.
- **P1 all-attempts:**
  - prep: prep + seconds of the failed TetGen calls;
  - solve: solve + seconds of the failed GPU rungs, plus the CPU wall time when the entry ended
    on the lean GPU path.
- **Parentheses** are printed iff the all-attempts total exceeds the successful attempt by more
  than 0.5 %.
- **To 10⁻²:** I recomputed it from each solve's own `log.txt` Newton timestamps (GPU entries;
  identical to the recorded value in all cases). For CPU and lean entries I used the recorded value.
- **Markers:**
  - C: `solve_device = cpu`, or a Nitsche PARDISO rung;
  - R: `lean-gpu`;
  - †: a Nitsche rung with `looser = True`. The only one is the loaf of bread at d8 (rung
    `direct_continuation_session_tol`).
- **Gray row:** comparison.json η_rig > 0.10 for any mesh.
- **Medians:**
  - accuracy: excludes the † depth (d8 n = 46/17);
  - cost: excludes only the † Nitsche solve cell (n = 49/17). The loaf's Nitsche prep and
    to-10⁻² values and its P1 cost cells stay in;
  - unflagged = no T/G/L/S;
  - C and R times are included in the P1 time medians (n = 47).

## 2. Cells checked

These are the checks run by compare.py. A time cell is checked three times: value, parenthesis
and markers. Each accuracy metric is checked twice: against comparison.json and against my
recomputation. η^H1 is checked once, against my recomputation only.

| Table | Checks | Printed data cells (excl. names) | Mismatches |
|---|---|---|---|
| dofs_d7 | 318 | 266 (50×(flags+4) + medians/n 16) | 0 |
| dofs_d8 | 318 | 266 | 0 |
| accuracy_d7 | 1576 | 860 (50×16 + 60) + 50 gray states + names/order | 0 vs records; 2 vs my recompute (rounding) |
| accuracy_d8 | 1576 | 860 + 50 gray + † | 0 vs records; 1 vs my recompute (rounding) |
| cost_d7 | 2133 | 1076 (50×20 + 76) | 0 |
| cost_d8 | 2133 | 1076 | 0 |

Every table also had these checked: the row order (objects_50.txt order) and the 50 names
against each session's own prompt (all 50 map one-to-one), and the flags letters against the
flags booleans. The booleans are also consistent with the stated thresholds: p50 wall < 4h,
crevice share > 1 %, load gap < 3h, small support / partial clamp lists.

## 3. Mismatches (all from my independent recompute; the table matches the stored record)

| Table | Row | Column | Printed | Stored (comparison.json) | My recompute | Likely cause |
|---|---|---|---|---|---|---|
| accuracy_d7 | a USB stick | surface s99 | 0.847 | rounds to 0.847 | 0.84755 → 0.848 | rounding boundary; my weighted-percentile/locator differ by < 5e-4 relative |
| accuracy_d7 | a road roller … | surface s95 | 0.571 | rounds to 0.571 | 0.57047 → 0.570 | same |
| accuracy_d8 | a thatched stone cottage | dofs s95 | 0.984 | rounds to 0.984 | 0.98450 → 0.985 | same |

How close my recomputation is to `comparison.json` over all 282 entries (maximum relative difference):

| Metric | Max relative difference |
|---|---|
| η_rig | 2.6e-7 |
| compliance | 2.9e-8 |
| s95 | 3.1e-4 |
| s99 | 4.7e-4 |

The percentiles are the most sensitive quantity: points at cell faces or outside cells (≤ 0.12 %
of volume) are evaluated slightly differently. **Values I could not trace to raw data: none.**
η^H1 (range 0.13 %–99 %) could only be traced through my own recomputation. All 282 H1 cells
match it after rounding.

## 4. Points for the authors (captions and definitions, not data errors)

1. **The DOFs caption overstates the match.** It says "Matched DOFs: vertex count within 5 % of
   the octree's node count". Eleven P1 entries miss the 5 % target. Their printed DOF counts are
   real; the secant search kept its closest trial, or the remesh fell back to the source surface.
   The SI text says the search keeps "the closest trial", so a wording such as "targeting within
   5 %" would be accurate. Ratios (P1 vertices / Nitsche nodes):

   | Entry | Ratio |
   |---|---|
   | Leaning Tower d7 dofs | 2.12 |
   | Leaning Tower d7 graded | 1.47 (meshed at source resolution) |
   | Leaning Tower d8 graded | 0.81 |
   | Adjustable wrench d8 dofs | 0.58 |
   | School chair d7 dofs | 0.72 |
   | Glass-top table d8 dofs | 0.74 |
   | Car wheel d7 dofs | 0.905 |
   | Lighthouse d7 dofs | 0.913 |
   | Teddy bear d8 dofs | 0.932 |
   | Rhinoceros d8 dofs | 1.055 |
   | Yo-yo d7 dofs | 1.094 |

2. **The mesh-retry definition is ambiguous for three entries.** The table counts rejected
   remeshes only for the surface of the kept mesh. Rejections inside successful but not-kept
   size-search trials are not counted, which is consistent with "successful trials are not
   retries". Under a literal reading of "each rejected remesh is one meshing retry" these three
   cells would be larger:

   | Entry | Printed | Literal reading |
   |---|---|---|
   | Leaning Tower d7 matched DOFs | 2 | 12 |
   | Wooden toy train d8 matched DOFs | 0 | 3 |
   | Yo-yo d7 matched DOFs | 0 | 2 |

3. **One recorded field disagrees with the printed all-attempts prep.** The record field
   `prep_seconds_incl_failed_tetgen` also adds the self-intersection-check seconds of rejected
   remeshes (and of non-kept trials). The tables do not use it: they print prep + failed TetGen
   calls. This affects 6 entries: Leaning Tower d7 (all three meshes), natural stone arch d8
   dofs, wooden toy train d8 dofs and yo-yo d7 dofs. The parenthesis shows in only one of them,
   Leaning Tower d7 graded: 48.4 is printed, the record field gives 55.7. The table's choice
   avoids double counting the kept trial's check, which is already inside `surface_remesh`. Only
   the record field is inconsistent with the SI wording.

**Borderline gray rows (correct under "above 10 %").** The wooden rocking horse d7 graded
η_rig = 9.97 % prints as "10" in a non-gray row. The jackhammer d8 dofs 9.998 % prints as "10";
its row is gray because of its other meshes (11 %).

## 5. Other verified facts

- **Missing entries.** Exactly 3 objects have no P1 entries at either depth: the car brake
  disc, Himeji Castle and the park bench. The record's reason is "generated surface
  self-intersects (TetGen -d); not repaired". `tetgen_evidence.json` gives TetGen -d warnings /
  PyMeshLab faces of 98/106, 8/17 and 3/8, as the SI states. No other entry is missing or failed,
  and all 282 other entries are compared.
- **Nitsche DOFs.** 3 × the log's inner node count equals 3 × the VTU NumberOfPoints in every
  entry (100 of 100).
- **P1 DOFs.** 3 × npz node rows = record `p1_dofs` = 3 × `p1_vertices` = 3 × mesh `n_nodes`
  in every one of the 282 entries.
- **Recorded times are internally consistent.** Nitsche prep equals the sum of its prep phases
  and P1 prep equals the sum of its phases, in every entry.
- **Retries and markers.** Nitsche retries > 0 occur only at d8:
  - bar stool, jackhammer, park bench and school chair: 1 (pyamgx);
  - horse and pogo stick: 1 (jaxamg_4000);
  - loaf of bread: 6, which ended on the session-tolerance rung (†, C).

  The jackhammer d8 converging on pyamgx agrees with the SI and its notes.json. The C and R
  markers match `solve_device` everywhere: R on 4 entries (d8 surface of the hamburger, jelly,
  cake and loaf), with the failed CPU wall time inside the parentheses (e.g. hamburger
  338 + 1135 = 1473).
