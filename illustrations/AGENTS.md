# Illustrations — Agent Memo

Working area for all paper figures for **implicit_optimize**. Read §1 to pick a toolchain,
then the matching toolchain section. §6 (lessons) and §7 (checklist) apply to every figure.
The agentic-results figures (`agenticresults`, `agenticcompare`, per-category builds) are documented in [`README.md`](README.md). The main
figure's rows are `agentic_results/ar_paper_runs.json`; a row drawn smaller than the sizing
rule gives it (the control arm, 0.9×, shrunk about its centre; the car brake disc it replaced had 0.75×) says so with that entry's `draw_scale`, never by
changing the rule or pinning a camera (README §1, "Per-object size override"). A prompt
too long for the card is held to fewer lines in a smaller face by its entry's
`prompt_max_lines` (the control arm: 2 lines, 5.8 pt), never by growing every row (README §1).

**Archived inputs.** The run folders some builds read are packed in
`output/archives/<name>.tar.zst`: the three batches `agentic_batch50{,_haiku,_opus}`
(`agenticresults`, `agenticcompare`, per-category builds, `gallery`), `showcase`
(`showcase`) and `geometry_cleanup` (`shapeoptmerged` row c, only with `SOM_CU_RENDER=1`
or when its cached panels are absent). Extract each to its original path before such a
build: `zstd -dc output/archives/<name>.tar.zst | tar -x -C output`.

---

## 1. START HERE — pick the toolchain by figure type

| Figure type | Toolchain | Section | Exemplar |
|---|---|---|---|
| Schematic / method diagram (boxes, meshes, BCs, callouts) | **hand-authored SVG → Inkscape** | §3 | `gen_octree_nitsche.py` → `octree_nitsche.{svg,pdf,png}` |
| Data plot (convergence, benchmark, field slice as a 2-D chart) | **matplotlib → SVG → Inkscape** | §4 | — |
| Algorithm walk-through (multi-panel, real fields/meshes, 2-D profiles) | **matplotlib, data from faithful ports of the real functions** | §4 | `wf_bunny_field.py` + `wf_quadtree.py` + `wf_figure.py` → `octree_step1..4.{png,pdf}` (one figure per step) |
| Genuinely 3-D (volumetric field, mesh w/ transparency/lighting, cutaway) | **headless ParaView `pvbatch`** | §5 | `prep_bunny_sdf.py` + `render_bunny_sdf.py` → `bunny_sdf_pv.png` |
| Multi-panel 3-D (a pipeline shown on real geometry) | **pvbatch panels → matplotlib compose** | §5 + §5b | `bc_bunny_{geom,select}.py` + `render_bc_bunny.py` + `bc_bunny_figure.py` → `bc_region_id.{pdf,png}` |
| Scalar field + the mesh regions derived from it, one cut object | **pvbatch panels → matplotlib compose** | §5 + §5c | `dn_prep.py` + `render_dn.py` + `dn_figure.py` → `diffused_nitsche.{pdf,png}` |
| A per-facet field on ONE surface, one camera, one panel | **pvbatch panel → matplotlib crop** | §5 + §5b-ter | `./build.sh pfbunny` → `pf_prep.py` → `render_pf_bunny.py` → `pf_figure.py` |
| One iteration of a live optimiser, as a closed forward/backward LOOP | **pvbatch panels → matplotlib compose**, panels captured from a real run | §5 + §5f (§5g when a station carries two fields) | `./build.sh shapeopt` → `so_prep.py` → `render_so.py` → `so_figure.py` |
| A remeshing/geometry operator shown step by step on real CAD | **pvbatch panels → matplotlib compose**, pair built from an SDF twist | §5 + §5e | `./build.sh bczip` → `bz_prep.py` → `render_bz.py` → `bz_figure.py` |
| The branches of that operator the real CAD never reaches | same, but one **fabricated minimal CSG mesh per branch**, grouped before/after | §5 + §5e-ter | `./build.sh bczipx` → `bzx_prep.py` → `render_bzx.py` → `bzx_figure.py` |
| A **numerical study** (convergence, timings) whose meshes are also worth showing | **pvbatch panels → matplotlib compose, where the last panels ARE the plots** | §4 + §5 + §5h | `./build.sh nitscheconv` → `nc_prep.py` → `render_nc.py` → `nc_figure.py` |
| **The shape-optimisation figure**: one forward/backward loop in Figure 1's block language, an OPTIONAL branch dashed, the two table runs paired (b), the geometry clean-up's BC + change panels (c) | **pvbatch panels → matplotlib** | §5j (round 7) | `./build.sh shapeoptmerged` → [`som_cleanup_prep.py` → `render_som_cleanup.py`] → `som_figure.py` → `shape_opt_merged.{pdf,png}` |
| **The introductory challenge figure** (before Figure 1): one prompt → human engineer / AI agent / MechFoundry lanes × five steps, and MechFoundry's answers, flat computed pictograms | **CPU only: shapely/scipy prep → matplotlib grid** | §5k | `./build.sh challenge` → `ch_palette_check.py` → `ch_prep.py` → `ch_figure.py` |
| **The opening page**: the challenge figure stacked above Figure 1, one vector PDF | **plain pdfTeX stacks the two finished PDFs** (no rebuild of either) | §5k-bis | `./build.sh introoverview` → `io_compose.tex` → `intro_overview.{pdf,png}` |
| **Figure 1**: the whole workflow as one forward/backward loop on one object, generative inputs included | **opt-in cached generative assets + pvbatch panels → matplotlib block grid** | §5i | `./build.sh ovassets` (once) + `./build.sh overview` → `ov_bunny_loop.py` → `ov_surfaces.py` → `render_ov_panels.py` + `render_ov_solve.py` → `ov_figure.py` |

**matplotlib 3-D is the WRONG tool** for anything truly 3-D — no real z-buffer, so
intersecting surfaces composite wrong and transparency is fake. Use ParaView (§5).

---

## 2. UNIVERSAL RULES (every figure)

- **Environment** (verified 2026-09-08): Inkscape `1.1.2` (`/usr/bin/inkscape`, headless CLI);
  use conda env **`mechfoundry-warp117`** (`conda run -n mechfoundry-warp117 python …`) for figure
  computation.  It contains Warp 1.17.0, NumPy 1.26.4, JAX-AMG and the plotting/mesh packages;
  headless **ParaView 5.13.3 EGL** is provided by `pvbatch` on PATH.  The separate `ffmpeg`
  conda environment supplies ffmpeg/ffprobe for movie encoding.
- **Font is Arial, everywhere.** SVG: `font-family="Arial, Helvetica, sans-serif"`.
  matplotlib: `font.sans-serif=['Arial']`. ParaView: the built-in "Arial" is a look-alike —
  use the real TTF `/usr/share/fonts/truetype/msttcorefonts/Arial.ttf` (see §5).
- **The signed distance is italic *s* in every rendered label** (user, 2026-10-14; it was *d*
  until then, and the octree-step figures always said *s*).  matplotlib: `$s$`, `$s = 0$`,
  `$\mathrm{D}\mathcal{J}/\mathrm{D}s$`, "Octree + $s$", "Step on $s$" with Arial mathtext; SVG: an italic
  `<tspan font-style="italic">s</tspan>`; a ParaView colorbar title: `"… $s$"`, which VTK sends
  through pvbatch's bundled matplotlib mathtext, so set its rcParams to Arial first
  (`bunny/render_bunny_sdf.py`).  `d` survives only in code (dict keys `"d"`, `hp_panel_*_d.png`
  slot names), in derivatives, and as an octree depth (`2^d`, `rotated d=3`).
- **Displacement, objective and sensitivity notation** (user, 2026-10-29; the paper's):
  - the displacement is the vector **bold upright 𝐮** — mathtext `$\mathbf{u}$`, `$|\mathbf{u}|$`
    ("Nitsche solve, $\mathbf{u}$", "Displacement $|\mathbf{u}|$", "max $|\mathbf{u}|$"); SVG: an Arial
    Bold `<tspan font-weight="bold">u</tspan>` (the zero vector likewise bold, "(**u** = **0**)"),
    never the Unicode maths-bold letter, which Arial does not carry.  A scalar component stays
    italic non-bold (`$u_z$`).
  - the objective is a **calligraphic 𝒥**: `$\mathcal{J}$` ("Objective $\mathcal{J}$").
  - the sensitivities with respect to the design fields take an upright total **D**:
    `$\mathrm{D}\mathcal{J}/\mathrm{D}s$`, `$\mathrm{D}\mathcal{J}/\mathrm{D}\varphi$`,
    `$\mathrm{D}\mathcal{J}/\mathrm{D}$` + φ̄ (`\overline{\varphi}`), the key's generic
    `$|\mathrm{D}\mathcal{J}/\mathrm{D}{\cdot}|$`; the derivative with respect to the displacement
    keeps the **partial**: `$\partial\mathcal{J}/\partial\mathbf{u}$` (the adjoint right-hand side).
    `\varphi`, never `\phi`.
  - **`\mathcal` needs `"mathtext.cal": "cmsy10"`** next to the Arial `custom` fontset
    (`_shared/loop_chain.py`, `voidnuc/vn_figure.py`): the default `cursive` family resolves to
    **Comic Sans** on this machine; cmsy10 (bundled with matplotlib) holds TeX's calligraphic
    capitals at the ASCII letters, so `\mathcal{J}` prints the TeX 𝒥 and embeds as Type 42.
  - words without a symbol ("displacement", "objective", "Shape sensitivity" in Figure 1) and
    code/log text (`dJ/dd`, `|u|max` in prints, slot names `gu`) are left alone.
- **Deliverables per figure `NAME`:** an editable master (SVG for §3/§4, the `.py` for §5),
  plus `NAME.pdf` (vector, for LaTeX) and/or `NAME.png`. `build.sh` regenerates everything
  from scratch; the final `NAME.pdf`/`NAME.png` are the only figure files that live in this
  root (intermediates are rebuilt).
- **Folder layout (reorganized 2026-08-11 — one folder per figure).** Each figure's scripts
  + regenerable intermediates live in **its own subfolder**; the final PDFs/PNGs are moved back
  to this **root** (so the paper's `\includegraphics{illustrations/NAME.pdf}` paths are
  unchanged). Every `build.sh` target runs its steps inside the subfolder — `( cd <sub> && … )`
  — so the scripts' cwd-relative and `__file__`-relative paths resolve there, then `mv`s the
  deliverable(s) up to `$ROOT`. **When you add or edit a figure, put its files in the subfolder
  and run through `build.sh` (or `cd <sub>` yourself); do not run a prep/render/compose script
  from this root** — its bare filenames would miss the intermediates.

  | Subfolder | `build.sh` target | Figure (root deliverable) |
  |---|---|---|
  | `octree/`    | `octree`    | `octree_nitsche.{pdf,png}` |
  | `workflow/`  | `workflow`  | `octree_step1..4.{pdf,png}` |
  | `bunny/`     | `bunny`     | `bunny_sdf_pv.png` |
  | `sdfoctree/` | `sdfoctree` | `bunny_sdf_octree.{pdf,png}` (uses `../bunny/prep_bunny_sdf.py`) |
  | `bcregion/`  | `bcregion`  | `bc_region_id.{pdf,png}` |
  | `bcautobunny/` | `bcautobunny` | `automatic_bc_bunny.{pdf,png}` |
  | `pfbunny/`   | `pfbunny`   | `partfield_feature.{pdf,png}` + `partfield_segmentation.{pdf,png}` (uses `../bunny/prep_bunny_sdf.py`; reads Figure 1's PartField run in `overview/assets/seg`) |
  | `nitsche/`   | `nitsche`   | `diffused_nitsche.{pdf,png}` |
  | `voidnuc/`   | `voidnuc`   | `void_nucleation.{pdf,png}` |
  | `shapeopt/`  | `shapeopt`  | `shape_opt_loop.{pdf,png}` + `so_objective.{pdf,png}` |
  | `hiddenphi/` | `hiddenphi` | `hidden_phi_loop.{pdf,png}` + `hp_objective.{pdf,png}` |
  | `shapeopt/`  | `soframes`  | `shape_opt_evolution{,_slides}.{gif,mp4}` + frame directory (run with `SO_SAVE_EVERY=0`, §2) |
  | `hiddenphi/` | `hpframes`  | `hidden_phi_evolution{,_slides}.{gif,mp4}` + frame directory (run with `HP_SAVE_EVERY=0`, §2) |
  | `bczip/`     | `bczip`     | `bc_zip_workflow.{pdf,png}` (reads the Poly Haven table from `bczip/cad/`) |
  | `bczipx/`    | `bczipx`    | `bc_zip_special_cases.{pdf,png}` |
  | `nitscheconv/` | `nitscheconv` | `nitsche_convergence.{pdf,png}` (keeps the `nc_ref_*.npz` reference cache) |
  | `nitscheconv_uniform/` | `nitscheconvuniform` | `nitsche_convergence_uniform.{pdf,png}` (fixed-depth companion; shared source in `nitscheconv/`) |
  | `nitscheconv_uniform_sharp/` | `nitscheconvuniformsharp` | `nitsche_convergence_uniform_sharp.{pdf,png}` (fixed-depth, sharp-volume companion) |
  | `nitscheconv_sharp/` | `nitscheconvsharp` | `nitsche_convergence_sharp.{pdf,png}` (adaptive, sharp-volume companion) |
  | `challenge/` | `challenge` | `challenge_overview.{pdf,png}` (flat pictograms only) |
  | `intro_overview/` | `introoverview` | `intro_overview.{pdf,png}` (`challenge_overview.pdf` above `method_overview.pdf`; reads both root PDFs, rebuilds neither) |
  | `overview/`  | `overview` (+ opt-in `ovassets`) | `method_overview.{pdf,png}` (reads `overview/assets/`, `nitsche/dn_prep.py` via runpy, and `render_ov_solve.py`, which `nitsche` shares) |
  | `gallery/`   | `gallery`   | `agentic_gallery.{pdf,png}` (the 50 Opus objects in one amphitheater, seen from the stage; imports `agentic_results/`'s prep and style; §5m) |
  | `demo_video/` | `demovideo` | `demo_video.{mp4,png}` (27 s demo movie; imports `agentic_results/`'s prep, style and panel builders; reads the park bench's geometry clean-up; §5n) |

  Shared, NOT figure-local: `_shared/bc_bunny_geom.npz` (the GPU/TripoSG cache read by both
  `bcregion` and `nitsche`; the 3 scripts that touch it use `../_shared/…`) and
  `bunny/prep_bunny_sdf.py` (shared by `bunny` + `sdfoctree`). `_scratch/` holds an orphan
  SimJEB-bracket experiment (`so_fill_bolts.py` + `bracket_*`), not wired into `build.sh`.
  The `voidnuc`/`hiddenphi` preps anchor their `output_root` default to the repo-root
  `output/` via `__file__` (not a cwd-relative `../output`), so runs still land beside every
  other optimiser run regardless of the subfolder `build.sh` cd's into.
- **The user is on the CLI and CANNOT see inlined images.** Always write outputs to this
  folder and give the **file path**; never rely on showing a PNG in the transcript.
- **Verify by zooming, always.** After rendering, crop sub-regions with PIL
  (`Image.open(p).crop(box).save(...)`) and open each. Nearly every real defect this project
  hit (sideways arrowhead, leader dots inside a fill, colorbar overlap, wrong font) was
  invisible at full size and obvious at 4–8×. Check every arrow, label endpoint, and the
  colorbar region specifically.

### Rebuilding and timing discipline

- **Run one build job at a time on each GPU.** Timings in the convergence, optimization,
  animation, and overview figures are measurements; overlapping jobs make them meaningless.
  Check `nvidia-smi` before launch and while a long job is active.  On this machine use GPU 1
  for our builds (`CUDA_VISIBLE_DEVICES=1`) and match ParaView with `PV_GPU=1`; the specialized
  targets also accept `SO_GPU=1`, `HP_GPU=1`, `NC_GPU=1`, and `OV_GPU=1`.
- **Optimizer figure launches disable JAX device preallocation.** The explicit-surface remeshing
  path now defaults its physical SDF truncation width to `h_target`, so the extractor's band is
  `0.9 h_target`; this retains more near-interface cells than the former fixed `0.01` on the
  depth-7 table.  Without `XLA_PYTHON_CLIENT_PREALLOCATE=false`, JAX can reserve the remaining
  device pool before AMGX rebuilds its hierarchy on iteration 1 and a 48 GiB A6000 can fail even
  though the mesh itself fits.  `build.sh` sets the variable for `voidnuc`, `shapeopt`,
  `hiddenphi`, `soframes`, `hpframes`, `ovassets` and `overview`'s `ov_bunny_loop.py`;
  keep the original solver precision.  (The retired `ov_bench.py` launched `so_prep.py` as a
  subprocess, and that child also needed it, or it preallocated the pool before AMGX setup.)
- **(Retired with the first `method_overview`, kept for `ov_bench`-style benchmarks.)**
  **Overview provenance compares discretization, not cross-iteration node counts.** The overview's
  four-iteration benchmark records its initial outer mesh, while `hp_stats.txt` records the mesh
  at the nucleating iteration.  The `0.9 h_target` band makes their node counts differ by more
  than the old 5% heuristic even though they are the same problem.  `ov_bench.py` now checks the
  logged `h_target`, physical truncation, refinement band, depth, and `gd_lr`, and reports both
  iteration-specific mesh counts without asserting equality.  The benchmark reads those three
  lengths from its generated scratch `so_stats.txt` (the early initialization message is stdout,
  before the solver file logger exists), not by scraping terminal output.
- **The explicit-surface refinement width is mesh-scaled and physical.** For
  `remeshing_trimesh`, omitted `sdf_truncation` means `h_target`, and
  `hierarchical_extraction` therefore refines a `0.9 h_target` band in addition to forcing the
  finest cells that contain supplied surface vertices.  An explicit override is a physical
  length.  This policy does not apply to `remeshing_geo_func`: its `0.01` is the decoder's
  normalized truncation bound and remains unchanged.  The optimizer figure stats record both
  the effective truncation and band so captions do not infer them from defaults.
- **The optimizer captures use the same illustration-only Newton floor.** With the
  `0.9 h_target` extraction band, the depth-7 table's float64 residual can settle around
  `2.7e-7` after remeshing.  `SO_ABSOLUTE_TOL`, `VN_ABSOLUTE_TOL`, and `HP_ABSOLUTE_TOL`
  therefore default to `5e-7`; they remain independently overrideable for diagnostics and do
  not change production solver defaults.
- A normal `./build.sh shapeopt` or `hiddenphi` may only rerender cached optimization output.
  For **new timing**, move/remove `shapeopt/so_design.vtk` or `hiddenphi/hp_pre.vtk` first.
  Likewise, `soframes` and `hpframes` skip their frame-mode optimization while their
  `*_panel_frbody_*.png` cache exists.  Back up/remove those panels and `*_fr_times.txt` to
  force fresh timed frames.  Prefer moving old sentinels/panels into `/tmp` so recovery is easy.
- **A target whose deliverable is a TIME must run with per-iteration output OFF.**  The frames
  movies print the optimiser's cumulative wall clock on every frame, so `soframes`/`hpframes` are
  measurements, not just renders — but `so_prep`/`hp_prep` default to
  `output_per_optimize_step=1` and `build.sh` does NOT override it, so a bare `./build.sh
  soframes` silently times a run that is also writing two VTUs, a surface and a GLB every
  iteration.  The full invocations are

  ```bash
  SO_SAVE_EVERY=0 SO_GPU=1 PV_GPU=1 ./build.sh soframes
  HP_SAVE_EVERY=0 HP_GPU=1 PV_GPU=1 ./build.sh hpframes
  ```

  — the same switch the retired `ov_bench` used for the first overview's timings.  Measured 2026-09-21 on the
  shipped problem: 101 frames took **246.27 s** with saving on and **231.21 s** with it off,
  against the shipped **231.07 s**.  Forget it and the clock printed on every frame is 6.6 % too
  slow.  Verify afterwards that the run's `log.txt` holds no `Start saving outputs` line and its
  `vtu/` is empty — the run directory is NOT in one place: `so_prep` leaves `output_root`
  cwd-relative, so its runs land under `src/warp_wrapper/output/`, while `hp_prep` overrides it to
  the repo-root `output/`.  The loop targets (`shapeopt`, `hiddenphi`) quote no wall clock and are
  correct either way; saving there only costs time.
- `overview` no longer reads other figures' PDFs, so it need not run last.  It does depend on
  `nitsche/dn_prep.py` (runpy: the octree), on `overview/render_ov_solve.py` (shared with
  `nitsche`, whose defaults must stay panel (f)'s), and on `overview/assets/` (`./build.sh
  ovassets`, opt-in, never in `all`).  `pfbunny` reads the same assets (§5b-ter).
- Compute scripts that import `core` must receive `PYTHONPATH="$SRC"`; `build.sh` does this for
  shape/hidden-phi and both BC-zip targets.  Hidden-phi cutting uses trimesh/rtree and must receive
  the `mechfoundry-warp117` `LD_LIBRARY_PATH` plus `SPATIALINDEX_C_LIBRARY`; this applies to both
  `hp_cut.py` and `hp_fr_cut.py`.

---

## 3. TOOLCHAIN A — SVG + Inkscape (schematics)

### Export (Inkscape 1.1 uses modern `--export-*` flags, NOT old `-e/-A`)
```bash
inkscape NAME.svg --export-type=png --export-dpi=300 --export-area-drawing --export-filename=NAME.png
inkscape NAME.svg --export-type=pdf --export-text-to-path --export-area-drawing --export-filename=NAME.pdf
```
- `--export-area-drawing` (`-D`) = tight bbox; `--export-area-page` (`-C`, default) = viewBox.
- `--export-text-to-path` (`-T`): ALWAYS for the PDF (portable, no font deps); keep live text in the SVG.
- `--export-dpi=N` (raster) or `--export-width/height=PX`; `--export-background=WHITE --export-background-opacity=1` for opaque.

### Debug layout headlessly
- `inkscape f.svg --query-all` → `id,x,y,w,h` of every object; `--query-id=ID --query-x/-y/-width/-height` for one.
- `inkscape f.svg --actions='select-all;object-to-path;export-filename:o.svg;export-do'` (actions joined by `;`).

### House style
- **Canvas:** set `width`/`height` in px AND matching `viewBox="0 0 W H"`; design ~600–1200 px wide.
- **Palette (seaborn-deep, matches `../.assets`):** `#4C72B0` blue · `#DD8452` orange · `#55A868` green ·
  `#C44E52` red · `#8172B3` purple · `#8C8C8C` gray. Gray for scaffolding/mesh, saturated for the active object.
- **Strokes:** uniform weights (2 for boxes, 1 for mesh), round joins/caps. **IDs:** stable `id` on every element.
- **Arrows:** one shared `<marker>` in `<defs>`. GOTCHA: `orient="auto"` aligns the marker **+x** axis to the
  path, so the arrowhead must point along +x (`M0.5,0.5 L8.5,4.5 L0.5,8.5 Z`, `refX≈8.5`); a +y triangle comes
  out rotated 90°. Markers scale with `stroke-width` → on short legend lines a fat head swallows the shaft;
  use a slim stroke + keep legend glyphs horizontal.
- **Math:** Inkscape can't typeset LaTeX. Use Unicode symbols (θ ∇ Ω Γ ε ρ φ, `<tspan baseline-shift="super">`)
  for labels; for real equations render via matplotlib mathtext or typeset in the LaTeX caption.

---

## 4. TOOLCHAIN B — matplotlib (data plots)

Author in the `mechfoundry-warp117` env, `savefig('NAME.svg')`, then Inkscape → text-to-path PDF.
rc params: `font.family='sans-serif'`, `font.sans-serif=['Arial']`, `savefig.transparent=True`,
seaborn-deep palette. Save as SVG (vector); only raster when unavoidable.

### Algorithm walk-through figures (learned building `octree_step1..4`)
To *prove* the method (not hand-wave it), drive every panel with **real fields/meshes from
faithful ports of the actual pipeline functions**, reduced to a 2-D profile. For the octree
workflow: a real bunny-slice SDF (`wf_bunny_field.py`, voxel-fill + EDT of a mid-width slice),
a 2-D quadtree port of `hierarchical_extraction` (`wf_quadtree.py`: mixed-sign/band identify,
2:1 ripple balance), and scipy/skimage stand-ins for the filter / marching-cubes reinit.
- **Render the maintained BACKGROUND grid, not the ground truth.** The pipeline keeps queried
  node values and *interpolates the rest*; reconstruct that as per-leaf **bilinear** interp of
  the 4 queried corners (`background_field`). Then the SDF is blocky far from the surface and
  **sharpens as the mesh refines** — show that progression across depths (it's the whole point).
- **Decouple background-grid depth from mesh depth.** Index the fine background lattice by its
  OWN depth (`log2(N-1)`), never the quadtree's final depth, or you silently sample a sub-quadrant.
- **2:1 ripple is empty at the first refinement** (a 3→4 split is only a 1-level jump). Illustrate
  balancing at the transition where it actually fires (pick the max-ripple *non-final* depth).
- **Composition pattern:** `fig.add_gridspec` of 4 row-blocks; each block = a thin **title sub-row**
  (`height_ratios=[th,1]`, title `va='top'` so it clears the panel captions — the #1 overlap bug)
  + a panel row. **Uniform panel size across rows** via a fixed span with **margin columns**
  centering the shorter rows; arrows are their own turned-off axes (`FancyArrowPatch`). Embed the
  bold panel tag in the title: `set_title(f"$\\bf{{({tag})}}$  {txt}")`. A centered color legend
  goes in a strip just below the row (`get_position` → `fig.text`), not crammed in the title.
- **Nabla:** `∇` as a literal char is missing from Arial (renders as ▯). Use mathtext
  `$\nabla$` everywhere, including band titles. mathtext Arial: set `mathtext.fontset='custom'`,
  `mathtext.rm/it/bf='Arial[:italic|:bold]'`.
- SDF diverging map centered at 0 via `TwoSlopeNorm(vcenter=0,...)`; a `|∇SDF|` map centered at 1
  (`PRGn`, `vcenter=1`) reads "=1" as neutral white and deviations as color. A true distance field
  legitimately shows a purple medial-axis skeleton (|∇| undefined there) — that's correct, keep it.
  **Always give a diverging palette an explicit legend** (a bordered-swatch `legend_strip`, so the
  neutral white `=1` swatch is visible) — a reader can't infer purple<1 / green>1 otherwise.
- **Don't use a truncated SDF in the illustrations.** To show a non-eikonal "before" field, do NOT
  clamp/`tanh(s/T)` (that IS the truncated SDF, and this figure's source is a real mesh SDF). Use a
  non-truncated smooth distortion — multiply the true distance by a low-frequency `m(x)~O(1)`; it
  preserves the zero level set but makes `|∇|` deviate both ways.
- **Let the panels tell the story — minimal words.** Terse **Title-Case** step headings (no
  sentence-long parentheticals), **Sentence-case** 2–3-word panel captions, **plain arrows** (no
  verb labels — the panel sequence carries the process). Keep only essential math and the legends.
  Consistent capitalization — but **§5b supersedes this**: Sentence case for panel titles AND legend
  keys, with named primitives ("Flood Fill") keeping their capitals.
- **Compact for the article** (a multi-row method figure eats column height — minimize its
  height/width aspect): fold color legends into the **empty right side of the title band**
  (`title_legend`, square swatches in figure coords) instead of separate strips that cost row-gap
  height; put a needed key on the colorbar label (`"|∇SDF| (1: Eikonal)"`); tight `hspace`, small
  margins, and size the canvas so square panels nearly fill their rows (too short → panels shrink
  and leave horizontal gaps instead).
- **Don't stack the steps in one column** — that leaves big side margins on the short rows. Pack
  into a **rectangle**: the detailed step as a full-width **hero row**, the rest as a single
  **strip below** with per-step sub-titles (`strip_title` spanning its columns of a shared
  `subgridspec`), sized by panel count so both bands fill edge-to-edge. Step panel counts (5/3/3/2)
  don't tile evenly, so the strip panels come out smaller than the hero's — that's the deliberate
  "hero + supporting detail" look, and it drops the aspect to ~0.35 (very compact, vector so
  zoomable). NOTE: the four steps were later split into **one self-contained figure per step**
  (`octree_step1..4`, panels re-lettered a,b,… within each) — cleaner for the article and for
  iterating on a single step; each is built by its own `fig_stepN()` in `wf_figure.py`.
  - **No figure titles (user, 2026-10):** the "Step N · …" headings were dropped — the paper
    caption names the step. `make_fig` keeps a band above the panels ONLY when a legend needs
    it (step 1, `tin=0.28` so the legend clears the (f) caption); the others start at the
    panel captions. **Rendered text says italic `$s$`, never "SDF"** (`$|\nabla s|$`, "Noisy
    $s$"); step 4's scale factor is therefore `$\lambda$` (`$s \times \lambda$`), not `s`.
  - **Background: user settled on opaque WHITE** (`savefig(facecolor="white")`) — they tried
    transparent and reverted. (If ever transparent again: draw needed backgrounds as a **patch**,
    not `set_facecolor`, which transparent-save blanks.)
  - **Connect panels with the arrows**: give each arrow its own gridspec column, set `wspace=0`, and
    span the arrow `(0,0.5)→(1,0.5)` with `clip_on=False` so it touches both panels. For this to
    look connected the panels must FILL their cells — square limits (equal x/y range), and push
    labels/dimension-arrows inward (or left-align) so an aspect-equal panel isn't inset from the
    cell edge with a gap to the arrow.
  - **Highlight cells with a SHADOW** (semi-transparent fill) — saturated outlines/fills (magenta,
    cyan, …) print weird; a shaded overlay reads as "these cells," keeps the field visible, and is
    print-safe. Neutral gray shadow (~0.24) for identify/active cells and a DISTINCT tinted shadow
    (muted blue, ~0.38) for 2:1-balance cells; under the identify shadow, fade the field (~0.42) so
    the shade reads. Line/marker colors for the other steps stay OFF the blue↔orange map and
    distinct: magenta (spurious object), teal (extracted surface), blue (bbox), gray (masked region).
  - **One node symbol everywhere**: query nodes are black dots with a white edge, same size in every
    panel — don't recolor "new" nodes.
  - **Colorbar in its OWN gridspec column**, not `colorbar(ax=…, fraction=…)` — `fraction` steals
    width from the panel, so that panel ends up smaller than its siblings. Add a thin `cax` column
    (+ a tiny gap column) and `colorbar(cax=…)`; all panels stay equal size.
  - **To show a kept SDF band** (filtered field), dilate the body enough that the retained
    positive-SDF rim is visibly reddish before the gray mask — a 2-voxel band is too pale to read.

---

## 5. TOOLCHAIN C — headless ParaView (real 3-D)

### Two-process pattern (MANDATORY)
Importing `paraview.simple` into conda `mechfoundry-warp117` **segfaults** (libstdc++/libpython clash).
So: compute data in conda → write plain files → render in a **separate `pvbatch` process**.
Never co-host torch/conda and ParaView in one interpreter. (Memory: `headless-paraview-setup`.)

### GPU env vars (MANDATORY — else silent llvmpipe/software fallback)
```bash
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
export VTK_DEFAULT_EGL_DEVICE_INDEX=0
pvbatch render_xxx.py
```
Print `GetOpenGLInformation().GetRenderer()` in the script; confirm `NVIDIA RTX A6000`, never `llvmpipe`.

### File formats that work
- **Scalar volume (SDF)** → legacy BINARY VTK `STRUCTURED_POINTS` (`.vtk`), scalar written
  **x-fastest**: `arr[i,j,k]` (i↔x) → `arr.ravel(order="F").astype(">f4")` (big-endian).
- **Surface mesh** → **STL, not PLY.** A trimesh `.ply` loads as a bare point cloud (renders as dots); STL reads faces.

### SDF prep gotchas (conda side)
- Close holes with **pymeshlab** `meshing_close_holes`; `trimesh.fill_holes` and `voxel.fill()` **leak**
  through an open base (bunny) → garbage sign.
- Compute the SDF on the **padded cube directly**: pad solid occupancy into a cube-sized voxel grid, THEN
  `distance_transform_edt`. Do NOT extrapolate a bbox grid onto the cube (multilinear overshoot at corners).
- **Publication-quality surface:** Loop-subdivide + **Taubin-smooth** (`trimesh.smoothing.filter_taubin`,
  no shrinkage) to kill faceting before export.

### Pipeline cheatsheet (`paraview.simple`)
- `Clip` (ClipType `Plane`) halves a volume/mesh; `Invert` flips the kept half (verify by rendering). `Outline(Input=…)` = box edges.
- `ColorBy(disp, ('POINTS','sdf'))`. **`ColorBy(disp, None)` raises `invalid association 'NONE'`** — for a
  solid color skip it and set `disp.DiffuseColor` (a no-scalar STL defaults to solid).
- `disp.Representation = "Surface"`; `GenerateSurfaceNormals` before Clip for smooth (Gouraud) shading;
  `Opacity` for translucency (depth peeling automatic).
- Camera: set `CameraPosition/FocalPoint/ViewUp`, then `ResetCamera`.
- `SaveScreenshot("out.png", rv, ImageResolution=[W,H])`. PNG is the deliverable.

### Publication quality (learned refining `bunny_sdf_pv.png`)
- **Lighting:** white bg; `UseFXAA=1`, `UseToneMapping=1`; soft LightKit
  (`KeyLightAzimuth/Elevation/Intensity/Warmth`, `FillLightKFRatio`). **`UseSSAO` does NOT exist** in this
  build (guard every optional setattr in a try). PBR on the mesh (`Interpolation="PBR"`, `Roughness`).
- **OSPRay is WORSE for a translucent scientific cutaway.** Its physically-correct multi-layer transparency
  turns a translucent object into busy "x-ray glass" (all internal back-faces show) and shifts framing;
  **depth-peeled GL reads cleaner.** Tried `EnableRayTracing` raycaster/pathtracer, reverted.
- **Colorbar for a journal** (`GetScalarBar`): use the **real Arial TTF** (`TitleFontFamily/LabelFontFamily="File"`
  + `TitleFontFile/LabelFontFile=/usr/share/fonts/truetype/msttcorefonts/Arial.ttf`). Big fonts (title ~64,
  labels ~54 at ~2000 px height), thick bar (`ScalarBarThickness≈90`), moderate `ScalarBarLength≈0.5`.
  Replace the ~20 auto ticks with a clean handful (`UseCustomLabels=1`, `CustomLabels=[-0.2,0,0.2,0.4,0.6]`,
  `AddRangeLabels=0`, `LabelFormat="%.1f"`). **Capitalize the title** ("Signed distance  s").
- **No colorbar/cube overlap:** `ResetCamera` fills the frame edge-to-edge, so the object touches an overlaid
  colorbar. Widen `ViewSize` for margin AND **dolly the camera back** — move `CameraPosition` away from the
  focal point by ~1.3× — to shrink the object and open a clear right margin. Always crop-inspect the colorbar edge.

### SDF colormap convention (reuse it)
- **SDF data is never clamped — the color SCALE is.** A symmetric `RescaleTransferFunction(-Vin, Vin)`
  keeps white at `s=0` but saturates the whole far field (flat max-color past +Vin).
- **Fix = 0-centered diverging map with INDEPENDENT slopes** via explicit `lut.RGBPoints` keyed to data
  values (NOT `ApplyPreset`+`Rescale`, which pins neutral to the range midpoint). Read the true range:
  `sdf.GetDataInformation().GetPointDataInformation().GetArrayInformation("sdf").GetComponentRange(0)`.
- **Diverging rules (dataviz skill): two hues + a NEUTRAL-GRAY midpoint; CVD-safe blue↔orange poles.**
  Shipped map (`ColorSpace="Lab"`): inside `navy(0.106,0.196,0.478)→blue→soft-blue→gray(0.894 at 0)`,
  outside `gray→sand→orange(0.944,0.600,0.282)→red→brick(0.478,0.129,0.094)`; ramp fast off 0.
- User preference: **no isolines** on the bunny figure — the smooth field carries outside/inside/transition.

### Octree-mesh SDF cube (learned building `bunny_sdf_octree`)
A 2-panel companion to `bunny_sdf_pv`: (a) the uniform SDF cube, (b) the **real octree hex mesh with
nodal SDF**, both cut in half. Generate the octree from the explicit bunny with
`src.triposg_wrapper.trimesh_sdf.trimesh_to_octree` — **Warp GPU SDF query, NO TripoSG** (its
`__init__.py` is empty, so importing that submodule doesn't pull the model). Depth 3→6 = 45k nodes /
35k hexes in ~50 ms. Write it as a legacy VTK `UNSTRUCTURED_GRID` (hex `CELL_TYPES=12`, nodal scalar
`sdf`) — the `sdf2octree` corner order already matches VTK_HEXAHEDRON. Render with the SAME settings
as `bunny_sdf_pv` but **`Representation="Surface With Edges"`** for the cell frames. **Clip gotcha:**
`Y=0` is a cell boundary at every octree level, and clipping EXACTLY on it makes ParaView emit
degenerate slivers → the cut face shows spurious "X"/stub edges (looks like broken frames, worst in
the uniform-fine band). **Fix = nudge the plane a hair INTO the cells** (`Origin=[0, 0.009, 0]`), so it
cuts clean quad cross-sections; the cut stays flat and ~at Y=0. (Crinkle-clip also removes the
artifacts but leaves whole cells sticking out toward the camera — the nudge is cleaner.) Frames must be
thick (edge `LineWidth~3.5`, near-black) AND the bunny more transparent (`Opacity~0.32`) or they vanish
behind it. Render both panels with an **identical camera** (no per-panel colorbar); crop both to the
SAME union bbox to keep the cubes aligned; compose at **high dpi (~340) with `imshow(interpolation=
"lanczos")`** or the downscale anti-aliases the thin frames away in parts. One shared colorbar in
matplotlib from the LUT's RGBPoints (RGB interp ≈ the field's Lab). `build.sh sdfoctree`.
- **PDF needs `dpi=` too.** When the figure embeds raster panels (imshow'd renders), the `.pdf` is
  vector but the panels are rasterized at `savefig`'s dpi — omit it and they come out ~100 dpi (a
  low-res PDF next to a crisp PNG). Pass the SAME `dpi` to BOTH `savefig` calls (here 340).
- **Camera:** dolly back enough (`×1.16` after `ResetCamera`) that the whole cube — *incl. the
  bottom wireframe corners* — is inside the viewport; a tight dolly clips them, and the union-bbox
  crop can't recover geometry the render already cut. Compactness is fine: the crop trims the added
  white margin, not the cube.
- **Report the mesh stats on the figure.** `prep_octree_sdf.py` times the **pure generation**
  (warm-up call to JIT the Warp kernels, then a `torch.cuda.synchronize()`-fenced timed call — depth
  3→6 ≈ 47 ms) **at the render depth AND one level denser** (3→7 ≈ 214k/174k in ~65 ms) and writes
  both to `octree_stats.txt` (N/M/ms/gpu, then N2/M2/ms2). The compose step prints, as **three lines
  under panel (b)'s subtitle**: `"{N:,} nodes · {M:,} hexes"` / `"generated in {ms:.0f} ms on an
  {gpu}"` / `"(one level denser: {N2:,} nodes · {M2:,} hexes · {ms2:.0f} ms)"`.
- **Subtitles:** (a) "Underlying signed distance", (b) "Hierarchically extracted Octree mesh". (User
  spells it "Hierachical/Octree"; fix the misspelling for a paper title but keep their capital "Octree".)
  Place subtitle via `ax.text(..., 1.15, va="bottom")` and stats at `1.13, va="top"` (transAxes) so
  they stack cleanly just above the image — `set_title` can't mix the bold subtitle with smaller
  stats. Keep the y's close to 1.0 (not ~1.2) so the block sits near the cube, not floating above it;
  `bbox_inches="tight"` trims the freed top margin so the cube ends up larger.

### §5b Multi-panel 3-D: pvbatch panels → matplotlib compose (learned building `bc_region_id`)
Four stages, each its own script so the expensive one is cached: **geometry** (GPU, production
pipeline → `.npz`) → **selection** (production API → STL/VTK per panel + a `*_stats.txt`) →
**render** (pvbatch → one PNG per panel) → **compose** (matplotlib → PDF+PNG).
- **Drive the figure with the production API, not a re-implementation.** `bc_bunny_geom.py` runs
  `pipe.mesh2latent → latent2octree → extract_surface` (what the solver itself calls), and
  `bc_bunny_select.py` calls the real `SurfaceSelector.resolve()`. Every number in a caption is read
  from `bc_stats.txt`, so captions can't drift from the code.
- **Tune each feature separately — one global parameter usually can't serve two.** The bunny's two
  ears need different flood-fill gates AND different ring counts (near 5.0°/6 rings, far 4.0°/2):
  at 4.5° the far ear escapes its rim and floods the whole body (57 159 facets), while the near
  ear's cup is split by a deep crease that no *safe* gate crosses. Split the features with a
  `HalfSpaceSelector` derived from the seed components (midpoint + the line joining them, never a
  hard-coded plane), fill each side, clip back to its side, then union.
- **Ring growth bridges a crease; a wider gate does not.** Raising the angle hits a cliff (fill
  escapes the feature entirely); growth is bounded and closes the stripe. But watch `y_min` — rings
  also creep the region outward past the feature's base, so raise them only until the gap closes.
- **Beware a "coverage %" whose denominator you didn't verify.** Scoring the ear as *(facets in its
  half-space above the base)* capped at ~50% and looked like a hard limit — that denominator was
  mostly **head**: its dominant component sat 0.42 away from the ear tip. Check the reference
  region's connected components before trusting a ratio, or just render candidates and look.
- **Derive the camera from the data.** The area-weighted mean facet normal of the selection is the
  direction that shows it. The bunny's inner-ear facets average ~(0.39, 0.91, −0.12) — nearly straight
  up — so every side view showed zero selection. Sweep a handful of directions and contact-sheet them.
- **But a close-up panel keeps its parent panel's direction.** Pick the shared camera once (from the
  data, as above) and reuse it. Re-aiming the close-up down the selection's own normal shows the
  detail more head-on yet flips the features' apparent arrangement, and the reader can no longer
  match the close-up to the overview. Continuity beats a marginally better angle.
- **`Render(rv)` BEFORE setting the camera.** ParaView auto-resets the camera on the first render
  after a `Show`, silently clobbering a camera set beforehand. Symptom: one panel is framed for the
  whole scene while others obey your numbers.
- **Fitting distance:** a sphere of radius `r` needs `r / tan(FOV/2)` — `3.73 r` at ParaView's
  default 30° `CameraViewAngle`, **`4.51 r` at the 25°** this figure inherits from `bunny_sdf_pv`.
  Add ~5 % so wireframes that stick out past the surface still clear (`bc_region_id` uses 4.7 r).
- **Uniform panel scale:** panels sharing a camera must be cropped to a **common** ink box or the
  object changes size between them. Then letterbox everything to a **square** — matching a wide
  close-up's native aspect instead shrinks the whole-object panels.
- **Highlight overlays must be DISJOINT from the body mesh.** Drawing the whole surface and then
  the selected subset on top makes the selected facets coincident geometry; the z-fight speckles
  body colour through the highlight and looks exactly like "facets the selector missed". Export
  the **complement** (`~mask`) as the body layer. Two distinct bugs produce grey dots in a
  highlighted region — z-fighting (uniform fine speckle over the whole patch) and genuinely
  unselected facets (irregular blobs/fjords); count the mask before blaming the renderer.
- **Never let a faint context body meet the frame edge** — it reads as a grey rectangle / render bug.
  Export a *local* context mesh instead of the whole object. Build it as a **collar** —
  `GrowSelectionSelector(sel, N) - sel` (N=30 here) — not as "everything in the selection's bbox":
  the bbox version drags in whatever else shares that box (a strip of head/neck) which hangs out of
  frame as a grey tail with a black back-facing edge.
- **Fade the collar out; never hard-clip it.** Any clipped mesh ends on a ragged facet boundary that
  reads as a zigzag tear. Tag each collar facet with its **ring distance** from the selection
  (`_bfs_expand` per ring), write it as a legacy VTK `POLYDATA` + `CELL_DATA` scalar, then in
  ParaView `ColorBy(disp, ("CELLS", "fade"))` with an RGB transfer function running body-colour →
  white *and* `EnableOpacityMapping=1` with an opacity function falling to 0.
  Two things this gets wrong if you are careless:
  - **Keep the inner half of the collar FULLY opaque** (here ~15 of 30 rings, opacity 1 up to
    fade 0.5, gone by 0.85). Fading from the patch outward leaves nothing occluding the *far* side
    of the selection, and the back of the far feature shows through the body as a stray coloured
    patch with no boxes on it — a dead giveaway that the mask is too transparent.
  - **Light the collar ambient-heavy, but do NOT flatten it** — `Ambient 0.65`, `Diffuse 0.45`,
    `Specular 0`. At the panels' usual 0.12/1.0 the part wrapping into a hollow goes dark enough to
    read as dirt on the figure; at a fully unshaded 1.0/0.0 the smudge is gone but so is every trace
    of form, and the mask looks like a flat cut-out pasted over the render. The middle keeps the
    shading that says "surface" while holding the shadowed side pale.
- **Restrict a close-up's data with the feature's own resolved selector**, not with its bounding
  box. Filtering combined results by the bbox pulls in the *other* feature's share and paints a
  detached cloud beside the close-up.
- `GetOpenGLInformation()` is a **module-level** call (`from paraview.simple import *`), not a method
  on the view proxy. `PolyDataNormals` is deprecated in 5.13 → `SurfaceNormals` (still works, warns).
- Points: `Representation="Points"` + `RenderPointsAsSpheres=1`; box edges: write one legacy ASCII
  VTK `POLYDATA` with `LINES` (12 per box) and `RenderLinesAsTubes=1` — far cheaper than N `Box` sources.
- **Flat-shade wireframes and point glyphs** (`Ambient 1`, `Diffuse 0`, `Specular 0`). Under the
  default diffuse lighting, thin strokes and small glyphs render dark and desaturated — the box
  edges washed out to grey-blue and red node dots came out navy. Then `LineWidth ≈ 2.6` +
  `Opacity 0.8` + `RenderLinesAsTubes` keeps ~350 boxes individually legible at print size;
  thin *shaded* strokes just vanish. Don't solve density by zooming into a fragment — an earlier
  panel already shows the whole region, so the close-up must still read as *that* region.
- **Cut layers that don't carry the message.** A dense node-marker layer inside a box lattice
  fights it for attention and neither reads; dropping the markers (and their legend key) made
  the boxes land. Keep the count in the caption instead of drawing every point.
- **Reuse `render_bunny_sdf.py`'s material verbatim** for a matching look: warm off-white
  `[0.92, 0.90, 0.86]`, `Specular 0.35`, `SpecularPower 30`, `Ambient 0.12`, PBR `Roughness 0.45`,
  `FeatureAngle 50` normals, `CameraViewAngle 25`, key light az −35 / el 45 / int 0.85 / warmth 0.55,
  and `SaveScreenshot(..., OverrideColorPalette="WhiteBackground")`. **`UseSSAO` really is absent**
  in this build (5.13.3) — keep it in the guarded loop, it just prints "(skip)".
- `paraview.simple._DisableFirstRenderCameraReset()` is the clean fix for the auto-reset trap above.
- **One single-line title per panel, above it** — `set_title(f"$\\bf{{({tag})}}$  {title}",
  fontsize=8.2, pad=6)`, with the descriptive phrase folded INTO the title
  ("Seed selection: top 5% of height"), not set as a second line under it. Below-panel captions and
  under-title subtitles both cost row height for text a LaTeX caption can carry: dropping them took
  this figure from 3.69 in to 2.74 in tall at the same panel size. 8.2 pt keeps the longest title
  inside its column; at 9 pt it overruns the gutter. Derive the panel height from the width the
  gridspec actually gives a column (`(right-left)*W/(ncol + (ncol-1)*wspace)`) or square panels get
  letterboxed inside a taller box. Measure the title band's ink extent afterwards — a long title
  clips silently.
- **Scale an inter-panel arrow by growing the SHAFT, not the head.** The §3 marker gotcha applies to
  `FancyArrowPatch` too: at `mutation_scale=26` a custom `head_length=0.55` consumed the entire
  0.034-fig-unit shaft and the arrow rendered as a bare triangle. Keep the default `-|>` proportions,
  set `mutation_scale≈20` + `lw≈2.6`, and lengthen the endpoints to ±0.016 fig units — then widen
  `wspace` (0.10 → 0.14) so the longer arrow still clears both panels.
- **Sentence case for panel titles and legend keys**, NOT Title Case — "Bounding boxes covering all
  facets", not "Bounding Boxes Covering All Facets". The exception is a **named primitive or
  operator**, which keeps its capitals because it names the thing, not the step:
  "Flood Fill & Grow by Rings". (This supersedes the older §4 "headings Title Case" rule.)
- **Arial has no thin space (U+2009)** — matplotlib renders it as a missing-glyph box. Use a plain
  space as the thousands separator.

#### The box cover this figure used to draw is gone (`core/bc_surface_selectors.py`)
Panel (c) was "Bounding boxes covering all facets": the per-component box lattice
(`max_bbox_size`) that `SurfaceSelector.resolve()` built as a pre-filter for the BVH.  It was removed
from production (2026-09-30).  A selector now keeps one box around its whole selected patch as a
far-point cull, and the Warp BVH distance test (≤ `eps_expansion`) decides.  The accepted points
are identical, and the query is up to 46× faster (project memory `bc-bbox-subdivision`).  Panel (c)
is therefore a close-up of the selected facets, with no box overlay and no "Bounding boxes" key.
The paper text's "a series of small bounding boxes" no longer describes the code.

#### Shipped settings for `bc_region_id` (read before re-tuning)
| Knob | Value | Where |
|---|---|---|
| seed | `HighestSelector('y', 0.05)` → 2 components | `bc_bunny_select.py` |
| near-ear gate / rings | **5.0° / 6** | `FRONT_ANGLE`, `FRONT_RINGS` |
| far-ear gate / rings | **4.0° / 2** (4.5° floods the body) | `BACK_ANGLE`, `BACK_RINGS` |
| collar | 30 rings, opaque to fade 0.5, gone by 0.85 | `COLLAR_RINGS`, `OPAQUE_TO/FADE_END` |
| collar material | `Ambient 0.65 / Diffuse 0.45 / Specular 0` | `render_bc_bunny.py` |
| camera | `(0.10, 0.62, 0.78)`, `CameraViewAngle 25`, all three panels | `VIEW_DIR` |
| figure | 7.16 × 2.69 in, square panels, `wspace 0.14` | `bc_bunny_figure.py` |

`build.sh bcregion` runs all four stages; `_shared/bc_bunny_geom.npz` is cached (GPU + TripoSG,
minutes) and gitignored along with every `bc_*.{stl,vtk}` / `bc_panel_*.png` intermediate. **The
select step must be re-run before any renderer experiment** — `build.sh` deletes those intermediates
when it finishes.

### §5b-bis Automatic shape-diameter BC selection (learned building `automatic_bc_bunny`)

This is a **new figure**, not a replacement for `bc_region_id`: keep both root deliverables.  It
shows the production `ShapeDiameterSegmentSelector` workflow in four panels: **(a)** top-5%-height
seed, **(b)** shape-diameter values, **(c)** connected surface segments, and **(d)** a close-up of
the selected facets.  There is deliberately no panel (e).  Run it with `./build.sh bcautobunny`; the
pipeline is `bunny/prep_bunny_sdf.py` → `bcautobunny/auto_bunny_prep.py` →
`render_auto_bunny.py` → `auto_bunny_figure.py`, producing
`automatic_bc_bunny.{pdf,png}` in this root.

- **Use the same surface source and appearance as the established bunny BC figure.**  Selection
  runs on the Loop-subdivided, Taubin-smoothed `bunny.stl` emitted by `prep_bunny_sdf.py`, mirrored
  and winding-repaired exactly as in the bunny renders.  Shape-diameter rays intersect this surface
  through the Warp mesh BVH.  Do not rebuild a marching-cubes surface from `bunny_sdf.vtk`, and do
  not reinitialize an SDF for the selector.  The retained signed-distance grid and the shape
  diameter are different quantities.
- **The selector call is the source of truth.**  `auto_bunny_prep.py` calls
  `ShapeDiameterSegmentSelector.resolve()` and reads its selected-face mask and segmentation.
  Do not reconstruct segments or adjacency expansion in illustration code.
  Core segmentation and selection are GPU Warp/highly parallel Torch operations.
- **Shipped selector settings:** `HighestSelector("z", fraction=0.05)`, 25 rays, `k=5`,
  smoothing `0.26`, maximum segment-area fraction `0.12`, and exactly one spatially downward
  adjacent segment per disconnected ear seed (`adjacent_segments=1`,
  `adjacent_direction=(0,0,-1)`).  Adjacency is between **connected segment IDs**, not merely equal
  cluster colours: one colour can occur on disconnected ear and head components, and colour alone
  caused the earlier head leak.
- **No facet/ring expansion or flood fill belongs to the automatic result.**  The one-adjacent-
  segment option remains part of the production selector, but there is no post-selection adjacent
  facet growth.  The only 30-ring operation is a render-only fading collar in panel (d); it never
  changes the selected mask.
- **Panel (c) is segmentation, not selection.**  Use the discrete nature-journal-style palette in
  `render_auto_bunny.py` and color the connected `segment_id` field.  Shape diameter in panel (b)
  is continuous and linearly normalized after CGAL-style missing-value repair and bilateral
  postprocessing; do not apply a categorical map to it.
- **Panel (d) must match `bc_region_id` panel (c)'s visual grammar:** same camera direction
  `(1.5,-3,1.25)`, 25° view angle, selected surface and ambient-heavy fading collar.  It is a
  close-up of the resolved region, not a whole-bunny fifth panel.  It used to carry the blue edges
  of the selector's small-box cover; that cover was removed from production (see §5b), and so were
  the edges.
- **The old method is comparison-only.**  The prep computes a legacy two-ear flood-fill/growth mask
  after the automatic result solely to report overlap; it is never passed into the automatic
  selector or shown as another workflow panel.  Its encoded settings are near ear 4.6°/14 rings and
  far ear 3.4°/14 rings, so do not call it parameter-identical to the current §5b shipped table.
  Current recorded comparison: 26,024 legacy facets versus 24,710 automatic facets, IoU 0.944,
  recall 0.947, precision 0.997.  `auto_bunny_prep.py` asserts both ear seeds, prevents a body leak,
  and requires minimum overlap.
- **Regression data:** `auto_stats.txt` is the caption/count record.  For code changes, set
  `BC_SAVE_BASELINE=/tmp/name.npz` when running the prep and compare `cluster_ids`, `segment_ids`,
  and `selected` array-by-array.  GPU BVH traversal can move shape-diameter floats by about
  `6e-7` between runs; this is acceptable only when every discrete downstream array is identical.
- **Performance record (RTX A6000, 161,568 facets):** the staged benchmark lives in
  `bc_selector_benchmark_relabel256.json`.  Capturing colored Warp discharge rounds and doing the
  costly full residual-graph global relabel every 256 launches reduced alpha expansion from
  23.72 s to 2.89 s without changing the cut; total staged selector time is 3.08 s.  Preserve the
  max-flow/min-cut duality check and convergence check when optimizing further.  Rejected frontier,
  lock-free atomic, and unary-dominance trials were slower and were removed—do not restore them
  without a production-mesh benchmark.
- `build.sh bcautobunny` deletes the STL/VTK/panel intermediates after composition but retains the
  scripts, `auto_stats.txt`, benchmark JSON, and final root PDF/PNG.  As with §5b, rerun prep before
  invoking the renderer directly.

### §5b-ter PartField instead of shape diameter (learned building the two `partfield_*` figures)

Two **single-panel** deliverables, `partfield_feature.{pdf,png}` and
`partfield_segmentation.{pdf,png}`, standing beside §5b-bis panels (b) and (c): the same bunny,
the same camera, the same material, but the *learned* backend the FEM workflow and the UI offer
as an alternative to shape diameter (`segmentation_method: "partfield"`).  `./build.sh pfbunny`;
the pipeline is `bunny/prep_bunny_sdf.py` → `pfbunny/pf_surface.py` → `pf_prep.py` →
`pf_palette_check.py` → `render_pf_bunny.py` → `pf_figure.py`.  **Not part of `all`** — it reads
the overview's opt-in generative assets.

- **Since 2026-09-29 both figures show Figure 1's PartField run, not one of their own.**  The user
  asked for `partfield_segmentation` to be what the overview's segmentation panel is, and
  PartField is not seeded (below), so a run of its own gives other parts.  `pf_prep.py` reads
  `overview/assets/seg/pf_cache/features.npy` and `pf_k8.npz` — written by the production
  `segment.py` in `./build.sh ovassets segments`, sha-checked against `assets/MANIFEST.json` — and
  asserts the run's surface (`assets/bunny_display.ply`) is `pf_surface` facet for facet (same
  vertices and order; only the winding differs, which labels do not see).  k = 8 is the
  overview's: the smallest k that gives each ear its own part.  The parts are coloured by cluster,
  in the overview's pastel tints, by the overview's own `ov_palette_check.segment_tints` (every
  assignment tried; touching parts get the most different pair, min normal ΔE 16.9), so every
  part has its Figure 1 colour.  The feature panel moved with it (one run feeds both), and its
  PCA hues changed: a different run's leading components point elsewhere.  The bullets below that
  quote k = 5 numbers, height-ordered colours and the saturated palette are the history of the
  first version.

- **Drive it with the production entry point, in the production environment** (now in
  `ov_assets.py segments`, which runs it with these flags).  The target invoked
  `src/partfield_wrapper/segment.py` through the interpreter `partfield_status()` resolves, with
  the same flags `fem_workflow/monitor.py` builds.  No illustration code runs the network, picks a
  clustering backend or re-derives labels.  (`--cluster-interpreter` is passed for that fidelity
  but is never exercised here: `gpu_agglomerative_available()` is true inside the `mechfoundry-partfield` env —
  cuml is installed there and not in `mechfoundry-warp117` — so clustering stays in process.)
- **PartField keys its feature cache on a FILE, not on arrays.**  `bcautobunny` mirrors and
  winding-repairs the bunny in memory; that is not an option here, because
  `PartFieldRuntime.features()` hashes the surface path.  `pf_surface.py` therefore writes the
  mirrored surface to disk once and everything downstream reads that same file.
- **STL is a triangle soup — it has no adjacency.**  Reading `pf_surface.stl` back with
  `process=False` (which is what PartField itself does, so the facet order is preserved) gives
  three private vertices per facet.  Feed that to `_build_adj_pairs` and *every facet is its own
  connected component*: the first run reported "5 clusters → 161 568 connected
  segments" instead of the couple of dozen there are.  `pf_surface.py` writes the welded
  vertices/faces beside the STL as `pf_topology.npz`;
  `Trimesh.export` emits triangles in `mesh.faces` order, so facet *i* matches, and the script
  asserts that (max centroid drift, currently exactly 0) rather than trusting it.
- **PartField stores CLUSTER labels; shape diameter stores CONNECTED segments.**  For
  `shape_diameter`, `segment_surface` already returns `segment_ids` from
  `_connected_segment_ids`.  For `partfield`, `gpu_cluster.run` hands `write_segmentation` the raw
  cluster labels under the same key.  The analogue of §5b-bis panel (c) therefore has to run
  `_connected_segment_ids(cluster_ids, adj_pairs)` itself — as the selector does, and for the
  reason §5b-bis gives: one cluster colour can sit on parts that do not touch.
  **Its third positional argument is `shape_diameter`, not a segment count**; passing a count
  fails with `'int' object has no attribute 'dtype'`.
- **The measured split:** 5 clusters → **19** connected segments (`pf_stats.txt`), but the five
  largest carry 0.3211 / 0.2657 / 0.1775 / 0.1730 / 0.0624 of the area — 99.97 % between them.
  The leftovers are 5–10 facets each; they are visible only past ~4× zoom and are *real output*,
  not render dirt (§5b's rule: count before blaming the renderer).  PartField also puts the two
  ears in different parts, which shape diameter does not.  That is the backend's answer, not a
  defect.
- **The inference is NOT deterministic, so do not quote a number you did not just read.**
  `PartFieldRuntime._loader` draws 100 000 surface samples with an unseeded
  `trimesh.sample.sample_surface`, so two cold runs give different features: across two here the
  part areas moved by up to 1.5 points, the explained variance by 0.004, and the connected-segment
  count went 14 → 19.  Every discrete conclusion held (5 parts, same parts, same 99.97 %), but the
  exact figures come from `pf_stats.txt` of the run that produced the shipped PNG.  A warm
  `pf_cache/` pins all of it; only a cache miss re-rolls.
- **Colour the segments by HEIGHT, not by area.**  Colour was first handed out by descending area,
  and two of the five parts sit 0.5 points apart — inside the run-to-run wobble above, so a rebuild
  could silently swap two parts' colours.  `pf_prep.py` orders the identity-carrying segments by
  area-weighted centroid height instead (ears, head, body, base on this object) and gives the rest
  the colours after them, biggest first.
- **A 448-channel feature has no colour map — project it, and project it into Lab.**  Three leading
  principal components of the L2-normalised features (the same normalisation `gpu_cluster` clusters
  under), percentile-clipped at 1 %, explaining 0.255 / 0.131 / 0.076.  Then:
  - **Straight to R/G/B is unusable.**  The dominant component's low end goes to near-black, the
    body becomes a black blob, and with no luminance left there is no shading — the bunny stops
    reading as a surface at all.
  - **Leading component → L\* is barely better.**  It ramps through grey: wherever it saturates,
    a\* and b\* are both ~0, so the head and ears came out near-white with no hue, and the
    silhouette dissolved into the white page.
  - **Shipped: the two leading components drive a\*/b\*, the third a narrow L\*.**
    `L* ∈ [52, 70]`, `a*, b* ∈ ±42`, converted to sRGB by hand (D65).  Hue then carries the
    variance that matters, every facet keeps the same mid lightness so the key light still models
    the form, and nothing clips to the page or to black.
- **Direct colours need `COLOR_SCALARS` + `MapScalars = 0`.**  ParaView paints an array as colour
  only when it is *unsigned char* with 3–4 components; a 3-component float array is pushed through
  a transfer function whatever you do.  Legacy VTK's `COLOR_SCALARS <name> 3` block (floats in
  [0, 1]) is read into a `vtkUnsignedCharArray`, so `ColorBy(disp, ("CELLS", "feature_rgb"))` +
  `disp.MapScalars = 0` paints the projection with no LUT in between — which is the point: no
  colour-map choice can bias what the feature field appears to do.
- **Now: the overview's pastel set, gated on TOUCHING parts.**  `pf_palette_check.py` fails a
  touching pair under normal ΔE 15 and reports the rest: 4 of the 12 touching pairs sit under CVD
  6 (worst 2.8), and no assignment of the eight tints lifts them (the best of all 8! reaches 2.8)
  — a limit of the pastel set the user chose for Figure 1, stated in its caption, not a bug here.
- **(history) The inherited segment palette FAILED the §5c gate on these parts — gate it, don't assume it.**
  `render_auto_bunny.py` panel (c) strides labels through `(id * 7) % 17 / 16` into a 10-entry step
  LUT.  On these five segments that lands an orange and a green **5.7 CVD ΔE** apart, under the
  hard floor of 6: exactly §5c's red/green collapse, inherited unnoticed because the stride picks
  different entries for different label sets.  Shipped instead: six explicit colours in
  `pf_prep.py` (`#dba854 #9c4331 #3a8f83 #2e4f6b #56b4e9 #d55e00`), assigned by the height order
  above, all-pairs **worst CVD 15.0 / worst normal 25.9** including the page white.
  `pf_palette_check.py` is the gate — it imports `../overview/ov_palette_check.py` so there is one
  implementation, and `build.sh` runs it **before** the renderer, so a recoloured figure cannot
  ship unvalidated.
- **Working a step LUT out on paper got three colours of five wrong.**  A `ColorSpace="Step"`
  transfer function that is then `Discretize`d into a different number of bins does not hand back
  the RGB point the value sits on.  Asking the transfer function
  (`lut.SMProxy.GetClientSideObject().GetColor(v, rgb)` — the output array is a required second
  argument) gave colours that did not match the list at all.  That readback is gone now: the
  segments are coloured as finished RGB in `pf_prep.py`, so the palette, the gate and the render
  all read one array and no transfer function can recolour them behind the gate's back.
- **`CellDatatoPointData` will not take an unsigned-char colour array.**  §5b-bis's panel (b) runs
  its shape diameter through it for a visually continuous surface; do the same to `feature_rgb` and
  the chained `GenerateSurfaceNormals` fails outright with *"Could not create a representation
  object"*.  Both panels here stay per-facet, which is what the quantity is, and at print size the
  facets are not resolvable anyway.
- **One camera, one crop.**  Both panels come off the identical camera, so `pf_figure.py` takes the
  **union** ink box over both, squares it, and crops both to it (§5b's uniform-panel-scale rule).
  `figsize × dpi` is set to the crop's pixel size so the render is placed 1:1 and neither
  deliverable resamples it.  No panel title: a single-panel figure lets the LaTeX caption carry the
  description instead of paying a title band for it.
- **Timing (RTX A6000, 161 568 facets):** 7.3 s for PartField's feature prediction; **44 s** for
  the whole `pfbunny` target cold, surface to composed PDFs (history: the target no longer runs
  PartField; the cache below now lives in `overview/assets/seg/pf_cache/`).  `pf_cache/` (290 MB `features.npy`
  + a manifest keyed on the surface hash, checkpoint and PartField revision) makes the inference a
  no-op on a rerun, and `Trimesh`'s STL export is byte-stable, so the manifest still hits after
  `build.sh` has deleted and regenerated the surface — which is also what keeps the quoted stats
  matching the shipped PNGs.  The cache is gitignored; `pf_stats.txt` is not.

#### Shipped settings for the `partfield_*` figures
| Knob | Value | Where |
|---|---|---|
| run | Figure 1's: `overview/assets/seg/` (features + `pf_k8.npz`), sha-checked | `pf_prep.py` |
| clusters | `8` (the overview's: each ear its own part) → 23 connected segments | `PF_CLUSTERS` |
| feature channels | 448, L2-normalised before PCA | `pf_prep.py` |
| PCA clip | 1st/99th percentile per component | `PF_CLIP_PCT` |
| colour | `L* ∈ [52, 70]` from PC3, `a*, b* ∈ ±42` from PC1/PC2 | `L_RANGE`, `CHROMA` |
| segment palette | the overview's 8 pastel tints, `segment_tints` assignment; touching ΔE ≥ 16.9 | `ov_style.SEGMENT_TINTS`, `pf_palette_check.py` |
| camera | `(1.5, -3, 1.25)`, `CameraViewAngle 25` — §5b-bis's | `VIEW_DIR` |
| figure | 1264 px square = 3.61 in at 350 dpi, common crop | `pf_figure.py` |

### §5c One cut object, many derived regions (learned building `diffused_nitsche`)
Five panels showing what a diffused Nitsche needs — `s` → `φ` → boundary region → the BC
intersections → the trimmed mesh — all on **one cut object with one camera**.  Same four stages as
§5b (`dn_prep.py` → `render_dn.py` → `dn_figure.py`), reusing `_shared/bc_bunny_geom.npz` for the MC
surface so the ear selection is *literally* `bc_region_id`'s.  **Every panel is the octree mesh
with its cell frames** — (a)/(b) the full mesh over `s` then `φ`, (c)/(d) the same with the tagged
regions drawn over it, (e) the **trimmed** mesh over `φ`.  Full vs trimmed is the point of the
last panel, so the trimmed mesh appears nowhere else.

- **Every region is a production call, and every port is ASSERTED against it.**  `dn_prep.py` runs
  the real predicates — `is_boundary` from `_build_adaptive_quadrature`, `SurfaceSelector.resolve` +
  its two-stage `__call__`, `_classify_bc_elements`' widened-`torch.isclose`
  `identify_boundary_nodes`, `_trim_exterior_elements` — and then asserts what it re-derived:
  the φ-threshold element set equals the closed-form `|s| < 3.4534 ε` set; the trimmer's element
  count equals the `keep_elem` mask used to carry tags across; `is_boundary ⊆ keep_elem` (the
  trimmed mesh really is interior ⊎ band); Dirichlet ∩ Neumann is empty; both BC sets are
  non-empty; the trimmer removes something; the band is neither empty nor everything.  These turn
  "I ported it faithfully" into something the build can fail on — and that last one is precisely
  the truncated-SDF failure below, encoded so it cannot come back silently.
- **The BC recipes are quoted from production, not invented.**  Neumann = `bc_region_id`'s split
  flood fill (seed `HighestSelector('y', 0.05)`, half-space from the two seed components, 5.0°/6
  and 4.0°/2 rings).  Dirichlet = the image-to-FEM surface-BC example's clamped base,
  `AlignedNormalSelector((0,−1,0), 45°, oriented=True) & LowestSelector('y', 0.05)`.  Neither is
  tuned here; the figure shows what the solver would actually select.
- **Render the field ON the mesh, not on a dense volume beside it.**  A separate `257³`
  `STRUCTURED_POINTS` cube gives a prettier `φ` ramp, but the figure is about a discretisation:
  the reader needs the cells.  Dropping it removed a 136 MB intermediate and a whole BVH grid
  query, and `(a)`'s far field going blocky is *correct* — that is the resolution the octree
  actually stores out there (cf. §4 "render the maintained background grid").
- **The TripoSG decoder SDF is TRUNCATED at `sdf_truncation_threshold = 0.01`, and that breaks
  every phi-threshold test.**  With `ε = h_min/2` the plateau only reaches `φ = 0.0059`, never the
  `1e-3` the boundary test wants, so `_build_adaptive_quadrature`'s `is_boundary` flagged **all
  583 164 elements** and `_trim_exterior_elements` removed **zero**.  The ratio is scale-free —
  `T/ε = 0.01·2^depth` — so it is 1.28 at depth 7 and 2.56 at depth 8; no depth escapes it.  Drive
  any φ/band figure with the production **BVH re-init** instead
  (`remeshing.build_initial_octree_from_trimesh` → `analytical_sdf.compute_nodal_sdf`, the shape-opt
  path), which returns a true untruncated distance in the same frame.  *Check the field's actual
  range before trusting a threshold that reads fine in the source.*
- **Matching a sibling figure's view is not a change of basis — CHECK THE DETERMINANT.**
  `prep_bunny_sdf.py` builds its frame with `v[:, [0,2,1]]`, an axis **transposition**, so
  `det = −1` and `bunny_sdf_pv` / `bunny_sdf_octree` render a **mirrored** bunny.  Pushing their
  camera through that same swap faithfully reproduces the mirror: the head came out on the right
  and the solid cut half on the left, both backwards, and it looked deliberate.  The bunny is
  near-bilaterally symmetric, so the same picture comes from viewing the un-mirrored bunny from the
  **other flank**.
- **Derive such a camera from the screen basis, don't hunt for it.**  With `r = f × up` and
  `up = +y`, `r = (−f_z, 0, f_x)`.  Two requirements pin the signs: the head (at `−x`) must project
  left → `r_x > 0` → `f_z < 0`; and the kept **far** half (`n·f > 0`, `n = +z`) must project right
  → `f_x < 0`.  Keep the source figure's component magnitudes for the same obliquity and you land on
  `CameraPosition (1.5, 1.25, 3.0)`, camera on `+z`, keeping `z ≤ CUT_Z`.  The light kit, view angle
  and dolly need no mapping and are used verbatim.
  *Verify by cropping both figures to their ink boxes and putting them side by side* — a mirrored
  pose is obvious there and nearly invisible panel-by-panel.
- **A legacy VTK `POINT_DATA` block may hold SEVERAL `SCALARS` arrays** — one mesh file then serves
  the `s` panel and the `φ` panels, and `ColorBy` picks between them.
- **A thin shell CANNOT read at whole-object scale on its own.**  Cut a band-only mesh and the hole
  in the ring is filled by the far half's inner surface, so it looks like a solid voxel bunny.  Give
  the region a **backdrop to sit on**: draw the whole mesh coloured by `φ` and lay the region on
  top.  Then the cut face reads as "grey cells straddling the `φ` transition" — the region, the
  field it comes from and the mesh it lives on, in one picture.
- **Overlay = a `Threshold` on the SAME grid**, not a second file: one source of truth for which
  cells are tagged, and the tags travel with the mesh.  `Threshold.Scalars = ["CELLS", "tag"]`,
  `LowerThreshold` / `UpperThreshold` / `ThresholdMethod = "Between"` in 5.13.
- **Two datasets sharing a cut plane must be offset — offset the BACKDROP backwards, never the
  overlay forwards.**  Coplanar cut faces z-fight, so one has to move.  Pushing the overlay 0.02
  (0.64 `h_min`) toward the camera exposed a *different* section and silently **dropped the ears**,
  which are only ~2 cells thick; pulling the backdrop back by the same amount keeps the region on
  exactly the plane (e) cuts, and a smooth field like `φ` shows no difference at all.  Move the
  thing whose exact geometry does not matter.
- **Hard-step a categorical cell tag** with duplicated control points
  (`1,c1, 1.5,c1, 1.51,c2, 2.5,c2, …`); (c) paints every tag one grey, (d) separates the BC subsets,
  so both panels are one geometry and one LUT apart.
- **`s` is DIVERGING, `φ` is SEQUENTIAL — pick the ramp by the job, not by the neighbour.**  `s` is
  signed, so it keeps the house two-hue map with the neutral grey at 0.  `φ ∈ [0,1]` is a magnitude,
  so it takes one hue, light→dark (dataviz `color-formula.md`) — the documented sequential blue,
  steps 100→700 evenly placed, validated single-hue (4.1° spread, monotone ΔL 0.095).  An earlier
  version mirrored the `s` map onto `φ` so the two would rhyme; it made the exterior — most of four
  panels — a saturated dark brick block, and it spent a diverging encoding on a quantity that does
  not diverge.  The sequential ramp lets the exterior recede (its lightest step is *meant* to sit
  near the surface) and puts the ink on the regions.  It also frees orange and teal for the BC
  overlays, which the blue↔orange `s` map did not.
- **Run the validator on the HOUSE palette too — ours failed.**  Neumann orange `#DD8452` against
  Dirichlet green `#55A868` is CVD ΔE **4.5**, under the hard floor of 6: the classic red/green
  collapse, on the two most important labels in the figure, and no choice of neutral rescues it.
  Replaced the green with teal `#12a594` and re-stepped the neutral to `#bababa`.  Validated
  all-pairs: worst CVD ΔE 8.7 (≥ 8 target), worst normal-vision 15.6 (≥ 15 floor).  The orange is
  unchanged, so the ears still match `bc_region_id`.  **There is no `node` on this box** — port
  `dataviz/scripts/validate_palette.js` to Python (same thresholds, same
  Machado-Oliveira-Fernandes 2009 matrices) rather than eyeballing ΔE.
- **A neutral must clear the chromatic slots by LIGHTNESS, and there are two ways up.**  Any grey
  sitting level with them (OKLCH L ≈ 0.58–0.77 here) collapses under CVD no matter how warm or cool
  you make it — warming it was tried and changes nothing, because the constraint is the chromatic
  pair, not the hue.  That leaves a dark band (≤ `#6c6c6c`) and a light one (≥ `#bababa`).  Take the
  **light** one: the boundary region is the base state and should recede, whereas a dark grey turns
  most of (c) and (d) into a heavy mass.  Its separation from the field's lightest steps is
  modest (ΔE 8) but it is a *neutral on a chromatic field* with a black frame on every cell, so it
  reads.  Give a pale legend chip a `0.5 pt` edge or it dissolves into the white page.
- **Show the SURFACE the selectors actually ran on**, not just the cells that came out.  Panel (d)
  carries `bunny_sdf_pv`'s translucent near half of the geometry with the selected facets painted in
  the same orange/green their cells carry, so "these facets → those cells" is one picture.  Export
  three **disjoint** pieces (`~(ears|base)`, `ears`, `base`) — body-plus-subset z-fights (§5b).
  Body at `bunny_sdf_pv`'s 0.42, but the selected facets **opaque** — at 0.85 the dark exterior
  behind them muddies the orange to a brown.  And opaque means **no PBR**: with the body's
  bunny_sdf_pv material (PBR, no environment map) the opaque facets came out dull brown anyway.
  Translucent body → PBR; opaque highlight → bc_region_id's plain Gouraud.  The §5b rule about
  PBR applies per *actor*, not per figure — and it is a prompt to LOOK, not a law: `void_nucleation`
  §5d puts PBR on an opaque half-section under a light kit + tone mapping and it comes out warmer
  and better-formed than the matte version, the opposite of what happened here.
- *(The selector's resolved bounding boxes were drawn here for a while, then hidden; the selector no
  longer builds them at all (§5b), so there is nothing left to draw.  The lesson stands for any dense
  wire overlay on a cut mesh: do NOT clip it at the cut plane (everything behind it is hidden, so the
  lattice vanishes; a slab does not help either — the pile-up is along the view axis).  Weight the
  strokes by 1/√N — 44 boxes over the ears vs 422 over the base, and at the ears' weight the base,
  tiling a nearly edge-on surface ~5 deep, merges into a solid blue blob that no opacity fixes
  (10 layers at 0.3 still composites to 0.97).  What worked was `lw = clip(18/√N, 1.0, 2.6)`,
  `opacity = clip(5.5/√N, 0.20, 0.85)`, `RenderLinesAsTubes = 0`.)*
- **Don't drop to close-ups until the wide shot has actually failed.**  The two BC regions first got
  a stacked pair of bbox-framed close-ups, which needed a highlight-driven auto-crop to keep the
  body out of the corners and still landed at half the other panels' size.  On the shared whole-cube
  camera, painted apart on one panel, they read fine — and the reader can place them.  A close-up is
  a cost (its own crop, its own framing bugs, half the area); pay it only when the feature really is
  invisible at full scale.
- **Label a panel with the condition the CODE writes, not the one you derived from it.**  The
  boundary test is a phi range, `1e-3 < φ < 1−1e-3` (`nitsche_field.py:915`).  Inverting
  `φ = ½(1 − tanh(s/ε))` gives `|s| < ε·atanh(1 − 2φ_lo) = ε·½ln((1−φ_lo)/φ_lo) = 3.4534 ε`, which
  is correct and was on the panel for a while — but it is a step the reader has to trust, and the
  factor swings with `φ_lo` on a log scale (1e-2 → 2.30 ε, 1e-3 → 3.45 ε, 1e-4 → 4.61 ε).  Print the
  range; keep the ε-multiple in `dn_stats.txt` for the caption.  Print it as plain **decimals**
  (`0.001 < φ < 0.999`), not `10⁻³`: mathtext sets superscripts high off the baseline, which at
  6.8 pt reads as a typo, and the decimals are exact, shorter, and need no mental arithmetic for the
  upper bound.  Derive both from the stats value and **assert they round-trip**, so the printed pair
  can never misstate the real one.
- **`\!` fights mathtext, it does not tighten it.**  mathtext ignores literal spaces in math mode
  and sets its own spacing around relations, so `"a < b"` and `"a<b"` render identically; `\!` then
  claws back that correct spacing and the `<` signs read as cramped.  When a title is 0.5 pt too
  wide, take the space from the **gutter** (`wspace`), not from the typography — a self-imposed
  MIN_GAP is not a reason to ship squeezed math.
- **Don't let a title claim of a CELL what the code only tests at a NODE.** The region is
  "cells with a node in `0.001 < φ < 0.999`", not "the region `0.001 < φ < 0.999`" — the drawn cells
  reach up to one element beyond it (§6, same trap as the bbox pre-filter).
- **Assert the layout, don't eyeball it.**  `dn_figure.py` ends with guards that every title,
  colour-bar tick, bar label and the key lie inside the canvas; that adjacent titles keep a minimum
  gap; that the three footer blocks (two bars, one key) do not overlap **in 2-D**; that each bar is
  centred under its own panel; and that the key's chips are centred on the bars.  Every one of them
  caught a real defect here — the φ bar's axis label landing on the key, `(b)`/`(c)` titles
  touching, the label falling off the canvas when the bars moved up, the key sitting a bar-height
  high.  Two footer facts worth keeping: a horizontal bar's axis label hangs ~0.27 in below it,
  which is what sets the footer height; and once bars and key share a vertical band in different
  columns, vertical clearance stops being the test.
- **To line a legend up with a colour bar, measure the CHIPS and iterate — the arithmetic lies
  twice.**  (i) matplotlib offsets a legend from a *point* `bbox_to_anchor` by `borderaxespad`
  (default 0.5 font-units — 0.024 figure heights at 6.8 pt), so solving "anchor = target − h/2"
  silently lands it a whole bar-height high; pass `borderaxespad=0`.  (ii) even then, centring the
  legend's *bbox* leaves the chips ~2.5 px off, because the bbox includes label ascenders and
  descenders that are not symmetric about the chip row — and the chips are what the eye lines up.
  Centre `leg.legend_handles`' union extent, correct by the measured residual in a short loop, and
  assert it.  Verify in **rendered pixels** (match the chip's fill colour, read its row range), not
  in figure coordinates: the figure-coordinate check passed exactly while the chips were still off.
- **Put each colour bar under the panel it belongs to, titled by its variable.**  The `φ` bar was
  briefly stretched across (b)–(e) to advertise that all four share the map; it just read as a
  detached full-width rule.  One bar per field, under its own panel, and the shared map is obvious
  from the panels themselves.
- **A key entry that needs explaining is not an entry.**  The translucent geometry in (d) was keyed
  "Embedded surface" and drew a "what is that?" — it is not a region like the other three, and the
  chip put a surface in a list of cell types.  Dropped; the caption carries it.

| Knob | Value | Where |
|---|---|---|
| octree | depth 3 → 6 on a 1.16 domain, `h_min = 1.16/2⁶ = 0.0181` | `INIT_DEPTH`, `OCTREE_DEPTH` |
| ε | `0.5 · h_min` (`epsilon_to_size_ratio`) | `EPS_RATIO` |
| band | `φ ∈ (1e-3, 1−1e-3)` ⇔ `\|s\| < 3.4534 ε = 1.727 h_min` | `PHI_LO` |
| BC tolerance | `max(1.5ε, 0.75 h_min) = 0.75 h_min` (solver `_bc_match_tolerance`) | `eps_tol` |
| Neumann | `bc_region_id`'s split flood fill (5.0°/6 · 4.0°/2) → 2 581 cells (2 954 at the old tolerance) | `dn_prep.py` |
| Dirichlet | image-to-FEM surface-BC example: ∠45° down ∧ lowest 5 % → 3 386 cells (3 756 at the old tolerance) | `dn_prep.py` |
| cut | `z = 0.009` (nudged off the cell boundary), keep `z ≤` (the far half) | `CUT_Z` |
| backdrop offset | `CUT_Z − 0.02`, mesh only; the overlay stays on `CUT_Z` | `BACKDROP_Z` |
| camera | `(1.5, 1.25, 3.0)`, `ViewAngle 25`, dolly 1.16 — matches `bunny_sdf_octree` | `VIEW_DIR` |
| surface | near half, body 0.42 translucent, selected facets opaque | `surface_actor` |
| `φ` ramp | sequential blue 100→700, `#cde2fb` → `#0d366b` | `PHI_RGB` |
| overlays | `#bababa` region · `#DD8452` Neumann · `#12a594` Dirichlet | `GRAY/ORANGE/TEAL` |
| figure | 7.16 × 1.98 in, 5 equal square panels, `wspace 0.14` | `dn_figure.py` |
| footer | `s` bar under (a), `φ` bar under (b), each titled by its symbol; key under (c)–(e) | `dn_figure.py` |

`build.sh nitsche` runs prep → render → compose and then **deletes its intermediates**, so a
renderer-only experiment must re-run `dn_prep.py` first (same trap as §5b; here it surfaces as
`RuntimeError: File not readable: dn_full.vtk`).  Rebuild from scratch before believing anything:
dropping the dense volume left `dn_stats.txt` still writing `NG` and `grid.min()`, and only a
from-empty build hit that line — every incremental run had the stale names in scope.

**BC tolerance (2026-09-30).**  The solver's selector tolerance went from `max(2ε, 0.75h)` to
`max(1.5ε, 0.75h)`: the old one clamped a strip ~h tall up the free faces next to a Dirichlet
face.  `dn_prep.py` follows it, and with `ε = h/2` it equals the `0.75·min_elem` that
`bc_bunny_select.py` always used, so the two figures now agree.

**Full rebuild for the `d → s` relabel (2026-10-15).**  `dn_figure.py` had said `$s$` since
5822271 but the committed PNG/PDF still read `d`; a from-scratch `./build.sh nitsche` fixed that and
re-solved panel (f) with current code.  Max |u| on the surface went **1.094e-1 → 9.251e-2** (−15 %;
same 38 898 nodes / 29 549 elements, 7 Newton iterations, ~2.85 s), so the ears droop a little less
at the true-scale warp.  The prep counts (`dn_stats.txt`) and panels (a)–(e) are unchanged
pixel-for-pixel apart from the label, so the drift is in the solver path (e.g. Neumann rescaling,
beta flooring), not the geometry or BC selection; it was accepted, not bisected.  8c2f794 (saved-segment BC tolerance) is not a cause: `ov_bunny_solve.py`
calls `NitscheNeoHookean` directly, not `workflow_tool.run_forward`, and the solver's own tolerance
is numerically unchanged by that commit.

### §5d Capture a live optimiser run (learned building `void_nucleation`)

**Current provenance (2026-09-24): `void_nucleation` is a focused composition of
the canonical `hiddenphi` capture, not a second optimizer run.**  The earlier
`vn_prep.py` duplicate was removed.  `build.sh voidnuc` first runs the cached
`hiddenphi` build, `voidnuc/render_vn.py` aliases its six field panels, carved
surface, final surface, and evolution strip without resampling, and
`vn_figure.py` reads `hiddenphi/hp_stats.txt` directly.  This makes every count
and every image in the two figures refer to one execution. The focused layout has three paired
field columns (current state, sensitivities, hidden-field update), then the carved and final
surfaces, plus the shared evolution strip. The detailed bullets below record lessons learned while
building the original capture; names such as `vn_prep.py` are historical. Their current production
home is `hiddenphi/hp_prep.py` unless a bullet explicitly says otherwise.

- **Subclass the production optimiser to RECORD; never re-implement the loop.**  `Capture(ShapeOptTrimmed)`
  hooks `_post_tape_backward` (both gradients + the SDF they belong to), `_pf_filter_thin_features`
  (hidden phi in *and* out, so the filter's verdict is data) and `_post_optimizer_step_remesh_impl`
  (to spot the cavity).  `p.step()` alone will NOT do — it is a forward solve and leaves
  `_last_e_phi_grad` unset; the gradients only exist inside `optimize()`.  Note `_snapshot_extra`
  **deletes** `_last_e_phi_grad` after each iteration, so it has to be taken in the hook.
- **Both `_hidden_phi` and the SDF dofs are indexed on the OUTER octree** (`_prev_octree_nodes` /
  `_prev_octree_elements`), not the trimmed FEM mesh — that is the mesh to export.
- **Counting connected components of the MC surface is NOT a void detector.**  It reads 4-7 from
  iteration 2 onward on a perfectly solid part — marching-cubes and BC-zip debris on the outer
  boundary — and wanders up and down with no relation to topology.  A void is unambiguous in the
  FIELD: label `sdf > 0` on the 65^3 background grid (`_sussman_grid_sdf` /
  `_sussman_grid_shape`) and keep the components that do not touch the domain boundary.  Confirm it
  a second way on the mesh — post-nucleation nodes with `s > 0` whose nearest PRE-nucleation node had
  `s < -3 eps` (`cKDTree`; the mesh is rebuilt every iteration so "same place" is a lookup) — and
  assert their centroid lies inside the part's bounding box.  An earlier version trusted the
  component count and "confirmed" a nucleation that never happened.
- **Gate the detector on the mechanism.**  This run also throws a transient 31-cell cavity at
  iterations 3-4, before any dip is viable, purely from the first MC re-inits.  Only a cavity that
  appears on an iteration where `_pf_filter_thin_features` COMMITTED is one the hidden field carved.
- **The volume penalty is ONE-SIDED (softplus, active only above target) and it is the only term
  that pushes phi down.**  The strain-energy term always wants more material, so once the design
  drops below `target_volume_fraction * V_0` the interior `dJ/dphi` collapses from 1.5e-2 to 5e-13
  and **no void can ever nucleate** — which is exactly what the first three configurations did, and
  it looks like a broken mechanism rather than an inactive constraint.  Set the target low enough
  (0.25 here) that the penalty stays on for the whole run, and read `penalty=` in the iteration log
  before blaming anything else.
- **`gd_lr` and `pf_phi_lr` are independent, and that is what makes this fit in a figure.**  `gd_lr`
  moves the visible boundary, `pf_phi_lr` the hidden field: hold the part still at `gd_lr = 0.05`
  (which also keeps the volume above target) and let `pf_phi_lr = 10` reach 0.5 in seven iterations.
  Scanned: 2 and 3 only ever reach the 4-18 bg-cell resolution floor; 5 makes a 14-cell cavity at
  iteration 13 and the real one at 28, which would put a hole in the "before" panels; **10 nucleates
  exactly once, at iteration 7, with nothing before it**.
- **A gradient that spans three decades needs a LOG ramp, and picking the normaliser is the whole
  panel.**  `|dJ/ds|` is dominated by the traction patch (median 3.4e-3 of max, 99th percentile
  2.4e-1), so over its maximum the entire boundary shell falls in the palest step and the panel
  shows a blank part with one bright sliver — which argues nothing.  The 99th percentile helped and
  a square root helped more, but only `log10(|g| / max)` floored at 4 decades puts the shell at a
  mid tone, the load at the peak and the deep interior at the floor.  The deep interior is EXACTLY
  0.0 (asserted), so the floor is where it belongs.  Label the bar in decades — the scaling goes on
  the axis, never inside the image.
- **Draw the `s = 0` curve on every panel.**  Contour the EXTRACTED SURFACE, not the volume: on the
  volume it returns the part's boundary as an isosurface standing in front of the cut face; on the
  surface it returns curves — the outline on the cut face and the silhouette on the sides.  Without
  it the two gradient panels are unreadable (`|dJ/ds|` is zero over the interior AND the exterior,
  so nothing says which side is material), and on (d) the same curve draws the new cavity's wall.
  A contour inherits the array it was cut from, so ParaView colours it from that LUT and the "ink"
  line comes out mid-blue — **`ColorBy(disp, None)` is what makes `DiffuseColor` take effect.**
- **The strip is a SOLID half-section with the cut-away half ghosted — three actors, not one.**
  Three treatments were tried in order.  (1) The whole closed surface, exterior only: reads as a
  clean outline, but once a void exists inside the slab there is no way to see it and the frames
  after nucleation are indistinguishable from the ones before.  (2) The surface clipped in half:
  a SHELL, so the frame fills with the inside of the far wall and the cross-section reads as a
  cavity everywhere.  (3) What ships — the half cut as a capped SOLID, its cut face flat-filled a
  step darker than the outer surface, and the removed half drawn over it at `Opacity 0.20` the way
  `render_bunny_sdf.py` ghosts its near half.  Now the section is unambiguously material, the void
  is a hole in that material, and the ghost keeps the round tabletop and the far legs — which the
  profile alone loses — visible.
- **Cap the half with a BOOLEAN, not with `slice_plane(cap=True)`.**  trimesh's capping path imports
  `shapely`, which is not installed in mechfoundry-warp117 (only `challenge`'s prep uses it, from a
  private venv, §5k); `trimesh.boolean.intersection` against a half-space box
  goes through manifold3d, which is, and returns a watertight capped solid with a proper hole where
  the cavity crosses the plane.  Build the box to meet the plane EXACTLY (pad only its far sides) and
  assert the two halves' volumes sum to the whole — that catches both an overlap and a missed cut.
- **Give the cut face its own mesh and render it UNLIT.**  Recolouring a face group of one mesh is
  not an option (coplanar actors z-fight, and nudging one toward the camera prints as a bright rim),
  so split the capped half on `normal ~ -n AND |y| < 1e-6` and export two STLs.  Then set
  `Ambient 1 / Diffuse 0` on the face: the boolean's cap is a fan of long slivers whose facet normals
  wobble a degree or two, and under ANY diffuse term that prints as a spray of diagonal scratches
  across the section.  Killing the diffuse term is also correct on its own terms — a section face is
  a fill, not a lit form.
- **Show the design EVOLVING, in a strip, and show it as the SURFACE.**  All four field columns sit
  at the nucleating iteration — where, by construction, the blank still looks like the blank — so on
  its own the figure looks like nothing happened.  Row (e) is the design's own marching-cubes
  surface (`_get_or_build_mc_surface_physical`, already captured in the backward hook and exported
  as one STL per frame), halved on the SAME plane and shot from the SAME camera, at iterations
  0/20/35/45/60/80.  Draw it with `bunny_sdf_pv`'s material **including PBR** (`Interpolation`
  PBR, `Roughness` 0.45, `Metallic` 0): §5b's "opaque + no environment map = dull brown" does not
  bite under this view's light kit + tone mapping, and the PBR solid holds its form better than the
  matte one.  That warning is a prompt to CHECK, not a rule.  Re-sectioning the octree over
  `phi` for the strip was the first attempt and it just repeated column (a) six times; the strip's
  job is the geometry, the columns above already carry the fields.  It is also what makes the TABLE
  legible: by iteration 20 the section reads as a top plate over a central spine with a leg on each
  side, which no single panel at iteration 4 could show.  Give the strip its own shared ink box and
  aspect — its content is a different shape from the field panels'.
- **A vertical elevation, not a plan section.**  A horizontal cut through this blank is a disk and
  says nothing about what the part is; the profile through two feet shows the blank standing, loaded
  on top.  With the feet on the axes that plane is `y = 0`, a coordinate plane.  It was the box
  DIAGONAL (`x = y`) for as long as the feet sat at 45 degrees, which cost the triangulated cells and
  the camera-fitting workaround in the next bullet.
- **A diagonal cut face is WIDER than the box, so size the camera off the FACE.**  No longer binding
  now the cut is `y = 0`, but the fitting code still sizes off the face and should stay that way.
  `0.5 * max(box edge)` is the fitting radius for an axis-aligned view; on the `x = y` plane the face is
  `2 * 1.14 * sqrt(2) = 3.22` across for a 2.28 box, and fitting to 1.14 put half the blank out of
  frame.  Use the face's half-diagonal, and trim the face IN ITS OWN PLANE (two more plane clips
  along `u = (1,1,0)/sqrt2`) or half the panel is the box's corners.
- **Stop letterboxing panels to squares when the content is not square.**  This cut face is a wide
  elevation (aspect ~1.9); padding each panel to a square spent 45 % of its height on white and made
  the figure 1.5 in taller for nothing.  Take the aspect from the shared ink box.
- **The GLB series is on, and the run is KEPT.**  The production `vtu_writer` emits one per iteration
  to `<output_root>/glb/surface_tNNNN.glb` when `output_surface_glb=True` (the cantilever config sets
  it); the canonical capture is `hp_prep.py`, with its `output_root` anchored to repo-root
  `output/`. `voidnuc` owns no numerical run directory or VTK/STL intermediates.
- **An over-aggressive hidden field stops the boundary moving at all.**  At `pf_phi_lr = 160` /
  `gd_lr = 0.1` the volume reduction was taken up entirely by INTERIOR cavities, so the one-sided
  volume penalty relaxed and the outer boundary had no reason to move: 110 iterations plateaued at
  -30 % with the profile still a cylinder and the interior a chaotic lattice.  The two rates compete
  for the same volume budget — whichever is faster does the carving.  Give the boundary the larger
  share.
- **Run the two mechanisms in SEQUENCE, as a manual schedule — not together.**  Phase 1 is pure
  shape optimisation (`pf_phi_lr = 0`): the boundary carves the blank towards a table and nothing
  touches the hidden field.  Phase 2 switches it on at a **fixed iteration** (`HP_PF_ON`, now 35 of
  91).  An earlier version triggered on volume instead (`VN_HALFWAY`); a fixed iteration is easier to
  reason about once `gd_lr` is settled, and at `gd_lr = 8` iteration 35 is both "recognisably a
  table" and 47 % of the way down from the start volume, so the two criteria coincide anyway.  The
  sequencing itself is not a convenience — running both at once is UNSTABLE: the
  hidden field opens cavities faster than the boundary update can cope with, and the run dies with
  `RuntimeError: BC region is being isolated` at `gd_lr` 0.3, 0.5 and 1.0 alike, somewhere between
  iteration 1 and 5, **non-reproducibly** (identical settings reached iteration 58 once and died at
  5 the next time).  Separated, phase 1 is exactly the pure shape-opt run that was always stable,
  and `gd_lr = 1.0` holds for the whole 70 iterations.
  *This cost a shipped figure.*  An earlier version was built from a combined run that happened to
  survive; re-running the same settings produced junk geometry from iteration 40 on and then died.
  **Re-run a stochastic prep before believing a figure, not after someone asks about it.**
- **Phase 2's step size sets the LAG and the CAVITY SIZE together, and not monotonically.**  A larger
  step pushes more nodes across 0.5 in the same iteration, so the first dip is a bigger contiguous
  region — but the relationship is noisy.  All switching on at iteration 31: `pf_phi_lr` 45 -> 10
  iterations later, four 10-31 cell cavities; 120 -> 3 later, three 8-11 cell; **150 -> 3 later,
  four 81-134 cell**; 200 -> 2 later, four 86-137; 250 -> 1 later, four 9-27; 300 -> 1 later, four
  53-144.  Scan it; do not extrapolate.  **And rescale it whenever the switch point or the MESH
  moves** — it scales inversely with the nodal support volume.  The later the switch, the less thick
  interior is left to dip: at `HALFWAY = 0.7` (switch at iteration 55) the same 150 gives only 14-15
  cell cavities and 200 is needed for 57-124.  And on the CURRENT graded octree (depth 7, init 4,
  switch at 25) the whole scale drops by an order of magnitude and the response goes bimodal —
  `pf_phi_lr` 2 -> never fires; 10 -> 8 iterations later, eight 10-26 cell cavities; 15 -> 5 later,
  eleven 8-27 cell; **25 -> 3 later, ONE 7.6k-cell lens**; 200 -> immediate, one 21k-cell cavity that
  is most of the interior.  Nothing in between gives "a few medium cavities": the graded interior
  sits at depth 4-5, so its nodes have near-identical support and cross 0.5 together.  One clean lens
  in the slab reads better in a cutaway than a scatter of specks, so **25 is the shipped value.**
- **Switch LATER to nucleate SMALLER, and it costs almost nothing.**  The switch point is a better
  size control than `pf_phi_lr`.  Moving it from iteration 25 to 35 (and dropping the rate 25 -> 12)
  takes the first cavity from 7265 background cells to **1130** — a thin slot instead of a fat lens
  — because by 35 the boundary has already carved the blank to within 6 % of target and the slab
  left to dip into is thin.  The lag stretches from 3 iterations to 7, which is not a cost: it means
  the figure can show the void GROWING (1130 -> 1482 -> 1943 -> ~3900 by iteration 55) rather than
  arriving full size.  At the same switch, `pf_phi_lr` 25 gives 3035 cells and 12 gives 1130.
- **The run converges at iteration ~80 and DEGRADES after it — more iterations are not more
  converged.**  Objective by iteration: 2.898 (20), 2.149 (40), 1.791 (60), **1.7125 (80)**, 1.7166
  (100), 1.7265 (119).  It bottoms out at 80 and then drifts back up, because `gd_lr = 8` puts the
  boundary in a limit cycle: the one-sided volume penalty pushes in above target and relaxes below
  it, so the volume hunts 1.60 <-> 1.69 every other iteration and `rel_red` alternates sign.  Past
  ~110 the surface starts to speckle — the enclosed-cavity list fills with 40-60 ONE-cell components
  and the MC surface area climbs from 1.199 at iteration 80 to 1.339 at 119, which shows on the
  render as pitting in the top plate and thinned legs.  The current requested build runs 91
  iterations and the strip ends at 90; that endpoint is a final state, not the objective minimum.
- **A `gd_lr` decay would fix the limit cycle, and it is NOT available from a subclass.**  The
  obvious cure for the hunting is to shrink the step late, the way `pf_phi_lr` is scheduled — but
  `optimize()` reads `gd_lr` ONCE into a local before the loop (`abstract_field.py:1619`, used at
  1728), so mutating `self.params["gd_lr"]` mid-run does nothing.  Getting it would mean editing
  production or reimplementing `optimize()`, both of which this figure's whole premise forbids.
  Stop at the objective minimum instead.
- **Run it out and the void MERGES INTO THE BOUNDARY — that is the ending, not a bug.**  The cavity
  grows until the slab above or below it is thinner than the band, and then it stops being an
  *enclosed* component: at iteration 57 the enclosed-cavity list drops to debris and the underside of
  the table is simply hollow.  The blank was solid and the converged part is not, which is the
  topology change the whole mechanism exists to produce — so the strip's last frame is allowed to
  show no enclosed void.  Do not read the empty cavity list as a failed run; check the geometry.
  (If a frame with the void still closed is wanted as the endpoint, stop at 56.)  By
  iteration 80 the underside is fully open and the part is a top plate on four slender legs.
- **Assert the schedule, both ways.**  `hp_prep.py` checks the hidden field actually switched on,
  that no void formed before it did (a leak), and that nucleation followed promptly
  (too small a `pf_phi_lr` never fires).
- **On the graded mesh the cavity is ONE cavity in the table's slab, not four above the feet.**  At
  uniform depth 6 the dips opened directly above the four feet (r = 0.6 = the bolt circle,
  z -0.32..+0.04) and one horizontal section caught all four.  At depth 7 / init 4 it is a single
  region centred on the axis, inside the slab under the top plate, which the `y = 0` elevation
  sections perfectly and (c)'s single `hidden phi = 0.5` outline lands on exactly.  Its SIZE follows
  the switch point: at iteration 25 a fat lens (bbox +-0.51 in x and y, z -0.02..0.26, ~7.3k cells);
  at the shipped iteration 35 a thin slot (z 0.075..0.255, ~1.1k cells) that then grows.
- **The feet's phase is a BC-zip constraint, and it relaxed when the grow went to 0.**  Putting them
  on the axes (0/90/180/270) instead of 45/135/225/315, so a coordinate-plane clip can section two
  of them, used to make `compute_bc_patch` fail with `invalid patch boundary
  (branching-out-degree)` — but only at `bc_zip_patch_grow` 2-4, where an axis-aligned foot centred
  on a spoke of the cylinder cap's triangle fan pinches its own patch boundary.  `so_prep.py` runs
  `bc_zip_patch_grow = 0` now, the phase is free, and the figure takes the axis phase: it buys the
  `y = 0` cut, which is worth real quality (next bullet).
- **A DIAGONAL cut plane triangulates the octree.**  On `x = y` every cell the plane crosses is cut
  off its faces, so ParaView's clip returns triangles and the panels showed a triangulated mess
  where the octree's quad grid should be — the mesh structure is half the point of these panels.  A
  coordinate-plane cut crosses cells face-parallel and the frames stay quads.  It also removes the
  diagonal-face camera fitting (the face is no longer wider than the box) and the in-plane trim,
  though `CUT_U` / `BOX_U` stay in the code because `fit` still sizes off the cut face.
- **Match the mask colour to the ramp's floor, and check what the panel's background ACTUALLY is.**
  These two panels are mostly BC mask — the frozen region covers most of the cropped block — so the
  mask colour *is* the background a reader sees the ramp against.  With a grey mask against a white
  ramp floor the panel read as a grey field with an orange shell floating on it, and the colour bar
  disagreed with the panel at every value near zero.  Sample the rendered PNG rather than reasoning
  about it: `vn_panel_b_d.png` came back `(167,167,172)` across its whole body while the page was
  `(255,255,255)`, which settled it in one command.  Mask and floor are both white now; a lit white
  surface still renders ~`(218,218,218)`, and that shading gap is fine — it is the same shading
  every other panel gets.
- **A sequential ramp for a "how much" field should start at the PAGE WHITE.**  The gradient ramp
  began at a pale tint, so everything below the log floor — all the air, and the whole interior in
  the `|dJ/ds|` panel — printed as a solid block of colour the reader had to notice and then
  dismiss, and the step from "no sensitivity" to "a little" landed as a visible edge.  Starting the
  ramp at white lets the nil region fall away into the paper: the only thing carrying ink is the
  part that has sensitivity.  The mask is white too (previous bullet), so floor, mask and page are
  one tone — the cost is that "excluded" and "zero" become indistinguishable, which is the price of
  the continuity and is worth naming in the caption.
- **A prep that dies partway leaves the renderer reading STALE panels, and the figure looks fine.**
  This is the most dangerous failure mode in the whole pipeline and it bit twice in one session.
  The hidden-phi build runs prep → render → compose; the panels are PNGs on disk that nothing invalidates, so
  when the prep aborted at strip frame 60 the renderer produced two frames from the PREVIOUS run's
  files and the composed figure mixed two runs while looking entirely plausible.  Corollaries:
  **never grep the tail of a failing build** — filtering for `built|Error` showed a bare `Traceback`
  with the actual message cut off, twice, and cost two debugging rounds; and when a build fails,
  assume every downstream artefact is suspect rather than re-running only the step that broke.
- **For a renderer-only experiment, rebuild the strip inputs from the GLB series, not the prep.**
  `build.sh` deletes the intermediates and the prep is ~13 minutes, which is a punishing loop for a
  material or colour-map tweak.  `../output/<run>/glb/surface_tNNNN.glb` is the same surface the
  capture holds, so the half-cut/cap/ghost STLs can be regenerated for a couple of frames in
  seconds and rendered on their own.  Verify the frame first — match its `volume` against
  `evo_vol` in `hp_stats.txt` — because the GLB index is the writer's save counter, not the
  capture's `_it`, and nothing guarantees they line up.
- **Sample the rendered PNG instead of reasoning about what a colour looks like.**  Two colour
  decisions in a row were argued from the code and both were wrong; `Image.getpixel` on the panel
  settled each in one command (`(167,167,172)` across the body of `vn_panel_b_d.png` against a
  `(255,255,255)` page proved the panel's background was the MASK, not the ramp floor).
- **The raw MC surface is NOT reliably a closed volume, and the boolean is strict about it.**  Late
  in the run it carries debris — 6 components at iteration 60, 36 at iteration 80, all but one under
  1e-4 in volume, several of them open — and manifold3d then refuses the ENTIRE half-space boolean
  with `ValueError: Not all meshes are volumes!`.  The strip export dies partway through, so some
  frames are written and some are not, and the render step then reads a STALE panel for the missing
  ones without complaining: the figure comes out looking plausible and mixing two runs.
  `merge_vertices` + `fill_holes` does not rescue it.  Split into components, keep the closed ones
  with real volume, discard the rest (`as_volume`).  **And note the failure is data-dependent** —
  the same iteration list passed on one run and failed on the next, so this is not something a
  single successful build proves absent.
- **Legacy ASCII VTK cannot carry NaN, and it fails SILENTLY.**  Masking a region by writing `nan`
  into a point array is the natural move — `NanColor` is right there — but `vtkDataReader` gives up
  on the array with `WARN| Error reading ascii data. Possible mismatch of datasize with declaration`
  and carries on, so the panels render blank-ish and NOTHING raises.  Write a SENTINEL below the
  ramp instead (`MASK_SENTINEL = -1.0`) and catch it with the LUT's `UseBelowRangeColor` /
  `BelowRangeColor`.  Same visual result, and it survives the reader.
- **Mask the BC region out of the sensitivity panels, and normalise over what is LEFT.**  Every node
  of every boundary element touching a Dirichlet or Neumann patch has its SDF frozen
  (`shape_opt.py` `_build_e_fixed_mask` -> `_e_fixed_bool`), so the sensitivity printed there is
  never acted on; drawing it makes the reader compare a live gradient against a dead one.  Mask it
  rather than zero it — zero would claim the sensitivity IS zero there, a different and false
  statement.  And take the normalising maximum over the FREE nodes: the traction patch sits inside
  the mask and owns the global max, so normalising by it pushes every live value down a decade.
  The mask is big — 31037 of 78031 nodes — but it only touches **3.4 %** of the deep interior (429
  nodes, all inside the solid frozen feet), so panel (b)'s claim survives it.  `hp_prep.py` asserts
  both: that the mask covers under half the deep interior, and that it covers NONE of the nodes
  where the hidden field dipped.
- **A fifth column costs the column titles.**  At 4 columns each panel is 1.63 in and the titles fit;
  at 5 it is 1.30 in and they do not — the existing layout assertion caught "(c) and (d) collide"
  rather than shipping an overlap.  Shorten the titles, do not shrink the font: the detail belongs
  in the in-panel labels and the caption.
- **`ColorBy(rep, None)` is not a safe way to switch off scalar colouring.**  The helper reads the
  rep's CURRENT association and hands it straight back as a string, so on a representation that has
  no colouring yet it raises `invalid association string 'NONE'` — and whether a contour has
  inherited colouring depends on the dataset, so the same line worked on one example and threw on
  the next.  Set `disp.ColorArrayName = ["POINTS", ""]` instead.
- **When retargeting a renderer, ASSERT every constant you meant to change.**  Half a dozen
  `str.replace` edits to `render_vn.py` silently no-opped on a one-word comment mismatch, leaving the
  previous example's `BOX_LO/BOX_HI/FOCUS/VIEW_DIR` in place.  The crop then kept `z >= -0.16` while
  the new cut kept `z <= -0.1603`, the pipeline produced **zero cells**, and all eight panels
  rendered pure white — with no error, because an empty render is a valid render.  `grep` the
  constants after editing, or print `GetDataInformation().GetNumberOfCells()` once per panel.
- **A blocky cavity is the ELEMENT quantum, and only a uniform interior fixes it.**  The carving is
  done by `_localized_phi_to_bg_overwrite`, which trilinear-fills each affected element from its 8
  nodal hidden-phi values and leaves every bg cell outside that set at its old deep-solid value — so
  the field is smooth within an element, discontinuous across the affected/unaffected face, and the
  zero crossing staircases at the element size.  On a GRADED octree every element passing the
  "all 8 nodes deeper than `3 eps`" test is `2 h_min`, because a depth-6 cell that close to the
  surface always owns a non-deep node: the cavity came out perforated, with islands, at that
  quantum.  `remesh_init_octree_depth` **does not fix this** — it is a floor on the FAR field, and
  3 -> 4 left all 270 affected elements at `2 h_min` (12 660 -> 12 744 nodes, nothing else moved);
  5 does not help either.  Only `init == octree_depth` puts `1 h_min` elements in the interior
  (46 080 cells, 50 421 nodes), and then the cavity is a single smooth region.  It costs nothing
  here — 27 s either way.  (Not the sparse scatter: all dipped interior nodes lie inside affected
  elements, so no orphan node spikes survive.  And nothing smooths the surface afterwards — the MC
  re-init re-derives distance FROM it, unlike the BC-zip path which Taubin-smooths its triangles.)
- **`pf_phi_lr` scales INVERSELY with nodal support volume, so it must be re-tuned with the mesh.**
  `dJ/dphi` at a node is `integral(W N dV)`, and the volume-penalty part of it is proportional to
  that node's support.  Going from the graded interior to the uniform one shrinks the support ~8x,
  and the `pf_phi_lr = 10` that nucleated at iteration 7 there produces no dip at all here; 80
  nucleates once at iteration 4, 200 fires at iteration 2 and hollows the part almost completely.
  A "the mechanism is broken" symptom is far more often this than a bug.
- **A uniform interior also changes WHERE the void goes, and it is worth knowing which effect you
  are looking at.**  On a uniform mesh every interior node has the same support, so the
  volume-penalty gradient is the same at all of them (`dphi_deep == dphi_max` every iteration) and
  the strain-energy modulation is ~8 orders smaller: the whole interior crosses 0.5 together and the
  part hollows into a shell.  The graded mesh staggered the crossings by support volume, which made
  the cavity look selectively placed — that localisation was substantially a MESH effect, not
  mechanics.  Smooth-and-global versus ragged-and-local is a real trade-off; `VN_INIT_DEPTH=3` gets
  the old behaviour back in one env var.
- **A panel labelled with a NEGATIVE claim has to be the right instant, or the label is false.**
  Column (c)'s lower panel first re-used column (a)'s render and called it "s (unchanged)" — the
  hidden-field update really does not touch `s`, so it read as obviously true.  It is not: the
  ORDINARY gradient step runs in the same iteration, between the backward pass `_post_tape_backward`
  captures and the `_pf_filter_thin_features` call, and it moves `s` by up to `1.2e-2` in the
  boundary band.  Capture the SDF inside the filter hook as its own array and render that; the
  claim then becomes the true and much stronger one — `max |delta s|` over deep-interior nodes is
  **exactly 0.0**, because `dJ/ds` is exactly 0 there — and the label becomes "interior unchanged".
  Assert both halves (it moved somewhere, it did not move deep), so a step size that starts leaking
  into the interior fails the build instead of quietly falsifying the panel.  Then make each
  column's `s = 0` outline contour ITS OWN instant's SDF, or the outline silently belongs to a
  different time than the field under it.
- **NEVER crop an octree with ParaView's Box clip — use successive PLANE clips.**  `vtkBox`'s
  implicit function is the *max* of six half-space distances, which is not linear, and the clipper
  interpolates it linearly across each cell: every cell large compared to the box has its corner
  chamfered into a triangle.  The far field here sits at octree depth 3, cells `0.4333` across (8x
  the leaf), four of which span the whole far corner — the block rendered with visibly bevelled
  corners that read as a deliberate modelling choice, not a bug, and it survived a full review.  A
  plane's implicit function IS linear, so five plane clips cut exactly at any cell size.  The
  *bounds* of a Box clip are correct, which is why checking them does not catch this: look at the
  silhouette where the cells are COARSEST.
- **Cut where the story is.**  The cavity opens as an L-shaped slab through the mid-plane, so one
  plane at `z = 0.2505` (nudged off the cell boundary) gives every panel the L cross-section going
  from solid to hollow in place — no panel is re-aimed, and no corner cutaway is needed.  Crop the
  octree to the part plus a band first, or the panels are mostly empty air cells.
- **Soften the diverging map's positive end.**  Almost half of every `s` panel is air at the clamp,
  and at the house map's dark brick that block outweighs the part it surrounds (§5c's trap, in the
  other direction).  Stop the positive end at a mid orange; keep the negative range in full — that
  is the part.  Clamp to `+-0.20` and label the tick `>=0.2`: the part is 0.5 thick so `s` only
  reaches -0.196 inside, and letting the far field (+1.56) set the scale flattens every interior
  value into the neutral.
- **The run is not assumed bit-reproducible** (GPU reductions), so do not compare two executions to
  infer an algorithmic difference.  There is now only one shipped execution: quote captured values
  through `hiddenphi/hp_stats.txt`, and have both compositions consume those same artifacts.
- **Sentence-case stage titles, field labels inside the panels.**  Titles name the STAGE ("Void
  carved at MC re-init"), the in-panel corner label names what is plotted (`phi`, `|dJ/dphi|`,
  `hidden phi`, `s  (unchanged)`).  Two labels, two jobs; neither has to carry both.
- **Check the boolean backend before trusting a union.**  `manifold3d` IS installed now and
  `trimesh.boolean.union` gives a watertight blank, which is what the current cylinder + four
  protruding feet relies on.  When it was missing, trimesh's fallback silently CONCATENATED the
  meshes — interior faces left inside the material, nothing for the sign test to work with, and no
  error.  Assert `is_watertight` on the result either way.

| Item | Current source |
|---|---|
| optimizer, mesh, schedule, gradients, counts | `hiddenphi/hp_prep.py`, `hiddenphi/hp_stats.txt` |
| (a) current hidden φ + s | `hp_panel_a_hid.png`, `hp_panel_a_d.png` |
| (b) sensitivities | `hp_panel_c_phi.png`, `hp_panel_c_d.png` |
| (c) post-step hidden φ + s | `hp_panel_d_hid.png`, `hp_panel_d_d.png` |
| (d) first carved surface | `hp_panel_e.png` |
| (e) final surface | `hp_panel_evo_<iter_final>.png` |
| (f) evolution | all `hp_panel_evo_<evo_iters>.png` listed by `hp_stats.txt` |
| adapter/composition | `voidnuc/render_vn.py`, `voidnuc/vn_figure.py` |

`build.sh voidnuc` therefore runs the cached hidden-φ render first and only then
aliases/composes the focused figure.  It never invokes an independent optimizer.

### §5e BC-zip: restoring an exact patch onto a changed body (learned building the BC-zip figure)

**STATUS: `bc_zip_workflow.{pdf,png}` is built and iterated to its current form.**  `./build.sh bczip` runs the three-stage pipeline —
`bz_prep.py` (GPU, captures every stage) → `render_bz.py` (pvbatch, 7 panels + 3 seam zooms) →
`bz_figure.py` (matplotlib compose) — and deletes its intermediates.  Layout: steps 1–4 across the
top as whole-body views (what they change is global — which triangles are selected, which loops
exist), steps 5–7 across the bottom as seam close-ups (at body scale they are the same picture; the
strip is already in place after step 5 and the last two only re-triangulate and relax it), key in the
eighth cell.  Every number on the figure is read back from `bz_stats.txt`, so the caption cannot
drift from the run.

**The figure's geometry is the TABLE (`BZ_CASE=table`, the default); the Y-bracket is a mini
example.**  `bz_prep.py` now takes both: the table is the real CC0 CAD with 45°-twisted legs and is
the ordinary path end to end (4 patch components, 1:1 loops, no clustering); the bracket is
fabricated and kept only because it reaches the two branches the ordinary path never does.

**FIXED — the per-panel VTK export used to assume ONE patch component.**  Everything from
`Pf_local = comps[0]["faces"]` down remapped only the first component, so (a)/(c) highlighted one
foot of four and (e)/(f) rendered with no patch at all.  The fix is not to extend that remap: **the
cached patch is exact and never moves, so write it once as its own mesh in patch-local coordinates**
(`bz_patch_all.vtk`) and overlay it wherever it is needed.  That sidesteps pre-compaction indexing
entirely.  Two more bugs of the same family, both found only by looking at the render:

- **`subdiv_wrap` assigned instead of appending.**  `_subdivide_bridge_faces` is called once per
  patch component, so panel (f) showed one leg bridged and the rest bare.  Accumulate, then
  `np.vstack`.  Step 6 reads 430 → 11 614 across four legs, not 108 → 2 918 for one.
- **Overlaying the patch on the body z-fights** — they are the same surface.  Matching the duplicate
  body faces away by centroid fails (**0 of 7028**): `compute_bc_patch` re-triangulates.  Cut them by
  *proximity* instead (centroid within `0.35 h`, drops 7 030).  Do **not** solve it by offsetting the
  overlay along its normals: the same file is drawn in (e)-(g), where nothing is underneath, and the
  offset visibly inflates the patch.

### §5e-bis  The figure's final form, and what each choice cost

**Layout.**  Eight panels in ONE horizontal strip: (a) the cache taken before the first optimizer
step, then a break carrying an **"Optimized" arrow**, then (b)-(h) all within one later iteration.
Group rules and labels over each region, legend as a centred horizontal row underneath, `(h)` alone
carrying a smaller grey sub-line ("with consistent BC").  No per-panel statistic captions.

**Palette — the BC patch is the house green `#55A868`** (what `bc_region_id` and `diffused_nitsche`
use for BC), the zip strip is house blue `#4C72B0`, removal candidate `#a13d41`, base `#ebe6db`.
Validated before committing (`scratchpad/validate_palette.py`):

| pair | normal | protan | deutan | min CVD |
|---|---|---|---|---|
| red / green | 28.4 | 26.0 | 15.5 | **15.5** |
| green / blue | 22.0 | 21.2 | 20.6 | **20.6** |

Worst CVD ΔE 15.5 against a target of 8; worst normal-vision pair 22.0 against a floor of 15.  One
WARN: green is 2.85:1 on white, under the 3:1 guideline — discharged because every colour is named
in the legend, so identity never rests on the swatch alone.  **The boxes wear the patch's colour**:
they outline its extent, and leaving them a different hue creates a colour with no legend entry.

**Two box sets, not one.**  (a) draws each patch's *own* AABB (scale 1) — what the cache
establishes; (b) draws the same boxes scaled by `aabb_scale` — what selects the initial candidate,
which BFS facet rings then grow until valid (the box itself never grows; panel (b) is titled "Grow
by rings" for that reason).  Drawing the scaled box in (a) claims the workflow has already
decided how far to reach.

**Draw nothing that is not in the legend.**  Boundary-loop polylines were rendered in (a), (b) and
(c); with no legend entry they read as a third material ("what is the blue at the bottom?").  They
are now drawn nowhere.

**Cameras are per case and per group, and shared within a group.**  The bracket is a flat plate
(y-up), the table is z-up with its subject a leg seam near the floor — reusing one camera renders the
tabletop face-on.  Within the whole-body group **all panels must share one radius**: widening only
panel (b)'s frustum to fit its boxes made its table smaller than (a)'s and (c)'s.  Size the shared
radius to hold the boxes instead.

**Layout traps, every one of which shipped once before a guard caught it:**

- A **title band above the axes is paid for twice** — top margin for row 1, row gap for row 2.
- **Axes-position guards do not catch text overflow**, because text lives outside its axes.  Guard
  the drawn text: titles, sub-lines, legend, group labels, arrow label.
- **The common letterbox aspect decides the strip height.**  `min` (the tallest crop) lets the seam
  close-ups set the height and fill it; `max` (the widest) makes every panel shorter and scales the
  close-ups down into it.
- **Pin the group rule to the title tops, not to the title band** — widening the band for (h)'s
  sub-line floated the rules away from the titles.
- **Blanking a caption is not the same as deleting its data.**  The per-panel statistic strings were
  left populated after the captions were dropped; the moment any sub-line was rendered, every one of
  them came back.
- Guards need `bbox()` defined **before** the first block that uses it, not beside the assertions.

**Step 4 (cluster + bridge) is not in THIS figure.**  The table never reaches it — the removal
already leaves one loop per patch loop — so `bz_prep` writes no panel-4 VTK, the renderer skips it
and the compose step drops the cell rather than captioning a copy of (c).  It, and the other two
branches the ordinary path skips, are now their own figure: **§5e-ter**, `./build.sh bczipx`.

**The Y-bracket case, for the branches the table skips.**  It is built so the cached
patch has MORE loops (3) than the initial removal candidate (2), which is exactly the condition
`_validate` rejects — so step 2's BFS ring expansion must run (17 rings, 2 → 5 loops), and it
overshoots to one more loop than the patch has, so step 4's clustering runs too (5 → 3 loops, 4
bridge + 32 cap tris).  The table demo in the scratchpad is the *ordinary* path: everything 1:1, so
neither branch fires.  Keep both — they cover different halves of the algorithm.

**Two layout traps, both hit on the first cut of `bz_figure.py`:**
- **A title band above the axes has to be paid for twice** in the figure height — once as top margin
  for row 1, once as the row gap for row 2.  Leaving it out of `fig_h` clipped the row-1 titles clean
  off the canvas.
- **Axes-position guards do not catch text overflow, because text lives outside its axes.**  The
  original guards checked every panel's `get_position()` and passed while the titles were off the top
  and the footer ran past the right edge.  Guard the *drawn text*: `bbox(ax.title)`, `ax.texts[…]`,
  the legend and the footer, each asserted inside `[0, 1]²`.

**Coverage against the paper's seven steps — every branch now has a figure:**

| step | branch | figure | status |
|---|---|---|---|
| 1 | cache patch + nested-loop check | workflow (a) | ✅ 4 patches, 1 loop each |
| 1 | nested-loop **rejection** (workflow refuses) | — | ⬜ text only, see §5e-ter |
| 2 | scale box, BFS ring expansion | workflow (b) | ✅ |
| 3 | remove | workflow (c) | ✅ |
| 3 | **fan-cap an inner (nested) hole loop** | special (a), (b) | ✅ 112 cap tris |
| 4 | cluster + bridge + cap the smaller | special (c), (d) | ✅ 3 → 1, 8+32 tris |
| 5 | 1:1 match + zip | workflow (e) | ✅ |
| 6 | subdivide long bridges | workflow (f) | ✅ |
| 7 | Taubin smooth | workflow (g) | ✅ |

### §5e-ter  The special-cases figure: the branches the ordinary path never reaches

**STATUS: `bc_zip_special_cases.{pdf,png}` is built.**  `./build.sh bczipx` runs
`bzx_prep.py` → `render_bzx.py` → `bzx_figure.py` and deletes its intermediates (~30 s, GPU).  Four
panels in one row as two before/after pairs, each pair one fabricated geometry under one camera, a
rule + label over each naming what that branch DOES ("Cap a nested hole loop", "Cluster & bridge
surplus loops").  Titles only — no grey sub-lines, no statistics — parallel where the panels are
("Removal exposes a bore" / "Removal leaves three loops"), joined with `&` rather than a comma as in
`bc_zip_workflow`.  The labels carry NO step numbers: the mapping back to the paper's enumeration
belongs in the caption, and `Step 4 — …` on the figure only invites the reader to hunt for 2 and 5.

| case | geometry | what fires |
|---|---|---|
| A | short round bar clamped on its end face, **bored right through** | candidate leaves the bar's rim + the bore's mouth concentric inside it → `_cap_inner_hole_loops`, 112 tris |
| B | clamped round trunk **split into three struts** just inside the box, running back into a common head | 3 hole loops against the patch's 1 → `_cluster_and_bridge_hole_loops`, 8 bridge + 32 cap tris |

Both are asserted watertight, and `bzx_figure` refuses to compose unless the capture says both
branches actually ran.

**Step 1's rejection had a third geometry and it was CUT.**  A plate whose clamp was the annular
washer seat around a bolt hole makes `compute_bc_patch` raise, and the panel showed the cached
annulus exploded off the plate with its nested loop in red.  It came out because its outcome is a
*refusal*: there is no "after" to pair it with, so it sat in the figure as the one odd panel, in the
one geometry nothing else used, carrying the only green on the page — a key entry for a colour drawn
nowhere else.  That branch belongs in the text.  (Recipe if it is ever wanted back: frozen nodes in
an annulus `r ∈ [0.17, 0.30]` one grid layer under the top face, so the patch is an annulus that
does NOT touch the hole — a clamp reaching the hole wall grows down it, which puts the inner loop at
a lower z than the outer and the AABB test never fires.  Draw it by calling `compute_bc_patch`
twice: once with `_check_no_nested_loops` monkey-patched to a no-op for the geometry, once for real
for the refusal.)

**Nesting is tested by AXIS-ALIGNED bounding boxes, and that dictates the geometry.**  The one
non-obvious constraint in the whole figure, and it kills the obvious designs:

- A bump escaping the removal box through a face has its loop *beyond* the outer loop on that axis,
  so it is never classified as nested.  What works is a feature escaping *inward* and concentric
  with the outer rim — a pocket, a bore, a counterbore.
- An axis-aligned bar is cut by the (axis-aligned) box in a rim whose AABB is degenerate along the
  axis, and the bore's mouth then fails containment by a fraction of an element for a reason that
  has nothing to do with nesting.  **Tilt the bar 18°** and both rims become oblique ellipses about
  one axis, where containment holds by construction.  Aligning the mesh rings with the cut plane
  instead would make both rims exactly planar — and then they sit at *different* ring planes and the
  test fails outright.  The tilt is what buys the slack, and the slack is worth more than exactness.

**Case A is a THROUGH bore in a SHORT bar, and both words are load-bearing.**  A blind pocket orphans
an island sealed inside the solid, where the panel after the cap has nothing to show — capping it is
correct, but a figure cannot draw it.  Through the bar, (a) sees daylight and (b) sees a violet plug:
the whole before/after in one glance.  The bar is then short enough that
`2·S_BAR·tan(view angle) < bore diameter` (0.38 long, 0.17 across, camera 19° off the face normal),
or the near wall occludes the far opening and the hole reads as a rod again.  Nesting is indifferent
to which it is, so the choice is purely what the panels can show.

**Case B is a trunk splitting into three struts that FLARE back into a head.**  Design constraints,
all of them load-bearing:

- **Not equilateral.**  The two fat struts are the closest pair, so they are the ones bridged, and
  the thin one has the smallest perimeter, so it is the one capped — both branches land where the
  eye expects.  Equidistant centroids would leave the choice to `argmin`.
- **A head, not three rounded tips.**  The removal leaves three separate pieces either way and the
  algorithm is indifferent, but three rods floating in space is not a *part*.
- **Flared, not butted.**  A boolean union of a rod into a trunk leaves a hard intersection circle,
  which no optimiser would produce; CSG cannot fillet, so `_tube` takes an `r_end` and grows the rod
  over the last `flare` of its run on a smoothstep — zero slope where the straight part ends, so no
  kink where the blend begins.  Bounded by the head: `offset + r_end <= TRUNK_R`, or the blend
  bulges out of the body it is supposed to run into.
- Struts short (0.26 of run, 0.06 of it flare) and the head close, or the panel is mostly empty rod.

**Neither geometry is marching cubes — both are hand-built uniform CSG meshes**, and for two
independent reasons.  `bzx_prep.set_grid()` still supplies the background node lattice, because BC
nodes are background nodes and `radius = 2h` is the production rule; only the *surface* changed, and
`compute_bc_patch` / `stitch_bc_patches` cannot tell.

- **The bridge is one loop EDGE wide.**  `_bridge_two_hole_loops` (the Fuchs-Kedem-Uselton
  dynamic program, spelled out — the figure and these notes say "bridge", since initials name
  the algorithm rather than the object a reader is looking at) spans the closest edge pair and
  subdivision only lengthens the strip, never widens it — so an MC loop, whose edges run anywhere
  from ~0 to h, hands the bridge whatever it drew.  It drew 0.003 (12 % of h) and the strip tapered
  to a point: a hairline no camera, grid or zoom can rescue, because the width is not h, it is a
  sample from MC's edge-length distribution.  Even segments give one edge length and a proper strip.
- **An MC cylinder's silhouette is scalloped** at the element scale, and these panels are almost
  entirely circles.  Raising the grid to 257 fixed the rims and not the impression; a tessellated
  tube is round by construction at a twentieth of the triangles.

`bzx_prep._tube` builds them, and every line of it is a trap that was hit:

- `trimesh.creation.cylinder` is ONE quad ring along its whole length, and a face is a removal
  candidate if any *one* vertex is in the box — an unsplit strut is removed end to end (874 of 1174
  faces went, and the panel had nothing left to show).
- `subdivide_to_size` is not the fix: it splits triangles independently and returns a non-watertight
  mesh.  Build the rings directly.
- The hemisphere tip must exclude both `phi = 0` (repeats the last barrel ring) and `phi = π/2`
  (collapses a whole ring onto the pole); a fan closes it.
- `fix_normals()` before any boolean — manifold3d refuses anything not consistently wound with
  "Not all meshes are volumes!".
- Flat ends are closed with CONCENTRIC RINGS, not a fan.  A fan is a valence-`sections` pole, and
  the Taubin pass below pulls a flat face containing one into a warped cone — radiating ridges right
  across the head's front.  Rings keep every interior vertex at valence 6.

**The struts' section count IS the bridge's width — that is the one knob that trades.**  32 sections
puts the loops at "circle" and the strip at 0.0137, half a millimetre printed: thin, but uniform and
solid, which is a different thing from the tapered hairline.  Past that the bridge goes back to
invisible.  Case A has no such coupling (96 sections, rings 0.0035), and the head carries its own
finer tessellation (72 sections) because it is the biggest thing in the panel and nowhere near a
loop.  Ring spacing is what makes the LOOPS smooth: the candidate boundary follows mesh edges, and
an oblique cut plane zigzags it between adjacent rings by exactly the ring spacing — 0.012 is 0.6 mm
on the page against a 0.34 mm loop tube, visibly serrated; 0.0035 puts it far under.

**Rendering, in the order the problems appeared:**

- **Solid colour per mesh, not the `stage` LUT.**  With `ColorBy(CELLS, stage)` on, VTK paints
  backfaces from the scalars too and the backface material below does nothing.  One material per
  file; step 4's bridge and cap are separate files for the same reason.
- **Backfaces take the same colour dimmed to 0.82** (base meshes only).  Every panel looks into
  something the removal opened, and ParaView casts no shadows in this build (`UseSSAO` does not
  exist on the 5.13 view), so a lit bore wall reads as a solid rod standing in a tube.  *Darkening*
  rather than dimming turned the fan cap black; 0.74 left the cut struts as two dark discs.
- **`BackfaceRepresentation = "Surface"` is a PLAIN surface — it drops the edges**, which on the fan
  cap is the entire point of drawing it.  Overlays stay on `"Follow Frontface"`.
- **Edges on the NEW triangles only** (`bzx_a_cap`, `bzx_b_bridge`, `bzx_b_cap`), drawn over the
  *unchanged* before-panel base file, so a pair differs by exactly what the step added.
- **`GenerateSurfaceNormals` at a 45° feature angle.**  Legacy polydata carries no normals, so VTK
  shades every triangle flat and a 32-gon rod reads as a 32-sided prism — the faceted look was
  SHADING, not geometry (that rod's silhouette is off a circle by 0.0013, i.e. 0.05 mm on the page).
  45° rounds every barrel while keeping the genuine creases: a cut face meets its barrel at 90°, the
  flare at ~5°.
- **Loop tubes at 14 px of an 1800 px render** — 0.34 mm printed.  At the 6 px they started at, the
  panels read as though the loops had not been drawn, and the loops are the subject of all four.
- **`fit_r = 4.55`.**  A 25° FOV needs 4.51 r; every 0.1 above that is white space in every panel.

**Taubin over the whole display mesh with the loops pinned.**  `bzx_prep.taubin` is production's own
λ/μ block (0.5 / −0.53) run over the base mesh instead of just the bridge strip, pinning every drawn
loop vertex.

**The two cases need wildly different sweep counts — 10 for B, 120 for A — and that is structural,
not taste.**  Case B has union SEAMS, which are high-frequency and relax in 10 sweeps into the
fillets that make it look cast rather than assembled.  Case A is a cylinder minus a cylinder:
developable everywhere, with nothing for a smoother to bite on, so the same 10 sweeps move it 0.0014
(0.7 % of the bar radius) and change *nothing visible*.  Its only creases are the bore mouth and the
crater-floor edge; 120 sweeps move 0.0020 and relax those into a cast-looking funnel, and 300 move
0.0035 and dish the crater floor into a bowl.  If a case looks untouched by smoothing, check whether
it has any creases before turning the count up — and measure: `bzx_prep` prints the max and mean
displacement per case for exactly this reason.  It needs **inverse-edge-length weights** (the scale-dependent umbrella): the
uniform graph Laplacian drags vertices along the long direction of a 0.022 × 0.014 quad, and at 6
iterations the head rippled, at 40 the part melted.  It also needs the ring-triangulated flat caps
above.  This is a DISPLAY pass — it runs after the workflow, so the drawn surface is no longer
bit-for-bit the one `stitch_bc_patches` was handed.  Everything the figure *claims* (loops, caps,
bridge) is pinned or built on pinned vertices, so nothing it asserts moved.  `iters=0` disables it.

**Layout — size the PANEL, then size the STRIP to its content.**  Dividing the page width among
however many panels there happen to be couples panel size to panel count: dropping this figure from
five panels to four made every panel 30 % bigger and the strip half an inch taller, for no reason
anyone chose.  A figure placed at `\textwidth` prints at its own aspect ratio, so panel width
*relative to* `FIG_W` is printed panel size.  `bzx_figure` fixes `PANEL_W = 1.05 in` (the scale
`bc_zip_workflow` settled on), makes every gap its own grid column with `wspace=0`, and sets
`FIG_W = 5.35` — the width the panels and gaps actually need, rather than the house 7.16 with a
quarter of the figure empty.  **It follows that this figure is placed at its own width, not
stretched to `\textwidth`**, or the panels come back 34 % larger.  The within-pair gap is set by the
TITLES, which are wider than the 1.05 in panels they sit over.

**Four key entries: bridge, cap, open loop, nested loop.**  Not the cream base surface — it is
everything no step has touched and no title refers to it.  Not the BC patch either: it is not drawn
in any of these four panels (in (c)/(d) it sat between the camera and the cut struts, hiding the
bridge and the cap), and a key entry for a colour that is nowhere on the page sends the reader
hunting for it.

**Several marching-cubes contexts in one process makes Warp's shutdown race Torch's** — the script
SIGSEGVs *after* its last line of real work about half the time, which under `set -e` aborts
`build.sh` long after `bzx_prep.py` has actually succeeded.  Everything is on disk and every failure
mode is an assert, so the script ends with `os._exit(0)`.  (Kept even though both cases are CSG now:
`extract_surface` may come back.)

**Status: geometry settled, panels not built.**  The demo lives in scratchpad `bz_same.py` (+ `rq.py`
to render, `dihed.py` to measure); `illustrations/bczip/bz_prep.py` still calls `stitch_bc_patches` with
defaults and must pick up the settled parameters below.

The subject is `stitch_bc_patches` in `core/remeshing.py`: cache the clamped surface off the
geometry, let the optimiser change the body, then remove the changed surface near the BC and zip the
exact cached patch back on.  **Do not run the cantilever to get one** — build the pair from an
unmodified CC0 Poly Haven table (see memory `cad-source-polyhaven`) and an SDF offset that stands in
for an optimiser step.

**The demo is a round trip, not a pairing of two CAD models.**  Earlier attempts paired a real table
with a fabricated body so the patch loops would have to find a different object.  That is an edge
case, it burns days on `MAX_RINGS` and `branching-out-degree` failures, and it is not what the
ordinary path looks like.  Cache off the table, erode the same table, stitch back: patch is 4
components × 1 loop, matching is 1:1, no bridge, no cap, watertight.  Edge cases (ring growth,
bridging, capping) are separate models.

**Three constants decide whether it looks like anything** — none discoverable by reading the
function, all measured (details in memory `bc-zip-geometry-constraints`):

| Knob | Value | Why |
|---|---|---|
| cut plane | **z = −0.40** (bottom 10 %) | the lower the cut, the more leg the ramps get |
| `aabb_scale` | **3.0** (production default 1.1) | sets the gap between the two zipped loops |
| `subdivision_size_ratio` | **3.0 — the production default, leave it alone** | the strip must not be refined finer than the mesh it joins |
| `smooth_order` | **1** (Taubin, the default) | see the fairing note below — order ≥ 2 exists but is not what the figure ships |
| `smooth_iters` | **600** (default 10) | Taubin passes over the strip interior; free at this size |
| `taubin_mu` | **−0.51** (default −0.53) | the knob that actually smooths — pass-band 0.039 not 0.113 |
| `smooth_cg_iters` | **20000** (was 200) | only matters at `smooth_order ≥ 2`; the cap, not the tolerance, governs convergence |
| erosion `EPS` | **off (0)** | the two loops are the same size — the strip is pure twist |
| twist `BZ_TWIST` | **45°** about each leg's own axis, **uniform through the band** | makes the transition surface twisted, not just tapered |
| twist ramp | smootherstep, flat to `z = −0.28`, zero at `z = +0.23` (length **0.51**) | shorter — bounded by where the legs stop being isolated |

**The zip band is `z ∈ [−0.390, −0.263]`; whatever is applied to the leg must be flat across all of
it.**  Key the ramps off `CUT` in the script (`FLAT_TO = CUT + 0.12`) rather than hard-coding them,
so moving the cut plane hands the ramps the extra leg automatically.  That is the one hard constraint on the ramp — everything above the band is free, and the
longer that fade runs, the smoother the leg reads.  A ramp that still varies inside the band makes
the leg thicken (or unwind) going up while the strip flares going down, and they meet in a V.

**Turning the erosion off is a direct test of `crease = arctan(ΔR / gap)`, and it passes.**  With
`ΔR = 0` the crease should vanish and only the twist should remain — measured, the seams go
8.6° / 11.4° → **3.3° / 5.0°**, both now at or under the mesh's own 7.0° faceting floor.  **A pure
rotation is something the strip absorbs smoothly; a change in cross-section is not.**  Keep this in
mind when reading any seam number from an older run here: most of what was being measured was the
radius change, not the zipper.  Pure-twist figures: base/patch/zip = 101 314 / 7 028 / 11 614,
13.9 % removed, strip p90 6.3°, 0 degenerate, watertight.

**Find the ramp's ceiling by slicing the SDF and counting connected components per slice — not by
eyeballing the render.**  The apron *looks* like it starts around `z = +0.16`; the slice count says
the legs are still **four isolated columns at +0.225**, the scalloped skirt begins at `+0.24`, and
they merge into one body at `+0.27`.  The eyeball guess cost 39 % of the available run, and that
showed up as a sharp helical fold at mid-leg where the twist rate peaked.  `bz_room.py` does the
measurement.  The two ramps then want **different** lengths: the twist must die below `+0.23`
  because a per-leg rotation applied to shared apron material shears the skirt apart.  (The erosion
ramp had no such ceiling — a uniform inward offset tears nothing — and only stopped at `+0.30` to
stay off the scallop teeth, which are ~2 cells thick and would visibly thin.  It is off now.)

**Moving the cut plane down costs loop gap, and you must pay it back with `aabb_scale`.**  The gap is
`(aabb_scale − 1)/2` of the ***patch's*** height, so a bottom-fifth cut — half the patch a lower-leg
cut gives — halves the band at an unchanged scale.  Measured, at `CUT = −0.30`:

| `aabb_scale` | band | crease (base / patch) | zip faces |
|---|---|---|---|
| 1.3 | 6 cells | 24.6° / 28.9° | 26 404 |
| 1.75 | 13 cells | 13.4° / 15.1° | 73 458 |
| 2.2 | 19 cells | 7.8° / 10.4° | 81 568 |

and again after dropping to the bottom 10 % (`CUT = −0.40`), where the patch halves again:

| `aabb_scale` | band | crease (base / patch) | zip faces |
|---|---|---|---|
| 1.75 | 7 cells | 22.1° / 24.8° | 26 958 |
| 2.4 | 12 cells | 13.1° / 15.6° | 55 428 |
| **3.0** | **17 cells** | **8.7° / 11.7°** (p90 12.5° / 21.9°) | 77 192 |

(measured with the leg erosion still on and `subdivision_size_ratio` overridden to 0.75 — both since
reverted; at the shipped settings the same `aabb_scale = 3.0` gives 3.3° / 5.0° and 11 614 zip faces)

**How much subdivision the bridge actually gets (measured with `bz_subcount.py`).**  The zipper emits
a **single quad-strip ring — 76 to 78 triangles per leg**; everything else is
`_subdivide_bridge_faces`.  Per leg:

| ratio | rounds | growth | faces / leg |
|---|---|---|---|
| **3.0** (default) | **3–4, all four legs converge** | ×3.00 each | 76 → **2 052** (×27) |
| 0.75 (don't) | 5 useful + 5 wasted; **two legs never converge** | ×3.00, then ×1.02, ×1.00… | 76 → 18 830 (×248) |

×3, not ×4, because the 2-marked-edge template (corner + quad split) dominates on a quad strip.

**Subdivision equalises the strip's face AREA but not its SHAPE, and no amount of refining fixes
that.**  Measured at the shipped settings:

| | median area | median edge | aspect ratio (med / p90) |
|---|---|---|---|
| base | 2.81e-5 | 0.00754 | 1.46 / 4.47 |
| patch | 2.84e-5 | 0.00757 | 1.43 / 3.57 |
| **zip** | 1.01e-5 | **0.01228** | **6.50 / 11.93** |

The strip's triangles are slivers with a *longer* median edge than the base despite a smaller area.
The cause is structural: `_subdivide_bridge_faces` never marks **external** bridge edges (that is what
keeps base/patch neighbours free of hanging vertices), so it can only ever refine the strip along one
direction.  Refining harder changes the resolution, not the ratio of the strip's two directions — a
grading scheme that also split *across* the strip is what would.  This is the same restriction that
makes an over-refined strip hit `max_iter`, seen from the other side.

**The tell for over-refining: the loop stops running out of work and starts running out of
iterations.**  At 0.75 two of the four legs exit on `max_iter = 10` with edges still 1.8× the
threshold (0.0106 against 0.0058).  Those leftovers are *external* bridge edges, which the algorithm
deliberately never marks so neighbouring base/patch triangles never get a hanging vertex — they
cannot be split, so the loop can only terminate on the cap.  **Don't override the default**: 9× the
strip faces bought 7.1° → 4.6° of strip p90, while one character of `taubin_mu` bought 13.4° → 7.1°.

Read `aabb_scale` in **cells of band height, never as a ratio** — what a value buys depends on the
patch's height, so it has to be re-tuned every time the cut plane moves.  At 1.75 the seam is better
than the `CUT = −0.10` figure managed (16.5° / 19.5°) *and* the ramps are half again as long.

**Sliver faces (area < 1e-12) at the removal front come and go with the cut plane, not with tuning.**
At `CUT = −0.30` there were exactly 4 — one per leg — at every `aabb_scale` from 1.68 to 1.80, because
the BFS removal front landed on a grid plane where the marching-cubes surface degenerates.  At
`CUT = −0.40` there are **0**.  Watertightness is never affected (0 open / 0 non-manifold / 1
component), but sliver normals are NaN, so **exclude them before computing any dihedral statistic** or
the whole figure comes back `nan`.

**`FLAT_TO` may overlap the top of the band a little without harm.**  Smootherstep has zero slope at
both ends, so a ramp that starts varying just below the band top changes nothing measurable —
`CUT + 0.12` and `CUT + 0.15` gave *identical* seam angles at `aabb_scale = 3.0`, where the former
overlaps the band by 0.017.  Use `CUT + 0.12`: it is the longer ramp and it produced no slivers.

**Twist each leg about its OWN axis, not the table's.**  A twist about the table axis swings the legs
sideways; the exact patch and the leg above it then no longer overlap and there is nothing for the
zipper to bridge.  Leg axes come from k-means on the xy of the frozen leg cross-sections (measured
here: radius 0.292, at 0°/90°/180°/270° — *not* the diagonals a crude quadrant split suggests).
Apply the twist by re-querying the BVH at rotated points, not by resampling the SDF grid: trilinear
resampling smears the leg's corners, which are exactly the feature the twist is meant to show.

- **Taubin cannot smooth the seam; only the loop gap can.**  `stitch_bc_patches` pins every bridge
  vertex shared with a non-bridge face, and *every* vertex the zipper emits sits on either the hole
  loop or the patch loop — so the only movable vertices are subdivision midpoints, and the default
  `subdivision_size_ratio=3.0` creates exactly **one row** of them (measured 339 free vs 339 pinned).
  Both rings are Dirichlet-pinned with no tangent constraint, so the relaxed strip is the tension
  surface between them: for a change in cross-section, a **cone with a G0 crease at each end**.  No
  smoothing parameter and no wider *band* removes it.
- **The crease angle is `arctan(ΔR / gap)`, and the gap was hard-coded.**  B.1 built the removal box
  as `1.1 ×` the patch AABB, putting the hole loop only `0.05 × patch height` above the patch loop —
  4 cells here against a 2.9-cell ΔR, i.e. 45°.  **This figure needed a production change**:
  `aabb_scale` is now a keyword argument of `stitch_bc_patches` with default 1.1, so behaviour is
  unchanged and all three existing callers (keyword-only) are unaffected.  Measured sweep:

  | `aabb_scale` | loop gap | crease (base / patch ring) | zip faces |
  |---|---|---|---|
  | 1.1 | 0.030 · 4 cells | 45.1° / 46.1° | 9 544 |
  | **1.3** | **0.075 · 10 cells** | **18.3° / 18.4°** | 26 676 |
  | 1.8 | 0.173 · 23 cells | ~2° | 115 102 |

  Watertight (0 open / 0 non-manifold / 1 component) at every setting.  **Stop at 1.3**: past it the
  strip stops *joining* the leg and starts *replacing* it — at 1.8 the band is 23 cells tall and the
  zip outweighs the base mesh, so the strip colour swallows the part it is supposed to annotate.
- **`smooth_order` — G1/G2 fairing of the strip (opt-in, and NOT what this figure ships).**  The
  default Taubin pass is a *membrane* energy pinned on one ring, so the strip is G0 at both seams no
  matter what the coefficients are.  `smooth_order=2` solves the thin-plate problem `Δ²x=0` with two
  clamped rings per side (G1), `3` solves `Δ³x=0` with three (G2); `1` is the default, is
  **bit-identical** to the old path (verified: max vertex move 0.0), and is what both the optimiser
  and the figure run.  Order ≥ 2 stays available for a figure that needs a clean seam, but it is a
  deliberate opt-in, never a default — the G0 crease is cheap and stable, the fairing is neither.
  Two things that look equivalent and are not, both found by measurement:
  - **Cotangent weights, not uniform `D−A`.**  Uniform weights ignore the strip's real edge lengths and
    drag vertices through the slivers the zipper leaves — 178° dihedrals, i.e. flipped faces.  The
    operator is `L (M⁻¹L)^(order−1)` with cotan `L` and barycentric mass `M`; `M` diagonal ⇒ symmetric
    ⇒ the free-set submatrix is SPD ⇒ plain CG is legal.  Clamp negative cotans (obtuse triangles) or
    it stops being PSD.
  - **The biharmonic conditions like h⁻⁴, so unpreconditioned CG silently does nothing.**  At order 2
    the strip was still sitting on the zipper's original cone after **20 000** iterations — it reads as
    "the feature has no effect", not as a failure.  Assemble `A` once by sparse-sparse product (also
    the only way to get `diag(A)`), then Jacobi-precondition: 200 iterations now beat 20 000 without.
    All three coordinates ride one `spmm`; poll the residual every 25 iterations, not every one, or the
    device syncs dominate.
  - **An under-converged biharmonic solve is indistinguishable from a feature that does nothing.**
    This cost the most time here, so read it before touching the solver.  Both seam rings stay
    *pinned* — the clamped condition takes its tangent from the ring's neighbours, which are already
    in the mesh — and that is sufficient, but only once the solve actually converges.  The iteration
    cap governs, not the tolerance:

    | `smooth_cg_iters` (order 2) | 200 | 2 000 | 20 000 |
    |---|---|---|---|
    | patch seam | 16.6° | 5.5° | **5.2°** |

    At the old default of 200 the strip sat on the zipper's original cone and every knob looked inert.
    **CG is now only the fallback**: the strip is local (13 214 free vertices out of a 60 k-vertex mesh,
    281 k nnz), and at that size the biharmonic is a *direct*-solve problem — sparse LU factors it in
    ~150 ms to a 1e-13 residual, so the whole fairing call is **0.285 s** against ~10 s of CG for a
    worse answer.  Not an AMG job: classical AMG targets 2nd-order elliptic operators, and at 13 k
    exact-in-0.3 s there is nothing left for an iterative solver to win.
    The default cap is now 20 000; it exits on `smooth_cg_tol` well before that, and the whole
    `stitch_bc_patches` call still runs in ~13 s.  Order 3 needs ~100 000 to converge and lands at
    11.6° / 5.0°, i.e. no better than order 2 — if you use the fairing at all, **use 2**.
  - **A better CG preconditioner was built, measured, and then deliberately removed — don't rebuild
    it.**  The right preconditioner for `A = L(M⁻¹L)^(order−1)` is the factored one, `M⁻¹ ≈ L⁻¹ M L⁻¹`,
    with each inner `L⁻¹` a *fixed-degree* Chebyshev polynomial in `D^-1/2 L D^-1/2` (fixed degree is
    what keeps it a linear, symmetric operator, so outer CG stays valid; `S W S` is then SPD by
    construction).  Never precondition the 4th-order operator directly — precondition it with the
    2nd-order one it factors into.  Measured on the real system (13 214 unknowns), against warp's
    built-in `cg`:

    | preconditioner | outer iterations | error vs sparse LU |
    |---|---|---|
    | none | 20 000 (cap) | 0.31 — never arrives |
    | Jacobi | 20 000 (cap) | 5.2e-6 |
    | factored Chebyshev | **30** | 5.6e-8 |

    Bounds came free: `λ_max ≤ 2` by Gershgorin, `λ_min` from 50 Lanczos steps whose smallest Ritz
    value bounds it from *above* — the safe side (underestimating it widens the interval and wrecks
    the polynomial: κ̂ = 50 000 against a true 1 988 took the outer count 33 → 1 121).  It worked, and
    it was still the wrong thing to ship: **388 ms against sparse LU's 160 ms** at the sizes this code
    actually sees, and order 3 needed a degree rule the order-2 tuning does not supply (65 s at the
    auto degree).  The scalable path only pays off at a strip size the figure never reaches.  Reverted
    to LU-then-Jacobi-PCG.
  - **Taubin CAN smooth the strip's interior, and `taubin_mu` is the knob — but only once the strip has
    an interior.**  The "339 free against 339 pinned" measurement below was taken at
    a geometry whose band was only 4 cells tall.  With the **17-cell** band this figure now uses, the
    strip has **4 008 free vertices against 308 pinned at the default ratio** — the room comes from
    the band being tall, not from refining it.  Its pass-band frequency is `1/λ + 1/μ`; the default
    `(0.5, −0.53)` gives 0.113, and `−0.51` gives **0.039** — a far more aggressive low-pass.
    Measured on the strip's own dihedral p90:

    | iters / μ | −0.53 | −0.51 |
    |---|---|---|
    | 10 | 13.4° (default) | — |
    | 150 | — | 7.5° |
    | 600 | — | **7.1°** |

    (On the over-refined 0.75 strip the same sweep ran 14.8° → 4.6°, and 600 iterations at the
    *default* μ came out **worse** than 150 — over-iterating a wide pass-band.  **Fix μ before
    reaching for more triangles.**)

    600 iterations on 38 k vertices leaves the whole run at 9.7 s, so it is free.  Taubin is
    volume-preserving, so the collar does not shrink: band height is 0.1275 at every setting.  **The
    seams move by 0.1°** — the same conclusion from the other side, that the crease is geometry
    (`aabb_scale`), not smoothing.  Iterations alone plateau around 7.8°; μ is what breaks through.
  - **The Taubin path cannot be given the same treatment — the information is unreachable, not merely
    unused.**  This is about the *seam*, not the strip interior (see above).  Measured on the
    assembled mesh at `subdivision_size_ratio = 3.0`: the strip's free vertices have **0** one-ring neighbours
    outside the strip (13 506 neighbours, every one of them a bridge vertex).  The pinned seam ring
    shields them completely, so a 1-hop membrane operator's stencil never touches base or patch
    geometry and extending its graph past the seam is provably a no-op.  Only a ≥2-hop operator
    (`smooth_order ≥ 2`) reaches across.  Don't spend time trying to fix order 1.
  - **What is left at the patch seam is the patch's own triangulation.**  Median is 5.2°, but 4 edges
    of 192 sit at ~100°, and every one of them is against a patch face **5–10× the strip's face area**.
    Coarsening the strip to match does not help (`subdivision_size_ratio` 0.75 → 1.5 → 2.5 keeps 4, 4,
    11 such edges while the median degrades 5.2° → 7.9° → 12.8°), because the offending triangles are in
    the exact cached patch, which by definition cannot be re-meshed.  A few visible spikes on that seam
    are the floor of the method, not a tuning failure.  Refining the strip only widens the fan it makes
    into each coarse patch vertex.

    **That sweep once read as an argument for `subdivision_size_ratio = 0.75`.  It is not — do not
    override the default.**  Those numbers (0.75 / 1.5 / 3.0 → 13 214 / 3 893 / 1 172 DOFs and
    5.2° / 7.9° / 14.3°) were taken at `smooth_order = 2` on a 4-cell band, where the fairing solve
    genuinely needs rows near the seam.  On the shipped order-1 geometry with a 17-cell band, the
    default ratio gives 4 008 free vertices and a strip p90 of 7.1° — against 4.6° for nine times the
    faces.  **The real lever was `taubin_mu`** (13.4° → 7.1°), and reaching for triangles first cost
    an unnecessary 9× in mesh size, pushed two of four legs past `max_iter`, and quietly changed a
    production default in `shape_opt.py`.  What is still true: `_subdivide_bridge_faces` uses one
    global threshold, so grading the refinement toward the two rings remains the open improvement if
    a future figure ever does need a finer strip.
  - **The two seams do not converge to the same place**: patch **5.2°** (below the mesh's own 8.6°
    faceting floor — effectively gone), base **14.0°**.  The base ring is the ragged BFS removal front,
    so the tangent data it supplies is noisier than the patch's BC-selection boundary.
  - `smooth_base_rings` also exists — it releases N rings of *base* surface so the fillet runs into the
    leg, taking the base seam to 3.3°.  **Not needed, and not used here**: it works by letting the base
    bend rather than by making the strip meet it properly, and the exact patch can never be released
    that way, so it fixes only the seam that was already fixable.  Default 0.  (If it ever is used:
    seed the growth from the seam ring — growing from the strip's *interior* never fires, since no base
    face touches an interior vertex.)  **Order 3 is worse than order 2** (16.1°, p90 31.7°, ~750 ms):
    the clamped data is itself ragged — the removal front zigzags by ~h and the neighbouring
    marching-cubes surface facets at 8.6° p90 / 130° max — and higher order propagates that boundary
    noise harder.  **Use 2; 3 is below the noise floor of its own boundary conditions.**  Neither
    reaches 0°, and no fairing order can: the seam *angle* is geometry (`aabb_scale`), the order only
    decides whether the corner is a crease or a fillet.
- **Whatever is applied to the leg must be uniform across the band** — erosion, twist, anything.
  Where it still ramps there, the leg thickens (or unwinds) going up while the strip flares going
  down and the two meet in a V — 27° at the top ring against 7° at the bottom.  Flattening the ramp
  through the band made it symmetric.
- **Do not "fix" the seam by removing the size difference *behind the reader's back*.**  Ramping the
  erosion to zero above the band drives both seams to 0.2–0.3° and looks immaculate — and, done
  silently to make a render prettier, destroys the figure, because the two loops become nearly
  identical and the zipper visibly has nothing to do.  A sharp-looking seam is a signal to widen the
  gap, never to quietly shrink ΔR.
  **This is not an argument against `EPS = 0`, which is what the figure now ships.**  The difference
  is that the twist replaced the radius change as the thing being illustrated: the two loops are the
  same size *and still genuinely different*, so the zipper has as much to do as before (107 faces per
  leg, against 77 with the erosion on).  The rule is about hiding the subject, not about ΔR.
- **Measure the seam, don't eyeball it.**  `dihed.py` reports median/p90 dihedral across
  `zip|base` and `zip|patch` edges, against the `base|base` faceting floor as the reference — the
  render cannot distinguish an 18° crease from a colour boundary.  Asymmetry between the two rings
  is the diagnostic that the *body* is wrong, not the zip.
- **Selecting BC nodes: filter by the SDF, not by a box.**  A grid box around the clamped region was
  2.1 M nodes and `compute_bc_patch` tried to allocate 447 GiB.  `pts[(sdf < 0) & (pts[:,2] <= CUT)]`
  gives 27 k.
- Watertightness must be checked by **edge degree** (every edge in exactly 2 faces), not
  `extract_boundary_loops_strict`.  Base meshes go through `_close_open_loops` first.

### §5f Forward+backward optimisation loop (learned building `shape_opt_loop`)

The inverse-problem figure: a rectangular loop of five panels (a-e) plus a design-evolution strip (f)
at the bottom (redesigned from the old 8-panel two-row form).  Loop: top row (a) design octree+SDF →
(b) diffused-Nitsche solve; a single right-angle down the RIGHT edge (labelled "Compute objective"
above / "Backward propagation" below its long horizontal) into (c) sensitivity ∂J/∂SDF; bottom row
(c) → (d) SDF gradient-descent update → (e) marching cubes & restore BC; and a single right-angle up
the LEFT edge (labelled "Extract octree" / "Reinitialize SDF") back into (a).  **Displayed letters are
consecutive a-e; the render/POS tags stay {a,b,d,e,g}** (so_figure's TITLES maps d→c, e→d, g→e) — the
old (c),(f),(h) panels are gone and the objective/volume trajectory is a SEPARATE figure
(`so_objective.{png,pdf}`).  (a,b) sit close at the centre-top; (c,d,e) span the middle; (f) is a full
width row across the bottom.  Three stages `so_prep.py` → `render_so.py` → `so_figure.py`, wired as
`build.sh shapeopt`; the prep IS a real `ShapeOptTrimmed` run — now to CONVERGENCE (`N_ITERS 101`, the
objective settles into a steady 2-iter limit cycle, about 4–5 min on the current A6000), the loop panels captured at
`CAPTURE_ITER 40` (the compliance turn).  `so_prep.py` is the canonical cylinder→table blank §5d uses.

- **Show the design at the COMPLIANCE TURN, not the raw blank or the converged part.**  The volume
  penalty is one-sided (softplus, active only above target, see §5d), so while volume > 50% it removes
  material and raw compliance RISES; once volume hits the target the penalty switches off and compliance
  DECREASES.  That peak (~iteration 40, ~50 %) is the legible round-top-on-four-legs table worth
  showing.  `so_prep.py` writes `so_traj.txt` (raw SE + volume per iter) so the SEPARATE `so_objective.py`
  plots the hump and marks the shown iter; find the turn from a no-capture trace (`SO_CAPTURE_ITER=999`)
  first, then capture.
- **The exact zip (`sdf_bc_fix_method="zip"`) is fragile; three things make it run every remesh.**
  (1) Load the WHOLE top, not a central disk — a disk cap gets pinched off ("BC region is being
  isolated") when material is removed around its perimeter; the full tabletop stays attached to the legs.
  (2) Small BC patches leave bowtie/pinch vertices that fail `extract_boundary_loops_strict` with
  "branching-out-degree" (the §5e failure) — now FIXED in production: `compute_bc_patch` grows the patch
  face set by `grow_rings` (default 2, config `bc_zip_patch_grow`, wired through `_init_bc_graft`);
  morphological CLOSING does NOT work (erosion re-opens the pinch), only pure dilation.  (3) That lets
  the small feet stitch at depth 7; without the grow the depth-7 floor is ~0.22.
- **The feet are REAL geometry, not a selection on a flat disk.**  Four short cylinders (`FOOT_R 0.24`
  radius, `FOOT_H 0.22` protrusion below the disk, `FOOT_OV 0.06` overlap up into it) are
  `trimesh.boolean.union`-ed onto the cylinder — a *watertight* union, so `manifold3d` must be installed
  (`pip install manifold3d`; concatenation would leave buried overlap faces that stamp a spurious near-0
  level set inside the ±0.01-truncated SDF band at the foot/disk crease → MC carves a hole).  Because the
  feet protrude to the mesh's lowest z, ONE percentile selector `LowestSelector('z', fraction)` grabs all
  four at once and keeps grabbing them as the shape evolves (frozen feet stay lowest).
- **Freeze only a SHORT band at the very bottom of the foot; let the optimizer build the leg above.**
  This is the arrangement that finally read right.  Everything from the freeze boundary up is where the
  ugliness lives (see next bullet), so keep that boundary LOW: freeze just the foot cap plus `FROZEN_H`
  (0.02, ≈ ⅔ cell) of wall.  `so_prep` derives `FOOT_FRAC = ((Z_BOT-FOOT_H+FROZEN_H) - (Z_BOT-FOOT_H)) /
  (Z_TOP - (Z_BOT-FOOT_H))`.  The tall `FOOT_H` is *scaffolding* — it exists only so the frozen band sits
  far below the flat disk bottom (`z=Z_BOT`); the optimizer reshapes the un-frozen upper foot into the
  leg, adding the volume it needs, as one continuous surface.  Do NOT freeze up to the disk: the frozen
  nodes + patch would wrap onto the flat bottom annulus (BC patch covering the large bottom face), and at
  `z=Z_BOT` exactly all four feet connect through the annulus into one region → the whole bottom freezes.
  Verify `np.load('so_bcnodes.npy')` foot `zmax` ≪ `Z_BOT`.
- **Two boundaries cause the visible defects; both are fixed by geometry, not smoothing.**
  (1) *MC slivers on the flat foot cap.*  The bg grid snaps to `bbox_lo = geom_min − pad`, and `pad = 0.3`
  is an integer ×`h`, so the flat cap at `geom_min` lands exactly on a grid plane → those nodes sit at
  SDF=0 and MC degenerates into sliver triangles (aspect ratio in the millions).  Fix: `pad = 0.3 + 0.5·h`
  so the cap falls mid-cell (slivers 88 → 0, max aspect 9e6 → ~200).  (2) *The frozen/free seam.*  Where
  the frozen foot meets the free MC-reinitialized leg there is a stitch scar; the zip-patch **grow** widens
  it into a band up the wall.  With **larger feet the exact patch is clean, so disable the grow**
  (`bc_zip_patch_grow 0`, `SO_PATCH_GROW`) — no dilated seam, and no branching (thin feet at `FOOT_R 0.13`
  needed grow 3 to avoid a bowtie "cup" patch; `FOOT_R 0.24` does not).  Net: the only remaining seam is a
  faint ring low on the foot.  Probe geometry changes with `SO_N_ITERS=3` first (zip-init failures are
  fast, ~30 s).  NB the reinit here is **marching cubes** (`reinit_sdf_method="marchingcubes"`), NOT
  Sussman — the free SDF is rebuilt from the MC surface each remesh, then the BC region is hole-cut and the
  iter-0 patch zip-stitched (`sdf_bc_fix_method="zip"`).
- **The clip box itself must end on initial-octree planes.**  The half-cell nominal pad fixes MC
  slivers, but its faces cut through mixed-depth octree cells; centroid clipping then removed an
  alternating one-cell boundary strip before ParaView ever saw the mesh.  `so_prep.py` now expands
  the nominal box to the nearest initial-depth planes while preserving the exact original full-cube
  origin.  Thus SDF samples, marching cubes, and BC zip do not move, while the retained octree tiles
  the whole box.  Static design/remesh captures and every frame assert matching extrema and equal
  summed hex/bbox volumes before export.
- **Depth 8 OOMs at ~iter 8 — stay at depth 7.**  Every remesh changes the mesh size and Warp's fem
  temporary cache (`warp.fem.borrow_temporary`, patched in `core/precision.py`) pools a fresh ~1.8 GB
  temporary per unique size and never frees the old ones → 48 GB gone by iter 8.  (The async VTU writer
  also leaked and raised a stale `disp[cells]` IndexError; `output_per_optimize_step` alone does not stop
  it — shape_opt queues surface saves separately.)  Depth 7 (leaf 0.03) has no leak and runs 90+ iters.
- **Run on GPU 1** (`CUDA_VISIBLE_DEVICES=1`, plus `VTK_DEFAULT_EGL_DEVICE_INDEX=1` for the render): GPU 0
  is shared and a co-tenant's job corrupted a run into a CUDA illegal-memory abort.
- **Octree field panels: render like `diffused_nitsche` — a CENTERED coordinate-plane cut, whole cube,
  no box crop.**  The old form box-clipped the far-field air and cut diagonally on `x=y`; BOTH were
  dropped.  Rotate the four feet onto the x/y AXES (0/90/180/270°, not 45° — one env change in `_ang`)
  so a centred `y=0` plane (normal `(0,1,0)`, origin nudged to `y=0.009` off the cell boundary, keep the
  far half) slices through two opposite feet: the table's ⊓ profile with two legs, exactly the bunny/DN
  cut.  Show the WHOLE octree cube — do NOT crop the air: `ResetCamera` to fit every actor (the cube
  outline included) then dolly ×1.05, so the frame is never clipped.  Cell edges `LineWidth 0.9`, cube
  outline `2.4` colour `[0.30,0.32,0.35]` — all == `render_dn.py`.  Shared view dir `(1.5,-3.0,1.5)`
  (mostly −y, near face-on to the cut), up `+z`, `ViewAngle 25`.  **Panel (b) is the exception**: the
  trimmed SOLVE mesh is shown IN FULL (`cut=False`), framed by the SAME octree cube outline
  (`frame_vtk="so_design.vtk"`) — not a cut profile.
- **The sensitivity is a BOUNDARY quantity and tiny at the turn; mask the BC region to 0, not NaN.**
  `∂J/∂SDF` is ~0 in the interior and lives on the boundary; at the turn the free-region max is only
  ~2e-3 while the frozen LOAD region holds the big values.  Mask the frozen BC nodes
  (`_reinit_fixed_bool` positions, `cKDTree` within `1.5 h`) to **0** — NaN contaminates the clip's
  cross-section interpolation and greys the whole face — then a tight `±0.001` diverging range shows the
  free boundary shell (leg/plate surfaces) while BC + interior stay neutral.
- **Give each scalar its OWN colormap so the four field panels never read as the same quantity.**
  SDF (a, e) keeps the house red↔blue diverging (`±0.30`).  `|u|` (b) is PLASMA (dark purple→yellow,
  sequential magnitude) — viridis was rejected as "ugly" and the original sequential Blues clashed with
  the SDF blue side.  Sensitivity `∂J/∂SDF` (d) is PRGn (purple↔green diverging, `±0.001`) so it is not
  read as the red↔blue SDF.  Define plasma/PRGn by explicit RGBPoints in `render_so.py` (the sequential
  plasma via `RGB` colour space, the diverging maps via `Lab`) and MIRROR them in `so_figure.py`'s
  colorbars with `plt.get_cmap("plasma"|"PRGn")`.  (Any sequential preset renders high values light, so
  set the ramp by hand or the loaded top comes out pale and the fixed feet dark, backwards.)
- **Panel (g) BC patches MUST come from the restoration's OWN face labels — never a proximity guess or a
  re-run selector.**  This was the crux of getting (g) right.  `stitch_bc_patches` returns
  `(verts, faces, face_labels)` with `face_labels` ∈ {0 base, 1 restored BC patch, 2 zip bridge}, and
  **label 1 IS the exact set of surfaces the run holds fixed** (the grown frozen-SDF / BC-zip patch).
  `so_prep` dumps `verts/faces/labels` to `so_bc_masks.npz`; `so_split_bc.py` splits label-1 by centroid
  z into Dirichlet feet (`z<0`) and Neumann top (`z>0`), and everything else (labels 0,2) is body.  Two
  WRONG detours were tried and rejected: (a) classifying MC faces by KDTree distance to the frozen nodes
  (`so_bcnodes.npy`) — a hard per-face threshold against the discrete node lattice yields a ragged
  ZIGZAG boundary; (b) resolving the production `LowestSelector('z', FOOT_FRAC)` on the surface — clean,
  but it is only the ~0.02 frozen *band* (the FEM selector's raw slab, downward-facing and occluded),
  NOT the restored region, so it is "not consistent with the surfaces being fixed".  The face-label path
  is the only one that is both clean AND faithful; it needs `so_prep` to keep `stitch_bc_patches`' third
  return (was discarded as `_`).
- **Panel (f) design-evolution strip == void_nucleation's strip mechanics.**  Run to convergence
  (`N_ITERS 101`) and dump the design MC surface at `SO_EVO` iterations (`0,20,40,60,80,100`; keep 40 so
  one frame matches the loop).  `so_prep._mc_surface()` (== vn's, via `_get_or_build_mc_surface_physical`)
  writes the full `so_evo_<it>.stl`; the CUT is done in the conda step (`so_split_bc.py`, NOT `so_prep`,
  so the strip re-renders without re-running the 10-min optimisation) — `trimesh.boolean.intersection`
  against a half-space box on `y=0` (through two legs, `manifold3d` backend) gives the capped far half
  (`_shell` + `_face`) and the near half (`_ghost`).  **Load the evo STL with `process=True`** or the
  boolean throws "Not all meshes are volumes!" (duplicate MC verts → not watertight). Some sampled
  runs also shed tiny open MC debris, so both `so_split_bc.py` and `so_fr_cut.py` retain only
  watertight components above `1e-4` volume before the boolean. `render_so.panel_evo`
  renders the three actors **exactly like `hidden_phi_loop`'s (f)** (`render_hp.section`, adopted
  2026-08-11): `_shell` as a PBR off-white body solid, `_face` UNLIT and a step **DARKER** than the body
  (`CUT_FACE = [0.760, 0.735, 0.690]`, the sectioned-solid convention — so the section reads as filled
  material, not a same-tone shell), `_ghost` at `Opacity 0.20`, under an SSAO view whose ambient
  occlusion gives the section its solidity, from a FIXED near-side elevation camera (`EVO_VIEW_DIR
  (0.30,-0.90,0.30)`, sized off a fixed frame box, NOT `ResetCamera` per frame) so frames don't jump in
  scale.  Only panel (f) uses this path — the loop panels (a-e,g) keep their own view/camera.  Render just
  the strip with `pvbatch render_so.py evo` (the script now takes optional panel-slot args like
  `render_hp.py`; no args = all panels, as `build.sh` calls it).

| Knob | Value | Where |
|---|---|---|
| blank | cylinder R 1.0 × H 1.0 + four short feet `boolean.union`-ed (needs `manifold3d`), centred | `so_prep.py` |
| BCs | four large cylindrical feet (`FOOT_R 0.24`, `FOOT_H 0.22` protrude, bolt circle 0.6, **on the x/y axes 0/90/180/270°** so a `y=0` cut hits two); freeze only the bottom `FROZEN_H 0.02` band via `LowestSelector('z', ~auto)`; whole top `HighestSelector('z',0.06)` = (0,0,-0.02) | `so_prep.py` |
| grid align | nominal pad `0.3 + 0.5·h` keeps flat foot caps off grid planes; the clip box is then expanded onto initial-octree planes without shifting the original cube lattice | `so_prep.py` |
| octree | uniform depth **7**, `h = 3.84/2⁷ = 0.03`; depth 8 OOMs | `SO_DEPTH` |
| BC fix | `zip`, `bc_zip_patch_grow 0` (grow OFF; large feet keep the exact patch clean, no seam band) | `SO_BCFIX`, `SO_PATCH_GROW` |
| optimiser | `gd_lr 8`, `w_vol 5`, `target_volume_fraction 0.5`, **101 iters (to convergence)**, capture 40 | `SO_GD_LR/W_VOL/SO_N_ITERS/CAPTURE_ITER` |
| evo strip (f) | design surface at `SO_EVO` iters `0,20,40,60,80,100`; half-cut on `y=0` (shell+cap+ghost), rendered like `hidden_phi_loop`'s (f): DARKER `CUT_FACE`, PBR+SSAO, fixed elevation camera | `so_prep.py`→`so_split_bc.py`→`render_so.panel_evo` |
| captured | iteration 40: design 90 374 nodes / 73 875 hexes, solve 78 223 / 61 804, 5 BC patches, raw volume 1.69589; bbox/hex volumes 15.9252480 / 15.9252481 | `so_stats.txt` |
| octree cut | NO box crop; centred plane `y=0` (normal `(0,1,0)`, origin `y=0.009`), keep far half; whole cube shown | `render_so.py` |
| camera | loop panels (a-e,g): `ResetCamera` fit + dolly ×1.05, view dir `(1.5,-3.0,1.5)`, up `+z`, `ViewAngle 25`.  evo strip (f): its OWN fixed elevation camera `(0.30,-0.90,0.30)` sized off a fixed box (== `render_hp`) | `render_so.py` |
| ramps | SDF diverging `±0.30` (house) for a,e-tags; `\|u\|` **plasma** 0..0.0125 for b; sensitivity **PRGn** `±0.001` (BC→0) for d-tag | `render_so.py` |
| panel e colour | Dirichlet/Neumann/body from `stitch_bc_patches` face-labels (1=BC patch, split by z), NOT proximity | `so_prep.py` → `so_bc_masks.npz` → `so_split_bc.py` |
| figure | ~11.5 × 10.6 in, loop (a,b top-centre; c,d,e middle) + full-width evo strip (f) at the bottom; right-angle loop arrows with labels above/below their horizontals; one-line BC legend; `bbox_inches='tight'` via figure-level artists | `so_figure.py` |

`build.sh shapeopt` runs prep → render → compose on GPU 1 and deletes its intermediates (keeps
`so_stats.txt` + `so_traj.txt`); a renderer-only experiment must re-run `so_prep.py` first (same trap as
§5b–§5e).

**§5f-frames — the design-evolution movie (`build.sh soframes` / `hpframes`).**  A separate
deliverable from the loop figure: one composed frame PER optimiser iteration, the octree+`s` section
cut (panel a's code) beside the sectioned design body (the evo strip's code), with the iteration
number, cumulative optimiser wall-clock, and octree node/element counts marked top-left.  Built by
re-running the SAME prep in FRAMES mode (`SO_FRAMES=1` / `HP_FRAMES=1`): it skips every loop-figure
capture and instead dumps `so_fr_<it>.vtk` (octree+sdf) + `so_fr_<it>.stl` (MC surface) each
iteration plus a cumulative wall-clock (dump I/O subtracted out) with node/element counts to
`so_fr_times.txt` — so the shipped loop cache is never written and stays intact.  Pipeline:
`so_prep.py`(FRAMES) → `so_fr_cut.py` (sections every `so_fr_*` like `so_split_bc`) →
`render_so_frames.py` (imports the main-guarded `render_so` panel fns) → `fr_figure.py so` (side-by-side
PIL compose → `shape_opt_evolution_frames/frame_*.png` + `.gif` + `.mp4` and the presentation-sized
`shape_opt_evolution_slides.{gif,mp4}`).  `hpframes` mirrors it with
`hp_prep.py`/`hp_fr_cut.py`/`render_hp_frames.py`/`fr_figure.py hp`; `hp_prep` needs the repository
`src` on `PYTHONPATH`, and `hp_fr_cut` needs the conda libspatialindex runtime (build.sh sets both).
Render fns are importable because
`render_so`/`render_hp`'s render loops are now under `if __name__ == "__main__"`.  ~1 h for the pair
(two optimiser re-runs + ~360 renders).  Heavy `*_fr_*.vtk/.stl` are dropped after render; the panel
PNGs + times are the cache — force a fresh run by backing up/removing
`shapeopt/panel_so_frbody_*.png` or `hiddenphi/hp_panel_frbody_*.png` plus its times file.
**Run both with `SO_SAVE_EVERY=0` / `HP_SAVE_EVERY=0`** (§2): the clock drawn on each frame is the
optimiser's compute only when the run is not also writing per-iteration VTU/GLB.
The coarse-bbox-corrected hidden-phi movie capture on 2026-09-24 contains **91 frames** and records
**232.29 s** of optimizer compute on an NVIDIA RTX A6000 (`hiddenphi/hp_fr_stats.txt`); all
91 frame exports passed the complete-bbox volume gate, all 91 surface sections completed, and the
raw `hp_fr_*.vtk/.stl` files were removed after rendering.
The corresponding coarse-bbox-corrected shape-opt capture contains **101 frames** and records
**239.96 s** of optimizer compute on the same GPU (`shapeopt/so_fr_stats.txt`); all 101 exports
passed the volume gate, all 101 surfaces were sectioned without a skip, the composed masters are
2163×1102, and the raw `so_fr_*.vtk/.stl` files were removed after rendering.
**Never make `_slides.gif` by scaling an existing GIF with ffmpeg**: that produced bad local palettes
and disposal trails.  `fr_figure.py` resizes each RGB master frame to 1920 px, quantizes each frame
adaptively, and saves with full-frame disposal; its slide MP4 is 1920 px wide, 30 fps, and 5 seconds.

---

### §5g The same loop with a second field in it (learned building `hidden_phi_loop`)

`shape_opt_loop`'s rectangular cycle carrying `void_nucleation`'s content: the same
cylinder→table problem (`so_prep.py`'s blank, BCs and schedule) run in PHASE-FIELD mode, so three of
the five stations carry TWO images and the extra one is always the hidden phase field.  (a) design
`s` + hidden φ → (b) diffused-Nitsche solve → (c) sensitivities `∂J/∂s` and `∂J/∂φ` → (d) the two
updates, `s` then hidden φ SIDE BY SIDE → (e) marching cubes + BC restore, sectioned so the cavity the cycle just
carved is visible → back to (a).  Four stages `hp_prep.py` → `hp_cut.py` → `render_hp.py` →
`hp_figure.py`, wired as `build.sh hiddenphi`, ~8 min on GPU 1.  **8.5 × 4.96 in for eleven renders
where `shape_opt_loop` spends 11.5 × 10.6 on eleven** — the pairs make the loop wider and taller, so
everything else moves in: one shared fixed camera, per-panel ink crops, colour bars in the row gaps
and inside (c)'s own white rather than at the figure foot.

- **The loop is the right frame for this mechanism, and the production order says so.**  Inside one
  `ShapeOptTrimmed` iteration: forward solve → `_post_tape_backward` (both gradients) → the optimizer
  step on `s` → `_post_optimizer_step_remesh_impl`, which *begins* by calling `_pf_update_hidden_phi`
  and `_pf_filter_thin_features` and only then scatters, carves (`_localized_phi_to_bg_overwrite`),
  marching-cubes and zips.  So the hidden field's update belongs with `s`'s in (d), the carve belongs
  on the (d)→(e) arrow, and the void first exists in (e) — the panel order is the call order, not a
  narrative.
- **Capture at an iteration you cannot name in advance.**  The interesting iteration is the one that
  NUCLEATES, which varies with the schedule.  Record every iteration into rolling buffers (`buf_bw`,
  `buf_flt`) and freeze them the moment `_find_voids` reports an enclosed cavity on an iteration where
  the thin filter COMMITTED; that recorder now lives in `hp_prep.py`. With the
  §5d schedule (PF on at 35, `pf_phi_lr` 12, `gd_lr` 8) its iteration is discovered from the
  current run.  Do not claim a fixed iteration across executions.
- **To record a production return value, wrap the function — do not copy the sequence around it.**
  Panel (e) needs `stitch_bc_patches`' third return (`face_labels`), which production discards
  (`out_verts_phys, faces, _ = …`).  `so_prep.py` gets it by re-implementing `_build_mc_surface_and_bvh`'s
  whole raw-MC → filter → close → stitch sequence in the subclass — which works but has to be kept in
  step with `shape_opt.py` by hand.  `hp_prep.py` instead patches `core.remeshing.stitch_bc_patches`
  with a recorder that calls the real function, keeps a copy of all three returns and hands the
  originals back.  Ten lines, cannot drift, and the import inside the production function
  (`from .remeshing import stitch_bc_patches`) resolves at call time so the patch takes.
- **THE ghost's own cut face is what was making every section muddy — drop it.**  Both halves come out
  of the half-space boolean capped, and the near half's cap lies exactly on the section plane,
  directly between the camera and the face the panel is about.  Even at 13 % opacity it prints as a
  translucent sheet over the entire cross-section.  `hp_cut.open_ghost` splits it off (`split_cap`
  with `sign=+1`, its normal faces away) and exports the ghost as an OPEN shell.  This is the single
  biggest legibility win in the figure and it applies to `void_nucleation`'s and `shape_opt_loop`'s
  strips too — they ship with the cap still on.
- **A cavity is a separate COMPONENT of the MC surface, so colour it as one.**  Even with the ghost
  cleaned up, at ~1.1 k background cells the void read as a smudge on the section.  Take the
  connected components of the face graph, keep the largest as the body, and mark every other
  component that the body `contains` (a containment test, not a size test — MC also throws debris
  specks that sit ON the outer boundary, and painting one of those as a void would be a fabrication).
  Those faces get a cold dark slate `#4f5866` and a legend entry.  The ghost drops them too, or the
  near half's copy of the cavity prints over the far half's.
- **PBR is fine on the off-white body and wrong on a coloured actor.**  §5d found that
  `bunny_sdf_pv`'s PBR material reads better than matte on the sectioned body, so it stayed.  On (e)'s
  BC surfaces it pulls the Neumann orange down to a brown and the teal to a slate — the panel no
  longer matched its own legend swatches.  Panel (e)'s three BC actors use `render_so.py`'s plain
  material (Ambient 0.14 / Diffuse 1.0 / Specular 0.30, no PBR), which is what the legend colours were
  chosen against; the strip keeps PBR.
- **Measure the arrow labels and let them set the layout, do not place them and hope.**  With five
  stations the horizontal runs on the loop's two legs are the scarcest space on the page and both
  two-line labels are wider than the run.  Centring put "Reinitialize SDF" through its own arrow one
  way and off the canvas the other; anchoring it to the panel edge fixed that and broke the other
  side.  The shipped script measures each label in a throwaway figure (`text_w`), sets `RUN_L`/`RUN_R`
  from the measurements, derives the top row's x from the runs, and asserts the whole band still fits
  between the two legs.  Same idea rescued the four-entry BC key, where a per-character width estimate
  put "Neumann" through the next swatch.
- **Give the pair-stations a shape and stick to it.**  (a), (c), and (d) put their two images SIDE BY SIDE —
  each pair is one state, one gradient, or one update of the same object, read at a glance; (d) keeps
  the same `s`, hidden-φ order as (a).  (e) is drawn 1.2 field-panel heights, not
  2.0: at 2.0 it took 2.5 in of width, more than the two sensitivity panels together, and a ghosted
  off-white section does not earn that against panels carrying a field and a mesh.
- **A ragged outer layer is a capture-domain bug, not a ParaView clipping bug.**  The old
  `clip_bbox` crossed mixed-depth octree cells and `clip_octree_to_bbox` retained them by centroid.
  On the `y=0` section this deleted a `0.12`-wide strip from `x_min` over the upper part and from
  `x_max` over the lower part: 256 of 5,888 finest-cell tiles were genuinely absent. Moving the
  render plane could not restore them. `hp_prep.py` now rounds every padded span up to an even
  number of initial-octree cells while preserving the original full-cube origin, placing all six
  faces on coarse-cell planes without moving any SDF sample near the geometry.
  It asserts the node extrema equal that box and that summed axis-aligned hex volumes equal the box
  volume before exporting. The physical surface boolean remains exactly `y=0`; the octree render
  uses `y=0.009` only to make VTK's far-side face choice unambiguous.
- **Panel (c) floats inside a two-panel-tall row — put its colour bar in that white.**  It was
  originally down in a band under the row with the keys, two inches from the panel it scales.  Moving
  it directly under (c) closed the gap and let `BOT_BAND` shrink to one line (0.46 → 0.26 in).
- **The `s` panels are mostly saturated air, and that is deliberate** (§5d): the ramp clamps at ±0.30
  and the positive end stops at a mid orange rather than the house brick, or the far field outweighs
  the part.  Do not "fix" the orange blocks in (a) and (d).
- **The sibling figures share one capture.** `voidnuc` is now a focused composition of
  `hiddenphi`; never restore an independent `vn_prep.py` run.  Source both figures' mesh and cavity
  counts from `hp_stats.txt` and map panels explicitly in `voidnuc/render_vn.py`.
  The section colour was run through the §5c palette validator: `#4f5866` is worst-pair CVD ΔE
  22.1 / normal 33.9 against teal, orange, body and cut face.

| Knob | Value | Where |
|---|---|---|
| problem | `so_prep.py`'s cylinder+four-feet blank verbatim, whole-top load, `LowestSelector` feet, 50 % target, `w_vol 5`, `gd_lr 8`, depth 7 | `hp_prep.py` |
| schedule | phase-field design variable, hidden φ ON at iteration **35** (`HP_PF_ON`), `pf_phi_lr` **12**, 91 iterations | `hp_prep.py` |
| captured | the NUCLEATING iteration, found at run time — **43** (41 dips in / 39 kept, 112 viable elements, 1 180-cell cavity, 91 096 nodes / 74 575 hexes, solve mesh 79 153 / 62 690) | `hp_stats.txt` |
| panels | (a) `sdf`+`hidden` side by side, (b) `umag` on the inner mesh UNCUT, (c) `gsdf`+`gphi` side by side, (d) `sdf_c`+`hidden_c` side by side, (e) sectioned surface | `render_hp.py` |
| ramps | `s` diverging ±0.30; φ sequential blue 0..1; both gradients log₁₀-of-own-max floored at 4 decades, BC region masked to a below-range sentinel; `\|u\|` plasma 0..`umag_max` | `render_hp.py` |
| contours | `s = 0` ink 4.0 on every field panel; hidden φ = 0.5 teal 8.0 on (a)-right and (d)-top — (a) has no interior dip and (d) does, which IS the figure | `render_hp.py` |
| section | surfaces: exact `y = 0` half-space boolean; octree fields: plane origin `y = 0.009`, keeping the far half; the captured bbox is expanded to initial-octree planes without shifting the original cube lattice, with extent and complete-volume assertions; cut face unlit `#c2bcb0`, cavity `#4f5866`, ghost OPEN at 0.13 (0.20 for the strip) | `hp_prep.py`, `hp_cut.py`, `render_hp.py` |
| camera | field panels: shared cube camera `CUBE_VIEW_DIR (0.42,-1.1,0.40)`, `CUBE_DOLLY 1.05`; surface/evolution panels: fixed `(0.30,-0.90,0.30)`, up `+z`, `ViewAngle 25`, sized from the fixed frame box | `render_hp.py` |
| figure | 8.50 × 4.96 in; field sub-panel 1.39 × 1.11, (e) 1.64 × 1.33, strip 6 frames; leg runs measured from the label widths | `hp_figure.py` |

`build.sh hiddenphi` runs prep → cut → render → compose on GPU `${HP_GPU:-1}` and deletes its
intermediates (keeps `hp_stats.txt`).  `hp_cut.py` is split out of the prep on purpose (as
`so_split_bc.py` is): every experiment with the section, the BC colours or the ghost needs those few
seconds of trimesh and none of the 7-minute run.

### §5f/§5g redesign (2026-10-12, revised after the designers' review): chain key on top, one 5-column grid

**This supersedes the U-shaped loop layouts described in §5f and §5g** (their capture notes
still hold).  Both figures are composed by `_shared/loop_chain.py` (Page in mm, y down;
Figure 1's tokens: ink `#1a1a21`, grey `#5f6672`, backward magenta `#a3195b`, forward
`#30343b`, 0.75 pt stealth arrows, the FEA tint `#ebf5e6` for every chain station; 8 pt bold
panel letters, 6.5 pt titles, 6 pt notes, 5.5 pt ticks, Type 42 Arial; `Page.check()` fails
the build on text under 5 pt, off the page, or overlapping).  Shape 180 × 90 mm, hidden
180 × 97 mm; a `loopstack` target used to draw both on one page (a–e / f–j) — RETIRED
2026-10-13 with `_shared/stack_loops.py` and its preview, superseded by §5j.  Standalone
builds letter a–e (`--letters` overrides).  Legend text lives in each folder's
`legend_notes.md`, not in the figure.

- **One 5-column grid (30.8 mm) for both**, so stacked stations line up: s | φ̄ | φ | u | J.
  Shape-opt leaves column 2 (and column 4 of row c, φ̄'s step) as deliberate whitespace.
- **Chain**: s → φ —Nitsche solve→ u → J; magenta backward lane with dot tails: J→u
  **Auto-diff** (taped `fem.integrate` + kernels), u→φ **Adjoint state method + auto-diff**
  (`AbstractField.step` registers `adjoint_fn` via `tape.record_func`: Kλ = −∂J/∂u, then
  ∂R/∂φ·λ through the taped residual), φ→s **Auto-diff** (`_sdf_to_phi_kernel` on the tape),
  arriving as ∂J/∂s; a thin grey RETURN closes the loop ("Update s → re-initialise / remesh
  + restore BC").  **Hidden-φ: φ̄ is NOT on the forward line** — `_set_e_phi_to_one` holds
  the interior weight at 1 through solve and objective.  φ̄ sits above it in a dashed box
  (dashes = optional, Figure 1's convention) with its two real links, both dashed: the
  gradient "∂J/∂φ at φ = 1 (interior)" (magenta) and "Carve φ̄ < ½" into s (ink).
- **Rows**: b forward under the stations (octree + s | hidden φ̄ | trimmed mesh + φ |
  diffused-Nitsche solve | J history with SE/SE₀ faint and the captured iteration);
  c |∂J/∂·| / own max, log 4 decades, Figure 1's magenta; d the update in call order
  (step on s with the previous s = 0 as light dashes OVER the new curve → [step on φ̄,
  φ = ½ thick, the dip labelled "To be carved", φ̄ greyed outside the body] → remesh + BC);
  e the strip at 70 % height with the captured iteration in bold (hidden-φ: the captured
  iteration's own restored surface `hp_e_*`, slot `l_evo_cap`, replaces the nearest frame).
- **One key band** right of the strip: s, φ(, φ̄), |u| (0 … max, the paper's cream → brick
  ramp), |∂J/∂·|/max (log); s = 0, previous s = 0, φ̄ = 0.5, BC swatches.
- **Renders sized for print**: cell frames 4.6–4.7 px = 0.3 pt at 30.8 mm in mid grey
  (black at that weight fills the finest cells), cube outline 4.6 px, s = 0 7 px, RGB
  interpolation so panels match their matplotlib bars; the trimmed-mesh φ panel draws
  frames on the cut face only (a Slice a hair in front of the clip; untriangulated) and
  the uncut solve panel none (grazing faces turned grey); strip ghost 0.34 in a darker
  grey.  Hidden-φ's loop panels are the `l_*` slots of `render_hp.py`; the un-prefixed slots
  keep their old look because `void_nucleation` aliases them.  `render_so.py` changes DO
  reach `render_so_frames.py` if the frames are re-rendered.
- VTK cannot dash lines: dashes are cut from the contour with a scalar clip on
  sin(40(x+z)), then translated 0.02 along the view direction so they draw over the
  coincident solid curve without z-fighting (a translation along −y shifted them on screen).


**Round 3 (designers' second review, 2026-10-12) — what changed on top of the above:**
- Sizes: shape standalone 180 × 95 mm on a **compact 4-column grid** (6 mm gaps, no empty
  columns); hidden 180 × 89 mm; stacked page 180 × 174 mm, where shape switches to the
  5-column `layout="stacked"` with NO key band (j's band is the combined key) and its strip
  matches j's.  Tightening: panel box aspect 1.35, strips at 62 %, rows/chain spacing.
- Gradient panels: each field over its OWN 99th percentile (of non-zero free-node values),
  clipped at 1, log 4 decades ("|∂J/∂·|/p99", ticks 10⁻⁴ … ≥1); the field drawn without
  frames and the frames over it as a 40 % wireframe (a separate `ExtractSurface` proxy —
  `Show` of the same proxy returns the SAME representation); s = 0 at 0.35 pt in a white
  casing (both lifted along the view direction).  Gotcha: a LUT with a below-range sentinel
  must get `AutomaticRescaleRangeMode = "Never"`, or it re-fits to [−1, 1] at render time
  and 0 prints mid-ramp pink.
- Chain: return label just "Update s" at the hook; letters at each chain's top-left (f level
  with the φ̄ box); f's φ̄ gradient label "∂J/∂φ̄ = ∂J/∂φ at φ = 1 (interior)" in magenta.
  "Nitsche solve" on the chain AND the heads ("Nitsche solve, u"); update heads "Remesh +
  restore BC" / "Carve + remesh + restore BC"; strip frame "After iteration 43".
- Objective: J and SE/SE₀ only (formula in legend_notes); hidden ticks 0/30/60/90.
- NOTE for the solver docs: `_pf_update_hidden_phi`'s docstring says the band tracks
  tanh(SDF/ε); the code uses `_sdf_to_phi` = ½(1 − tanh(SDF/ε)).  Docstring slip only.

### §5j The merged shape-optimisation figure (2026-10-13, `shape_opt_merged`)

The user's verdict on the loop figures' grid of panels: "no logical connection between the
panels … it should correspond to the method overview figure or to panel (a)".  The
answer is ONE figure whose renders ARE the loop: `shapeopt/som_figure.py`, built by
`./build.sh shapeoptmerged` (runs `shapeopt` and `hiddenphi` first for their render
caches; no optimiser re-runs while the caches exist).  Legend text:
`shapeopt/legend_notes_merged.md`.

- **a, left: the loop in Figure 1's language** (`ov_figure.block/station/arrow`): every
  station a light, edge-less, rounded block of its part's tint (Pre-processing `#eaf1fa`
  for s, step and remesh; FEA & optimization `#ebf5e6` for φ, u, the gradients and the J
  pill), title in a 3.3 mm head band, the render on the floor, Figure 1's green GPU chip on
  the stations Figure 1 marks.  Row 1, left to right: Remesh + restore BC → Signed
  distance s → Phase field φ → Nitsche solve, u; the solve drops into the Objective J pill
  (row 2, Figure 1's place); the magenta lane runs under row 2 from J and rises into ∂J/∂s,
  carrying ∂J/∂u / ∂J/∂φ / ∂J/∂s with their methods under the stations they belong to;
  ∂J/∂s → Step on s → up into Remesh → s closes the loop.  The OPTIONAL hidden phase
  field is dashed: ∂J/∂φ̄ (Optional) rises dashed from the lane at φ's column ("at φ = 1,
  interior") and leaves through a dashed "Step φ̄, carve φ̄ < ½" into the remesh riser.
  No objective plot.
- **b, c, right: results**, each run's evolution as a 2 × 3 grid, captured iteration bold
  ("After 43" for the carve frame).  **One key band** as a footer under the loop.
- **Renders onto the block tint, opaque** (Figure 1's rule): `render_so.py merged`
  writes `panel_so_m_*` with `BG` = the tint (no `WhiteBackground` palette override),
  `render_hp.py merged` writes `hp_panel_m_*`.  Crops key on the CORNER colour, not white.
  Crop each run's renders on their own: the two renderers' images differ in size, and a
  common box from the larger one pads the smaller with BLACK bars.
- Which run feeds which block (main loop: shape run, iteration 40; optional block:
  hidden-φ run, iteration 43) is stated in the legend notes.  The standalone `shapeopt` /
  `hiddenphi` loop figures still build (their compact layouts above) as supplementary
  material.

**Round 2 of §5j (designers, 2026-10-13):** "Geometry + BC" (was Remesh + restore BC)
with an "Initial design" entry stub from the left (gutter widened to 10 mm); Figure 1's
titles "Octree + s" / "Trimmed mesh + φ"; Step on s retinted green (render re-made on the
FEA tint); the φ̄ step shown as a small dashed INSET on its dashed connector, in the band
between the rows (`hp_panel_m_d_hid`, label "To be carved" with a leader to the dip
loop's right end, found as the teal component strictly inside the outline); ∂J/∂φ̄ block
tagged "run c, it. 43" in its head band; results as 2 × 2 grids (b: 0/40/60/100; c:
35/43/60/90, starting where the hidden field switches on), b's first row level with the
loop's top and c's last row ending with row 2; enclosed cavity walls slate (`hp_cut.py`
now writes `hp_evo_<it>_shellbody.stl` + `_void.stl`; only enclosed cavities qualify, so
iteration 90's opened voids are not coloured); a two-row key band ~3 mm under the lane
labels.  180 × 94 mm.

**Round 3 of §5j (the user, 2026-10-13) — supersedes the layout above:** a takes the LEFT
HALF (~90 mm), b and c side by side the right half, exactly as tall as a (their rows are
spread to a's height; b uses c's row pitch so the rows align).  a is compact (blocks
~18 mm, renders ~16 mm, 6.5 pt titles, Figure 1's text-to-image balance) and organised BY
VARIABLE: row 1 forward (Initial design → Geometry + BC → Octree + s → Trimmed mesh + φ
→ Nitsche solve, u → Objective J pill below); row 2 the backward chain UNDER its stations
(∂J/∂u pill under u, ∂J/∂φ pill under φ, the ∂J/∂s block under s, magenta arrows labelled
auto-diff / adjoint + auto-diff / auto-diff) and Step on s returning up into Geometry +
BC; row 3 dashed: ∂J/∂φ̄ (subtitle "at φ = 1, interior", fed dashed from the ∂J/∂φ pill)
→ Step on φ̄, carve φ̄ < ½ (full block, "To be carved"), returning dashed through the left
gutter into the same riser; tag "run c, it. 43" on the row.  Keys INSIDE their panels:
a's four bars (min/max only, "log" on the gradient) in a's empty bottom-right cell and
s = 0 / Dirichlet / Neumann / Optional along a's foot; c's "Carved void" under c.  Body,
part tints, the chip and the p99 detail live in legend_notes_merged.md.  Iteration
numbers above the shapes.  180 × 75 mm.

**Minor fixes (2026-10-13):** column gap 4.6 mm (Figure 1's) so every arrow is the same
glyph; both horizontal backward labels under their arrows; bars flush with column 4's
edge, s's bar has a 0 tick; the chip re-keyed ("Differentiable GPU code (Warp)"); the run
tag moved to the notes; "To be carved" under its block with a vertical leader.  Ghosts
darkened ~10 % — which exposed diagonal hatching on the SHAPE strip: its ghost still had
its own cut cap, coincident with the far half's cut face (§5g's lesson, never applied to
render_so); `_evo_body(open_cut=True)` now clips it off at y < −0.003.  Opened voids
(iterations 80, 90) are not coloured: the surfaces do not record which faces the carve
made, so c's key says "Carved void (enclosed; opens by 80)".

**Round 4 of §5j (the user, 2026-10-13) — supersedes the rounds above where they differ:**
no tinted fills (the user: "pink and green background look awful"): every station renders
on WHITE (`render_so/render_hp` merged slots now use a white BG) inside a 0.45 pt `#c9ced6`
rounded outline, dashed for the optional φ̄ row; one-line 6 pt titles ABOVE the outlines
(they may overhang the outline into the 4 mm column gap; the page gate checks
neighbours); b OVER c, one row of three frames each (0/40/100, 35/43/90), together as
tall as a; frames gamma-lifted 0.72 at compose with slate pixels (B − R > 6) masked out;
"Auto-diff"/"Adjoint + auto-diff" all under their arrows; the Step-on-d riser runs at the
column's left so it misses the centred title; texts "run c", "To be carved", "carve φ̄ <
½" and the Carved-void key removed (legend notes carry them).  Opened voids ARE now
coloured: `hp_cut.py` marks every face within 1.5 h (0.045) of an earlier frame's enclosed
cavity wall (frames in iteration order) and splits the ghost the same way
(`_ghostbody`/`_ghostvoid`, the latter slate at 0.55).  Ghost darkening reverted.
180 × 64.5 mm.

**Round 5 of §5j (the user, 2026-10-13) — supersedes the above where they differ:** a
spans the full width; b and c BELOW it, one row of six frames each (b 0/20/40/60/80/100;
c "Concurrent topology and shape optimization", 20/35/43/60/80/90), iteration above each
shape.  No void colour any more (`hp_cut.py` reverted to e8127d3's; the opened-void
experiment is gone), no GPU chips, no "at φ = 1, interior" (legend notes).  Titles INSIDE
the outlines.  Initial design = b's iteration-0 frame as a thumbnail with its arrow into
Geometry + BC.  Lower rows (and their pills) shifted 5 mm right: s's step rises straight
up into Geometry + BC; φ̄'s step returns dashed through its OWN lane left of the lower
rows to its own entry point (asserted apart).  J sits right of the solve; its magenta
elbow runs down and left into ∂J/∂u.  Backward labels: "Auto-diff" above its arrow,
"Adjoint" above / "auto-diff" below the adjoint arrow.  Bars 2 × 2 and the line keys in
row 3's free right part.  180 × 138 mm.

**Round 6 of §5j (the user, 2026-10-13) — supersedes the above where they differ:** a has
exactly TWO rows plus one legend row.  Row 1: Initial design (a station, b's iteration-0
frame) → Geometry + BC → Octree + s → Trimmed mesh + φ → Nitsche solve, u → Objective J
pill.  Row 2, right to left, image stations only: ∂J/∂u (NEW render, see below), ∂J/∂φ̄
(dashed), ∂J/∂s, and ONE step station over columns 0–1 (Step on φ̄ in a dashed sub-frame,
Step on s) whose arrow rises into Geometry + BC.  Truthful routing: the magenta lane runs
in the gap ABOVE row 2 — ∂J/∂u → ("Adjoint" above / "auto-diff" below) → a dot labelled
∂J/∂φ → ("Auto-diff" above) → ∂J/∂s; ∂J/∂φ̄ drops dashed from the dot; J reaches ∂J/∂u by
an elbow whose "Auto-diff" sits above its horizontal; ∂J/∂φ̄ reaches the step's φ̄ half by
a dashed lane UNDER row 2, so nothing crosses.  b and c titles carry the wall clock from
`*_fr_stats.txt` (the frames-mode, save-off timings).  **∂J/∂u** is not stored by the run:
`shapeopt/so_dJdu.py` (pvpython, CPU) recomputes it from the cached run's own VTU of the
captured iteration as the internal-force vector Σ φ_w P(F)·∇N (2 × 2 × 2 Gauss,
hanging-node entries to masters, strong Dirichlet nodes zeroed); boundary elements use the
standard rule, not the solver's adaptive Nitsche quadrature (stated in the legend notes).
Check printed by the script: deep-interior p90 ≈ 1.3e-7 vs loaded-top median 5e-6.
`build_shapeopt` runs it when `so_dJdu.vtk` is absent (SO_RUN_VTU overrides the VTU).
Station width = the longest title ("Trimmed mesh + φ" at 6 pt, measured) + pads, the
step station's halves alike; a's rows are centred on the page (b, c and the legend row
stay left-aligned at the letter gutter).  180 × 119 mm.
**Then (same day):** a, b, c and the legend share ONE span [AL, AR] (a's row-2 left edge …
the J pill's right edge), centred on the page, panel letters LET left of AL; b/c frames
shrink to that span; the legend row is justified to it (measured on a scratch page, the
7 gaps spread; wraps to two rows only if it cannot fit).  Row 2 sits S2 = (GAP + B)/2
right of row 1 so the step station (2B + 3BP, both halves) is centred under Geometry +
BC and its return rises from its centre straight up (asserted); J moved right so its
elbow clears ∂J/∂u by ≥ 6 mm.  180 × 112 mm.

**The standalone loop figures follow §5j (2026-10-13).** `shape_opt_loop` and
`hidden_phi_loop` are now built by the SAME layout class as `shape_opt_merged`
(`_shared/loop_layout.py`, `LoopFigure`; `som_figure.py`, `so_figure.py` and
`hp_figure.py` only gather renders): two rows + one legend row, then ONE six-frame row (b)
on a's span.  phibar=None (shape: row 2 = ∂J/∂u, the ∂J/∂φ dot, ∂J/∂s, Step on s; no
Optional key), "optional" (merged, dashed) or "solid" (hidden-φ: φ̄ is the method).  The
hidden figure's renders are `render_hp.py`'s `l_*` slots, plus `l_geom` (the whole
restored surface split by stitch_bc_patches' labels, `hp_full_*.stl`, written by
`hp_cut.py`); its ∂J/∂u is the SHAPE run's (`panel_so_m_gu.png`) because the hidden run
kept no displacement vector — stated in its legend notes.  `render_hp.py loop` renders
every `hp_evo_<it>_shell.stl` present (80 included; `HP_EVO` default now lists 80).
Sizes: shape 180 × 80.7 mm, hidden 180 × 85.9 mm; the merged figure is pixel-identical to
before the refactor.  The old U-loop and chain layouts of §5f/§5g are retired.

**Notation pass (2026-10-29, all three loop figures + `void_nucleation`):** the rounds above
quote the labels of their day (∂J/∂s, ∂J/∂φ, ∂J/∂φ̄, ∂J/∂u, "Objective J", "Nitsche solve,
u", |u|, |∂J/∂·|).  They now read D𝒥/Ds, D𝒥/Dφ, D𝒥/Dφ̄, **∂𝒥/∂𝐮** (the partial stays for
the displacement), "Objective 𝒥", "Nitsche solve, **u**", |**u**| and |D𝒥/D·| (§2 notation
rule).  The J pill keeps the width measured on the old "Objective $J$" (`LoopFigure.jw`): the
calligraphic J is ~0.3 mm wider and still clears the outline, and measuring it would widen
a's span and move every station.  The legend row is re-justified by its own code (the wider
|**u**| and |D𝒥/D·| symbols shift the later bars by a few px).  Panels unchanged.

**Load arrows on Geometry + BC (2026-10-14, all three figures).**  The orange top alone did
not say "pressed down"; the station now carries agentic_results' load arrows.
`_shared/table_load_arrows.py PREFIX OUT.stl` (conda, CPU, a second) reads the station's
three STLs (`so_body/so_bc_dir/so_bc_neu` or `hp_full_*`) and writes one arrow STL
(`so_load_arrows.stl`, `hp_load_arrows.stl`); `render_so.load_arrows` / `render_hp`'s
`geom` slot draw it in `ar_style.LOAD_ARROW` with `ARROW_FINISH`.  The glyph IS
`ar_prep.arrow_mesh` at `ar_prep.ARROW_*` × the body extent (L = 2.0 → 0.36 long), tip on
the top (ray cast onto the Neumann surface), pointing along the traction (0, 0, −1).
Sites are a fixed quincunx, not ar_prep's search (the top is one flat visible disk):
the centre + four on a ring at 0.58 R, turned 22.5° off the camera azimuth — on the
camera's own 90° grid a front shaft prints through the arrow behind it.  The printed
gap is asserted ≥ `ar_prep.ARROW_GAP`·L; a smaller ring (0.48) fails it (0.184 < 0.2).
Cost: the back arrows stand above the rim, so the cropped render is taller and the table
prints ~13 % smaller in its station (crop aspect 1.22 → 1.05).  The legend's Neumann key
is the swatch plus a flat downward arrow in the same colour (`LoopFigure.load_key`,
ar_figure.bc_keys' proportions); the legend row re-justifies around it.  Re-render only
the station with `pvbatch render_so.py g m_g` (the merged slots now take their own tags,
`m_a … m_gu`) or `pvbatch render_hp.py l_geom` — no optimiser runs.  A worktree has none
of hiddenphi's git-ignored cache: copy `hp_pre.vtk hp_solve.vtk hp_bc_masks.npz hp_e_*.stl
hp_evo_<it>.stl` from the main checkout, then `hp_cut.py` → arrows → `render_hp.py`.

**Round 7 of §5j (the user, 2026-10-14) — b paired, c the geometry clean-up; a
unchanged (pixel-identical).**  180 × 158 mm.
- **b, ONE paired row**: "Shape optimization" (0, **40**, 100) beside "Concurrent topology
  and shape optimization" (20, **43**, 90), three frames each, each run's title on two lines
  (bold run, grey iterations + wall clock).  Recovered from the deleted branch's
  `05bb8d6` (`LoopFigure._pair_geometry` / `draw_pairs`, `PG`/`PROW`/`TL2`/`PAIR_HMAX`):
  `results` given as a LIST of two runs selects it; a tuple per row keeps the six-frame
  rows `shape_opt_loop` / `hidden_phi_loop` use (shape_opt_loop re-composed pixel-identical).
  That branch's row-c examples and `optexamples/` pipeline were NOT brought back.
  `LoopFigure.results_bottom()` is the hook a subclass draws under.
- **c, geometry clean-up** (`experiments/geometry_cleanup`, its README): 14 objects in
  3 rows (4 / 5 / 5), one panel height for all (14.4 mm), each row justified to a's span.
  `som_figure.MergedFigure(LoopFigure)` draws it.  `ROWS_C` is hard-coded: the split with
  the largest common height over every 3-row split (slot = max(two panels, label)); the
  rows mix tall and wide bodies.  Two rows would be ~10 mm (change maps unreadable); a
  thematic split costs ~2.5 mm.  Per object: name (6 pt) + grey stats line (5 pt:
  "[self-weight, ]N updates, SE/SE₀ x", from `results.json`), then two panels 0.4 mm apart:
  - **BC** — agentic_results' look, `render_ar.bc_panel` verbatim: clay, flat-lit teal /
    orange faces from the session's `resolved.npz` on `geometry/physical.glb`,
    `ar_prep.load_arrows` along `case.json`'s tractions, `ar_prep.support_outline` around
    supports facing away (ar_prep's marker rule).  Gravity-only bodies (crocodile,
    elephant, rhinoceros, stone arch) have no arrow → "self-weight" in their label.
  - **Change** — the clean-up's signed change in h on the final body (`render/it_<final>.vtk`),
    drawn by `geometry_cleanup/render.py`'s change-mode code (its `view()`, `lut()`,
    normals, finish).  NOT the clean-up's PNG: its camera (0.85 × the fit distance) cuts
    off the hammer's head, both femur ends, the bench's feet.  `render_som_cleanup.py
    --check crocodile` renders at the clean-up's dolly/size and asserts the montage tile
    is reproduced (max pixel difference 1/255).
  - **Camera**: both panels the clean-up's left-hand view (`render.camera`, azimuth 35°,
    elevation 25°, +y up, framed on `camera.json`'s starting box) dollied back ×1.45 (1.3
    still cut the femur's load arrow), 1150 px; one crop box per object over both panels.
    `physical.glb` is asserted to lie within 3 h of the clean-up's iteration-0 box.
- **c's keys** right-aligned on its title row: Support swatch, Load swatch + arrow
  (`LoopFigure.load_key(label=)`), and "Change in h" −3 … 3 (`CHANGE_PTS`, RGB, asserted
  equal to `render.lut()` / `CHANGE_MAX` by parsing its source).  Row a's key keeps
  "Dirichlet" / "Neumann": same colours, the user asked for support / load in c.
- **Pipeline** (`build.sh shapeoptmerged`): `som_cleanup_prep.py` (conda, CPU only:
  `CUDA_VISIBLE_DEVICES=""`, warp's ray queries run on the CPU; writes `_cleanup/*.vtk`,
  git-ignored, and the record `som_cleanup.json`) → `render_som_cleanup.py --check` →
  `render_som_cleanup.py` (`panel_som_cu_<object>_{bc,change}.png`, committed as the
  cache) → `som_figure.py`.  The render steps run only when the panels are absent or with
  `SOM_CU_RENDER=1`; they read `output/geometry_cleanup/` and re-run nothing.  Legend
  text for b and c in `legend_notes_merged.md`.
- **Worktree caveat (still true)**: hiddenphi's render cache is git-ignored and was missing
  in BOTH checkouts.  `build_shapeoptmerged` calls `build_hiddenphi`, which RE-RUNS the
  hidden-φ optimiser when `hp_pre.vtk` is absent.  To recompose without it: copy
  `hp_pre.vtk hp_solve.vtk hp_bc_masks.npz hp_evo_{20,90}.stl` from the main checkout, run
  `hp_cut.py`, `pvbatch render_hp.py m_g_phi m_d_hid l_evo_20 l_evo_90 l_evo_cap`, then
  `som_figure.py` alone.  (The 43 frame is `hp_e_*`, written by hp_cut.)

**Round 8 of §5j (the user, 2026-10-14) — c re-titled, 17 objects, one-line labels, key at
the foot; a and b pixel-identical.**  180 × 183.2 mm (c 92.5 mm, was 66.5).
- **Title** "Generated shape finetuning", grey "(Left: Supports and loads. Right: Change
  in shape)".  **Key line under c's rows**, centred: Support, Load (swatch + arrow),
  "Change in shape" bar ticked −3*h* / 0 / 3*h*, then grey "*h* = octree leaf size" (the
  clean-up's analysis leaf, 1.92 · extent / 2^depth; full definition in
  `legend_notes_merged.md`).
- **Labels**: ONE line, name (6 pt) + grey "SE/SE₀ x" (5 pt); updates / self-weight moved
  to the legend notes.
- **Objects**: + dental-crown, pogo-stick, truck-leaf-spring ("Leaf spring") in
  `som_cleanup_prep.OBJECTS`, same pipeline/camera/×1.45 dolly; the 14 old records in
  `som_cleanup.json` are unchanged (only a `view_override: false` key added) and their
  panels were NOT re-rendered.  **One declared view override** (the user asked):
  `VIEW_OVERRIDE = {"dental-crown": (90, 25)}` in the prep, recorded per object in
  `som_cleanup.json` and used for both panels; `render_som_cleanup.place` reads the
  object's recorded azimuth/elevation and asserts every other object is at `VIEWS[0]`.
  The angle was scored (area-weighted |change| facing the camera, 15° × {10…70°} grid):
  azimuth 35 sees 0.59 of the best, 90/25 0.91; 90/10 is the best but loses the cusps.
- **Rows**: constraints robot arm + femur + crown in row 1, cake + crocodile not;
  `som_figure.py --search K` = bisection on the height with an exact bin-packing check.
  One-line labels are ~20 mm wide, so the 17 are label-bound: 3 rows → 10.3 mm (narrow
  bodies become slivers; ~10 mm was already judged unreadable in round 7), **4 rows →
  16.7 mm** (5 / 4 / 4 / 4), chosen.  `ROWS_C` is the pasted answer.
- **a/b pixel-identical trick**: matplotlib anchors the page at its BOTTOM, so a new c
  height shifts a and b by the page height's fractional pixel.  `PX_PHASE` pads < 1 px
  under c's key to keep round 7's phase (3712.859 px) — verified: no pixel differs above
  c's title.  Also: `render_hp.py`'s translucent `l_evo_*` frames are NOT deterministic
  run to run (±1…6 levels in a few pixels); re-rendering in a fresh checkout changes b by
  a handful of pixels.  Round 8 re-rendered until each frame matched the committed figure.

**Round 9 of §5j (the user, 2026-10-15) — c's text only; a and b pixel-identical, no
re-renders.**  180 × 178.8 mm.  Title "Refining generated shapes for physics" (grey
parenthesis kept); h → `$h_{\min}$` (`HMIN`) in c's key: ticks −3 h_min / 0 / 3 h_min and
"h_min = smallest octree leaf size"; labels "SE/SE₀ = x" (`se_text`).  The wider labels
shrink round 8's four rows from 16.7 to 15.6 mm (c 88.2 mm); kept on the user's word
(`--search 4` now gives 16.4 mm with a reshuffled split).

**Round 10 of §5j (the user, 2026-10-15) — c's pair arrows and key; a and b
pixel-identical, no re-renders.**  180 × 176.5 mm (c 85.8 mm).  A small arrow from each
object's BC panel to its change panel: a's pipeline arrow redrawn at 0.6 × (head, width,
line weight; `som_figure.small_arrow`, `Page.arrow`'s head is fixed in mm) in a 2.4 mm
gap (`IG`, was 0.4), 0.35 mm clear of either panel, so it can never touch a body or a
load arrow (checked on the hammer, femur and pogo stick).  The gap comes out of the
panels: round 8's rows kept, 15.6 → 14.6 mm (`--search 4`: 15.5 mm reshuffled).  Key:
"Change in shape" centred ABOVE a 16 mm bar on the swatches' centre line, ticks below;
"h_min = …" dropped from the figure (the legend notes still define h_min).

**Round 11 of §5j (the user, 2026-10-16) — b's frames and right-hand title; a and c
pixel-identical (rows above b's label and from c's label down: 0 pixels differ).**  Same
size.  Right run titled "Topology optimization" (was "Concurrent topology and shape
optimization"; same run).  b now has its OWN frames, `panel_so_m_evo_<it>.png` (committed)
and `hp_panel_m_evo_{20,cap,90}.png` (git-ignored), styled in ONE module,
`_shared/evo_style.py`, used by `render_so.py m_evo` and `render_hp.py m_evo_*`:
- **Near half fainter**: `M_GHOST_OPACITY` 0.34 → 0.12, both runs (0.20 tried first: at
  print size it barely changed).  The section face now shows through, so the frames read
  greyer; the far half's opaque body keeps each outline, frame 0 included.
- **Voids dark grey, by a rule**: `_shared/void_open.py` (conda, CPU warp, ~1 s a mesh)
  gives every shell/ghost vertex its openness — the fraction of 256 cosine-weighted
  hemisphere rays leaving the UNCUT design — written as `<base>_{shell,ghost}_open.vtk`;
  `evo_style.open_actor` ramps body colour (≥ 0.22) → `VOID_RGB` [0.40, 0.385, 0.365]
  (≤ 0.08).  Not hp_cut's `cavity_faces`: a component test loses the void once it breaks
  through to the underside (~57).  Enclosed cavity 0, tunnels a few hundredths, outside
  ≥ 0.3 except one sub-pixel crevice at shape 100 (0.2 % of vertices).  Warm grey, not a
  hue: `loop_layout.lift`'s gamma 0.72 brightens it to ~0.52, still half the body.
- **Standalone loop figures untouched**: `shape_opt_loop` and `hidden_phi_loop` still use
  `evo_*` / `l_evo_*` (0.34, voids in body colour).  Re-composed: shape_opt_loop
  pixel-identical; hidden_phi_loop (all `l_*` re-rendered in a scratch copy) 19 pixels at
  ±1, round 8's translucent-frame noise.  a's "Initial design" still crops the old
  `panel_so_evo_0`, which is what keeps a identical.
- **By hand** (hiddenphi's cache copied in from the main checkout, as round 7 says): the
  `void_open.py` calls and two pvbatch lines in `build_shapeoptmerged`, then
  `som_figure.py`.  `build.sh` now runs them (shape frames only when absent or
  `SOM_B_RENDER=1`).

**Round 12 of §5j (the user, 2026-10-19) — c's title only; no re-renders.**  c is titled
"Refining generated shapes" (round 9's "… for physics" dropped; grey parenthesis kept),
in `MergedFigure.draw_cleanup` and `legend_notes_merged.md`.  Built in a fresh worktree:
hiddenphi's git-ignored cache was absent again, so round 7's recipe (copy `hp_pre.vtk
hp_solve.vtk hp_bc_masks.npz hp_evo_{20,90}.stl` from the main checkout, `hp_cut.py`,
`table_load_arrows.py`, round 11's `void_open.py` calls, `pvbatch render_hp.py m_g_phi
m_d_hid m_evo_20 m_evo_cap m_evo_90` on the NVIDIA EGL renderer) and then `som_figure.py`
alone; row c's panel cache was not touched.  Against the committed PNG: a and c identical
outside c's title line; b's re-rendered hidden-φ frames differ in 10 pixels by 1 level
(round 8's translucent-frame noise).  Do NOT use `/tmp/somb/g20/hp_panel_m_evo_*`: that is
round 11's 0.20-opacity trial and moves ~330 k pixels of b.

**Round 13 of §5j (the user, 2026-10-19) — the BC key matches agentic_results'.**  a's
key read "Dirichlet" / "Neumann" (1.5 mm swatches edged `#888888`, a DOWNWARD arrow) and
c's "Support" / "Load" in the same glyphs; both now draw `LoopFigure.ar_bc_key`, a copy of
`ar_figure.bc_keys`' key: the words "Support" / "Load", 1.9 mm swatches edged in ink
(0.35 pt), the flat load arrow pointing RIGHT (3.2 mm, shaft 0.46, head 2.55 × the shaft
wide and 36 % of the length, `ar_style.LOAD_ARROW` #a45123), 0.7 mm swatch → arrow,
0.8 mm glyph → label.  Colours were already the same (`#12a594` / `#dd8452`, in the
legend and in every render), so no panel was re-rendered for it.  Kept at this figure's
5.5 pt (`L.TICK_FS`; agentic_results' KEY_FS is 5.6) so a's legend row stays one size.
Opt-in (`AR_BC_KEY = True` on `MergedFigure`): the standalone `shape_opt_loop` /
`hidden_phi_loop` keep the old key and were not rebuilt.  a's legend row re-justifies
around the new widths, so every key in it moves a little; the legend notes name supports
/ loads (with Dirichlet / Neumann in brackets).  Built by round 12's recipe (hiddenphi's
cache copied from the main checkout, `hp_cut.py`, `table_load_arrows.py`, the three
`void_open.py` calls, `pvbatch render_hp.py m_g_phi m_d_hid m_evo_20 m_evo_cap m_evo_90`
on GPU 0, NVIDIA EGL, opacity 0.12), then `som_figure.py`; row c's cache untouched.
Against the committed PNG: changes only in a's legend row and c's key line, plus 7 pixels
at ±1 in b (translucent-frame noise).

### §5h A numerical study whose last two panels are the plots (learned building `nitsche_convergence`)

The twist-cube benchmark actually run, in seven panels.  **Top row, the solution:** (a) the classical
body-fitted hex mesh, (b) the same box embedded in an octree, (c) the box rotated 45° in an octree —
all deformed by their own solution, coloured by `|u|` on one scale from one camera.  **Bottom row,
what the embedded scheme is really handed:** (d)/(e) each octree cut open over the phase field, in
the `bunny_sdf_octree` manner — domain wireframe, translucent ghost of the geometry inside it.
**Right, the measurements:** (f) relative `L2` error against a body-fitted reference over four levels
of refinement, (g) what those same runs cost, sharing (f)'s x axis, with the key beneath both.

Four scripts, `build.sh nitscheconv`, ~20 s warm: `nc_palette_check.py` (a **gate, not a step** — the
dataviz validator, run first so a re-coloured series cannot ship unvalidated) → `nc_prep.py` (GPU
sweep → `nc_data.txt` + `nc_stats.txt` + seven VTKs) → `render_nc.py` (pvbatch → five PNGs) →
`nc_figure.py` (compose → `nitsche_convergence.{pdf,png}`, 7.16 × 2.87 in).

The renders are the plots' key — a reader who has never seen an embedded-boundary mesh cannot read
"Diffused Nitsche, rotated" off a legend line — so they sit in a 3×2 block on the left with the two
plots stacked on the right.  **The cell under the body-fitted panel is empty on purpose: a
body-fitted mesh has no phase field, and the gap says so.**  The two colour bars live in it, `|u|` at
the top against the row it serves and `φ` below against its own.

#### Getting the numbers right

- **Drive it from the production comparison script, and change only what you can defend.**
  `nc_prep.py` imports its local `comparison_helpers.py` and calls `run_standard` /
  `run_nitsche` / `run_nitsche_rotated` with its geometry, BCs and `SOLVER_CONFIG`.  Three deliberate
  departures, each with its own bullet below: no refined quadrature (`quad_refinement_depth=0`
  everywhere — the figure's subject is plain mesh refinement, so `ε = h/2`), a different error metric,
  and a tighter Newton tolerance.
- **Measure the error as a volume integral on a fixed lattice, not as a sum over each mesh's own
  nodes.**  Both fields are EVALUATED at the cell centres of one shared 64³ lattice — a Q1 field, on
  a uniform grid or inside an octree cell, IS the trilinear interpolant of that element's corner
  values, so evaluating it is not resampling it — and the relative `L2` error is the midpoint rule
  over the whole box.  That replaces two separate problems.  (i) The source script's
  `displacement_error` matches each test node to its nearest reference node, and that `O(h_ref·|∇u|)`
  mismatch is a noise floor exactly where the finest meshes land.  (ii) Summing over a mesh's own
  nodes weights the norm by that mesh's node density — an octree carries most of its nodes in the
  fine layer near the boundary, precisely where its error is largest — so each scheme was measured in
  a different norm, and the octree's in one biased towards its own worst region.  Not a small effect:
  it inflated `rotated d=3` by **33 %** (0.238 → 0.317) and every Nitsche point by 4–12 %, while
  barely touching the body-fitted curve (uniform mesh, uniform weights).  The old number is still
  computed and written as `err_nodal`, one column over, so the two compare without a re-run.
- **An evaluator you wrote is a claim; pin it against one you did not.**  `hex_eval` locates a point
  among ~58 k variable-size cells (`cKDTree` on cell centres, widening `k` until every point lands,
  containment-tested) and reads each corner's trilinear weight off the SIGN of its offset about the
  cell centre — deliberately ordering-agnostic, because octree corner conventions are exactly the
  thing that is right until it is not.  The body-fitted mesh is a case BOTH it and
  `RegularGridInterpolator` handle, so the prep runs both on the same field at the same points and
  asserts they agree: **3.3e-16**.  Everything the octree errors mean rests on that.
- **The dof vector's layout is an assumption — assert it.**  Everything downstream is wrong by a
  permutation if `u` is not laid out like the analytic `meshgrid('ij')` lattice the comparison script
  builds.  Two asserts pin it: the clamped face is 0 and the twisted face equals the prescribed
  rotation, both to 1e-6.  (Gate at 1e-6, not machine zero — constraints are imposed to the linear
  solve's tolerance, measured 1.4e-9 at n=16 — and a wrong permutation lands at **0.75**, not 1e-8,
  so the loose gate still catches it.)
- **A convergence study is only as good as its reference, so build two.**  This solver configuration
  tops out at **96³** body-fitted: 64³ converges on the shared 20-step schedule in 134 s; 96³ *dies*
  on it (line search rejects the step at load step 19 of 20, after 20 min) and needs **40 load steps**
  to converge, in 779 s; 128³ never leaves the first load step (residual stuck at rel = 1.000, ~66 s
  per wasted iteration — the AMGX/mixed-precision solve returns nothing usable at 6.3 M dofs).  Both
  usable references were built and the whole sweep measured against each: **every Nitsche error moves
  by ≤ 0.4 %, the body-fitted curve's finest point by 15 %** (1.448e-4 → 1.660e-4, measured with the
  node-sum metric and the production tolerance, which is what the sweep used at the time) — exactly
  where a reference only 1.33× finer than the test mesh stops being one.  Ship the finer one.  More
  load steps for the reference is a continuation detail, not a discretisation change (same solution,
  same tolerance); every test run keeps `SOLVER_CONFIG`'s schedule.
- **`solver.final_residual` is the ABSOLUTE ‖R‖, and the two schemes do not share a scale.**  A
  Nitsche solve converges at ‖R‖ = 2.7e-3 because its penalty terms put ‖R‖₀ at 9.8e3; a body-fitted
  solve of the same problem starts at 0.15 and ends at 4e-11.  Checking that absolute number against
  `relative_tol` rejected **every Nitsche run** and nearly cost the coarsest two levels ("depth 3 does
  not converge" — it converges in 2 Newton iterations per load step).  The relative residual is
  `solver.relative_residual_history[-1]`; the gate uses it, and every run's value goes into
  `nc_data.txt`.
- **Reporting COST forced a change to how everything is solved.**  The first version of (g) said the
  embedded scheme was 3× cheaper (15 s against 43 s) on half the Newton iterations (40 against
  80–100).  Both were artefacts of that same asymmetry: the solver exits on
  `‖R‖ < absolute_tol OR ‖R‖/‖R‖₀ < relative_tol`, Nitsche's `‖R‖₀` is dominated by a LINEAR penalty
  term that one Newton step kills, and the relative test then passes at an absolute residual five
  orders above where the body-fitted run finishes.  Comparing iteration counts or wall clock across
  that is meaningless.  The sweep now runs every variant at `relative_tol = 1e-10` (`NC_RTOL`), where
  both end near 1e-8 absolute.  Measured consequences:
  - the ERRORS move by **0.24 % / 0.12 %** (nitsche / rotated at d=5) — the loose gate never
    contaminated the accuracy study, so (f) was right either way;
  - the COST story inverts: iterations become **80 (Nitsche) vs 80–120 (body-fitted)**, times
    **43 s vs 54 s** at h = 1/32.  At equal `h` and matched convergence the embedded solve is ~20 %
    cheaper on half the elements (59 k against 111 k) and ~75× less accurate.  That is the real
    trade; the loose-gate version overstated the cheap side by 3×.
- **Warm up before timing, and expect the coarse end to be overhead-bound anyway.**  The first solve
  of each kind in a process pays Warp's kernel JIT, so the prep throws one run of each runner away
  before the sweep (~5 s; same reasoning as `prep_octree_sdf`'s timed generation).  It did NOT flatten
  the left end of the cost curve, which is the useful finding: rotated d=3 and d=4 both cost ~2.9 s
  with 3.4× different element counts, because at that size the run is AMGX/BVH/quadrature setup, not
  arithmetic.  Don't read a speed-up off the coarse points.

#### Getting the pictures right

- **Say which scheme, not which mesh.**  The panels are titled "Classical body-fitted FEM" and
  "Diffused Nitsche, aligned / rotated 45°", and the key repeats those words exactly.  An earlier pass
  titled them by geometry ("Body-fitted hex mesh", "Aligned box in an octree") and drew the question
  it deserved — *are (a), (b) and (c) all diffused Nitsche?*  They are not, and at a glance (a) and
  (b) look nearly identical because at depth 6 the aligned box IS grid-conforming (0.75 = 24 h); what
  differs is how the BCs are imposed.  When two panels differ in a way the picture cannot show, the
  title has to carry it.
- **Crop to the geometry FIRST, then deform — that order is the whole difference between a picture of
  the mesh and a picture of the body.**  The solution panels began as "keep the cells that carry
  material, warp them", and (c) came out a sawtooth staircase: what an octree's *cells* do, not what
  the *body* does.  The prep now dumps the octree UNDEFORMED with the displacement as a `VECTORS u`
  array; the renderer cuts it to the 1.5³ box with six plane clips (exact, and their properties do not
  move between ParaView versions) and only then applies `WarpByVector`.  All three solution panels are
  then the same twisted box and compare as solutions, while the non-conformity moves to (d)/(e), where
  it belongs.  Two things to know: the clip's interpolated boundary values come from cells that
  straddle the interface, which is exactly the field the Nitsche form controls there (the solution,
  not an artefact); and the clip tetrahedralises, so `Surface With Edges` draws the triangulation —
  33 k cells in, 38 k out.
- **The `bunny_sdf_octree` treatment is two elements, and both have to be there.**  A wireframe of the
  BACKGROUND DOMAIN cube (2.0 aligned, 2.25 rotated — written from the prep so the rotated one goes
  through the same rotation as its mesh, and is therefore the thing that comes out oblique), plus a
  translucent ghost of the geometry inside it.  The ghost is the WHOLE box, not the removed half: the
  camera's screen-right is `+y`, so the two halves sit side by side rather than one behind the other,
  and a half-box seen into from its open side reads as two flat grey panels.  It also needs
  `Surface With Edges` — a translucent bunny reads because a bunny has form; a translucent box is a
  grey area until its 12 edges are drawn.  The frame costs framing: the camera opens to `r = 2.05` to
  hold the rotated domain's corners, which shrinks the mesh inside it.
- **A close-up of the interface was the wrong trade.**  At whole-box scale the diffuse band is
  `2.3 ε ≈ 1.15 h` — a 0.5 mm hairline on a 1.18 in panel — and 3× in, the band and the cells crossing
  it read beautifully while the object stops being recognisable.  A panel whose job is "this is the
  mesh the solver saw" has to show the mesh.
- **When two panels look like different resolutions, measure before believing either of them.**
  (d) and (e) are both at depth 6, cells 0.0312 and 0.0352 — 12.5 % apart, and the closest a common
  depth allows (the rotated domain is 2.25 against 2.0; the next aligned level down is 0.0625, twice
  as coarse).  On screen the drawn grids are 8 px and 13 px, a ratio of 1.63 ≈ 1.125 × **√2**: (e)'s
  cut plane meets its cells at 45°, so each cell's cross-section is √2 wider than the cell's edge.
  The cells are not coarser; their SECTIONS are.  Coarsening (d) to match the appearance would have
  misstated the mesh — instead both titles carry `h`, read from `nc_data.txt`.
- **`φ = 0.5` lands exactly ON nodes when the box is grid-aligned.**  0.75 = 24 h at depth 6, so the
  box's faces run through nodes where `s = 0`, and any strict `> 0.5` test keeps or drops those nodes
  at random — an outer layer that comes out pitted.  Use 0.4: `φ` falls off fast either side (one cell
  out is already 0.018), so it means "on or inside" unambiguously.  (The solution panels no longer
  threshold — they clip to the box — but the trap is general, and `nc_prep.py` still selects the
  dumped cells with `φ > 0.5`, where the *any-node* form makes the ambiguity harmless.)
- **A render that clipped its object cannot be rescued at compose time, so assert it there.**  At
  fit `r = 1.08` the rotated octree touched the bottom of its 1500² viewport; the ink crop cannot
  invent the missing rows, and the panel just ends flat against the figure edge — which looks like a
  layout bug, not a render bug.  `nc_figure.py` asserts every panel's ink box clears its own border by
  a pixel, and that assert is what found it.
- **Two fields on one page need two ramps, and the older figure decides which keeps its hue.**  Both
  `|u|` and `φ` are magnitudes, so both take a sequential ramp (dataviz `color-formula.md`), and they
  cannot both be blue.  `φ` keeps `diffused_nitsche`'s validated blue — the phase field should look
  the same in both figures — so `|u|` takes a warm one, GENERATED rather than picked: the same OKLCH
  lightness ladder as that blue (L 0.905 → 0.338 in even 0.095 steps), hue fixed at 52°, chroma at
  92 % of the in-gamut maximum at each step, so equal fractions of the two fields carry equal
  perceptual weight.  (The house diverging map's warm arm was tried first: its lightness steps run
  0.049 → 0.168 and its hue walks 70° → 30° — not a single-hue sequential ramp at all.)  `|u|` then
  stops at the SIXTH step, not the seventh: it covers the whole of every solution panel, and at
  L 0.34 the twisted end went to a near-black brown that swallowed its own cell frames.
- **Thin plot lines need more contrast than filled cells, and the house palette does not have it.**
  Two of three series had to move (`nc_palette_check.py`, the dataviz validator ported to Python —
  there is still no `node` on this box): seaborn's orange `#DD8452` is **2.8:1** against white, under
  the 3:1 gate → `#c8622a` (4.0:1); a mid-grey baseline `#59606b` is **ΔE 10.9** from the house blue,
  under the normal-vision floor of 15 → `#3b4250` (19.2).  Final worst pair: CVD 18.5, normal 19.2,
  every line ≥ 4:1.  The baseline is a **neutral by design** — it is what the other two are measured
  against, not a third competitor — so the lightness band and chroma floor are waived for it (they
  would forbid any neutral at any lightness) while every separation gate still applies; the checker
  encodes that as a `role`, it does not just skip the row.

#### Getting the page right

- **One ink box per ROW, one aspect for ALL five.**  The crop has to be per row — the three solution
  panels share a camera and must stay at one scale, and so do the two φ panels, but the two rows do
  not share a framing.  The letterbox is then common to all five, so every panel is the same size: a
  grid of unequal cells reads as a hierarchy that is not there.  It costs the φ panels some side
  padding, since their ink is squarer than the solution renders'.  That is the price of the grid.
- **The key belongs under both plots, not inside one of them.**  While it lived in (f)'s top-right
  corner it cost that panel a decade of headroom bought purely to clear the curves, and it forced (f)
  to be taller than (g).  Beneath both, the two plots are the same height — they are two measurements
  of the same twelve runs, so neither is the senior partner — and both keep their height for data.
  It is **one column, three rows**: three of these labels side by side need 3.7 in against a 2.6 in
  column, and at `ncol=2` matplotlib fills column-major, which put "rotated" above "aligned" when read
  across.  The fitted orders left the labels at the same time — they are in the caption and in
  `nc_data.txt`, and a key crammed with `h^1.8` was what made every row too tall to start with
  (mathtext superscripts set the row height).
- **One quantity per axis.**  (g) plots time; the Newton count is flat in `h`, so it is a line of TEXT
  in the panel, not a horizontal curve pretending to be data, and a second y-scale is out (dataviz:
  never a dual axis).  (g)'s title carries the load-step count — 20 steps × ~4 Newton iterations is
  what every number in it is made of.
- **Place the Newton note by measuring it against the curves.**  Three hand-picked spots each failed
  on some variant (top left as one line hit the rotated curve's last point; bottom right as one line
  hit the coarsest points; bottom right as three lines, half the panel tall, hit the uniform runs'
  1/16 points).  `nc_figure.py` now tries top left, bottom right, top centre in that order, keeps the
  first whose box clears every plotted segment and marker, and fails the build if none does.  Top
  left wins on all four figures: cost rises with 1/h, so that corner is empty by construction.
- **Name the OBJECT on the axis, not just the element.**  `Element size h (box size 1.5×1.5×1.5)`:
  `h` alone says nothing about how many elements are across the body, and converting between the two
  is the first thing a reader does.  Both numbers come from `nc_stats.txt`.
- **Assert the layout, don't eyeball it** (as §5c): every title inside the canvas; adjacent titles
  clear of each other; a row-2 title clear of the panel above; the rightmost title in each row clear
  of the plots' y labels (they sit 0.30 in away now); the colour bars inside their own cell; the key
  clear of the x label and of the page edge.  Each of those caught something.

#### Caching, and what it costs you

- **Cache the sweep, not just the reference.**  Twelve nonlinear solves is ~4 min and every iteration
  on the renderer or the compose step needs none of it, so `nc_prep.py` skips itself when its nine
  products (`nc_data.txt`, `nc_stats.txt`, seven VTKs) are all present, and `build.sh` deletes only
  the panel PNGs.  Warm, the target is ~20 s.  Two things keep it honest: an INCOMPLETE cache re-runs
  and says which product was missing, and **`NC_FORCE=1` re-runs unconditionally — use it before
  trusting a from-scratch rebuild**, because with the cache warm this target no longer proves the
  sweep still works.  §6's rule about incremental runs hiding breakage now applies here by
  construction.
- **Pin the GPU when the figure plots wall clock.**  `NC_GPU` (default 0).  (g) is a timing panel;
  numbers measured on whichever card happened to be free are not comparable between rebuilds.

**What it measures** (relative `L2` in `u`, volume-weighted on the shared 64³ lattice, vs a
body-fitted 96³ reference; the shipped float32 / JAX-AMG sweep at `relative_tol = 2e-5`, rebuilt
2026-09-30 with the solver's BC tolerance `max(1.5ε, 0.75h)` and `beta_safety_factor = 5`;
`err_nodal` in `nc_data.txt` is the older node-sum, for comparison):

| `h` | Classical body-fitted | Diffused Nitsche, aligned | Diffused Nitsche, rotated 45° |
|---|---|---|---|
| 1/4  | 7.14e-3 | 4.383e-2 | 9.481e-2 |
| 1/8  | 2.13e-3 | 2.030e-2 | 4.544e-2 |
| 1/16 | 6.20e-4 | 8.900e-3 | 1.684e-2 |
| 1/32 | 1.62e-4 | 4.059e-3 | 8.907e-3 |
| **fitted order** | **1.82** | **1.15** | **1.17** |
| **s / Newton it. at 1/32** | **20.8 s / 100** | **16.3 s / 60** | **12.6 s / 60** |

The two settings cut every Nitsche error about 3× (at 1/32: aligned 1.31e-2 → 4.06e-3, rotated
1.80e-2 → 8.91e-3) and leave the rate where it was.  The rate is first order because the penalty holds
`u = ū` across the inner half of the diffuse band, which moves each Dirichlet face ~ε into the
body (project memory `nitsche-first-order-root-cause`).  Both Nitsche families converge at the same
rate, and full non-conformity costs a **constant factor of ~2.2**, not the rate.  The body-fitted
scheme is second-order and ~25× more accurate at equal `h`, for about the same money — the honest
framing of what an embedded method buys (any geometry, no meshing) and what it costs.  The uniform
companion measures the same to within 3 % (orders 1.15 / 1.16).  The two float32 sharp-volume
companions fit 1.42 / 1.29 (uniform) and 1.45 / 1.21 (adaptive) over the same depths 3–6.

| Knob | Value | Where |
|---|---|---|
| geometry / BCs / material | local `comparison_helpers.py` (1.5³ box, θ = 45°, E = 10, ν = 0.3) | imported |
| depths | 3 → 6 (`h = 2/2^d`; rotated `2.25/2^d`, 12.5 % coarser at the same depth) | `NC_DEPTHS` |
| quadrature refinement | **0** — the figure's subject | `nc_prep.py` |
| ε | `h_coarse/2` (the `r=0` default) | `run_nitsche` |
| Newton tolerance | **2e-5** for every target | `NC_RTOL` |
| precision | **float32** for every target (the project rule for Nitsche runs) | `NC_USE_MIXED_PRECISION=1` for float64 |
| linear system (sharp companions) | scaled to unit diagonal; JAX-AMG iteration cap 4000 instead of 1000 | `NC_SCALE_DIAG`, `NC_AMG_MAX_ITERS` |
| reference | body-fitted **96³**, 40 load steps, 779 s, cached in `nc_ref_96.npz` | `NC_REF_N`, `NC_REF_LOAD_STEPS` |
| body-fitted comparison meshes | `n = 0.75·2^d` = 6, 12, 24, 48 (same `h` as the octree) | `_std_n` |
| error metric | volume-weighted `L2` on a 64³ sample lattice (`err`); node-sum over `0.1 < \|z\|/hs < 0.9` kept as `err_nodal` | `NC_SAMPLE_N` |
| solution panels | crop to the 1.5³ box (6 plane clips), THEN `WarpByVector` | `crop_to_geometry`, `warp` |
| φ panels | domain wireframe + whole translucent 1.5³ ghost, cut at `y = 0.006` (off the cell boundary) | `domain_frame`, `ghost_geometry`, `CUT_Y` |
| camera | `(1.15, −0.95, 0.62)`, up `+z`, 25°; solution panels fit r = 1.35 × 1.10, φ panels r = 2.05 about the origin (the domain frame sets it) | `render_nc.py` |
| caches | reference `nc_ref_96.npz` (~13 min) and the sweep itself (`nc_data.txt` + 7 VTKs, ~4 min) | `NC_FORCE=1` re-runs |
| GPU | 0 by default — (g) plots wall clock, so it must be the same card every time | `NC_GPU` |
| figure | 7.16 × 2.87 in, panels 1.18 in, plots 2.6 in wide and EQUAL height, key beneath both | `nc_figure.py` |

`build.sh nitscheconv` runs gate → prep → render → compose and deletes only the panel PNGs.
Keep this original target on its established adaptive depth-3–6 sweep: its body-fitted meshes are
6³, 12³, 24³, and 48³, and its two embedded meshes retain the depth-3 far field. Do not add the
96³/depth-7 point or a JAX preallocation override here; those were tried and then deliberately
removed so the existing figure and build behavior stay unchanged. A forced GPU verification run
(`NC_FORCE=1 NC_GPU=1 ./build.sh nitscheconv`) completed all 12 cases after the shared runner gained
the configurable initial depth. Against the checked-in baseline, the convergence orders were
identical (1.82, 1.13, 1.21), the maximum absolute error change was 2.3e-9, and wall time varied by
at most about 9%. The regenerated original artifacts were restored after that check so only the two
new companion experiments contribute generated-output changes.

The fixed-depth companion is `build.sh nitscheconvuniform`. It runs the same benchmark and figure
code with `NC_UNIFORM=1`, passing `init_octree_depth == octree_depth` at every level for both embedded
variants. Its independent sweep cache lives in `nitscheconv_uniform/`, while the identical 96³
body-fitted reference cache is shared from `nitscheconv/`. The prep asserts one cell edge length on
the actual trimmed FEM mesh before accepting each Nitsche result. It writes
`nitsche_convergence_uniform.{pdf,png}`.

The sharp-volume companions are `build.sh nitscheconvuniformsharp` (fixed depth, cache in
`nitscheconv_uniform_sharp/`) and `build.sh nitscheconvsharp` (adaptive, cache in
`nitscheconv_sharp/`).  Both run float32 at depths 3–6 like the rest, with `NC_SCALE_DIAG=1
NC_AMG_MAX_ITERS=4000`.  Under `sharp_volumetric` the exterior cells carry only the φ-floor
stiffness, so the tangent's diagonal spans decades.  Unscaled, the float32 sweep fails at depth 6:
the aligned uniform tangent sits at the float32 rounding floor, Newton creeps from load step 11, and
an AMG solve stalls 300× above tolerance.  (An earlier version of this sweep failed sooner, at
rotated depth 4, which is why it once ran float64 at depths 3–5; float64 aligned depth 6 exhausts the
47 GiB A6000.)  Scaled, both uniform depth-6 cases converge in 40 Newton iterations.  The adaptive
rotated depth-6 solve still needed more than the default 1000 AMG iterations (a β cap made it
worse), hence the 4000 cap, set on both targets.  The
solver's sharp-only rule that a partly claimed, wholly exterior element is claimed whole
(`_evaluate_bcs_at_positions`) is part of the fix too: a floor-only cell pinned at some of its
points and not others stalled AMG.
- **(g) has a step at the coarse end in both sharp figures, and it is not a trend.**  The scaling
  costs ~1.4 s per new mesh size (JAX recompiles for the new shape), which dominates the few-second
  depth-3 and depth-4 solves (depth 4, standard: 1.20 → 2.55 s; Nitsche: 0.71 → 2.10 s).  The
  body-fitted runs in those figures pay it too, because `NC_SCALE_DIAG` is set for the whole sweep.
`nc_data.txt`, `nc_stats.txt`, the VTKs and `nc_ref_96.npz` all stay: the first two are the record of
what the sweep measured, the rest are its cache.  Warm, the target is ~20 s; `NC_FORCE=1` puts the 12
solves back, and deleting `nc_ref_96.npz` puts the 13-minute reference back.

---

### §5k The challenge figure, before Figure 1 (2026-10-17, `challenge_overview`)

Introductory and illustrative, not technical.  7.09 × 2.91 in (180 × 73.9 mm), Arial, ONLY
`method_overview`'s type sizes (6.0 / 6.5 pt), its arrows and identity colours; violet is
MechFoundry's alone.

**Layout, read left to right.**  A speech bubble, with NO column head over it, holds the
object's name only ("A sitting rabbit").  It forks into the three lanes' headers: **Human engineer** (clock,
"Slow and manual"), **AI agent** (a chat-bubble icon, grey "!", "Error-prone" in the same grey as "Slow and manual"; NOT violet;
the robot icon means MechFoundry's agents only), **MechFoundry** (violet robot + a SHADED
person: no human needed; "Fully automatable").  Then five columns: 1 Shape (AFTER the roles, the author),
2 Mesh, 3 Support & load, 4 Materials, 5 Design.  In column 1 the AI agent and MechFoundry
share ONE "Shape generation" cell spanning both lanes (the same generated shape).  There are
NO arrows from the AI-agent row down to the MechFoundry row (removed at the user's request,
2026-10-17; SGAP 0.12 vs RGAP 0.07 still sets the solved lane apart).

| | Human engineer | AI agent | MechFoundry |
|---|---|---|---|
| 1 Shape | CAD modeling | Shape generation (shared) | (shared) |
| 2 Mesh | Partitioning and meshing | Mesher failure | Adaptive meshing |
| 3 Support & load | Handbook and manual pick (a book icon) | Mask functions (code returning True per mesh node: "z < 0.05") | Segmented part selection (pressing arrows down onto the ears, pins under the feet) |
| 4 Materials | Part-by-part setup | One material | Material per part |
| 5 Design | Remesh | Repeated mesher failure | Auto remesh |

The load in "Segmented part selection" PRESSES the ears DOWN (the user, 2026-10-17; it used to
pull them up and back).  `ear_presses()`/`press()`: one vertical arrow per ear over the middle of
that ear's hump of the upper edge (the right ear's slanted top halfway out, which also keeps it off
the cell's centre line under the incoming violet arrow), its tip PRESS_GAP = 0.012 in above the
highest edge point under the WHOLE head (so the wings never sit on the slanted edge), every arrow
PRESS_LEN = GAP long (the lane arrows' length), lw 1.1; `press_lim()` raises that cell's frame
top just enough (solved, not typed) that the tails stay TOP_PAD below the cell's top edge, which
makes this one bunny ~2.5 % smaller than before (frame top 1.16 → ~1.19).  No other cell carries a load arrow (the Design
cells' arrows are magenta surface motion, not a load).

Every cell label is a NOUN PHRASE (the author asked for consistent grammar; the engineer's
Design label "Remesh" is theirs, 2026-10-27, shortened from "Manual/Constrained remesh").  The CAD bunny uses the row's own frame (`BN_LIM_R`), with a height
dimension on its right.  "Isolated components" is set on two lines.

The generated shape is drawn WITHOUT element edges (the author); its defects are an ISLAND off
the back and a SELF-INTERSECTION: on the scan's own tail (do NOT enlarge it), a segment runs
down from the notch between the back and the tail, HALFWAY (it does not cut through the tail),
with a curved shaded crescent of the same length on its tail side; never holes, an extra ellipse
or dashed lines.  Each defect is named where it sits, grey with a leader: "Self-intersection" under the tail,
"Isolated components" above the island.  No sparkles/stars anywhere.  The mask chip has a teal
border and points at the selected nodes with a teal arrow.  The engineer's partition runs ALONG
the CAD model's own part curves, inked solid black (not new dashed red cuts); ears, head and tail
are meshed, the rest not yet.  The MechFoundry row's step cells AND the shared "Shape generation"
cell are one light violet (#efecf8); only the MechFoundry header keeps its white box.  Every lane arrow runs level with its boxes' vertical
centre, and each header feeds column 1 with its own arrow (the agent's and MechFoundry's both
into the shared shape); HGAP = GAP so that arrow is as long as the others.  Text in fields
(TPU, PLA, the mask) is centred on its capitals (`ptext_mid`), not on matplotlib's full box.  The agent's
"Repeated mesher failure" cell meshes the redesigned body and flags the elements of radius ratio
< 0.3 (all at the ear/body junction, where the thinned body meets the kept ears at sharp
corners) with one red ring: the failure is low mesh quality at sharp corners, NOT a
self-intersection carried along with the optimized body (author's choice).  No commas in labels ("and").
Labels may wrap to two lines (cells carry a two-line band).  The mask cell shows the surface mesh's own nodes.  No cursor in
"Part-by-part setup".  MechFoundry's octree is
refined round the island too, but the island is not solid (drawn as a dashed ghost).

**Revision of 2026-10-17 (the user), as rules in `ch_figure.py`:**
- **Type sizes = method_overview's, like for like**, read from `ov_figure.py`'s own
  `TITLE_FS, CARD_FS, LEGEND_FS, TAB_FS = ...` line by `ast` (no import: it would run its
  compose), and the gate refuses any text at another size.  Cell labels = its block titles
  6.5 (were 6.0); column heads and lane names = its bold tabs 6.0 bold (were 6.5 and 7.0
  bold); the prompt = its prompt card 6.0 italic (was 6.5); badges and every note inside a
  picture (defect names, TPU/PLA, the mask) = its legend text 6.0 (notes were 5.0); the red
  "?" marks take the title size 6.5 bold (were 7.0).  At 6.5 pt the label pitch `LINE_H` is
  0.094 in (rule unchanged: the "lp" line box + LEAD_PT), so the figure grew 2.873 → 2.909 in.
- **All grey notes of the Shape-generation cell at one size** (asserted); the two lines of
  "Isolated components" sit one label pitch apart at their own size, not a typed data step.
- **Text fields are sized by their text**: the mask chip is its text + `FIELD_PAD` 0.018 in
  round the capitals, the TPU/PLA fields its cap height + 2 `FIELD_PAD`.
- **The two left columns are fitted to what they hold** (not typed): bubble = prompt width +
  `BUBBLE_PAD` 0.10 a side (0.84 → 0.782 in), header column = `HDR_IN` + max(lane name,
  `HDR_BADGE` + badge) + `HDR_OUT` (0.92 → 0.903 in); the step columns get the rest
  (0.893 → 0.908 in).  This, not a smaller font, is what made "Manual/Constrained" (0.818 in
  at 6.5 pt, from the old label "Manual/Constrained remesh", now "Remesh") fit its cell's picture width; `label()` now asserts every line fits
  (`wrap()` keeps an over-long word whole and would not catch it).
- **The header rule spans only the five step columns** (column 1's left edge to column 5's
  right edge); the "PROMPT" head is gone.
- **Centring rule, every cell**: a frame (`BN_LIM_R` & co.) fixes a drawing's SCALE and
  height (one baseline and size per row), then the drawing is slid sideways so its VISUAL
  bounding box (every patch, line, collection and text in the axes: cursors, dimension
  lines, the book, the mask chip, defect names; clipping turned off so the box is what is
  seen) is centred in the cell's picture area (cell less BP), to ≤ 0.002 in, asserted, and
  must stay inside the cell less 0.01 in.  Vertically the frames stand (row baselines; the
  load-arrow cell is solved to fit its top margin).  The build prints each slide: the CAD,
  partition and part-by-part pictograms moved +0.029, +0.071, +0.018 in; the mask cell
  −0.039; the rest ≤ 0.014.

**Pictograms are computed** (`ch_prep.py`) from the display bunny's side silhouette: a
Delaunay mesh of it with dropped faces and a torn fragment; parts cut by a line under the
ears, two discs (head, tail) and a strip (feet), in `SEGMENT_TINTS` with feet teal / ears
orange; a quadtree (uniform to depth 3, depth 5 at the outline) shaded by the SIGN of the
distance only; a redesign = inward offset + one hole, feet and ears kept, meshed again for
the two lanes that must remesh.  Material colours are `e_rgb` soft/stiff, the same two in
every lane (TPU = ears, PLA = body).

**Settled with the author (2026-10-17), do not re-litigate without them:**
- Few words: one 1–3 word label per cell; evidence goes in the caption.
- Show ONLY what MechFoundry solves: no body-fitted meshing or mesh repair of generated
  shapes; supports, loads and materials assigned to named parts; redesign without remeshing,
  holes allowed.  NO solve column (no advantage there: commercial solvers do nonlinear
  elasticity), no value-check column.
- No method_overview panels: flat drawings only, so Figure 2 stays the method.
- The engineer's part is CAD (faces exist, so "Pick faces"); when the mesher fails on it the
  engineer partitions the model by hand ("Partition, mesh").  A design change means remeshing
  by hand, with NO re-pick (the author).  For the agent, the meshing algorithm fails on the
  generated shape ("Mesher fails", not "mesh breaks").  Wording fixed by the author: "Design"
  not "Redesign", "Support & load" not "Fix & load", "Generate shape" not "Generate mesh",
  "Adaptive meshing" not "Octree, no repair".
- MechFoundry DOES remesh after a redesign (the octree is rebuilt): the claim is that it is
  automatic and never needs repair, "Auto remesh", never "no remesh".
- Rejected after review: "No meshing" (the octree is a mesh; "Octree, no repair"), "Pick
  every face" (a generated mesh has no CAD faces), "Often fails" (uncited; "Error-prone"),
  a violet agent lane (violet = ours in Figure 2), a green redesign cell, an SDF colour ramp
  with no legend.
- `FAIL` `#c0392b` is the one new identity colour; `ch_palette_check.py` gates it.

**Building it (merged into dev 2026-10-17; two fixes so it rebuilds off the author's machine).**
- **shapely.**  `ch_prep.py` cuts the silhouette with GEOS (`buffer`, booleans, topology-preserving
  `simplify`), and mechfoundry-warp117 has no shapely.  A dependency-free port was tried and rejected:
  manifold3d (Clipper2, in the env) matches the areas to 1e-7 but not the vertices (the inward
  offset has 308 vertices against GEOS's 300), so the drawn meshes would change.  Instead
  `build.sh`'s `ch_prep_python` runs the prep under `$CH_PREP_PYTHON` if set, under mechfoundry-warp117
  if it ever gains shapely, else under a private venv
  `~/.cache/implicit_optimize/venv-shapely-2.0.7` (`--system-site-packages` on mechfoundry-warp117, ONLY
  `shapely==2.0.7` added with `--no-deps`; made once, needs pip/network that once; never install
  into a shared conda env).  The pin also pins GEOS (3.11.4, bundled in the wheel), which is what
  keeps the geometry fixed: with it `ch_stats.json` is byte-identical to the PR's
  committed one, and `ch_geom.npz` is the same on every run.  The venv's python is run with the env's `lib/` first on `LD_LIBRARY_PATH` (the system
  libstdc++ is too old for the env's compiled modules).
- **"texts 'Partitioning and' and 'meshing' touch" (5 wrapped labels) was the gate's ruler, not the
  layout.**  The gate measured text extents on the Agg canvas at the figure's default 100 dpi,
  where every metric is rounded to whole pixels: each 6 pt line box came out 9 px = 0.090 in,
  against 0.0775 in at 400–600 dpi and a 0.088 in line pitch, so the two lines of every wrapped
  label "overlapped" by 0.002 in that exist only in the rounding (it depends on `figure.dpi` and
  the FreeType build, which is why it passed for the author).  Fix, as rules in `ch_figure.py`:
  the gate draws at `PNG_DPI` = 400 (one pixel = 0.0025 in, a quarter of the real 0.010 in gap)
  and restores the dpi afterwards; and `LINE_H` is no longer a typed number but the font's own
  "lp" line box (matplotlib sizes EVERY line by that pair; measured on the TextPath outline, no
  dpi) plus `LEAD_PT` = 0.85 pt, rounded to 0.001 in (= 0.088 in, the author's value, so the page
  does not move; unrounded, the 1.7e-5 in excess re-antialiases ~750 PNG pixels).  The lines stay
  separate texts so the gate still checks each one.  Output is pixel-identical to the PR's code run
  here with `CH_DRAFT=1`; against the PR's committed PNG only glyph antialiasing differs (that PNG
  was rasterised by another matplotlib/FreeType), plus ≤1 px on the dimension line and the ear
  junction of the redesign outline.  Never "fix" this gate with `CH_DRAFT=1`.
- The build writes only the root `challenge_overview.{pdf,png}`; the stale copies the PR had
  committed inside `challenge/` were removed.

**Caption (draft).**  *From an object's name to a simulated, redesigned part.*  The same prompt
is taken up by an engineer, an AI agent and MechFoundry.  The engineer (top) models the shape in
CAD; when the mesher fails they partition the model by hand and mesh it block by block, pick the
support and load faces and look their values up in references and handbooks, assign a material to each part, and remesh by hand after every design
change; geometry preparation and meshing dominate an analyst's effort (Hardwick et al.).  The AI
agent (middle) has a generative model make the shape, a triangle mesh with no CAD faces that can
carry islands and self-intersections: the meshing algorithm fails on it, supports and loads
become coded masks over the mesh nodes such as "z < 0.05" (boundary conditions are the least reliable step
for current agents; FEABench, HELM), the whole shape gets one material, and after each design change
the mesher fails again on poor elements at the sharp corners.  MechFoundry (bottom, Fig. 2), fully automatable by agents and starting from the same
generated shape, meshes it adaptively on an octree of the signed distance (refined near the island, which
is not kept as solid), selects segmented parts as supports, loads and material regions, and remeshes
automatically after each design change (magenta: the surface moves, a hole opens) while the
supports and loads stay attached.

### §5k-bis The opening page: challenge above Figure 1 (2026-10-17, `intro_overview`)

The user asked for the challenge figure "appended above the method overview", in its own folder.
`intro_overview/io_compose.tex` is PLAIN pdfTeX (no LaTeX class or package; `pdftex` is in
`/usr/bin`, while pypdf/pikepdf/cairo are not in any env here): it embeds the two root PDFs as form
XObjects, so every path, glyph and image is exactly what each figure's build wrote (vector stays
vector, method_overview's rendered panels keep their resolution, its Type 3 fonts stay Type 3),
and stacks them with nothing rescaled.  The page width is the figures' common 180 mm (510.264 bp;
the script REFUSES to run if the two widths differ, since a rescale would give two text sizes),
and the height is measured: 210.2 + 2 mm gap + 202.5 pt = 416.86 bp = 7.087 × 5.790 in (180 ×
147.1 mm; 207.6 pt on top before the 2026-10-17 type-size revision).  The 2 mm gap adds to each figure's own ~1.5 mm margin.  PNG: `pdftocairo -r 600`
(4253 × 3453).  **Panel letters a (challenge, top) and b (Figure 1, bottom)**, added at the
user's request (2026-10-17) in the lettered figures' house style (`loop_chain.LETTER_FS`: 8 pt
bold lower case Arial, ink #1a1a21, Type 42).  pdfTeX has no Arial metrics, so
`io_letters.py` draws them with matplotlib on ONE transparent page of exactly the stacked page's
size (sizes from `pdfinfo`, the gap read from `io_compose.tex`), and the tex lays it over the stack
LAST (both figures paint an opaque white background, so a letters form placed first is hidden;
`\shipout\hbox{\hbox to 0pt{stack\hss}letters}`) and refuses a letters page of another size.
Rule: ink starts 1 mm from the left edge (loop_chain's ML) for both; vertically each has its
own rule (the user, 2026-10-17: one shared corner rule left "a" ~1.6 mm above the challenge's
first text).  **a** sits ON challenge_overview's column-head baseline, MEASURED (pdftotext
-bbox finds "MESH", flat-bottomed capitals; the lowest ink row in that box on the 600 dpi
raster is the baseline: 4.44 mm below the top, 1.67 mm lower than before); the row's left end is free since "PROMPT"
went.  **b** keeps an ascender 0.75 mm below method_overview's top edge.  The top-left corner is the only
spot free in both: method_overview's violet frame runs up 1.5 mm in and turns into the "AI
operator" pill; at a 1 mm inset "b" came 0.45 mm from that corner.  Gate in `io_letters.py`: each
sub-figure rasterised alone at 600 dpi must be white within 0.5 mm of the letter's ink box
(measured: a > 9.9 mm (the search cap), b 0.65 mm clear).  Both sub-figures stay the vector forms their builds wrote.  Known wording mismatch left to the author: the challenge prompt says
"A sitting rabbit", Figure 1's card "a sitting rabbit".  `./build.sh introoverview` rebuilds
neither figure (method_overview needs the GPU and `overview/assets`), so run `challenge` /
`overview` first when either changes; it is last in `all`.

### §5i Figure 1: one forward/backward loop on one bunny (learned building `method_overview`)

The first figure of the NMI submission: every way a shape gets in, the pre-processing it goes
through, and one differentiable FEA pass and its gradients, all on ONE bunny.  Rebuilt from scratch
on 2026-09-28/29 over fourteen reviewed drafts (`output/overview_review/draft*.png`).  It replaced
a first version that EXTRACTED panels from the other figures' PDFs (the retired part is summarised
at the end of this section).

**What it shows** (7.09 × 2.80 in, 180 mm, Arial only):

- **Shape generation** starts from ONE text prompt, "a sitting rabbit", as the workflow's geometry
  card does since SDXL can draw an image from text (`draw_image`, 2026-09-29).  The generation
  part is two sub-columns:
  - The prompt card sits level with the forward row's centre line.  It goes straight into
    **Text2CAD / CAD-Coder**, and on through that block into segmentation as one straight line.  It
    also branches down into **TripoSG** (its text mode) and down into **SDXL**.
  - **SDXL**'s drawing goes up into TripoSG (its image mode; the arrow joins the prompt's, text OR
    image, one head) and down into **Hunyuan3D**.
  - That order (CAD, TripoSG, Hunyuan3D, top down; prompt above SDXL) is the only one with no
    crossing arrows: TripoSG takes both sources, so it has to sit between them.
  - The three shapes merge into one arrow into segmentation.  The user's rule: TripoSG, Hunyuan3D and
    the texture step all SHOW THE SAME MESH (the display bunny; grey, painted, painted), and only
    the CAD lane has its own part.
- **Forward row**: Segmentation → BC regions → Octree + *s* → Trimmed mesh + φ → Nitsche solve.
- **Second row**, each block under its partner at the same width, its panel at the same size:
  Texture (Optional) (under segmentation) · (Heterogeneous) Materials (under BC regions, and
  joining the BC arrow into the octree) · ∂J/∂s (under the octree) · ∂J/∂φ (Optional) (under the
  trimmed mesh) · Objective *J*, a small pill under the solve.
- **Octree after pre-processing (the user, 2026-10-16).**  The octree used to come first, as a
  pre-processing step.  It is rebuilt every shape-optimization iteration, so it now sits after BC
  regions, and it and ∂J/∂s are in the green FEA & optimization part (`ov_style.PANEL_PART`
  octree = "fea"; `ov_p_octree.png` was re-rendered for its tint, nothing else).  The loop
  octree → trimmed mesh → solve → J → ∂J/∂s → octree closes over green blocks only: no arrow
  goes back to segmentation, BC regions, texture or materials.
- **Backward**: J → a lane under row B → ∂J/∂s → up into Octree + *s* (solid magenta), and J →
  ∂J/∂φ → up into the trimmed mesh's φ (dashed: the optional hidden phase field).
- **Who drives it**: the frame has TWO tabs on its top line, the two ways the whole pass is run.
  - The violet tab (a `>_` prompt and a robot head) says the AI operator drives every step via
    the workflow CLI (called the meta-agent before 2026-10-16).  The AI operator has its OWN icon: the star belongs to the LLM/VLM
    SUBAGENTS inside the steps (the user, 2026-09-29), and the legend says "subagent".
  - The grey tab (a UI-window and a person icon, drawn in the ink family so it never reads as an
    agent) says human users drive every step via the web UI.  This is the Streamlit app, which
    shares the checkpointed graph with the CLI.
- **Marks and parts**, both legends on the frame's BOTTOM line (the second tab took the top line's
  room):
  - The three workflow parts (shape generation / pre-processing / FEA & optimization) are LIGHT,
    EDGE-LESS block tints; their legend sits at the left.
  - The marks' legend sits at the right: a violet star = "LLM / VLM subagent can assist", a
    green chip = differentiable GPU code in NVIDIA Warp.
  - `ov_figure.py` asserts that the two legends do not meet and that both tabs stay inside the
    frame.

**Pipeline.**  Stage 1 is opt-in and cached; everything else is deterministic and rebuilt:

```
./build.sh ovassets [input paint segments cadmodel cutout]   # once; OV_ASSETS_FORCE=1 redoes a step
./build.sh overview                                      # every build, GPU ${OV_GPU:-1}
  ov_palette_check (gate) → ov_style (gate: d/φ ramps == render_dn) → assets gate
  → ../bunny/prep_bunny_sdf.py → ov_bunny_loop.py → ov_surfaces.py
  → render_ov_panels.py (pvbatch) + render_ov_solve.py (pvbatch, dn (f)'s renderer) → ov_figure.py
```

| Script | Env | Does |
|---|---|---|
| `ov_assets.py` | mechfoundry-warp117 (+ subprocess envs) | the generative assets into `assets/`, each file in `assets/MANIFEST.json` with sha256, command, env, model, seed, prompt |
| `ov_manifest.py` | — | `asset(rel)`: a path, refused unless its sha matches the manifest |
| `ov_bunny_loop.py` | mechfoundry-warp117, GPU | `runpy` of `nitsche/dn_prep.py` (its asserted 45 082-node / 35 442-hex octree), the BC sets, one forward solve and both gradients; writes `ov_grad_full.vtk`, `ov_trim_bc.vtk`, `ov_bc_{ears,base,rest}.stl`, `ov_bc_faces.npz`, `ov_loop_solve.vtu`, `ov_loop.json` |
| `ov_surfaces.py` | mechfoundry-warp117 | the renderable surfaces `ov_s_*.vtk` (painted, PartField tints, BC colours, materials, CAD) + `ov_surfaces.json` |
| `ov_style.py` | plain Python | ONE camera / cut / colour table for pvbatch, the prep and the compose |
| `render_ov_panels.py` | pvbatch | every `ov_p_*.png` except the solve |
| `render_ov_solve.py` | pvbatch | the solve (shared with `diffused_nitsche` (f)) |
| `ov_figure.py` | mechfoundry-warp117 | the compose + every layout assertion |
| `ov_cad_tessellate.py` | mechfoundry-cadcoder | a STEP → triangles with B-rep face ids (cadcoder sandbox's `tessellate`) |

**The user's brief, as it stands after fourteen drafts (the later rule wins; do not re-litigate):**

- **Titles only.**  No timings, counts, formulas, colour bars or arrow labels.  The exceptions
  are the two legends and the words "(Optional)" / "(Heterogeneous)" in titles.  `ALLOWED_TEXT` in
  `ov_figure.py` is the registry of every string the page may carry, and the build fails on any
  other.
- **Surface panels = `pfbunny/render_pf_bunny.py`'s recipe** (mirrored display bunny, tone mapping
  OFF, Ambient 0.12 / Diffuse 1 / Specular 0.35, fitted to the display bunny with a 1.02 dolly).
  **Cube panels = `nitsche/render_dn.py`'s (d)/(e) exactly** (unmirrored, seen from the back, cut at
  y = +0.009 keeping the far half, fitted to the cube, 1.16).  The mesh and octree are
  `diffused_nitsche`'s; verified identical (`bunny.stl` sha, `dn_stats`).
- **The gradient lives on the octree only.**  No "gradient on the surface" panel.
- **Trimmed mesh + φ**: dn (e) with the BC cells, and ONLY the ear and base BC facets drawn
  translucent and self-lit (denser and brighter than (d)'s body so they read at this size).  No
  body, so φ shows beneath.  It is seen from the back, as dn shows it.
- **Nitsche solve**: `diffused_nitsche` (f)'s look, but the **BC panel's camera and size**.  Its
  colours are not to be changed.
- **BC regions**: rendered like the TripoSG clay bunny.  The ears = the PartField ear parts via the
  production `SurfaceSegmentSelector`; the base = dn_prep's fixed bottom (lowest 5 %).  The ears'
  deformation is enlarged (traction −0.05).
- **Segmentation**: PartField only, pastel tints, no seam lines.  **Materials**: one panel, only the
  ears softer.  **Texture**: a pre-processing step between segmentation and materials; the sources
  go straight to segmentation (to the octree before 2026-10-16).  **Objective J**: a small pill, not a panel, under the solve.
- **Inputs**: only the text prompt (the image and scribble cards went with the text-first front
  end, 2026-09-29), in a LARGER, ROUNDED text field.  It is set at 6 pt and wrapped over two
  lines: on one line at that size it would widen the generation part past what the
  title-limited row can give.  SDXL's drawing is a transparent cut-out on the generation tint, at the shapes'
  height.  The Hunyuan bunny must look like a REAL rabbit (not a toy, no dark eye patches).
- **REJECTED — do not retry**: per-step category shown as a thin coloured rule under each title
  ("even worse"); deeper tints with edges ("awkward for a nature-series journal"); "optional"
  subtitle lines; the texture step as a generation lane; J as a panel; greyscale or light-green
  materials; a separate shape-diameter segmentation panel.

**Lessons:**

- **`bunny.stl` is wound INWARD (signed volume −0.198).**  The display bunny is `bunny.stl`
  mirrored in y with its faces unchanged, and the reflection turns it outward (`ov_assets.bunny()`
  asserts volume > 0).  Hunyuan-Paint on an inward mesh returned an **all-black texture without an
  error**.  That is a production bug in `src/hunyuan3d_wrapper`, reported and NOT fixed.
- **Hunyuan-Paint's frame.**  Hunyuan is y-up and paints the SDXL reference from its front camera
  (−z), so the display bunny goes in as (x, z, −y) (`TO_HUNYUAN`, det +1 asserted: a mirror would
  paint a mirrored bunny).  The reference is SDXL + depth ControlNet on the bunny's depth, prompt
  "a realistic brown wild rabbit with soft natural fur, photograph", seed 1, picked from a screen of
  4 prompts × 3 seeds.  The paint runs at `--views 9 --paint-resolution 768 --texture-size 2048`
  (the defaults smeared the fur), and the fur renders at Specular 0.08 (0.35 looked plastic).
- **A PLY in ParaView renders as a point cloud** (the speckled first input card).  Keep a `.stl`
  twin for rendering.
- **Render every panel OPAQUE onto its block's tint.**  A transparent render composited later
  keeps a dark anti-aliased rim; un-premultiplying it leaves a light halo.  `render_ov_panels.save()`
  sets the background to `S.fill_rgb(S.PANEL_PART[name])`.  `render_ov_solve.py` takes
  `DN_SOLVE_BG`; its background is not tone-mapped (checked: the corner pixel equals the tint).
- **The solve panel is `diffused_nitsche`'s own renderer, not a copy.**  `render_ov_solve.py` grew
  four env overrides, each defaulting to panel (f)'s value so `build_nitsche` is unchanged:
  `DN_SOLVE_PREFIX` (the VTU), `DN_SOLVE_FIT` (the surface the camera is fitted to; default
  `dn_full.vtk`), `DN_SOLVE_DOLLY` (1.16) and `DN_SOLVE_BG` (white).  The loop writes
  `ov_loop_solve.vtu` in the DISPLAY frame (points and displacement × MIRROR), so with
  `FIT=assets/bunny_display.stl DOLLY=1.02` the deformed bunny is exactly the other bunnies' size
  (asserted in the compose; the ears stay inside the frame).
- **Capturing the gradients (`ov_bunny_loop.py`).**
  - `ShapeOptTrimmed` runs with `remeshing=False` on dn's octree, one iteration,
    `output_per_optimize_step=0`, objective = compliance alone (`shape_opt_w_vol = 0`).
  - A `_post_tape_backward` hook copies `_last_e_grad` (outer, untrimmed) and, in phase-field mode,
    `_last_e_phi_grad` before `save()` drops them.
  - The nodes `_e_fixed_bool` freezes (every node of a boundary element touching a BC patch) are
    written as masked; the field shown is log10(|g|/max) over 3 decades.
  - **Traction −0.05 NaN'd in one load step**: 4 load steps and a `forward()` before `optimize()`
    fixed it.
  - Each gradient panel cuts through the band its own sensitivity lives in (`cut_plane`:
    dJ/ds −0.045, dJ/dφ +0.009, in `ov_loop.json`); the paper's +0.009 misses the neck, where
    dJ/ds sits.
- **Check the adjoint with finite differences that account for the discrete step**
  (`OV_GRAD_FD=1`).  The FD/adjoint ratio GREW as h shrank: the objective carries a fixed discrete
  jump.  Fit FD(h) = a + c/h and compare a: a/adjoint = 1.009 (asserted within 2 %).  Compare
  u_max on the SURFACE interpolation, not nodal against surface (they differ by the interpolation,
  not by a bug; rel 1.9e-4 once like is compared with like).
- **PartField k is chosen, not typed.**  `ov_assets.py segments` clusters k = 5…10 from ONE feature
  cache.  The loop takes the smallest k whose parts give each ear its own part: k = 8, ear parts
  [5, 7], 23 connected segments.  The BC selector is the production `SurfaceSegmentSelector` on
  those parts.  `pfbunny` now reads this same run (§5b-ter), so `partfield_segmentation` IS this
  panel.
- **Segment tints: one assignment, shared.**  `ov_palette_check.segment_tints` tries all 8!
  assignments of the eight pastel `SEGMENT_TINTS` and keeps the one whose closest TOUCHING pair is
  farthest apart (normal ΔE 16.9).  Parts that do not touch may look alike.  **CVD limit**: 4 of
  the 12 touching pairs are under CVD ΔE 6 (worst 2.8), and no assignment of these tints does
  better.  It is a property of the pastel set the user chose, and the caption must carry the part
  identities.
- **BC regions as ONE surface with cell colours.**  Split into rest / ears / base meshes, each part
  got its own normals and the seams sparkled white.  The body is `S.CLAY`, the TripoSG panel's grey.
- **Materials**: ears hard rubber (E = 1.0e8), the rest ABS (2.04e9), each the geometric mean of
  its range.  The ramp is `PHI_RGB`'s light end → its 0.667 blue, in the figure's family:
  viridis's light green read out of place, and greys read as missing data.  The first try, a
  deepseek-flash VLM per-part call, returned one material for every part; the user said to infer
  the materials ourselves.
- **The pictured SDXL drawing is the paint step's depth-conditioned reference, standing in for
  `draw_image`'s.**  Production `src/hunyuan3d_wrapper/draw_image.py` is plain text-to-image SDXL
  (object-on-white style prompt), cut out with RMBG-1.4 and an Otsu threshold.  A free drawing of
  "a sitting rabbit" would sit in another pose than the display bunny every generator is pictured
  producing, so the figure shows `assets/paint/sdxl.png` (the same SDXL, guided by the bunny's
  depth) in its RMBG-1.4 cut-out.  Say so in the caption.
- **Input cut-outs (`ov_assets.py cutout`).**  Only `sdxl_cut.png` is drawn now; the clay-render
  and scribble cut-outs (and `input`, the step that makes them) stay in the assets unused.
  - Hunyuan's own background mask for the SDXL image was BINARY and left a grey rim.  Both images
    are matted with RMBG-1.4, TripoSG's own background remover, run as its pipeline runs it (1024²,
    min-max stretch).
  - The mattes are then colour-decontaminated: I = aF + (1−a)B, with B smoothed in from the
    background pixels.  The whiskers survive and no rim remains.
  - The scribble's ink is keyed off its white paper.  The compose composites all three onto the
    generation tint.
- **(Unused since the text-first front end.)  A scribble must not be a traced outline** (that reads as a silhouette).  `_trace_scribble`
  simplifies the silhouette hard (`approximate_polygon`, 0.022 of the size), splines it, jitters it,
  overshoots the start, and adds a quicker second pass over the back and ears.  Then the marks a
  hand adds: the ear split from the notch, an eye dot, a front leg, a tail loop.
- **Layout (`ov_figure.py`).**
  - **A grid**: two rows, one column per forward step.  Row-B blocks take their partner's column,
    and every row-B panel is its partner's size (same crop group, same height; asserted).
  - **One gap** (`GAP` 0.19 in) between all columns, so every forward arrow is one length.  The
    panel height H is solved by bisection so the forward row exactly fills the width.
  - **A long title widens its column.**  "(Heterogeneous) Materials" is two lines, each centred on the text (left-aligned, the short
    "Materials" hung off to one side and the user read the title as off-centre),
    the star centred left of both.
  - **Checks**: the text registry, font sizes 5–7 pt, icon count, blocks not overlapping, rows
    aligned, titles inside their blocks, nothing off the canvas.
- **The forward row is limited by its TITLES, not its panels.**  At H ≈ 0.8 in every column but
  the octree's is as wide as its title ("Trimmed mesh + φ", "(Heterogeneous)", "Texture
  (Optional)", the Objective pill), so the width is flat in H until the octree panel outgrows its
  title.  A generation part only 0.1 in wider left the bisection no solution at all.  Width comes
  back from the gaps, the pill's padding, the fans' corner radius, and the CAD title's two lines
  (on one line, "Text2CAD / CAD-Coder" alone is 1.2 in); it does not come from a smaller H.
- **Arrows are drawn, not `FancyArrowPatch`.**  A 0.75 pt shaft with rounded corners and a slender
  concave (stealth) head.  Merges bend smoothly into the shared shaft (the three lanes into the
  octree, the materials into the BC → trimmed arrow), and only that shaft carries a head.
  **A head straight off a bend reads as a kink**: the ∂J/∂s riser did it when the lane sat 0.09 in
  below the blocks.  `arrow()` now asserts a straight run of ≥ `MIN_RUN` (0.035 in) between the
  last bend and the head; `DROP` 0.15 gives it.
- **Pastel block tints do not separate under CVD** (generation / FEA 0.8, pre / FEA 2.3).  The
  parts are also told apart by place and by the legend, and the caption names them.
- **The CAD lane is a real B-rep, reverse-engineered from the Stanford bunny into a CAD-Coder-style
  CadQuery script (2026-09-30).**  The user first supplied a Gemini-generated picture of a B-rep
  bunny (`assets/cad/brep_bunny_gemini.jpeg`, cut out by `ov_assets.py cadimage` along its closed
  outline; kept, unused), then asked for the model to be BUILT, "almost like the Stanford bunny but
  CAD-style, with many more B-rep patches".  `ov_assets.py cadmodel` runs `ov_cad_fit.py`, which
  writes `assets/cad/bunny_cadcoder.py`, and `ov_cad_build.py`, which builds it into
  `cad_bunny.npz`.
  - **What works: a loft through latitude rings of the body's radius function.**  The ear-less body
    (PartField ear parts 5 and 7 removed from its voxels) is star-shaped from its centroid to 99 %:
    1 % of rays cross it twice, and the farthest crossing is taken.  A (θ, φ) ray grid of 48 rings
    × 64 points, with the axis along y (flank to flank), gives r(θ, φ), blurred 0.5 grid steps.
    - Each ring is a periodic spline through the SAME longitudes, so neighbours correspond exactly.
    - The poles sit on the flanks, and the loft's end rings are flattened (it caps them flat).
    - Head, chest, paws, haunch and tail lie between the poles, and the head is its own lobe.
  - **Ears**: a loft through the moment ellipses of 9 slices along each ear part's principal axis,
    the root pushed 0.05 into the head, plus a shrunken closing ellipse at the tip.
  - **Order, and the patch grid (208 faces)**:
    - The ears are fused onto the smooth, UNSPLIT body first.  A fuse onto patch edges failed
      wherever an edge met an ear root.
    - Then one `Shape.split` of the rabbit's SHELL cuts it along FULL planes, and the solid is
      rebuilt from the shell and never `clean()`ed.  The planes: 12 rings in x, 5 waterlines, a
      centre line in y, and small squares for 3 rings per ear.
    - Every sheet must cross the whole surface.  Sheets that ended inside it (rings stopped under
      the ear roots, flank and face lines stopped short) left sliver faces the mesher could not
      close.
    - Measured: voxel IoU 0.94 with the bunny, ICP mean distance 0.50 % of its diagonal.
  - **The poles (the user: "a small chunk of curves clustered near the leg")**: the near pole sits
    on the flank by the front leg, in view.
    - With a cap angle θ0 of 0.06–0.22 rad it showed as a flat disc with a dent round it and
      patch lines crowding in.
    - Swapping the flat caps for `makeNSidedSurface` domes through the measured pole points left a
      creased knob.
    - Tilting the axis so both poles face away from the camera broke the build in every variant.
    - What works: θ0 = 0.025, only the end ring flattened (a cap of about 1 px in the figure), and
      waterlines evenly spaced and kept 0.07 clear of each pole's height.  Two waterlines either
      side of the pole read as a doubled line, and flank lines at y = ±0.17 ran beside a waterline.
  - **What failed first, in order** (so nobody retries it):
    - Moment ellipsoids per k-means cluster (IoU 0.83) read as a balloon animal, and OCC could not
      fillet their junctions.
    - Hand-placed ellipsoids made an egg on legs.
    - A loft along x through convex-hull sections (IoU 0.88) turned head, chest and paws into one
      block, and hull ear lofts were invalid solids.
    - A loft along x through the REAL outlines still could not make the head a lobe.
    - Splitting parts or runs by planes and fusing them back, even with glue, left coincident faces
      that broke the Boolean (slivers, invalid pieces, cracks).
  - **Built in process, not through `run_sandboxed`.**  Its child is capped at 120 s of CPU, and
    OCC runs every Boolean on all 64 cores here: about 30 s of wall time costs 150–400 CPU-seconds,
    and the child is killed.  `ov_cad_build.py` therefore calls the sandbox module's own
    `check_source` (the script passes CAD-Coder's static safety check), `execute`, `tessellate`
    (default tolerance) and `finish_mesh`.
  - `ov_surfaces.py` aligns it (ICP) and draws it in the TripoSG panel's clay with its B-rep face
    edges in ink.  `render_ov_panels.py` renders `ov_p_cad.png` as a surface panel, so it stands
    as tall as the other bunnies.
  - The caption should say the CAD part was reverse-engineered from the bunny rather than
    generated by Text2CAD / CAD-Coder from the prompt.
- **Caption must carry**: the colour meanings (s, φ, |∂J/∂s|, E, Neumann/Dirichlet), that the
  pictured solve uses uniform material, where the BC sets come from, that the three part tints and
  the pastel segment tints are not CVD-separable, and the model and CAD licences.

| Knob | Value | Where |
|---|---|---|
| page | 7.087 × 2.873 in; panels H = 0.809 in, both rows | `ov_figure.py` |
| gaps | columns 0.18, prompt/SDXL → generators 0.23 (fan corners 0.035), sources → octree 0.24, rows 0.17, backward lane 0.15 below row B | `GAP`, `G1`, `RF`, `GAP_SRC`, `VGAP`, `DROP` |
| generation | CAD block's content centred on the forward row's centre line; the three shapes one size, 2.5 lane_c = 1.5 H + VGAP + HEAD_B − 2 HEAD − 2 BP − 2 LANE_GAP | `layout()` |
| type | titles 6.5 pt, legends / tabs 6.0 (bold tabs), the prompt 6.0 italic over two lines in a rounded 0.25 in field | `TITLE_FS`, `CARD_FS`, `TXT_H` |
| part tints | `#fcf0e6` / `#eaf1fa` / `#ebf5e6`, no edge | `ov_style.PARTS` |
| identity colours | ink `#1a1a21`, agent `#6b4fbb`, backward `#a3195b`, GPU `#3b7d23` (NVIDIA's `#76b900` fails CVD 5.9), Neumann `#dd8452`, Dirichlet `#12a594` | `ov_style.PALETTE`, gated all-pairs |
| arrows | 0.75 pt, head 0.062 × 0.048 in, notch 0.28, corner radius 0.045, forward `#30343b` | `ARROW_LW`, `HL`, `HW`, `R` |
| load | traction −0.05 on the ears, 4 load steps, base clamped; u_max 0.280 on the surface | `ov_bunny_loop.py`, `ov_loop.json` |
| PartField | k = 8 (auto), ear parts [5, 7] | `ov_loop.json` |
| paint | SDXL-base + depth ControlNet, seed 1, 9 views / 768 px / 2048² | `ov_assets.PAINT_*` |

**Retired with the first version** (no longer in `build.sh`; files may linger until removed):
`ov_thumbs.py` / `ov_thumbs.json` (panels extracted from built PDFs by `pdfimages`, index plus a
pixel sha as the guard), `ov_bench.py` / `ov_bench_log.txt` / `ov_stats.txt` (measured timings, now
banned by the titles-only rule), and `ov_t_*`, `ov_sheet_*`, `ov_input_*`.  `ov_bunny_solve.py`
stays: it is `build_nitsche`'s solve.  Two lessons from that version still hold anywhere:
- A figure built from other figures has a moving foundation; pin each extracted panel by index AND
  pixel hash.
- Its palette gate found `hidden_phi_loop` / `shape_opt_loop` shipping an orange/green loop-leg pair
  at CVD 4.9.  That is still unfixed there: a one-constant edit plus a rebuild.

---

### §5m The 50-object gallery (`gallery`, learned building `agentic_gallery`)

`./build.sh gallery`: `gl_prep.py` → `am_layout.py` → `render_gl.py` (pvbatch) → `am_figure.py`.
All 50 Opus objects stand in ONE amphitheater, seen by a person standing on the stage.
The layout rules are the user-approved schematic's (job notes:
`schematic/amphi_schematic_v5.{py,txt,png}`), ported, not re-tuned.

- **Same data as agentic results' last column, imported rather than copied.**
  `gl_prep` calls `ar_prep._prep_solve` as it is (surface interpolation and the ×N rule), then
  asserts ×N and |u|max against each object's category record. `ar_prep.py` is not
  refactored. Each object's shown side is its category figure's camera heading, turned to
  face the eye.
- **Revision 2 (designer eye-check of the first render + the user's binding requirements,
  2026-10-17):**
  - No shadows; plain white floor.
  - **Labels are obstacles for every object.** The shared rule `am_labels.place` takes, nearest
    object first, the first candidate (below the silhouette with nudges, then beside the base,
    then above the top for the back row) that lies on no object's ink and is ≥ 0.5 mm from other
    labels. It must also be owned: the centroid nearer its own object by ≥ 1.0 mm, and the
    nearest glyph ≤ 3 mm from it. The same function runs in the layout search (Warp
    silhouettes, 0.25 mm) and in the final compose (id passes, 0.085 mm), where a failure stops
    the build and names the label.
  - **Labels are vector again,** since nothing may cover them.
  - **Rows leave floor for labels.**
    - The radial gap between rows is at least the next row's label strip: two label lines
      plus clearances, foreshortened by the sight line.
    - Tiers also rise until the sight line clears each row's label strip (fix 1).
    - Label-bound seats use gap 0.55 (fix 4); odd rows are staggered (fix 4).
  - **Arcs** are centred ARC_BACK × R0 behind the stage (fix 3).
  - **Repair.** A label with no legal position widens its own seat ×1.15, for up to 3 rounds.
  - **Search.** ARC_BACK ∈ {0.5, 1, 1.5}, eye height ∈ {60, 70, 80}, span ∈ {110, 130, 150}° and
    stagger ∈ {0, 0.5} are searched, 54 candidates in 6 processes. The winner is the passing
    layout covering most 2.5 mm page cells, with the smaller largest empty rectangle as
    tie-break.
  - **Outline** 0.15 mm `#b3b9c1` (fix 6).
  - **`GL_SEAT_SWAPS="a skyscraper:a lighthouse"`** is the user's choice of 2026-10-17 and
    the one named exception to rule-only placement. The two exchange seats after row
    assignment, before tiers and labels, and the size-order gate exempts them by name.
- **Revision 3 (the user, after revision 2 read as sparse):**
  - Labels are 4.0 pt (cap 1.01 mm).
  - **Two label lanes.** Odd seats' labels sit one label height + 0.5 mm further toward the
    viewer, so a label competes for row length only with the next-but-one seat.
    - A seat needs max(object × 1.35, half its label + 0.8 mm).
    - Neighbours keep their bodies apart; seats two apart keep their labels apart.
    - The row gap holds both lanes. ROW_GAP is 0.3.
  - **Labels fall back** to the other lane, then beside the base (left or right, turned by
    the side rule), then above the top for the back row.
  - **Rows spaced by footprint.** Rows are spaced by each object's drawn footprint (its
    largest horizontal side), not its largest extent: a tall tower needs little floor.
  - **Short last row.** It takes the largest objects of the row before (size order kept)
    while it has fewer objects and still fits, so no lone back row leaves empty sky.
  - **Search margin.** The search adds 0.25 mm to every label clearance, so its passes survive
    the compose's finer raster. Without it, the first rev-3 build passed the search and then
    failed 2 labels in the compose.
  - **Search grid:** 36 points (arc centre {0.5, 1} R0 × eye {60, 70, 80} ×
    span {100, 110, 120}° × stagger {0, 0.5}), 8 processes.
  - A failed label's seat widens ×1.15, for up to 4 rounds.
- **Revision 4 (the user, after revision 3):**
  - **Lower eye:** eye height searched over {30, 38, 45} D0, down from 60–80.
  - **Flatter rows:** the arc centre sits {2, 3.5, 5} R0 behind the stage.
  - **Closer sizes:** p goes from 0.25 to 0.20.
  - **Fuller sides:** the search objective is the coverage of the least-covered page third
    (left / centre / right), then total coverage.
  - **Rows span the frame.** Each row may fill 95 % of the camera's frame width at its own
    depth, so near and far rows reach the same page edges (no fan; previously span × r).
    The label turn follows the seat's place along its row.
  - **Grid:** 27 points: arc centre {2, 3.5, 5} R0 × eye {30, 38, 45} × row gap
    {0.3, 1.5, 3.0}. A low eye foreshortens the rows, so a larger row gap lets the scene
    reach the page height instead of being width-bound.
  - **Relaxed label rule** (the user, superseding revision 2's hard rule):
    - A nearer object may cover part of a farther object's label or body.
    - Hard rules kept: no label touches another label; every label is owned (≥ 1.0 mm
      centroid margin, ≤ 3 mm from its own object); every label stays ≥ 70 % visible.
    - Among passing positions, an uncovered one is preferred, else the least covered.
    - Labels are composited as ink in the 1200 dpi render under nearer bodies, so the PDF
      labels are raster at 600 dpi.
  - **Per-object overrides.** The user allows named per-object arrangement (2026-10-17). None
    are used so far; any future override must be listed here with a one-line reason.
- **Revision 5 (the user, 2026-10-19: "not compact enough"):**
  - **Taller page.** The page stays 180 mm wide (full text width, like `method_overview`);
    its height is searched over {110, 122.5, 135} mm.
  - **Labels at 5 pt,** the journal minimum (cap 1.26 mm). The gate band is now
    1.2–1.35 mm.
  - **Lower eye:** {24, 32} D0.
  - **Equal-count rows.** The rows are `rows_n` ∈ {5, 6, 7, 8} contiguous size groups of
    equal count, replacing frame-width capacity. Labels set most seats, so equal counts give
    rows of about equal printed length, and the row count sets the block's height.
  - **Rows reach the same edges.** Every row is spread evenly to the printed width of the
    widest row, so all rows reach the same page edges. Spreading to the camera frame instead
    fed back into the FOV fit and ran away (FOV 84°).
  - **Search:** arc centre 5 R0, row gap 0.6; 24 points.
  - **Overlap:** objects may overlap; nearer objects may cover farther labels down to
    70 % visible. Labels never overlap each other, and ownership stays ≥ 1.0 mm and ≤ 3 mm.
- **Revision 9: the user kept revision 8's layout** (2026-10-19), with named per-object
  arrangement in `gallery/placement_overrides.json`. Each entry carries its reason
  ("user request 2026-10-19"), is copied into `am_scene.json` / `gl_stats.json` under
  "overrides", and is selected by `GL_OVERRIDES` (default that file; "" runs the search
  again). With overrides set, there is no search: revision 8's parameters and seat
  widenings are fixed in the file.
  - Row 1 (front) spacing ×1.12; row 4 (back) spacing ×1.09.
  - "a USB stick" +0.5 D0 radially (away from the viewer); "an electric toothbrush" −0.5 D0
    (toward the viewer).
  - "a skyscraper", "the Leaning Tower of Pisa", "a lighthouse", "a natural stone arch":
    −2.0 D0 radially (forward) and floor −1.5 D0 (lower).
  - Labels are re-placed by the unchanged rules. No gate needed a smaller nudge.
- **Revision 8 (the user: "the top row is too far away"):**
  - **Lower tiers** (`close = 1`). A row's tier now rises only until the sight line clears
    ONE label line of floor in front of it, not three. Labels whose floor is covered move to
    another lane, above or beside.
  - **No closer.** Dropping the strip entirely (`close = 2`: rows spaced by footprint, tiers
    by the body rule alone) left 14+ labels unplaceable and a 27 % empty band.
  - **Shorter page.** The page height is searched over {90, 100, 110} mm.
  - **Closer row steps.** The radial step between rows shrinks by `row_overlap` ∈ {0.3, 0.6}
    of the two rows' half-footprints, so rows overlap the row in front, as asked.
- **Revision 7 (the user: "more objects in the front rows … remove one or two rows"):**
  - **Rows filled to a common printed width.** With `balance_rows`, rows are filled front to
    back, smallest first, each until it prints W wide. W is the smallest width that seats
    everyone in `rows_n` rows, found by bisection. The front rows then hold more, smaller
    objects, and the crowd is a rectangle, not a funnel.
    - Pairwise boundary swapping oscillated, which is why bisection is used.
  - **Search:** rows_n {4, 5} × pack {0, 0.3} × eye {24, 32} × page height {110, 122.5, 135}.
  - **Label preference** (to stop ambiguous labels): among the passing positions, take a
    label that nothing covers, that lies on no other object (≤ 2 %), and whose own object is
    nearer than any other by 1.5 mm. Failing that, the one with the fewest such faults.
    1.0 mm stays the hard floor. Labels short of 1.5 mm or over another object are listed
    as "ambiguous_labels" in `gl_stats.json`.
- **Revision 6 (the user: "the packed audience view, less spacing, accept overlapping"):**
  - **Packed rows.** Rows are `rows_n` equal-count size groups, each packed from its centre:
    seat = the object's footprint × (1 + pack). They are no longer spread to the widest row.
    Labels no longer widen seats.
  - **Labels.** They take room by lane (three lanes by seat parity, a fourth as fallback),
    by sliding up to ±8 mm along the floor, or by sitting beside the object. Only one label
    line of floor is kept between rows.
  - **Label turn** at the row ends reduced from 25° to 15°.
  - **"Above" fallback for every row.** A label may go above its object's top in any row,
    where it lies over farther rows, which nearer things may cover. Without it, every packed
    layout left 2–13 labels unplaceable.
  - **Packed repair.** A failing seat gains 2 mm of printed row length per round, up to
    5 rounds. Multiplying a toothbrush's footprint by 1.15 gained nothing.
  - **Search:** rows_n {3, 4, 5, 6} × eye {24, 32, 40} × page height {110, 122.5} ×
    pack {0, 0.3}; 48 points.
  - **Speed.** The label rule now caches glyph outlines and checks candidates in a local
    window (one candidate 100 s instead of > 10 min).
- **Size.** D = (L/Lmin)^p in world units (D0 = 1), defined before projection; p = 0.25 until revision 3, 0.20 from revision 4.
- **Size.** D = (L/Lmin)^p in world units (D0 = 1), defined before projection; p = 0.25 until revision 3, 0.20 from revision 4.
- **Seats.**
  - Arcs over 130°, with a front radius R0 = 18.
  - Rows are filled smallest first. A seat is max(D, label world width × cos 25°) × 1.35.
  - The label world width is what the fixed 4.5 pt print needs at that depth. That depends
    on the fitted focal length, so layout and camera fit are iterated until the rows are
    stable (3 iterations).
  - Within a row, the largest object is at the centre and the rest alternate outward.
- **Tiers.** A row's floor rises only where the sight line over the row in front would hide
  more than 80 % of its tallest object. With the eye 50 D0 high, no tier rises (all 0).
- **Camera.**
  - True perspective. The eye is 2 R0 behind the stage centre at height 50.
  - The aim is re-pointed until the scene's projected box (objects plus label rectangles) is
    centred.
  - The horizontal FOV is fitted to fill 97 % of the 16:9 page; it came out at 51.1°.
  - `render_gl` sets position, focal point, view-up and the vertical view angle verbatim.
- **Labels.**
  - They are raster ink composited in the 1200-dpi render, not a vector layer, because a
    nearer body may cover part of a farther label.
  - Anchor: 0.6 mm below the object's unoccluded silhouette (Warp-cast from the eye), under
    its middle half.
  - Turn: sign(a) × 25° × |a| / 65°.
  - Collision rule, nearest object first: the first offset in a size-ordered list (down to
    4 mm toward the viewer, 1 mm up, sideways up to 6 mm) that keeps the label ≥ 0.5 mm from
    every placed label, off its own body and inside the page. Among those, prefer offsets
    with no nearer body over the label, then labels nearest their own object (+0.5 mm).
- **Keys removed** at the user's request: no key, colour bar or size line in the figure; the
  caption carries them.
- **Gates (`am_figure.py`):**
  - projection: id pixels inside the Warp silhouette ≥ 99 %, bounding boxes within 1
    quarter-resolution pixel on unoccluded objects;
  - object visible fraction ≥ 0.25 (heavy overlap allowed);
  - no two labels overlap, every label ≥ 60 % visible, labels inside the page;
  - cap height 1.0–1.2 mm (a named waiver of the 5 pt floor);
  - size order, and ×N equal to the category records;
  - the batch root untouched.
- **History.** The earlier orthographic 45° row gallery (commit e97a5ec, `gl_layout.py` /
  `gl_figure.py`) was text-bound. Its row labels in front of the objects needed p ≤ 0.12 to fit
  on 16:9. The amphitheater fits p = 0.25 because perspective shrinks the back-row labels'
  world footprint relative to the objects, and overlap is allowed.

### §5n The demo video (`demovideo`, 2026-10-25, revised the same day, 2026-10-26 and 2026-11-07)

`./build.sh demovideo`: `dv_cleanup.py` (cached) → `dv_prep.py` → `render_dv.py` (pvbatch,
two processes: bench and tiles) → `dv_compose.py` (PIL) → ffmpeg. 28.8 s, 1920×1080, 30 fps, H.264 CRF 20
yuv420p faststart, ~9.2 MB, ~15–19 min on one A6000 (+ ~11 min once for the cached clean-up re-run). The mp4 and its poster png are tracked
(the user reversed an earlier "not tracked"). Times live in ONE module, `dv_timeline.py`,
which both the renderer and the compose import, so a pass is rendered exactly on the
frames the compose blends it; `dv_camera.py` is ParaView's projection in numpy, shared by
the camera fit and the compose.

- **Timeline (s).** CLI 0–2.2 (opaque from frame 0, typing 0.15–1.75, faded 2.2–2.4 while
  it moves up to 2.5, drawn UNDER the rising bench, opaque by 2.35); rise 2.2–3.1; looks 4.2 / 5.7 / 7.2 / 8.7 (0.6 s fades); E→|u|
  10.3–10.8; octree 10.8–11.1 in, held to 13.6, out 13.9; ghost and "×1" 13.9–14.1;
  s(t) from t0 = 14.0, T = 2 s, peak 15.0; the optimisation beat 16.0–22.8 (below); the
  optimised bench into its cell 22.8–23.35; tiles 23.4–24.05; hold to 28.8 (3 s longer at
  the user's request). Every scene-8 time is the 23 s cut's + `OPT_DT` = `SHRINK0` − 17.0
  = 5.8 s (derived, not typed); the poster is frame 747 (24.9 s). Title changes dip (out 0.18 s, in 0.22 s); a
  two-line-over-one-line dissolve read as mush on a phone.
- **Looks are agentic results', by import.** `dv_prep` calls `ar_prep.prep_run`
  (`AR_SEGMENTATION=1`) and `write_run_files`; `render_dv` builds each look with
  `render_ar`'s `surface_panel` / `bc_panel` / `solve_panel`. Tiles go through
  `ar_prep._prep_solve`, gated (×N equal, |u|max to 1e-6) against the main or Opus
  category record, as the gallery does. Video-only changes, all in `dv_prep`:
  - E on log [5, 200] GPa. The figure's shared scale puts the bench's 12 and 120 GPa in
    two similar blues.
  - Support tubes are 2.5× the figure's radius.
  - The figure's 3 print-sized arrows are replaced by 5 seat + 3 backrest arrows. They
    have `arrow_mesh`'s shape, are 150 px long, and sit in a pass of their own that the
    compose lays OVER the body, so the backrest never hides them while the bench turns.
- **Camera.** Every object orbits a vertical axis through its bbox centre at ONE distance
  (no zoom pump).
  - The bench turns at 11.4°/s (12°/s, then 5 % slower, the user) from heading +25.55°.
  - Chosen so the BC scene's middle (7.95 s) is a −40° front three-quarter, where the
    backrest's backward load arrows read along their length (the user: "clearly
    visible").
  - The octree hold's middle (12.35 s) is then at −90°, face-on to the lateral cut.
  - Headings: the rise ends at +15°; the hold sees −76° to −104°; the ×200 peaks sit at
    −120° and −143°; the clip ends at −212°.
  - The +40° side would put the hold near 0°, the cut seen edge-on. The cost of −40° is a
    rear three-quarter view from 12.34 s to the end of scene 7.
  - Its fit is the at-rest height over the headings the clip shows = 92 % of the 780 px
    viewport. The viewport is y = 220–1000, 70 px below the frame centre (the user: "a
    little bit lower"); `fit_camera` centres the bench there, not at y = 540.
  - Tiles turn at 17.1°/s ±10 %, golden-ratio phases. A tile's largest projected side over
    all azimuths, undeformed and at peak k, = 88 % of the short side of its cell's body
    area.
- **Ghost.** A 4 px `#9aa1ab` line just outside the undeformed silhouette, drawn only
  where it is more than 4 px from the deformed silhouette, with no fill. The first cut's
  fill + 6 px ring read as a sticker halo. Tiles use the same rule.
- **The octree beat (the user: "the full octree mesh, with the front half cropped, the
  shape is still there", held ~2.5 s).**
  - The solve writes `vtu_sdf/` beside `vtu/`. That is the adaptive octree the solver
    holds (136 801 leaves, 11–182 mm). `vtu/` holds only the 107 k trimmed cells; the
    first cut showed those, and no grading showed.
  - But `vtu_sdf/` is not a full box (the user: "some cells on the outer boundary are
    missing").
    - `workflow_tool.run_forward` builds the octree in an isotropic cube: 2.909 m,
      140 470 leaves, for the bench.
    - It then sheds the padding with `core.remeshing.clip_octree_to_bbox`, which keeps a
      cell iff its CENTROID lies in the geometry bbox + pad.
    - Coarse cells straddling that box drop out by their centres. The saved leaves fill
      only 92.0 % of their own 184 × 128 × 128-leaf bbox: notches and steps.
  - `dv_prep.full_octree` rebuilds the cube with the repo's own code, exactly as
    `ShapeOptTrimmed._init_remeshing_state` path A does:
    `build_initial_octree_from_trimesh` (same mesh, mesh_center, h, depths 4 → 8,
    clip_bbox for the lattice snap), then `sdf2octree`, then
    `scale_octree_nodes_to_physical`.
    - Gate: `clip_octree_to_bbox` of the rebuilt cube must equal vtu_sdf's cells, cell
      for cell.
    - It keeps every leaf inside the clip box grown out to the 182 mm lattice. That is
      137 334 whole cells filling the box exactly, x −0.33–1.85, y −0.38–1.44,
      z −0.29–1.17 m (`ar_0_octree.vtk`).
    - The whole 2.9 m cube would not fit the frame.
  - The octree is cut EXACTLY by a plane fixed to the bench, through the orbit centre:
    the bench's lateral mid-plane, normal −x. That is the front/back mid-plane's +z
    turned 90° clockwise seen from above (the user: "rotated 90 degs clockwise").
    - The +x half is kept, around the whole |u| surface. It is the half away from the
      camera through the hold, and the plane turns with the bench.
    - Cells straddling the plane are cut, not kept whole. Round 2 kept them whole with a
      crinkle clip.
    - Fill and lines are separate, so no triangulation shows:
      - Fill: an exact `Clip` of the hex grid, drawn as a surface with no edges,
        near-white and flat, so the clip's tetrahedra never show.
      - Edges: `ExtractEdges` of the uncut grid, clipped by the plane. Every kept cube
        edge is an exact segment, 1 px.
      - Cut face: every cell's six faces as quads (`dv_prep` writes
        `ar_0_octfaces.vtk`, 163 k points, shared faces once), sliced by the plane.
        Each cut cell's own outline is drawn 2 px, darker.
  - Axis-aligned, the plane cuts every cube in a square, so the cut face is a true
    quadtree pattern in perspective.
  - Plane history:
    - A fully camera-facing (tilted) plane cut cubes into triangles and hexagons.
    - A vertical, camera-following plane is oblique to x and z, so its x- and z-face
      crossings interleave. Cubes cut near a vertical edge became narrow slivers that
      read as lines through big cells. They were true outlines: on frame 370, 0 of
      3 782 slice segments lay inside a cut cell.
    - The user then asked for the plane fixed to the bench: first the front/back
      mid-plane (normal +z), then that plane turned 90° clockwise.
  - The complete box's 12-edge frame is drawn over the octree look (`dv_compose.
    box_frame`, 2 px `#5f6672`, fading with it), so the removed front half's extent
    shows.
    - Only edges with a box face turned to the camera are drawn; nothing inside a convex
      box can hide them, so they correctly pass in front of the bench.
    - The three hidden edges were tried dashed and lighter. They crossed the cut face
      and read as cell outlines, so they are left out.
  - A ParaView `Slice` drawn as wireframe shows the cutter's triangulation. That is why
    the cut face's lines come from the face quads, not from a slice of the cells.
  - The domain box reaches into the title band, so a white 92 % band sits behind the
    title while the octree is up.
  - Scene 7's title is two lines, "Diffuse Nitsche FEA in NVIDIA Warp" / "on
    non-conforming octree mesh" (72 px Arial Bold, baselines 100 / 186; 1228 and 1104 px
    wide, so ≥ 346 px side margins). The white band (92 % to y = 205) covers both lines'
    descenders (y ≤ 201). `dv_compose` reads the lines from `dv_timeline.TITLES`.
- **Floor grid.** Rendered at 2× with 4 px lines and Lanczos-downsampled, then faded to
  0 above y = 330 (the far field, whose thin lines crawled). The E colour bar sits on a
  white 90 % panel so grid lines do not cross its 44 px ticks. The bar stands upright
  right of the bench (x = 1560, y = 380–920). Framed 70 px lower, the bench's feet reach
  y = 980, where the horizontal bar was, and that bar could not move lower and keep
  40 px from the frame's bottom edge.
- **Finale.** A 7×3 grid inset to x = 64–1856 below a two-line, 64 px title. The second
  line, "each in 6.5 minutes on average", is the mean `metrics.json` wall clock of the 50
  Opus runs (6.46 min); `dv_prep` asserts it. The bench shrinks into the top-left cell
  by the tiles' rule, and its Nitsche title shrinks toward it and gives way to its
  prompt.
  - Every cell carries its object's prompt above it as the agentic-results card prints
    it: the objects_50 wording with its article, curly quotes, typographic apostrophes,
    Arial Italic in ink, wrapped by `ar_figure.balanced_wrap` into ≤ 2 lines across the
    cell. The size is 28 px for every tile, in a 76 px band.
  - "a school chair with a steel frame and a plastic seat" needed 20 px to fit two
    lines, so that tile became "a bar stool" (bar-stool-20261013-213800, 4.0 min, ×500).
    Every prompt now fits at 28 px.
  - The control arm, glass-top table and rocking horse had already been swapped for
    shorter names when the labels were bare names.
  - Tiles pop as a diagonal wave from the bench's cell, 50 ms a step.
- **The optimisation beat (2026-11-07, the user's third design; a dolly-in close-up and a
  pop-up card came first and were dropped).** At 16.0 (s back to 1 after one cycle) the
  deformation stops and the bench is optimised in the main view; the spin never changes.
  - One eased move 16.0–16.7 into a CLOSE-UP of one end (the user: "closer view
    focused on ONE side"): the end whose change sites sum largest (−x, 7.43 h vs 7.14 h;
    the continuous heading also faces it), 0.85 m of height in the 780 px viewport
    (2.05× the orbit), its sites' centroid at (470, 620). The camera orbits that point,
    its focal point slid along the camera's own right/up axes (`dv_prep.close_params`,
    `render_dv.bench_camera`). The rest of the bench runs off the frame's left/top/bottom
    (intended); the body and its arrows fade to 0 across x 840–960 while the plot is up
    (`body_clear_mask`; no vertex reaches it at these headings). Earlier cuts: 1.1× on
    the whole bench, then fitted with 64 px margins after it touched the left edge.
  - The spin eases from 11.4 to 3.5°/s over the move in, holds through the iterates and
    the hold, and eases back during the shrink; the heading is the rate's closed-form
    integral (`_slowdown`), so it never jumps (−132° at 16.0, −137° at 16.7, −152° at the
    end of the shrink). |u| crossfades 16.0–16.6 into the iterates' surface-change
    colouring: each iterate's signed change from it_0 in h (render_prep.py's `change_h`:
    distance to it_0, + outside = added) under a NAVY-CYAN diverging map (the user,
    2026-11-08: the clean-up figure's red end read like the |u| colour; PRGn purple–green
    (−3 h #40004b … +3 h #00441b) came next and was rejected). Its mechanics are
    `experiments/geometry_cleanup/render.py`'s `lut()`, read from that file's source by
    `dv_change.py` (CHANGE_MAX = 3 h, white 0.97 within ±0.25 h, the ±1 h stops, RGB
    interpolation); the ends are the video's own: −3 h #0c7c99 deep teal, −1 h #41b6c4
    cyan, +1 h #253494 navy, +3 h #081d58, plus ONE extra knot on the cyan side, −0.5 h
    #9cd7de (half-way to the −1 h cyan): cyan is light, and on the plain ramp it sat only
    ΔE76 16.8 off white at the 0.5 h arrow threshold, below the old red end's 21.6; with
    the knot it is 24.6 (navy +0.5 h: 28.5). The two ends are far apart (ΔE76 80 at
    ±1 h, 51 at ±3 h). render.py's blue–red lut stays for the paper's shape_opt_merged.
    it_0 is all white 0.97 (ΔE76 8.3 from the zero-|u| cream: close, and crossfaded);
    navy grows where material is added.
    The change is averaged over each vertex's neighbours, 6 passes (~1 h): raw, the
    slats' sharp edges showed thin orange lines that flickered between iterates (each
    iterate's marching-cubes edge sits a little off it_0's). The iterate view's normals
    are not split at creases (split, faint grey streaks ran along the slats). A bar
    sits bottom left over the close-up (≥ 56 px from the frame's edges, on a blurred white
    backing, no panel edge), fading with the change colouring. (A first cut held a
    uniform cream body; the bar first sat under the plot.) Ghost and ×N out 16.0–16.2. The title
    dips to "Optionally, it can be optimized / for load-carrying ability" (72 px, two
    lines: one is 1860 px); that title is the one that shrinks into the bench's cell.
  - Data: the bench's geometry clean-up (`experiments/geometry_cleanup`, archived). Its run
    kept the solver output only at iterations 0 and 40, so `dv_cleanup.py` RE-RUNS it
    unchanged (its own `setup.py` + `run.py`, uniform stand-in E 57.2 GPa, step 0.5 h,
    cap 40, reinit off) with `GEOMETRY_CLEANUP_SAVE_EVERY=1` into
    `output/geometry_cleanup_demovideo/` (cached; `DV_CLEANUP_FORCE=1` redoes it). Gate
    against the archived `traj.tsv`: p99 distance within 0.012 h, max within 0.22 h at
    every iteration; SE +1.6 % at iteration 0 drifting to −2.25 % at 40 (SE/SE0 0.575 vs
    0.598). The SE offset is commit 964ca7c (2026-10-15, the day after the archived run):
    saved-segment BC selectors use the solver's BC tolerance (0.75 h, was h), so the
    frozen BC band shrank (38 933 → 33 991 cells). Same shapes, current code's numbers.
  - The 41 iterates play 16.8–21.8 at 8 a second (the user: 15 a second was too
    quick, then 10 a second one second short; `ITER_RATE`, and every later time follows), neighbours crossfaded (their marching-cubes meshes do not correspond), in the
    slowly turning close-up; then a 1.0 s pause on it_40 (21.8–22.8, frames 654–683;
    the shrink starts at frame 684), the bench still turning at 3.5°/s, the arrows full
    (bar the look-ahead occlusion fades) and the plot up: the moment the result is read
    (a 1.2 s "−42 %" hold came first, then no pause, then 0.3 s).
  - The plot (right half, x 1000–1880, no panel; the designer's review, 2026-11-07):
    Iteration 0–40 vs "Strain energy (vs. initial)" (SE/SE0), y from 0 (an honest
    baseline) with ticks 0 / 0.5 / 1.0; a dashed 2 px #5f6672 reference at 1.0; 4 px
    #3d424b axes and 4 × 16 px outward ticks (the user: thinner ones vanished at 360 px),
    40 px tick and 44 px axis labels, Arial; a 6 px ink curve drawn in step with the
    iterate, a 22 px brick dot with a 3 px white ring (on (0, 1) as the plot fades in);
    a LIVE reduction (the user): a 3 px bracket from the dashed 1.0 down to the current
    dot, at its x, and "−NN %" (64 px bold, #1f1f24) over "Lower → Stiffer" (42 px
    #3d424b, the axis colour, medium = Arial regular + a 0.5 px stroke since Arial has no
    medium face; 32 px #5f6672 was too faint (the user); Arial has U+2192) counting from −0 % to −42 % with the iterate (the curve's
    interpolated value, integer %; "0 %" without a minus), at a FIXED spot in the plot's
    lower-right interior (right edge ~1810 px, baseline ~690 px, below the curve's end,
    its dot and bracket at every iteration), right-aligned (Arial's digits are tabular:
    no jitter), visible whenever the plot is. y label "Strain energy (vs. initial)". Plot in 16.3–16.8, out 22.75–23.05. (The
    designer's version popped "−42 %" in at the end and held it 1.2 s.) No panel: a white one cut the floor grid in a hard seam, so the grid
    itself fades to 0 under the plot over a 180 px feather. Every size is a constant at
    the top of `dv_compose`'s plot section.
  - Change arrows: `ar_prep.arrow_mesh`'s shape, their own pass over the body (like the
    load arrows), coloured with the change map's own family: navy #253494 (the +1 h
    stop; contrast 9.8 on white) for added material, deep teal #0c7c99 (the −3 h end;
    4.5 on white, 2.0 on the 1 h cyan) for removed, each ringed by a 4 px WHITE halo (the
    user, "not very visible", twice; at 2 px the navy arrows still sank into the navy
    junction: PRGn mid tones #5aae61 / #9970ab with a dark #1f1f24
    outline, then dark #238b45 / #7b3294, came before; a dark fill alone merges into the
    same-coloured surface, contrast 1.0–1.5, and the white halo separates it from both
    the coloured surface and the white body). The halo's occlusion is its own: a halo
    pixel lies OFF the arrow, so it is hidden only where the body there is nearer than the
    arrow's depth at its nearest arrow pixel (it took the arrow pixels' depth mask, 0 off
    the arrow, so only a ~1 px fringe of the "2 px" halo ever showed). Shaft 7 px, head
    20 px wide (4.7 / 12 px, then 7 / 17 px; `dv_prep`'s
    own `oarrow_mesh` sizes them separately, `ar_prep.arrow_mesh` ties the shaft to the
    head). Red #e34948 / blue #2a78d6 came first. Sites (the user, 2026-11-08: emphasis by DENSITY, not size; every
    visibly coloured patch gets arrows): the threshold is tied to the colour map, 0.5 h
    (5.7 mm), where it has clearly left white (ΔE76 28.5 navy / 24.6 cyan from the
    white, the latter by the −0.5 h knot; the old red end was 21.6 there) (1.0 / 0.6 / 0.8 h came
    first, and 0.8 h left red patches bare). On the focused end, |smoothed change| ≥ 0.5 h,
    biggest first, spaced 4 cm · (0.5 h / c)² down to 0.8 cm, kept if seen (not end-on,
    on the left half) in ≥ 20 % of the iterates' frames and NOT OVERLAPPING any kept arrow
    on screen (tail–tip segments ≥ 18 px apart whenever both are on; 12 px for 12 px
    heads, 17 for 17; with 20 px heads and the 4 px halo, base length / gap → arrows:
    56/24 → 22, 45/24 → 25, 50/20 → 26, 45/20 → 28, 50/18 → 31, the user asking ≥ ~30; a 22 px mid-point
    gap capped density, and 70 px arrows along the junction's converging normals crossed),
    no cap, and not on a mesh piece < 3 % of the body at any iterate (that removed the
    1.69 h floating fragment at (0.055, 0.663, 0.735)).
    - Coverage check (prep, printed): on the last iterate's frame and the end pause, every
      connected visible patch of it_40 with |change| ≥ 0.5 h (≥ 8 vertices, fragments
      excepted; vertices visible by a BVH ray from the camera) must hold a visible arrow
      within 1.5 cm. An uncovered patch adds all its vertices as candidates and keeps the
      one whose arrow is seen best (a patch facing the camera squarely would get only
      end-on arrows, i.e. dots, and would be reported exempt; none is). Result: PASS
      after one round, 28 arrows of 364 candidates (junction 16, arm 6, legs/feet 6,
      all growth), on at it_0 / 10 / 20 / 40: 0 / 10 / 22 / 25, checked over the 30
      frames of the 1.0 s end pause; one 8-vertex patch at (0.08, 0.45, 0.32) faces the
      camera there and is exempt (every arrow on it would be end-on). 31 with the
      0.3 s pause (47: 29 / 11 / 7 with the
      thinnest arrows and their 12 px gap; 36 with 17 px heads and a 17 px gap).
    - Arrows nearly uniform: 50 px at the threshold (45, then +25 % asked; +11 % kept the
      count), up to 1.5× (68 px) with the site's
      change so far, a constant 20 px head; anchors are fixed sites tracked smoothly
      through the iterates (never re-sampled), the direction their smoothed normal.
    - Each anchor is tracked back from it_40 (nearest point on it_k of its point on
      it_k+1). The chain slides sideways where a junction fills, so the direction is the
      anchor's NORMAL, Gaussian-smoothed over iterations (σ = 6: σ = 3 left 25° jumps
      per update; now ≤ 10°, median ~3°), signed by the site's change.
    - Occlusion is EXACT, per pixel, from ONE renderer (the user: arrows still showed
      over the slats in front of them; two views reset their clipping ranges to their
      own bounds, so their depth buffers were not comparable). The arrows are a second
      display in the iterate's view; each arrow frame is shot twice from it, body only
      (the arrows at opacity 0, so bounds and clipping range stay the same) and body +
      arrows, and the compose keeps the pixels where the combined depth is nearer than
      the body-only depth, and only inside the arrows' own footprint (the ID shot, grown
      2 px): along body silhouettes the two depth shots differ slightly, and those pixels
      drew black outline polylines across the body (the user: "weird lines"). Kept
      pieces under 30 px are dropped (an arrow partly behind a slat left scattered pixels,
      each ringed by the outline: dotted specks; a 3×3 opening instead ate into the thin
      shafts and drew the outline inside them as dashes). The |u| kit's FXAA and tone mapping render through a
      post-processing buffer, and the window's depth then reads back EMPTY (all 1): the
      first per-pixel cut was void because of it, so `grab_z` turns both off for the
      depth read only (it asserts a non-empty buffer). Check over all 129 arrow frames:
      the two depth buffers agree exactly (max |Δz| = 0.0 over 24.2 M body pixels with
      no arrow), and 0 kept arrow pixels have the body nearer. A white knock-out behind
      the live "−NN %" let the bracket pass behind the label; it read as a broken line, so
      the label now sits in the plot's empty lower-left interior (left edge at iteration
      1.6, baseline at 0.20), which neither the curve (≥ 0.575) nor the bracket reaches.
    - Per-arrow fade (the user: partly occluded arrows left red specks): a flat ID shot
      (arrows only, red channel = index + 1, ambient only, no FXAA) gives each arrow's
      pixels; its visible share = kept pixels / its side-on area at its current length
      (`arrow_nominal.json` from dv_prep; against its own footprint an end-on or tiny
      arrow counted as fully seen), averaged over ±4 frames, off below 35 % and on again
      at ≥ 45 %, the state averaged over ±4 frames (`arrow_fades.json`). Every arrow pixel
      (AA fringe included) takes its nearest valid-ID pixel's fade, outline too.
    - ONE visibility per arrow, owned by the compose (the user: "the arrows flash or
      blink"). Measured first (per arrow, per frame, visible ID pixels × fade): 35 flips,
      10 of them reversed within 0.5 s, some frame to frame. Causes: prep scaled each
      arrow's geometry by its own end-on state and threshold gate (arrows grew from dots)
      while the compose's share fade moved the other way, and the per-frame visible count
      jumped as the crossfade's dominant iterate (the occluding body) changed. Now prep
      writes every kept arrow at full size in every frame, and `arrow_fades` sets its
      alpha: target = site past the threshold (`arrow_meta.json` t_cross) AND visible
      share (ID pixels kept by depth / side-on area) Gaussian-smoothed σ = 4 frames, on
      at ≥ 50 %, off below 35 %, ≥ 0.4 s between changes; alpha slew-limited, a full fade
      ≥ 0.3 s (now: rises ≥ 0.3 s, falls ≥ 0.5 s with look-ahead, below). The 30 px piece drop (another per-frame switch) is gone; instead each
      visible piece of an arrow, and its outline, weighs smoothstep((px − 15) / 45), so
      slivers beside a slat fade continuously and leave no black outline squiggles.
      After: 25 flips (each a single fade-in at the threshold or one occlusion fade, ≥ 1 s
      apart), 0 reversals within 0.5 s, max alpha step per frame 0.111.
    - No arrow disappears suddenly (the user). Rule: no arrow loses more than 1/15 of its
      alpha a frame (a full fade-out ≥ 0.5 s). The frames are rendered offline, so
      `arrow_fades` LOOKS AHEAD: a forward pass slew-limits rises (1/9 a frame), a
      backward pass caps each frame at the next frames' alpha + 1/15 per frame between, so
      an arrow that will go behind the body (smoothed visible share below 35 %) has
      finished fading by the frame it does. The beat's global arrow opacity is folded in
      as a MIN (a product of two falling ramps falls faster than either). The end fade
      starts with the shrink: full through the last iterate and the whole end pause (the
      user: that is where the result is read; a 20.55–21.3 smoothstep left them faint
      there), then LINEAR 22.8–23.3 at exactly 1/15 a frame, ending with the iterate body
      they sit on (OPT_FADE; the arrows render in its view). It was 0.15 s, 20.95–21.1,
      0.32 a frame, at first. The arrows are rendered on the shrink's oversized frames too, and
      the coverage check runs on every end-pause frame (30 with the 1.0 s pause, the spin
      turning patches into and out of view) whatever the fade.
      Two more pops went with it: the halo ring and the piece labels were cut from the
      FADED alpha at 0.5, so every halo vanished the frame its arrow's fade passed 0.5 (they
      are now cut from the coverage before the fade), and the ID lookup took AA-edge pixels
      (alpha > 128) whose blended red value is another arrow's id, drawing 1 px dots of
      hidden arrows (now alpha ≥ 250 only). Off is also triggered by the RAW share (an
      iterate switch can bury an arrow's base in one frame). Audit (`DV_ARROW_LOG=DIR` on
      the compose writes each arrow's visible weight per frame): before, largest alpha
      drop 0.324 a frame (the 0.15 s end fade), 0.111 elsewhere; after (the late linear
      end fade, 31 arrows), largest alpha drop 0.067 everywhere. Visible weight against
      the arrow's own footprint that frame: before, largest drop 0.32 a frame, 23 arrows
      above 1/15, one going from ≥ 30 % to ≤ 5 % within 3 frames; after, 0.14 (arrow 8
      at the shrink's first frame, where the oversized canvas reveals its off-frame part:
      a bookkeeping jump, not on screen), 12 above 1/15 (fade plus the camera pulling
      back), none abrupt. No arrow rises once the end fade has begun (one fading in as
      the camera pulled back came and went within it).
    - The end pause shows every arrow FULLY on or FULLY off (the user: half-faded and
      head-only arrows at the end read as noise). `dv_prep` marks an arrow `pause_on`
      (`arrow_meta.json`) if, by ray casting 9 points along it against it_40's BVH, ≥ 60 %
      of it is unoccluded and it is not end-on in ≥ half the pause's frames, AND it stays
      ≥ 60 % unoccluded through the end fade's first frames (alpha > 0.3): an arrow full
      at the pause's end, hidden by the body 8 frames into the shrink, would vanish at
      alpha 0.5. The coverage check counts only those (an arrow on a patch it could only
      show in part leaves it to an exemption). `arrow_fades` then holds pause_on arrows
      at 1 from the pause's first frame (their fade-in starts 9 frames before it) until
      the first pause frame the body hides > 40 % of one (exact: its ID pixels against
      the depth test, not the nominal-area share, which reads ~0.75 for a fully visible
      but foreshortened arrow), with the look-ahead fade-out ending there; the rest are
      faded out before the pause. Audit over the pause's 30 frames (2026-11-09): 12 of
      the 28 arrows on in every frame, 16 off in every frame, none with alpha in
      (0.1, 0.9), none at full alpha with an unoccluded share in (5 %, 60 %); coverage
      PASS on the shown 12 (one 8-vertex patch exempt). Of the 16 off, 8 are fully behind
      the body at the pause's headings; the others were half-hidden, head-only or hidden
      early in the shrink.
    - `dv_prep` writes one arrow mesh per frame (`_work/oarrows/`); the renderer swaps
      the file.
  - The shrink (21.0–21.55) starts from the close-up: the camera pulls back out
    (closeup 1 → 0) while the compose shrinks the layer into the cell by the tiles' rule
    on `box_close_px` (the OPTIMISED bench, at rest and at ×200, under the ORBIT camera
    over the remaining headings), so it lands as the whole bench, not a cropped close-up.
    The layer holds only what the camera saw: in the close-up the bench runs up to 770 px
    off the frame's left and below its bottom, so a frame-sized layer, once scaled down,
    showed its own border as a straight cut through the legs and grid (f0636, y ≈ 963).
    The move's frames (0 < move < 1, f0631–f0646) are therefore rendered on a 2× canvas
    (same camera and pixel scale, the view angle widened so tan(fov/2) doubles;
    `TL.oversize`), and the compose anchors the frame's origin at the canvas's (960, 540);
    the grid alone fades out over 240 px at that canvas's border (it is fading with the
    move anyway). Checked: the oversized centre equals the normal render (mean |Δ| 0.003
    levels), and an automated test finds no alpha jump at any layer border inside the
    frame over 16.0–16.7 s and 21.0–21.6 s (now 22.7–23.3 s). (A soft feather at the frame-sized layer's
    border was rejected: the legs would fade away.)
    Its colour crossfades back to |u| 22.8–23.3 and s(t) restarts from 1 (same k = 200);
    its ghost (undeformed it_40) returns 22.9–23.2. Its |u| is a forward solve of it_40
    with the session's per-part materials (`dv_cleanup.run_parts`, pyamgx as the
    session's own solve), coloured on the SAME |u| range as the original bench (the
    session's |u|max), so the stiffer bench reads lighter: |u|max 0.586 mm vs 0.857 mm
    (−32 %). New material (7.2 % of it_40's volume, outside the original body) takes the
    nearest occupied cell's part (`dense_grid_from_sparse`): 53 % steel, 47 % wood. The
    other 20 tiles are unchanged.
  - The batch records name the old checkout path `/home/xl2254/implicit_optimize`, which no
    longer exists; `dv_prep.run_key` matches records to runs by their path below
    `agentic_batch50_opus/`, and the batch is read from the checkout's own `output/`.
- Check frames at 360 px wide (phone) before shipping, and a frame sequence around every
  transition.

---

## 6. HARD-WON LESSONS (read before shipping any figure)

- **Compute the geometry, don't fake it.** Generate real structure (quadtree, SDF, hanging nodes) in
  Python and emit from data; a hand-drawn "quadtree" isn't 2:1-balanced and an expert spots it.
- **Verify numerically, not by eye.** Assert invariants in code (e.g. 0 balance violations; every drawn
  hanging node = coarse-edge midpoint = finer corner, one level apart).
- **Never make a figure prettier by deleting its subject.** A blemish is usually the operator's real
  behaviour under a badly chosen input, so tune the *input* until the operator has genuine work to do
  — do not tune the input until there is nothing to see. §5e is the cautionary case: the BC-zip seam
  can be driven below the mesh's own faceting by shrinking the difference the zipper has to bridge,
  which yields a flawless render of nothing happening.
- **Zoom into PNG crops region-by-region** (see §2) — non-negotiable; the colorbar especially.
- **Leaders/labels must land ON the feature they name** — target the surface (normal-offset by the stroke
  radius), not a centerline, or the endpoint floats inside the fill.
- **SVG marker gotcha** (§3) and **legend glyphs horizontal**.
- **Show the story where it's visible; filter the rest.** Refinement grading reads in open regions; draw the
  meaningful subset (52 hanging nodes, not 398). Less well-chosen detail beats a cluttered dump.
- **Notation/claims exact.** 2-D octree figure = **quadtree**; interface = `s=0` / `φ=½` (NOT φ=0); no
  overclaiming ("topology-optimized", not "compliance-optimal", unless solved). Experts read labels first.
  A caption must not state a *sufficient* condition the code only treats as *necessary*: BC nodes are
  "near-boundary nodes identified", NOT "nodes inside the boxes" (a box hit is only the pre-filter).
- **Check a field's actual range before trusting a threshold applied to it.** The TripoSG decoder
  truncates its SDF at 0.01, so `φ = ½(1−tanh(s/ε))` with `ε = h/2` never leaves (0.006, 0.994) and
  every `1e-3` phase-field test degenerates — the "boundary region" became the whole mesh and
  trimming removed nothing. The code looked right; the input made it vacuous (§5c).
- **A from-scratch rebuild is part of the work, not a formality.** Incremental runs keep deleted
  names in scope and keep deleted intermediates on disk, so they pass over exactly the breakage a
  reader will hit first. Every real reference/ordering bug in `diffused_nitsche` — a stats writer
  still naming a removed grid, a renderer run against files its own build had cleaned — showed up
  only on `build.sh <target>` from an empty folder (§5c).
- **Validate the palette; don't trust it because it is the house palette.** seaborn-deep's orange
  and green — this project's Neumann/Dirichlet pair — sit at CVD ΔE 4.5, below the hard floor of 6.
  The dataviz skill's validator finds that in one command; no amount of looking at it does (§5c).
- **Measure the code before believing the prose about it.** The paper text said selections become "a
  series of small bounding boxes"; `_cc_bboxes` emitted **one box per connected component**, so a
  flood-filled ear got 2 boxes at ~30% of the body diagonal that were **71% air** (27 521 candidate
  nodes → 7 871 accepted). Run the real selector on the real mesh and count before drawing it.  (The
  box cover has since been removed from the selector altogether, §5b.)
- **A reviewer's sentence may be an observation, not an instruction.** "The ears are just zoomed in
  at the same angle" was a *description*; it was read as a complaint, the close-up was re-aimed, and
  the next round of feedback was "the ears are not shown in the same angle as the previous two".
  When a remark could be either, ask — a wrong guess costs a full render + review cycle.
- **Before comparing two methods on a number, check that both sides were measured the same way.**
  The twist-cube study reported the embedded scheme as 3× cheaper on half the Newton iterations; both
  were artefacts of a stopping criterion that is relative to a residual the two schemes do not share
  (Nitsche's is penalty-dominated, five orders above the body-fitted one).  At a matched tolerance the
  costs are within 20 % and the iteration counts equal (§5h).  Anything the figure would *praise* one
  side for is where to look first.
- **An appearance is a measurement you have not made yet.**  Two mesh panels looked like different
  resolutions; the cells were 12.5 % apart and the drawn grids 1.63× apart, the difference being
  exactly the √2 of a section cut at 45° (§5h).  Fixing what it looked like would have misstated what
  it was.  Measure the pixels before changing the thing.
- **A measured error is only as good as what it is measured against — build two references.** A
  convergence study's finest points are exactly where a too-coarse reference bends the curve, and no
  amount of staring finds it. Running the whole `nitsche_convergence` sweep against both a 64³ and a
  96³ body-fitted reference settled it in 90 s: the Nitsche curves moved ≤ 0.4 %, the body-fitted
  curve's last point 15 % (§5h). The same habit applies to any "vs. ground truth" number in a figure.
- **When a fix creates a new artefact, the answer is usually the middle, not the other extreme.**
  Dark collar → flattened it to unshaded → "why does the white region have no shadow". Both ends
  were wrong; the fix was an ambient-heavy but still-lit material. Sweep the parameter, don't flip it.

---

## 7. CHECKLIST before declaring a figure done

- [ ] Master committed; `build.sh` regenerates the figure **from an empty folder** — delete the
      intermediates and run the target, or a stale name/file will only break for the next reader.
- [ ] **Zoomed PNG crops inspected** — every arrow, label endpoint, and the colorbar region.
- [ ] Computed geometry **numerically validated** (invariants asserted in code).
- [ ] Every leader/label lands ON the feature it names (not inside a fill).
- [ ] Arrowheads point the intended way; legend glyphs horizontal and un-swallowed.
- [ ] Fonts = **Arial** (real TTF in ParaView); colors/strokes consistent; no text/colorbar overlap or overflow.
- [ ] Labels/claims notationally correct (dimension name, interface value, no overclaim).
- [ ] SVG: `--export-text-to-path` PDF looks identical to the SVG. 3-D: confirmed GPU renderer, STL not PLY.
- [ ] Multi-panel: every quoted count read from the generated `*_stats.txt`, not typed by hand;
      panels sharing a camera cropped to a common box; rasterized PDF compared against the PNG.
- [ ] Overlay/highlight meshes are **disjoint** from the body layer (no z-fight speckle), and any
      faded context mask still **occludes** — nothing from the far side shows through it.
- [ ] Titles and legend keys in **Sentence case** (named primitives excepted); title band's ink
      extent measured against the canvas so nothing clips.
- [ ] Every colour that carries identity **run through the dataviz validator** (§5c) — the house
      palette is not exempt; ours had a CVD ΔE 4.5 pair in it.
- [ ] A panel label states the condition the **code** tests, not one derived from it; a port of a
      production predicate is **asserted equal** to it, not just believed.
- [ ] Any figure reporting **cost or iteration counts** for two methods: both were solved to a
      comparable convergence level, and the criterion is scale-free across them (§5h).
- [ ] Any figure whose deliverable includes a **measured wall clock**: the run had per-iteration
      output OFF (`SO_SAVE_EVERY=0` / `HP_SAVE_EVERY=0`, §2) — confirmed by `Start saving outputs`
      being absent from its `log.txt` — and nothing else was on the GPU while it ran.
- [ ] Any figure with a **cache** that skips the expensive step: the from-scratch path was forced
      once (`NC_FORCE=1` and friends) before shipping — a warm cache no longer proves it runs.

---

## 8. PROJECT CONTEXT — what the figures depict

5-stage implicit-geometry optimization pipeline:
1. Geometry → SDF → 2:1-balanced non-conforming **octree hex mesh** (image/trimesh → TripoSG latent).
2. GPU FEM + **density topology optimization** (Helmholtz filter → tanh-SIMP → GCMMA).
3. **Diffused Nitsche** BCs on non-conforming / rotated embedded meshes (twist-cube benchmark).
   Figures: `diffused_nitsche` — `s` → `φ` (ε = h_min/2) → boundary region → the Neumann/Dirichlet
   intersections with the resolved selector boxes → the trimmed octree mesh (§5c);
   `nitsche_convergence` — that benchmark actually run, the three discretisations of the twisting
   box beside the error curves they produce under refinement (§5h).
4. **SDF shape optimization** (boundary evolution, adaptive octree, MC/BVH reinit, BC-zip freeze).
   Figures: `shape_opt_loop` — one iteration as a closed forward/backward cycle (§5f);
   `void_nucleation` — the hidden phase field opening an interior void, four stages across (§5d);
   `hidden_phi_loop` — the same cycle run in phase-field mode, two fields at every station (§5g).
5. **Latent-space shape optimization** (backprop FEM loss through TripoSG decoder; pull-back-metric
   preconditioner ladder — NO figure yet, README section empty). See project memory files for methods.

`method_overview` (§5i) is Figure 1: the agent-assisted workflow as one forward/backward loop on
the bunny.  It covers three generative front ends, PartField segmentation, BC regions, optional
texture and per-part materials, the trimmed octree, the Nitsche solve, J, and ∂J/∂s (plus the
optional ∂J/∂φ) back into the octree.  It renders its own panels from the same octree, camera and
colours as `diffused_nitsche`, `partfield_segmentation` and `automatic_bc_bunny`.

### Objective plots `so_objective` / `hp_objective` (2026-10-13, designer's review)

Both are drawn by ONE function, `_shared/objective_style.py::draw()`, so they cannot drift:
88 × 55 mm single column (fixed mm layout, no `tight_layout`/`bbox_inches`), two stacked
panels sharing x 0–100 — compliance C/C₀ on top (23 mm, 0.9–2.2, ticks 1.0/1.5/2.0),
relative volume V/V₀ below (20 mm, 0.45–1.05, ticks 0.5/0.75/1.0).  No twin axes, no
arrows, no in-figure title.  Ink `#1a1a21` compliance, magenta `#a3195b` volume, grey
`#5f6672` for reference lines + annotations; 0.9 pt data, 0.5 pt axes/ticks/reference,
2 pt ticks; Arial incl. mathtext (`custom` fontset), 6.5/6/5.5 pt, Type 42, PNG 600 dpi.
The shown iteration is an open marker + dashed drop line, labelled at its foot
(`ha='left'`); HP shades iteration 35 → end in `#e7e2f5` ("Hidden field active").
`check()` fails the build on text < 5 pt, within 1 mm of the page edge, two overlapping
texts, or an annotation within 0.3 mm of a plotted line.  The volume panel is 20 mm (not
18) because the 22.6 mm "Relative volume, V/V₀" label would otherwise reach the x-tick row.
Legend line (not in the figure): "Compliance normalised by the initial design's, C/C₀, and
relative volume V/V₀ versus iteration; dotted line, the volume target V/V₀ = 0.5; open
marker, the iteration shown in the loop figure (shading in the hidden-φ run: the hidden
phase field is active from iteration 35)."

### 5l. `operator_loop` — the AI agent operator's loop (`./build.sh operator`)

`operator/op_figure.py`, plain matplotlib, NO pictures (the author's ask): a three-lane swim-lane
diagram for the Supplementary Methods subsection "Operation by a coding agent" (methods_draft
`methods.tex`).  Lanes: Operator (violet, the coding agent), Driver (script), Workflow gates (the
state machine's six approval gates).  Column heads are the operator's four steps: 1 Brief, 2 Plan,
3 Driver runs the workflow, 4 Judgment.  Author's rules (2026-10-18): NO grey text inside any
block (titles only, so the optional painting gate is not drawn); grey text elsewhere (lane
subtitles, heads) at >= 6.5 pt; legend in black at 6 pt, worded "Operator runs the driver or
approves the stop" / "Operator edits the plan and rewinds to the gate", no "Approval gate" entry.
Each look (geometry, regions, materials, fields) is a violet card with an eye.  Checked against
drive.py on dev (2026-10-17): the four looks are the run's always-or-usual stops in STOP_ORDER;
the route stops (revision, drawing, web_image), the texture stop and the pre-solve triage stop are
left out as a simplification.  The materials stop is `mode: parts` or an inferred uniform material
(plan_answer), hence "Materials", not "Per-part materials"; the regions-leaving check (bc_values:
contacts, statics, holds, sliding, load binding and shortfall) is "Support and load check".  The
orange stop arrow runs the same span as the violet call arrow beside it, with its octagon on the
arrow as the play mark sits on the violet one.  Driver-to-gate arrows leave each check card's
centre; Physical scale branches off halfway down the arrow to Geometry input, Boundary values and
Material off halfway down the arrow to Boundary regions.  Teal-bordered driver cards are the
driver's numeric checks.
