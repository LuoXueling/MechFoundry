# demo_video

A 29 s (28.8 s), 1920×1080, 30 fps demonstration movie. It starts with a Claude Code prompt
("Look at AGENTS.md, simulate a park bench"). The bench then goes through every look of
the agentic-results figure: clay, paint, segmentation, supports and loads (seen from the
front), Young's modulus, then, under "Diffuse Nitsche FEA in NVIDIA Warp / on non-conforming
octree mesh", the solve's full adaptive octree cut by a plane fixed to the
bench, inside the box's frame, and finally |u| breathing 1× → 200× → 1×. After one cycle the deformation stops,
the camera moves into a close-up of one end in the left half, the spin slows, and under "Optionally, it can be
optimized for load-carrying ability" it plays its geometry clean-up's 41 iterates (5 s, then a 1 s hold on the result, still turning), coloured by their surface change (cyan–white–navy), while a
strain-energy plot draws on the right and navy arrows with a white halo mark where material is added; the optimised bench, deforming again, then shrinks into the grid. It ends on a 7×3 grid: the bench and 20
more Opus runs, each under its prompt, under "AI operator does all these fully autonomously / each
in 6.5 minutes on average".

```shell
./build.sh demovideo                 # from illustrations/, ~15–19 min on one A6000 (GPU 1),
                                     # + ~11 min once for the clean-up re-run (cached)
DV_DRAFT=1 ./build.sh demovideo      # half size, every 2nd frame, 15 fps
DV_KEEP_WORK=1 ./build.sh demovideo  # keep _work/ (passes + frames) for compose-only edits
DV_CLEANUP_FORCE=1 ./build.sh demovideo  # redo the clean-up re-run
```

Deliverables `../demo_video.mp4` (~9.2 MB) and `../demo_video.png` (the finale at 24.9 s, frame 747)
are tracked. `dv_stats.json` is the record: runs, ×N, |u|max, cameras, octree levels,
wall clocks. "6.5 minutes on average" is the mean `metrics.json` wall clock of all 50
Opus runs (6.46 min), read from the Opus category records; these are the same numbers
as the Opus columns of `agentic_results_compare_time.csv`'s `wall_clock` row.
`dv_prep` asserts the rounded value.

| Script | Env | Does |
|---|---|---|
| `dv_cleanup.py` | `mechfoundry-warp117` | the bench's clean-up re-run with every iteration's solver output, gated against the archived `traj.tsv`, and a per-part forward solve of its final shape (both cached in `output/geometry_cleanup_demovideo/`) |
| `dv_change.py` | numpy | the surface-change colour map: `experiments/geometry_cleanup/render.py`'s `lut()` mechanics (read from its source) with navy/cyan ends (and a half-way cyan knot at −0.5 h); the arrow colours |
| `dv_timeline.py` | plain Python | every time-dependent quantity (looks, spin, s(t), fades, tile pops) |
| `dv_camera.py` | numpy | ParaView's projection, shared by the camera fit and the compose |
| `dv_prep.py` | `mechfoundry-warp117` | `agentic_results/ar_prep` (imported) → `_work/ar_*.vtk`, arrows, orbit cameras; the beat's iterates, strain-energy curve, change arrows (one mesh per frame), framing and the optimised bench's \|u\| |
| `render_dv.py` | `pvbatch` (EGL) | `render_ar`'s panels (imported), one view per look, one PNG per pass per frame |
| `dv_compose.py` | `mechfoundry-warp117` | crossfades, ghost, grid, titles, names, CLI, colour bar, tiles → `_work/out/`; the change arrows' visibility (`arrow_fades`: look-ahead, no arrow loses more than 1/15 of its alpha a frame; `DV_ARROW_LOG=DIR` writes a per-arrow, per-frame audit) |

Inputs: `output/agentic_batch50_opus` and `output/geometry_cleanup/park-bench` (read
only; extract both from `output/archives/` first, the latter with
`zstd -dc output/archives/geometry_cleanup.tar.zst | tar -x -C output geometry_cleanup/park-bench`).
The clean-up re-run (`experiments/geometry_cleanup/run.py` with its opt-in
`GEOMETRY_CLEANUP_SAVE_EVERY=1`) is written to `output/geometry_cleanup_demovideo/` and kept
across builds. The choices and their reasons are in `../AGENTS.md` §5n.
