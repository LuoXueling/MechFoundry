# demo_video

A 23 s, 1920×1080, 30 fps demonstration movie. It starts with a Claude Code prompt
("Look at AGENTS.md, simulate a park bench"). The bench then goes through every look of
the agentic-results figure: clay, paint, segmentation, supports and loads (seen from the
front), Young's modulus, then, under "Diffuse Nitsche FEA in NVIDIA Warp / on non-conforming
octree mesh", the solve's full adaptive octree cut by a plane fixed to the
bench, inside the box's frame, and finally |u| breathing 1× → 200× → 1×. It ends on a 7×3 grid: the bench and 20
more Opus runs, each under its prompt, under "AI operator does all these fully autonomously / each
in 6.5 minutes on average".

```shell
./build.sh demovideo                 # from illustrations/, ~11 min on one A6000 (GPU 1)
DV_DRAFT=1 ./build.sh demovideo      # half size, every 2nd frame, 15 fps
DV_KEEP_WORK=1 ./build.sh demovideo  # keep _work/ (passes + frames) for compose-only edits
```

Deliverables `../demo_video.mp4` (~9 MB) and `../demo_video.png` (the finale at 19.1 s)
are tracked. `dv_stats.json` is the record: runs, ×N, |u|max, cameras, octree levels,
wall clocks. "6.5 minutes on average" is the mean `metrics.json` wall clock of all 50
Opus runs (6.46 min), read from the Opus category records; these are the same numbers
as the Opus columns of `agentic_results_compare_time.csv`'s `wall_clock` row.
`dv_prep` asserts the rounded value.

| Script | Env | Does |
|---|---|---|
| `dv_timeline.py` | plain Python | every time-dependent quantity (looks, spin, s(t), fades, tile pops) |
| `dv_camera.py` | numpy | ParaView's projection, shared by the camera fit and the compose |
| `dv_prep.py` | `mechfoundry-warp117` | `agentic_results/ar_prep` (imported) → `_work/ar_*.vtk`, arrows, orbit cameras |
| `render_dv.py` | `pvbatch` (EGL) | `render_ar`'s panels (imported), one view per look, one PNG per pass per frame |
| `dv_compose.py` | `mechfoundry-warp117` | crossfades, ghost, grid, titles, names, CLI, colour bar, tiles → `_work/out/` |

Input: `output/agentic_batch50_opus` (read only; extract its archive first). The
choices and their reasons are in `../AGENTS.md` §5n.
