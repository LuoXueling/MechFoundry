# GPU usage

## The rule

**Every GPU-intensive step runs on an idle GPU, unless the user explicitly says otherwise**,
with its VRAM tracked while it runs. The driver enforces this; do not bypass it. The
GPU-intensive steps:
- 3-D generation: TripoSG, Hunyuan3D, CAD-Coder;
- SDXL drawing (`draw_image` on the geometry card), a one-shot process that loads SDXL and exits;
- texture painting;
- segmentation and region preview: PartField, shape diameter, `resolve_bcs`;
- VoMP;
- every solve and optimization.

**"Idle"** (flags `--max-foreign-mib`, `--max-util`): on each of 3 samples over about 3 s,
*foreign* memory under 1 GiB and utilization under 10 %.
- Memory is attributed per process: our resident workers, the driver's children and this
  user's workflow processes count as ours. Another user's model holding a small context (a few
  hundred MiB) does not make a card busy.
- Utilization cannot be attributed, so a card running anything is busy, including our own
  just-finished solve; the guard re-polls every 10 s.

**When nothing is idle**, the driver waits up to `--gpu-wait` (30 min), then returns `ok:false`
without running anything. Only the user may authorise `--gpu N` (a specific card, reported as a
user override) or `--allow-busy-gpu` (the least-loaded busy card).

CPU-only steps (scaling, applying values or materials, approving) are pinned to an idle card
when there is one, and otherwise run anyway, since they barely touch the GPU.

## How pinning works

- The driver sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<idle index>` in its
  own environment; every tool subprocess and worker it launches inherits them.
- **Resident workers keep the GPU they started on.** Before a GPU step the driver checks each
  worker the step needs (by its compute apps, or `/proc/<pid>/environ`) and stops any on the
  wrong card, so it restarts on the chosen one; the result's `notes` say so.
- **Output roots.** Workers are shared per output root and compute env. The Streamlit UI (root
  `output/fem_workflow`) starts them *unpinned*, defaulting to GPU 0, so the driver uses its own
  root, `output/fem_workflow_operator`.
  - If you must use the UI's root: check each result's `gpu.suspected_elsewhere`, and run
    `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu --stop-workers` when a UI-started worker is in the way.
  - Worker keys hash the *resolved* root, so relative and absolute spellings share workers;
    `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu --stop-workers` finds them by their pid files.
- **Before a solve** (the card's or `solve-backend`'s) the background paint is joined, then the
  model workers (TripoSG, PartField, CAD, Hunyuan, VoMP) are stopped and waited out, so the solve
  has the card to itself; a solve step's `baseline_mib` is read before that and may include them.

## Reading the `gpu` block of a result

`peak_delta_mib` is the step's extra memory; `workflow_peak_mib` all the workflow held on the
card, preloaded models included (the delta alone understates a preloaded step).
`foreign_appeared` (another job arrived) or `suspected_elsewhere` (the step ran on another card
through an unpinned worker: stop stray workers and re-run) means the result is re-checked.

`operator/gpu_log.jsonl` keeps one line per GPU step; `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py status` shows the live table.

## Measured VRAM and time (RTX A6000, 48 GB)

VRAM is `workflow_peak_mib` where recorded, otherwise `peak_mib` minus the other card's usage.

| Step | Time (warm) | VRAM on the card |
|---|---|---|
| TripoSG text, 16 steps (worker preloaded) | 4–9 s | about 7 GB |
| TripoSG image, 50 steps | 11–18 s | about 7.2–7.8 GB |
| SDXL drawing (`draw_image`), one 1024² image, 30 steps, with its cut-out | 13 s, loading included | 8–13 GB |
| TripoSG scribble + text | 9 s | about 7.3 GB |
| CAD-Coder (model resident) | about 47 s for 3 attempts | about 17.9 GB (15.2 GB model) |
| Hunyuan3D-2.1 image → shape (no texture) | cold about 71 s, warm about 36 s | 7.3 GB resident, up to about 13 GB |
| Hunyuan3D paint (texture) | warm about 30 s, cold about 50 s | **about 30 GB** |
| PartField segmentation (+ vision call) | 8–11 s | 6–7 GB |
| shape-diameter segmentation | 7 s | about 2.4 GB |
| region preview (`resolve_bcs`) | 2–4 s | < 1 GB extra |
| per-part material inference | 4 s | about 3 GB |
| VoMP inference | 48 s | **about 27 GB** |
| forward solve, depth 7, float32, 61–62k nodes | 17–21 s | +3.9–4.3 GB |
| forward solve, depth 7, float64, 45–83k nodes | 7–12 s | +4.5–7.5 GB (about 5–8 GB total) |
| forward solve, depth 8, float64, 210–264k nodes | 18–19 s | about 19–20 GB |
| shape optimization, depth 7, float64, 23k nodes | about 1.5 s per iteration (100 in 146 s) | about 4 GB |
| shape optimization, depth 7, float64, `load_steps` 2, 8 iterations | 36 s | about 8 GB |
| shape optimization, depth 8, float64, 84k nodes | about 5 s per iteration (30 in 152 s) | 17.4 GB at the start, 11 GB by the end |

**Rules of thumb:**
- Depth 8 costs about 3× the memory of depth 7. A depth-8 optimization peaks in its first
  iterations and needs less as material goes, but each iteration takes about 3.5× as long:
  optimize at depth 7 unless the members need depth 8.
- The Hunyuan3D paint and VoMP need about 30 GB each: they must not share a card with anything big.
- Only one generator model stays resident at a time; switching costs a cold start.
- The first use of a kind pays Warp JIT and model loading. Kernels are cached per output root
  under `.runtime_cache/`, so a fresh root recompiles once.

## Contention symptoms

- Steps much slower than the table: check `python3 -u src/agents_wrapper/fem_workflow/operator_skill/scripts/drive.py gpu` for another job.
- `foreign_appeared: true`: re-run the step and compare.
- Garbled or NaN solutions after a run shared a card: re-run on an idle card before trusting
  them. Contention can corrupt optimization runs.
