#!/usr/bin/env python
"""Paint a session's surface in the background, ahead of its material card.

Started by ``drive.py`` (``start_paint_ahead``) once the geometry is final and the plan is known
to paint at the material card (``plan.texture``, or per-part materials, which read colour).  It
runs the card's own paint (``graph.paint_material_texture_ahead``: the Hunyuan3D paint worker and
the six-view render, through the tool monitor and its receipts), so the card's ``texture_surface``
is served from those receipts when the runner reaches it -- after ``join_paint_ahead`` has waited
for this process to end.

    paint_ahead.py SESSION OUTPUT_ROOT ANSWER_JSON GPU_INDEX

The outcome goes to ``operator/paint_ahead.json`` (``state``, ``elapsed_s``, the VRAM report) and
one line of ``operator/gpu_log.jsonl``.  A failure only means the card paints for itself.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))
import sessionlib as lib  # noqa: E402

sys.path.insert(0, os.fspath(lib.REPO_ROOT))

RECORD = "paint_ahead.json"


def main(argv: list[str]) -> int:
    session, output_root = Path(argv[0]), Path(argv[1])
    answer = json.loads(argv[2])
    gpu = int(argv[3]) if len(argv) > 3 and argv[3].isdigit() else None
    record_path = lib.operator_dir(session) / RECORD
    sampler = lib.GPUSampler(output_root, gpu) if gpu is not None else None
    if sampler:
        sampler.start()
    started = time.time()
    outcome: dict = {}
    try:
        from src.agents_wrapper.fem_workflow.graph import paint_material_texture_ahead
        outcome = {"state": "done", **paint_material_texture_ahead(session, answer)}
    except Exception as exc:  # the card paints for itself
        outcome = {"state": "failed", "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                   "traceback_tail": traceback.format_exc().splitlines()[-6:]}
    report = sampler.finish() if sampler else None
    record = lib.read_json(record_path, {}) or {}
    record.update(outcome, elapsed_s=round(time.time() - started, 1), finished=time.time(),
                  gpu=report)
    lib.write_json(record_path, record)
    if report is not None:
        lib.append_jsonl(lib.operator_dir(session) / "gpu_log.jsonl", {
            "time": time.time(), "kind": "material_input", "action": "texture_surface (ahead, background)",
            "workers": ["hunyuan3d"], "ok": outcome.get("state") == "done", "index": gpu, **report})
    return 0 if outcome.get("state") == "done" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
