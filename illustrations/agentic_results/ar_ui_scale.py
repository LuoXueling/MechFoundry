"""The Streamlit UI's own deformation scale, called, not re-derived (env mechfoundry-agents).

    python ar_ui_scale.py SESSION_DIR [...]      # prints {session_dir: scale | null}

``src/ui/telemetry.py:deformation_scales`` (lines 727-751) is the slider the UI's
3-D viewer gets: zero up to DEFORMATION_REACH (0.3) x ``physical_extent`` /
max |u| (``solve/displacement_magnitude.glb.json`` "maximum") in ten steps.  The
figure shows the slider's TOP, ``scales[-1]``: the UI's "visibly deformed"
setting (the slider opens at 0, i.e. undeformed).  ``workflow`` is session.json,
exactly what ``controller.read_workflow`` hands the UI.  telemetry imports
altair / pandas, which only mechfoundry-agents carries, so ar_prep.py runs this
script there.
"""
import json
import os
import sys
from pathlib import Path

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, REPO)
from src.ui.telemetry import DEFORMATION_REACH, deformation_scales  # noqa: E402

out = {}
for d in sys.argv[1:]:
    root = Path(d)
    try:
        workflow = json.loads((root / "session.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        workflow = {}
    scales = deformation_scales(root, workflow if isinstance(workflow, dict) else {})
    out[d] = {"scale": scales[-1] if scales else None, "reach": DEFORMATION_REACH,
              "physical_extent": workflow.get("physical_extent"), "steps": len(scales)}
print(json.dumps(out))
