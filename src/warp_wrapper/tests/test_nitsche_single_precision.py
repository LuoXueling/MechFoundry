"""Accuracy regression for all-f32 JAX-AMG Diffused Nitsche."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = Path(__file__).with_name("run_nitsche_precision_case.py")


@pytest.mark.skipif(
    importlib.util.find_spec("jaxamg") is None,
    reason="jaxamg is required",
)
def test_diffused_nitsche_all_f32_has_fair_accuracy(tmp_path):
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    output = tmp_path / "f32.npz"
    completed = subprocess.run(
        [sys.executable, str(HELPER), "--precision", "f32", "--output", str(output)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"f32 Nitsche subprocess failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    result = np.load(output)
    assert str(result["dtype"]) == "float32"
    assert float(result["residual"]) < 1.0e-5
    # The published depth-3 result is 0.1371 against the 96^3 reference.  This
    # uses the same 64^3 volume metric with the shipped 64^3 reference; 0.16 is
    # a regression guard, not a claim that this deliberately coarse mesh is
    # high-accuracy.
    assert float(result["error"]) < 0.16
