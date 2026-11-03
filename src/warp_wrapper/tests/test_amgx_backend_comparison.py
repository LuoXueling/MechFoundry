"""End-to-end comparison of AMGX bindings on a Diffused Nitsche solve."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
HELPER = Path(__file__).with_name("run_nitsche_backend_case.py")


@pytest.mark.skipif(
    importlib.util.find_spec("jaxamg") is None
    or importlib.util.find_spec("pyamgx") is None,
    reason="both jaxamg and pyamgx are required",
)
def test_diffused_nitsche_jaxamg_matches_pyamgx(tmp_path):
    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    outputs = {}
    for backend in ("jaxamg", "pyamgx"):
        output = tmp_path / f"{backend}.npz"
        completed = subprocess.run(
            [
                sys.executable,
                str(HELPER),
                "--backend",
                backend,
                "--output",
                str(output),
            ],
            cwd=ROOT_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            pytest.fail(
                f"{backend} subprocess failed\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        outputs[backend] = np.load(output)

    jax_result = outputs["jaxamg"]
    py_result = outputs["pyamgx"]
    assert float(jax_result["residual"]) < 1.0e-4
    assert float(py_result["residual"]) < 1.0e-4
    assert str(jax_result["dtype"]) == "float32"
    assert np.max(np.abs(jax_result["displacement"])) > 1.0e-6
    np.testing.assert_allclose(
        jax_result["displacement"],
        py_result["displacement"],
        rtol=5.0e-3,
        atol=1.0e-6,
    )


def test_a_stalled_solve_says_so_and_names_what_to_turn_on():
    """A plateau and a shortfall are different failures, and read differently.

    Stopping 10^5 above the tolerance is the float32 rounding floor of an
    ill-conditioned system -- the same matrix stalls at a different level from
    run to run -- and no tolerance or iteration count reaches past it; the
    conditioning has to change.  Stopping just short of the tolerance is a
    solve that was still working, and wants iterations.  Saying only
    "NOT_CONVERGED" leaves the reader to guess which one they have.
    """
    # This module drives the solver through a subprocess helper, so its own
    # header leaves sys.path alone; reading one function out of it needs the
    # same insertion every other test module makes at import.
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from core.jax_amg_solver import _solve_failure_message, jaxamg_config

    config = jaxamg_config(3, use_mixed_precision=False)
    tolerance = float(config["solver"]["tolerance"])
    max_iters = int(config["solver"]["max_iters"])

    def message(residual, iterations, *, mixed=False, scaled=False):
        return _solve_failure_message(
            {"status": "AMGXStatus.NOT_CONVERGED", "iterations": iterations,
             "residual": residual},
            config, use_mixed_precision=mixed, scaled=scaled,
        )

    stalled = message(3.476e-02, max_iters)
    assert "stalled" in stalled and "rounding floor" in stalled
    assert "in float32" in stalled
    # Every knob that is still off, in the order worth trying them.
    assert stalled.index("scale_system_to_unit_diagonal=True") < stalled.index(
        "beta_max_over_e"
    ) < stalled.index("use_mixed_precision=True")
    assert "max_iters" not in stalled, "more iterations cannot reach past a floor"

    # What is already on is not offered again.
    done = message(3.476e-02, max_iters, mixed=True, scaled=True)
    assert "scale_system_to_unit_diagonal" not in done
    assert "use_mixed_precision" not in done
    assert "in float32" not in done
    assert "beta_max_over_e" in done

    short = message(tolerance * 8.0, max_iters)
    assert "ran out of iterations" in short and "max_iters" in short
    assert "8x short" in short
    assert "scale_system_to_unit_diagonal" not in short, "it was still converging"

    # A failure that is not about running out of iterations makes no guess.
    other = message(1.0e-09, 12)
    assert other.endswith("(residual=1.000e-09)")
    assert "try" not in other
