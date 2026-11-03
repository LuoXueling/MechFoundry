"""Shared pytest isolation for process-global AMGX backends."""

import os
import subprocess
import sys

import pytest


_PYAMGX_CHILD = "IMPLICIT_PYAMGX_TEST_CHILD"
_PYAMGX_PREFIX = "src/warp_wrapper/tests/test_preconditioners.py::TestAMGXSolve"


def pytest_pyfunc_call(pyfuncitem):
    """Run direct pyamgx tests in a clean AMGX process.

    pyamgx and JAX-AMG both own AMGX's process-global initialization.  Either
    binding can make the other's later initialization fail or become unstable,
    so a combined suite must not exercise them in one interpreter.
    """
    if os.environ.get(_PYAMGX_CHILD) or not pyfuncitem.nodeid.startswith(
        _PYAMGX_PREFIX
    ):
        return None

    env = os.environ.copy()
    env[_PYAMGX_CHILD] = "1"
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", pyfuncitem.nodeid],
        cwd=pyfuncitem.config.rootpath,
        env=env,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"isolated pyamgx test failed\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return True
