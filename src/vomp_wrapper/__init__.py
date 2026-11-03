"""Optional high-performance bridge to NVIDIA VoMP material inference.

The package deliberately performs no eager imports.  The orchestration/UI
environment does not need NumPy/SciPy from the VoMP compute environment, and a
uniform solve must not acquire optional dependencies merely while releasing a
possibly-running worker.
"""

__all__: list[str] = []
