"""Conda environment discovery and readiness checks for FEM compute tools."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_ENV_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_PROBE = r"""
import json
result = {"torch": False, "warp": False, "torch_cuda": False, "warp_cuda": False}
try:
    import torch
    result.update(torch=True, torch_version=str(torch.__version__),
                  torch_cuda=bool(torch.cuda.is_available()))
except Exception as exc:
    result["torch_error"] = f"{type(exc).__name__}: {exc}"
try:
    import warp as wp
    result.update(warp=True, warp_version=str(wp.__version__),
                  warp_cuda=bool(wp.is_cuda_available()))
except Exception as exc:
    result["warp_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(result, separators=(",", ":")))
"""


@dataclass(frozen=True)
class ComputeEnvironmentStatus:
    name: str
    torch: bool
    warp: bool
    torch_cuda: bool
    warp_cuda: bool
    torch_version: str | None = None
    warp_version: str | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.torch and self.warp and self.torch_cuda and self.warp_cuda

    @property
    def summary(self) -> str:
        if self.error:
            return self.error
        return (
            f"Torch {self.torch_version or 'unavailable'} "
            f"(CUDA {'yes' if self.torch_cuda else 'no'}); "
            f"Warp {self.warp_version or 'unavailable'} "
            f"(CUDA {'yes' if self.warp_cuda else 'no'})"
        )


class ComputeEnvironmentError(RuntimeError):
    """Raised when a selected environment cannot run the GPU workflow."""


@lru_cache(maxsize=1)
def conda_environments() -> tuple[str, ...]:
    """Return named Conda environments detected by the active Conda executable."""
    result = subprocess.run(
        ["conda", "env", "list", "--json"], capture_output=True, text=True,
        timeout=15, check=False,
    )
    if result.returncode != 0:
        raise ComputeEnvironmentError(result.stderr.strip() or "could not list Conda environments")
    paths = json.loads(result.stdout).get("envs", [])
    names = []
    for raw in paths:
        path = Path(raw)
        name = "base" if path.parent.name not in ("envs", "env") else path.name
        if _ENV_NAME.fullmatch(name) and name not in names:
            names.append(name)
    return tuple(names)


@lru_cache(maxsize=32)
def probe_compute_environment(name: str, timeout: int = 45) -> ComputeEnvironmentStatus:
    """Check Torch, Warp, and CUDA from inside a named Conda environment.

    A pre-rename ``implicit-*`` name (a resumed session's) is probed under its ``mechfoundry-*``
    name once the old environment is gone; the status carries the name actually probed.
    """
    name = _current_environment(name)
    if not _ENV_NAME.fullmatch(name):
        raise ValueError("invalid Conda environment name")
    try:
        result = subprocess.run(
            ["conda", "run", "-n", name, "python", "-c", _PROBE],
            capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ComputeEnvironmentStatus(
            name, False, False, False, False,
            error=f"{type(exc).__name__}: {exc}",
        )
    payload = None
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "torch" in candidate:
            payload = candidate
            break
    if payload is None:
        detail = result.stderr.strip() or result.stdout.strip() or "probe produced no result"
        return ComputeEnvironmentStatus(
            name, False, False, False, False,
            error=detail.splitlines()[-1][:500],
        )
    error = None if result.returncode == 0 else (result.stderr.strip() or "probe failed")
    return ComputeEnvironmentStatus(
        name=name,
        torch=bool(payload.get("torch")), warp=bool(payload.get("warp")),
        torch_cuda=bool(payload.get("torch_cuda")),
        warp_cuda=bool(payload.get("warp_cuda")),
        torch_version=payload.get("torch_version"),
        warp_version=payload.get("warp_version"), error=error,
    )


# A ready environment stays ready until something is installed into it, which touches its folder.
COMPUTE_ENV_FRESH_S = 86_400.0


def _ready_cache() -> Path:
    """Where ready verdicts are kept: the user's own cache, so every output root and front end
    (the app, the CLI, the operator runner and its batch slots) shares one."""
    home = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return Path(home) / "implicit_optimize" / "compute_env.json"


def _current_environment(name: str) -> str:
    """``runtime_files.current_environment``: the one table of the MechFoundry env rename."""
    try:  # imported here: the package imports this module, and builds the graph on import
        from src.agents_wrapper.fem_workflow.runtime_files import current_environment
    except ImportError:  # no LangGraph in this interpreter: the name as given
        return name
    return current_environment(name) if isinstance(name, str) else name


def _env_stamp(name: str) -> float | None:
    """The environment folder's modification time, or None when it cannot be found."""
    try:
        from src.agents_wrapper.fem_workflow.runtime_files import direct_interpreter
        interpreter = direct_interpreter(name)
        return interpreter.parents[1].stat().st_mtime if interpreter is not None else None
    except Exception:  # a cache key only: without it, the check simply runs
        return None


def require_compute_environment(name: str) -> ComputeEnvironmentStatus:
    """``probe_compute_environment`` or raise; a ready verdict remembered for a day.

    The probe starts the environment and imports torch and warp with CUDA: 5-8 s at every
    session start, in every new output root (each operator batch slot paid it, 2026-10-09).  A
    ready verdict is reused while the environment's folder is unchanged; a failed one never is.
    """
    name = _current_environment(name)
    stamp = _env_stamp(name)
    path = _ready_cache()
    try:
        cache = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cache = {}
    hit = cache.get(name) if isinstance(cache, dict) else None
    if (stamp is not None and isinstance(hit, dict) and hit.get("stamp") == stamp
            and time.time() - float(hit.get("time") or 0) < COMPUTE_ENV_FRESH_S):
        return ComputeEnvironmentStatus(**hit["status"])
    status = probe_compute_environment(name)
    if not status.ready:
        raise ComputeEnvironmentError(
            f"Conda environment {name!r} is not GPU-ready: {status.summary}"
        )
    if stamp is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            temporary.write_text(json.dumps({**(cache if isinstance(cache, dict) else {}), name: {
                "time": time.time(), "stamp": stamp, "status": dict(status.__dict__)}}), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            pass
    return status
