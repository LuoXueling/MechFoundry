#!/usr/bin/env python
"""A resident worker for the predefined FEM tools, kept warm between calls.

Every tool the monitor runs is its own process, so every call pays its imports
before it touches any data: 0.45s of numpy, trimesh and matplotlib for a render
or an export, and 3.4s for the GPU tools, which add torch and the warp kernel
modules.  That is pure latency in front of work that often takes less.

This serves the same tools out of one process that has already paid it.  The
request is the argv the subprocess would have been given, and it is parsed by
`workflow_tool`'s own parser, so the resident route and the one-shot route
cannot drift apart -- a subcommand added there is servable here with no change.
Nothing about correctness depends on this process existing: the monitor falls
back to launching the tool itself whenever the worker is absent, busy, or
broken.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import socket
import sys
import traceback
from pathlib import Path

# Importing the tool module is also what puts the repository root and
# src/warp_wrapper on the path, so `core` below resolves exactly as it does
# inside a tool.  Nothing runs on import; its own entry point is guarded.
sys.path.insert(0, os.fspath(Path(__file__).resolve().parent))
from workflow_tool import parser  # noqa: E402

# Half an hour with nothing to do is long enough to conclude the conversation
# that wanted us has moved on.  Exiting returns the memory; the next preload
# starts a new one, and a call that arrives meanwhile simply runs cold.
IDLE_TIMEOUT = 1800.0

TOOL_MODULE = Path(__file__).resolve().parent / "workflow_tool.py"
# The solver package the GPU tools and the forward solve import.  A worker holds
# whatever it imported for as long as it lives, so an edit here is as stale as
# an edit to the tools themselves.
SOLVER_PACKAGE = Path(__file__).resolve().parents[3] / "warp_wrapper" / "core"


def tool_digest() -> str:
    """What the tools looked like, so we can notice they no longer do.

    A resident worker holds the code it imported for as long as it lives, so an
    edit to the tools reaches every fresh subprocess and none of this one's
    calls -- which surfaces as the tool rejecting an argument the caller can
    plainly see it accepts.  The monitor already fingerprints this same file to
    decide whether a cached result still stands; reading it here keeps the
    worker honest about the same thing.
    """
    digest = hashlib.sha256()
    try:
        digest.update(TOOL_MODULE.read_bytes())
        for source in sorted(SOLVER_PACKAGE.glob("*.py")):
            digest.update(source.name.encode())
            digest.update(source.read_bytes())
    except OSError:
        return ""
    return digest.hexdigest()


LOADED_DIGEST = tool_digest()

# How long a caller that has opened a connection has to say what it wants.  It
# is abandoning the call, not thinking, that this is here for.
REQUEST_TIMEOUT = 30.0


def warm(tier: str) -> None:
    """Pay a tool's imports now, so the call that needs them does not.

    `light` covers everything that reads a mesh or writes a picture.  `cuda`
    adds what the GPU tools import, of which `core` is nearly all of it:
    `wp.init()` and a recursive `load_module` cost ~3s even against a warm
    kernel cache.  `solver` is what a forward solve pays before its first line
    of work: the solver classes, a CUDA context for torch and for warp, and
    AMGX's library initialisation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot  # noqa: F401
    import numpy  # noqa: F401
    import trimesh  # noqa: F401
    if tier in ("cuda", "solver"):
        import torch  # noqa: F401
        import core  # noqa: F401
    if tier == "solver":
        import warp as wp
        import core.shape_opt  # noqa: F401

        torch.zeros(1, device="cuda")
        wp.zeros(1, device="cuda")


def run_tool(argv: list[str], cwd: str = "") -> dict:
    """Run one tool exactly as its own process would, and report as one.

    stdout and stderr are captured and handed back so the receipt the monitor
    writes, and the backend log the UI reads from it, say the same thing they
    would have for a subprocess.  A traceback ends in the `ValueError:` or
    `RuntimeError:` line the monitor quotes when a tool fails, so a failure
    reads identically too.
    """
    out, err = io.StringIO(), io.StringIO()
    try:
        # The monitor runs a one-shot tool with the session directory as its
        # working directory, so a relative path in the arguments means something
        # there.  Serving the same call from a process that happens to sit
        # somewhere else would quietly resolve it against the wrong directory --
        # so this stands where the subprocess would have stood.  One request is
        # served at a time, which is what makes changing directory safe here.
        if cwd:
            os.chdir(cwd)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args = parser().parse_args(argv)
            artifacts = args.func(args)
            print(json.dumps({"artifacts": artifacts}, separators=(",", ":")))
    except SystemExit as exc:
        # argparse refused the arguments and would have exited the process.
        return {"ok": False, "stdout": out.getvalue(),
                "stderr": err.getvalue() or f"argument error: {exc}"}
    except BaseException:
        return {"ok": False, "stdout": out.getvalue(),
                "stderr": err.getvalue() + traceback.format_exc()}
    return {"ok": True, "stdout": out.getvalue(), "stderr": err.getvalue()}


def handle(request: dict) -> tuple[dict, bool]:
    """Answer one request, and say whether to keep serving."""
    action = request.get("action")
    if action == "shutdown":
        return {"ok": True, "result": "shutting_down"}, False
    if tool_digest() != LOADED_DIGEST:
        # The tools have been edited since they were imported.  Say so and stop:
        # the caller runs this one itself, and the next preload starts a worker
        # holding the code that is actually on disk.
        return {"ok": False, "stale": True, "stdout": "",
                "stderr": "tool module changed on disk since this worker loaded it"}, False
    if action == "warm":
        warm(request.get("tier", "light"))
        return {"ok": True, "result": "warm"}, True
    argv = request.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return {"ok": False, "stdout": "", "stderr": "argv must be a list of strings"}, True
    return run_tool(argv, str(request.get("cwd") or "")), True


def serve(socket_path: Path, tier: str, idle_timeout: float = IDLE_TIMEOUT,
          one_shot: bool = False) -> None:
    """Warm up, then answer until nobody has asked for anything in a while.

    The socket appears only once warming is done, so its presence is what says
    the worker is worth using.  Binding first would let a call arrive and queue
    behind the very imports it was hoping to skip.

    ``one_shot`` serves a single tool and exits: the forward solve's spare
    process, which wants its start-up paid in advance but the GPU to itself
    and every byte of it back afterwards.
    """
    warm(tier)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(os.fspath(socket_path))
        server.listen(8)
        server.settimeout(idle_timeout)
        running = True
        while running:
            try:
                connection, _ = server.accept()
            except (socket.timeout, TimeoutError):
                break
            with connection:
                try:
                    connection.settimeout(REQUEST_TIMEOUT)
                    # Greeting first, before a request has been read.  A caller
                    # that does not hear this promptly knows we are busy with
                    # something else and runs its tool itself -- having sent us
                    # nothing we might go on to run as well, which is how the
                    # same output would end up with two writers.
                    connection.sendall(b'{"ready":true}\n')
                    stream = connection.makefile("r", encoding="utf-8")
                    line = stream.readline()
                    if not line.strip():
                        continue  # the caller thought better of it
                    request = json.loads(line)
                    response, running = handle(request)
                    if one_shot and "argv" in request:
                        running = False
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    response = {"ok": False, "stdout": "",
                                "stderr": f"{type(exc).__name__}: {exc}"}
                # The caller may be gone by now -- it is free to stop waiting on
                # a tool it has decided to run itself.
                with contextlib.suppress(OSError):
                    connection.sendall(
                        (json.dumps(response, separators=(",", ":")) + "\n").encode()
                    )
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--serve", type=Path, required=True)
    args.add_argument("--warm", choices=("light", "cuda", "solver"), default="light")
    args.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT)
    args.add_argument("--one-shot", action="store_true",
                      help="exit after serving one tool (the forward solve's spare)")
    parsed = args.parse_args()
    serve(parsed.serve, parsed.warm, parsed.idle_timeout, parsed.one_shot)
    # Warp and torch can make interpreter teardown linger; the socket is already
    # gone and every artifact is written, so let the OS reclaim the process.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
