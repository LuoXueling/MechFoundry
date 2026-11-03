#!/usr/bin/env python
"""Resident Hunyuan3D-2.1 model server with the workflow's safe greeting protocol."""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import hashlib
import io
import json
import os
import socket
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))

from src.hunyuan3d_wrapper.generate import Hunyuan3DRuntime, parser, run

IDLE_TIMEOUT = 1800.0
REQUEST_TIMEOUT = 30.0
FILES = tuple(sorted(Path(__file__).resolve().parent.glob("*.py")))


def source_digest() -> str:
    digest = hashlib.sha256()
    try:
        for path in FILES:
            digest.update(path.read_bytes())
        return digest.hexdigest()
    except OSError:
        return ""


LOADED_DIGEST = source_digest()


def handle(request: dict, runtime: Hunyuan3DRuntime) -> tuple[dict, bool]:
    if request.get("action") == "shutdown":
        return {"ok": True, "result": "shutting_down"}, False
    if source_digest() != LOADED_DIGEST:
        return {"ok": False, "stale": True, "stdout": "",
                "stderr": "Hunyuan3D wrapper changed on disk"}, False
    argv = request.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        return {"ok": False, "stdout": "", "stderr": "argv must be a list of strings"}, True
    out, err = io.StringIO(), io.StringIO()
    try:
        cwd = str(request.get("cwd") or "")
        if cwd:
            os.chdir(cwd)
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            run(parser().parse_args(argv), runtime)
    except BaseException:
        return {"ok": False, "stdout": out.getvalue(),
                "stderr": err.getvalue() + traceback.format_exc()}, True
    return {"ok": True, "stdout": out.getvalue(), "stderr": err.getvalue()}, True


def fault_report(socket_path: Path) -> Path:
    """Where this worker reports a fatal signal: beside its socket, as the workflow's monitor
    (monitor.fault_report) reads it."""
    return Path(socket_path).with_name(Path(socket_path).name + ".fault")


def serve(socket_path: Path, runtime: Hunyuan3DRuntime, idle_timeout: float):
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    # A paint can die in native code (SIGSEGV in upstream's inpainting, a held-out run of
    # 2026-10-10), taking this worker with it.  faulthandler writes which signal to a file the
    # monitor reads, so it records the crash instead of running the same paint again cold.
    # Opened afresh by each worker: a report there is this worker's.
    fault = fault_report(socket_path).open("w", encoding="utf-8")
    faulthandler.enable(file=fault, all_threads=False)
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
                    connection.sendall(b'{"ready":true}\n')
                    line = connection.makefile("r", encoding="utf-8").readline()
                    if not line.strip():
                        continue
                    response, running = handle(json.loads(line), runtime)
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    response = {"ok": False, "stdout": "",
                                "stderr": f"{type(exc).__name__}: {exc}"}
                with contextlib.suppress(OSError):
                    connection.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)
        # A clean exit leaves no report (a crash never gets here, and its report stays).
        faulthandler.disable()
        fault.close()
        fault_report(socket_path).unlink(missing_ok=True)


def main():
    choices = argparse.ArgumentParser()
    choices.add_argument("--serve", type=Path, required=True)
    choices.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT)
    stages = choices.add_mutually_exclusive_group()
    stages.add_argument("--no-texture", action="store_true",
                        help="load the shape stage only; the paint stage loads on first use")
    stages.add_argument("--paint", action="store_true",
                        help="load the paint stage and SDXL's reference models only, for a "
                             "card that paints an existing surface; the shape stage loads on first use")
    args = choices.parse_args()
    runtime = Hunyuan3DRuntime(texture=not args.no_texture)
    if args.paint:
        runtime.preload(texture=True, shape=False)
        # Painting from a description draws its reference first; an image
        # reference never needs SDXL, but a description is the common case.
        # Without SDXL's weights the worker still paints from images.
        try:
            runtime.preload_reference()
        except Exception:
            traceback.print_exc()
    else:
        runtime.preload()
    serve(args.serve, runtime, args.idle_timeout)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
