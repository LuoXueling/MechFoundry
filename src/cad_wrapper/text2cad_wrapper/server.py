#!/usr/bin/env python
"""Resident Text2CAD model server with the workflow's safe greeting protocol."""

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

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(REPO_ROOT))

from src.cad_wrapper.text2cad_wrapper.availability import settings
from src.cad_wrapper.text2cad_wrapper.generate import Text2CADRuntime, parser, run

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


def handle(request: dict, runtime: Text2CADRuntime) -> tuple[dict, bool]:
    if request.get("action") == "shutdown":
        return {"ok": True, "result": "shutting_down"}, False
    if source_digest() != LOADED_DIGEST:
        return {"ok": False, "stale": True, "stdout": "",
                "stderr": "Text2CAD wrapper changed on disk"}, False
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


def serve(socket_path: Path, runtime: Text2CADRuntime, idle_timeout: float):
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


def main():
    _, _, default_checkpoint = settings()
    choices = argparse.ArgumentParser()
    choices.add_argument("--serve", type=Path, required=True)
    choices.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT)
    choices.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    choices.add_argument("--device", default="cuda")
    args = choices.parse_args()
    runtime = Text2CADRuntime(args.checkpoint, device=args.device)
    runtime.preload()
    serve(args.serve, runtime, args.idle_timeout)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
