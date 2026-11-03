#!/usr/bin/env python
"""Run one generated CadQuery script in isolation and save its tessellation.

CAD-Coder answers with Python, so building its part means running code a model
wrote.  This script is the only place that happens, always in a child process of
its own (see ``run_sandboxed``), and it trusts nothing about the code:

- the source must pass an AST allowlist: imports of cadquery, math and numpy
  only; no name or attribute starting with an underscore (which closes the
  ``().__class__.__subclasses__()`` escapes); no builtins that reach files,
  code or attributes by name; no attribute that exports, imports, saves or
  loads anything;
- it runs with a reduced ``__builtins__`` and an import hook that admits the
  same three modules;
- the process runs with memory, CPU-time and file-size limits, in a throwaway
  directory, with an environment carrying no credentials, and -- where the
  kernel allows an unprivileged network namespace -- with no network at all.

What comes back is plain arrays: the fused solid tessellated face by face, the
B-rep face of each triangle, and each face's surface type.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ALLOWED_MODULES = frozenset({"cadquery", "math", "numpy"})
FORBIDDEN_NAMES = frozenset({
    "eval", "exec", "compile", "open", "input", "breakpoint", "help", "globals",
    "locals", "vars", "dir", "getattr", "setattr", "delattr", "hasattr",
    "memoryview", "exit", "quit", "type", "object", "super",
})
# Attributes that reach the file system, however a library spells them.
FORBIDDEN_ATTRIBUTE = re.compile(
    r"export|import|save|load|file|svg|stl|step|dxf|vtk|brep|pickle|memmap|ctypes|system|popen",
    re.IGNORECASE,
)
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "int", "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "Exception", "IndexError", "KeyError", "TypeError",
    "ValueError", "ZeroDivisionError", "True", "False", "None",
)
MEMORY_LIMIT = 8 << 30
CPU_SECONDS = 120
FILE_LIMIT = 256 << 20


class UnsafeCode(ValueError):
    pass


def check_source(code: str) -> None:
    """Reject anything outside the small language a CAD script needs."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise UnsafeCode(f"not Python: {exc}") from None
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            if module.split(".")[0] not in ALLOWED_MODULES:
                raise UnsafeCode(f"import of {module!r} is not allowed")
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in FORBIDDEN_NAMES):
            raise UnsafeCode(f"use of {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute) and (
                node.attr.startswith("_") or FORBIDDEN_ATTRIBUTE.search(node.attr)):
            raise UnsafeCode(f"attribute {node.attr!r} is not allowed")
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await)):
            raise UnsafeCode(f"{type(node).__name__} is not allowed")


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed")
    return builtins.__import__(name, globals, locals, fromlist, level)


def execute(code: str):
    """Run the checked script and return its result ``r`` as one CadQuery shape."""
    import cadquery as cq

    check_source(code)
    safe = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    safe["__import__"] = _guarded_import
    namespace = {"__builtins__": safe, "__name__": "cad_script"}
    exec(compile(code, "<cad-coder>", "exec"), namespace)
    result = namespace.get("r")
    if result is None:
        raise ValueError("the script did not assign its model to r")
    if isinstance(result, cq.Workplane):
        shapes = [value for value in result.vals() if isinstance(value, cq.Shape)]
    elif isinstance(result, cq.Shape):
        shapes = [result]
    else:
        raise ValueError(f"r is a {type(result).__name__}, not a CadQuery shape")
    solids = [solid for shape in shapes for solid in shape.Solids()]
    if not solids:
        raise ValueError("the script produced no solid")
    fused = solids[0].fuse(*solids[1:]).clean() if len(solids) > 1 else solids[0]
    return fused


def tessellate(shape, relative_tolerance: float = 1e-3):
    """Triangles of each B-rep face, with the face each came from and its type."""
    import numpy as np
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location

    # Mesh the solid once, so faces sharing an edge share its discretisation;
    # CadQuery's per-face tessellate meshes each face alone and the seams open.
    tolerance = relative_tolerance * shape.BoundingBox().DiagonalLength
    BRepMesh_IncrementalMesh(shape.wrapped, tolerance, False, 0.5, True)
    vertices, triangles, labels, surface_types = [], [], [], []
    for index, face in enumerate(shape.Faces()):
        surface_types.append(face.geomType().lower())
        location = TopLoc_Location()
        mesh = BRep_Tool.Triangulation_s(face.wrapped, location)
        if mesh is None:
            continue
        offset = len(vertices)
        transform = location.Transformation()
        for i in range(1, mesh.NbNodes() + 1):
            point = mesh.Node(i).Transformed(transform)
            vertices.append((point.X(), point.Y(), point.Z()))
        reverse = face.wrapped.Orientation() == TopAbs_REVERSED
        for i in range(1, mesh.NbTriangles() + 1):
            a, b, c = (offset + k - 1 for k in mesh.Triangle(i).Get())
            triangles.append((a, c, b) if reverse else (a, b, c))
        labels.extend([index] * mesh.NbTriangles())
    return (np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64),
            np.asarray(labels, dtype=np.int64), surface_types)


def run_sandboxed(code: str, timeout: float = 180.0):
    """Build ``code`` in a fresh sandboxed child; return its arrays or raise.

    Returns ``(vertices, triangles, labels, surface_types)``.
    """
    import numpy as np

    with tempfile.TemporaryDirectory(prefix="cad-coder-") as scratch:
        script = Path(scratch) / "script.py"
        script.write_text(code, encoding="utf-8")
        output = Path(scratch) / "tessellation.npz"
        command = [sys.executable, "-I", os.fspath(Path(__file__).resolve()),
                   "--code", os.fspath(script), "--output", os.fspath(output)]
        if shutil.which("unshare") and subprocess.run(
                ["unshare", "-rn", "true"], capture_output=True).returncode == 0:
            command = ["unshare", "-rn", *command]
        environment = {"PATH": "/usr/bin:/bin", "HOME": scratch, "OMP_NUM_THREADS": "1"}
        try:
            result = subprocess.run(
                command, cwd=scratch, env=environment, capture_output=True,
                text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"the generated script ran past {timeout:.0f} s") from None
        if result.returncode != 0 or not output.is_file():
            lines = [line for line in result.stderr.strip().splitlines() if line.strip()]
            raise RuntimeError(lines[-1] if lines else f"sandbox exited with {result.returncode}")
        with np.load(output) as saved:
            return (saved["vertices"], saved["triangles"], saved["labels"],
                    [str(kind) for kind in saved["surface_types"]])


def _limit_resources() -> None:
    for limit, value in ((resource.RLIMIT_AS, MEMORY_LIMIT), (resource.RLIMIT_CPU, CPU_SECONDS),
                         (resource.RLIMIT_FSIZE, FILE_LIMIT)):
        resource.setrlimit(limit, (value, value))


def main():
    choices = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    choices.add_argument("--code", type=Path, required=True)
    choices.add_argument("--output", type=Path, required=True)
    args = choices.parse_args()
    _limit_resources()
    import numpy as np

    try:
        shape = execute(args.code.read_text(encoding="utf-8"))
        vertices, triangles, labels, surface_types = tessellate(shape)
    except BaseException as exc:  # reported to the parent, which decides
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        sys.exit(1)
    np.savez_compressed(args.output, vertices=vertices, triangles=triangles, labels=labels,
                        surface_types=np.asarray(surface_types, dtype=str))


if __name__ == "__main__":
    main()
