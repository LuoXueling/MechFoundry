"""Benchmark bunny octree construction and four-support shape optimization."""

import argparse
import datetime as dt
import json
import os
import re
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(HERE))
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from core.bc_surface_selectors import HighestSelector, LowestSelector
from core.shape_opt import ShapeOptTrimmed
from src.triposg_wrapper.trimesh_sdf import trimesh_to_octree


# Void kept between the geometry and the octree root cube, per side, as a
# fraction of the geometry's own extent.
PAD_FRACTION = 0.15


def cube_spacing(mesh, depth, pad_fraction=PAD_FRACTION):
    """Leaf size and clip padding for an octree root cube sized to the mesh.

    The root cube side is ``h_target * 2**depth`` and is centred on the origin,
    so it has to span ``2 * max|bounds|`` plus the void on either side.  Solving
    that for ``h_target`` spends the whole depth budget on the geometry; a fixed
    cube side spends part of it resolving empty padding whenever the part is
    smaller than the cube.

    The padding is an integer number of cells plus half a cell.  The background
    grid is snapped so its planes land on ``clip_bbox_lo + n*h_target``, and
    ``clip_bbox_lo = geometry_min - padding``; a whole number of cells would put
    the flat foot caps exactly on a grid plane, where the iso-surface is tangent
    to the lattice and marching cubes degenerates into slivers.
    """
    half_extent = float(np.abs(np.asarray(mesh.bounds)).max())
    cells = 2**depth
    # Provisional spacing from the requested void fraction, then round the void
    # to whole cells and re-solve so the cube side closes exactly.
    h_provisional = 2.0 * half_extent * (1.0 + 2.0 * pad_fraction) / cells
    pad_cells = max(1, round(2.0 * half_extent * pad_fraction / h_provisional - 0.5))
    if cells <= 2 * pad_cells + 1:
        raise ValueError(f"depth {depth} is too coarse to hold {pad_cells} pad cells")
    h_target = 2.0 * half_extent / (cells - (2 * pad_cells + 1))
    return h_target, (pad_cells + 0.5) * h_target


def build_geometry():
    body = trimesh.creation.cylinder(radius=1.0, height=1.0, sections=96)
    feet = []
    for angle in np.deg2rad([0.0, 90.0, 180.0, 270.0]):
        foot = trimesh.creation.cylinder(radius=0.24, height=0.28, sections=48)
        foot.apply_translation(
            [0.6 * np.cos(angle), 0.6 * np.sin(angle), -0.5 - 0.22 + 0.14]
        )
        feet.append(foot)
    mesh = trimesh.boolean.union([body, *feet])
    trimesh.repair.fix_normals(mesh)
    return mesh


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


_TIMESTAMP = re.compile(r"^\[(\d\d-\d\d-\d{4},\d\d:\d\d:\d\d\.\d+)\]\s*(.*)$")


def iteration_times(log_path):
    events, origin = [], None
    for line in log_path.read_text(errors="replace").splitlines():
        match = _TIMESTAMP.match(line)
        if not match:
            continue
        stamp = dt.datetime.strptime(match.group(1), "%m-%d-%Y,%H:%M:%S.%f")
        origin = origin or stamp
        events.append(((stamp - origin).total_seconds(), match.group(2)))
    starts = [i for i, (_, message) in enumerate(events)
              if message.startswith("===== Optimization Iteration")]
    rows = []
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(events)
        segment = events[start:stop]
        end = next((t for t, message in segment
                    if message.lstrip().startswith("Opt iter")), segment[-1][0])
        rows.append(end - segment[0][0])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--init-depth", type=int, default=3)
    parser.add_argument("--optimization-depth", type=int, default=7)
    parser.add_argument("--optimization-iters", type=int, default=4)
    parser.add_argument("--output", type=Path, default=HERE / "benchmark.json")
    args = parser.parse_args()

    bunny = trimesh.load(HERE / "bunny.stl", force="mesh")
    half_side = float((HERE / "bunny_sdf.meta").read_text().splitlines()[0])
    octree_call = lambda: trimesh_to_octree(
        bunny,
        bounds=(-half_side, -half_side, -half_side, half_side, half_side, half_side),
        init_octree_depth=args.init_depth,
        octree_depth=args.depth,
        domain_length=2.0 * half_side,
        sdf_truncation=0.01,
    )
    octree_call()  # exclude one-time JIT compilation
    octree, octree_seconds = timed(octree_call)
    dense_call = lambda: trimesh_to_octree(
        bunny,
        bounds=(-half_side, -half_side, -half_side, half_side, half_side, half_side),
        init_octree_depth=args.init_depth,
        octree_depth=args.depth + 1,
        domain_length=2.0 * half_side,
        sdf_truncation=0.01,
    )
    dense_octree, dense_seconds = timed(dense_call)

    mesh = build_geometry()
    h_target, padding = cube_spacing(mesh, args.optimization_depth)
    bounds = np.asarray(mesh.bounds)
    fixed_fraction = float(f"{0.02 / (0.5 - (-0.5 - 0.22)):.4f}")
    work = tempfile.TemporaryDirectory(prefix="benchmark-", dir=HERE)
    overrides = {
        "project_name": "four_support_benchmark",
        "output_root": work.name,
        "phase_field_epsilon": 0.5 * h_target,
        "remeshing": True,
        "remeshing_method": "marchingcubes",
        "reinit_sdf_method": "marchingcubes",
        "sdf_bc_fix_method": "zip",
        "remesh_octree_depth": args.optimization_depth,
        "remesh_init_octree_depth": 4,
        "remeshing_trimesh": {
            "mesh": mesh,
            "mesh_center": [0.0, 0.0, 0.0],
            "h_target": h_target,
            "clip_bbox": (tuple(bounds[0] - padding), tuple(bounds[1] + padding)),
        },
        "design_variable": "sdf",
        "bc_zip_patch_grow": 0,
        "output_per_optimize_step": 0,
        "max_optimize_iters": args.optimization_iters,
        "gd_lr": 8.0,
        "target_volume_fraction": 0.5,
        "shape_opt_w_vol": 5.0,
        "optimize_relative_tol": 1e-9,
        "dirichlet_bcs_dict": {LowestSelector("z", fixed_fraction): 0.0},
        "neumann_bcs_dict": {HighestSelector("z", 0.06): (0.0, 0.0, -0.02)},
    }
    output_root = Path(work.name)
    problem = ShapeOptTrimmed("config", override_params=overrides)
    problem.optimize()
    created = sorted(path for path in output_root.iterdir() if path.is_dir())
    if len(created) != 1:
        raise RuntimeError(f"expected one benchmark run directory, found {created}")
    per_iteration = iteration_times(created[0] / "log.txt")
    timed_iterations = per_iteration[1:]

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "octree_depth": args.depth,
        "octree_nodes": int(len(octree["nodes"])),
        "octree_elements": int(len(octree["elements"])),
        "octree_seconds": octree_seconds,
        "dense_octree_depth": args.depth + 1,
        "dense_octree_nodes": int(len(dense_octree["nodes"])),
        "dense_octree_elements": int(len(dense_octree["elements"])),
        "dense_octree_seconds": dense_seconds,
        "optimization_depth": args.optimization_depth,
        "optimization_iterations": args.optimization_iters,
        "optimization_iteration_seconds": per_iteration,
        "optimization_median_seconds_excluding_warmup": (
            statistics.median(timed_iterations) if timed_iterations else None
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    work.cleanup()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
