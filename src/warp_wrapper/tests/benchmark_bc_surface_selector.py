"""Stage benchmark for ShapeDiameterSegmentSelector on a production mesh."""

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh
import warp as wp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.analytical_sdf import build_warp_mesh_patch  # noqa: E402
from core.bc_surface_segmentation import (  # noqa: E402
    SurfaceSegmentation, _bilateral_smooth, _connected_segment_ids,
    _fill_missing, _gmm_1d, _regularize_labels, compute_shape_diameter,
)
from core.bc_surface_selectors import (  # noqa: E402
    HighestSelector, ShapeDiameterSegmentSelector, _build_adj_pairs,
    _compute_face_attributes,
)


def sync():
    wp.synchronize()
    torch.cuda.synchronize()


def timed(name, fn, repeats=3):
    values, result = [], None
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        values.append(1000.0 * (time.perf_counter() - start))
    timings[name] = {
        "median_ms": statistics.median(values), "samples_ms": values,
    }
    return result


parser = argparse.ArgumentParser()
parser.add_argument("mesh")
parser.add_argument("--output", default="bc_selector_benchmark.json")
args = parser.parse_args()
device = torch.device("cuda")
mesh = trimesh.load(args.mesh)
vertices = np.asarray(mesh.vertices) * np.array([1.0, -1.0, 1.0])
triangles = np.asarray(mesh.faces)[:, [0, 2, 1]]
verts = torch.tensor(np.ascontiguousarray(vertices), dtype=torch.float32, device=device)
faces = torch.tensor(np.ascontiguousarray(triangles), dtype=torch.long, device=device)
timings = {}

# Warm basic Torch/Warp paths before recording.
_compute_face_attributes(verts, faces)
sync()
centroids, normals, areas = timed("face_attributes", lambda: _compute_face_attributes(verts, faces))
adj = timed("face_adjacency", lambda: _build_adj_pairs(faces))


def orient():
    tri = verts[faces]
    volume6 = (tri[:, 0] * torch.linalg.cross(tri[:, 1], tri[:, 2])).sum()
    return ((faces[:, [0, 2, 1]].contiguous(), -normals)
            if bool(volume6 < 0) else (faces, normals))


oriented_faces, oriented_normals = timed("orientation", orient)
wp_mesh, keep_alive = timed(
    "warp_bvh_build", lambda: build_warp_mesh_patch(verts, oriented_faces, device="cuda")
)
_ = keep_alive
raw = timed("ray_cast_and_mad", lambda: compute_shape_diameter(
    verts, oriented_faces, centroids, oriented_normals, num_rays=25, wp_mesh=wp_mesh
), repeats=3)
repaired = timed("missing_value_repair", lambda: _fill_missing(raw, adj))
window = int(math.sqrt(len(faces) / 2000.0)) + 1
smooth = timed("bilateral_filter", lambda: _bilateral_smooth(repaired, adj, window), repeats=3)


def normalize():
    linear = (smooth - smooth.amin()) / (smooth.amax() - smooth.amin()).clamp_min(1e-8)
    logged = torch.log(5.0 * linear + 1.0) / math.log(6.0)
    return linear, logged


linear, logged = timed("normalization", normalize)
probabilities = timed("gmm", lambda: _gmm_1d(logged, 5), repeats=3)
cluster_ids = timed("alpha_expansion_graphcut", lambda: _regularize_labels(
    probabilities, verts, oriented_faces, adj, smoothing=0.26
), repeats=3)
segment_ids = timed("connected_segments", lambda: _connected_segment_ids(
    cluster_ids, adj, linear
), repeats=3)
segmentation = SurfaceSegmentation(
    linear, probabilities, cluster_ids, segment_ids, int(segment_ids.amax()) + 1
)
selector = ShapeDiameterSegmentSelector(
    HighestSelector("z", fraction=0.05), num_clusters=5, smoothing=0.26,
    num_rays=25, max_segment_area_fraction=0.12, adjacent_segments=1,
    adjacent_direction=(0.0, 0.0, -1.0),
)
key = (verts.data_ptr(), faces.data_ptr(), tuple(verts.shape), tuple(faces.shape),
       0, 5, 0.26, 25, 120.0, 0.12)
selector._CACHE[key] = segmentation
selected = timed("seed_and_spatial_selection", lambda: selector.select(
    verts, faces, centroids, normals, areas, None, adj_pairs=adj
), repeats=3)
h_min = 1.16 / 64.0
selected_faces = faces[selected].contiguous()


def patch_box():
    v = verts[selected_faces.reshape(-1)]
    return v.amin(0) - 0.75 * h_min, v.amax(0) + 0.75 * h_min


timed("patch_box", patch_box, repeats=3)
timed("selected_surface_bvh", lambda: build_warp_mesh_patch(
    verts, selected_faces, device="cuda"
), repeats=3)

total = sum(v["median_ms"] for v in timings.values())
for value in timings.values():
    value["percent_of_staged_total"] = 100.0 * value["median_ms"] / total
report = {
    "mesh_faces": len(faces), "mesh_vertices": len(verts), "num_rays": 25,
    "bilateral_window": window, "segments": segmentation.num_segments,
    "selected_faces": int(selected.sum()),
    "staged_total_ms": total, "timings": timings,
}
Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
