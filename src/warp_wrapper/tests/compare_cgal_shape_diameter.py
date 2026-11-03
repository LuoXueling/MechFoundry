"""Compare GPU and CGAL normalized shape diameter on one indexed mesh.

This diagnostic deliberately stops before GMM/graph cuts, isolating the ray,
MAD, missing-value repair, and bilateral-filtering stages.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bc_surface_segmentation import (
    _bilateral_smooth,
    _fill_missing,
    compute_shape_diameter,
)
from core.bc_surface_selectors import (
    _build_adj_pairs,
    _compute_face_attributes,
)


def write_off(path, vertices, faces):
    with open(path, "w") as out:
        out.write(f"OFF\n{len(vertices)} {len(faces)} 0\n")
        np.savetxt(out, vertices, fmt="%.17g")
        np.savetxt(out, np.column_stack((np.full(len(faces), 3), faces)), fmt="%d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh")
    parser.add_argument("oracle")
    parser.add_argument("--mirror-y", action="store_true")
    args = parser.parse_args()

    # Match the production bunny loader: STL needs vertex merging so adjacent
    # triangles share indices and both implementations see the same graph.
    mesh = trimesh.load(args.mesh)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if args.mirror_y:
        vertices = vertices * np.array([1.0, -1.0, 1.0])
        faces = faces[:, [0, 2, 1]]

    device = torch.device("cuda")
    verts_t = torch.as_tensor(np.ascontiguousarray(vertices), dtype=torch.float32,
                              device=device)
    faces_t = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.long,
                              device=device)
    centroids, normals, _ = _compute_face_attributes(verts_t, faces_t)
    adjacency = _build_adj_pairs(faces_t)
    raw = compute_shape_diameter(verts_t, faces_t, centroids, normals)
    repaired = _fill_missing(raw, adjacency)
    window = int(math.sqrt(len(faces) / 2000.0)) + 1
    smooth = _bilateral_smooth(repaired, adjacency, window)
    gpu = ((smooth - smooth.amin()) /
           (smooth.amax() - smooth.amin()).clamp_min(1.0e-8)).cpu().numpy()

    with tempfile.TemporaryDirectory() as tmp:
        off = os.path.join(tmp, "shared.off")
        write_off(off, vertices, faces)
        text = subprocess.check_output([args.oracle, off, "6", "0.26"], text=True)
    rows = [line.split() for line in text.splitlines() if line[:1].isdigit()]
    cgal = np.asarray([float(row[1]) for row in rows])
    cgal_raw = np.asarray([float(row[3]) for row in rows])
    if len(cgal) != len(gpu):
        raise RuntimeError(f"face count changed: GPU {len(gpu)}, CGAL {len(cgal)}")
    error = np.abs(gpu - cgal)
    print(f"faces {len(gpu)} window {window}")
    print(f"correlation {np.corrcoef(gpu, cgal)[0, 1]:.9f}")
    print(f"mae {error.mean():.9f} rmse {np.sqrt(np.mean(error**2)):.9f} "
          f"max {error.max():.9f}")
    for name, values in (("gpu", gpu), ("cgal", cgal), ("abs_error", error)):
        print(name, *[f"{q:.7f}" for q in np.quantile(values, [0, .01, .1, .5, .9, .99, 1])])
    valid = cgal_raw >= 0.0
    raw_np = raw.cpu().numpy()
    raw_error = np.abs(raw_np[valid] - cgal_raw[valid])
    print(f"raw_valid_cgal {valid.sum()} raw_valid_gpu {(raw_np >= 0).sum()}")
    print(f"raw_correlation {np.corrcoef(raw_np[valid], cgal_raw[valid])[0, 1]:.9f} "
          f"raw_mae {raw_error.mean():.9f} raw_rmse {np.sqrt(np.mean(raw_error**2)):.9f}")


if __name__ == "__main__":
    main()
