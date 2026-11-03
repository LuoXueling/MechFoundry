"""GPU prep for the automatic bunny BC workflow figure.

Consumes ``bunny.stl`` and ``bunny_sdf.vtk`` from
``../bunny/prep_bunny_sdf.py``.  Selection runs on exactly the same
Loop-subdivided, Taubin-smoothed surface as ``bcregion``.  Shape-diameter rays
therefore intersect that exact surface through Warp's mesh BVH; the coarser
voxel SDF has a different zero set and is not substituted for it.  Neither the
surface nor a distance field is rebuilt here.
"""

import os
import sys

import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "warp_wrapper"))

from core.bc_surface_selectors import (
    FloodFillSelector,
    GrowSelectionSelector,
    HalfSpaceSelector,
    HighestSelector,
    ShapeDiameterSegmentSelector,
    _bfs_expand,
    _build_adj_pairs,
    _cc_by_label_prop,
    _compute_face_attributes,
)
DEVICE = "cuda"
SEED_FRACTION = 0.05
NUM_CLUSTERS = int(os.environ.get("BC_NUM_CLUSTERS", 5))
NUM_RAYS = 25
SMOOTHING = float(os.environ.get("BC_SMOOTHING", 0.26))
OCTREE_DEPTH = 6
MIRROR = np.array([1.0, -1.0, 1.0])  # identical to bcregion


def read_structured_points(path):
    with open(path, "rb") as fh:
        dims = origin = spacing = None
        while True:
            line = fh.readline().decode("ascii").strip()
            if line.startswith("DIMENSIONS"):
                dims = tuple(map(int, line.split()[1:]))
            elif line.startswith("ORIGIN"):
                origin = np.asarray(line.split()[1:], np.float32)
            elif line.startswith("SPACING"):
                spacing = np.asarray(line.split()[1:], np.float32)
            elif line == "LOOKUP_TABLE default":
                break
        raw = np.frombuffer(fh.read(), dtype=">f4").astype(np.float32)
    assert dims and origin is not None and spacing is not None
    return raw.reshape(dims, order="F"), origin, spacing


def write_surface(path, verts, faces, fields):
    v = verts.detach().cpu().numpy().astype(np.float32)
    f = faces.detach().cpu().numpy().astype(np.int64)
    with open(path, "w") as out:
        out.write("# vtk DataFile Version 3.0\nautomatic BC surface\nASCII\nDATASET POLYDATA\n")
        out.write(f"POINTS {len(v)} float\n")
        np.savetxt(out, v, fmt="%.7g")
        out.write(f"POLYGONS {len(f)} {4 * len(f)}\n")
        np.savetxt(out, np.column_stack([np.full(len(f), 3), f]), fmt="%d")
        out.write(f"CELL_DATA {len(f)}\n")
        for name, data in fields.items():
            arr = data.detach().cpu().numpy().reshape(-1)
            out.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(out, arr, fmt="%.7g")


def write_stl(path, verts, faces, mask):
    v = verts.detach().cpu().numpy()
    f = faces[mask].detach().cpu().numpy()
    trimesh.Trimesh(v, f, process=False).export(path)


sdf_np, origin_np, spacing_np = read_structured_points("bunny_sdf.vtk")
side = float(spacing_np[0] * (sdf_np.shape[0] - 1))
bunny = trimesh.load("bunny.stl")
verts = torch.tensor(np.ascontiguousarray(bunny.vertices * MIRROR),
                     dtype=torch.float32, device=DEVICE)
# Reflection reverses orientation; this is exactly bcregion's winding repair.
faces = torch.tensor(np.ascontiguousarray(bunny.faces[:, [0, 2, 1]]),
                     dtype=torch.long, device=DEVICE)
centroids, normals, areas = _compute_face_attributes(verts, faces)
adj = _build_adj_pairs(faces)

seed_selector = HighestSelector("z", fraction=SEED_FRACTION)
seed_mask = seed_selector.select(
    verts, faces, centroids, normals, areas, None, adj_pairs=adj
)
selector = ShapeDiameterSegmentSelector(
    seed_selector, num_clusters=NUM_CLUSTERS, smoothing=SMOOTHING,
    num_rays=NUM_RAYS, max_segment_area_fraction=0.12,
    adjacent_segments=1, adjacent_direction=(0.0, 0.0, -1.0),
)
h_min = side / (2 ** OCTREE_DEPTH)
selector.resolve(verts, faces, eps_expansion=0.75 * h_min)
selected = selector._selected_face_mask
seg = selector._surface_segmentation
assert len(_cc_by_label_prop(seed_mask, adj)) == 2, "expected two ear-tip seeds"
assert int(selected.sum()) > int(seed_mask.sum()), "automatic segmentation did not grow seed"
if os.environ.get("BC_ALLOW_FAILED_MASK", "0") != "1":
    assert int(selected.sum()) < faces.shape[0] // 4, "ear seed leaked into the body segment"

# Reproduce bcregion's established flood-fill/growth selection for an honest
# after-the-fact comparison.  It is never passed to the automatic selector.
seed_comps = sorted(_cc_by_label_prop(seed_mask, adj), key=len, reverse=True)
tip_c = [centroids[c].mean(0) for c in seed_comps]
axis = tip_c[0] - tip_c[1]
axis = axis / axis.norm()
mid = 0.5 * (tip_c[0] + tip_c[1])
view_dir = torch.tensor([1.5, -3.0, 1.25], device=DEVICE)
near_first = float(tip_c[0] @ view_dir) > float(tip_c[1] @ view_dir)
front_half = HalfSpaceSelector(mid, axis if near_first else -axis)
back_half = HalfSpaceSelector(mid, -axis if near_first else axis)
front_allowed = front_half.select(
    verts, faces, centroids, normals, areas, None, adj_pairs=adj)
back_allowed = back_half.select(
    verts, faces, centroids, normals, areas, None, adj_pairs=adj)
front_reference = GrowSelectionSelector(
    FloodFillSelector(seed_selector & front_half, 4.6) & front_half,
    num_rings=14,
).select(verts, faces, centroids, normals, areas, None, adj_pairs=adj)
back_reference = GrowSelectionSelector(
    FloodFillSelector(seed_selector & back_half, 3.4) & back_half,
    num_rings=14,
).select(verts, faces, centroids, normals, areas, None, adj_pairs=adj)
reference = front_reference | back_reference
# Rendering-only fading collar, matching bc_region_id panel (c).  It does not
# change the selected BC region.
COLLAR_RINGS = 30
ring_id = torch.zeros(faces.shape[0], dtype=torch.long, device=DEVICE)
prev_collar = selected.clone()
for ring in range(1, COLLAR_RINGS + 1):
    cur_collar = _bfs_expand(selected, adj, ring)
    ring_id[cur_collar & ~prev_collar] = ring
    prev_collar = cur_collar
context_mask = prev_collar & ~selected
fade = (ring_id.float() - 1.0) / max(COLLAR_RINGS - 1, 1)
write_surface("auto_selected_context.vtk", verts, faces[context_mask], {
    "fade": fade[context_mask],
})
selected_vertices = verts[faces[selected].reshape(-1)]
selected_lo, selected_hi = selected_vertices.amin(0), selected_vertices.amax(0)
selected_center = 0.5 * (selected_lo + selected_hi)
selected_radius = 0.5 * torch.linalg.vector_norm(selected_hi - selected_lo)
intersection = (selected & reference).sum()
union = (selected | reference).sum()
iou = float(intersection / union)
recall = float(intersection / reference.sum())
precision = float(intersection / selected.sum())
auto_only = selected & ~reference
reference_only = reference & ~selected
# 0 body, 1 automatic only, 2 agreement, 3 previous only.  Keeping this as one
# cell field guarantees the comparison uses the exact same facets/camera.
comparison = (auto_only.to(torch.int32)
              + 2 * (selected & reference).to(torch.int32)
              + 3 * reference_only.to(torch.int32))
print(f"ear verification: IoU={iou:.3f}, recall={recall:.3f}, "
      f"reference={int(reference.sum())} facets")
if os.environ.get("BC_ALLOW_FAILED_MASK", "0") != "1":
    assert iou >= 0.65 and recall >= 0.75, "automatic mask does not cover both reference ears"

write_surface("auto_surface.vtk", verts, faces, {
    "shape_diameter": seg.shape_diameter,
    "segment_id": seg.segment_ids.float(),
    "segment_color": ((seg.segment_ids * 7) % 17).float() / 16.0,
    "seed": seed_mask.float(),
    "selected": selected.float(),
    "previous_selected": reference.float(),
    "comparison": comparison.float(),
})
write_stl("auto_seed.stl", verts, faces, seed_mask)
write_stl("auto_seed_rest.stl", verts, faces, ~seed_mask)
write_stl("auto_selected.stl", verts, faces, selected)
write_stl("auto_selected_rest.stl", verts, faces, ~selected)
write_stl("previous_selected.stl", verts, faces, reference)
write_stl("previous_selected_rest.stl", verts, faces, ~reference)

with open("auto_stats.txt", "w") as out:
    out.write(f"num_clusters {NUM_CLUSTERS}\n")
    out.write("adjacent_segment_depth 1\n")
    out.write(f"smoothing {SMOOTHING}\n")
    out.write(f"surface_faces {faces.shape[0]}\nseed_faces {int(seed_mask.sum())}\n")
    out.write(f"selected_faces {int(selected.sum())}\nsegments {seg.num_segments}\n")
    out.write(f"previous_faces {int(reference.sum())}\n")
    out.write(f"overlap_faces {int(intersection)}\n")
    out.write(f"auto_only_faces {int(auto_only.sum())}\n")
    out.write(f"previous_only_faces {int(reference_only.sum())}\n")
    out.write(f"iou {iou:.8f}\nrecall {recall:.8f}\nprecision {precision:.8f}\n")
    out.write("selected_center " + " ".join(map(str, selected_center.cpu().tolist())) + "\n")
    out.write(f"selected_radius {float(selected_radius):.9g}\n")
print(f"surface {faces.shape[0]} faces; {seg.num_segments} segments; "
      f"selected {int(selected.sum())}")
if os.environ.get("BC_SAVE_BASELINE"):
    np.savez_compressed(
        os.environ["BC_SAVE_BASELINE"],
        shape_diameter=seg.shape_diameter.cpu().numpy(),
        cluster_ids=seg.cluster_ids.cpu().numpy(),
        segment_ids=seg.segment_ids.cpu().numpy(),
        selected=selected.cpu().numpy(),
    )
