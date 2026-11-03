"""Turn PartField's per-face features and clusters into renderable fields.

THE RUN.  Both panels show the PartField run method_overview (Figure 1) shows:
``../overview/assets/seg/`` — ``pf_cache/features.npy``, one learned feature
vector per facet, and ``pf_k8.npz``, the cluster labels per facet at the
overview's k = 8 (the smallest k that gives each ear its own part), both
written by the production ``src/partfield_wrapper/segment.py`` in
``ov_assets.py segments`` and sha-checked here against ``assets/MANIFEST.json``.
PartField's inference is not seeded (§5b-ter), so a second run would move the
parts; reading the recorded run is what makes ``partfield_segmentation`` the
overview's segmentation panel rather than a look-alike.  Nothing here re-runs
the network or re-clusters.  The run was made on the overview's display bunny —
this figure's ``pf_surface`` to the facet (same vertices, same facet order; only
the winding differs, which labels do not see); asserted below.

Two derived fields are added, both render-only:

* ``feature_rgb`` — the features projected onto their three leading principal
  components and mapped to colour.  Shape diameter is one number per facet and
  a sequential colour map shows it directly; a PartField feature is 448 numbers,
  so the analogous picture is the projection that keeps the most variance.
* ``segment_rgb`` — the overview's pastel tint per PartField part, assigned by
  the overview's own ``ov_palette_check.segment_tints`` (touching parts get the
  most different pair), so each part has the colour it has in Figure 1.
"""

import os
import sys

import numpy as np
import torch
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "warp_wrapper"))
sys.path.insert(0, os.path.join(HERE, "..", "overview"))

from core.bc_surface_segmentation import _connected_segment_ids  # noqa: E402
from core.bc_surface_selectors import _build_adj_pairs  # noqa: E402

import ov_manifest  # noqa: E402
import ov_palette_check as G  # noqa: E402
import ov_style as S  # noqa: E402

DEVICE = "cuda"
K = int(os.environ.get("PF_CLUSTERS", 8))          # the overview's k
# Percentile clip for the PCA->RGB map.  A handful of outlying facets otherwise
# own both ends of every channel and the whole body washes out to mid grey.
CLIP_PCT = float(os.environ.get("PF_CLIP_PCT", 1.0))
# L* stays off both ends: nothing black enough to swallow the shading,
# nothing white enough to blow out under the key light.
L_RANGE = (52.0, 70.0)
CHROMA = 42.0  # half-width of the a*/b* spread


def write_surface(path, verts, faces, scalars=None, colors=None):
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    with open(path, "w") as out:
        out.write("# vtk DataFile Version 3.0\nPartField surface\nASCII\n"
                  "DATASET POLYDATA\n")
        out.write(f"POINTS {len(v)} float\n")
        np.savetxt(out, v, fmt="%.7g")
        out.write(f"POLYGONS {len(f)} {4 * len(f)}\n")
        np.savetxt(out, np.column_stack([np.full(len(f), 3), f]), fmt="%d")
        out.write(f"CELL_DATA {len(f)}\n")
        for name, data in (scalars or {}).items():
            out.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(out, np.asarray(data).reshape(-1), fmt="%.7g")
        for name, data in (colors or {}).items():
            # Legacy COLOR_SCALARS: the reader turns these into an unsigned-char
            # array, which is what ParaView needs to paint them directly
            # (disp.MapScalars = 0) instead of pushing them through a LUT.
            rgb = np.asarray(data, dtype=np.float32).reshape(-1, 3)
            out.write(f"COLOR_SCALARS {name} 3\n")
            np.savetxt(out, rgb, fmt="%.5f")


def _lab_to_srgb(lab):
    """CIELab (D65) -> sRGB, clamped.  Plain float maths, no colour package."""
    lightness, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fy = (lightness + 16.0) / 116.0
    f = np.stack([fy + a / 500.0, fy, fy - b / 200.0], axis=1)
    cube = f ** 3
    linear = np.where(cube > 0.008856, cube, (f - 16.0 / 116.0) / 7.787)
    xyz = linear * np.array([0.95047, 1.00000, 1.08883])
    matrix = np.array([[3.2406, -1.5372, -0.4986],
                       [-0.9689, 1.8758, 0.0415],
                       [0.0557, -0.2040, 1.0570]])
    rgb = xyz @ matrix.T
    rgb = np.clip(rgb, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb,
                    1.055 * rgb ** (1 / 2.4) - 0.055)


def pca_rgb(features):
    """Project to the 3 leading principal components and map to colour.

    The components go to CIELab rather than straight to RGB.  Raw R/G/B drives
    the dominant component's dark end to near-black, and a black body shows no
    shading at all -- the bunny stops reading as a surface.  Sending the leading
    component to a restricted L* instead, and the next two to a* and b*, keeps
    the same ordering while every facet stays light enough to be lit.
    """
    x = torch.as_tensor(np.ascontiguousarray(features), device=DEVICE,
                        dtype=torch.float32)
    # Clustering compares directions, not magnitudes (gpu_cluster normalises
    # before k-means), so project the unit features for the same reason.
    x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True).clamp_min(1e-12)
    x = x - x.mean(0, keepdim=True)
    # 161k x 448 -- the 448 x 448 covariance eigenproblem is the cheap way round.
    cov = (x.T @ x) / (x.shape[0] - 1)
    values, vectors = torch.linalg.eigh(cov)
    order = torch.argsort(values, descending=True)[:3]
    basis = vectors[:, order]
    # eigh's sign is arbitrary; pin it so reruns give the same picture.
    sign = torch.sign(basis[torch.argmax(basis.abs(), dim=0),
                            torch.arange(3, device=DEVICE)])
    basis = basis * torch.where(sign == 0, torch.ones_like(sign), sign)
    scores = x @ basis
    lo = torch.quantile(scores, CLIP_PCT / 100.0, dim=0)
    hi = torch.quantile(scores, 1.0 - CLIP_PCT / 100.0, dim=0)
    unit = ((scores - lo) / (hi - lo).clamp_min(1e-12)).clamp(0.0, 1.0)
    unit = unit.cpu().numpy().astype(np.float64)
    # The two leading components go to the CHROMATIC axes and the third to a
    # narrow L*.  Putting the dominant component on L* instead reads as a grey
    # ramp -- the bunny's head came out near-white with no hue at all -- because
    # a*, b* are both near zero wherever that component saturates.
    lab = np.stack([
        L_RANGE[0] + (L_RANGE[1] - L_RANGE[0]) * unit[:, 2],
        (unit[:, 0] - 0.5) * 2.0 * CHROMA,
        (unit[:, 1] - 0.5) * 2.0 * CHROMA,
    ], axis=1)
    explained = (values[order] / values.clamp_min(0).sum()).cpu().numpy()
    return _lab_to_srgb(lab).astype(np.float32), explained


# The welded twin of pf_surface.stl, in the same facet order (see pf_surface.py).
# STL itself is a soup, so a mesh read back from it has no facet adjacency.
topology = np.load("pf_topology.npz")
verts, faces = topology["vertices"], topology["faces"]
surface = trimesh.Trimesh(verts, faces, process=False)
# the run's surface is the overview's display bunny: the same facets, in order
display = trimesh.load(ov_manifest.asset("bunny_display.ply"), process=False)
assert len(display.faces) == len(faces), "the run was made on another surface"
drift = np.abs(display.triangles.mean(1) - surface.triangles.mean(1)).max()
assert drift < 1e-5, f"facet order differs from the run's surface ({drift:g})"
features = np.load(ov_manifest.asset("seg/pf_cache/features.npy"), mmap_mode="r")
clusters = np.load(ov_manifest.asset(f"seg/pf_k{K}.npz"))["segment_ids"].astype(np.int64)
assert features.shape[0] == len(faces), "one feature vector per facet"
assert clusters.shape[0] == len(faces), "one cluster label per facet"

rgb, explained = pca_rgb(features)

# the connected segments the clusters split into, for the stats only: the
# figure, like the overview, colours the PartField parts themselves
faces_t = torch.as_tensor(faces, dtype=torch.long, device=DEVICE)
adj = _build_adj_pairs(faces_t)
segments = _connected_segment_ids(torch.as_tensor(clusters, device=DEVICE), adj).cpu().numpy()
num_clusters = int(clusters.max()) + 1
num_segments = int(segments.max()) + 1

colour, touching, min_de = G.segment_tints(clusters, surface.face_adjacency, S.SEGMENT_TINTS)
tint_rgb = np.array([S._hex(c) for c in S.SEGMENT_TINTS], dtype=np.float32)
segment_rgb = tint_rgb[np.array([colour[int(c)] for c in clusters])]

areas = surface.area_faces
area_by_part = np.bincount(clusters, weights=areas, minlength=num_clusters) / areas.sum()
height = np.bincount(clusters, weights=areas * surface.triangles[:, :, 2].mean(1),
                     minlength=num_clusters) / np.maximum(
    np.bincount(clusters, weights=areas, minlength=num_clusters), 1e-30)

write_surface(
    "pf_surface.vtk", verts, faces,
    scalars={"cluster_id": clusters.astype(np.float32),
             "segment_id": segments.astype(np.float32)},
    colors={"feature_rgb": rgb, "segment_rgb": segment_rgb},
)

# for the gate: every part's tint, and which parts touch
with open("pf_palette.txt", "w") as out:
    for part in sorted(colour):
        out.write(f"part{part} {S.SEGMENT_TINTS[colour[part]]} {area_by_part[part]:.5f}\n")
    for a, b in touching:
        out.write(f"touch part{a} part{b}\n")

with open("pf_stats.txt", "w") as out:
    out.write("run ../overview/assets/seg (ov_assets.py segments, see assets/MANIFEST.json)\n")
    out.write(f"surface_faces {len(faces)}\n")
    out.write(f"feature_channels {features.shape[1]}\n")
    out.write(f"num_clusters {num_clusters}\n")
    out.write(f"num_segments {num_segments}\n")
    out.write("explained_variance " + " ".join(f"{v:.6f}" for v in explained) + "\n")
    out.write(f"explained_variance_total {float(explained.sum()):.6f}\n")
    out.write(f"clip_percentile {CLIP_PCT}\n")
    out.write(f"lab_lightness_range {L_RANGE[0]} {L_RANGE[1]}\n")
    out.write(f"lab_chroma {CHROMA}\n")
    by_height = np.argsort(-height)
    out.write("part_areas_top_down " + " ".join(f"{area_by_part[p]:.5f}" for p in by_height)
              + "\n")
    out.write("part_heights_top_down " + " ".join(f"{height[p]:.5f}" for p in by_height) + "\n")
    out.write("part_tints_top_down "
              + " ".join(S.SEGMENT_TINTS[colour[int(p)]] for p in by_height) + "\n")
    out.write(f"touching_pairs {len(touching)}\n")
    out.write(f"min_touching_dE {min_de:.1f}\n")
print(f"surface {len(faces)} faces; {features.shape[1]} feature channels; "
      f"{num_clusters} parts -> {num_segments} connected segments")
print("explained variance " + ", ".join(f"{v:.3f}" for v in explained)
      + f" (total {explained.sum():.3f})")
print("part areas " + ", ".join(f"{v:.4f}" for v in np.sort(area_by_part)[::-1]))
