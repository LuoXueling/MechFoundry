"""Unit tests for SurfaceSelector BC region selectors.

No GPU, no Warp, no FEM solve — pure Python/PyTorch on CPU.
Tests use simple synthetic meshes where the exact selection can be inferred:

  make_box_mesh()    [-1,1]^3 surface, 12 triangles (2 per face × 6 faces).
                     Face layout:  0-1=bottom(z=-1), 2-3=top(z=+1),
                                   4-5=front(y=-1), 6-7=back(y=+1),
                                   8-9=left(x=-1), 10-11=right(x=+1).
                     All normals are unit axis-aligned vectors.

  make_sphere_mesh() UV sphere r=1 centered at origin, ~400 triangles.
                     All face mean curvatures are in [1.99, 7.15].
                     All face Gaussian curvatures are > 1.0.

  make_row_mesh()    Three non-adjacent triangles at x≈1, 3, 10.
                     Mean centroid is at x≈4.67; distances from mean are
                     3.67 (x=1), 1.67 (x=3), 5.33 (x=10) — strictly ordered.

Run with:
    conda activate mechfoundry-warp117
    cd src/warp_wrapper
    python -m pytest tests/test_bc_surface_selectors.py -v
"""

import sys
import math
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.bc_surface_selectors import (
    AlignedNormalSelector,
    BoundaryEdgeSelector,
    BoundingBoxSelector,
    ConnectedComponentSelector,
    ComplementSelector,
    CurvatureSelector,
    CylinderSelector,
    DihedralAngleSelector,
    FlatRegionSelector,
    FloodFillSelector,
    FurthestFromCenterSelector,
    GrowSelectionSelector,
    HalfSpaceSelector,
    HighestSelector,
    IntersectSelector,
    LowestSelector,
    NormalSelector,
    PercentileSelector,
    SharpFeatureSelector,
    SlabSelector,
    SphereSelector,
    SubtractSelector,
    SurfaceSegmentSelector,
    UnionSelector,
    _compute_face_attributes,
    _build_adj_pairs,
)

DEVICE = "cpu"


# ===========================================================================
# Mesh factories
# ===========================================================================


def make_box_mesh():
    """[-1,1]^3 surface mesh: 8 vertices, 12 triangles.

    Face layout (2 triangles per face, 6 faces):
      0-1  Bottom (z=-1): normals (0,0,-1), centroids z=-1
      2-3  Top    (z=+1): normals (0,0,+1), centroids z=+1
      4-5  Front  (y=-1): normals (0,-1,0)
      6-7  Back   (y=+1): normals (0,+1,0)
      8-9  Left   (x=-1): normals (-1,0,0)
      10-11 Right (x=+1): normals (+1,0,0)

    Centroid x-y radial distances: top/bottom ≈ 0.471, sides ≈ 1.054.
    HalfSpace(z≥0) selects faces {2,3,5,7,8,11}.
    """
    verts = torch.tensor(
        [
            [-1, -1, -1],  # 0
            [+1, -1, -1],  # 1
            [+1, +1, -1],  # 2
            [-1, +1, -1],  # 3
            [-1, -1, +1],  # 4
            [+1, -1, +1],  # 5
            [+1, +1, +1],  # 6
            [-1, +1, +1],  # 7
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    faces = torch.tensor(
        [
            # Bottom (z=-1): indices 0-1
            [0, 2, 1],
            [0, 3, 2],
            # Top (z=+1): indices 2-3
            [4, 5, 6],
            [4, 6, 7],
            # Front (y=-1): indices 4-5
            [0, 1, 5],
            [0, 5, 4],
            # Back (y=+1): indices 6-7
            [2, 3, 7],
            [2, 7, 6],
            # Left (x=-1): indices 8-9
            [0, 4, 7],
            [0, 7, 3],
            # Right (x=+1): indices 10-11
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=torch.int32,
        device=DEVICE,
    )
    return verts, faces


def make_sphere_mesh(n_lat: int = 10, n_lon: int = 20):
    """UV sphere r=1, center (0,0,0).  n_lat*n_lon*2 triangles."""
    verts_list = []
    for i in range(n_lat + 1):
        lat = math.pi * (i / n_lat - 0.5)
        for j in range(n_lon):
            lon = 2.0 * math.pi * j / n_lon
            x = math.cos(lat) * math.cos(lon)
            y = math.cos(lat) * math.sin(lon)
            z = math.sin(lat)
            verts_list.append([x, y, z])
    verts = torch.tensor(verts_list, dtype=torch.float32, device=DEVICE)

    faces_list = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + (j + 1) % n_lon
            faces_list.append([a, b, d])
            faces_list.append([a, d, c])
    faces = torch.tensor(faces_list, dtype=torch.int32, device=DEVICE)
    return verts, faces


def make_row_mesh():
    """Three isolated triangles at x≈1, 3, 10 — no shared edges.

    Centroids: (1, 1/3, 0), (3, 1/3, 0), (10, 1/3, 0).
    Mean centroid ≈ (4.67, 1/3, 0); distances from mean:
      face 0 ≈ 3.67,  face 1 ≈ 1.67,  face 2 ≈ 5.33  (strictly ordered).
    FurthestFromCenterSelector(1/3) → k=1 → selects face 2 only.
    """
    verts = torch.tensor(
        [
            [0, 0, 0], [2, 0, 0], [1, 1, 0],   # tri 0: centroid (1, 1/3, 0)
            [2, 0, 0], [4, 0, 0], [3, 1, 0],   # tri 1: centroid (3, 1/3, 0)
            [9, 0, 0], [11, 0, 0], [10, 1, 0], # tri 2: centroid (10, 1/3, 0)
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    faces = torch.tensor(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
        dtype=torch.int32,
        device=DEVICE,
    )
    return verts, faces


# ===========================================================================
# Helper
# ===========================================================================


def _run(selector, verts, faces, eps=0.05):
    """Call select() then resolve(); return (face_mask, resolved_selector)."""
    centroids, normals, areas = _compute_face_attributes(verts, faces)
    from core.bc_surface_selectors import _compute_face_curvatures
    curvatures = _compute_face_curvatures(verts, faces)
    face_mask = selector.select(verts, faces, centroids, normals, areas, curvatures)
    selector.resolve(verts, faces, eps_expansion=eps)
    return face_mask, selector


def _centroids_of(verts, faces, mask):
    fi = faces.long()
    c = (verts[fi[:, 0]] + verts[fi[:, 1]] + verts[fi[:, 2]]) / 3.0
    return c[mask]


def test_surface_segment_selector_reuses_reviewed_labels():
    verts, faces = make_box_mesh()
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
    mask, _ = _run(SurfaceSegmentSelector([1, 4], labels), verts, faces)
    assert torch.nonzero(mask, as_tuple=False).reshape(-1).tolist() == [2, 3, 8, 9]

    with pytest.raises(ValueError, match="unknown segment ids"):
        _run(SurfaceSegmentSelector([9], labels), verts, faces)


# ===========================================================================
# Geometric selector tests
# ===========================================================================


def test_bounding_box():
    """BoundingBoxSelector selects faces whose centroid is inside the box.

    On the box mesh, only face 7 (back face, centroid=(0.333,1.0,0.333)) has
    all centroid coordinates in (0, 1.1] — selecting exactly 1 face.
    """
    verts, faces = make_box_mesh()
    sel = BoundingBoxSelector((0.0, 0.0, 0.0), (1.1, 1.1, 1.1))
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 1, f"Expected 1 face in +octant, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c > 0.0).all(), "Selected centroid must be strictly in positive octant"


def test_slab():
    """SlabSelector(z, -1.1, -0.5) selects only the 2 bottom-face triangles."""
    verts, faces = make_box_mesh()
    sel = SlabSelector("z", -1.1, -0.5)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Bottom face = 2 triangles, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] < -0.5).all()


def test_sphere_selector():
    """SphereSelector at (0,0,1.5) radius 0.8 selects both top-face triangles.

    Top-face centroids are at (±1/3, ∓1/3, 1); distance to (0,0,1.5) ≈ 0.60 < 0.8.
    All other face centroids are > 0.8 away from (0,0,1.5).
    """
    verts, faces = make_box_mesh()
    sel = SphereSelector((0.0, 0.0, 1.5), 0.8)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Expected 2 top faces, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] > 0.0).all(), "Selected faces should be on the top (z=+1) face"


def test_cylinder():
    """CylinderSelector along z-axis with radius 0.5 selects top+bottom (4 faces).

    Top/bottom face centroids have x-y radial distance ≈ 0.471 < 0.5.
    Side face centroids have x-y radial distance ≈ 1.054 > 0.5.
    """
    verts, faces = make_box_mesh()
    sel = CylinderSelector((0.0, 0.0, -2.0), (0.0, 0.0, 2.0), radius=0.5)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 4, f"Expected 4 top+bottom faces, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    radial = c[:, :2].norm(dim=1)
    assert (radial < 0.5).all(), "All selected centroids must be within r=0.5 of z-axis"


def test_half_space():
    """HalfSpaceSelector(z≥0) selects 6 of 12 box-mesh faces.

    Faces with z-centroid > 0: top (2), and one triangle each from
    front, back, left, right (4) — total 6.  The other 6 have z < 0.
    """
    verts, faces = make_box_mesh()
    sel = HalfSpaceSelector((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 6, f"Expected 6 faces with z centroid ≥ 0, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] >= -1e-6).all()


def test_lowest_selector():
    """LowestSelector('z', 0.1) selects only the 2 bottom faces.

    Threshold = -1 + 0.1*(2) = -0.8; bottom centroids at z=-1 qualify,
    all other centroids are at z ≥ -1/3.
    """
    verts, faces = make_box_mesh()
    sel = LowestSelector("z", fraction=0.1)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Expected 2 bottom faces, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] < 0.0).all()


def test_highest_selector():
    """HighestSelector('z', 0.1) selects only the 2 top faces.

    Threshold = +1 - 0.1*(2) = +0.8; top centroids at z=+1 qualify,
    all other centroids are at z ≤ +1/3.
    """
    verts, faces = make_box_mesh()
    sel = HighestSelector("z", fraction=0.1)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Expected 2 top faces, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] > 0.0).all()


def test_percentile_selector():
    """PercentileSelector('z', 0, 10) selects only the 2 bottom faces.

    Box z-centroids: two at z=-1, four at z=-1/3, four at z=+1/3, two at z=+1.
    The 10th percentile falls at z≈-0.933 (between the two z=-1 entries and the
    z=-1/3 entries), so only the 2 bottom faces satisfy z ∈ [-1, -0.933].
    """
    verts, faces = make_box_mesh()
    sel = PercentileSelector("z", lo_pct=0.0, hi_pct=10.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Expected 2 faces in bottom 10th pct, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] < 0.0).all()


def test_furthest_from_center():
    """FurthestFromCenterSelector(1/3) selects the single farthest triangle.

    Row mesh has 3 faces with centroids at x≈1, 3, 10. Mean centroid is at x≈4.67.
    Distances from mean: x=1→3.67, x=3→1.67, x=10→5.33 (strictly ordered).
    k = max(1, round(1/3 * 3)) = 1 → selects only face 2 (farthest at x≈10).
    """
    verts, faces = make_row_mesh()
    sel = FurthestFromCenterSelector(fraction=1 / 3)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 1, f"Expected 1 face (farthest tri), got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert float(c[0, 0].item()) > 9.0, "Selected face must be the x≈10 triangle"


# ===========================================================================
# Normal & Curvature selector tests
# ===========================================================================


def test_normal_selector():
    """NormalSelector is unsigned: both +z and -z normals match target (0,0,1).

    Top (normal +z) and bottom (normal -z) → |dot|=1.0 ≥ cos(20°) ≈ 0.94.
    Side normals are ±x or ±y → |dot|=0 < threshold → excluded.
    Total: 2 (top) + 2 (bottom) = 4 faces.
    """
    verts, faces = make_box_mesh()
    sel = NormalSelector((0.0, 0.0, 1.0), max_angle_deg=20.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 4, f"Expected 4 (top+bottom), got {int(mask.sum())}"


def test_aligned_normal_oriented():
    """AlignedNormalSelector(+z, oriented=True) selects only the 2 top faces."""
    verts, faces = make_box_mesh()
    sel = AlignedNormalSelector((0.0, 0.0, 1.0), max_angle_deg=20.0, oriented=True)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Only top 2 faces have +z normals, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] > 0.0).all()


def test_aligned_normal_not_oriented():
    """AlignedNormalSelector(+z, oriented=False) behaves like NormalSelector → 4 faces."""
    verts, faces = make_box_mesh()
    sel = AlignedNormalSelector((0.0, 0.0, 1.0), max_angle_deg=20.0, oriented=False)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 4, f"Expected 4 (top+bottom), got {int(mask.sum())}"


def test_flat_region():
    """FlatRegionSelector selects all 12 box faces — the box is piecewise flat.

    The cotangent mean curvature of every box vertex is 0, so all face curvatures
    are 0, satisfying κ ≤ 5.0.
    """
    verts, faces = make_box_mesh()
    sel = FlatRegionSelector(max_curvature=5.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 12, f"All box faces are flat, got {int(mask.sum())}"


def test_sharp_feature():
    """SharpFeatureSelector(1.0) selects all sphere faces.

    For a unit sphere, mean curvature ≈ 1.0 everywhere (analytically).
    The discrete cotangent approximation gives values in [1.99, 7.15] for
    sphere(10, 20), all well above the threshold 1.0 → all 400 faces selected.
    """
    verts, faces = make_sphere_mesh()
    F = int(faces.shape[0])
    sel = SharpFeatureSelector(min_curvature=1.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == F, f"Expected all {F} sphere faces to be sharp, got {int(mask.sum())}"


def test_curvature_selector_gaussian():
    """CurvatureSelector(0.5, 500, 'gaussian') selects all sphere faces.

    The Gaussian curvature of a unit sphere is exactly 1/R²=1 everywhere.
    The discrete angle-defect approximation gives values > 1.0 for all faces
    in sphere(10, 20) — all fall in (0.5, 500) → all 400 faces selected.
    """
    verts, faces = make_sphere_mesh()
    F = int(faces.shape[0])
    sel = CurvatureSelector(0.5, 500.0, curvature_type="gaussian")
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == F, f"Expected all {F} sphere faces, got {int(mask.sum())}"


def test_dihedral_angle():
    """DihedralAngleSelector(55, 65) selects all 12 box faces.

    Each box triangle has one co-planar neighbor (dihedral 0°) and two perpendicular
    neighbors (dihedral 90°).  Mean = (0+90+90)/3 = 60° — within [55, 65] for all 12.
    """
    verts, faces = make_box_mesh()
    sel = DihedralAngleSelector(min_angle_deg=55.0, max_angle_deg=65.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 12, f"All 12 box triangles have mean dihedral ~60°, got {int(mask.sum())}"


# ===========================================================================
# Topological selector tests
# ===========================================================================


def test_flood_fill_point_stops_at_crease():
    """FloodFill from top face with 20° threshold stays on top (2 faces).

    Adjacent faces meet at 90° dihedral; pairwise criterion dot(n_i, n_j) = 0
    which is below cos(20°)≈0.94 → BFS cannot cross the box crease.
    """
    verts, faces = make_box_mesh()
    sel = FloodFillSelector(seed=(0.0, 0.0, 1.1), max_feature_angle_deg=20.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Should select only 2 top triangles, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 2] > 0.0).all()


def test_flood_fill_soft_crosses_crease():
    """FloodFill with 91° threshold traverses all box edges and reaches all 12 faces.

    Box faces meet at 90°; cos(90°)=0 ≥ cos(91°)≈-0.017 → criterion satisfied
    for every shared edge.  The box surface is connected, so all 12 are reached.
    """
    verts, faces = make_box_mesh()
    sel = FloodFillSelector(seed=(0.0, 0.0, 1.1), max_feature_angle_deg=91.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 12, f"With 91° threshold all 12 faces should be selected, got {int(mask.sum())}"


def test_flood_fill_selector_seed():
    """FloodFill with SurfaceSelector seed uses multi-source BFS from all seed faces.

    HighestSelector('z', 0.1) selects the 2 top triangles as the seed.
    With max_feature_angle=20° the BFS cannot cross the 90° box creases, so
    exactly those 2 faces are selected.
    """
    verts, faces = make_box_mesh()
    seed_sel = HighestSelector("z", fraction=0.1)
    sel = FloodFillSelector(seed=seed_sel, max_feature_angle_deg=20.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Expected 2 top triangles, got {int(mask.sum())}"


def test_flood_fill_int_seed():
    """FloodFill with integer face index 2 (a top face) and 20° threshold → 2 faces."""
    verts, faces = make_box_mesh()
    sel = FloodFillSelector(seed=2, max_feature_angle_deg=20.0)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Expected 2 top triangles, got {int(mask.sum())}"


def test_grow_selection():
    """GrowSelectionSelector expands top+bottom (4 faces) by 1 ring → all 12 faces.

    NormalSelector(z, 10°) [unsigned] selects top (2) and bottom (2) = 4 faces.
    Each top face is adjacent to 2 unique side faces; each bottom face to 2 others.
    Growing by 1 ring adds all 8 remaining side faces → 4 + 8 = 12.
    """
    verts, faces = make_box_mesh()
    sel = GrowSelectionSelector(NormalSelector((0.0, 0.0, 1.0), max_angle_deg=10.0), num_rings=1)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 12, f"Expected all 12 faces after 1-ring grow, got {int(mask.sum())}"


def test_boundary_edge_selector():
    """BoundaryEdgeSelector returns all 4 selected faces as boundary faces.

    NormalSelector(z, 10°) selects top (2) and bottom (2) = 4 faces.
    Every selected face has 2 non-selected side neighbors, so each is a boundary
    face → all 4 are returned.
    """
    verts, faces = make_box_mesh()
    sel = BoundaryEdgeSelector(NormalSelector((0.0, 0.0, 1.0), max_angle_deg=10.0))
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 4, f"All 4 top+bottom faces are boundary faces, got {int(mask.sum())}"


def test_connected_component_selector_whole():
    """ConnectedComponentSelector(None, 0, 'size') returns all 12 box faces (one CC)."""
    verts, faces = make_box_mesh()
    sel = ConnectedComponentSelector(base_selector=None, component_index=0, sort_by="size")
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 12, f"One CC = all 12 triangles, got {int(mask.sum())}"


def test_connected_component_by_area():
    """ConnectedComponentSelector sort_by='area' — same result on a single-CC mesh."""
    verts, faces = make_box_mesh()
    sel = ConnectedComponentSelector(base_selector=None, component_index=0, sort_by="area")
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 12


# ===========================================================================
# Boolean composite tests
# ===========================================================================


def test_intersect():
    """A & B selects faces in both A and B.

    NormalSelector(z, 20°) selects top+bottom (4 faces).
    HighestSelector(z, 0.4) selects top 40% of z range → only top 2 faces (z=+1).
    Intersection: 2 faces.
    """
    verts, faces = make_box_mesh()
    sel = NormalSelector((0.0, 0.0, 1.0), max_angle_deg=20.0) & HighestSelector("z", fraction=0.4)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Intersection: top face only (2 tris), got {int(mask.sum())}"


def test_union():
    """A | B selects faces in A or B.

    LowestSelector(z, 0.1) → 2 bottom faces.  HighestSelector(z, 0.1) → 2 top faces.
    Union: 4 faces total (no overlap).
    """
    verts, faces = make_box_mesh()
    sel = LowestSelector("z", 0.1) | HighestSelector("z", 0.1)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 4, f"Union: top(2)+bottom(2)=4, got {int(mask.sum())}"


def test_complement():
    """~A selects all faces NOT selected by A.

    AlignedNormalSelector(-z, oriented) selects 2 bottom faces.
    Complement: 12 - 2 = 10 faces.
    """
    verts, faces = make_box_mesh()
    bottom_sel = AlignedNormalSelector((0.0, 0.0, -1.0), max_angle_deg=20.0, oriented=True)
    sel = ComplementSelector(bottom_sel)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 10, f"All except bottom 2 = 10, got {int(mask.sum())}"


def test_subtract():
    """A - B selects faces in A but not in B.

    LowestSelector(z, 0.1) → same 2 bottom faces as AlignedNormal(-z, 20°).
    Subtract: 2 - 2 = 0 faces.
    """
    verts, faces = make_box_mesh()
    bottom = AlignedNormalSelector((0.0, 0.0, -1.0), max_angle_deg=20.0, oriented=True)
    also_bottom = LowestSelector("z", 0.1)
    sel = SubtractSelector(also_bottom, bottom)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 0, f"Bottom minus bottom-normal should be empty, got {int(mask.sum())}"


def test_operator_syntax():
    """Python operator shortcuts &, |, -, ~ return the correct composite selector types."""
    verts, faces = make_box_mesh()
    a = NormalSelector((0.0, 0.0, 1.0), 20.0)
    b = HighestSelector("z", 0.4)

    sel_and = a & b
    sel_or = a | b
    sel_sub = a - b
    sel_inv = ~a

    assert isinstance(sel_and, IntersectSelector)
    assert isinstance(sel_or, UnionSelector)
    assert isinstance(sel_sub, SubtractSelector)
    assert isinstance(sel_inv, ComplementSelector)

    mask_and, _ = _run(sel_and, verts, faces)
    mask_or, _ = _run(sel_or, verts, faces)
    # Intersection ⊆ Union
    assert int(mask_and.sum()) <= int(mask_or.sum())


# ===========================================================================
# Resolution and __call__ tests
# ===========================================================================


def test_resolve_patch_box():
    """resolve() stores the selected patch's box, expanded by eps_expansion."""
    verts, faces = make_box_mesh()
    eps = 0.15
    sel = LowestSelector("z", fraction=0.4)
    sel.resolve(verts, faces, eps_expansion=eps)
    assert sel.is_resolved
    sel_verts = verts[faces[sel._selected_face_mask].reshape(-1).long()]
    assert torch.allclose(sel._patch_lo, sel_verts.amin(0) - eps)
    assert torch.allclose(sel._patch_hi, sel_verts.amax(0) + eps)
    # Bottom face vertices are at z=-1
    assert float(sel._patch_lo[2]) <= -1.0 - eps + 1e-6


def test_call_after_resolve():
    """__call__ returns True for a point on/near the selected surface.

    Bottom face is at z=-1 (x,y ∈ [-1,1]²).  Point (0,0,-0.95) is 0.05 above
    the face, within eps=0.1 — accepted by both bbox and BVH stages.
    """
    verts, faces = make_box_mesh()
    sel = LowestSelector("z", fraction=0.1)
    sel.resolve(verts, faces, eps_expansion=0.1)
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])
    z = torch.tensor([-0.95])
    result = sel(x, y, z)
    assert result.shape == (1,)
    assert bool(result[0].item()), "Point 0.05 above bottom face should be inside BC region"


def test_call_unresolved_raises():
    """Calling an unresolved selector raises RuntimeError."""
    sel = LowestSelector("z", fraction=0.1)
    with pytest.raises(RuntimeError, match="not resolved"):
        sel(torch.zeros(1), torch.zeros(1), torch.zeros(1))


def test_separate_patches_share_one_selector():
    """Top + bottom faces: points near either are accepted, the gap is not.

    The culling box spans both patches; the BVH distance test decides.
    """
    verts, faces = make_box_mesh()
    sel = UnionSelector(LowestSelector("z", 0.1), HighestSelector("z", 0.1))
    sel.resolve(verts, faces, eps_expansion=0.05)
    x = torch.zeros(3)
    y = torch.zeros(3)
    z = torch.tensor([-0.97, 0.0, 0.97])
    assert sel(x, y, z).tolist() == [True, False, True]


def test_cull_matches_bvh_on_every_point():
    """Culling by the patch box must not change the accepted set."""
    from core.analytical_sdf import compute_unsigned_dist

    verts, faces = make_sphere_mesh(n_lat=16, n_lon=32)
    eps = 0.05
    sel = HighestSelector("z", fraction=0.5)
    sel.resolve(verts, faces, eps_expansion=eps)

    g = torch.linspace(-1.3, 1.3, 24, device=DEVICE)
    X, Y, Z = torch.meshgrid(g, g, g, indexing="ij")
    x, y, z = X.reshape(-1), Y.reshape(-1), Z.reshape(-1)
    pts = torch.stack([x, y, z], dim=1)
    d = compute_unsigned_dist(sel._wp_mesh, pts, max_dist=2 * eps, device=DEVICE)
    assert torch.equal(sel(x, y, z), d <= eps)
    assert bool(sel(x, y, z).any())


def test_far_points_skip_the_bvh(monkeypatch):
    """Only points inside the patch box are sent to the BVH query."""
    import core.analytical_sdf as asdf

    verts, faces = make_box_mesh()
    sel = LowestSelector("z", fraction=0.1)
    sel.resolve(verts, faces, eps_expansion=0.1)
    seen = []
    real = asdf.compute_unsigned_dist

    def spy(wp_mesh, query_pts, max_dist, device="cuda"):
        seen.append(int(query_pts.shape[0]))
        return real(wp_mesh, query_pts, max_dist, device=device)

    monkeypatch.setattr(asdf, "compute_unsigned_dist", spy)
    x = torch.tensor([0.0, 0.0, 5.0])
    y = torch.tensor([0.0, 0.0, 0.0])
    z = torch.tensor([-0.95, 0.9, -1.0])     # near, same column but far, far off
    assert sel(x, y, z).tolist() == [True, False, False]
    assert seen == [1], f"expected only the near point at the BVH, got {seen}"


# ===========================================================================
# Repr and is_resolved
# ===========================================================================


def test_repr_unresolved():
    sel = NormalSelector((0.0, 0.0, 1.0), 30.0)
    assert "unresolved" in repr(sel)


def test_repr_resolved():
    verts, faces = make_box_mesh()
    sel = NormalSelector((0.0, 0.0, 1.0), 30.0)
    sel.resolve(verts, faces, eps_expansion=0.05)
    assert "resolved" in repr(sel)


def test_is_resolved_false_before():
    assert not LowestSelector("z").is_resolved


def test_is_resolved_true_after():
    verts, faces = make_box_mesh()
    sel = LowestSelector("z")
    sel.resolve(verts, faces, eps_expansion=0.05)
    assert sel.is_resolved


def test_zero_selection_warns():
    """Selecting zero faces emits a warning and resolves to an empty region."""
    verts, faces = make_box_mesh()
    sel = SphereSelector((10.0, 10.0, 10.0), radius=0.01)
    with pytest.warns(UserWarning):
        sel.resolve(verts, faces, eps_expansion=0.05)
    assert sel.is_resolved
    assert sel._patch_lo is None and sel._wp_mesh is None
    result = sel(torch.zeros(5), torch.zeros(5), torch.zeros(5))
    assert not result.any()


# ===========================================================================
# Axis variants
# ===========================================================================


def test_lowest_x_axis():
    """LowestSelector('x', 0.1) selects the 2 left-face triangles (x=-1)."""
    verts, faces = make_box_mesh()
    sel = LowestSelector("x", fraction=0.1)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Left face = 2 triangles, got {int(mask.sum())}"
    c = _centroids_of(verts, faces, mask)
    assert (c[:, 0] < 0.0).all()


def test_slab_y_axis():
    """SlabSelector(y, 0.5, 1.1) selects the 2 back-face triangles (y=+1)."""
    verts, faces = make_box_mesh()
    sel = SlabSelector(1, 0.5, 1.1)
    mask, _ = _run(sel, verts, faces)
    assert int(mask.sum()) == 2, f"Back face = 2 triangles, got {int(mask.sum())}"
