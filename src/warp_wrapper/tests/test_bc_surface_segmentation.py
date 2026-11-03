"""Minimal correctness tests for GPU automatic BC surface segmentation."""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.bc_surface_segmentation import (  # noqa: E402
    SignedDistanceGrid,
    _alpha_move_terms,
    _alpha_expansion_labels,
    _bilateral_smooth,
    _cgal_dihedral_costs,
    _connected_segment_ids,
    _fill_missing,
    _gmm_1d,
    _robust_ray_mean,
    _regularize_labels,
    _warp_binary_min_cut,
)
from core.bc_surface_selectors import (  # noqa: E402
    HighestSelector,
    ShapeDiameterSegmentSelector,
    _build_adj_pairs,
)


def test_fill_missing_matches_single_pass_and_minimum_fallback():
    values = torch.tensor([1.0, float("nan"), float("nan"), 4.0])
    adjacency = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
    )
    got = _fill_missing(values, adjacency)
    assert torch.isfinite(got).all()
    assert got.tolist() == pytest.approx([1.0, 1.0, 4.0, 4.0])
    longer = torch.tensor([1.0, float("nan"), float("nan"), float("nan"), 4.0])
    line = torch.tensor([[i, i + 1] for i in range(4)]
                        + [[i + 1, i] for i in range(4)], dtype=torch.long)
    got = _fill_missing(longer, line)
    assert got.tolist() == pytest.approx([1.0, 1.0, 1.0, 4.0, 4.0])


def test_even_median_and_mad_match_cgal_convention():
    hits = torch.tensor([[1.0, 2.0, 4.0, 100.0], [-1.0, -1.0, -1.0, -1.0]])
    got = _robust_ray_mean(hits)
    # Median=3 and MAD=1.5, hence 1,2,4 remain and 100 is rejected.
    # CGAL uses -1 to mark a facet with no accepted ray for neighbor repair.
    assert got.tolist() == pytest.approx([7.0 / 3.0, -1.0])


def test_bilateral_uses_full_unique_graph_window_once():
    values = torch.tensor([0.0, 0.0, 1.0, 1.0])
    adjacency = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]], dtype=torch.long
    )
    one = _bilateral_smooth(values, adjacency, window=1)
    two = _bilateral_smooth(values, adjacency, window=2)
    assert torch.isfinite(one).all() and torch.isfinite(two).all()
    assert not torch.equal(one, two)


def test_gmm_separates_two_thickness_bands_deterministically():
    values = torch.cat([torch.linspace(0.02, 0.18, 80), torch.linspace(0.82, 0.98, 80)])
    p0 = _gmm_1d(values, 2)
    p1 = _gmm_1d(values, 2)
    assert torch.equal(p0, p1)
    labels = p0.argmax(1)
    assert int(labels[:80].mode().values) != int(labels[80:].mode().values)
    assert float(p0.sum(1).sub(1.0).abs().max()) < 1.0e-6


def test_connected_components_split_disconnected_equal_clusters():
    clusters = torch.tensor([0, 0, 1, 0, 0], dtype=torch.long)
    adjacency = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 4], [4, 3]],
        dtype=torch.long,
    )
    segments = _connected_segment_ids(clusters, adjacency)
    assert segments[0] == segments[1]
    assert segments[3] == segments[4]
    assert segments[0] != segments[3]
    assert segments.unique().numel() == 3


def test_connected_components_are_ordered_by_mean_sdf_like_cgal():
    clusters = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    adjacency = torch.tensor([[0, 1], [1, 0], [2, 3], [3, 2]], dtype=torch.long)
    sdf = torch.tensor([0.8, 1.0, 0.1, 0.3])
    segments = _connected_segment_ids(clusters, adjacency, sdf)
    assert segments.tolist() == [1, 1, 0, 0]


def test_cgal_oriented_dihedral_cost_is_symmetric_and_fold_sensitive():
    verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [1., -1., 0.], [1., -1., 1.]])
    pairs = torch.tensor([[0, 1], [1, 0]])
    flat_faces = torch.tensor([[0, 1, 2], [1, 0, 3]])
    fold_faces = torch.tensor([[0, 1, 2], [1, 0, 4]])
    flat = _cgal_dihedral_costs(verts, flat_faces, pairs, 0.25)
    fold = _cgal_dihedral_costs(verts, fold_faces, pairs, 0.25)
    assert flat[0] == pytest.approx(float(flat[1]))
    assert fold[0] == pytest.approx(float(fold[1]))
    assert float(flat[0]) > float(fold[0])


def test_alpha_move_binary_terms_equal_original_potts_energy():
    unary = torch.tensor([[0.2, 1.1, 0.7], [0.8, 0.1, 1.2],
                          [0.5, 0.9, 0.3]])
    labels = torch.tensor([0, 1, 2])
    eu = torch.tensor([0, 1])
    ev = torch.tensor([1, 2])
    ew = torch.tensor([0.6, 1.3])
    alpha = 2
    d0, d1, cu, cv, cw, constant = _alpha_move_terms(
        unary, labels, alpha, eu, ev, ew
    )
    for bits in range(8):
        x = torch.tensor([(bits >> i) & 1 for i in range(3)], dtype=torch.bool)
        moved = torch.where(x, torch.full_like(labels, alpha), labels)
        original = unary[torch.arange(3), moved].sum()
        original += (ew * (moved[eu] != moved[ev])).sum()
        binary = torch.where(x, d1, d0).sum() + constant
        binary += (cw * (x[cu] != x[cv])).sum()
        assert float(binary) == pytest.approx(float(original), abs=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by Warp max-flow")
def test_warp_binary_min_cut_matches_exhaustive_search():
    device = torch.device("cuda")
    d0 = torch.tensor([0.2, 1.3, 0.1, 0.8], device=device)
    d1 = torch.tensor([1.1, 0.1, 1.4, 0.2], device=device)
    eu = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
    ev = torch.tensor([1, 2, 3, 3], dtype=torch.long, device=device)
    ew = torch.tensor([0.7, 0.4, 0.9, 0.3], device=device)
    got = _warp_binary_min_cut(d0, d1, eu, ev, ew)
    got_energy = torch.where(got, d1, d0).sum() + (ew * (got[eu] != got[ev])).sum()
    energies = []
    for bits in range(16):
        x = torch.tensor([(bits >> i) & 1 for i in range(4)],
                         dtype=torch.bool, device=device)
        energies.append(torch.where(x, d1, d0).sum() + (ew * (x[eu] != x[ev])).sum())
    optimum = torch.stack(energies).amin()
    assert float(got_energy) == pytest.approx(float(optimum), abs=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by Warp max-flow")
def test_alpha_expansion_reaches_small_potts_optimum():
    device = torch.device("cuda")
    unary = torch.tensor([[0.1, 1.2, 0.8], [0.3, 0.4, 1.1],
                          [1.0, 0.5, 0.2], [0.9, 0.7, 0.1]], device=device)
    eu = torch.tensor([0, 1, 2], dtype=torch.long, device=device)
    ev = torch.tensor([1, 2, 3], dtype=torch.long, device=device)
    ew = torch.tensor([0.6, 0.8, 0.5], device=device)
    got = _alpha_expansion_labels(unary, unary.argmin(1), eu, ev, ew)
    got_energy = unary[torch.arange(4, device=device), got].sum()
    got_energy += (ew * (got[eu] != got[ev])).sum()
    energies = []
    for code in range(3 ** 4):
        q = code
        lab = []
        for _ in range(4):
            lab.append(q % 3); q //= 3
        lab = torch.tensor(lab, device=device)
        energies.append(unary[torch.arange(4, device=device), lab].sum()
                        + (ew * (lab[eu] != lab[ev])).sum())
    assert float(got_energy) == pytest.approx(float(torch.stack(energies).amin()), abs=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by Warp max-flow")
def test_cgal_graphcut_keeps_flat_strip_coherent():
    device = torch.device("cuda")
    probabilities = torch.tensor(
        [[0.99, 0.01], [0.90, 0.10], [0.10, 0.90], [0.01, 0.99]],
        dtype=torch.float32, device=device,
    )
    verts = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                          [1., 1., 0.], [2., 0., 0.], [2., 1., 0.]], device=device)
    faces = torch.tensor([[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3]], device=device)
    adjacency = torch.tensor(
        [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2]],
        dtype=torch.long, device=device,
    )
    labels = _regularize_labels(
        probabilities, verts, faces, adjacency, smoothing=0.25
    )
    assert len(set(labels.cpu().tolist())) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required by Warp segmentation")
def test_grid_backed_selector_on_two_radius_capsule():
    """The public selector reuses a signed-distance grid and retains its seed."""
    from core.analytical_sdf import extract_surface

    device = torch.device("cuda")
    n = 65
    q = torch.linspace(-1.0, 1.0, n, device=device)
    x, y, z = torch.meshgrid(q, q, q, indexing="ij")
    # Two overlapping spheres of different radii form a watertight thin/thick object.
    small = torch.sqrt((x + 0.42) ** 2 + y * y + z * z) - 0.30
    large = torch.sqrt((x - 0.28) ** 2 + y * y + z * z) - 0.52
    sdf = torch.minimum(small, large).contiguous()
    verts, faces, _ = extract_surface(sdf, domain_length=2.0, device="cuda")
    context = SignedDistanceGrid(
        sdf,
        torch.full((3,), -1.0, device=device),
        torch.full((3,), 2.0 / (n - 1), device=device),
    )
    selector = ShapeDiameterSegmentSelector(
        HighestSelector("x", fraction=0.12), num_clusters=3, num_rays=9
    )
    selector.resolve(
        verts, faces, eps_expansion=0.04, sdf_grid=context
    )
    assert selector._surface_segmentation is not None
    assert selector._surface_segmentation.shape_diameter.is_cuda
    assert selector._surface_segmentation.num_segments >= 2
    assert int(selector._selected_face_mask.sum()) > 0
    assert selector._wp_mesh is not None

    # Surface-only callers use the Warp BVH ray backend rather than rebuilding
    # a signed-distance volume.
    fallback = ShapeDiameterSegmentSelector(
        HighestSelector("x", fraction=0.12), num_clusters=2, num_rays=5
    )
    fallback.resolve(verts, faces, eps_expansion=0.04)
    assert torch.isfinite(fallback._surface_segmentation.shape_diameter).all()
    assert int(fallback._selected_face_mask.sum()) > 0


def test_spatial_adjacent_selector_options_are_validated():
    with pytest.raises(ValueError, match="adjacent_segments"):
        ShapeDiameterSegmentSelector(HighestSelector("z"), adjacent_segments=-1)
    with pytest.raises(ValueError, match="adjacent_direction"):
        ShapeDiameterSegmentSelector(
            HighestSelector("z"), adjacent_segments=1,
            adjacent_direction=(0.0, 0.0, 0.0),
        )
