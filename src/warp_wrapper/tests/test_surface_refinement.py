"""Tests for explicit-surface-driven octree refinement."""

import sys
from pathlib import Path

import pytest
import torch


WRAPPER_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = WRAPPER_ROOT.parent.parent
for _path in (str(WRAPPER_ROOT), str(REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.remeshing import (  # noqa: E402
    _surface_vertices_to_must_query_positions,
    resolve_explicit_surface_sdf_truncation,
)
from src.triposg_wrapper.hierarchical_extraction import (  # noqa: E402
    hierarchical_extraction,
)


def _legacy_surface_chain(vertices, init_depth, final_depth):
    """Pre-optimization implementation, retained as an equivalence oracle."""
    unit = ((vertices.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)
    positions = []
    for depth in range(init_depth + 1, final_depth + 1):
        n_cells = 2**depth
        cells = torch.floor(unit * n_cells).long().clamp(0, n_cells - 1)
        parents = torch.unique(cells // 2, dim=0)
        positions.append(-1.0 + (parents * 2 + 1).double() * (2.0 / n_cells))
    return torch.cat(positions, dim=0)


@pytest.mark.parametrize("h_target", [0.03, 30.0])
def test_explicit_surface_band_defaults_to_leaf_width(h_target):
    truncation = resolve_explicit_surface_sdf_truncation(h_target)
    assert truncation == pytest.approx(h_target)
    assert 0.9 * truncation / h_target == pytest.approx(0.9)


def test_explicit_surface_band_accepts_physical_override():
    assert resolve_explicit_surface_sdf_truncation(0.03, 0.012) == pytest.approx(0.012)
    with pytest.raises(ValueError, match="positive physical length"):
        resolve_explicit_surface_sdf_truncation(0.03, 0.0)


def test_surface_positions_are_unique_final_parent_centres():
    vertices = torch.tensor(
        [
            [-0.73, 0.12, 0.44],
            [-0.72, 0.11, 0.43],  # shares coarse parents with the first
            [0.99, -1.0, 1.0],  # domain-boundary clamping
            [float("nan"), 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    positions, counts = _surface_vertices_to_must_query_positions(
        vertices, init_octree_depth=2, octree_depth=5
    )

    assert counts == [(5, positions.shape[0])]
    assert torch.isfinite(positions).all()

    finite_vertices = vertices[torch.isfinite(vertices).all(dim=-1)]
    unit = ((finite_vertices + 1.0) / 2.0).clamp(0.0, 1.0)
    n_cells = 2**5
    idx = ((positions + 1.0) * n_cells / 2.0).round().long()
    assert bool((idx.remainder(2) == 1).all())

    cells = torch.floor(unit * n_cells).long().clamp(0, n_cells - 1)
    expected_parents = torch.unique(cells // 2, dim=0)
    actual_parents = torch.unique((idx - 1) // 2, dim=0)
    assert torch.equal(actual_parents, expected_parents)
    assert positions.shape[0] == expected_parents.shape[0]


def test_compact_positions_match_legacy_refinement_chain():
    init_depth = 2
    final_depth = 5
    vertices = torch.tensor(
        [
            [-0.73, 0.12, 0.44],
            [-0.72, 0.11, 0.43],
            [0.31, -0.57, 0.18],
            [0.99, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    compact, _ = _surface_vertices_to_must_query_positions(
        vertices, init_octree_depth=init_depth, octree_depth=final_depth
    )
    legacy = _legacy_surface_chain(vertices, init_depth, final_depth)

    def plane_sdf(points):
        return points[..., :1].float()

    _, compact_anchors = hierarchical_extraction(
        geometric_func=plane_sdf,
        init_octree_depth=init_depth,
        octree_depth=final_depth,
        device="cpu",
        must_query_positions=compact,
        sdf_truncation=0.01,
    )
    _, legacy_anchors = hierarchical_extraction(
        geometric_func=plane_sdf,
        init_octree_depth=init_depth,
        octree_depth=final_depth,
        device="cpu",
        must_query_positions=legacy,
        sdf_truncation=0.01,
    )
    assert torch.equal(compact_anchors, legacy_anchors)
    assert compact.shape[0] < legacy.shape[0]


def test_hierarchical_extraction_forces_complete_finest_group():
    init_depth = 2
    final_depth = 4
    vertex = torch.tensor([[0.71, 0.37, -0.22]], dtype=torch.float64)
    must_query, counts = _surface_vertices_to_must_query_positions(
        vertex, init_octree_depth=init_depth, octree_depth=final_depth
    )

    # The ordinary interface is x=0.  The forced vertex is deliberately far
    # away, so its final group cannot be attributed to the normal band test.
    def plane_sdf(points):
        return points[..., :1].float()

    _, anchors = hierarchical_extraction(
        geometric_func=plane_sdf,
        init_octree_depth=init_depth,
        octree_depth=final_depth,
        device="cpu",
        must_query_positions=must_query,
        sdf_truncation=0.01,
    )

    n_final = 2**final_depth
    cell = torch.floor(((vertex[0] + 1.0) / 2.0) * n_final).long()
    cell = cell.clamp(0, n_final - 1)
    parent_lo = (cell // 2) * 2
    offsets = torch.tensor(
        [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
        dtype=torch.long,
    )
    children = parent_lo.unsqueeze(0) + offsets
    child_levels = anchors[children[:, 0], children[:, 1], children[:, 2]]
    assert torch.equal(child_levels, torch.full_like(child_levels, final_depth))

    # Anchor volumes must tile the cube exactly once: gaps reduce this sum and
    # overlapping coarse/fine leaves increase it.
    levels = anchors[anchors > 0].to(torch.long)
    finest_cell_volume = (2.0 / n_final) ** 3
    volume = (
        (2 ** (final_depth - levels)).to(torch.float64).pow(3).sum()
        * finest_cell_volume
    )
    assert float(volume) == pytest.approx(8.0, abs=1e-12)

    assert counts[-1][0] == final_depth

    # Expand leaf levels over finest-cell occupancy and verify all face-adjacent
    # cells obey the extractor's 2:1 balance invariant.
    leaf_level = torch.zeros_like(anchors)
    for anchor in torch.nonzero(anchors > 0, as_tuple=False):
        level = int(anchors[tuple(anchor.tolist())])
        width = 2 ** (final_depth - level)
        x, y, z = anchor.tolist()
        leaf_level[x : x + width, y : y + width, z : z + width] = level
    assert bool((leaf_level > 0).all())
    for axis in range(3):
        assert int(torch.diff(leaf_level, dim=axis).abs().max()) <= 1


def test_finest_only_position_gets_balanced_ancestry():
    """Hidden-phi points are supplied only on the finest lattice."""
    final_depth = 4
    point = torch.tensor([[0.625, -0.375, 0.125]], dtype=torch.float64)

    def plane_sdf(points):
        return points[..., :1].float()

    _, anchors = hierarchical_extraction(
        geometric_func=plane_sdf,
        init_octree_depth=2,
        octree_depth=final_depth,
        device="cpu",
        must_query_positions=point,
        sdf_truncation=0.01,
    )

    idx = ((point[0] + 1.0) * (2**final_depth) / 2.0).round().long()
    touching = idx.unsqueeze(0) + torch.tensor(
        [[x, y, z] for x in (-1, 0) for y in (-1, 0) for z in (-1, 0)]
    )
    touching = touching[((touching >= 0) & (touching < 2**final_depth)).all(dim=1)]
    touching_levels = anchors[touching[:, 0], touching[:, 1], touching[:, 2]]
    assert bool((touching_levels == final_depth).all())
