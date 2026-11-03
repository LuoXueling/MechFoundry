"""Sign-convention detection for the SDF used by the remeshing path."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.analytical_sdf import (  # noqa: E402
    _closed_surface_orientation, build_warp_mesh, compute_nodal_sdf,
)


def _hollow_box(inner=0.4, outer=1.0):
    """A box with a concentric cavity: its bbox centroid lies in air."""
    trimesh = pytest.importorskip("trimesh")
    shell = trimesh.creation.box(extents=(outer, outer, outer))
    cavity = trimesh.creation.box(extents=(inner, inner, inner))
    cavity.invert()          # inward-facing normals enclose the void
    return trimesh.util.concatenate([shell, cavity])


def _as_tensors(mesh, device):
    verts = torch.from_numpy(np.ascontiguousarray(mesh.vertices, dtype=np.float32)).to(device)
    faces = torch.from_numpy(np.ascontiguousarray(mesh.faces, dtype=np.int32)).to(device)
    return verts, faces


def test_orientation_follows_signed_volume_not_the_probe_point():
    """A hollow part's bbox centroid is in a void, which inverted the field."""
    trimesh = pytest.importorskip("trimesh")
    if not torch.cuda.is_available():
        pytest.skip("sign detection runs on the CUDA BVH")
    mesh = _hollow_box()
    verts, faces = _as_tensors(mesh, "cuda")

    orientation = _closed_surface_orientation(verts, faces)
    assert orientation is not None and orientation > 0.0

    # The probe the caller supplies falls inside the cavity, not the material.
    probe = torch.zeros(3, dtype=torch.float32, device="cuda")
    diagonal = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    wp_mesh, negate, _keep = build_warp_mesh(verts, faces, probe, diagonal, device="cuda")
    assert negate is False

    # A point in the shell wall must read negative under the standard convention.
    wall = torch.tensor([[0.35, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    value = float(compute_nodal_sdf(wp_mesh, wall, negate, max_dist=10 * diagonal,
                                    device="cuda")[0])
    assert value < 0.0


def test_solid_box_orientation_is_unchanged():
    """The common case must keep behaving exactly as before."""
    trimesh = pytest.importorskip("trimesh")
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    verts = torch.from_numpy(np.ascontiguousarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.ascontiguousarray(mesh.faces, dtype=np.int32))
    orientation = _closed_surface_orientation(verts, faces)
    assert orientation == pytest.approx(1.0, rel=1e-5)

    # Inverted winding is detected as such, so the caller negates.
    mesh.invert()
    verts = torch.from_numpy(np.ascontiguousarray(mesh.vertices, dtype=np.float32))
    faces = torch.from_numpy(np.ascontiguousarray(mesh.faces, dtype=np.int32))
    assert _closed_surface_orientation(verts, faces) == pytest.approx(-1.0, rel=1e-5)


def test_open_surface_defers_to_the_probe():
    """An open sheet has no meaningful signed volume, so the probe still decides."""
    verts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
    assert _closed_surface_orientation(verts, faces) is None
