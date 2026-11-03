import numpy as np
import pytest

from src.cad_wrapper.cad_common.mesh import HALF_EXTENT
from src.cad_wrapper.text2cad_wrapper.availability import face_labels_path, settings, text2cad_status


def test_missing_environment_is_not_ready(monkeypatch, tmp_path):
    text2cad_status.cache_clear()
    monkeypatch.setenv("TEXT2CAD_CONDA_ENV", "mechfoundry-text2cad-missing")
    monkeypatch.setenv("TEXT2CAD_ROOT", str(tmp_path / "missing"))
    try:
        status = text2cad_status()
        assert not status.ready
        assert status.model_dump()["environment"] == "mechfoundry-text2cad-missing"
    finally:
        text2cad_status.cache_clear()


def test_the_checkpoint_and_face_labels_have_fixed_homes(monkeypatch, tmp_path):
    monkeypatch.setenv("TEXT2CAD_ROOT", str(tmp_path))
    monkeypatch.delenv("TEXT2CAD_CHECKPOINT", raising=False)
    _, root, checkpoint = settings()
    assert checkpoint == root / "pretrained_weights" / "text2cad_v1.0" / "Text2CAD_1.0.pth"
    assert face_labels_path(tmp_path / "generated.glb").name == "generated.glb.faces.npz"


def test_normalize_fits_the_generator_box():
    trimesh = pytest.importorskip("trimesh")
    generate = pytest.importorskip("src.cad_wrapper.text2cad_wrapper.generate")
    box = trimesh.creation.box(extents=(8.0, 2.0, 1.0))
    box.apply_translation((10.0, -3.0, 2.0))
    _, labels = np.unique(box.face_normals.round(6), axis=0, return_inverse=True)  # one per side
    surface, split = generate.normalize(box, faces=2000, labels=labels)
    assert surface.is_watertight
    assert split.shape == (len(surface.faces),)
    # Every piece keeps the side it was cut from: one normal per label.
    for side in range(6):
        normals = surface.face_normals[split == side]
        assert len(normals) and np.allclose(normals, normals[0], atol=1e-9)
    assert abs(surface.bounds).max() == pytest.approx(HALF_EXTENT)
    assert surface.bounds.mean(axis=0) == pytest.approx([0.0, 0.0, 0.0], abs=1e-9)
    assert len(box.faces) < len(surface.faces) <= 2000


def test_cad_parts_stand_up_in_the_workflow_frame():
    """CAD is modelled z-up; the workflow reads +y as up, so a tall part is tall in y."""
    trimesh = pytest.importorskip("trimesh")
    from src.cad_wrapper.cad_common.mesh import normalize

    tower = trimesh.creation.box(extents=(1.0, 2.0, 8.0))  # tall along CAD z
    top = tower.face_normals[:, 2] > 0.5                 # its upward-facing cap
    surface, labels = normalize(tower, faces=2000, labels=top.astype(int))
    extents = surface.extents
    assert extents[1] == pytest.approx(extents.max())    # tall along workflow y
    assert extents[2] == pytest.approx(extents.max() / 4)  # CAD y (2) now along z
    assert surface.is_watertight and surface.volume > 0   # still outward-facing
    cap = surface.face_normals[labels == 1]
    assert np.allclose(cap, [0.0, 1.0, 0.0])              # the top faces +y


def test_plate_with_hole_builds_a_closed_solid():
    pytest.importorskip("OCC.Core")
    generate = pytest.importorskip("src.cad_wrapper.text2cad_wrapper.generate")
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

    plate = BRepPrimAPI_MakeBox(gp_Pnt(-1, -1, 0), 2.0, 2.0, 0.2).Shape()
    bore = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0, 0, -1), gp_Dir(0, 0, 1)), 0.1, 3.0).Shape()
    mesh, labels, surface_types, discarded = generate.cad_to_mesh(BRepAlgoAPI_Cut(plate, bore).Shape())
    assert mesh.is_watertight and discarded == 0
    assert mesh.euler_number == 0  # one through-hole
    # Six B-rep faces -- top, bottom, four sides -- plus the bore, which is the
    # one cylinder and lies at the hole's radius from the axis.
    assert sorted(surface_types) == ["cylinder"] + ["plane"] * 6
    assert labels.shape == (len(mesh.faces),) and len(set(labels.tolist())) == 7
    bore_faces = labels == surface_types.index("cylinder")
    radii = np.linalg.norm(mesh.triangles_center[bore_faces][:, :2], axis=1)
    assert np.allclose(radii, radii.mean(), rtol=0.05)
