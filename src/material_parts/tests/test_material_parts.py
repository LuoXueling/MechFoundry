from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
trimesh = pytest.importorskip("trimesh")

from src.material_parts import field_from_parts
from src.vomp_wrapper.artifact import load_artifact
from src.vomp_wrapper.visualize import sample_vertices


def two_blocks():
    """Two unit cubes a gap apart, labelled 0 and 1 by face."""
    left, right = trimesh.creation.box((1, 1, 1)), trimesh.creation.box((1, 1, 1))
    right.apply_translation((1.5, 0, 0))
    mesh = trimesh.util.concatenate([left, right])
    labels = np.r_[np.zeros(len(left.faces), int), np.ones(len(right.faces), int)]
    return mesh, labels


def test_each_part_carries_its_own_material_through_its_volume(tmp_path):
    mesh, labels = two_blocks()
    field_from_parts(mesh, labels, {0: (1e9, 0.3, 1000.0), 1: (2e11, 0.28, 7850.0)},
                     tmp_path / "field", provenance={"test": True})
    artifact = load_artifact(tmp_path / "field", verify=True)
    assert artifact.manifest["provenance"] == {"method": "parts", "test": True}
    inside = np.array([[0, 0, 0], [0.4, 0.4, -0.4], [1.5, 0, 0], [1.1, -0.4, 0.4]])
    sampled = sample_vertices(artifact, inside)
    np.testing.assert_allclose(sampled[:, 0], [1e9, 1e9, 2e11, 2e11], rtol=1e-6)
    np.testing.assert_allclose(sampled[:, 1], [0.3, 0.3, 0.28, 0.28], rtol=1e-6)
    np.testing.assert_allclose(sampled[:, 2], [1000, 1000, 7850, 7850], rtol=1e-6)


def test_a_part_without_a_material_is_refused(tmp_path):
    mesh, labels = two_blocks()
    with pytest.raises(ValueError, match=r"no material for parts \[1\]"):
        field_from_parts(mesh, labels, {0: (1e9, 0.3, 1000.0)}, tmp_path / "field")


def textured_box(path):
    """A box whose UV texture is red for x < 0 and blue for x > 0, exported as glTF."""
    from PIL import Image

    box = trimesh.creation.box((1, 1, 1))
    pixels = np.zeros((64, 64, 3), np.uint8)
    pixels[:, :32], pixels[:, 32:] = (255, 0, 0), (0, 0, 255)
    uv = np.c_[(box.vertices[:, 0] + 0.5) * 0.98 + 0.01, np.full(len(box.vertices), 0.5)]
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(pixels)))
    box.export(path)
    return trimesh.load(path, force="mesh", process=False)


def test_a_uv_texture_is_read_as_the_surface_colour(tmp_path):
    """An unpainted textured upload has colour: the montage reads its texture."""
    pytest.importorskip("matplotlib")
    import argparse
    import json

    import importlib.util

    # By path: the package would import the workflow graph, which needs the
    # agents environment, while the montage needs the compute one's matplotlib.
    script = (Path(__file__).resolve().parents[2] / "agents_wrapper" / "fem_workflow"
              / "scripts" / "workflow_tool.py")
    spec = importlib.util.spec_from_file_location("workflow_tool", script)
    workflow_tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow_tool)

    mesh = textured_box(tmp_path / "box.glb")
    colours = workflow_tool._vertex_face_colors(mesh)
    centroid_x = mesh.triangles_center[:, 0]
    # Faces on the -x side are red, on the +x side blue; the rest straddle both.
    assert np.allclose(colours[centroid_x < -0.4, :3], [1, 0, 0], atol=0.02)
    assert np.allclose(colours[centroid_x > 0.4, :3], [0, 0, 1], atol=0.02)

    labels = (centroid_x > 0).astype(int)
    np.savez(tmp_path / "segments.npz", segment_ids=labels)
    (tmp_path / "segments.json").write_text(json.dumps({"segments": [
        {"id": 0, "color": "#1f77b4"}, {"id": 1, "color": "#ff7f0e"}]}))
    args = argparse.Namespace(surface=tmp_path / "box.glb", segments_npz=tmp_path / "segments.npz",
                              segments_json=tmp_path / "segments.json", output=tmp_path / "montage.png")
    workflow_tool.render_part_montage(args)
    assert (tmp_path / "montage.png").stat().st_size > 0

    np.savez(tmp_path / "segments.npz", segment_ids=labels[:-1])
    with pytest.raises(ValueError, match="belong to a different surface"):
        workflow_tool.render_part_montage(args)


def test_editing_a_part_redraws_the_field_s_previews(tmp_path):
    """The field is rewritten in place; previews of the values it replaced are not current."""
    import json
    from types import SimpleNamespace

    from src.agents_wrapper.fem_workflow.scripts import workflow_tool
    from src.vomp_wrapper.visualize import surface_visualizations_current

    mesh, labels = two_blocks()
    surface = tmp_path / "physical.glb"
    mesh.export(surface)
    np.savez(tmp_path / "segments.npz", segment_ids=labels)
    materials = tmp_path / "materials.json"
    field = tmp_path / "field"
    args = SimpleNamespace(surface=surface, segments_npz=tmp_path / "segments.npz",
                           materials_json=materials, unit_system="SI (m-kg-s)", output=field)

    def build(soft_modulus):
        materials.write_text(json.dumps({"parts": [
            {"id": 0, "youngs_modulus": soft_modulus, "poisson_ratio": 0.3, "density": 1000.0},
            {"id": 1, "youngs_modulus": 2e11, "poisson_ratio": 0.28, "density": 7850.0},
        ]}))
        workflow_tool.build_part_field(args)
        return json.loads((field / "youngs_modulus.glb.json").read_text())

    assert build(1e9)["data_minimum"] == pytest.approx(1e9, rel=1e-6)
    assert surface_visualizations_current(field)
    # The page's own check sees the grid change even when nothing redraws them.
    field_from_parts(mesh, labels, {0: (5e6, 0.3, 1000.0), 1: (2e11, 0.28, 7850.0)}, field)
    assert not surface_visualizations_current(field)
    # The tool draws them with the field, so an edit reaches the viewer.
    assert build(5e6)["data_minimum"] == pytest.approx(5e6, rel=1e-6)
    assert surface_visualizations_current(field)
