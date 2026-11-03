"""A body generated from an image (a drawing's or a web photograph's cut-out) is painted from that
image: the runner adds it to the material card's paint and to the background paint alike, so the
card is still served from the background paint.  Bodies from text, CAD and built bodies keep
the text-only paint."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402

PLAN = {
    "geometry": {"description": "a plastic handheld body", "draw": True},
    "scale": {"unit_system": "SI (m-kg-s)", "up_axis": "z", "physical_extent": 0.17},
    "material": {"youngs_modulus": 2.3e9, "poisson_ratio": 0.35, "density": 1050},
    "texture": {"texture_prompt": "light grey ABS plastic", "texture_seed": 0},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png(path: Path, value: int) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 4), value, np.uint8), "RGBA").save(path)


def _session(tmp_path: Path, mode: str = "image", recorded: str | None = None) -> Path:
    """A session whose body was generated: from the staged input image (mode "image") or from text.
    ``recorded`` is the folder session.json names (a copied session names another)."""
    import trimesh
    session = tmp_path / "s"
    (session / "geometry").mkdir(parents=True)
    (session / "operator").mkdir()
    root = recorded or str(session)
    trimesh.creation.box(extents=(1, 2, 3)).export(session / "geometry" / "generated.glb")
    trimesh.creation.box().export(session / "geometry" / "physical.glb")       # bare
    body = _sha(session / "geometry" / "generated.glb")
    receipts, state = [], {"geometry_mode": mode, "session_dir": root,
                           "surface_path": f"{root}/geometry/physical.glb"}
    if mode == "image":
        image = session / "inputs" / "original.png"
        _png(session / "inputs" / "sdxl" / "a_s0.png", 40)
        _png(image, 40)
        state["input_asset"] = {"source_name": "a_s0.png", "path": f"{root}/inputs/original.png",
                                "sha256": _sha(image)}
        receipts.append({"tool": "stage_asset",
                         "arguments": {"source": f"{root}/inputs/sdxl/a_s0.png", "output": f"{root}/inputs/original.png"},
                         "artifact_digests": {f"{root}/inputs/original.png": _sha(image)}})
        receipts.append({"tool": "generate_image_surface",
                         "arguments": {"output": f"{root}/geometry/generated.glb", "input": f"{root}/inputs/original.png"},
                         "artifact_digests": {f"{root}/geometry/generated.glb": body}})
    else:
        receipts.append({"tool": "generate_text_surface",
                         "arguments": {"output": f"{root}/geometry/generated.glb", "prompt": "x"},
                         "artifact_digests": {f"{root}/geometry/generated.glb": body}})
    state["tool_receipts"] = receipts
    lib.write_json(session / "session.json", state)
    return session


def _card_answer(session: Path) -> dict:
    """The material card's answer as the runner builds and lints it (run: plan_answer, then
    with_body_image, then prepare_answer)."""
    card = {"kind": "material_input", "surface_has_colour": False, "textured_surface": None,
            "material_parts": []}
    answer, _, _ = drive.plan_answer("material_input", card, PLAN)
    answer = drive.with_body_image(session, answer)
    workflow = lib.read_json(session / "session.json", {})
    prepared = drive.prepare_answer(card, workflow, answer, merge=True, force=False)
    assert not prepared.errors
    return prepared.answer


def test_an_image_route_body_is_painted_from_its_image_and_the_background_paint_is_adopted(tmp_path):
    session = _session(tmp_path)
    image = str((session / "inputs" / "original.png").resolve())
    assert drive.body_image(session) == image
    ahead = drive.paint_ahead_answer(session, PLAN)
    assert ahead["texture_image"] == image
    assert ahead["texture_prompt"] == "light grey ABS plastic"            # the words stay
    card = _card_answer(session)
    assert card == ahead
    assert drive._paint_ahead_key(session, card) == drive._paint_ahead_key(session, ahead)


def test_a_copied_image_route_session_finds_its_image_in_its_own_folder(tmp_path):
    session = _session(tmp_path, recorded="/elsewhere/slot/abc")
    assert drive.body_image(session) == str((session / "inputs" / "original.png").resolve())
    assert _card_answer(session) == drive.paint_ahead_answer(session, PLAN)


def test_a_text_route_body_keeps_the_text_only_paint(tmp_path):
    session = _session(tmp_path, mode="text")
    assert drive.body_image(session) is None
    ahead = drive.paint_ahead_answer(session, PLAN)
    assert "texture_image" not in ahead and ahead == _card_answer(session)


def test_a_body_not_generated_from_the_image_on_disk_is_not_painted_from_it(tmp_path):
    # a later image staged over the input (a new drawing or photograph) without a generation from it
    session = _session(tmp_path)
    state = lib.read_json(session / "session.json")
    _png(session / "inputs" / "original.png", 200)
    state["input_asset"]["sha256"] = _sha(session / "inputs" / "original.png")
    state["tool_receipts"].append({"tool": "stage_asset", "arguments": {
        "source": "x.png", "output": f"{session}/inputs/original.png"},
        "artifact_digests": {f"{session}/inputs/original.png": state["input_asset"]["sha256"]}})
    lib.write_json(session / "session.json", state)
    assert drive.body_image(session) is None
    # the body regenerated afterwards from text (a CAD or text route): the last generation decides
    session = _session(tmp_path / "b")
    state = lib.read_json(session / "session.json")
    body = state["tool_receipts"][-1]["artifact_digests"]
    state["tool_receipts"].append({"tool": "generate_cadcoder_text_surface", "arguments": {"prompt": "x"},
                                   "artifact_digests": body})
    lib.write_json(session / "session.json", state)
    assert drive.body_image(session) is None


def test_a_built_or_uploaded_body_is_not_painted_from_an_image(tmp_path):
    import trimesh
    session = _session(tmp_path)
    trimesh.creation.box().export(session / "geometry" / "original.glb")
    state = lib.read_json(session / "session.json")
    state["geometry_mode"] = "surface"
    lib.write_json(session / "session.json", state)
    assert drive.body_image(session) is None
    assert "texture_image" not in drive.paint_ahead_answer(session, PLAN)


def test_only_a_painting_answer_takes_the_image(tmp_path):
    session = _session(tmp_path)
    flat = {"action": "texture_surface", "per_part_colours": True}
    assert drive.with_body_image(session, flat) == flat
    own = {"action": "texture_surface", "texture_image": "/x.png"}
    assert drive.with_body_image(session, own) == own
    material = {"material_mode": "uniform", "youngs_modulus": 1.0}
    assert drive.with_body_image(session, material) == material
    assert drive.with_body_image(session, None) is None


def test_the_texture_stop_judges_an_image_paint_against_the_image_not_the_words(tmp_path, monkeypatch):
    import trimesh
    monkeypatch.setattr(drive, "mark_paint_views", lambda stats, out, compute_env=None: None)
    session = _session(tmp_path)
    painted = session / "materials" / "texture" / "textured.glb"
    painted.parent.mkdir(parents=True)
    mesh = trimesh.creation.icosphere(subdivisions=4)        # light grey with a black side, as the image has
    rgb = np.where(mesh.vertices[:, :1] < -0.5, 5.0, 200.0) + np.random.default_rng(0).normal(0, 3, (len(mesh.vertices), 3))
    mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.clip(rgb, 0, 255).astype(np.uint8))
    mesh.export(painted)
    card = {"textured_surface": str(painted), "texture_prompt": "light grey ABS plastic", "texture_seed": 0}
    state = lib.read_json(session / "session.json")
    image = f"{session}/inputs/original.png"
    paint = {"tool": "paint_hunyuan3d_surface", "arguments": {"output": str(painted), "input": image},
             "artifact_digests": {str(painted): _sha(painted)}}
    # painted from the words: the words' colour is checked (a black body asked light grey)
    lib.write_json(session / "session.json", {**state, "tool_receipts": state["tool_receipts"] + [
        {**paint, "arguments": {"output": str(painted)}}]})
    assert drive.painted_from_image(session, painted) is None
    payload = drive._stop_payload("texture", session, tmp_path, card, PLAN)
    assert "near-black" in payload["warnings"][0]
    # painted from the body's image: compared with the image, the words' colour is not checked
    lib.write_json(session / "session.json", {**state, "tool_receipts": state["tool_receipts"] + [paint]})
    assert drive.painted_from_image(session, painted) == image
    payload = drive._stop_payload("texture", session, tmp_path, card, PLAN)
    assert not payload.get("warnings") and payload["texture"]["painted_from_image"] == image
    assert "compare the views with it" in payload["note"]
