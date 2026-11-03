import numpy as np
import pytest

from src.hunyuan3d_wrapper.availability import hunyuan3d_status, sdxl_status


def test_missing_environment_is_not_ready(monkeypatch, tmp_path):
    hunyuan3d_status.cache_clear()
    monkeypatch.setenv("HUNYUAN3D_CONDA_ENV", "mechfoundry-hunyuan3d-missing")
    monkeypatch.setenv("HUNYUAN3D_ROOT", str(tmp_path / "missing"))
    try:
        status = hunyuan3d_status()
        assert not status.ready
        assert status.model_dump()["environment"] == "mechfoundry-hunyuan3d-missing"
    finally:
        hunyuan3d_status.cache_clear()


def test_sdxl_needs_its_environment_and_its_weights(monkeypatch, tmp_path):
    """SDXL is ready without Hunyuan3D's own models, but not without its snapshots."""
    from src.hunyuan3d_wrapper import availability

    sdxl_status.cache_clear()
    monkeypatch.setenv("HUNYUAN3D_CONDA_ENV", "mechfoundry-hunyuan3d-missing")
    try:
        assert "was not found" in sdxl_status().summary
        sdxl_status.cache_clear()
        monkeypatch.setattr(availability, "conda_interpreter", lambda name: tmp_path / "python")
        monkeypatch.setenv("HUNYUAN3D_WEIGHTS", str(tmp_path / "empty-cache"))
        status = sdxl_status()
        assert not status.ready
        assert "stable-diffusion-xl-base-1.0" in status.summary
        assert "sdxl-vae-fp16-fix" in status.summary
    finally:
        sdxl_status.cache_clear()


def test_the_drawing_is_cut_out_where_rmbg_is_sure():
    from src.hunyuan3d_wrapper.draw_image import otsu

    levels = np.concatenate([np.full(900, 12), np.full(100, 240)]).astype(np.uint8)
    threshold = otsu(levels)
    assert 12 <= threshold < 240 and (levels > threshold).sum() == 100


def _seam_split_box():
    """A welded box, and a copy split along every face into its own UV island."""
    trimesh = pytest.importorskip("trimesh")
    Image = pytest.importorskip("PIL.Image")

    box = trimesh.creation.box(extents=(2.0, 1.0, 1.0))
    # Each triangle gets its own three vertices, as a UV unwrap splits seams.
    vertices = box.vertices[box.faces].reshape(-1, 3)
    # Left half of the texture red, right half blue; +x corners sample the right.
    u = np.where(vertices[:, 0] > 0, 0.75, 0.25)
    uv = np.column_stack([u, np.full_like(u, 0.5)])
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    texture[:, :4] = (255, 0, 0)
    texture[:, 4:] = (0, 0, 255)
    return box, vertices, uv, Image.fromarray(texture)


def test_colours_come_back_onto_the_welded_surface():
    pytest.importorskip("scipy")
    from src.hunyuan3d_wrapper.surface import transfer_colours

    box, vertices, uv, texture = _seam_split_box()
    colours = transfer_colours(box, vertices, uv, texture)
    assert colours.shape == (len(box.vertices), 4)
    assert (colours[:, 3] == 255).all()
    right = box.vertices[:, 0] > 0
    assert (colours[right, :3] == (0, 0, 255)).all()
    assert (colours[~right, :3] == (255, 0, 0)).all()


def test_a_moved_surface_is_refused():
    pytest.importorskip("scipy")
    from src.hunyuan3d_wrapper.surface import transfer_colours

    box, vertices, uv, texture = _seam_split_box()
    with pytest.raises(RuntimeError, match="does not match"):
        transfer_colours(box, vertices + 0.1, uv, texture)


def test_cleanup_keeps_the_largest_part_and_fits_the_frame():
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")  # trimesh's component split
    from src.hunyuan3d_wrapper.surface import HALF_EXTENT, clean_surface, fit_to_frame

    body = trimesh.creation.icosphere(subdivisions=3, radius=3.0)
    floater = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    floater.apply_translation((10.0, 0.0, 0.0))
    # Tessellation seams: duplicate the body's vertices per face, as marching
    # cubes output from a chunked extraction can.
    split = trimesh.Trimesh(body.vertices[body.faces].reshape(-1, 3),
                            np.arange(3 * len(body.faces)).reshape(-1, 3), process=False)
    surface, discarded = clean_surface(trimesh.util.concatenate([split, floater]))
    assert discarded == 1 and surface.is_watertight and surface.volume > 0
    fitted = fit_to_frame(surface)
    assert np.isclose(np.abs(fitted.bounds).max(), HALF_EXTENT)
    assert np.allclose(fitted.bounds.sum(axis=0), 0.0, atol=1e-9)
    assert fitted.is_watertight


def test_the_upstream_pipelines_import(monkeypatch):
    from src.hunyuan3d_wrapper.generate import Hunyuan3DRuntime

    for name in ("HF_HOME", "HF_HUB_OFFLINE", "U2NET_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("sys.path", list(__import__("sys").path))
    Hunyuan3DRuntime()._prepare()
    pytest.importorskip("hy3dshape.pipelines", reason="run in the mechfoundry-hunyuan3d environment")
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: F401
    from textureGenPipeline import Hunyuan3DPaintPipeline  # noqa: F401


def test_colours_come_back_from_a_simplified_copy():
    """A surface painted through a decimated copy takes the colour of the nearest painted point."""
    trimesh = pytest.importorskip("trimesh")
    pytest.importorskip("scipy")
    Image = pytest.importorskip("PIL.Image")
    from src.hunyuan3d_wrapper.surface import transfer_colours_nearest

    fine = trimesh.creation.icosphere(subdivisions=4)
    coarse = trimesh.creation.icosphere(subdivisions=2)
    # The coarse copy's UV puts x > 0 on the texture's blue right half, the rest on red.
    coarse.visual = trimesh.visual.TextureVisuals(uv=np.column_stack([
        np.where(coarse.vertices[:, 0] > 0, 0.75, 0.25), np.full(len(coarse.vertices), 0.5)]))
    texture = np.zeros((8, 8, 3), dtype=np.uint8)
    texture[:, :4] = (255, 0, 0)
    texture[:, 4:] = (0, 0, 255)
    colours = transfer_colours_nearest(fine, coarse, Image.fromarray(texture))
    assert colours.shape == (len(fine.vertices), 4) and (colours[:, 3] == 255).all()
    # Away from the seam at x = 0, where a face straddles both halves.
    far = np.abs(fine.vertices[:, 0]) > 0.3
    right = fine.vertices[:, 0] > 0
    assert (colours[far & right, :3] == (0, 0, 255)).all()
    assert (colours[far & ~right, :3] == (255, 0, 0)).all()


def test_colour_is_detected_only_where_there_is_some(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from src.hunyuan3d_wrapper.surface import has_colour

    box = trimesh.creation.box()
    box.export(tmp_path / "grey.glb")
    assert not has_colour(tmp_path / "grey.glb")
    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=np.tile((200, 30, 30, 255), (8, 1)))
    box.export(tmp_path / "red.glb")
    assert has_colour(tmp_path / "red.glb")


def test_painting_never_loads_the_shape_stage(tmp_path):
    """A paint asks for the paint stage alone, and cuts out an image with the remover loaded on demand."""
    pytest.importorskip("torch")
    trimesh = pytest.importorskip("trimesh")
    from src.hunyuan3d_wrapper.generate import Hunyuan3DRuntime, paint_surface, parser

    surface = tmp_path / "surface.glb"
    trimesh.creation.box().export(surface)
    asked = []

    class Stop(Exception):
        pass

    class Runtime(Hunyuan3DRuntime):
        def preload(self, texture=None, shape=True):
            asked.append((texture, shape))
            raise Stop

    args = parser().parse_args(["--surface", str(surface), "--output", str(tmp_path / "out.glb"),
                                "--prompt", "oak"])
    with pytest.raises(Stop):
        paint_surface(args, Runtime())
    assert asked == [(True, False)]

    removed = []
    runtime = Hunyuan3DRuntime()
    runtime.background = lambda image: removed.append(image) or image
    assert runtime.remove_background("image") == "image" and removed == ["image"]


def test_the_reference_is_framed_on_the_silhouette():
    pytest.importorskip("PIL")
    from src.hunyuan3d_wrapper.generate import framed

    near = np.zeros((100, 100), dtype=np.float32)
    mask = np.zeros((100, 100), dtype=bool)
    mask[40:60, 45:50] = True  # a thin sliver
    near[mask] = 0.6
    near2, mask2 = framed(near, mask, 0.1)
    assert mask2.shape == (100, 100)
    rows, cols = np.nonzero(mask2)
    # The sliver now spans the frame's height, less the margin, and stays centred.
    assert rows.max() - rows.min() > 70 and abs((cols.min() + cols.max()) / 2 - 50) < 3
    assert np.allclose(near2[mask2], 0.6, atol=0.05) and (near2[~mask2] == 0).all()
    # Nothing to frame: unchanged.
    empty = np.zeros_like(mask)
    assert framed(near, empty, 0.1)[1] is empty


def _gated_views(tiles_per_attempt, reference_rgb, boxes=None, reference=None, **options):
    """A gate over a fake multiview model that returns ``tiles_per_attempt[attempt]`` (flat colours).

    ``boxes`` gives each view's body in its normal map (default the middle 8x8);
    ``reference`` an RGBA reference image in place of a flat ``reference_rgb`` one.
    """
    Image = pytest.importorskip("PIL.Image")
    from src.hunyuan3d_wrapper.generate import _GatedViews

    size = 16
    views = len(tiles_per_attempt[0])
    normals = []
    for box in boxes or [(4, 4, 12, 12)] * views:
        normal = Image.new("RGB", (size, size), (255, 255, 255))
        normal.paste((128, 128, 255), box)  # a normal map: the body in the box
        normals.append(normal)
    calls = []

    def model(images, conditions, **kwargs):
        colours = tiles_per_attempt[len(calls)]
        calls.append(kwargs)
        return {"albedo": [Image.new("RGB", (size, size), colour) for colour in colours], "mr": []}

    baked = []
    if reference is None:
        reference = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        reference.paste(reference_rgb + (255,), (4, 4, 12, 12))
    seeds = []
    gate = _GatedViews(model, lambda *lists: baked.append(lists) or "baked", reference,
                       reseed=seeds.append, **options)
    conditions = normals * 2
    return gate, conditions, calls, baked, seeds


def _two_colour_reference(first, second, share, size=100):
    """An opaque RGBA reference, ``second`` on the top ``share`` of its rows and ``first`` below."""
    Image = pytest.importorskip("PIL.Image")
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3], rgba[..., 3] = first, 255
    rgba[:int(round(size * share)), :, :3] = second
    return Image.fromarray(rgba)


WOOD, METAL, MID_GREY = (196, 150, 100), (70, 72, 78), (150, 150, 150)


def test_the_multicolour_spread_tells_shading_and_specks_from_a_second_colour():
    """The synthetic check behind MULTICOLOUR_SPREAD_DELTA_E (numbers quoted in generate.py)."""
    Image = pytest.importorskip("PIL.Image")
    from src.hunyuan3d_wrapper.generate import MULTICOLOUR_SPREAD_DELTA_E, _body_lab, colour_spread

    limit = MULTICOLOUR_SPREAD_DELTA_E
    ramp = np.zeros((100, 100, 4), np.uint8)
    ramp[..., 3] = 255
    ramp[..., :3] = np.clip(np.array(WOOD) * np.linspace(0.8, 1.2, 100)[:, None, None], 0, 255)
    noisy = np.zeros((100, 100, 4), np.uint8)
    noisy[..., 3] = 255
    noisy[..., :3] = np.clip(np.array(WOOD) + np.random.default_rng(1).normal(0, 8, (100, 100, 3)), 0, 255)
    single = [colour_spread(_body_lab(Image.fromarray(image))) for image in (ramp, noisy)]
    speck = colour_spread(_body_lab(_two_colour_reference(WOOD, METAL, 0.05)))
    assert max(single) < limit - 5 and speck < limit
    # A second colour on any share from under a tenth to most of the body counts, the near pair
    # too; a spread from the median would halve at an even split.
    for share in (0.08, 0.3, 0.5, 0.9):
        for second in (METAL, MID_GREY):
            assert colour_spread(_body_lab(_two_colour_reference(WOOD, second, share))) > limit + 5


def test_a_view_unlike_the_front_is_not_baked():
    brown, grey = (140, 90, 50), (128, 128, 128)
    gate, conditions, calls, baked, seeds = _gated_views([[brown, brown, brown, brown, grey, brown]], brown)
    gate(["reference"], conditions)
    assert len(calls) == 1 and seeds == []
    assert gate.kept == [0, 1, 2, 3, 5] and gate.distances[4] > 20
    assert gate.bake(list("abcdef"), list(range(6)), list(range(6)), [1] * 6) == "baked"
    assert baked[0][0] == ["a", "b", "c", "d", "f"] and baked[0][1] == [0, 1, 2, 3, 5]


def test_views_unlike_the_reference_are_drawn_again():
    light, black = (190, 190, 185), (5, 5, 5)
    gate, conditions, calls, baked, seeds = _gated_views(
        [[black] * 6, [light] * 6, [black] * 6], light, retries=2)
    result = gate(["reference"], conditions)
    # The first set swapped body and background; the second matched and was kept.
    assert len(calls) == 2 and seeds == [1] and gate.attempts == 2 and gate.kept_attempt == 1
    assert np.asarray(result["albedo"][0])[8, 8].tolist() == list(light)
    assert gate.front_distance < 28 and gate.kept == list(range(6)) and gate.gate == "applied"


def test_a_multicoloured_reference_keeps_the_views_of_its_second_colour():
    reference = _two_colour_reference(WOOD, METAL, 0.3)
    gate, conditions, calls, baked, seeds = _gated_views(
        [[WOOD, WOOD, METAL, METAL, WOOD, METAL]], WOOD, reference=reference)
    gate(["reference"], conditions)
    # Views where the second material dominates are far from the anchor but kept, and not redrawn.
    assert gate.multicoloured and len(calls) == 1 and seeds == []
    assert gate.distances[2] > 20 and gate.kept is None and gate.gate.startswith("not applied")
    gate.bake(list("abcdef"), list(range(6)), list(range(6)), [1] * 6)
    assert baked[0][0] == list("abcdef")


def test_a_multicoloured_reference_still_redraws_a_swapped_body():
    reference = _two_colour_reference(WOOD, MID_GREY, 0.3)
    black = (5, 5, 5)
    gate, conditions, calls, baked, seeds = _gated_views(
        [[black] * 6, [MID_GREY] * 6], WOOD, reference=reference)
    gate(["reference"], conditions)
    # Near-black is in no colour of the reference; mid grey is, though not its median.
    assert len(calls) == 2 and seeds == [1] and gate.kept_attempt == 1 and gate.front_distance < 28


def test_the_anchor_is_the_view_showing_the_most_body():
    brown, grey = (140, 90, 50), (128, 128, 128)
    edge = (0, 7, 8, 8)     # the front of a thin plate seen edge-on: 8 body pixels, too few to trust
    sliver = (0, 7, 16, 8)  # 16 body pixels: just enough
    boxes = [edge, (4, 4, 12, 12), (0, 0, 16, 16), (4, 4, 12, 12), (4, 4, 12, 12), sliver]
    gate, conditions, calls, baked, seeds = _gated_views(
        [[grey, brown, brown, brown, grey, brown]], brown, boxes=boxes)
    gate(["reference"], conditions)
    # The grey front no longer turns the gate off or anchors it: the full view does, and
    # the grey views are left out; the front's few pixels are not trusted, so it is kept.
    assert gate.anchor == 2 and gate.gate == "applied"
    assert gate.distances[0] is None and gate.distances[4] > 20 and gate.kept == [0, 1, 2, 3, 5]


def test_the_gate_says_when_no_view_shows_enough_body():
    brown = (140, 90, 50)
    gate, conditions, calls, baked, seeds = _gated_views(
        [[brown] * 6], brown, boxes=[(0, 0, 3, 3)] * 6)
    gate(["reference"], conditions)
    assert gate.anchor is None and gate.kept is None and len(calls) == 1
    assert gate.gate.startswith("not applied") and "body pixels" in gate.gate


def _shaded_body(rgb, size=40):
    """An RGBA image: a body of colour ``rgb`` under a ±15 % shading ramp, on a grey background."""
    rgba = np.zeros((size, size, 4), np.uint8)
    rgba[..., :3] = (128, 128, 128)                                  # background, not body
    ramp = np.linspace(0.85, 1.15, size // 2)[:, None, None]
    body = np.clip(np.asarray(rgb, float)[None, None, :] * ramp * np.ones((1, size // 2, 1)), 0, 255)
    rgba[size // 4:3 * size // 4, size // 4:3 * size // 4, :3] = body.astype(np.uint8)
    rgba[size // 4:3 * size // 4, size // 4:3 * size // 4, 3] = 255
    return rgba


def test_a_mid_grey_reference_is_lifted_keeping_its_shading():
    """lifted_reference: a near-neutral mid grey's median L* goes to REFERENCE_LIFT_TO, its
    shading keeps its order, and the background and alpha are untouched."""
    Image = pytest.importorskip("PIL.Image")
    from src.hunyuan3d_wrapper.generate import REFERENCE_LIFT_BELOW, REFERENCE_LIFT_TO, lifted_reference

    rgba = _shaded_body((128, 128, 130))                             # L* about 54, C* about 1
    lifted, before, after = lifted_reference(Image.fromarray(rgba))
    assert before < REFERENCE_LIFT_BELOW and abs(after - REFERENCE_LIFT_TO) < 1.0
    out = np.asarray(lifted)
    assert (out[:10] == rgba[:10]).all() and (out[..., 3] == rgba[..., 3]).all()
    assert (np.diff(out[10:30, 20, 0].astype(int)) >= 0).all()      # the shading keeps its order


@pytest.mark.parametrize("rgb", [(200, 30, 30), (150, 50, 40), (20, 30, 90), (25, 25, 25),
                                 (70, 45, 30), (200, 200, 200)],
                         ids=["red", "brick", "navy", "black", "dark wood", "light grey"])
def test_a_coloured_black_or_light_reference_is_left_as_drawn(rgb):
    """Hue-bearing colours, blacks and light bodies are the look asked for: returned unchanged."""
    Image = pytest.importorskip("PIL.Image")
    from src.hunyuan3d_wrapper.generate import lifted_reference

    image = Image.fromarray(_shaded_body(rgb))
    same, before, after = lifted_reference(image)
    assert same is image and after is None and before is not None


def _banded_body(parts, size=100):
    """An RGBA image whose body is horizontal bands of CIE L*a*b* colours ``parts`` [(lab, share)]."""
    from src.hunyuan3d_wrapper.generate import _srgb_from_lab

    rgba = np.zeros((size, size, 4), np.uint8)
    row = 0
    for lab, share in parts:
        rows = int(round(share * size))
        rgba[row:row + rows, :, :3] = np.rint(_srgb_from_lab(np.asarray(lab, float))).astype(np.uint8)
        rgba[row:row + rows, :, 3] = 255
        row += rows
    return rgba


@pytest.mark.parametrize("parts", [
    [((50, 0, 0), 0.7), ((45, 60, 40), 0.3)],
    [((50, 0, 0), 0.55), ((30, 15, -45), 0.45)],
    [((50, 0, 0), 0.6), ((70, 8, 28), 0.4)],
    [((42, 0, 0), 1.0)],
], ids=["grey 70 + red 30", "grey 55 + navy 45", "grey 60 + light wood 40", "dark grey L42"])
def test_a_multicoloured_or_dark_grey_reference_is_left_as_drawn(parts):
    """A grey majority beside a red, navy or wood part has a neutral median a*/b* and a neutral
    per-pixel median chroma; it is still several colours and is not lifted.  Nor is a dark grey
    under REFERENCE_LIFT_ABOVE."""
    Image = pytest.importorskip("PIL.Image")
    from src.hunyuan3d_wrapper.generate import lifted_reference

    image = Image.fromarray(_banded_body(parts))
    same, before, after = lifted_reference(image)
    assert same is image and after is None and before is not None


def test_a_single_mid_grey_band_is_still_lifted():
    Image = pytest.importorskip("PIL.Image")
    from src.hunyuan3d_wrapper.generate import REFERENCE_LIFT_TO, lifted_reference

    _, before, after = lifted_reference(Image.fromarray(_banded_body([((55, 0, 1), 1.0)])))
    assert abs(before - 55) < 1.0 and abs(after - REFERENCE_LIFT_TO) < 1.0


def test_a_dark_object_painted_at_its_drawn_lightness_is_not_drawn_again():
    """The gate measures the views against the reference as drawn (paint_surface passes the
    un-lifted image), so a dark body painted dark is accepted at the first drawing."""
    dark = (60, 40, 30)
    gate, conditions, calls, baked, seeds = _gated_views([[dark] * 6] * 3, dark)
    gate(["reference"], conditions)
    assert len(calls) == 1 and seeds == [] and gate.front_distance < 1.0
    # paint_surface's wiring (its models need the hunyuan3d env): the gate and reference.png get
    # the drawn image, the multiview model the lifted one, and a user's --input is never lifted.
    import inspect

    from src.hunyuan3d_wrapper.generate import paint_surface
    source = inspect.getsource(paint_surface)
    assert "_GatedViews(multiview, processor.bake_from_multiview, image, reseed)" in source
    assert "_replace_atomically(Path(args.reference), image.save)" in source
    assert 'painted_from.save(scratch / "input.png")' in source
    assert "(image, None, None) if args.input is not None else lifted_reference(image)" in source


def test_the_tilt_moves_every_axis_off_every_axis_and_turns_back():
    """A face square to any axis is no longer square-on to any axis camera after the tilt."""
    from src.hunyuan3d_wrapper.generate import FLAT_VIEW_TILT
    from src.hunyuan3d_wrapper.surface import tilt_rotation

    rotation = tilt_rotation(FLAT_VIEW_TILT)
    assert np.allclose(rotation @ rotation.T, np.eye(3)) and np.isclose(np.linalg.det(rotation), 1.0)
    turned = np.eye(3) @ rotation.T  # each axis direction, turned
    # Every turned axis is off all three axes by at least the tilt (no component is 1).
    assert np.abs(turned).max() < np.cos(np.radians(FLAT_VIEW_TILT)) + 1e-9
    points = np.random.default_rng(0).normal(size=(50, 3))
    assert np.allclose((points @ rotation.T) @ rotation, points)


def test_colours_come_back_onto_a_surface_painted_tilted():
    """paint_surface turns the painted copy back before the exact transfer, which then matches."""
    pytest.importorskip("scipy")
    from src.hunyuan3d_wrapper.surface import tilt_rotation, transfer_colours

    box, vertices, uv, texture = _seam_split_box()
    rotation = tilt_rotation(3.0)
    painted = vertices @ rotation.T
    with pytest.raises(RuntimeError, match="does not match"):
        transfer_colours(box, painted, uv, texture)
    colours = transfer_colours(box, painted @ rotation, uv, texture)
    right = box.vertices[:, 0] > 0
    assert (colours[right, :3] == (0, 0, 255)).all() and (colours[~right, :3] == (255, 0, 0)).all()


def test_a_view_seeing_one_flat_face_is_found(monkeypatch):
    """flat_views flags a view whose visible depth is float rounding, not one seeing two depths."""
    from src.hunyuan3d_wrapper import generate

    mask = np.zeros((8, 8), bool)
    mask[2:6, 2:6] = True
    flat = np.where(mask, 0.25 + np.float32(2.4e-7) * (np.arange(64).reshape(8, 8) % 2), 0.0)
    stepped = np.where(mask, 0.25, 0.0)
    stepped[2:4, 2:6] = 0.25 + 1.6e-3  # a small step in front of a large face still bakes
    depths = {(0, 0): flat, (0, 90): stepped, (90, 0): np.zeros((8, 8))}
    seen = []

    class Render:
        def set_mesh(self, vertices, faces):
            seen.append((vertices.dtype, faces.dtype))

    monkeypatch.setattr(generate, "view_depth", lambda render, resolution, elevation, azimuth: (
        depths[(elevation, azimuth)], depths[(elevation, azimuth)] > 0))
    found = generate.flat_views(Render(), np.zeros((3, 3)), np.array([[0, 1, 2]]), list(depths))
    # The empty view (nothing visible) is not called flat: there is nothing to bake from it.
    assert found == [0] and seen == [(np.float32, np.int32)]


def test_a_painting_that_never_reached_the_surface_is_refused():
    """The failures of an acceptance run (2026-10-10): every view baked 0 texels, the vertex
    colours came back all (0, 0, 0), and the tool still reported a painting."""
    from src.hunyuan3d_wrapper.generate import painting_failure

    rng = np.random.default_rng(0)
    wood = np.clip(np.array([196, 150, 100]) + rng.normal(0, 12, (500, 3)), 0, 255).astype(np.uint8)
    black = np.zeros((500, 4), np.uint8)
    black[:, 3] = 255
    assert "0 texels" in painting_failure(wood, baked_texels=0)
    assert "black (0, 0, 0)" in painting_failure(black)
    assert "black (0, 0, 0)" in painting_failure(black, baked_texels=1000, view_spread=9.0)
    flat_grey = np.full((500, 3), 128, np.uint8)
    assert "one colour" in painting_failure(flat_grey, baked_texels=1000, view_spread=9.0)
    # A real painting passes, and so does a uniform one painted from uniform views.
    assert painting_failure(wood, baked_texels=1000, view_spread=9.0) is None
    assert painting_failure(flat_grey, baked_texels=1000, view_spread=1.0) is None
    # A matte black painted near-black is a painting, not a failure (spread 1.25 in the batches).
    near_black = np.clip(rng.normal(3, 2, (500, 3)), 0, 255).astype(np.uint8)
    assert painting_failure(near_black, baked_texels=1000, view_spread=1.5) is None


def test_the_gate_records_the_bake_and_the_views_spread():
    """The gate keeps what the albedo bake covered (its first bake) and the baked views' spread."""
    brown, lighter, grey = (140, 90, 50), (165, 110, 65), (128, 128, 128)
    gate, conditions, calls, baked, seeds = _gated_views([[brown, lighter, brown, lighter, grey, brown]], brown)
    gate(["reference"], conditions)
    # Measured over the baked views only: the dropped grey view does not count.
    assert gate.kept == [0, 1, 2, 3, 5] and 5 < gate.view_spread < 20
    covered = np.zeros((4, 4, 1), bool)
    covered[:2] = True
    gate._bake = lambda *lists: ("texture", covered)
    gate.bake(list("abcdef"), list(range(6)), list(range(6)), [1] * 6)
    gate._bake = lambda *lists: ("texture", np.zeros((4, 4, 1), bool))  # the metal-roughness bake
    gate.bake(list("abcdef"), list(range(6)), list(range(6)), [1] * 6)
    assert gate.baked_texels == 8


def test_a_failed_painting_is_never_written():
    """paint_surface's wiring (its models need the hunyuan3d env): the painting is checked before
    it is written, and a body with flat axis views is painted tilted and turned back."""
    import inspect

    from src.hunyuan3d_wrapper.generate import paint_surface
    source = inspect.getsource(paint_surface)
    check, write = source.index("painting_failure("), source.index("_replace_atomically(output")
    assert check < write and 'raise RuntimeError(f"the painting failed: {failure}")' in source
    assert "flat_views(paint.render, copy.vertices, copy.faces, cameras)" in source
    # painted in the unit frame (paint_frame), before the tilt; turned, then scaled back
    frame = source.index("centre, half = paint_frame(copy.vertices)")
    assert frame < source.index("flat = flat_views(") < source.index("copy.export(")
    assert "back = back @ tilt" in source and "textured.vertices = back * half + centre" in source
    assert source.index("back = back @ tilt") < source.index("back * half + centre") \
        < source.index("transfer_colours_nearest(surface")
    # the UV check is installed for the paint and removed after it
    assert "paint.render.load_mesh = _checked_uv_load(paint.render)" in source
    assert 'paint.render.__dict__.pop("load_mesh", None)' in source


def test_a_small_body_is_painted_in_a_unit_frame():
    """xatlas counts faces under float32 epsilon (~1.2e-7) as degenerate and unwraps them to NaN:
    a body of a few centimetres in metres has every face that small (a held-out run, 2026-10-10).
    The paint frame takes any body to half-extent 1, and back exactly."""
    from src.hunyuan3d_wrapper.surface import paint_frame

    rng = np.random.default_rng(0)
    vertices = rng.random((200, 3)) * np.array([0.062, 0.026, 0.0023]) + np.array([0.5, -1.0, 2.0])
    centre, half = paint_frame(vertices)
    unit = (vertices - centre) / half
    assert np.isclose(np.abs(unit).max(), 1.0)
    assert np.allclose(unit.min(axis=0) + unit.max(axis=0), 0.0)
    assert np.allclose(unit * half + centre, vertices)
    assert paint_frame(np.zeros((3, 3)))[1] == 1.0


def test_the_plain_colour_is_read_from_the_texture_words(tmp_path):
    from src.hunyuan3d_wrapper.surface import PLAIN_COLOUR_WORDS, PLAIN_DEFAULT, has_colour, plain_colour

    assert plain_colour("red painted steel") == (PLAIN_COLOUR_WORDS["red"], "red")
    light, word = plain_colour("light yellow metal")
    assert word == "light yellow" and all(a >= b for a, b in zip(light, PLAIN_COLOUR_WORDS["yellow"]))
    dark, word = plain_colour("dark green plastic")
    assert word == "dark green" and all(a <= b for a, b in zip(dark, PLAIN_COLOUR_WORDS["green"]))
    assert plain_colour("brushed brass")[1] == "brass"
    # a per-part prompt's first colour; a neutral is moved off neutral, so it reads as painted
    assert plain_colour("grey steel plate, brown wood rail")[1] == "grey"
    (r, g, b), word = plain_colour("")
    assert word.startswith("mid grey") and (r, g) == PLAIN_DEFAULT[:2] and b != g
    trimesh = pytest.importorskip("trimesh")
    box = trimesh.creation.box()
    rgba = np.array([*plain_colour("grey")[0], 255], np.uint8)
    box.visual = trimesh.visual.ColorVisuals(box, vertex_colors=np.tile(rgba, (len(box.vertices), 1)))
    box.export(tmp_path / "grey.glb")
    assert has_colour(tmp_path / "grey.glb")


def test_non_finite_uvs_are_refused_before_the_inpainting_crashes():
    """Upstream's inpainting indexes the texture with every UV unchecked: a NaN UV is SIGSEGV."""
    from src.hunyuan3d_wrapper.generate import _checked_uv_load

    class Render:
        def load_mesh(self, mesh):
            self.vtx_uv = mesh

    load = _checked_uv_load(Render())
    load(np.full((4, 2), 0.5, np.float32))
    with pytest.raises(RuntimeError, match=r"cannot be painted: its UV unwrap failed \(4 of 4"):
        load(np.full((4, 2), np.nan, np.float32))
