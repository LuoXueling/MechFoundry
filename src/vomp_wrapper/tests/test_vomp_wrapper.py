import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from src.vomp_wrapper.artifact import (
    dense_grid_from_sparse, load_artifact, unit_scales, write_artifact,
)
from src.vomp_wrapper.inference import VompRuntime
from src.vomp_wrapper.visualize import (
    sample_vertices, surface_visualizations_current, write_surface_visualizations,
)


def test_sparse_voxels_become_a_complete_validated_grid(tmp_path):
    coords = np.asarray([[-0.4921875] * 3, [0.4921875] * 3], dtype=np.float32)
    props = np.asarray([[1.0e6, 0.2, 1000.0], [2.0e6, 0.4, 2000.0]], dtype=np.float32)
    results = {
        "voxel_coords_world": coords,
        "youngs_modulus": props[:, 0], "poisson_ratio": props[:, 1],
        "density": props[:, 2], "transform_center": np.zeros(3),
        "transform_scale": 1.0,
    }
    artifact = write_artifact(tmp_path / "material", results,
                              unit_system="SI (m-kg-s)", provenance={"test": True})
    loaded = load_artifact(artifact.root, verify=True)
    assert loaded.values.shape == (64, 64, 64, 3)
    assert loaded.valid.sum() == 2
    assert np.all(np.isfinite(loaded.values))
    assert loaded.values[0, 0, 0] == pytest.approx(props[0])
    assert loaded.values[-1, -1, -1] == pytest.approx(props[1])
    assert loaded.manifest["provenance"] == {"test": True}
    assert loaded.manifest["ranges"] == {
        "youngs_modulus": [1.0e6, 2.0e6],
        "poisson_ratio": pytest.approx([0.2, 0.4]),
        "density": [1000.0, 2000.0],
    }


@pytest.mark.parametrize("units,e_scale,rho_scale", [
    ("SI (m-kg-s)", 1.0, 1.0),
    ("SI-mm (mm-tonne-s)", 1e-6, 1e-12),
    ("US-inch (in-lbf-s)", 1 / 6894.757293168, 9.357254e-8),
])
def test_material_unit_conversion(units, e_scale, rho_scale):
    assert unit_scales(units) == pytest.approx((e_scale, rho_scale))
    grid, valid, _ = dense_grid_from_sparse(
        np.asarray([[0.0, 0.0, 0.0]]), np.asarray([[1e6, 0.3, 1000.0]]),
        center=np.zeros(3), scale=1.0, unit_system=units,
    )
    assert valid.sum() == 1
    assert grid[0, 0, 0] == pytest.approx([1e6 * e_scale, 0.3, 1000 * rho_scale])


@pytest.mark.parametrize("properties", [
    [[0.0, 0.3, 1000.0]], [[1.0, 0.5, 1000.0]], [[1.0, 0.3, -1.0]],
])
def test_invalid_physical_properties_are_rejected(properties):
    with pytest.raises(ValueError):
        dense_grid_from_sparse(
            np.zeros((1, 3)), np.asarray(properties), center=np.zeros(3),
            scale=1.0, unit_system="SI (m-kg-s)",
        )


def test_runtime_passes_absolute_weight_paths_without_conda_activation(
        tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    captured = {}

    class FakeModel:
        device = "cpu"

        def eval(self):
            captured["evaluated"] = True

    class FakeVomp:
        @classmethod
        def from_checkpoint(cls, **kwargs):
            captured.update(kwargs)
            return FakeModel()

    package = types.ModuleType("vomp")
    inference = types.ModuleType("vomp.inference")
    inference.Vomp = FakeVomp
    utilities = types.ModuleType("vomp.inference.utils")

    class FakeLazyDino:
        def __init__(self, **kwargs):
            captured["dino"] = kwargs

        def get_model(self):
            captured["dino_warmed"] = True

    utilities.LazyLoadDino = FakeLazyDino
    monkeypatch.setitem(sys.modules, "vomp", package)
    monkeypatch.setitem(sys.modules, "vomp.inference", inference)
    monkeypatch.setitem(sys.modules, "vomp.inference.utils", utilities)
    runtime = VompRuntime(tmp_path / "source", weights)
    runtime.preload()

    assert captured["evaluated"] is True
    assert captured["dino_warmed"] is True
    for name, leaf in (
        ("config_path", "inference.json"),
        ("geometry_checkpoint_dir", "geometry_transformer.pt"),
        ("matvae_checkpoint_dir", "matvae.safetensors"),
        ("normalization_params_path", "normalization_params.json"),
    ):
        assert captured[name] == str((weights / leaf).resolve())


def test_material_fields_are_projected_to_reviewable_surface_assets(tmp_path):
    import trimesh

    coords = np.asarray([[-0.49] * 3, [0.49] * 3], dtype=np.float32)
    results = {
        "voxel_coords_world": coords,
        "youngs_modulus": np.asarray([1e6, 2e6]),
        "poisson_ratio": np.asarray([0.2, 0.4]),
        "density": np.asarray([1000.0, 2000.0]),
        "transform_center": np.zeros(3), "transform_scale": 1.0,
    }
    artifact = write_artifact(
        tmp_path / "material", results, unit_system="SI (m-kg-s)"
    )
    surface = tmp_path / "box.glb"
    trimesh.creation.box(extents=(0.98, 0.98, 0.98)).export(surface)
    sampled = sample_vertices(artifact, trimesh.load(surface, force="mesh").vertices)
    assert sampled.shape[1] == 3
    outputs = write_surface_visualizations(artifact, surface)
    assert {path.name for path in outputs} == {
        "youngs_modulus.glb", "poisson_ratio.glb", "density.glb",
    }
    for path in outputs:
        assert path.is_file()
        metadata = json.loads(path.with_suffix(".glb.json").read_text())
        assert metadata["minimum"] <= metadata["maximum"]
        assert metadata["scale"] in ("linear", "logarithmic")
    assert surface_visualizations_current(artifact.root)



def _capture_inference(tmp_path, **runtime_kwargs):
    """Run `infer` as far as the call into VoMP and return its arguments."""
    captured = {}

    class Stop(Exception):
        """Raised once the arguments are in hand, before any work happens."""

    class FakeModel:
        device = "cpu"

        def get_mesh_materials(self, mesh_path, **kwargs):
            captured.update(kwargs)
            raise Stop

    tmp_path.mkdir(parents=True, exist_ok=True)
    surface = tmp_path / "surface.glb"
    __import__("trimesh").creation.box().export(surface)
    runtime = VompRuntime(tmp_path / "source", tmp_path / "weights",
                          blender=str(tmp_path / "blender"), **runtime_kwargs)
    runtime.model = FakeModel()
    with pytest.raises(Stop):
        runtime.infer(
            surface=surface, artifact=tmp_path / "material",
            cache=tmp_path / "cache", unit_system="SI (m-kg-s)",
            num_views=150, dino_batch_size=16, seed=0,
        )
    return captured


def test_each_renderer_is_asked_for_the_parallelism_it_can_use(tmp_path):
    """Rendering is where this call spent its time, and it was sequential.

    150 Cycles frames at 128 samples took 159s of a 207s run on an idle
    machine, and 21s a frame on a busy one.  Blender renders them in one
    process unless asked for more, and four took it to 71s while eight took 74s
    -- so four, not the core count, since this runs beside a solver that wants
    the rest of the box.  The rasteriser draws every view on one device in
    4.12s and comes in through the same single-process path, so it asks for
    one; asking for more would fan out a renderer that does not fan out.
    """
    blender = _capture_inference(tmp_path / "a", renderer="blender")
    assert blender["num_render_jobs"] == 4
    raster = _capture_inference(tmp_path / "b", renderer="raster")
    assert raster["num_render_jobs"] == 1

    # The regime the released model was trained and published on, unchanged
    # under either: the speed came from how the frames are drawn, not from
    # fewer of them.
    for captured in (blender, raster):
        assert captured["num_views"] == 150
        assert captured["max_voxels"] == 32768
        assert captured["query_points"] == "voxel_centers"


def test_the_two_renderers_do_not_share_a_cached_artifact(tmp_path):
    """They draw different pixels, so they are different inference requests."""
    import json

    surface = tmp_path / "surface.glb"
    __import__("trimesh").creation.box().export(surface)
    keys = []
    for renderer in ("raster", "blender"):
        runtime = VompRuntime(tmp_path / "source", tmp_path / "weights",
                              blender=str(tmp_path / "blender"), renderer=renderer)

        class FakeModel:
            device = "cpu"

            def get_mesh_materials(self, mesh_path, **kwargs):
                raise RuntimeError("stop")

        runtime.model = FakeModel()
        with pytest.raises(RuntimeError):
            runtime.infer(
                surface=surface, artifact=tmp_path / f"material_{renderer}",
                cache=tmp_path / "cache", unit_system="SI (m-kg-s)",
                num_views=150, dino_batch_size=16, seed=0,
            )
        keys.append(renderer)
    jobs = sorted(p.name for p in (tmp_path / "cache").iterdir())
    assert len(jobs) == 2, f"one job directory per renderer, got {jobs}"


def test_a_blender_that_is_named_relatively_is_found_from_the_session_directory(
        tmp_path, monkeypatch):
    """The environment ships a relative BLENDER_BIN; inference runs elsewhere.

    `./blender-3.0.1-linux-x64/blender` resolves only for a process whose
    working directory is the VoMP root, and inference runs from the session
    directory, where VoMP reported the path as missing without saying that the
    environment, not the bundle, had supplied it.
    """
    from src.vomp_wrapper.availability import settings

    root = tmp_path / "VoMP"
    bundled = root / "blender-3.0.1-linux-x64" / "blender"
    bundled.parent.mkdir(parents=True)
    bundled.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("VOMP_ROOT", str(root))
    monkeypatch.delenv("VOMP_BLENDER", raising=False)

    monkeypatch.setenv("BLENDER_BIN", "./blender-3.0.1-linux-x64/blender")
    assert settings()[3] == str(bundled.resolve())

    # A setting that names nothing gives way to the bundle rather than
    # travelling on to fail inside VoMP.
    monkeypatch.setenv("BLENDER_BIN", "/nowhere/blender")
    assert settings()[3] == str(bundled)

    # An absolute setting that exists is still what the operator asked for.
    elsewhere = tmp_path / "own-blender"
    elsewhere.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("BLENDER_BIN", str(elsewhere))
    assert settings()[3] == str(elsewhere)


def test_the_feature_mean_is_taken_where_the_features_already_are():
    """18.8 GiB does not need to visit the host to be averaged.

    The feature stage builds (150, 32768, 1024) float32, copies it over and
    means it with numpy -- 11.23s and 2.89s measured, against 0.09s for the
    same mean on the device.  The stand-in answers the two calls the stage
    makes of that block and nothing else, so the arithmetic is unchanged: what
    differs is the order the sum is accumulated in, which lands well inside the
    float16 the result is stored as.
    """
    import numpy as np
    import torch

    from src.vomp_wrapper.patches import LAZY_MEAN_THRESHOLD, _MeanWhereItLies

    if not torch.cuda.is_available():
        pytest.skip("the stand-in only stands in for device tensors")
    block = torch.randn(32, 512, 256, device="cuda")
    upstream = np.mean(block.cpu().numpy(), axis=0).astype(np.float16)
    patched = np.mean(_MeanWhereItLies(block), axis=0).astype(np.float16)
    assert patched.shape == upstream.shape
    assert np.abs(patched.astype(np.float32) - upstream.astype(np.float32)).max() < 1.0e-3

    # The caller reshapes the grid_sample result before it means it.
    reshaped = _MeanWhereItLies(torch.randn(4, 256, 1, 32, device="cuda"))
    assert np.mean(reshaped.squeeze(2).permute(0, 2, 1), axis=0).shape == (32, 256)

    # Anything that wants the whole block still gets it, rather than failing.
    assert np.asarray(_MeanWhereItLies(block)).shape == (32, 512, 256)
    # Only the block worth intercepting is intercepted.
    assert LAZY_MEAN_THRESHOLD == 1 << 30


def test_only_the_oversized_sampling_is_intercepted(monkeypatch):
    """A grid_sample that is not the feature block passes through untouched."""
    import types

    import torch

    from src.vomp_wrapper.patches import _MeanWhereItLies, patch_feature_aggregation

    module = types.ModuleType("fake_vomp")
    module.np = __import__("numpy")
    module.F = None
    assert patch_feature_aggregation(module) is True
    assert patch_feature_aggregation(module) is False, "applied once, not twice"

    # The compressed write was 2.27s for 64 MiB that np.load reads either way.
    assert module.np.savez_compressed is module.np.savez

    if not torch.cuda.is_available():
        pytest.skip("the threshold is measured on device tensors")
    small = module.F.grid_sample(
        torch.zeros(1, 4, 8, 8, device="cuda"),
        torch.zeros(1, 1, 16, 2, device="cuda"),
        align_corners=False,
    )
    assert isinstance(small, torch.Tensor), "a small sampling is not stood in for"
    assert not isinstance(small, _MeanWhereItLies)
    # Everything the module asks of torch.nn.functional still reaches it.
    assert module.F.normalize(torch.ones(2, 3)).shape == (2, 3)


def test_the_view_encoder_runs_half_and_hands_back_float():
    """DINOv2 is 11.42s in float32 here, 2.19s under autocast, and it shows.

    The paper gives its whole DINO computation 0.86s on an A100, which is this
    regime on faster silicon.  Only the encoder's arithmetic changes: end to
    end against the float32 run on the same renders, the materials move by
    3.7e-4 at the median with a correlation of 0.9989.  What it returns is cast
    back, because everything downstream -- the sampling, the mean, the cached
    features -- was written for float32.
    """
    import torch

    from src.vomp_wrapper.patches import patch_dino_precision

    class FakeDino:
        def __init__(self):
            self.dtypes = []

        def forward_features(self, batch):
            # Whatever the encoder computes in is what autocast chose for it.
            weight = torch.ones(4, 4, device=batch.device)
            self.dtypes.append((batch @ weight).dtype)
            return {"x_norm_patchtokens": (batch @ weight), "meta": "unchanged"}

    if not torch.cuda.is_available():
        pytest.skip("autocast is asked for on cuda")
    dino = FakeDino()
    assert patch_dino_precision(dino) is True
    assert patch_dino_precision(dino) is False, "wrapped once, not twice"
    out = dino.forward_features(torch.ones(4, 4, device="cuda"))
    assert dino.dtypes == [torch.float16], "the encoder ran half"
    assert out["x_norm_patchtokens"].dtype is torch.float32, "and handed back float"
    assert out["meta"] == "unchanged"


def test_a_render_is_not_decoded_again_by_the_stage_that_just_wrote_it(tmp_path):
    """0.44s of a 4.90s prediction went on re-reading this process's own PNGs.

    The renderer keeps what it wrote and `Image.open` answers from it, so the
    feature stage is handed the pixels that are on disk rather than decoding
    them a second time.  Anything the renderer did not write -- a Blender run,
    a rerun, a path from somewhere else -- still goes to Pillow.
    """
    import types

    import numpy as np
    from PIL import Image

    from src.vomp_wrapper import raster
    from src.vomp_wrapper.patches import patch_image_cache

    module = types.ModuleType("fake_vomp")
    assert patch_image_cache(module) is True
    assert patch_image_cache(module) is False, "applied once, not twice"

    written = tmp_path / "000.png"
    picture = Image.fromarray(
        np.full((8, 8, 4), 200, dtype=np.uint8), mode="RGBA"
    )
    picture.save(written)
    raster.RENDERED.clear()
    raster.RENDERED[str(written.resolve())] = picture
    assert module.Image.open(written) is picture, "the kept render is handed back"
    # And it is the render that is on disk, so either way gives the same pixels.
    assert np.array_equal(np.array(module.Image.open(written)),
                          np.array(Image.open(written)))

    elsewhere = tmp_path / "other.png"
    picture.save(elsewhere)
    assert module.Image.open(elsewhere) is not picture
    assert np.array(module.Image.open(elsewhere)).shape == (8, 8, 4)
    raster.RENDERED.clear()
    assert module.Image.open(written) is not picture, "cleared between renders"


def test_the_sparse_backend_is_warmed_before_a_prediction_waits_on_it():
    """The first prediction in a process cost 9.75s and the next 5.70s.

    The difference is spconv generating and compiling its kernels, which does
    not depend on the object, so the worker pays it while loading rather than
    on the first button press.  Warming is best effort: a model that cannot be
    warmed still answers.
    """
    from src.vomp_wrapper.patches import install, warm_inference

    calls = {}

    class FakeModel:
        device = "cpu"

        def predict_materials(self, coordinates, features):
            calls["shape"] = (tuple(coordinates.shape), tuple(features.shape))

    assert warm_inference(FakeModel()) is True
    assert calls["shape"] == ((64, 4), (64, 1024)), "voxels as [batch, x, y, z]"

    class Broken:
        device = "cpu"

        def predict_materials(self, *_):
            raise RuntimeError("no kernels today")

    assert warm_inference(Broken()) is False
    # A worker that could not warm up is still a worker.
    assert "warm_inference" not in install(
        Broken(), rasterize=False, fast_features=False, half_encoder=False
    )


def test_the_renderer_finds_its_compiler_without_the_environment_activated(monkeypatch):
    """The plugin build needs ninja, and the workflow does not activate.

    nvdiffrast compiles a C++ extension through torch on first use, and torch
    writes and runs a ninja file every time, cached build or not.  pip puts
    ninja beside the interpreter; the monitor launches this environment's
    python directly because activation costs ~1.8s a call, so PATH does not
    carry that directory.  A prediction died on it -- "Ninja is required to
    load C++ extensions" -- from inside the render, after the model had
    loaded.
    """
    import os
    import sys

    from src.vomp_wrapper.raster import ensure_toolchain_on_path

    interpreter = str(Path(sys.executable).resolve().parent)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    ensure_toolchain_on_path()
    assert os.environ["PATH"].split(os.pathsep)[0] == interpreter
    assert "/usr/bin" in os.environ["PATH"], "what was there is still there"

    # Asked twice, it does not stack up.
    ensure_toolchain_on_path()
    assert os.environ["PATH"].split(os.pathsep).count(interpreter) == 1


def test_the_renderer_draws_a_surface_in_its_own_colour():
    """Blender renders a mesh with its material, so a coloured one is not grey."""
    import trimesh
    from PIL import Image
    from src.vomp_wrapper import raster

    box = trimesh.creation.box()
    assert raster.surface_albedo(box) is None

    # COLOR_0 as the Hunyuan3D wrapper writes it: sRGB bytes, decoded to linear.
    box.visual = trimesh.visual.ColorVisuals(
        box, vertex_colors=np.tile([[188, 0, 255, 255]], (len(box.vertices), 1)))
    kind, colours = raster.surface_albedo(box)
    assert kind == "vertex"
    np.testing.assert_allclose(colours[0], [0.5029, 0.0, 1.0], atol=1e-3)

    # A base-colour texture, flipped so its first row is v = 0 and decoded.
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0] = 255  # the top row, which trimesh puts at v = 1
    textured = trimesh.Trimesh(box.vertices, box.faces, visual=trimesh.visual.TextureVisuals(
        uv=np.zeros((len(box.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(image)),
    ), process=False)
    kind, uv, texture = raster.surface_albedo(textured)
    assert kind == "texture" and uv.shape == (len(box.vertices), 2)
    assert texture[0].max() == 0.0 and texture[-1].min() == 1.0


def test_a_texture_too_big_for_pil_still_colours_its_part(tmp_path, monkeypatch):
    """GVM's Jobba sofa has 16k textures, over PIL's decompression-bomb limit.

    trimesh then dropped the texture silently and the black leather rendered
    in its white Kd.  Blender loads it; so must we, shrunk to what 512 px views
    can show.
    """
    import trimesh
    from PIL import Image
    from src.vomp_wrapper import raster

    Image.fromarray(np.full((64, 64, 3), 20, dtype=np.uint8)).save(tmp_path / "dark.png")
    (tmp_path / "model.mtl").write_text("newmtl m\nKd 0.8 0.8 0.8\nmap_Kd dark.png\n")
    box = trimesh.creation.box()
    uv = "\n".join(f"vt {u} {v}" for u, v in np.random.default_rng(0).random((len(box.vertices), 2)))
    (tmp_path / "model.obj").write_text(
        "mtllib model.mtl\nusemtl m\n"
        + "\n".join(f"v {x} {y} {z}" for x, y, z in box.vertices) + "\n" + uv + "\n"
        + "\n".join(f"f {a}/{a} {b}/{b} {c}/{c}" for a, b, c in box.faces + 1) + "\n")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)  # 64x64 is now a "bomb"
    monkeypatch.setattr(raster, "MAX_TEXTURE_SIZE", 16)

    surface = raster.SceneSurface(tmp_path / "model.obj")
    assert (surface.face_texture == 0).all()
    assert surface.textures[0].shape == (16, 16, 3)
    np.testing.assert_allclose(surface.textures[0], raster.srgb_to_linear(20 / 255), atol=1e-6)
    assert Image.MAX_IMAGE_PIXELS == 100, "the limit is restored for everyone else"


def test_seams_do_not_crease_the_shading():
    import trimesh
    from src.vomp_wrapper import raster

    sphere = trimesh.creation.icosphere(subdivisions=3)
    # Split every face apart, as UV seams would everywhere.
    split = trimesh.Trimesh(sphere.vertices[sphere.faces].reshape(-1, 3),
                            np.arange(3 * len(sphere.faces)).reshape(-1, 3), process=False)
    normals = raster.welded_normals(split)
    radial = split.vertices / np.linalg.norm(split.vertices, axis=1, keepdims=True)
    assert np.einsum("ij,ij->i", normals, radial).min() > 0.99


def test_filmic_matches_blender_at_middle_grey():
    torch = pytest.importorskip("torch")
    from src.vomp_wrapper import raster

    try:
        raster._filmic_tables("cpu")
    except FileNotFoundError:
        pytest.skip("VoMP's Blender is not installed")
    grey = raster.filmic(torch.tensor([[0.0, 0.0, 0.0], [0.18, 0.18, 0.18], [16.0, 16.0, 16.0]]))
    # Filmic puts scene-referred 0.18 at display 0.5 and clips far above white.
    np.testing.assert_allclose(grey.numpy(), [[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]], atol=2e-3)


def test_voxels_land_where_their_features_were_sampled(tmp_path):
    """VoMP's reported voxels come back to the lattice points they were lifted at.

    VoMP reports each voxel half a cell above its lattice point, in the file's
    own frame (the frame it was trained in); the artifact must put each one in
    the cell centred on its point, and the solver's sampler must read it back
    there.
    """
    import trimesh
    pytest.importorskip("vomp")
    from src.vomp_wrapper.artifact import GRID_SIZE
    from src.vomp_wrapper.inference import restore_frame

    box = trimesh.creation.box(extents=(2.0, 1.0, 0.5))
    box.apply_translation((3.0, -1.0, 0.25))
    surface = tmp_path / "box.glb"
    box.export(surface)
    # Lattice points of VoMP's unit box, distinct in every axis.
    lattice = np.array([[-8, 3, 1], [5, -2, 0], [0, 0, -1]], dtype=np.float64) / GRID_SIZE
    reported = lattice + 0.5 / GRID_SIZE
    properties = np.array([[1e6, 0.2, 900.0], [2e6, 0.3, 1000.0], [3e6, 0.4, 1100.0]])
    restored = restore_frame({"voxel_coords_world": reported}, surface)
    np.testing.assert_allclose(restored["transform_scale"], 2.0)
    np.testing.assert_allclose(restored["voxel_coords_world"],
                               lattice * 2.0 + np.array([3.0, -1.0, 0.25]), atol=1e-12)

    artifact = write_artifact(tmp_path / "material", {
        **restored, "youngs_modulus": properties[:, 0],
        "poisson_ratio": properties[:, 1], "density": properties[:, 2],
    }, unit_system="SI (m-kg-s)")
    expected = np.rint((lattice + 0.5) * GRID_SIZE).astype(int)
    assert set(map(tuple, np.argwhere(artifact.valid))) == set(map(tuple, expected))
    # Each point reads its own voxel back, in the preview and in the solver.
    points = restored["voxel_coords_world"]
    np.testing.assert_allclose(sample_vertices(artifact, points), properties.astype(np.float32))
    torch = pytest.importorskip("torch")
    import importlib.util
    # By path: the solver package's __init__ needs the solver environment.
    source = Path(__file__).resolve().parents[2] / "warp_wrapper" / "core" / "material_field.py"
    spec = importlib.util.spec_from_file_location("material_field", source)
    material_field = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = material_field  # its dataclass looks itself up there
    spec.loader.exec_module(material_field)
    sampled = material_field.sample_material_artifact(artifact.root, torch.as_tensor(points), non_blocking=False)
    np.testing.assert_allclose(sampled.values.numpy(), properties.astype(np.float32))


def test_voxels_are_projected_as_vomp_was_trained():
    """No patch turns VoMP's voxels: its training features were lifted unturned."""
    from src.vomp_wrapper import patches

    class Model:
        def _voxelize_mesh(self, mesh_path, output_dir, **kwargs):
            return "untouched"

    model = Model()
    patches.install(model, rasterize=False, fast_features=False, half_encoder=False, warm=False)
    assert model._voxelize_mesh("a.glb", "out") == "untouched"


def test_a_gltf_surface_is_shown_to_vomp_z_up_and_its_voxels_come_back(tmp_path):
    """VoMP was trained on Z-up assets; a Y-up glTF goes in turned and comes back unturned."""
    import trimesh
    pytest.importorskip("vomp")
    from src.vomp_wrapper.artifact import GRID_SIZE
    from src.vomp_wrapper.inference import present_z_up, restore_frame, to_z_up

    tall = trimesh.creation.box(extents=(0.5, 2.0, 1.0))           # Y-up: tallest along y
    tall.apply_translation((1.0, 2.0, -3.0))
    surface = tmp_path / "tall.glb"
    tall.export(surface)
    presented = present_z_up(surface, tmp_path / "job")
    turned = trimesh.load(presented, force="mesh")
    assert np.argmax(np.ptp(turned.vertices, 0)) == 2, "tallest along z once presented"

    # A voxel VoMP reports on the presented (Z-up) copy lands back on the
    # lattice point it came from, in the surface's own frame.
    lattice = np.array([[3, -7, 12], [0, 5, -2]], dtype=np.float64) / GRID_SIZE
    results = {"voxel_coords_world": lattice + 0.5 / GRID_SIZE}
    restored = restore_frame(results, surface, presented)
    lo, hi = turned.vertices.min(0), turned.vertices.max(0)
    expected_z_up = lattice * (hi - lo).max() + (lo + hi) / 2
    np.testing.assert_allclose(to_z_up(restored["voxel_coords_world"]), expected_z_up, atol=1e-9)
    np.testing.assert_allclose(to_z_up(restored["transform_center"][None])[0], (lo + hi) / 2, atol=1e-9)
    # An OBJ is left as it is.
    obj = tmp_path / "part.obj"; tall.export(obj)
    assert present_z_up(obj, tmp_path / "job2") == obj
