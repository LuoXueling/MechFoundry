"""Runtime repairs to VoMP, applied to a loaded model rather than to its source.

``availability.py`` refuses to run when the checkout's revision differs from
``PINNED_SOURCE_REVISION``, so the tree under ``src/dependency/VoMP`` is not
ours to edit.  Everything here is therefore installed onto the model instance
and the module it came from, the way ``partfield_wrapper`` patches its own
upstream, and every patch reports itself so a receipt can say what ran.
"""

from __future__ import annotations

import types
from pathlib import Path

# Above this, a sampled-feature block is the (views, points, channels) one the
# feature stage means over -- 18.8 GiB at the shipped settings.  Below it, a
# grid_sample is something else and is left alone.
LAZY_MEAN_THRESHOLD = 1 << 30


class _MeanWhereItLies:
    """The feature block, kept on the device until the mean has been taken.

    ``_extract_dino_features`` builds (150, 32768, 1024) float32 -- 18.8 GiB --
    copies it to the host and means it with numpy: 11.23s and 2.89s measured,
    against 0.09s for the same mean on the device.  The block is used for
    nothing else, and ``np.mean`` asks a non-array for its own ``mean``, so
    standing in for it here is enough to move the work without touching the
    lines that do it.
    """

    def __init__(self, tensor):
        self._tensor = tensor

    @property
    def shape(self):
        return tuple(self._tensor.shape)

    @property
    def dtype(self):
        return self._tensor.dtype

    def squeeze(self, dim):
        return _MeanWhereItLies(self._tensor.squeeze(dim))

    def permute(self, *dims):
        return _MeanWhereItLies(self._tensor.permute(*dims))

    def cpu(self):
        return self

    def numpy(self):
        return self

    def mean(self, axis=None, dtype=None, out=None, **kwargs):
        reduced = self._tensor.float().mean(dim=0 if axis is None else axis)
        result = reduced.cpu().numpy()
        return result if dtype is None else result.astype(dtype)

    def __array__(self, dtype=None):
        # Whatever asks for the whole block in host memory gets it, so a caller
        # this stand-in did not anticipate still works -- slowly, as before.
        result = self._tensor.cpu().numpy()
        return result if dtype is None else result.astype(dtype)


class _Delegating:
    """A module stand-in that forwards everything it is not asked to change."""

    def __init__(self, wrapped, **overrides):
        self.__dict__["_wrapped"] = wrapped
        self.__dict__["_overrides"] = overrides

    def __getattr__(self, name):
        override = self.__dict__["_overrides"].get(name)
        return override if override is not None else getattr(self._wrapped, name)


def _lazy_grid_sample(original):
    def grid_sample(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.is_cuda and result.element_size() * result.nelement() >= LAZY_MEAN_THRESHOLD:
            return _MeanWhereItLies(result)
        return result
    return grid_sample


def patch_feature_aggregation(module) -> bool:
    """Take the feature mean on the device, and stop recompressing the cache.

    The cache is 64 MiB of float16 that compresses to 55 MiB in 2.27s, against
    0.08s for writing it as it stands; it is read back by ``np.load``, which
    does not care which it was.
    """
    import torch.nn.functional as functional

    if getattr(module, "_fem_features_patched", False):
        return False
    module.F = _Delegating(functional, grid_sample=_lazy_grid_sample(functional.grid_sample))
    module.np = _Delegating(module.np, savez_compressed=module.np.savez)
    module._fem_features_patched = True
    return True


def _dino_image_size(model, fallback: int = 518) -> int:
    """The size the feature stage resamples every render to, read from VoMP.

    Taken from the signature rather than written down, so a released model that
    reads a different size still gets renders at that size instead of renders
    this wrapper has quietly resampled to the wrong one.
    """
    import inspect

    try:
        parameter = inspect.signature(model.get_mesh_materials).parameters["image_size"]
    except (AttributeError, KeyError, TypeError, ValueError):
        return fallback
    return int(parameter.default) if isinstance(parameter.default, int) else fallback


def patch_mesh_renderer(model, image_size_override: int | None = None) -> bool:
    """Rasterise the views instead of path-tracing them in Blender.

    Blender renders VoMP's 150 mesh views in 159s, or 71s across four
    processes; the paper's own budget spends 2.11s on rendering because it
    rasterises Gaussians, which is a path a mesh does not have upstream.  The
    scene is one matte grey object under three fixed lights, and
    ``raster.render_views`` reproduces it to a silhouette IoU of 0.9999 and a
    per-pixel correlation of 0.954 in 4.12s.
    """
    from src.vomp_wrapper.raster import render_views

    wanted = _dino_image_size(model)

    def _render_views_sequential(self, views, mesh_path, renders_dir, image_size,
                                 blender_path, use_gpu, gpu_device):
        return render_views(
            mesh_path, views, renders_dir,
            image_size=image_size_override or image_size,
            output_size=wanted,
        )

    model._render_views_sequential = types.MethodType(_render_views_sequential, model)
    return True


def patch_image_cache(module) -> bool:
    """Let the feature stage take the renders it was about to decode.

    It opens each of the 150 PNGs this process wrote seconds earlier and
    decodes it again, 0.44s of a 4.90s prediction.  ``Image.open`` here answers
    from what the renderer kept, and falls through to Pillow for any path it
    does not recognise -- a Blender render, a rerun, anything else.  The pixels
    are the ones that were written, so the prediction is unchanged: checked
    bit for bit against the decoding path.
    """
    from PIL import Image as pillow

    from src.vomp_wrapper.raster import RENDERED

    if getattr(module, "_fem_images_patched", False):
        return False
    original = pillow.open

    def open_image(path, *args, **kwargs):
        if not args and not kwargs:
            cached = RENDERED.get(str(Path(path).resolve()))
            if cached is not None:
                return cached
        return original(path, *args, **kwargs)

    module.Image = _Delegating(pillow, open=open_image)
    module._fem_images_patched = True
    return True


def patch_dino_precision(dino) -> bool:
    """Run the view encoder in half precision, as its own benchmark implies.

    DINOv2 ViT-L/14 over 150 views at 518 costs 11.42s here in float32, 7.16s
    with TF32 allowed, and 2.19s under autocast -- and the paper's budget gives
    the whole DINO computation 0.86s on an A100, which is that same regime on
    faster silicon.  Only the encoder's arithmetic changes: what it returns is
    cast back, so the sampling and the mean downstream see float32 exactly as
    before.
    """
    import torch

    if getattr(dino, "_fem_autocast", False):
        return False
    original = dino.forward_features

    def forward_features(*args, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            features = original(*args, **kwargs)
        if isinstance(features, dict):
            return {
                name: value.float() if torch.is_tensor(value) else value
                for name, value in features.items()
            }
        return features.float() if torch.is_tensor(features) else features

    dino.forward_features = forward_features
    dino._fem_autocast = True
    return True


# The channel count DINOv2 ViT-L/14 hands the transformer, and enough voxels
# to exercise a sparse convolution without being worth timing.
FEATURE_CHANNELS = 1024
WARM_VOXELS = 64


def warm_inference(model) -> bool:
    """Pay the sparse backend's code generation before a user is waiting on it.

    The first prediction in a process costs 9.75s and every one after it 5.70s.
    The difference is spconv generating and building the C++ for its kernels --
    186882 calls into pccm's code model on that first pass -- plus CUDA
    compiling them.  None of it depends on the object, so it belongs beside the
    DINO warm-up rather than inside the first button press.  Best effort: a
    worker that cannot warm up still answers, just slower once.
    """
    import torch

    try:
        coordinates = torch.stack([
            torch.zeros(WARM_VOXELS, dtype=torch.int32),
            *(torch.randint(0, 64, (WARM_VOXELS,), dtype=torch.int32) for _ in range(3)),
        ], dim=1).to(model.device)
        features = torch.zeros(
            WARM_VOXELS, FEATURE_CHANNELS, dtype=torch.float32, device=model.device
        )
        with torch.inference_mode():
            model.predict_materials(coordinates, features)
        return True
    except Exception:
        return False


def warm_rasterizer() -> bool:
    """Build nvdiffrast's plugin now rather than inside the first prediction.

    It is a C++ extension torch compiles on first use, and the compile needs
    ninja on PATH -- which is what the workflow's direct interpreter launch
    does not provide until ``ensure_toolchain_on_path`` has run.  Doing it here
    means the build, and any reason it cannot happen, lands while the worker is
    loading.
    """
    try:
        from src.vomp_wrapper.raster import build_context

        build_context()
        return True
    except Exception:
        return False


# Blender's glTF and OBJ importers turn a Y-up file Z-up on the way in; its
# PLY and STL importers leave it as it is.  VoMP projects its voxels, which
# stay in the file's own frame, through cameras placed around the turned
# object.  That is not corrected here: its training features were lifted the
# same way (render_usd.py exports each asset to OBJ for Blender and voxelizes
# it in the USD's frame), and the model expects it.  Projecting the voxels in
# the turned frame instead moved a GVM test chair's features to a median
# cosine of 0.92 from the dataset's own (0.98 as upstream does it) and its
# modulus error from x1.7 to x6.3.
TURNED_FORMATS = (".glb", ".gltf", ".obj")


def install(model, *, rasterize: bool = True, fast_features: bool = True,
            half_encoder: bool = True, warm: bool = True, dino=None) -> list[str]:
    """Apply the patches this model should run with, and name the ones applied."""
    applied = []
    if fast_features:
        import sys

        module = sys.modules.get(type(model).__module__)
        if module is not None and patch_feature_aggregation(module):
            applied.append("feature_aggregation")
        if module is not None and rasterize and patch_image_cache(module):
            applied.append("image_cache")
    if rasterize and patch_mesh_renderer(model):
        applied.append("mesh_renderer")
    if half_encoder and dino is not None and patch_dino_precision(dino):
        applied.append("half_encoder")
    if warm and warm_inference(model):
        applied.append("warm_inference")
    if rasterize and warm_rasterizer():
        applied.append("warm_rasterizer")
    return applied
