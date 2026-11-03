#!/usr/bin/env python
"""Image -> watertight surface with vertex colours, using Hunyuan3D-2.1.

The shape stage (a DiT flow model) turns the image into a mesh, which is
cleaned up as TripoSG's is: floaters removed, reduced to the face budget with
upstream's own reducer, welded, the largest component kept, single-triangle
holes closed, and checked watertight.  It is then fitted into the [-1, 1] box the other
generators use.  The paint stage (multiview PBR diffusion) textures that same
surface without remeshing it; its UV-mapped albedo is carried back onto the
watertight surface as vertex colours (``COLOR_0``), which survive every later
step of the workflow where UV seams would not.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from src.hunyuan3d_wrapper.availability import (
    CONTROLNET_REPO_ID, REALESRGAN, REFERENCE_REPO_ID, REFERENCE_VAE_REPO_ID, REPO_ID, U2NET, settings,
)
from src.hunyuan3d_wrapper.surface import (
    DEFAULT_FACES, clean_surface, fit_to_frame, paint_frame, texture_of, tilt_rotation,
    transfer_colours, transfer_colours_nearest,
)

# Appended to the user's words (which come first, so SDXL's 77-token limit
# never cuts them) when a reference image is drawn from text.  The paint stage
# takes the reference's body colour as the albedo and removes its lighting, so
# the drawing should show that colour plainly: shadows, glossy highlights and
# low-key lighting in the drawing came back as a near-black body.  No "white
# background": SDXL then paints recesses and gaps in the body white as well.
REFERENCE_STYLE = ("one object, its surface colour shown clearly, bright even studio lighting, matte, "
                   "no shadows, no reflections, plain background, product photo")
REFERENCE_NEGATIVE = ("background scenery, text, watermark, multiple objects, shadows, dark, underexposed, "
                      "dramatic lighting, glossy highlights, reflections, cropped, blurry")
REFERENCE_SIZE = 1024
# Where the reference is drawn from: (elevation, azimuth) in the paint stage's
# camera.  The paint stage reads the reference for its look, not as one of its
# own views, so it need not be the front view; seen from above and aside, the
# drawing shows the top and a side as well, where a front view showed no top.
REFERENCE_VIEW = (25.0, 30.0)
# The drawing is framed on the silhouette with this margin (of its side, each
# way), so a thin or small body fills it rather than a sliver of the frame.
REFERENCE_FRAME = 0.08


class Hunyuan3DRuntime:
    """Both Hunyuan3D-2.1 stages loaded on the GPU, kept resident by the server."""

    def __init__(self, texture: bool = True, paint_resolution: int = 512, views: int = 6):
        self.environment, self.root, self.weights = settings()
        self.texture = texture
        self.paint_resolution = paint_resolution
        self.views = views
        self.shape = None
        self.paint = None
        self.background = None
        self.upscaler = None
        self.reference = None
        self.flashvdm = False
        self._fixed = False

    def set_flashvdm(self, enabled: bool):
        """Upstream's faster volume decoding for the shape stage (its app's --enable_flashvdm)."""
        if enabled != self.flashvdm:
            self.shape.enable_flashvdm(enabled=enabled, mc_algo="mc")
            self.flashvdm = enabled

    def set_texture_size(self, texture_size: int):
        """Bake into a ``texture_size`` square texture, rendering views at up to 2048.

        Only the renderer is rebuilt; the paint models stay loaded.
        """
        config = self.paint.config
        render_size = min(texture_size, 2048)
        if (config.texture_size, config.render_size) == (texture_size, render_size):
            return
        from DifferentiableRenderer.MeshRender import MeshRender
        from utils.pipeline_utils import ViewProcessor

        config.texture_size, config.render_size = texture_size, render_size
        self.paint.render = MeshRender(
            default_resolution=render_size, texture_size=texture_size,
            bake_mode=config.bake_mode, raster_mode=config.raster_mode,
        )
        self.paint.view_processor = ViewProcessor(config, self.paint.render)

    def set_upscale(self, enabled: bool):
        """Whether RealESRGAN enlarges each painted view 4x before baking."""
        self.paint.models["super_model"] = self.upscaler if enabled else (lambda image: image)

    def _prepare(self):
        # Every weight comes from the cache under the checkout; nothing is fetched.
        os.environ["HF_HOME"] = os.fspath(self.weights)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("U2NET_HOME", os.fspath(self.root / U2NET.parent))
        for folder in ("hy3dshape", "hy3dpaint", ""):
            path = os.fspath(self.root / folder)
            if path not in sys.path:
                sys.path.insert(0, path)
        # Both stages import torchvision's removed functional_tensor (the paint
        # stage through RealESRGAN), so the fix goes in before either loads.
        if not self._fixed:
            try:
                from torchvision_fix import apply_fix
                apply_fix()
            except ImportError:
                pass
            self._fixed = True

    def load_background(self):
        """Upstream's background remover; the shape stage loads it, a paint only on demand."""
        if self.background is None:
            self._prepare()
            from hy3dshape.rembg import BackgroundRemover

            self.background = BackgroundRemover()
        return self.background

    def remove_background(self, image):
        """``image`` cut out with upstream's background remover."""
        return self.load_background()(image)

    def preload(self, texture: bool | None = None, shape: bool = True):
        """Load the shape stage (unless ``shape`` is False) and, if ``texture``, the paint stage.

        Painting an existing surface needs no shape stage: loading it anyway
        cost 31 of the 80 s a cold paint took, and its GPU memory.
        """
        texture = self.texture if texture is None else texture
        self._prepare()
        if shape and self.shape is None:
            from huggingface_hub import snapshot_download
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

            snapshot = snapshot_download(REPO_ID, local_files_only=True)
            self.shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(snapshot)
            # Every image the shape stage takes is cut out first.
            self.load_background()
        if texture and self.paint is None:
            from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

            config = Hunyuan3DPaintConfig(self.views, self.paint_resolution)
            config.realesrgan_ckpt_path = os.fspath(self.root / REALESRGAN)
            config.multiview_cfg_path = os.fspath(self.root / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml")
            config.custom_pipeline = os.fspath(self.root / "hy3dpaint" / "hunyuanpaintpbr")
            self.paint = Hunyuan3DPaintPipeline(config)
            self.upscaler = self.paint.models["super_model"]
        return self

    def preload_reference(self):
        """SDXL with a depth ControlNet, which draws a reference image for a bare mesh.

        Loaded on first use, or at the start of a worker serving a paint card
        (``server.py --paint``): generating from an image never needs it.
        """
        self._prepare()
        if self.reference is None:
            import torch
            from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
            from huggingface_hub import snapshot_download

            fp16 = torch.float16
            self.reference = StableDiffusionXLControlNetPipeline.from_pretrained(
                snapshot_download(REFERENCE_REPO_ID, local_files_only=True),
                controlnet=ControlNetModel.from_pretrained(
                    snapshot_download(CONTROLNET_REPO_ID, local_files_only=True),
                    variant="fp16", torch_dtype=fp16),
                # The stock SDXL VAE overflows in half precision.
                vae=AutoencoderKL.from_pretrained(
                    snapshot_download(REFERENCE_VAE_REPO_ID, local_files_only=True), torch_dtype=fp16),
                variant="fp16", torch_dtype=fp16,
            ).to("cuda")
            self.reference.set_progress_bar_config(disable=True)
        return self


class _SeededPipeline:
    """The paint stage's multiview diffusion with a chosen seed.

    Upstream seeds it with 0 on every call; this stands in for its pipeline
    and swaps in a generator seeded as asked, leaving everything else alone.
    """

    def __init__(self, pipeline, seed: int):
        self._pipeline, self._seed = pipeline, seed

    def __getattr__(self, name):
        return getattr(self._pipeline, name)

    def __call__(self, *args, **kwargs):
        import torch

        kwargs["generator"] = torch.Generator(device=self._pipeline.device).manual_seed(self._seed)
        return self._pipeline(*args, **kwargs)


def _lab(rgb):
    """CIE L*a*b* (D65) of sRGB colours given as 0..255, shape (..., 3)."""
    import numpy as np

    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


# A painted view whose body colour is further than this (CIE76 ΔE between
# medians) from the anchor view's is not baked; the faces only it saw are filled
# by upstream's inpainting from the painted faces around them.  The anchor is
# the view showing the most body: the front view, which the paint stage weights
# most, is a sliver on a thin plate seen edge-on, and under GATE_MIN_PIXELS it
# would switch the whole gate off.  The others, a top or a side that one large
# flat face fills, now and then come out as the background's flat grey or
# washed out, and would be baked as a grey face.
VIEW_GATE_DELTA_E = 20.0
# When the anchor view's body is further than this from the reference's, the
# views are drawn again with other seeds (up to VIEW_RETRIES more times, the
# k-th retry with seed + RESEED_STRIDE * k) and the closest set kept: now and
# then the multiview model swaps body and background, painting a light body
# near-black.
FRONT_GATE_DELTA_E = 28.0
VIEW_RETRIES = 2
RESEED_STRIDE = 1000
# Fewer body pixels than this and a view's median colour is not trusted.
GATE_MIN_PIXELS = 16
# A reference whose body is itself several colours (per-part materials, painted
# so a vision call can tell the parts apart by colour) is not gated view by
# view: a side where the second material dominates would be dropped as unlike
# the anchor, erasing the very cue the painting is for.  The measure is the
# 90th percentile of ΔE between body pixels paired at random, so a second
# colour on about 6 % or more of the body counts (its cross pairs pass 10 %)
# and a speck (a hole's rim, a highlight) does not.  A spread from the median
# was tried first and is not robust: at a 50/50 split the median falls between
# the two colours and the spread halves.  In the synthetic check
# (test_hunyuan3d_wrapper) one colour under a ±12 L* shading ramp spreads 19,
# under σ 8 RGB noise 17, with a 5 % speck of another colour 0; a second colour
# on 8 % to 90 % of the body spreads its full distance, 35 for a light wood on
# a mid grey, 52 on a dark metal.  27 sits midway between shading and the
# nearer pair.
MULTICOLOUR_SPREAD_DELTA_E = 27.0
# For a multi-coloured reference the retry measures the anchor against the
# reference's closest colours (the 5th percentile of ΔE to its body pixels),
# not its median: a view showing mostly the second material is fine, a body
# darker or greyer than anything in the reference (a body/background swap) is not.
MULTICOLOUR_NEAREST_PERCENTILE = 5.0
# An SDXL-drawn reference whose body is a near-neutral mid grey -- median per-pixel chroma under
# REFERENCE_LIFT_MAX_CHROMA, median L* from REFERENCE_LIFT_ABOVE up to REFERENCE_LIFT_BELOW, and one
# colour (colour_spread at most MULTICOLOUR_SPREAD_DELTA_E) -- is lightened (lifted_reference) to
# a median of REFERENCE_LIFT_TO before the multiview model reads it.  The multiview model now and
# then reads such a grey as a dark metal: over 57 paints of the agentic batches (2026-10-08..10) a
# mid-grey metal reference at L* 55 (C* 1.9) painted at 7.7, while 32 references at 56 or over
# never painted dark.  Replayed from the stored references (seed 0, 2026-10-10; median L* painted,
# off / lifted to 60 / to 70): that mid grey 7.7 / 23 / 72, a dark cast steel at 38 (C* 4.3)
# 40 / 57 / 69.  Lifting to 60 is not enough.  With these bounds (same replay): the mid grey is
# lifted and paints 72 at the first drawing (3.1 ΔE from the drawn reference, which the gate
# measures against); a mid green (L* 54, C* 45) 60, a dark glazed brown (20) 8.4 and a dark oak
# (26) 25, all left alone and each at the first drawing.
# Only near-neutral mid greys: a hue-bearing colour (red, navy, a wood) lifted along L* alone came
# back pale -- a strong red as salmon, navy as lavender, dark walnut as beige -- which is the
# words' colour changed, not a rendering fixed; dark hue-bearing references are the wording
# rule's (a mid or light shade).  In the survey the grey, steel and concrete references had a
# median C* of 0-6.8 (those that painted near-black 0-5.3), every wood and brown 10 or more:
# C* 8 separates them.  The chroma is the median of each pixel's chroma, not the chroma of the
# median a*/b*: a red, navy or wood part beside a grey one pulls the median a*/b* back toward
# zero and would pass for neutral.  The per-pixel median does not catch it either when the grey
# is the majority (its median is then the grey's), so a multi-coloured body (per-part materials,
# painted so the parts are told apart by colour) is never lifted: the lift would lighten the
# second colour too, a red to a pink, a light wood to near white.
# The floor is L* 48, not 40.  The only grey in the replay under the failure's 55, the cast steel
# at 38, painted 40 un-lifted -- its own shade -- so nothing shows a dark grey needing the lift,
# and lifting one to 70 makes a drawn "dark grey" a light grey.  Scaling the lift instead (less
# for a darker body) is no middle way: the replay shows a partial lift of the failing mid grey
# (to 60) still painted dark (23), so a reduced lift fixes nothing and still changes the shade.
# 48 sits about midway between the faithful 38 and the failing 55.  Not under it either for a
# black or charcoal: a near-black reference (L* 32) lifted to 70 still painted 1.5, so the lift
# would only make the reference lie.  Only a drawn reference: a user's image is the look they
# asked for.
REFERENCE_LIFT_MAX_CHROMA = 8.0
REFERENCE_LIFT_ABOVE = 48.0
REFERENCE_LIFT_BELOW = 60.0
REFERENCE_LIFT_TO = 70.0


def _srgb_from_lab(lab):
    """sRGB 0..255 (float) of CIE L*a*b* (D65) colours, shape (..., 3): ``_lab``'s inverse."""
    import numpy as np

    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16) / 116
    f = np.stack([fy + lab[..., 1] / 500, fy, fy - lab[..., 2] / 200], -1)
    xyz = np.where(f > 0.20690, f ** 3, (f - 16 / 116) / 7.787) * np.array([0.95047, 1.0, 1.08883])
    c = xyz @ np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]).T
    c = np.clip(c, 0.0, 1.0)
    return 255.0 * np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)


def lifted_reference(image, below: float | None = None, to: float | None = None):
    """``image`` (RGBA) with its body lightened when it is a near-neutral mid grey.

    The body (alpha over half) qualifies when its median per-pixel chroma is under
    REFERENCE_LIFT_MAX_CHROMA, its median L* lies in [REFERENCE_LIFT_ABOVE, ``below``) and it is
    one colour (colour_spread at most MULTICOLOUR_SPREAD_DELTA_E).  A
    gamma on L* alone (a*, b* kept) then takes the median to ``to``; white stays white and black
    stays black, so shading keeps its order.  Any other body, a hue-bearing colour, a dark grey,
    a black or several colours, is returned as it is.  Returns the image, the body's median L*
    before, and after (None when it was left alone).  See REFERENCE_LIFT_BELOW.
    """
    import numpy as np
    from PIL import Image

    below = REFERENCE_LIFT_BELOW if below is None else below
    to = REFERENCE_LIFT_TO if to is None else to
    rgba = np.asarray(image.convert("RGBA")).copy()
    body = rgba[..., 3] > 127
    if not body.any():
        return image, None, None
    lab = _lab(rgba[..., :3][body])
    median = float(np.median(lab[:, 0]))
    chroma = float(np.median(np.hypot(lab[:, 1], lab[:, 2])))
    if (below is None or not REFERENCE_LIFT_ABOVE <= median < below
            or chroma >= REFERENCE_LIFT_MAX_CHROMA
            or colour_spread(lab[::max(1, len(lab) // 20000)]) > MULTICOLOUR_SPREAD_DELTA_E):
        return image, round(median, 1), None
    gamma = np.log(to / 100.0) / np.log(median / 100.0)
    lab[:, 0] = 100.0 * (np.clip(lab[:, 0], 0.0, 100.0) / 100.0) ** gamma
    rgba[..., :3][body] = np.rint(_srgb_from_lab(lab)).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA"), round(median, 1), round(float(np.median(lab[:, 0])), 1)


def _body_lab(image, limit: int = 20000):
    """L*a*b* of an RGBA image's opaque pixels (all pixels when none is), at most about ``limit``."""
    import numpy as np

    rgba = np.asarray(image.convert("RGBA"))
    body = rgba[..., 3] > 127
    pixels = rgba[..., :3][body] if body.any() else rgba[..., :3].reshape(-1, 3)
    return _lab(pixels[::max(1, len(pixels) // limit)])


def colour_spread(lab) -> float:
    """90th percentile of the CIE76 ΔE between colours ``lab`` (N, 3) paired at random (fixed seed)."""
    import numpy as np

    lab = np.asarray(lab, dtype=np.float64)
    partner = lab[np.random.default_rng(0).permutation(len(lab))]
    return float(np.percentile(np.linalg.norm(lab - partner, axis=-1), 90))


# A painted surface whose vertex colours spread (colour_spread) less than this is one colour.
# Over the 74 paints of the agentic batches (2026-10-08..10) the least spread of a real painting
# was 1.25 (a matte black), most 18; the two that baked nothing spread 0.
PAINT_UNIFORM_SPREAD = 0.5
# ... and that is a failure when the painted views' body pixels spread more than this, i.e. the
# views showed shading or grain that never reached the surface.
PAINT_VIEWS_VARIED_SPREAD = 2.0


def painting_failure(colours, baked_texels: int | None = None, view_spread: float | None = None) -> str | None:
    """Why a painting's vertex ``colours`` (N, 3 or 4, 0..255) are not a painting, or None.

    ``baked_texels`` is how many texels the views' bake covered (None when unknown) and
    ``view_spread`` the colour_spread of the baked views' body pixels.  A paint stage that
    baked nothing still hands back a texture -- the inpainting of an empty one, all black -- so
    without this a failed bake was reported as a painting.
    """
    import numpy as np

    rgb = np.asarray(colours)[:, :3]
    if baked_texels == 0:
        return ("no painted view reached the surface: the bake covered 0 texels "
                "(every view saw the body as one flat face, or none saw it)")
    if len(rgb) and not rgb.any():
        return "every vertex colour is black (0, 0, 0): the bake wrote nothing onto the surface"
    if len(rgb) and view_spread is not None and view_spread > PAINT_VIEWS_VARIED_SPREAD:
        spread = colour_spread(_lab(rgb[::max(1, len(rgb) // 20000)]))
        if spread < PAINT_UNIFORM_SPREAD:
            return (f"the surface came out one colour (spread {spread:.2f} ΔE) where the painted "
                    f"views vary ({view_spread:.1f} ΔE): the views did not reach the surface")
    return None


def _median_body(image):
    """Median RGB of an RGBA image's opaque pixels."""
    import numpy as np

    rgba = np.asarray(image.convert("RGBA"))
    body = rgba[..., 3] > 127
    return np.median((rgba[..., :3][body] if body.any() else rgba[..., :3].reshape(-1, 3)), axis=0)


class _GatedViews:
    """The paint stage's multiview model and baking, checked against the reference.

    Stands in for the multiview model: after it draws the views, each one's
    body (where its normal map is not background) is compared in colour with
    the anchor view's (the one showing the most body), and the anchor's with
    the reference's.  An anchor unlike the reference has the views drawn
    again with the next seed (``reseed``); ``bake`` then bakes only the views
    that matched the anchor.  A multi-coloured reference keeps every view and
    retries only a gross mismatch.  ``gate`` says whether views were gated
    and, when not, why; ``kept_attempt`` which drawing was kept (0 the first).
    """

    def __init__(self, model, bake, reference, reseed=None, threshold: float = VIEW_GATE_DELTA_E,
                 front_threshold: float = FRONT_GATE_DELTA_E, retries: int = VIEW_RETRIES):
        import numpy as np

        self._model, self._bake, self._reseed = model, bake, reseed
        self._threshold, self._front_threshold, self._retries = threshold, front_threshold, retries
        self.reference_pixels = _body_lab(reference)
        self.reference_lab = np.median(self.reference_pixels, axis=0)
        self.reference_spread = round(colour_spread(self.reference_pixels), 1)
        self.multicoloured = self.reference_spread > MULTICOLOUR_SPREAD_DELTA_E
        self.kept, self.distances, self.front_distance, self.attempts = None, [], None, 0
        self.anchor, self.kept_attempt, self.gate = None, 0, "not run"
        # What the albedo bake covered and how varied the baked views were, for painting_failure.
        self.baked_texels, self.view_spread = None, None

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _reference_distance(self, colour) -> float:
        import numpy as np

        if not self.multicoloured:
            return float(np.linalg.norm(colour - self.reference_lab))
        to_pixels = np.linalg.norm(self.reference_pixels - colour, axis=-1)
        return float(np.percentile(to_pixels, MULTICOLOUR_NEAREST_PERCENTILE))

    def _measure(self, tiles, conditions):
        import numpy as np
        from PIL import Image

        colours, counts = [], []
        for index, tile in enumerate(tiles):
            normal = np.asarray(conditions[index].convert("RGB").resize(tile.size, Image.NEAREST))
            body = ~(normal > 250).all(axis=-1)
            pixels = np.asarray(tile.convert("RGB"))[body]
            counts.append(len(pixels))
            colours.append(_lab(np.median(pixels, axis=0)) if len(pixels) >= GATE_MIN_PIXELS else None)
        anchor = int(np.argmax(counts)) if counts else None
        if anchor is None or colours[anchor] is None:
            # No view shows enough body to trust a colour: nothing is gated or retried.
            return [None] * len(colours), 0.0, None
        distances = [None if colour is None else round(float(np.linalg.norm(colour - colours[anchor])), 1)
                     for colour in colours]
        return distances, round(self._reference_distance(colours[anchor]), 1), anchor

    def __call__(self, images, conditions, **kwargs):
        best = None
        for attempt in range(1 + (self._retries if self._reseed is not None else 0)):
            if attempt:
                self._reseed(attempt)
            result = self._model(images, conditions, **kwargs)
            distances, front_distance, anchor = self._measure(result["albedo"], conditions)
            if best is None or front_distance < best[2]:
                best = (result, distances, front_distance, anchor, attempt)
            self.attempts = attempt + 1
            if front_distance <= self._front_threshold:
                break
        result, self.distances, self.front_distance, self.anchor, self.kept_attempt = best
        if self.anchor is None:
            self.gate = f"not applied: no view showed {GATE_MIN_PIXELS} or more body pixels"
            self.kept = None
        elif self.multicoloured:
            self.gate = (f"not applied: the reference's body is several colours (spread "
                         f"{self.reference_spread} > {MULTICOLOUR_SPREAD_DELTA_E:g} ΔE), "
                         f"so only a view unlike every reference colour was drawn again")
            self.kept = None
        else:
            self.gate = "applied"
            self.kept = [index for index, distance in enumerate(self.distances)
                         if index == self.anchor or distance is None or distance <= self._threshold]
        self.view_spread = self._view_spread(result["albedo"], conditions)
        return result

    def _view_spread(self, tiles, conditions):
        """colour_spread of the body pixels of the views that will be baked."""
        import numpy as np
        from PIL import Image

        keep = self.kept if self.kept is not None else range(len(tiles))
        pixels = []
        for index in keep:
            tile = tiles[index]
            normal = np.asarray(conditions[index].convert("RGB").resize(tile.size, Image.NEAREST))
            pixels.append(np.asarray(tile.convert("RGB"))[~(normal > 250).all(axis=-1)])
        pixels = np.concatenate(pixels) if pixels else np.zeros((0, 3))
        if len(pixels) < GATE_MIN_PIXELS:
            return None
        return round(colour_spread(_lab(pixels[::max(1, len(pixels) // 20000)])), 1)

    def bake(self, views, elevations, azimuths, weights):
        keep = self.kept if self.kept is not None else range(len(views))
        pick = lambda values: [values[i] for i in keep]  # noqa: E731
        result = self._bake(pick(views), pick(elevations), pick(azimuths), pick(weights))
        # Upstream bakes the albedo first (then metal-roughness): its mask is what reached the surface.
        if self.baked_texels is None and isinstance(result, tuple) and len(result) == 2:
            self.baked_texels = int(result[1].sum())
        return result


def view_depth(render, resolution: int, elevation: float, azimuth: float):
    """Camera-space depth and silhouette of the loaded mesh from one of the paint stage's cameras.

    Elevation and azimuth are the paint stage's own (orthographic; elevation
    0, azimuth 0 is its front view, positive elevation looks down on the top).
    """
    from DifferentiableRenderer.MeshRender import RenderConfig

    view = render._create_view_state(RenderConfig(elevation, azimuth, resolution=resolution))
    raster, _ = render.raster_rasterize(view.pos_clip, render.pos_idx, resolution=view.resolution)
    z = (view.pos_camera[..., 2:3] / view.pos_camera[..., 3:4]).reshape(1, -1, 1).contiguous()
    depth, _ = render.raster_interpolate(z, raster, render.pos_idx)
    return depth[0, ..., 0].cpu().numpy(), raster[0, ..., 3].cpu().numpy() > 0


# A view whose visible depth spans less than this (in the paint stage's frame, where the body's
# bounding sphere is 1.15 across) sees one flat face square-on, and upstream's bake takes nothing
# from it.  Its back-projection divides the depth by its span to find depth edges, so a span of
# float rounding (1e-7) turns into noise: Canny then marks 28-31 % of the visible pixels as
# edges, the dilated edge band covers the rest, and the view bakes 0 texels.  A body all of
# whose six axis views are like that -- a flat-sided prism built square to the axes -- baked
# nothing at all: the inpainting of an empty texture is black, and in an acceptance run
# (2026-10-10) both paints of such a prism came back with every vertex colour (0, 0, 0) under a
# good set of painted views.  The spans measured on the built bodies of that run: 2.4e-7 to
# 4.8e-7 on every square-on view, against 1.6e-3 and up on any view that sees two depths
# (a view seeing one large face and a small step still bakes).  Every built body of the two
# acceptance batches of that day lost views this way -- one to four of its six, the faces only
# they saw filled by inpainting from the neighbours' colours -- and the prism all six.
FLAT_VIEW_SPAN = 1e-4
# Such a body is painted turned by this much (degrees) about two of its axes, which takes every
# face square to an axis off every camera axis; the colours come back onto the unturned surface.
# Replayed on those bodies (2026-10-10), 3 degrees bakes every one of the six views: the prism's
# 0 texels per view became 8,800-397,000.
FLAT_VIEW_TILT = 3.0
# The views checked: upstream always bakes its first six candidates (the six axis views), and
# picks any further view among those at +-20 degrees elevation, which no face square to an
# axis presents square-on.
AXIS_VIEWS = 6


def flat_views(render, vertices, faces, cameras, resolution: int = 512) -> list[int]:
    """Indices of ``cameras`` ((elevation, azimuth) pairs) that see ``vertices``/``faces`` as one flat face.

    The mesh is set on ``render`` geometry-only, as the reference drawing does; the paint stage
    loads its own UV-mapped copy afterwards.
    """
    import numpy as np

    render.set_mesh(np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32))
    flat = []
    for index, (elevation, azimuth) in enumerate(cameras):
        depth, mask = view_depth(render, resolution, elevation, azimuth)
        if mask.any() and float(np.ptp(depth[mask])) < FLAT_VIEW_SPAN:
            flat.append(index)
    return flat


def front_depth(render, resolution: int = REFERENCE_SIZE, elevation: float = 0.0, azimuth: float = 0.0):
    """Depth (near bright) and silhouette of the loaded mesh from one of the paint stage's cameras.

    Elevation and azimuth are the paint stage's own (orthographic; elevation
    0, azimuth 0 is its front view, positive elevation looks down on the top).
    """
    import numpy as np

    depth, mask = view_depth(render, resolution, elevation, azimuth)
    near = np.zeros_like(depth)
    if mask.any():
        values = depth[mask]
        # The camera looks down -z, so larger z is nearer; map the object's
        # depth range to 0.2..1 and leave the background black.
        span = max(float(values.max() - values.min()), 1e-6)
        near[mask] = 0.2 + 0.8 * (depth[mask] - values.min()) / span
    return near, mask


def framed(near, mask, margin: float):
    """``near`` and ``mask`` cropped to a square around the silhouette, ``margin`` of its side each way.

    Resized back to the input's size, so a thin or small body fills the
    drawing (and the paint stage's 512-pixel copy of it) instead of a sliver.
    """
    import numpy as np
    from PIL import Image

    if not mask.any():
        return near, mask
    size = mask.shape[0]
    rows, cols = np.nonzero(mask)
    centre = ((rows.min() + rows.max()) / 2.0, (cols.min() + cols.max()) / 2.0)
    side = max(rows.max() - rows.min(), cols.max() - cols.min()) + 1
    half = max(side * (0.5 + margin), 8.0)
    top, left = int(round(centre[0] - half)), int(round(centre[1] - half))
    extent = int(round(2 * half))
    # Pad so the square never leaves the image, then cut it out and resize.
    pad = max(0, -top, -left, top + extent - size, left + extent - size)
    near = np.pad(near, pad)[top + pad:top + pad + extent, left + pad:left + pad + extent]
    mask = np.pad(mask, pad)[top + pad:top + pad + extent, left + pad:left + pad + extent]
    # Depth and silhouette are resized apart and divided, so the background's
    # zero never darkens the body's rim.
    def resized(values):
        return np.asarray(Image.fromarray(values.astype(np.float32)).resize((size, size), Image.BILINEAR))

    weight = resized(mask.astype(np.float32))
    depth = resized(np.where(mask, near, 0.0)) / np.maximum(weight, 1e-6)
    mask = weight > 0.5
    return np.where(mask, depth, 0.0), mask


def load_image(path: Path, runtime: Hunyuan3DRuntime):
    """The input as RGBA, with the background removed unless it already has alpha."""
    import numpy as np
    from PIL import Image

    image = Image.open(path)
    if image.mode == "RGBA":
        alpha = np.asarray(image.getchannel("A"))
        if alpha.min() < 250:
            return image
    return runtime.remove_background(image.convert("RGB"))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--input", type=Path,
                        help="the image to generate from; with --surface, an optional reference image")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--surface", type=Path,
                        help="paint this existing surface instead of generating one; its geometry is kept exactly")
    result.add_argument("--prompt", default="",
                        help="with --surface and no --input: what the reference image should show")
    result.add_argument("--reference", type=Path,
                        help="with --surface: where to save the reference image the paint stage followed")
    result.add_argument("--reference-steps", type=int, default=30)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--faces", type=int, default=-1,
                        help=f"face budget; -1 uses upstream's {DEFAULT_FACES}")
    result.add_argument("--steps", type=int, default=30)
    result.add_argument("--octree-resolution", type=int, choices=(256, 384), default=384)
    result.add_argument("--no-flashvdm", action="store_true",
                        help="decode the shape with upstream's plain volume decoder (about 2x slower)")
    result.add_argument("--texture-size", type=int, choices=(1024, 2048, 4096), default=1024,
                        help="baked texture size; the surface keeps one colour per vertex either way")
    result.add_argument("--upscale", action="store_true",
                        help="enlarge each painted view with RealESRGAN before baking (slower)")
    result.add_argument("--paint-resolution", type=int, choices=(512, 768), default=512)
    result.add_argument("--views", type=int, choices=range(6, 10), default=6)
    result.add_argument("--textured-glb", action="store_true",
                        help="also keep upstream's UV-textured PBR mesh as <output>.pbr.glb, for viewing")
    result.add_argument("--no-texture", action="store_true", help="shape only, no colour")
    return result


def run(args, runtime: Hunyuan3DRuntime | None = None) -> list[str]:
    if args.surface is not None:
        return paint_surface(args, runtime)
    import torch
    import trimesh

    if args.input is None or not Path(args.input).is_file():
        raise FileNotFoundError(f"input image not found: {args.input}")
    if args.faces != -1 and args.faces < 1000:
        raise ValueError("--faces must be -1 or at least 1000")
    texture = not args.no_texture
    runtime = runtime or Hunyuan3DRuntime(texture, args.paint_resolution, args.views)
    runtime.preload(texture)
    from hy3dshape.postprocessors import DegenerateFaceRemover, FaceReducer, FloaterRemover

    image = load_image(Path(args.input), runtime)
    runtime.set_flashvdm(not args.no_flashvdm)
    mesh = runtime.shape(
        image=image, num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
        generator=torch.manual_seed(args.seed),
    )[0]
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=DEFAULT_FACES if args.faces == -1 else args.faces)
    surface, discarded = clean_surface(mesh)
    surface = fit_to_frame(surface)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = [str(output)]
    if texture:
        paint = runtime.paint
        # The paint stage reads its view count and resolution from its config.
        paint.config.max_selected_view_num = args.views
        paint.config.resolution = args.paint_resolution
        runtime.set_texture_size(args.texture_size)
        runtime.set_upscale(args.upscale)
        with tempfile.TemporaryDirectory(prefix="hunyuan3d-") as scratch:
            scratch = Path(scratch)
            surface.export(scratch / "surface.obj")
            image.save(scratch / "input.png")
            painted = paint(
                mesh_path=os.fspath(scratch / "surface.obj"),
                image_path=os.fspath(scratch / "input.png"),
                output_mesh_path=os.fspath(scratch / "textured.obj"),
                use_remesh=False, save_glb=args.textured_glb,
            )
            textured = trimesh.load(painted, process=False, force="mesh")
            colours = transfer_colours(surface, textured.vertices, textured.visual.uv,
                                       texture_of(textured))
            failure = painting_failure(colours)
            if failure is not None:
                raise RuntimeError(f"the painting failed: {failure}")
            if args.textured_glb:
                pbr = output.with_suffix(".pbr.glb")
                os.replace(scratch / "textured.glb", pbr)
                artifacts.append(str(pbr))
        surface.visual = trimesh.visual.ColorVisuals(surface, vertex_colors=colours)
    surface.export(output)
    print(json.dumps({
        "artifacts": artifacts, "vertices": len(surface.vertices),
        "faces": len(surface.faces), "discarded_components": discarded,
        "coloured": texture,
    }))
    return artifacts


def draw_reference(runtime: Hunyuan3DRuntime, mesh, prompt: str, seed: int, steps: int):
    """A reference image of ``mesh``'s front view, drawn by SDXL from ``prompt`` and its depth.

    Returned RGBA, with the mesh's silhouette as alpha, as the paint stage
    takes an uploaded image.
    """
    import numpy as np
    import torch
    from PIL import Image

    # Geometry only: upstream's load_mesh assumes UVs, which a bare mesh has none of.
    runtime.paint.render.set_mesh(np.asarray(mesh.vertices, dtype=np.float32),
                                  np.asarray(mesh.faces, dtype=np.int32))
    near, mask = front_depth(runtime.paint.render, REFERENCE_SIZE, *REFERENCE_VIEW)
    if REFERENCE_FRAME is not None:
        near, mask = framed(near, mask, REFERENCE_FRAME)
    runtime.preload_reference()
    depth = Image.fromarray(np.rint(near * 255).astype(np.uint8)).convert("RGB")
    words = ", ".join(part for part in (prompt.strip(), REFERENCE_STYLE) if part)
    image = runtime.reference(
        prompt=words, negative_prompt=REFERENCE_NEGATIVE, image=depth,
        controlnet_conditioning_scale=0.8, num_inference_steps=steps, guidance_scale=7.0,
        width=REFERENCE_SIZE, height=REFERENCE_SIZE,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0].convert("RGBA")
    image.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    return image


def _checked_uv_load(render):
    """``render.load_mesh`` that raises when the unwrapped UVs are not finite.

    Upstream's vertex inpainting turns every UV into a texel index unchecked, so a NaN UV is
    an out-of-range read and the process dies with SIGSEGV after minutes of drawing, leaving
    nothing in its log (paint_frame says when xatlas returns them).  Raised here, before any
    view is drawn, it is an error that says what happened.
    """
    import numpy as np

    load = type(render).load_mesh

    def checked(*args, **kwargs):
        load(render, *args, **kwargs)
        uv = getattr(render, "vtx_uv", None)
        if uv is None:
            return
        uv = uv.detach().cpu().numpy() if hasattr(uv, "detach") else np.asarray(uv)
        bad = int((~np.isfinite(uv)).any(axis=-1).sum())
        if bad:
            # "cannot be painted": no seed changes this (the workflow then gives a plain colour).
            raise RuntimeError(
                f"the surface cannot be painted: its UV unwrap failed ({bad} of {len(uv)} texture "
                "coordinates are not finite), and upstream's inpainting would crash on them")

    return checked


def _replace_atomically(path: Path, write) -> None:
    """``write(temporary)`` beside ``path``, then renamed over it in one step.

    A paint may run in the background while the workflow records the session's files (graph's
    branch record, which skips dot-named files): it must never see half a painting.
    """
    path = Path(path)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    try:
        write(os.fspath(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def paint_surface(args, runtime: Hunyuan3DRuntime | None = None) -> list[str]:
    """Colour an existing surface with the paint stage, leaving its geometry untouched.

    The reference image is the user's (``--input``) or drawn from ``--prompt``
    and the surface's front-view depth.  The paint stage runs on the surface
    itself, or on a copy reduced to the face budget when it is larger; either
    way the colours come back onto the surface's own vertices.
    """
    import numpy as np
    import torch
    import trimesh

    if not Path(args.surface).is_file():
        raise FileNotFoundError(f"surface not found: {args.surface}")
    if args.input is not None and not Path(args.input).is_file():
        raise FileNotFoundError(f"reference image not found: {args.input}")
    runtime = runtime or Hunyuan3DRuntime(True, args.paint_resolution, args.views)
    runtime.preload(texture=True, shape=False)
    from hy3dshape.postprocessors import FaceReducer

    loaded = trimesh.load(args.surface, force="mesh", process=False)
    surface = trimesh.Trimesh(loaded.vertices, loaded.faces, process=False)
    budget = DEFAULT_FACES if args.faces == -1 else args.faces
    reduced = len(surface.faces) > budget
    copy = FaceReducer()(surface.copy(), max_facenum=budget) if reduced else surface.copy()

    if args.input is not None:
        image = load_image(Path(args.input), runtime)
    else:
        image = draw_reference(runtime, copy, args.prompt, args.seed, args.reference_steps)
    # A drawn mid grey is lightened for the multiview model alone (lifted_reference).  The saved
    # reference.png stays what SDXL drew, and the view gate measures the painting against it: a
    # body that painted at the drawn grey is right, and only one far darker is drawn again.
    painted_from, reference_lightness, lifted_lightness = (
        (image, None, None) if args.input is not None else lifted_reference(image))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = [str(output)]
    if args.reference is not None:
        Path(args.reference).parent.mkdir(parents=True, exist_ok=True)
        _replace_atomically(Path(args.reference), image.save)
        artifacts.append(str(args.reference))

    paint = runtime.paint
    paint.config.max_selected_view_num = args.views
    paint.config.resolution = args.paint_resolution
    runtime.set_texture_size(args.texture_size)
    runtime.set_upscale(args.upscale)
    # A body that one of the axis views sees as a single flat face is painted turned a little
    # (FLAT_VIEW_SPAN): drawn from the unturned copy above, the reference is unaffected.
    cameras = list(zip(paint.config.candidate_camera_elevs[:AXIS_VIEWS],
                       paint.config.candidate_camera_azims[:AXIS_VIEWS]))
    # Painted in a frame of half-extent 1 (paint_frame): xatlas unwraps a small body in metres
    # to NaN UVs, on which upstream's inpainting segfaults.  The views are unchanged by it.
    centre, half = paint_frame(copy.vertices)
    copy = trimesh.Trimesh((np.asarray(copy.vertices) - centre) / half, copy.faces, process=False)
    flat = flat_views(paint.render, copy.vertices, copy.faces, cameras)
    tilt, flat_after = None, []
    if flat:
        tilt = tilt_rotation(FLAT_VIEW_TILT)
        copy = trimesh.Trimesh(np.asarray(copy.vertices) @ tilt.T, copy.faces, process=False)
        flat_after = flat_views(paint.render, copy.vertices, copy.faces, cameras)
    multiview = paint.models["multiview_model"]
    pipeline = multiview.pipeline
    processor = paint.view_processor
    gate = None
    try:
        # Patched inside the try, so a resident worker never keeps a seeded
        # pipeline or a half-installed gate when building one raises.
        seeded = multiview.pipeline = _SeededPipeline(pipeline, args.seed)
        if VIEW_GATE_DELTA_E is not None:
            def reseed(attempt):
                seeded._seed = args.seed + RESEED_STRIDE * attempt

            gate = _GatedViews(multiview, processor.bake_from_multiview, image, reseed)
            paint.models["multiview_model"] = gate
            processor.bake_from_multiview = gate.bake
        paint.render.load_mesh = _checked_uv_load(paint.render)
        with tempfile.TemporaryDirectory(prefix="hunyuan3d-paint-") as scratch:
            scratch = Path(scratch)
            copy.export(scratch / "surface.obj")
            painted_from.save(scratch / "input.png")
            painted = paint(
                mesh_path=os.fspath(scratch / "surface.obj"),
                image_path=os.fspath(scratch / "input.png"),
                output_mesh_path=os.fspath(scratch / "textured.obj"),
                use_remesh=False, save_glb=False,
            )
            textured = trimesh.load(painted, process=False, force="mesh")
    finally:
        multiview.pipeline = pipeline
        paint.models["multiview_model"] = multiview
        processor.__dict__.pop("bake_from_multiview", None)
        paint.render.__dict__.pop("load_mesh", None)
    torch.cuda.empty_cache()
    # Turned and scaled back onto the surface's frame (the paint stage keeps the positions it
    # was given).
    back = np.asarray(textured.vertices, dtype=np.float64)
    if tilt is not None:
        back = back @ tilt
    textured.vertices = back * half + centre
    if reduced:
        colours = transfer_colours_nearest(surface, textured, texture_of(textured))
    else:
        colours = transfer_colours(surface, textured.vertices, textured.visual.uv, texture_of(textured))
    failure = painting_failure(colours, gate.baked_texels if gate is not None else None,
                               gate.view_spread if gate is not None else None)
    if failure is not None:
        # Never written: a failed painting must not stand in the place of one.
        raise RuntimeError(f"the painting failed: {failure}")
    surface.visual = trimesh.visual.ColorVisuals(surface, vertex_colors=colours)
    _replace_atomically(output, surface.export)
    kept_attempt = gate.kept_attempt if gate is not None else 0
    print(json.dumps({
        "artifacts": artifacts, "vertices": len(surface.vertices), "faces": len(surface.faces),
        "painted_faces": len(copy.faces), "reference": "image" if args.input is not None else "prompt",
        "seed": args.seed,
        # Axis views that saw one flat face (FLAT_VIEW_SPAN), before and after the turn by
        # tilt_degrees that the paint was then made under.
        "flat_views": flat, "tilt_degrees": FLAT_VIEW_TILT if tilt is not None else 0.0,
        "flat_views_after_tilt": flat_after,
        "baked_texels": gate.baked_texels if gate is not None else None,
        "view_spread": gate.view_spread if gate is not None else None,
        "reference_lightness": reference_lightness, "reference_lifted_to": lifted_lightness,
        # Which drawing was kept and its seed (retry k draws with seed + RESEED_STRIDE * k), so
        # "change the seed" can start from the one actually seen.
        "kept_attempt": kept_attempt, "kept_seed": args.seed + RESEED_STRIDE * kept_attempt,
        **({"view_gate": gate.gate, "view_anchor": gate.anchor, "view_distances": gate.distances,
            "views_baked": gate.kept if gate.kept is not None else list(range(len(gate.distances))),
            "reference_spread": gate.reference_spread, "front_distance": gate.front_distance,
            "view_attempts": gate.attempts} if gate is not None else {"view_gate": "not applied: disabled"}),
    }))
    return artifacts


def main():
    run(parser().parse_args())
    sys.stdout.flush(); sys.stderr.flush()
    # Release the GPU at once instead of waiting on interpreter teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
