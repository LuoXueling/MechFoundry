#!/usr/bin/env python
"""Text -> an image of one object, drawn by SDXL, as an input for the image-to-3D generators.

TripoSG's text mode has no picture to follow -- it runs the scribble checkpoint
on the prompt alone -- while its image checkpoint, and Hunyuan3D-2.1, follow a
picture closely.  Drawing the object first gives either of them one.  SDXL is
the painter's reference model, already in the Hunyuan3D cache with its
half-precision-safe VAE, so this runs in the Hunyuan3D environment and needs
none of Hunyuan3D's own models.

SDXL draws a studio floor under the object whatever the prompt says, and
Hunyuan3D's background remover keeps it: the floor came back as a slab fused to
the object's legs.  So the drawing is cut out with TripoSG's RMBG-1.4,
thresholded as TripoSG's own image loader does, and saved with that mask as its
alpha, which both generators take as given.  TripoSG's loader then skips its own
background removal and gets the same mask it would have made.  The drawing as
drawn is kept beside it.

    python draw_image.py --prompt "a sturdy oak bench" --output cut.png --output-raw drawn.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(REPO_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(REPO_ROOT))

from src.hunyuan3d_wrapper.availability import (  # noqa: E402
    REFERENCE_REPO_ID, REFERENCE_VAE_REPO_ID, settings, snapshot_dir,
)

# What the image-to-3D generators were trained on: one whole object, centred on
# a plain background, seen from a little above so its top and one side show.
# A cropped or scenic picture gives a cropped or fused shape.
#
# SDXL still draws several at some seeds: a row of remotes, a pair of clogs, a spread of
# burgers, now and then a grid of the same pagoda.  "multiple objects" alone does not stop a
# lineup or a collage, so the negative names those too.  Measured 2026-10-13 on 16 descriptions
# agents wrote (seeds 0-3, the same seeds each way): drawings with more than one object or view
# fell from 22 of 64 to 17, six fixed and one new, with no more cropped or wrong objects (12 to
# 11).  Dropping "product photo" and "three-quarter view" from STYLE changed nothing (17 again),
# so STYLE stays.  What remains is mostly a subject drawn as a set (a pair of shoes, a loaf and
# its slices), which another seed fixes.  The negative has its own 77 tokens: this is 53.
STYLE = ("a single object, the whole object in frame, three-quarter view from slightly above, "
         "centered, plain white background, soft even lighting, product photo")
NEGATIVE = ("cropped, cut off, partial view, close-up, multiple objects, person, background "
            "scenery, floor, shadow, text, watermark, blurry, multiple views, collage, grid, "
            "turnaround, reference sheet, split image, duplicate, two views")
SIZE = 1024
RMBG = REPO_ROOT / "src" / "dependency" / "TripoSG" / "pretrained_weights" / "RMBG-1.4"


def otsu(values) -> int:
    """Otsu's threshold of a uint8 array: pixels above it are foreground."""
    import numpy as np

    p = np.bincount(values.ravel(), minlength=256).astype(float)
    p /= p.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(256))
    between = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    return int(np.argmax(between))


def load_rmbg():
    """TripoSG's background remover, fetched as TripoSG fetches it; None if unavailable."""
    import torch
    from huggingface_hub import snapshot_download

    try:
        if not (RMBG / "config.json").is_file():
            snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=RMBG)
        from src.dependency.TripoSG.scripts.briarmbg import BriaRMBG

        return BriaRMBG.from_pretrained(RMBG).to("cuda").eval()
    except Exception as exc:  # offline, or no TripoSG checkout: draw without the cut-out
        print(f"no RMBG-1.4 cut-out: {type(exc).__name__}: {exc}", file=sys.stderr)
        torch.cuda.empty_cache()
        return None


def cut_out(net, image):
    """``image`` as RGBA, its alpha the object's mask from RMBG-1.4.

    The same steps as TripoSG's loader (``scripts/image_process.py``): the
    network on a 1024² copy, rescaled to 0..1, Otsu's threshold, and specks
    under 200 pixels dropped.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from scipy import ndimage

    rgb = np.array(image.convert("RGB"))  # a writable copy, which torch can share
    x = torch.from_numpy(rgb).cuda().float().permute(2, 0, 1)[None] / 255.0
    x = F.interpolate(x, size=(1024, 1024), mode="bilinear", antialias=True)
    with torch.no_grad():
        mask = net(x - 0.5)[0][0]
    mask = F.interpolate(mask, size=rgb.shape[:2], mode="bilinear")[0, 0]
    mask = (mask - mask.min()) / (mask.max() - mask.min()).clamp_min(1e-8)
    levels = (mask * 255).to(torch.uint8).cpu().numpy()
    keep = levels > otsu(levels)
    labels, count = ndimage.label(keep)
    if count:
        sizes = ndimage.sum(keep, labels, range(1, count + 1))
        keep = np.isin(labels, 1 + np.flatnonzero(sizes >= 200))
    return Image.fromarray(np.dstack([rgb, keep.astype(np.uint8) * 255]))  # 4 channels: RGBA


def draw(args) -> dict:
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    if not args.prompt.strip():
        raise ValueError("--prompt must describe the object to draw")
    _, _, weights = settings()
    base, vae = snapshot_dir(weights, REFERENCE_REPO_ID), snapshot_dir(weights, REFERENCE_VAE_REPO_ID)
    if base is None or vae is None:
        raise FileNotFoundError(f"SDXL weights are missing in {weights}")
    started = time.time()
    fp16 = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base,
        # The stock SDXL VAE overflows in half precision.
        vae=AutoencoderKL.from_pretrained(vae, torch_dtype=fp16),
        variant="fp16", torch_dtype=fp16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    prompt = ", ".join(part for part in (args.prompt.strip(), STYLE) if part)
    dropped = dropped_words(pipe.tokenizer, prompt)
    image = pipe(
        prompt=prompt, negative_prompt=NEGATIVE, num_inference_steps=args.steps,
        guidance_scale=args.guidance, width=SIZE, height=SIZE,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    ).images[0]
    drawn = time.time()
    del pipe
    torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = []
    if args.output_raw is not None:
        args.output_raw.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output_raw)
        artifacts.append(os.fspath(args.output_raw))
    net = load_rmbg()
    (image if net is None else cut_out(net, image)).save(args.output)
    artifacts.insert(0, os.fspath(args.output))
    return {"artifacts": artifacts, "cut_out": net is not None, "prompt": prompt,
            "seed": args.seed, "steps": args.steps, "draw_s": round(drawn - started, 1),
            "torch_peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            **dropped}


def dropped_words(tokenizer, prompt: str) -> dict:
    """What of the prompt SDXL never reads, by the pipeline's own tokenizer.

    Its CLIP text encoders take 77 tokens (start and end marks included) and diffusers drops
    the rest with only a log line.  The description comes first and STYLE after it, so a long
    description pushes out the very words that keep the drawing one whole object on a plain
    background (2026-10-09: a clog described in about 60 words lost the whole STYLE, and its
    drawings showed pairs).  The same slice diffusers' warning prints; ``description_cut`` when
    the cut reaches into the description itself.
    """
    limit = int(tokenizer.model_max_length)
    ids = tokenizer(prompt, truncation=False).input_ids
    if len(ids) <= limit:
        return {"prompt_tokens": len(ids), "token_limit": limit}
    text = tokenizer.decode(ids[limit - 1:-1])
    style_tokens = len(tokenizer(STYLE, truncation=False).input_ids) - 2
    return {"prompt_tokens": len(ids), "token_limit": limit, "dropped": text,
            "description_cut": len(ids) - 2 - (limit - 2) > style_tokens + 1}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    result.add_argument("--prompt", required=True, help="the object; a single-object style is appended")
    result.add_argument("--output", type=Path, required=True,
                        help="the drawing cut out from its background (RGBA PNG)")
    result.add_argument("--output-raw", type=Path, help="the drawing as drawn")
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--steps", type=int, default=30)
    result.add_argument("--guidance", type=float, default=7.0)
    return result


def main():
    print(json.dumps(draw(parser().parse_args())))
    sys.stdout.flush(); sys.stderr.flush()
    # Release the GPU at once instead of waiting on interpreter teardown.
    os._exit(0)


if __name__ == "__main__":
    main()
