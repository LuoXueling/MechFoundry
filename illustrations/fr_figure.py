"""Compose the design-evolution FRAME series (conda `mechfoundry-warp117`, CPU).

One frame per optimiser iteration: the octree + d section cut (left) and the
sectioned design surface (right) side by side, rendered by the shipped panel
code, with the iteration number and the cumulative optimiser wall clock marked
in the top-left corner.

    python fr_figure.py so     # shape_opt (shapeopt/panel_so_fr_*, so_fr_times.txt)
    python fr_figure.py hp     # hidden_phi (hiddenphi/hp_panel_fr_*, hp_fr_times.txt)

Writes  <out>_frames/frame_XXXX.png   (the numbered series, the deliverable)
and     <out>.gif  (+ <out>.mp4 when imageio/ffmpeg is available)  in illustrations/.
Every panel is cropped to ONE common ink box across the whole run so the part
does not resize between frames — the movie holds still while the design evolves.
"""
import glob
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "so": dict(sub="shapeopt", oct="panel_so_fr_{}.png",
               body="panel_so_frbody_{}.png", times="so_fr_times.txt",
               out="shape_opt_evolution", title="Shape optimization"),
    "hp": dict(sub="hiddenphi", oct="hp_panel_fr_{}.png",
               body="hp_panel_frbody_{}.png", times="hp_fr_times.txt",
               out="hidden_phi_evolution", title="Void-nucleation optimization"),
}

VIDEO_SECONDS = 5.0      # every movie finishes in this many seconds (fps = n/5)
SLIDE_WIDTH = 1920       # presentation-friendly variants, preserving aspect ratio
TH = 820                 # panel display height (px); both panels share it
PAD = 46                 # outer margin
GAP = 44                 # between the two panels
HEADER = 150             # top band that holds the iteration/time annotation
SUBLBL = 40              # room under each panel for its small caption
BG = (255, 255, 255)
INK = (26, 26, 33)
GREY = (120, 120, 128)


def _font(path_names, size):
    for p in path_names:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


_AR = "/usr/share/fonts/truetype/msttcorefonts/arial.ttf"
_ARB = "/usr/share/fonts/truetype/msttcorefonts/arialbd.ttf"
F_IT = _font([_ARB, _AR], 58)      # "Iteration N"
F_T = _font([_AR], 44)             # "t = ... s"
F_SUB = _font([_AR], 32)           # panel captions


def ink_box(im):
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    m = np.abs(a - 255).sum(2) > 12
    ys, xs = np.where(m)
    if len(xs) == 0:
        return (0, 0, im.width, im.height)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def union_box(boxes, w, h, pad=10):
    x0 = max(min(b[0] for b in boxes) - pad, 0)
    y0 = max(min(b[1] for b in boxes) - pad, 0)
    x1 = min(max(b[2] for b in boxes) + pad, w)
    y1 = min(max(b[3] for b in boxes) + pad, h)
    return x0, y0, x1, y1


def scaled(im, box, th):
    c = im.crop(box)
    w = max(1, round(c.width * th / c.height))
    return c.resize((w, th), Image.LANCZOS)


def main(key):
    cfg = CONFIG[key]
    sub = os.path.join(ROOT, cfg["sub"])

    meta = {}          # iter -> (wallclock_seconds, nodes, elements)
    tpath = os.path.join(sub, cfg["times"])
    if os.path.exists(tpath):
        for line in open(tpath):
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            meta[int(p[0])] = (float(p[1]),
                               int(p[2]) if len(p) > 2 else None,
                               int(p[3]) if len(p) > 3 else None)

    # iterations for which BOTH panels rendered
    its = []
    for f in glob.glob(os.path.join(sub, cfg["oct"].format("*"))):
        it = int(re.search(r"_fr_?(\d+)\.png$", os.path.basename(f)).group(1))
        if os.path.exists(os.path.join(sub, cfg["body"].format(it))):
            its.append(it)
    its = sorted(set(its))
    assert its, f"no rendered frames found in {sub} for {key}"
    print(f"[{key}] {len(its)} frames: {its[:6]}{' ...' if len(its) > 6 else ''}")

    oct_raw = {it: Image.open(os.path.join(sub, cfg["oct"].format(it))).convert("RGB")
               for it in its}
    body_raw = {it: Image.open(os.path.join(sub, cfg["body"].format(it))).convert("RGB")
                for it in its}
    w0, h0 = next(iter(oct_raw.values())).size
    ob = union_box([ink_box(im) for im in oct_raw.values()], w0, h0)
    w1, h1 = next(iter(body_raw.values())).size
    bb = union_box([ink_box(im) for im in body_raw.values()], w1, h1)

    # canvas size is fixed by the common crops -> every frame is identical in size
    oct_w = round((ob[2] - ob[0]) * TH / (ob[3] - ob[1]))
    body_w = round((bb[2] - bb[0]) * TH / (bb[3] - bb[1]))
    CW = PAD + oct_w + GAP + body_w + PAD
    CH = PAD + HEADER + TH + SUBLBL + PAD
    last = its[-1]

    outdir = os.path.join(ROOT, cfg["out"] + "_frames")
    os.makedirs(outdir, exist_ok=True)
    for old in glob.glob(os.path.join(outdir, "frame_*.png")):
        os.remove(old)

    frames = []
    for k, it in enumerate(its):
        canvas = Image.new("RGB", (CW, CH), BG)
        canvas.paste(scaled(oct_raw[it], ob, TH), (PAD, PAD + HEADER))
        canvas.paste(scaled(body_raw[it], bb, TH), (PAD + oct_w + GAP, PAD + HEADER))
        d = ImageDraw.Draw(canvas)
        # top-left: the iteration count and the cumulative optimiser wall clock
        d.text((PAD, PAD - 4), f"Iteration {it} / {last}", font=F_IT, fill=INK)
        sec, nn, ne = meta.get(it, (None, None, None))
        tstr = f"Wallclock time = {sec:.1f} s" if sec is not None else "Wallclock time = —"
        d.text((PAD, PAD + 74), tstr, font=F_T, fill=INK)
        # under the octree: its size this iteration
        if nn is not None:
            d.text((PAD, PAD + HEADER + TH + 4),
                   f"{nn:,} nodes   ·   {ne:,} elements", font=F_SUB, fill=GREY)
        fp = os.path.join(outdir, f"frame_{k:04d}.png")
        canvas.save(fp)
        frames.append(fp)
    print(f"[{key}] wrote {len(frames)} frames to {outdir}  ({CW}x{CH})")

    # --- assemble a movie that finishes in VIDEO_SECONDS regardless of frame
    # count: fps = n / VIDEO_SECONDS.  mp4 via the `ffmpeg` conda env (ffmpeg is
    # not in `mechfoundry-warp117`); the gif is a self-contained fallback at the same pace.
    fps = len(frames) / VIDEO_SECONDS
    gif = os.path.join(ROOT, cfg["out"] + ".gif")
    imgs = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=max(1, round(1000.0 / fps)), loop=0, optimize=True)
    print(f"[{key}] wrote {gif}  ({len(frames)} frames, {VIDEO_SECONDS:g}s, {fps:.1f} fps)")

    # Build the slide GIF from the RGB masters, not by transcoding the GIF.
    # Direct GIF-to-GIF scaling in ffmpeg can reuse/dispose local palettes in a
    # way that produces visibly corrupted colours and trails between frames.
    slide_h = round(CH * SLIDE_WIDTH / CW)
    slide_h += slide_h % 2
    slide_imgs = [
        Image.open(f).convert("RGB").resize((SLIDE_WIDTH, slide_h), Image.LANCZOS)
        .convert("P", palette=Image.ADAPTIVE)
        for f in frames
    ]
    slide_gif = os.path.join(ROOT, cfg["out"] + "_slides.gif")
    slide_imgs[0].save(
        slide_gif,
        save_all=True,
        append_images=slide_imgs[1:],
        duration=max(1, round(1000.0 / fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"[{key}] wrote {slide_gif}  ({SLIDE_WIDTH}x{slide_h})")

    mp4 = os.path.join(ROOT, cfg["out"] + ".mp4")
    cmd = ["conda", "run", "-n", "ffmpeg", "ffmpeg", "-y", "-framerate", f"{fps}",
           "-i", os.path.join(outdir, "frame_%04d.png"),
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",   # libx264 needs even dims
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[{key}] wrote {mp4}  ({VIDEO_SECONDS:g}s, {fps:.1f} fps)")
    except Exception as e:                                             # noqa: BLE001
        if os.path.exists(mp4):
            os.remove(mp4)
        msg = getattr(e, "stderr", "") or str(e)
        print(f"[{key}] mp4 skipped (need `conda create -n ffmpeg -c conda-forge "
              f"ffmpeg`): {msg.strip().splitlines()[-1] if msg else e}; gif only")

    slide_mp4 = os.path.join(ROOT, cfg["out"] + "_slides.mp4")
    slide_cmd = [
        "conda", "run", "-n", "ffmpeg", "ffmpeg", "-y", "-framerate", f"{fps}",
        "-i", os.path.join(outdir, "frame_%04d.png"),
        "-vf", f"scale={SLIDE_WIDTH}:{slide_h},fps=30",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        slide_mp4,
    ]
    try:
        subprocess.run(slide_cmd, check=True, capture_output=True, text=True)
        print(f"[{key}] wrote {slide_mp4}  ({VIDEO_SECONDS:g}s, 30 fps)")
    except Exception as e:                                             # noqa: BLE001
        if os.path.exists(slide_mp4):
            os.remove(slide_mp4)
        msg = getattr(e, "stderr", "") or str(e)
        print(f"[{key}] slide mp4 skipped: {msg.strip().splitlines()[-1] if msg else e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "so")
