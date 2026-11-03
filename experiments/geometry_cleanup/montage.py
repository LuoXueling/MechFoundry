"""Contact sheets of the before/after renders (CPU, PIL).

    python montage.py [--tag TAG] [--per-page 10] [--views 0,1] [--its K1,K2,...] [object ...]

One row per object: for each view, the starting surface (iteration 0), the final one,
and the final one coloured by its signed change from the start (blue = removed, red =
added, saturated at +-render.CHANGE_MAX h).  The row label carries the object, the
updates run and stop reason, final distance/h, V/V0 and SE/SE0 (results.json).  With
--its, the row instead shows those iterations of one run, and the change maps of the
last two (trial sheets).  Pages are
written to <OUT>/montage[_<tag>]_<page>.png.
"""

import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

import common as C

ap = argparse.ArgumentParser()
ap.add_argument("names", nargs="*")
ap.add_argument("--tag", default="")
ap.add_argument("--per-page", type=int, default=10)
ap.add_argument("--views", default="0,1")
ap.add_argument("--its", default="")
ap.add_argument("--out", default="")
a = ap.parse_args()

TILE = 300
LABEL = 300
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 17)
    BIG = ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial.ttf", 20)
except OSError:
    FONT = BIG = ImageFont.load_default()

views = [int(v) for v in a.views.split(",")]
names = a.names or sorted(C.runs())


def tiles(rd, res):
    cam = json.load(open(os.path.join(rd, "render", "camera.json")))
    last = cam["its"][-1]
    out = []
    for v in views:
        if a.its:
            ks = [int(i) for i in a.its.split(",")]
            for k in ks:
                out.append((f"it {k}", os.path.join(rd, "render", f"it_{k}_v{v}_plain.png")))
            for k in ks[-2:]:
                out.append((f"change it {k}", os.path.join(rd, "render", f"it_{k}_v{v}_change.png")))
        else:
            out += [("start", os.path.join(rd, "render", f"it_0_v{v}_plain.png")),
                    (f"final ({res['updates'] if res else '?'} updates)", os.path.join(rd, "render", f"it_{last}_v{v}_plain.png")),
                    ("change", os.path.join(rd, "render", f"it_{last}_v{v}_change.png"))]
    return out


def label(name, res, st):
    if res is None:
        return [name, "NO RESULT"]
    L = [name[:30], f"{res['updates']} updates, stop: {res['stop_reason']}"]
    if res["final_d_h"] is not None:
        L.append(f"d_p99 {res['final_d_h']:.2f} h  max {res['final_d_max_h']:.2f} h")
    if res["vol_ratio"] is not None:
        L.append(f"V/V0 {res['vol_ratio']:.3f}  SE/SE0 {res['se_ratio']:.3f}")
    L.append(f"h {res['h']:.3g} m" + ("  [stand-in mat]" if res["standin"] else ""))
    if res["error"]:
        e = res["error"]
        L += [e[i:i + 34] for i in range(0, min(len(e), 136), 34)]
    return L


pages, row_imgs = [], []
for n in names:
    rd = os.path.join(C.obj_dir(n), a.tag) if a.tag else C.obj_dir(n)
    rp = os.path.join(rd, "results.json")
    res = json.load(open(rp)) if os.path.isfile(rp) else None
    st = json.load(open(os.path.join(C.obj_dir(n), "setup.json")))
    try:
        ts = tiles(rd, res)
    except FileNotFoundError:
        ts = []
    ncol = len(ts) or 3 * len(views)
    row = Image.new("RGB", (LABEL + TILE * ncol, TILE), "white")
    d = ImageDraw.Draw(row)
    for i, line in enumerate(label(n, res, st)):
        d.text((8, 8 + 24 * i), line, fill=(0, 0, 0), font=BIG if i == 0 else FONT)
    for j, (cap, p) in enumerate(ts):
        if os.path.isfile(p):
            row.paste(Image.open(p).convert("RGB").resize((TILE, TILE)), (LABEL + TILE * j, 0))
        d.text((LABEL + TILE * j + 6, TILE - 22), cap, fill=(90, 90, 90), font=FONT)
    d.line([(0, TILE - 1), (row.width, TILE - 1)], fill=(200, 200, 200))
    row_imgs.append(row)

suf = a.out or (f"_{a.tag}" if a.tag else "")
for p0 in range(0, len(row_imgs), a.per_page):
    rows = row_imgs[p0:p0 + a.per_page]
    W = max(r.width for r in rows)
    page = Image.new("RGB", (W, TILE * len(rows)), "white")
    for i, r in enumerate(rows):
        page.paste(r, (0, TILE * i))
    path = os.path.join(C.OUT, f"montage{suf}_{p0 // a.per_page + 1}.png")
    page.save(path)
    print("wrote", path)
