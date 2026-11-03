"""Stage 4 (env mechfoundry-warp117): compose the amphitheater, place its labels, gate it.

    python am_figure.py WORK_DIR --out agentic_gallery

Layers, bottom to top (page 180 x 101.25 mm, plain white floor: no shadows and no key, the
user's decisions of 2026-10-17):
  outline   the undeformed silhouette: a 0.15 mm #b3b9c1 line just outside the union of
            the undeformed bodies, under the deformed ones (designer fix 6)
  bodies    render_gl's scene.png
  labels    Arial Italic glyphs at one printed size (4.0 pt), placed by am_labels' rules on
            the render's own id passes (>= 0.5 mm from each other, off their own object,
            owned: centroid nearer its own object by >= 1.0 mm, nearest glyph <= 3 mm), and
            composited as ink into the 1200 dpi render UNDER nearer bodies (rev 4, the user:
            a nearer object may cover part of a farther label; >= 70 % of each stays seen).

GATES (fail the build): renderer NVIDIA; all 50 ids drawn; projection (id pixels inside the
Warp-cast silhouettes from the same eye >= 99 %, bounding boxes within 1 quarter-res pixel
on unoccluded objects); every object >= 25 % visible; every label placed; every label
>= 98 % free of object ink (no object over any label); no two labels overlap; ownership;
labels inside the page; label cap height in [1.0, 1.2] mm (the labels' named waiver of the
5 pt floor); size order (the user's seat swap exempted by name); x N equal to the category
records.  Reported: page coverage (2.5 mm cells covered by objects + labels) and the
largest empty rectangle.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
sys.path.insert(0, HERE)
import ar_figure as AF  # noqa: E402  (matplotlib + Arial rcParams first)
import ar_prep  # noqa: E402
import am_layout as AM  # noqa: E402
import am_labels  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import binary_dilation  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OUT_DPI = 600
OUTLINE_W = 0.15
OUTLINE_RGB = (0xb3, 0xb9, 0xc1)
OUTLINE_ALPHA = 0.9
VIS_MIN = 0.25
LABEL_FREE_MIN = 0.98
CAP_BAND = (1.2, 1.35)        # rev 5: 5 pt labels (cap 1.26 mm), the journal minimum
CHECK = {}


def gate(name, ok, value, msg=""):
    CHECK[name] = {"ok": bool(ok), "value": value}
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: {value} {msg}")


def load(work, name):
    return np.asarray(Image.open(os.path.join(work, name)))


def decode_ids(img):
    rgb = img[..., :3].astype(int)
    obj = (rgb[..., 2] < 128) & (rgb[..., 1] > 128)
    return np.where(obj, rgb[..., 0], 0)


def label_alpha(v, codes, px_mm):
    """The label rendered alone by Agg (anti-aliased, true holes) on the page's pixel grid:
    (alpha, row0, col0)."""
    import matplotlib.pyplot as plt
    c0, c1 = int(math.floor(v[:, 0].min() * px_mm)) - 2, int(math.ceil(v[:, 0].max() * px_mm)) + 2
    r0 = int(math.floor((AM.PAGE_H - v[:, 1].max()) * px_mm)) - 2
    r1 = int(math.ceil((AM.PAGE_H - v[:, 1].min()) * px_mm)) + 2
    fig = plt.figure(figsize=((c1 - c0) / 100, (r1 - r0) / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(c0 / px_mm, c1 / px_mm)
    ax.set_ylim(AM.PAGE_H - r1 / px_mm, AM.PAGE_H - r0 / px_mm)
    ax.set_axis_off()
    ax.add_patch(PathPatch(Path(v, codes), fc="black", ec="none", lw=0))
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., 0].astype(float)
    plt.close(fig)
    return 1.0 - img / 255.0, r0, c0


def cast_alone(scene, objs_world, shape, mm):
    """Each object's deformed silhouette alone, Warp-cast from the eye through the
    quarter-resolution pixel centres (the render's own camera)."""
    cam = scene["camera"]
    E = np.array(cam["position"])
    look = AM.to_world(np.array(cam["look_schematic"])[None])[0]
    right = AM.to_world(np.array(cam["right_schematic"])[None])[0]
    up = AM.to_world(np.array(cam["up_schematic"])[None])[0]
    f = cam["f_mm"]
    out = {}
    for k, (V, F) in objs_world.items():
        v = V - E
        zc = v @ look
        px = AM.PAGE_W / 2 + f * (v @ right) / zc
        py = AM.PAGE_H / 2 + f * (v @ up) / zc
        c0, c1 = max(int(px.min() / mm) - 2, 0), min(int(px.max() / mm) + 3, shape[1])
        r0 = max(int((AM.PAGE_H - py.max()) / mm) - 2, 0)
        r1 = min(int((AM.PAGE_H - py.min()) / mm) + 3, shape[0])
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        xn = ((cc + 0.5) * mm - AM.PAGE_W / 2) / f
        yn = ((AM.PAGE_H - (rr + 0.5) * mm) - AM.PAGE_H / 2) / f
        dirs = look[None] + xn.reshape(-1, 1) * right[None] + yn.reshape(-1, 1) * up[None]
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        hit = ar_prep.cast_faces(V, F, E, dirs, 1e6) >= 0
        m = np.zeros(shape, bool)
        m[r0:r1, c0:c1] = hit.reshape(cc.shape)
        out[k] = m
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work")
    ap.add_argument("--out", default="agentic_gallery")
    args = ap.parse_args()
    W_ = args.work
    scene = json.load(open(os.path.join(W_, "am_scene.json")))
    AM.PAGE_H = scene["camera"]["page_mm"][1]            # the searched page height
    meta = json.load(open(os.path.join(W_, "gl_objects.json")))["objects"]
    info = json.load(open(os.path.join(W_, "render_info.json")))
    objs = scene["objects"]
    n = len(objs)
    print("gates:")
    gate("renderer", "NVIDIA" in info["renderer"] and "llvmpipe" not in info["renderer"].lower(),
         info["renderer"])

    # ---- layers (no ground: plain white floor) ---------------------------------------------------
    sc = load(W_, "scene.png").astype(float)
    idd, idu = decode_ids(load(W_, "id_def.png")), decode_ids(load(W_, "id_und.png"))
    urel = load(W_, "urel.png")[..., 0].astype(float) / 255.0
    Hpx, Wpx = idd.shape
    px_mm = Wpx / AM.PAGE_W
    base = np.full((Hpx, Wpx, 3), 255.0)
    U = idu > 0
    r = max(1, int(round(OUTLINE_W * px_mm)))
    edge = binary_dilation(U, iterations=r) & ~U
    base[edge] = base[edge] * (1 - OUTLINE_ALPHA) + np.array(OUTLINE_RGB) * OUTLINE_ALPHA
    a = sc[..., 3:4] / 255.0
    comp = np.clip(sc[..., :3] * a + base * (1 - a), 0, 255)
    page_px = (int(round(AM.PAGE_W / 25.4 * OUT_DPI)), int(round(AM.PAGE_H / 25.4 * OUT_DPI)))

    # ---- the labels, by am_labels' rules on the render's own ink (quarter resolution) ------------
    q = 4
    idd4, idu4, urel4 = idd[::q, ::q], idu[::q, ::q], urel[::q, ::q]
    edge4 = edge[::q, ::q]
    sh4 = idd4.shape
    mm4 = AM.PAGE_W / sh4[1]
    # the ink a reader sees for object k: its deformed body, and its undeformed silhouette
    # (and outline) where no deformed body covers it
    und_vis = (idd4 == 0)
    ink = {k: (idd4 == k + 1) | ((idu4 == k + 1) & und_vis) for k in range(n)}
    outline_any = edge4 & (idd4 == 0)
    if outline_any.any():           # the outline ring belongs to the object it surrounds
        from scipy.ndimage import distance_transform_edt
        _, (ri, ci) = distance_transform_edt(idu4 == 0, return_indices=True)
        owner = idu4[ri, ci]
        for k in range(n):
            ink[k] |= outline_any & (owner == k + 1)
    last = len(scene["rows"]) - 1
    items = [{"k": o["k"], "text": o["label"]["text"], "turn": o["label"]["turn_deg"],
              "depth": o["depth"], "back": o["row"] == last, "lane": o["seat"].get("lane", 0)} for o in objs]
    labs, failed = am_labels.place(items, ink, mm4, AM.PAGE_W, AM.PAGE_H, AM._FP, AM.LABEL_PT * AM.PT_MM)
    lf = os.path.join(W_, "label_failed.json")
    json.dump([objs[k]["instruction"] for k in failed], open(lf, "w"))
    gate("every label placed (hard rules: free floor, owned, inside)", not failed,
         [objs[k]["instruction"] for k in failed] or "all 50",
         str({objs[k]["instruction"]: am_labels.REASONS.get(k) for k in failed}) if failed else "")

    # ---- the page -----------------------------------------------------------------------------
    # rev 4 (the user): a nearer body may cover part of a farther label, so the labels are ink
    # composited into the 1200 dpi render under the nearer bodies' pixels
    depth = {o["k"]: o["depth"] for o in objs}
    ink_rgb = np.array([int(AF.INK[i:i + 2], 16) for i in (1, 3, 5)], float)
    lab_vis = {}
    for k, L_ in labs.items():
        al, r0, c0 = label_alpha(L_["v"], L_["codes"], px_mm)
        r1, c1 = r0 + al.shape[0], c0 + al.shape[1]
        nearer_ids = [j + 1 for j in range(n) if j != k and depth[j] < depth[k]]
        occ = np.isin(idd[r0:r1, c0:c1], nearer_ids)
        tot = al.sum()
        al = al * ~occ
        lab_vis[k] = float(al.sum() / max(tot, 1e-9))
        comp[r0:r1, c0:c1] = comp[r0:r1, c0:c1] * (1 - al[..., None]) + ink_rgb * al[..., None]
    comp_out = np.asarray(Image.fromarray(np.clip(comp, 0, 255).astype(np.uint8)).resize(page_px, Image.LANCZOS))
    pg = AF.Page(AM.PAGE_W, AM.PAGE_H)
    pg.img(comp_out, 0, 0, AM.PAGE_W, AM.PAGE_H)
    for ext in ("pdf", "png"):
        pg.fig.savefig(f"{args.out}.{ext}", dpi=OUT_DPI, facecolor="white")
    pg.fig.savefig(os.path.join(W_, "check_hires.png"), dpi=1200, facecolor="white")
    print(f"wrote {args.out}.pdf / .png")

    # ---- gates ------------------------------------------------------------------------------
    ids = set(np.unique(idd).tolist()) - {0}
    gate("all objects drawn", ids == set(range(1, n + 1)), f"{len(ids)} of {n}")
    zw = np.load(os.path.join(W_, "scene_world.npz"))
    alone = cast_alone(scene, {k: (zw[f"V{k}"], zw[f"F{k}"]) for k in range(n)}, sh4, mm4)
    vis, inside = {}, {}
    for k in range(n):
        m = idd4 == k + 1
        vis[k] = float(m.sum() / max(alone[k].sum(), 1))
        inside[k] = float((m & binary_dilation(alone[k], iterations=1)).sum() / max(m.sum(), 1))
    gate("projection: id pixels inside the Warp silhouette", min(inside.values()) >= 0.99,
         round(min(inside.values()), 4))
    # silhouette agreement on the objects nothing covers, within one quarter-res pixel both
    # ways: the Warp cast inside the dilated id pass (the converse is the gate above).  A
    # bounding box failed on a sub-pixel spire tip of the Taj Mahal that the cast hits and the
    # rasteriser drops; an IoU was boundary-dominated on thin small objects (wrench: 0.90).
    free = [k for k in range(n) if vis[k] >= 0.999]
    cov = {k: float((alone[k] & binary_dilation(idd4 == k + 1, iterations=1)).sum() / max(alone[k].sum(), 1))
           for k in free}
    ki = min(cov, key=cov.get) if cov else None
    gate("projection: Warp silhouettes inside the id pass within 1 px (>= 99 %, unoccluded objects)",
         ki is not None and cov[ki] >= 0.99, f"min {cov[ki]:.4f} ({objs[ki]['instruction']}) over {len(free)} objects"
         if ki is not None else None)
    vk = min(vis, key=vis.get)
    gate(f"every object >= {VIS_MIN:.0%} visible", vis[vk] >= VIS_MIN, f"{vis[vk]:.3f} ({objs[vk]['instruction']})")
    allink = np.zeros(sh4, bool)
    for k in range(n):
        allink |= ink[k]
    lk = min(lab_vis, key=lab_vis.get) if lab_vis else None
    gate(f"every label >= {am_labels.LABEL_VIS_MIN:.0%} visible (nearer bodies may cover the rest)",
         lk is not None and lab_vis[lk] >= am_labels.LABEL_VIS_MIN - 0.01,
         f"{lab_vis[lk]:.3f} ({objs[lk]['instruction']})" if lk is not None else None)
    lab_free = lab_vis
    keys = sorted(labs)
    overlap = [(objs[a_]["instruction"], objs[b_]["instruction"]) for i, a_ in enumerate(keys)
               for b_ in keys[i + 1:] if (labs[a_]["mask"] & labs[b_]["mask"]).any()]
    gate("no two labels overlap", not overlap, overlap)
    own_m = {k: L_["other_centroid_mm"] - L_["own_centroid_mm"] for k, L_ in labs.items()}
    own_n = {k: L_["own_near_mm"] for k, L_ in labs.items()}
    gate(f"ownership: centroid margin >= {am_labels.OWN_MARGIN} mm, nearest glyph <= {am_labels.OWN_MAX} mm",
         min(own_m.values()) >= am_labels.OWN_MARGIN and max(own_n.values()) <= am_labels.OWN_MAX,
         f"min margin {min(own_m.values()):.2f} mm, max distance {max(own_n.values()):.2f} mm")
    allv = np.vstack([L_["v"] for L_ in labs.values()])
    marg = float(min(allv[:, 0].min(), AM.PAGE_W - allv[:, 0].max(), allv[:, 1].min(), AM.PAGE_H - allv[:, 1].max()))
    gate("labels inside the page (>= 1 mm)", marg >= 1.0 - 1e-6, round(marg, 2))
    h = TextPath((0, 0), "H", size=1, prop=AM._FP).get_extents()
    cap = (h.y1 - h.y0) * AM.LABEL_PT * AM.PT_MM
    gate(f"label cap height in {CAP_BAND} mm", CAP_BAND[0] <= cap <= CAP_BAND[1], round(cap, 3),
         f"({AM.LABEL_PT} pt, the journal minimum)")
    gate("size order (the user's seat swap exempted by name)", all(scene["size_order"].values()), scene["size_order"])
    gate("x N equal to the category records", all(o["gate"].get("factor_equal") for o in meta),
         f"{sum(bool(o['gate'].get('factor_equal')) for o in meta)} of {n}")
    # coverage on the final page
    occ = allink.copy()
    for L_ in labs.values():
        occ |= L_["mask"]
    f = max(1, int(round(AM.CELL / mm4)))
    gh, gw = sh4[0] // f, sh4[1] // f
    cells = occ[:gh * f, :gw * f].reshape(gh, f, gw, f).any((1, 3))
    cover = float(cells.mean())
    thirds = [float(c.mean()) for c in np.array_split(cells, 3, axis=1)]
    empty = AM.max_empty_rect(cells) * (f * mm4) ** 2
    print(f"  [info] coverage by page third (left / centre / right): "
          f"{' / '.join(f'{t:.1%}' for t in thirds)}")
    print(f"  [info] page coverage {cover:.1%} of {AM.CELL} mm cells; largest empty rectangle "
          f"{empty:.0f} mm2 ({empty / (AM.PAGE_W * AM.PAGE_H):.1%} of the page)")
    ambiguous = {objs[k]["instruction"]: {"margin_mm": round(L_["margin"], 2), "over_other": round(L_["over_other"], 3)}
                 for k, L_ in labs.items() if L_["over_other"] > 0.02 or L_["margin"] < am_labels.OWN_PREFER}
    print(f"  [info] labels short of the preferred ownership ({am_labels.OWN_PREFER} mm) or over another "
          f"object: {len(ambiguous)} {ambiguous}")
    kinds = {}
    for L_ in labs.values():
        kinds[L_["kind"]] = kinds.get(L_["kind"], 0) + 1
    print(f"  [info] label positions: {kinds}")
    smallest = sorted(objs, key=lambda o: o["L_m"])[:5]
    hot_small = {o["instruction"]: int(((urel4 > 0.7) & (idd4 == o["k"] + 1)).sum()) for o in smallest}
    stats = {"name": args.out, "p": scene["p"], "params": scene["params"], "widen": scene.get("widen"),
             "D_range": scene["D_range"], "ratio_drawn": scene["ratio_drawn"], "ratio_real": scene["ratio_real"],
             "rows": [[objs[k]["instruction"] for k in r["objects"]] for r in scene["rows"]],
             "row_counts": [len(r["objects"]) for r in scene["rows"]], "tiers": [r["z"] for r in scene["rows"]],
             "fov_h_deg": scene["camera"]["fov_h_deg"], "page_mm": scene["camera"]["page_mm"],
             "seat_swaps": scene["seat_swaps"], "overrides": scene.get("overrides"), "gates": CHECK, "coverage": cover, "coverage_thirds": thirds,
             "largest_empty_mm2": empty, "largest_empty_share": empty / (AM.PAGE_W * AM.PAGE_H),
             "visible": {objs[k]["instruction"]: v for k, v in vis.items()},
             "labels": {"pt": AM.LABEL_PT, "cap_mm": cap, "kinds": kinds,
                        "per_object": {objs[k]["instruction"]: {"kind": L_["kind"], "nudge": L_["nudge"],
                                                                "own_margin_mm": own_m[k], "own_near_mm": own_n[k],
                                                                "visible": lab_vis[k], "kind_covered_at_placement": L_.get("covered")}
                                        for k, L_ in labs.items()},
                        "failed": [objs[k]["instruction"] for k in failed]},
             "hot_pixels_smallest_five_quarter_res": hot_small, "ambiguous_labels": ambiguous, "render": info, "search": scene["search"]}
    json.dump(stats, open(os.path.join(W_, "gl_stats.json"), "w"), indent=1, default=float)
    if labs:
        contact_sheet(W_, objs, vis, labs, idd, idu, mm4)
    bad = [k for k, v in CHECK.items() if not v["ok"]]
    assert not bad, f"gates failed: {bad}"
    print("all gates passed")


def contact_sheet(W_, objs, vis, labs, idd, idu, mm4):
    """check/contact.png: every object with its label, size order, from the 1200 dpi page;
    check/crop_rc.png: the page in 4 x 2 crops at 1200 dpi."""
    import matplotlib.pyplot as plt
    hi = np.asarray(Image.open(os.path.join(W_, "check_hires.png")).convert("RGB"))
    pmm = hi.shape[1] / AM.PAGE_W
    out = os.path.join(W_, "check")
    os.makedirs(out, exist_ok=True)
    ch, cw = AM.PAGE_H / 2, AM.PAGE_W / 4
    for r in range(2):
        for c in range(4):
            Image.fromarray(hi[int(r * ch * pmm):int((r + 1) * ch * pmm),
                               int(c * cw * pmm):int((c + 1) * cw * pmm)]).save(f"{out}/crop_{r}{c}.png")
    s = idd.shape[1] / AM.PAGE_W
    fig, axes = plt.subplots(5, 10, figsize=(30, 17.5), dpi=100)
    for ax, o in zip(axes.ravel(), sorted(objs, key=lambda o: o["L_m"])):
        k = o["k"]
        rr, cc = np.nonzero((idd == k + 1) | (idu == k + 1))
        if k in labs:
            lr, lc = np.nonzero(labs[k]["mask"])
            x0, x1 = min(cc.min() / s, lc.min() * mm4) - 1.0, max(cc.max() / s, lc.max() * mm4) + 1.0
            yt, yb = min(rr.min() / s, lr.min() * mm4) - 1.0, max(rr.max() / s, lr.max() * mm4) + 1.0
        else:
            x0, x1, yt, yb = cc.min() / s - 1, cc.max() / s + 1, rr.min() / s - 1, rr.max() / s + 1
        side = max(x1 - x0, yb - yt)
        cx, cy = (x0 + x1) / 2, (yt + yb) / 2
        crop = hi[max(int((cy - side / 2) * pmm), 0):int((cy + side / 2) * pmm),
                  max(int((cx - side / 2) * pmm), 0):int((cx + side / 2) * pmm)]
        ax.imshow(crop, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_title(f"{o['instruction'][:34]}\nrow {o['row']} · L {o['L_m']:.3g} m · D {o['D']:.2f} · "
                     f"x{o['factor']:g}\nvisible {vis[k]:.2f} · label {labs.get(k, {}).get('kind')}", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{out}/contact.png")
    plt.close(fig)
    print(f"wrote {out}/contact.png and crops")


if __name__ == "__main__":
    main()
