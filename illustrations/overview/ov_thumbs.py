"""Prep step 1 (conda ``mechfoundry-warp117``, CPU): thumbnails for the method-overview figure.

The overview summarises figures that are already built, so it must not re-render
anything.  It also must not *crop* them: every panel of every source figure is
its own embedded image object inside that figure's PDF, because the compose
scripts ``imshow`` the ParaView render and draw the panel title / in-panel label
as vector text on top.  ``pdfimages`` therefore hands back the **bare render**,
at the dpi the source figure was saved with (600, or 340 for
``bunny_sdf_octree``) — higher than the root PNGs, and with no text to crop
around.

    pdfimages -png -f 1 -l 1 ../hidden_phi_loop.pdf <prefix>

Extraction is by image INDEX, which is stable only while a source figure's panel
set is.  So every station asserts the extracted image's ``(w, h)`` and a hash of
its raw pixels; a source figure rebuilt with a different panel count, size or
order fails the build here instead of silently putting the wrong picture on the
page.  ``--sheet`` writes a labelled contact sheet of every image in every source
so the indices can be re-mapped when that happens.

Writes ``ov_t_<station>.png`` (+ ``ov_thumbs.json`` provenance) into this folder.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# --------------------------------------------------------------------------
# station -> (source figure, image index, expected (w, h), pixel sha1[:12])
#
# The hashes are filled in by the first run (they print as a ready-to-paste
# block) and are what make a changed source loud rather than silent.
# --------------------------------------------------------------------------
SOURCES = {
    # --- block A: the bunny, implicit geometry -> FE analysis ---------------
    "sdf":         ("bunny_sdf_octree",  0, (1083, 1095),  "1c1509221061"),
    "octree":      ("bunny_sdf_octree",  1, (1083, 1095),  "2d30fcd138dc"),
    # the diffuse interface alone, then the same field carrying the band and the
    # two BC sets (bc_region_id's own panel is no longer used: the overview shows
    # the BC sets where they land, on the mesh, not the surface selection step)
    "phi":         ("diffused_nitsche",  1, (629, 629),    "96e62ae3f003"),
    "bandbc":      ("diffused_nitsche",  3, (629, 629),    "8fb9bab8738e"),
    # --- block B: the table, one design iteration ---------------------------
    # (bc_zip_workflow's panel is no longer used: the BC restoration is named on
    # the arrow that produces the new geometry rather than pictured)
    # hidden_phi_loop is an ACTIVE target: it was rebuilt three times while this
    # figure was written (911x672 -> 732x612 -> 824x689 -> 814x715), and one of
    # those rebuilds ALSO swapped (d)'s two halves, moving the dip panel 5 -> 6.
    # The size assertion caught every one; `--sheet` plus one look re-maps it in
    # a minute.  Expect to redo this after each rebuild over there — and check
    # the ORDER on the sheet, not just the sizes.
    "initial":     ("hidden_phi_loop",   0, (832, 667),    "ecf88fe485fb"),
    "solve":       ("hidden_phi_loop",   2, (836, 667),    "41bb557e2861"),
    "grad_d":      ("hidden_phi_loop",   3, (832, 667),    "d897c7aba1b0"),
    "grad_phi":    ("hidden_phi_loop",   4, (832, 667),    "2bb6b0743ec5"),
    "upd_d":       ("hidden_phi_loop",   5, (832, 667),    "3c43b0e8cb7a"),
    "upd_phi":     ("hidden_phi_loop",   6, (832, 667),    "9b6635b6e14a"),
    "result":      ("hidden_phi_loop",   7, (983, 800),    "44d9442b9afe"),
}

# (The colour bars are embedded images too — indices 8-11 of hidden_phi_loop —
# but the overview quotes ranges in words rather than drawing bars, so none is
# extracted.  `--sheet` lists them if one is ever wanted.)
BARS = {}


def _extract(stem, workdir):
    """All page-1 images of ``<ROOT>/<stem>.pdf``, in object order."""
    pdf = os.path.join(ROOT, f"{stem}.pdf")
    if not os.path.isfile(pdf):
        raise SystemExit(
            f"missing source figure {pdf}\n"
            f"  build it first (./build.sh <target>); the overview only reads "
            f"deliverables, it never rebuilds them."
        )
    prefix = os.path.join(workdir, stem)
    subprocess.run(
        ["pdfimages", "-png", "-f", "1", "-l", "1", pdf, prefix],
        check=True,
    )
    out = sorted(
        f for f in os.listdir(workdir)
        if f.startswith(os.path.basename(prefix) + "-") and f.endswith(".png")
    )
    return [os.path.join(workdir, f) for f in out]


def _pixel_hash(im):
    """Hash the RAW pixels, not the file — poppler's PNG encoding is not the
    thing we are pinning, the picture is."""
    return hashlib.sha1(im.convert("RGB").tobytes()).hexdigest()[:12]


def sheet(workdir):
    """Labelled contact sheet per source, for re-mapping indices."""
    for stem in sorted({s[0] for s in list(SOURCES.values()) + list(BARS.values())}):
        files = _extract(stem, workdir)
        TH, cols = 210, 6
        rows = (len(files) + cols - 1) // cols
        out = Image.new("RGB", (cols * (TH + 10) + 10, rows * (TH + 28) + 10), "white")
        d = ImageDraw.Draw(out)
        for i, f in enumerate(files):
            src = Image.open(f)
            im = src.convert("RGB")
            im.thumbnail((TH, TH))
            x, y = 10 + (i % cols) * (TH + 10), 10 + (i // cols) * (TH + 28)
            out.paste(im, (x, y))
            d.rectangle([x - 1, y - 1, x + im.size[0], y + im.size[1]], outline="#bbbbbb")
            d.text((x, y + im.size[1] + 4),
                   f"{i}  {src.size[0]}x{src.size[1]}  {_pixel_hash(src)}", fill="black")
        p = os.path.join(HERE, f"ov_sheet_{stem}.png")
        out.save(p)
        print(f"wrote {p}  ({len(files)} images)")


def bless(workdir):
    """Take the CURRENT sizes and hashes at the recorded indices.

    The index mapping is NOT re-derived: it cannot be, reliably.  Matching a
    station to its panel by image signature was built and then measured — the
    panels are near-duplicates of each other (the design `d` before and after a
    gradient step differ by a hair) and each rebuild of the source shifts every
    panel's appearance, so the best match for `initial` came out as index 6
    rather than its own index 0.  A tool that guesses there would hand back
    exactly the silent substitution this whole mechanism exists to prevent.

    So: look at `--sheet` first, confirm the ORDER by eye, then `--bless` does
    the tedious half — reading off eleven sizes and hashes without a typo.
    """
    print("BLESS: this trusts the recorded indices.  Run --sheet and check the\n"
          "       panel ORDER first; --bless only refreshes sizes and hashes.\n")
    cache, updates = {}, {}
    for name, (stem, idx, wh, want) in SOURCES.items():
        if stem not in cache:
            cache[stem] = _extract(stem, workdir)
        files = cache[stem]
        if idx >= len(files):
            raise SystemExit(f"{name}: {stem}.pdf has only {len(files)} images")
        im = Image.open(files[idx])
        h = _pixel_hash(im)
        if im.size == tuple(wh) and h == want:
            print(f"  {name:9s} unchanged")
            continue
        updates[name] = (stem, idx, im.size, h)
        print(f"  {name:9s} {stem}.pdf[{idx}]  {wh[0]}x{wh[1]} -> "
              f"{im.size[0]}x{im.size[1]}")
    if not updates:
        print("nothing to bless")
        return
    out = []
    for line in open(__file__).read().splitlines(True):
        st = line.strip()
        if st.startswith('"') and '(' in st and st.split('"')[1] in updates:
            n = st.split('"')[1]
            stem, idx, size, h = updates[n]
            wh = f"({size[0]}, {size[1]})"
            line = (f'    "{n}":{" " * max(1, 12 - len(n))}("{stem}",'
                    f'{" " * max(1, 18 - len(stem))}{idx}, {wh},'
                    f'{" " * max(1, 14 - len(wh))}"{h}"),\n')
        out.append(line)
    open(__file__, "w").write("".join(out))
    print(f"\nblessed {len(updates)} station(s) in {os.path.basename(__file__)}")


def main():
    workdir = tempfile.mkdtemp(prefix="ov_thumbs_")
    try:
        if "--sheet" in sys.argv:
            sheet(workdir)
            return
        if "--bless" in sys.argv:
            bless(workdir)
            return

        cache, manifest, drift = {}, {}, []
        for name, (stem, idx, wh, want_hash) in {**SOURCES, **BARS}.items():
            if stem not in cache:
                cache[stem] = _extract(stem, workdir)
            files = cache[stem]
            if idx >= len(files):
                raise SystemExit(
                    f"{name}: {stem}.pdf has {len(files)} embedded images, "
                    f"index {idx} requested.  Re-map with --sheet."
                )
            im = Image.open(files[idx])
            got_hash = _pixel_hash(im)
            if im.size != wh:
                raise SystemExit(
                    f"{name}: {stem}.pdf image {idx} is {im.size}, expected {wh}.\n"
                    f"  That source figure was rebuilt.  Check the panel ORDER "
                    f"with `ov_thumbs.py --sheet`, then `--bless` to take the "
                    f"new sizes and hashes at the recorded indices."
                )
            if got_hash != want_hash:
                drift.append((name, stem, idx, want_hash, got_hash))
            dst = os.path.join(HERE, f"ov_t_{name}.png")
            im.convert("RGB").save(dst)
            manifest[name] = {
                "source": f"{stem}.pdf", "image_index": idx,
                "size": list(im.size), "pixel_sha1": got_hash,
            }
            print(f"  {name:9s} <- {stem}.pdf[{idx}]  {im.size[0]}x{im.size[1]}  {got_hash}")

        if drift:
            print("\nPixel hashes differ from the recorded ones:")
            for name, stem, idx, want, got in drift:
                print(f"  {name}: {stem}.pdf[{idx}]  recorded {want}  now {got}")
            print("\nIf that source figure was legitimately re-run, paste these in:")
            for name, stem, idx, _, got in drift:
                wh = {**SOURCES, **BARS}[name][2]
                print(f'    "{name}": ("{stem}", {idx}, {wh}, "{got}"),')
            raise SystemExit(
                "\nrefusing to compose from unverified thumbnails — a source "
                "figure changed under this figure's feet."
            )

        with open(os.path.join(HERE, "ov_thumbs.json"), "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"wrote {len(manifest)} thumbnails + ov_thumbs.json")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
