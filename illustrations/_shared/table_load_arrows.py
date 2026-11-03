"""Load arrows for the table's "Geometry + BC" station (conda `mechfoundry-warp117`).

    python ../_shared/table_load_arrows.py PREFIX OUT.stl     # e.g. so_ / hp_full_

Reads PREFIX{body,bc_dir,bc_neu}.stl (shapeopt: so_body / so_bc_dir / so_bc_neu) or
PREFIX{body,dir,neu}.stl (hiddenphi: hp_full_*), and writes the arrows as ONE STL
that the pvbatch renderers draw over the surface (render_so.panel_restore_bc,
render_hp.loop_panel("geom")).

The arrows are agentic_results' load arrows, not a look-alike: the mesh is
``ar_prep.arrow_mesh`` and the proportions are ``ar_prep.ARROW_*`` x the body's
largest extent, so a table arrow and an agentic-results arrow are the same glyph.
They point along the run's traction, (0, 0, -0.02) -> straight down, TIP on the loaded
top (a push: ar_prep's convention), the length fixed relative to the body (the
station shows direction, not magnitude).

Sites, not ar_prep's search: the top is one flat, fully visible disk, so the sites
are a fixed quincunx -- the centre plus four on a ring at RING x the top's radius,
the ring turned PHASE degrees off the camera's azimuth.  On the 90-degree grid a
front arrow's shaft prints straight through the arrow behind it (they share a
screen column); turned 22.5 degrees the five arrows sit in five separate columns.
The printed gap between every pair is asserted against ar_prep.ARROW_GAP, the
agentic figures' own spacing floor.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
import ar_prep  # noqa: E402  (ar_figure first, then trimesh: see ar_prep's import note)
import trimesh  # noqa: E402

DIRECTION = np.array([0.0, 0.0, -1.0])     # hp_prep.TRACTION / so_prep: (0, 0, -0.02)
VIEW_DIR = np.array([0.42, -1.1, 0.40])    # render_so.VIEW_DIR == render_hp.CUBE_VIEW_DIR
RING = 0.58                                # ring radius / the top's radius
PHASE = 22.5                               # degrees off the camera azimuth (see above)


def _parts(prefix):
    names = (("body", "bc_dir", "bc_neu") if os.path.exists(prefix + "bc_neu.stl")
             else ("body", "dir", "neu"))
    return [trimesh.load(prefix + n + ".stl", process=False) for n in names]


def build(prefix, out):
    body, _dir, neu = _parts(prefix)
    allv = np.vstack([body.vertices, _dir.vertices, neu.vertices])
    L = float((allv.max(0) - allv.min(0)).max())          # ar_prep's "x extent"
    c = neu.vertices[:, :2].mean(0)
    rad = 0.5 * float((neu.vertices[:, :2].max(0) - neu.vertices[:, :2].min(0)).min())
    az = np.degrees(np.arctan2(VIEW_DIR[1], VIEW_DIR[0]))
    sites = [c] + [c + RING * rad * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])
                   for a in az + PHASE + 90.0 * np.arange(4)]
    # each tip ON the loaded top: a ray straight down onto the Neumann surface
    org = np.array([[s[0], s[1], allv[:, 2].max() + 1.0] for s in sites])
    loc, ray, _ = neu.ray.intersects_location(org, np.tile(DIRECTION, (len(org), 1)))
    arrows = []
    for k in range(len(sites)):
        hits = loc[ray == k]
        assert len(hits), f"site {k} misses the loaded top"
        tip = hits[hits[:, 2].argmax()]
        tail = tip - ar_prep.ARROW_LEN * L * DIRECTION
        arrows.append((tail, tip))
    gaps = [ar_prep.printed_gap(a[0], a[1], b[0], b[1], VIEW_DIR)
            for i, a in enumerate(arrows) for b in arrows[i + 1:]]
    assert min(gaps) >= ar_prep.ARROW_GAP * L, f"arrows print {min(gaps):.3f} apart"
    V, F = [], []
    for tail, tip in arrows:
        v, f = ar_prep.arrow_mesh(tail, tip, L)
        F.append(f + sum(len(x) for x in V))
        V.append(v)
    trimesh.Trimesh(np.vstack(V), np.vstack(F), process=False).export(out)
    print(f"wrote {out}: {len(arrows)} arrows, length {ar_prep.ARROW_LEN * L:.3f} "
          f"(L {L:.3f}), min printed gap {min(gaps):.3f} >= {ar_prep.ARROW_GAP * L:.3f}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
