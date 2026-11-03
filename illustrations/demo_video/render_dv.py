"""Stage 2 (pvbatch + EGL, GPU): every 3-D pass of the demo video, one PNG per frame.

    pvbatch render_dv.py WORK [--scale S] [--step N] [--part bench|tiles|all]

after dv_prep.py.  The looks are agentic_results' own panels, built by
render_ar's functions (imported, not copied), each in ONE long-lived view whose
camera is moved every frame (the spin) -- nothing is re-created per frame:

  clay / paint / seg / E   render_ar.surface_panel (AR_SURF_VIEW kit, SURF_FINISH)
  bc                       render_ar.bc_panel, support tubes 2.5x the figure's radius
  solve                    render_ar.solve_panel (AR_SOLVE_VIEW, |u| ramp); the
                           WarpByVector factor is set per frame to s(t)
  octree                   the |u| surface inside the solve's FULL adaptive octree
                           (vtu_sdf/) cut exactly by the camera-facing plane through
                           the orbit centre, far half kept: fill without edges, the
                           cube edges and the cut-face cell outlines as clipped lines
  arrows                   the 8 load arrows alone (the compose lays them on top)
  grid                     the UI floor grid (ar_prep.ui_grid, as render_ar.add_ground
                           draws it), at 2x with 2x-wide lines, downsampled in the compose
  tiles                    render_ar.solve_panel per object at 2x its body area, plus its
                           undeformed pass (factor 0) for the ghost

Each pass is saved with a transparent background (straight alpha) and is rendered
only on frames where dv_timeline gives it a weight.  Frames that exist are skipped,
so an interrupted run resumes.  ``--scale 0.5 --step 2`` is the draft.
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
sys.path.insert(0, HERE)
from paraview.simple import *  # noqa: F403,E402
import render_ar as R  # noqa: E402
import ar_style as A  # noqa: E402
import dv_timeline as TL  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("work")
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--step", type=int, default=1)
ap.add_argument("--part", default="all", choices=("bench", "tiles", "all"))
ap.add_argument("--frames-dir", default=None, help="default WORK/frames")
ap.add_argument("--only", default=None, help="comma-separated frame numbers (checks)")
args = ap.parse_args()
WORK = os.path.abspath(args.work)
S = args.scale
FR = os.path.abspath(args.frames_dir or os.path.join(WORK, "frames"))


def view_dir(heading, elev):
    a, e = math.radians(heading), math.radians(elev)
    return [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)]


def set_camera(rv, c, D, heading, elev):
    d = view_dir(heading, elev)
    rv.CameraFocalPoint = list(c)
    rv.CameraPosition = [c[i] + D * d[i] for i in range(3)]
    rv.CameraViewUp = A.VIEW_UP
    rv.CameraViewAngle = A.VIEW_ANGLE
    rv.CameraParallelProjection = 0
    return d


def finish_view(rv):
    rv.Background = A.BG
    R._try(lambda: setattr(rv, "UseColorPaletteForBackground", 0))


def shoot(rv, path, size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Render(rv)  # noqa: F405
    SaveScreenshot(path, rv, ImageResolution=size, TransparentBackground=1)  # noqa: F405


def sized(w, h):
    size = [int(round(w)), int(round(h))]
    R.SIZE[:] = size
    return size


def last_warp(path):
    """The WarpByVector render_ar.solve_panel just built on ``path``."""
    hits = [p for p in GetSources().values()  # noqa: F405
            if p.GetXMLName() == "WarpVector" and path in list(p.Input.FileNames)]
    assert hits, f"no WarpByVector on {path}"
    return hits[-1]


OCT_FACE = [0.94, 0.94, 0.95]     # the cut octree's fill: near-white, flat, lit by the kit
OCT_EDGE = [0.45, 0.48, 0.53]     # the kept cells' edges, 1 px
OCT_CUT = [0.20, 0.22, 0.26]      # the cells' outlines on the cut face, 2 px, darker


# The octree cut is a plane fixed to the bench (the user, 2026-10-26), through the orbit
# centre: the front/back mid-plane's normal +z turned 90 deg clockwise seen from above
# ("rotated 90 degs clockwise"), i.e. -x, the bench's lateral axis.  Clip keeps the side
# the normal points away from (+x): the half farther from the camera through the octree
# hold, whose headings (-76 .. -104 deg) put the camera on the -x side.  Being
# axis-aligned, it cuts every cube in a square.
CUT_NORMAL = [-1.0, 0.0, 0.0]


def _plane(flt, centre, attr="ClipType"):
    setattr(flt, attr, "Plane")
    getattr(flt, attr).Origin = list(centre)
    getattr(flt, attr).Normal = CUT_NORMAL


def _lines(src, rv, rgb, width):
    d = Show(src, rv)  # noqa: F405
    d.Representation = "Surface"
    d.ColorArrayName = ["POINTS", ""]
    d.AmbientColor = rgb
    d.DiffuseColor = rgb
    d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    R._try(lambda: setattr(d, "LineWidth", max(1.0, width * S)))
    R._try(lambda: setattr(d, "RenderLinesAsTubes", 0))
    return d


def octree_view(grid, faces, surface, umax, centre):
    """The |u| surface (render_ar.solve_panel at s = 1, the solve look itself, whole)
    inside the solve's adaptive octree, COMPLETE over a box (dv_prep.full_octree:
    ar_0_octree.vtk, every leaf, 11 mm at the surface to 182 mm far out) -- CUT by the
    bench's lateral mid-plane (CUT_NORMAL, body-fixed), the far half kept (the user:
    cells straddling the plane are cut, not kept whole; the plane is fixed to the bench).
    Fill and lines are separate, so no triangulation shows:
      fill   Clip (exact) of the hex grid, a surface WITHOUT edges, near-white and flat:
             the cut face reads as one plane;
      edges  ExtractEdges of the uncut grid, clipped by the plane: every kept cube edge
             as an exact segment, 1 px;
      cut    the cells' quad faces (dv_prep's ar_0_octfaces.vtk) sliced by the plane:
             each cut cell's own outline on the cut face, 2 px and darker, so the
             section shows the refinement levels as a clean quadtree pattern.
    The complete box's 12-edge frame is drawn by the compose (dv_compose.box_frame).
    Returns the view and the three plane filters (the plane is fixed in the body)."""
    rv = R.solve_panel(surface, 1.0, umax)
    src = LegacyVTKReader(FileNames=[grid])  # noqa: F405
    fill = Clip(Input=src)  # noqa: F405
    edges = Clip(Input=ExtractEdges(Input=src))  # noqa: F405
    for flt in (fill, edges):
        _plane(flt, centre)
        R._try(lambda flt=flt: setattr(flt, "Invert", 1))      # the +x half (far from the camera)
        R._try(lambda flt=flt: setattr(flt, "Crinkleclip", 0))
    cut = Slice(Input=LegacyVTKReader(FileNames=[faces]))  # noqa: F405
    _plane(cut, centre, "SliceType")
    d = Show(fill, rv)  # noqa: F405
    d.Representation = "Surface"
    d.ColorArrayName = ["POINTS", ""]
    d.DiffuseColor = OCT_FACE
    d.AmbientColor = OCT_FACE
    for k, v in A.SOLVE_FINISH.items():
        setattr(d, k, v)
    d.Specular = 0.0
    d.Interpolation = "Flat"
    _lines(edges, rv, OCT_EDGE, 1.0)
    _lines(cut, rv, OCT_CUT, 2.0)
    finish_view(rv)
    return rv, (fill, edges, cut)


def grid_view(path, size):
    rv = R.new_view(A.AR_SURF_VIEW)
    rv.ViewSize = size
    d = Show(LegacyVTKReader(FileNames=[path]), rv)  # noqa: F405
    d.Representation = "Surface"
    ColorBy(d, ("POINTS", "rgba"))  # noqa: F405
    d.MapScalars = 0
    d.Ambient, d.Diffuse, d.Specular = 1.0, 0.0, 0.0
    d.LineWidth = 4.0 * S          # at 2x: 2 px lines after the compose's 2x downsample
    d.SetScalarBarVisibility(rv, False)
    finish_view(rv)
    return rv


def arrows_view(path):
    """The load arrows alone, lit as render_ar.bc_panel lights them (normals at 60 deg,
    ARROW_FINISH, their own cell colour); the compose draws this pass over the body."""
    rv = R.new_view(A.AR_SURF_VIEW)
    nrm = GenerateSurfaceNormals(Input=LegacyVTKReader(FileNames=[path]))  # noqa: F405
    R._try(lambda: setattr(nrm, "FeatureAngle", 60))
    d = Show(nrm, rv)  # noqa: F405
    d.Representation = "Surface"
    for k, v in A.ARROW_FINISH.items():
        setattr(d, k, v)
    ColorBy(d, ("CELLS", "rgb"))  # noqa: F405
    d.MapScalars = 0
    d.SetScalarBarVisibility(rv, False)
    return rv


def render_bench(st, times):
    b = st["bench"]
    f = lambda name: os.path.join(WORK, f"ar_0_{name}.vtk")  # noqa: E731
    size = sized(TL.W * S, TL.H * S)
    sup = f("support") if os.path.exists(f("support")) else None
    views = {"clay": R.surface_panel(f("clay"), "POINTS"),
             "paint": R.surface_panel(f("painted"), "POINTS", specular=A.PAINT_SPECULAR),
             "seg": R.surface_panel(f("seg"), "CELLS"),
             "bc": R.bc_panel(f("bc"), sup, None, b["tube_r"]),
             # the load arrows alone: the compose lays them OVER the body (round 2)
             "arrows": arrows_view(f("bigarrows")),
             "E": R.surface_panel(f("materials"), "POINTS"),
             "solve": R.solve_panel(f("solve"), 1.0, b["umax"])}
    warp = last_warp(f("solve"))
    for rv in views.values():
        finish_view(rv)
    views["octree"], _ = octree_view(f("octree"), f("octfaces"), f("solve"), b["umax"], b["centre"])
    gsize = sized(2 * TL.W * S, 2 * TL.H * S)
    grid = grid_view(f("grid"), gsize)
    n = 0
    for fi, t in times:
        if not TL.bench_visible(t):
            continue
        h = TL.bench_heading(t, b["heading0"])
        need = set(TL.look_weights(t))
        if TL.ghost_opacity(t) > 0:
            need.add("clay")          # the ghost is the undeformed clay pass's alpha
        if "bc" in need:
            need.add("arrows")
        for look in sorted(need):
            out = os.path.join(FR, look, f"f{fi:04d}.png")
            if os.path.exists(out):
                continue
            rv = views[look]
            set_camera(rv, b["centre"], b["distance"], h, b["elev"])
            if look == "solve":
                warp.ScaleFactor = TL.osc(t, b["k"])
            shoot(rv, out, size)
            n += 1
        if TL.grid_opacity(t) > 0:
            out = os.path.join(FR, "grid", f"f{fi:04d}.png")
            if not os.path.exists(out):
                set_camera(grid, b["centre"], b["distance"], h, b["elev"])
                shoot(grid, out, gsize)
                n += 1
    return n


def render_tiles(st, times):
    n = 0
    for tile in st["tiles"]:
        i = tile["j"]                       # tile j sits in cell j (reading order)
        path = os.path.join(WORK, f"ar_t{i}_solve.vtk")
        size = sized(2 * TL.TILE_BODY_W * S, 2 * TL.TILE_BODY_H * S)
        rv_u = R.solve_panel(path, 1.0, tile["umax"])
        warp = last_warp(path)
        rv_g = R.solve_panel(path, 0.0, tile["umax"])    # undeformed: the ghost's alpha
        for rv in (rv_u, rv_g):
            finish_view(rv)
        for fi, t in times:
            if not TL.tile_visible(t, i):
                continue
            h = TL.tile_heading(t, i, tile["heading0"])
            for rv, tag in ((rv_u, "u"), (rv_g, "g")):
                out = os.path.join(FR, f"t{i}", f"{tag}_f{fi:04d}.png")
                if os.path.exists(out):
                    continue
                set_camera(rv, tile["centre"], tile["distance"], h, tile["elev"])
                if tag == "u":
                    warp.ScaleFactor = TL.osc(t, tile["k"], TL.tile_phase(i))
                shoot(rv, out, size)
                n += 1
        Delete(rv_u)  # noqa: F405
        Delete(rv_g)  # noqa: F405
        print(f"tile {i} done ({tile['name']})", flush=True)
    return n


if __name__ == "__main__":
    paraview.simple._DisableFirstRenderCameraReset()  # noqa: F405
    _probe = CreateRenderView()  # noqa: F405
    Render(_probe)  # noqa: F405
    renderer = GetOpenGLInformation().GetRenderer()  # noqa: F405
    Delete(_probe)  # noqa: F405
    print("RENDERER:", renderer, flush=True)
    assert "llvmpipe" not in renderer.lower(), "software rendering: set the EGL env (illustrations/AGENTS.md §5)"
    st = json.load(open(os.path.join(WORK, "dv_stats.json")))
    times = TL.frame_times(args.step)
    if args.only:
        keep = {int(x) for x in args.only.split(",")}
        times = [(f, t) for f, t in TL.frame_times(1) if f in keep]
    log = os.path.join(FR, f"render_times_{args.part}.json")
    rec = {"renderer": renderer, "scale": S, "step": args.step}
    for part, fn in (("bench", render_bench), ("tiles", render_tiles)):
        if args.part in (part, "all"):
            t0 = time.time()
            n = fn(st, times)
            rec[part] = {"renders": n, "seconds": round(time.time() - t0, 1)}
            print(f"{part}: {n} renders in {time.time() - t0:.0f} s", flush=True)
    json.dump(rec, open(log, "w"), indent=1)
