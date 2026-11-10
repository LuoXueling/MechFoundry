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
  iterA / iterB            the clean-up's iterates in the uniform zero-|u| colour (point
                           rgb under the |u| kit, uniform_view): ONE view whose reader's
                           file is swapped per pass -- the two iterates a frame crossfades
  oarrows                  the change arrows (dv_prep: one mesh per frame), arrows_view
  opt / optclay            the optimised bench's |u| (per-part solve) on the session's |u|
                           range, warped by s(t); its undeformed pass for the ghost
  (camera)                 bench_camera: the orbit, eased into the beat's close-up of one
                           end by dv_timeline.closeup and back out in the shrink
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


def uniform_view(path):
    """A surface in its own point rgb under the |u| look's kit (AR_SOLVE_VIEW,
    SOLVE_FINISH), so the uniform zero-|u| body crossfades with the |u| look with no
    change of light, only of colour."""
    rv = R.new_view(A.AR_SOLVE_VIEW)
    nrm = GenerateSurfaceNormals(Input=LegacyVTKReader(FileNames=[path]))  # noqa: F405
    R._try(lambda: setattr(nrm, "FeatureAngle", A.FEATURE_ANGLE))
    # no splitting: the iterates are marching-cubes surfaces, and splitting at their small
    # creases drew faint grey streaks along the slats (the user: "weird lines")
    R._try(lambda: setattr(nrm, "Splitting", 0))
    d = Show(nrm, rv)  # noqa: F405
    d.Representation = "Surface"
    for k, v in A.SOLVE_FINISH.items():
        setattr(d, k, v)
    ColorBy(d, ("POINTS", "rgb"))  # noqa: F405
    d.MapScalars = 0
    d.SetScalarBarVisibility(rv, False)
    return rv


def grab_z(rv, path):
    """The view's depth buffer (as rendered at its ViewSize) -> float32 .npy, top row
    first: the compose cuts the change arrows exactly where the body is in front."""
    import numpy as np
    from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
    from vtkmodules.util.numpy_support import vtk_to_numpy
    # FXAA and tone mapping (the |u| kit's) render through a post-processing buffer, and
    # the window's depth then reads back empty (all 1): the first cut's masks were void.
    # Depth does not depend on either, so they are off for this read only.
    keep = {k: getattr(rv, k) for k in ("UseFXAA", "UseToneMapping") if hasattr(rv, k)}
    for k in keep:
        setattr(rv, k, 0)
    Render(rv)  # noqa: F405
    f = vtkWindowToImageFilter()
    f.SetInput(rv.GetRenderWindow())
    f.SetInputBufferTypeToZBuffer()
    f.ReadFrontBufferOff()
    f.Update()
    img = f.GetOutput()
    w, h, _ = img.GetDimensions()
    z = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w)[::-1]
    for k, v in keep.items():
        setattr(rv, k, v)
    assert (z < 1).any(), f"empty depth buffer for {path}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, z.astype(np.float32))


def reader_of(path):
    """The LegacyVTKReader built on ``path`` (its file is swapped per pass)."""
    hits = [p for p in GetSources().values()  # noqa: F405
            if hasattr(p, "FileNames") and path in list(p.FileNames)]
    assert hits, f"no reader on {path}"
    return hits[-1]


def bench_camera(b, t):
    """(focal, distance, heading, elev) of the bench at t: the orbit of scenes 2-7, eased
    by dv_timeline.closeup into the beat's close-up of one end (the camera orbits that
    end's focus point, its focal point slid along the camera's own right/up axes so the
    focus sits at CLOSE_CENTRE at every heading) and back out during the shrink; the
    heading is dv_timeline.bench_heading's, slowed in the beat (dv_prep.bench_cam is the
    twin)."""
    z = TL.closeup(t)
    h = TL.bench_heading(t, b["heading0"])
    c, D = list(b["centre"]), b["distance"]
    if z > 0:
        cl = b["cleanup"]["close"]
        a, e = math.radians(h), math.radians(b["elev"])
        f = [-math.cos(e) * math.sin(a), -math.sin(e), -math.cos(e) * math.cos(a)]
        r = [-f[2], 0.0, f[0]]                           # f x (0, 1, 0)
        n = math.sqrt(sum(x * x for x in r))
        r = [x / n for x in r]
        u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]]
        cc = [cl["focus"][i] + r[i] * cl["ox"] + u[i] * cl["oy"] for i in range(3)]
        c = [c[i] + z * (cc[i] - c[i]) for i in range(3)]
        D = 1.0 / ((1 - z) / D + z / cl["distance"])
    return c, D, h, b["elev"]


def render_bench(st, times):
    b = st["bench"]
    f = lambda name: os.path.join(WORK, f"ar_0_{name}.vtk")  # noqa: E731
    size = sized(TL.W * S, TL.H * S)
    sup = f("support") if os.path.exists(f("support")) else None
    umax = b["umax"]
    views = {"clay": R.surface_panel(f("clay"), "POINTS"),
             "paint": R.surface_panel(f("painted"), "POINTS", specular=A.PAINT_SPECULAR),
             "seg": R.surface_panel(f("seg"), "CELLS"),
             "bc": R.bc_panel(f("bc"), sup, None, b["tube_r"]),
             # the load arrows alone: the compose lays them OVER the body (round 2)
             "arrows": arrows_view(f("bigarrows")),
             "E": R.surface_panel(f("materials"), "POINTS"),
             "solve": R.solve_panel(f("solve"), 1.0, umax)}
    warp = last_warp(f("solve"))
    # the optimisation beat: the iterates in the uniform zero-|u| colour, ONE view whose
    # reader steps through the files (two passes a frame, crossfaded by the compose); the
    # change arrows, one mesh per frame (dv_prep), in their own pass over the body; the
    # optimised bench's |u| on the SESSION's |u| range (its smaller deflection reads
    # lighter) and its undeformed pass for the ghost
    it0 = f("it00")
    views["iter"] = uniform_view(it0)
    it_reader = reader_of(it0)
    # the change arrows live IN the iterate's view (the user: occlusion must be exact;
    # two views reset their clipping ranges to their own bounds, so their depth buffers
    # were not comparable and arrows showed through the slats).  Each arrow frame is shot
    # twice from this one view: body only (the arrows' display kept but at opacity 0, so
    # the bounds and clipping range do not change) and body + arrows; the compose keeps
    # the pixels where the combined depth is nearer than the body-only depth.
    oa0 = os.path.join(WORK, "oarrows", sorted(os.listdir(os.path.join(WORK, "oarrows")))[0])
    oa_nrm = GenerateSurfaceNormals(Input=LegacyVTKReader(FileNames=[oa0]))  # noqa: F405
    R._try(lambda: setattr(oa_nrm, "FeatureAngle", 60))
    oa_disp = Show(oa_nrm, views["iter"])  # noqa: F405
    oa_disp.Representation = "Surface"
    for k_, v_ in A.ARROW_FINISH.items():
        setattr(oa_disp, k_, v_)
    ColorBy(oa_disp, ("CELLS", "rgb"))  # noqa: F405
    oa_disp.MapScalars = 0
    oa_disp.SetScalarBarVisibility(views["iter"], False)
    oa_disp.Opacity = 0.0
    oa_reader = reader_of(oa0)
    # the same arrows in flat ID colours (ambient only, no FXAA / tone mapping: the red
    # channel reads back exactly as the arrow's index + 1), arrows only, same camera
    id0 = os.path.join(WORK, "oarrows_id", sorted(os.listdir(os.path.join(WORK, "oarrows_id")))[0])
    views["oarrows_id"] = R.new_view(A.AR_SURF_VIEW)
    for k_ in ("UseFXAA", "UseToneMapping"):
        R._try(lambda k_=k_: setattr(views["oarrows_id"], k_, 0))
    id_disp = Show(LegacyVTKReader(FileNames=[id0]), views["oarrows_id"])  # noqa: F405
    ColorBy(id_disp, ("CELLS", "rgb"))  # noqa: F405
    id_disp.MapScalars = 0
    id_disp.Ambient, id_disp.Diffuse, id_disp.Specular = 1.0, 0.0, 0.0
    id_disp.Interpolation = "Flat"
    id_disp.SetScalarBarVisibility(views["oarrows_id"], False)
    id_reader = reader_of(id0)
    views["opt"] = R.solve_panel(f("optsolve"), 1.0, umax)
    warp_opt = last_warp(f("optsolve"))
    views["optclay"] = R.solve_panel(f("optsolve"), 0.0, umax)
    for rv in views.values():
        finish_view(rv)
    views["octree"], _ = octree_view(f("octree"), f("octfaces"), f("solve"), umax, b["centre"])
    gsize = sized(2 * TL.W * S, 2 * TL.H * S)
    grid = grid_view(f("grid"), gsize)
    n = 0
    for fi, t in times:
        if not TL.bench_visible(t):
            continue
        c, D, h, el = bench_camera(b, t)
        ov = TL.oversize(t)
        # the shrink's frames on a canvas ov x larger each way: same camera and pixel
        # scale, the view angle widened so tan(fov/2) grows by ov (dv_timeline.oversize)
        fov = math.degrees(2 * math.atan(ov * math.tan(math.radians(A.VIEW_ANGLE) / 2)))
        osize = [int(round(ov * x)) for x in size]
        ogsize = [int(round(ov * x)) for x in gsize]
        weights = TL.look_weights(t)
        need = set(weights) - {"iter"}
        need |= set(TL.ghosts(t))     # each ghost is its undeformed pass's alpha
        if "bc" in need:
            need.add("arrows")
        files = {}
        arrows_on = TL.arrows_visible(t) and "iter" in weights
        if arrows_on:
            oa_reader.FileNames = [os.path.join(WORK, "oarrows", f"f{fi:04d}.vtk")]
        if "iter" in weights:
            i0, i1, w1 = TL.vm_iterate(t)
            files["iterA"] = (it_reader, f(f"it{i0:02d}"), "iter")
            if w1 > 1e-6:
                files["iterB"] = (it_reader, f(f"it{i1:02d}"), "iter")
            if arrows_on:              # body + arrows, on the iterate that dominates the blend
                files["oarrows"] = (it_reader, f(f"it{(i1 if w1 >= 0.5 else i0):02d}"), "iter")
                files["oarrows_id"] = (id_reader, os.path.join(WORK, "oarrows_id", f"f{fi:04d}.vtk"),
                                       "oarrows_id")
        for look in sorted(need) + sorted(files):
            out = os.path.join(FR, look, f"f{fi:04d}.png")
            if os.path.exists(out):
                continue
            if look in files:
                reader, path, vname = files[look]
                reader.FileNames = [path]
                rv = views[vname]
            else:
                rv = views[look]
            set_camera(rv, c, D, h, el)
            rv.CameraViewAngle = fov
            if look == "solve":
                warp.ScaleFactor = TL.bench_osc(t, b["k"])
            if look == "opt":
                warp_opt.ScaleFactor = TL.bench_osc(t, b["k"])
            rv.ViewSize = osize
            if look == "oarrows":
                # the same view, the same camera, the same bounds: body only, then body +
                # arrows (both depth buffers), then the combined colour
                oa_disp.Opacity = 0.0
                grab_z(rv, os.path.join(FR, "oarrows_body_z", f"f{fi:04d}.npy"))
                oa_disp.Opacity = 1.0
                grab_z(rv, os.path.join(FR, "oarrows_z", f"f{fi:04d}.npy"))
                shoot(rv, out, osize)
                oa_disp.Opacity = 0.0
                n += 1
                continue
            shoot(rv, out, osize)
            n += 1
        if TL.grid_opacity(t) > 0:
            out = os.path.join(FR, "grid", f"f{fi:04d}.png")
            if not os.path.exists(out):
                set_camera(grid, c, D, h, el)
                grid.CameraViewAngle = fov
                grid.ViewSize = ogsize
                shoot(grid, out, ogsize)
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
