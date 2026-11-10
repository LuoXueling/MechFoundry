"""Stage 1 (env mechfoundry-warp117, small GPU use): the demo video's surfaces and cameras.

    python dv_prep.py [--work _work]

Everything drawn is what the agentic-results figure draws, computed by its own code
(``agentic_results/ar_prep.py``, imported, not copied):

  bench   ar_prep.prep_run on the park-bench Opus run (AR_SEGMENTATION=1), with the
          main figure's camera heading (ar_stats.json) as its explicit view, then
          ar_prep.write_run_files -> _work/ar_0_{painted,seg,bc,materials,solve,grid}.vtk.
          Three video-only changes, each made here and not in ar_prep:
            * the figure's load arrows (3 here, sized for print) are replaced by 5 seat
              + 3 backrest arrows of ar_prep.arrow_mesh's shape, 150 px long on
              screen (ar_0_bigarrows.vtk), which the compose draws ON TOP of the body;
            * supports the figure would draw "seen through" are drawn as ordinary
              outlines (the same reason), and nothing else is seen through;
            * Young's modulus is coloured on log E in [5, 200] GPa (the figure's
              shared [10^5.5, 10^11.5] Pa puts wood and iron in two similar blues).
          plus ar_0_clay.vtk, the same surface in the overview's clay.
  octree  the solve's own full adaptive octree, solve/<run>/vtu_sdf/solution_t0000.vtu
          (written by the solver beside the trimmed vtu/: every leaf over the domain,
          11-182 mm), read by render_dv as is; its cells' faces as quads ->
          _work/ar_0_octfaces.vtk, which render_dv slices for the cut-face outlines
  tiles   20 other Opus runs, each as the figure's last column: ar_prep._prep_solve
          (|u| on the surface, the x N rule), GATED against the run's record in the
          main figure (ar_stats.json) or its Opus category figure (factor equal,
          |u|max to 1e-6) -> _work/ar_t<j>_solve.vtk

Cameras (written to _work/dv_stats.json; render_dv.py sets them verbatim):
  every object orbits a vertical axis through its bounding-box centre at its record's
  elevation, at ONE distance for the whole clip (no zoom pump).  The bench starts at
  heading +55 deg (front three-quarter) and its distance makes its at-rest projected
  height, over the headings the clip shows, 92 % of the 780 px viewport;
  each tile's makes its largest projected side over all azimuths, undeformed and at its
  peak k, 88 % of the cell's short side.  The focal height is moved until that union
  box is centred vertically (the bench's on its viewport's centre, y = 610).
  Projection is ParaView's (vertical view angle
  ar_style.VIEW_ANGLE), computed here with numpy.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("AR_SEGMENTATION", "1")      # ar_figure reads it at import
HERE = os.path.dirname(os.path.abspath(__file__))
AR = os.path.join(HERE, "..", "agentic_results")
sys.path.insert(0, AR)
sys.path.insert(0, os.path.join(HERE, "..", "gallery"))
sys.path.insert(0, HERE)
import ar_prep  # noqa: E402  (imports ar_figure before trimesh, as it must)
import ar_runs  # noqa: E402
import ar_style as A  # noqa: E402
import gl_prep  # noqa: E402  (bc_regions, category_records: the gallery's tile prep)
import numpy as np  # noqa: E402
import dv_timeline as TL  # noqa: E402
from dv_camera import basis, project  # noqa: E402

# the checkout's own extract (the figure records name the batch under the checkout's old
# path, /home/xl2254/implicit_optimize, which no longer exists: run_key matches a record to
# a run by its path below agentic_batch50_opus/, wherever the batch sits)
BATCH = os.environ.get("DV_BATCH_ROOT", os.path.join(ar_prep.REPO, "output", "agentic_batch50_opus"))
BENCH = "park-bench-20261013-212923"
# The 20 tiles beside the bench, in reading order of the 7 x 3 grid (the bench is
# cell 0, 0).  Every tile carries its object's prompt above it, which must fit a 256 px
# cell at 28 px italic in <= 2 lines.  Earlier swaps: the car suspension control arm,
# the glass-top dining table and the wooden rocking horse (whose bare names needed
# 18-22 px) gave way to a coffee mug, a sofa, the Pantheon, a car brake disc, an
# adjustable wrench and a lumbar vertebra; the car wheel had gone (its tyre stayed
# edge-on as a blank white drum); the school chair ("a school chair with a steel frame
# and a plastic seat" fits only at 20 px) gave way to a bar stool.  Categories
# alternate along the rows; all solved, 3.4-9.3 min each.
TILES = [
    "claw-hammer-20261014-002959",
    "giraffe-20261013-224524",
    "taj-mahal-20261014-021519",
    "coffee-mug-20261013-210839",
    "human-femur-20261014-010205",
    "himeji-castle-20261014-020226",
    "teddy-bear-20261013-214602",
    "truck-leaf-spring-20261013-231223",
    "elephant-20261013-224920",
    "sofa-20261013-214158",
    "electric-toothbrush-20261013-210405",
    "natural-stone-arch-20261014-021045",
    "car-brake-disc-20261013-233035",
    "bar-stool-20261013-213800",
    "pantheon-20261014-015416",
    "industrial-robot-arm-20261014-002011",
    "igloo-20261014-013439",
    "adjustable-wrench-20261014-002549",
    "crocodile-20261013-230026",
    "lumbar-vertebra-20261014-010944",
]


def object_name(instruction):
    """objects_50's wording as a tile name: no article, no trailing " with ..." clause,
    first letter capital ("the Taj Mahal" -> "Taj Mahal")."""
    name = instruction.split(" with ")[0].strip()
    for art in ("a ", "an ", "the "):
        if name.lower().startswith(art):
            name = name[len(art):]
    return name[0].upper() + name[1:]
E_RANGE_GPA = (5.0, 200.0)
TUBE_MULT = 2.5          # support outline tubes, x the figure's radius (phone legibility)


# -- records ------------------------------------------------------------------------------
def run_key(path):
    """A session's path from agentic_batch50_opus/ on (or the whole path if not in it)."""
    p = os.path.normpath(path)
    i = p.find(os.sep + "agentic_batch50_opus" + os.sep)
    return p[i:] if i >= 0 else p


def figure_records():
    """session folder -> its record: the main figure's (ar_stats.json) first, then the
    Opus category figures' (the heading each figure chose for it)."""
    out = {}
    for r in json.load(open(os.path.join(AR, "ar_stats.json")))["runs"]:
        out[run_key(r["session"])] = dict(r, _file="ar_stats.json")
    for instr, r in gl_prep.category_records().items():
        out.setdefault(run_key(r["session"]), r)
    return out


# -- the camera ---------------------------------------------------------------------------
def union_box(sets, c, elev, D, w_px, h_px, headings):
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for h in headings:
        for P in sets:
            x, y = project(P, c, h, elev, D, w_px, h_px)
            x0, x1 = min(x0, x.min()), max(x1, x.max())
            y0, y1 = min(y0, y.min()), max(y1, y.max())
    return x0, y0, x1, y1


def fit_camera(fit_sets, elev, w_px, h_px, target_px, measure, all_sets=None, heads=None,
               cy=None):
    """Orbit centre c (bbox centre; its height then moved to centre the box) and the
    distance D at which ``measure`` (of the union box over 72 headings) == target_px."""
    allP = np.vstack(fit_sets)
    lo, hi = allP.min(0), allP.max(0)
    c = 0.5 * (lo + hi)
    R = 0.5 * float(np.linalg.norm(hi - lo))
    heads = np.arange(0, 360, 5.0) if heads is None else heads
    D = 3 * R
    for _ in range(4):
        a, b = 1.05 * R, 200 * R
        for _ in range(60):
            m = 0.5 * (a + b)
            if measure(union_box(fit_sets, c, elev, m, w_px, h_px, heads)) > target_px:
                a = m
            else:
                b = m
        D = 0.5 * (a + b)
        box = union_box(all_sets or fit_sets, c, elev, D, w_px, h_px, heads)
        dy = 0.5 * (box[1] + box[3]) - (h_px / 2 if cy is None else cy)   # px, + = too low
        # a px of screen height at the orbit centre is (2 D tan(fov/2) / h_px) world units
        c = c.copy()
        c[1] -= dy * 2 * D * math.tan(math.radians(A.VIEW_ANGLE) / 2) / h_px
    box = union_box(all_sets or fit_sets, c, elev, D, w_px, h_px, heads)
    return c, D, box


def heading_elev(vd):
    vd = np.asarray(vd, float)
    vd = vd / np.linalg.norm(vd)
    return math.degrees(math.atan2(vd[0], vd[2])), math.degrees(math.asin(vd[1]))


# -- the bench's load arrows ---------------------------------------------------------------
ARROW_PX = 150           # length on screen at the orbit centre
ARROW_HEAD_PX = 34       # head width; the shaft and head keep ar_prep.arrow_mesh's proportions
ARROWS_PER = {"seat": 5, "back": 3}


def big_arrows(work, mesh, masks, regions, D):
    """ar_prep.arrow_mesh arrows along each load's traction, sized in screen px at the
    camera distance D: 5 on the region facing up (the seat), 3 on the other (the
    backrest), at x quantiles 0.1..0.9 of the region's faces that face the load, tip on
    the surface (a push).  Written to ar_0_bigarrows.vtk (cell colour LOAD_ARROW)."""
    ppm = (TL.H / 2) / (D * math.tan(math.radians(A.VIEW_ANGLE) / 2))     # px per metre
    length = ARROW_PX / ppm
    Ls = ARROW_HEAD_PX / ppm / (2 * ar_prep.ARROW_HEAD_R)                  # arrow_mesh's "L"
    vs, fs, off = [], [], 0
    cen = mesh.triangles_center
    L = float(np.ptp(mesh.vertices, axis=0).max())
    for g, t_ in regions:
        m = np.asarray(masks[g["key"]], bool)
        dvec = np.asarray(t_, float) / np.linalg.norm(t_)
        n = g["mean_normal"] or [0, 1, 0]
        count = ARROWS_PER["seat"] if n[1] > 0.7 else ARROWS_PER["back"]
        idx = np.nonzero(m & (mesh.face_normals @ -dvec > 0.5))[0]
        if len(idx) == 0:
            idx = np.nonzero(m)[0]
        mid = np.median(cen[idx], axis=0)
        for q in np.linspace(0.1, 0.9, count):
            xq = np.quantile(cen[idx, 0], q)
            j = idx[np.argmin((cen[idx, 0] - xq) ** 2 + 0.25 * ((cen[idx, 1:] - mid[1:]) ** 2).sum(1))]
            tip = cen[j] - dvec * 0.004 * L
            v_, f_ = ar_prep.arrow_mesh(tip - dvec * length, tip, Ls)
            vs.append(v_)
            fs.append(f_ + off)
            off += len(v_)
    V_, F_ = np.vstack(vs), np.vstack(fs)
    ar_prep.write_polydata(os.path.join(work, "ar_0_bigarrows.vtk"), V_, F_,
                           cell_rgb=np.tile(np.array(A.LOAD_ARROW), (len(F_), 1)))
    return len(vs)


# -- bench --------------------------------------------------------------------------------
def prep_bench(work, ctx, recs):
    path = os.path.join(BATCH, BENCH)
    run0 = ar_runs.load_run(path)
    fig = recs[run_key(run0.session)]
    run = ar_runs.load_run({"path": path, "view_dir": fig["view"]["dir"]})
    rec, out = ar_prep.prep_run(0, run, work, ctx)
    s = run.session
    mesh = ar_prep.load_mesh(os.path.join(s, "geometry", "physical.glb"))
    V = out["V"]
    L = rec["extent"]
    assert rec["scale"]["factor"] == fig["scale"]["factor"], "x N differs from the figure's"
    assert abs(rec["solve"]["umax_surface"] / fig["solve"]["umax_surface"] - 1) < 1e-6

    # The figure's arrows (load_arrows: 3 here, sized for a print panel) are replaced by
    # big ones drawn in a pass of their own, ON TOP of the body (round 2): 5 on the seat
    # and 3 on the backrest, spread along the bench, the figure's arrow mesh scaled up.
    masks = np.load(os.path.join(s, "boundary_conditions", "resolved.npz"))
    case = json.load(open(os.path.join(s, "case", "case.json")))
    tractions = {nb["name"]: nb["traction"] for nb in
                 (case.get("boundary_conditions") or {}).get("neumann") or [] if nb.get("traction")}
    out["arrows"], out["hidden_arrows"] = [], []
    regions = [(g, tractions.get(g["name"])) for g in rec["bc"]["regions"]
               if g["category"] == "neumann" and tractions.get(g["name"]) is not None]
    out.pop("bc_hidden", None)
    if "support_hidden" in out:      # drawn as ordinary outlines: the body turns
        parts = [out.pop("support_hidden")] + ([out["support_marker"]] if "support_marker" in out else [])
        off, vs, fs, ids = 0, [], [], []
        for sv, sf, sid in parts:
            vs.append(sv)
            fs.append(sf + off)
            ids.append(sid)
            off += len(sv)
        out["support_marker"] = (np.vstack(vs), np.vstack(fs), np.concatenate(ids))
    lo, hi = (math.log10(e * 1e9) for e in E_RANGE_GPA)
    ar_prep.write_run_files(rec, out, work, lo, hi)
    ar_prep.write_polydata(os.path.join(work, "ar_0_clay.vtk"), V, out["F"],
                           point_rgb=np.tile(np.array(A.CLAY), (len(V), 1)))
    # the two part moduli (wood slats, iron frame); values between are the interpolation
    # across a part seam
    Etab = [round(float(out["E"].min()) / 1e9, 2), round(float(out["E"].max()) / 1e9, 2)]

    k = float(rec["scale"]["factor"])
    _, elev = heading_elev(rec["view"]["dir"])
    h0 = TL.BENCH_HEADING0           # round 2: front three-quarter, never the back
    # the headings of the 23.0 s cut, kept so the optimisation beat (2026-11-07) does not
    # reframe the clip (the beat has its own framing, dv_prep.close_params, and the
    # shrink its own box, box_close_px)
    span = TL.SPIN_DPS * (23.0 - TL.SPIN_START)
    heads = np.arange(h0, h0 - span - 2.0, -2.0)
    vp = TL.VIEWPORT[1] - TL.VIEWPORT[0]
    Vd = V + k * out["u"]
    c, D, box_rest = fit_camera([V], elev, TL.W, TL.H, TL.FIT_REST * vp,
                                measure=lambda b: b[3] - b[1], heads=heads,
                                cy=0.5 * (TL.VIEWPORT[0] + TL.VIEWPORT[1]))
    box_all = union_box([V, Vd], c, elev, D, TL.W, TL.H, heads)
    n_arrows = big_arrows(work, mesh, masks, regions, D)
    vtu_sdf = os.path.join(s, rec["solve"]["vtu"]).replace(os.sep + "vtu" + os.sep, os.sep + "vtu_sdf" + os.sep)
    assert os.path.exists(vtu_sdf), f"no full-octree output {vtu_sdf}"
    bench = {"session": s, "instruction": rec["meta"].get("instruction"),
             "wall_clock_s": rec["meta"].get("wall_clock_s"),
             "record": fig["_file"], "k": k, "umax": rec["solve"]["umax_surface"],
             "extent": L, "heading0": h0, "elev": elev, "centre": c.tolist(), "distance": D,
             "box_rest_px": box_rest, "box_all_px": box_all,
             "tube_r": TUBE_MULT * ar_prep.MARKER_TUBE * L, "E_GPa_min_max": Etab,
             # the octree insert reads the solve's own FULL octree: vtu_sdf/ beside vtu/
             # is the adaptive octree over the whole domain (5 levels, 11-182 mm leaves)
             # with the SDF; vtu/ holds only the trimmed cells the solve kept
             "vtu": vtu_sdf, "octree": octree_stats(vtu_sdf, s, work),
             "solve_mesh": {"nodes": rec["solve"]["nodes"], "hexes": rec["solve"]["hexes"]},
             "E_range_GPa": E_RANGE_GPA, "arrows": n_arrows, "headings_deg": [float(heads[0]), float(heads[-1])],
             "files": sorted(os.path.basename(p) for p in os.listdir(work) if p.startswith("ar_0_")),
             "notes": rec["notes"]}
    print(f"bench: x{k:g}, |u|max {bench['umax']:.4g}, heading {h0:.1f}, elev {elev:.1f}, "
          f"D {D:.4g}, rest box {np.round(box_rest, 1).tolist()}, all {np.round(box_all, 1).tolist()}, "
          f"E {Etab} GPa, {n_arrows} arrows")
    return bench


def full_octree(session, vtu, work):
    """The solve's octree, COMPLETE over a box (the user: "some octree cells on the
    outer boundary are missing").

    Why cells are missing from vtu_sdf/: workflow_tool.run_forward builds the octree in
    an isotropic cube (side h * 2^8 = 2.909 m for the bench) and then sheds the cube
    padding with core.remeshing.clip_octree_to_bbox, which keeps a cell iff its CENTROID
    lies in the geometry bbox + pad.  Cells straddling that box are kept or dropped by
    their centre, so the saved mesh is a stepped, notched block: 136 801 leaves filling
    92.0 % of their own bounding box (184 x 128 x 128 finest leaves).

    Rebuilt here with the repo's own code, exactly as ShapeOptTrimmed._init_remeshing_state
    path A builds it: build_initial_octree_from_trimesh (same mesh, mesh_center, h,
    depths 4 -> 8, clip_bbox for the lattice snap, sdf_truncation) -> sdf2octree on its
    anchors -> scale_octree_nodes_to_physical.  Gate: clip_octree_to_bbox of the rebuilt
    cube must give exactly vtu_sdf's cells.  Drawn: every leaf of the rebuilt cube that
    lies inside the clip box grown outward to the coarsest-cell lattice (182 mm), so the
    block is a full box of whole cells.  Writes _work/ar_0_octree.vtk (hexes) and
    ar_0_octfaces.vtk (their faces as quads); returns the record."""
    import torch
    import trimesh
    from core.remeshing import (build_initial_octree_from_trimesh, clip_octree_to_bbox,
                                scale_octree_nodes_to_physical)
    from src.triposg_wrapper.utils import sdf2octree
    sys.path.insert(0, os.path.join(ar_prep.REPO, "src", "agents_wrapper", "fem_workflow", "scripts"))
    case = json.load(open(os.path.join(session, "case", "case.json")))
    so = case["shape_opt"]
    mesh = trimesh.load(os.path.join(session, "geometry", "physical.glb"), force="mesh", process=False)
    h = float(case["h_target"]) * 2.0 ** (int(case["octree_depth"]) - int(so["remesh_octree_depth"]))
    lo_m, hi_m = (np.asarray(b, float) for b in mesh.bounds)
    clip_pad = 0.3 / 2.0 * float((hi_m - lo_m).max()) + 0.5 * h      # workflow_tool.CLIP_BBOX_PAD
    clip = (tuple(lo_m - clip_pad), tuple(hi_m + clip_pad))
    init = build_initial_octree_from_trimesh(
        mesh, case["mesh_center"], h, int(so["remesh_octree_depth"]),
        int(so["remesh_init_octree_depth"]), device="cuda", clip_bbox=clip, build_octree=False)
    nodes, _, elements, _, _ = sdf2octree(cell_tensor=init["init_anchors"],
                                          sdf_tensor=init["init_grid_h16"], domain_length=2.0)
    dev = nodes.device
    full = scale_octree_nodes_to_physical(
        {"nodes": nodes, "elements": elements, "sdf": torch.zeros(len(nodes), device=dev)},
        torch.tensor(init["axis_scale"], dtype=torch.float64, device=dev),
        torch.tensor(init["geo_min"], dtype=torch.float64, device=dev))
    P = full["nodes"].double().cpu().numpy()
    C = full["elements"].long().cpu().numpy()
    clipped = clip_octree_to_bbox(full, *clip)
    Pc = clipped["nodes"].double().cpu().numpy()
    Cc = clipped["elements"].long().cpu().numpy()
    d = ar_prep.read_vtu(vtu)
    Ps = np.asarray(d["Points"], float).reshape(-1, 3)
    Cs = np.asarray(d["connectivity"], np.int64).reshape(-1, 8)

    def boxes(Pa, Ca):
        return {tuple(np.round(np.r_[Pa[Ca[i]].min(0), Pa[Ca[i]].max(0)] / h * 4).astype(np.int64))
                for i in range(len(Ca))}
    saved = boxes(Ps, Cs)
    rebuilt = boxes(Pc, Cc)
    assert rebuilt == saved, f"rebuilt clip {len(rebuilt)} cells != vtu_sdf {len(saved)}"
    # the box: the clip box grown to the coarsest-cell lattice of the cube
    lo, hi = P[C].min(1), P[C].max(1)
    size = (hi - lo)[:, 0]
    big = size.max()
    origin = P.min(0)
    blo = origin + np.floor((np.asarray(clip[0]) - origin) / big + 1e-9) * big
    bhi = origin + np.ceil((np.asarray(clip[1]) - origin) / big - 1e-9) * big
    keep = np.all(lo >= blo - 1e-6 * big, 1) & np.all(hi <= bhi + 1e-6 * big, 1)
    vol = float((size[keep] ** 3).sum())
    assert abs(vol / np.prod(bhi - blo) - 1) < 1e-6, "the kept cells do not fill the box"
    Ck = C[keep]
    used, inv = np.unique(Ck, return_inverse=True)
    Pk, Ck = P[used], inv.reshape(-1, 8)
    with open(os.path.join(work, "ar_0_octree.vtk"), "w") as out:
        out.write("# vtk DataFile Version 3.0\nfull octree\nASCII\nDATASET UNSTRUCTURED_GRID\n"
                  f"POINTS {len(Pk)} double\n")
        np.savetxt(out, Pk, fmt="%.9g")
        out.write(f"CELLS {len(Ck)} {9 * len(Ck)}\n")
        np.savetxt(out, np.column_stack([np.full(len(Ck), 8), Ck]), fmt="%d")
        out.write(f"CELL_TYPES {len(Ck)}\n")
        np.savetxt(out, np.full(len(Ck), 12), fmt="%d")
    cube_lo, cube_hi = P.min(0), P.max(0)
    return {"source": "rebuilt with build_initial_octree_from_trimesh + sdf2octree "
                      "(gate: its clip_octree_to_bbox == vtu_sdf cell for cell)",
            "why_vtu_sdf_is_ragged": "clip_octree_to_bbox keeps a cell iff its centroid is in "
                                      "the geometry bbox + pad",
            "vtu_sdf_cells": len(saved), "vtu_sdf_fill_of_its_bbox": None,
            "root_cube_m": [cube_lo.tolist(), cube_hi.tolist()], "root_cube_cells": int(len(C)),
            "clip_bbox_m": [list(clip[0]), list(clip[1])],
            "drawn_box_m": [blo.tolist(), bhi.tolist()], "drawn_cells": int(keep.sum()),
            "leaf_m": float(size.min()), "coarsest_m": float(big)}, Pk, Ck


def write_octfaces(P, C, work):
    """Every cell's 6 faces as axis-aligned QUADS (shared faces once) ->
    _work/ar_0_octfaces.vtk, which render_dv slices with the plane to draw each cut
    cell's exact outline on the cut face -- line segments, no triangulation."""
    lo, hi = P[C].min(1), P[C].max(1)
    ext = hi - lo
    assert np.allclose(ext, ext[:, :1], rtol=1e-3), "octree cells must be cubes"
    quads = ((0, 2, 6, 4), (1, 3, 7, 5), (0, 1, 5, 4), (2, 3, 7, 6), (0, 1, 3, 2), (4, 5, 7, 6))
    bits = np.array([[i & 1, (i >> 1) & 1, (i >> 2) & 1] for i in range(8)], float)
    corners = lo[:, None, :] + ext[:, None, :] * bits[None]                 # (N, 8, 3)
    fq = corners[:, np.array(quads)].reshape(-1, 4, 3)                    # (6N, 4, 3)
    key = np.round(fq.reshape(-1, 3) / ext.min() * 4).astype(np.int64)    # quarter-leaf lattice
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    pts = np.zeros((len(uniq), 3))
    pts[inv] = fq.reshape(-1, 3)
    faces = inv.reshape(-1, 4)
    _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    faces = faces[np.sort(keep)]
    with open(os.path.join(work, "ar_0_octfaces.vtk"), "w") as out:
        out.write(f"# vtk DataFile Version 3.0\noctree faces\nASCII\nDATASET POLYDATA\n"
                  f"POINTS {len(pts)} double\n")
        np.savetxt(out, pts, fmt="%.9g")
        out.write(f"POLYGONS {len(faces)} {5 * len(faces)}\n")
        np.savetxt(out, np.column_stack([np.full(len(faces), 4), faces]), fmt="%d")


def octree_stats(vtu, session, work):
    """Leaves per size of the solver's vtu_sdf/ octree and how much of its own bbox they
    fill; then the complete box (full_octree) whose cells render_dv draws."""
    d = ar_prep.read_vtu(vtu)
    P = np.asarray(d["Points"], float).reshape(-1, 3)
    C = np.asarray(d["connectivity"], np.int64).reshape(-1, 8)
    assert np.all(np.asarray(d["types"]) == 12), "not an all-hex octree"
    ext = np.ptp(P[C], axis=1).max(1)
    h, n = np.unique(np.round(ext, 5), return_counts=True)
    bb = np.ptp(P, axis=0)
    rec, Pk, Ck = full_octree(session, vtu, work)
    rec["vtu_sdf_fill_of_its_bbox"] = float((ext ** 3).sum() / np.prod(bb))
    rec["vtu_sdf_bbox_m"] = [P.min(0).tolist(), P.max(0).tolist()]
    write_octfaces(Pk, Ck, work)
    return {"cells": int(len(C)), "nodes": int(len(P)),
            "leaf_sizes_m": h.tolist(), "cells_per_size": n.tolist(), "drawn": rec}


# -- the optimisation beat ----------------------------------------------------------------
SITES_GROW, SITE_MIN_H, SITE_SEP_M = 8, 1.0, 0.15   # the coarse sites that choose the end
END_MIN_H = 0.5                   # the close-up's arrows: |change| >= 0.5 h (5.7 mm), either sign:
                                  # where the change lut has clearly left white for orange (the user:
                                  # every visibly red patch gets arrows); 1.0 / 0.6 / 0.8 h came first
END_SEP_M0, END_SEP_MIN = 0.04, 0.008   # ... spaced by change, steeply (the user: emphasis by
END_SEP_POW = 2.0                 #     density, not size): 4 cm (0.5 h / c)^2, >= 0.8 cm
PAUSE_MIN_SHARE = 0.6             # the end pause shows an arrow only if >= 60 % of it is
PAUSE_MIN_FRAC = 0.5              # unoccluded (not end-on) in >= half its frames (the user:
                                  # half-hidden arrows and head-only stubs at the end); else it
                                  # is off for the whole pause (the compose fades it out ahead)
COVER_MIN_VERTS, COVER_NEAR_M = 8, 0.015   # coverage: a visible red patch of >= 8 vertices
                                           # needs a visible arrow within 1.5 cm
END_PIECE_MIN = 0.03              # ... not on a mesh piece < 3 % of the body's largest, any iterate
OARROW_MAX_PX = 176               # (former: the largest change's arrow)
OARROW_BASE_PX = 50               # nearly uniform arrows: 50 px at the threshold (45 until the user
                                  # found them hard to see at the end; +25 % = 56 px with the bigger
                                  # heads and halo kept 22 arrows, so +11 %), up to 1.5x (70 px
                                  # arrows along converging normals crossed: half as many fit)
END_MAX_SITES = 1000              # no practical cap: the on-screen gap limits density
END_MIN_SEEN = 0.20               # ... seen (on) in >= 20 % of the iterates' frames
END_MIN_GAP_PX = 18               # ... and no on-screen overlap: arrow segments >= 18 px apart
                                  #     whenever both are on (12 for 12 px heads, 17 for 17 px).
                                  #     Measured with 20 px heads and the 4 px halo (base, gap ->
                                  #     arrows): 56/24 -> 22, 50/24 -> 23, 45/24 -> 25, 50/21 -> 25,
                                  #     50/20 -> 26, 45/20 -> 28, 50/18 -> 31 (the user: >= ~30);
                                  #     footprints checked for contact in the rendered ID shots
SMOOTH_SIGMA = 6.0                # the arrows' direction: Gaussian over iterations (3 left 25 deg jumps)
OARROW_PX, OARROW_HEAD_PX, OARROW_GAP_PX = 110, 20, 6  # length / head width (17 -> 20, +20 %,
                                  # the user) / gap (px)
OARROW_SHAFT_PX = 7.0             # shaft diameter (px).  The user ("the arrows are not very
                                  # visible"): shaft 1.5x the former 4.7 px, head 1.4x the former
                                  # 12 px; with the 4 px white halo the shaft is 15 px = 2.8 px on a
                                  # 360 px phone.  The head's length keeps ar_prep's head proportions.
import dv_change as _DC  # noqa: E402  (the change map; the arrows take its own navy / teal)
GROW_RGB = tuple(_DC._hex(_DC.ARROW_GROW))      # #253494 navy (dv_change)
SHRINK_RGB = tuple(_DC._hex(_DC.ARROW_SHRINK))  # #0c7c99 deep teal


def oarrow_mesh(tail, tip, ppm):
    """One change arrow (shaft cylinder + head cone, closed) at OARROW_SHAFT_PX /
    OARROW_HEAD_PX on screen (ppm: px per m at the arrow): ar_prep.arrow_mesh's geometry,
    but with the shaft and the head sized separately (its one scale ties the shaft to the
    head; the user asked for 1.5x the shaft but 1.4x the head)."""
    ax = tip - tail
    ln = np.linalg.norm(ax)
    a = ax / ln
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(a, u)
    rs, rh = 0.5 * OARROW_SHAFT_PX / ppm, 0.5 * OARROW_HEAD_PX / ppm
    hl = min(rh * ar_prep.ARROW_HEAD_L / ar_prep.ARROW_HEAD_R, 0.45 * ln)
    sides = 16
    th = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = lambda c, r: c + r * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * w)  # noqa: E731
    neck = tip - hl * a
    V = np.vstack([ring(tail, rs), ring(neck, rs), ring(neck, rh), [tail, tip]])
    i0, i1, i2, ct, cp = 0, sides, 2 * sides, 3 * sides, 3 * sides + 1
    F = []
    for k in range(sides):
        k1 = (k + 1) % sides
        F += [[i0 + k, i0 + k1, i1 + k1], [i0 + k, i1 + k1, i1 + k],      # shaft
              [ct, i0 + k1, i0 + k],                                      # tail cap
              [i1 + k, i1 + k1, i2 + k1], [i1 + k, i2 + k1, i2 + k],      # head base ring
              [i2 + k, i2 + k1, cp]]                                      # cone
    return V, np.array(F)


CHANGE_SMOOTH_ITERS = 6   # neighbour averaging of change_h (~1 h): the user saw thin orange
                          # lines along the slats' sharp edges, where each iterate's marching-
                          # cubes edge sits a little off it_0's and flickers; broad changes stay


def smooth_on_mesh(x, F, iters):
    """``iters`` passes of one-ring averaging of a vertex scalar (each vertex with its
    neighbours, equal weights)."""
    from scipy.sparse import coo_matrix
    n = len(x)
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A_ = coo_matrix((np.ones(2 * len(e)), (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])),
                    shape=(n, n)).tocsr()
    A_.data[:] = 1.0
    deg = np.asarray(A_.sum(1)).ravel() + 1.0
    for _ in range(iters):
        x = (A_ @ x + x) / deg
    return x


def view_dir_np(heading_deg, elev_deg):
    a, e = math.radians(heading_deg), math.radians(elev_deg)
    return np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])


def close_params(bench, focus):
    """The beat's close-up (the user: "closer view focused on ONE side of the bench"):
    the camera orbits the focus point (the focused end's change sites' centroid) at the
    distance that frames CLOSE_SPAN_M of height in the 780 px viewport, its focal point
    slid along the camera's own right/up axes so the focus lands on TL.CLOSE_CENTRE at
    every heading.  The rest of the bench may run off the frame's left/top/bottom (now
    intended); the compose fades the body out across BODY_CLEAR_X so it never reaches
    the plot."""
    tan_ = math.tan(math.radians(A.VIEW_ANGLE) / 2)
    vp = TL.VIEWPORT[1] - TL.VIEWPORT[0]
    ppm = vp / TL.CLOSE_SPAN_M
    Dc = (TL.H / 2) / (ppm * tan_)
    return {"focus": np.asarray(focus, float).tolist(), "distance": Dc, "px_per_m": ppm,
            "zoom": bench["distance"] / Dc,
            "ox": (TL.W / 2 - TL.CLOSE_CENTRE[0]) / ppm, "oy": (TL.CLOSE_CENTRE[1] - TL.H / 2) / ppm}


def bench_cam(bench, cl, t, z=None):
    """(focal, distance, heading, elev) of the bench at t (render_dv.bench_camera's twin):
    the orbit camera eased by closeup(t) into the close-up's (focus + right ox + up oy)."""
    z = TL.closeup(t) if z is None else z
    h = TL.bench_heading(t, bench["heading0"])
    _, r, u = basis(h, bench["elev"])
    c0 = np.asarray(bench["centre"], float)
    cc = np.asarray(cl["focus"], float) + r * cl["ox"] + u * cl["oy"]
    return c0 + z * (cc - c0), 1.0 / ((1 - z) / bench["distance"] + z / cl["distance"]), h, bench["elev"]


def prep_cleanup(work, bench, ctx):
    """The optimisation beat: the clean-up's 41 iterates in the uniform zero-|u| colour,
    the strain-energy curve, the change arrows (one mesh per frame), the beat's framing,
    and the optimised bench's |u| (its per-part solve) for the shrink and the tile."""
    from scipy.ndimage import gaussian_filter1d
    import dv_cleanup as DC
    crec = json.load(open(os.path.join(DC.CACHE, "dv_cleanup.json")))
    setup = json.load(open(os.path.join(DC.RUN, "setup.json")))
    assert os.path.realpath(setup["session"]) == os.path.realpath(bench["session"]), \
        "the clean-up is not of the video's bench session"
    h = float(setup["h"])
    its = []
    for k in range(TL.N_ITERS + 1):
        z = np.load(os.path.join(DC.RUN, "surfaces", f"it_{k}.npz"))
        its.append((z["v"].astype(np.float64), z["f"].astype(np.int64)))
    traj = DC.read_traj(os.path.join(DC.RUN, "traj.tsv"))
    assert [int(r["it"]) for r in traj] == list(range(TL.N_ITERS + 1))
    se = np.array([float(r["se"]) for r in traj])
    se_norm = (se / se[0]).tolist()
    # -- the change field and the arrow sites ------------------------------------------
    bvh = [DC.BVH(V, F) for V, F in its]
    # each iterate coloured by its signed change from it_0 (h) under the clean-up figure's
    # lut (the user: as shape_opt_merged panel c; render_prep.py's change_h): white at
    # it_0, red where material grows.  The |u| look crossfades straight into it_0's white
    # (0.97 grey, Delta E76 8.3 from the zero-|u| cream #ebe6db: close, and crossfaded)
    import dv_change
    change_stats = []
    for k, (V, F) in enumerate(its):
        if k == 0:
            ch = np.zeros(len(V))
        else:
            _, dk, sk = bvh[0].closest(V)
            ch = smooth_on_mesh(np.where(sk < 0, -dk, dk) / h, F, CHANGE_SMOOTH_ITERS)
        change_stats.append({"it": k, "max_h": float(ch.max()), "min_h": float(ch.min()),
                             "p99_abs_h": float(np.percentile(np.abs(ch), 99))})
        ar_prep.write_polydata(os.path.join(work, f"ar_0_it{k:02d}.vtk"), V, F,
                               point_rgb=dv_change.change_rgb(ch))
    V40 = its[-1][0]
    _, dist, sign = bvh[0].closest(V40)
    sd = np.where(sign < 0, -dist, dist) / h                  # + = material added (outside it_0)
    def track(sites):
        """Each site's anchor, tracked back through the iterates (nearest point on it_k
        of its point on it_k+1), its outward normal, and the smoothed direction / rate.
        The surface moves along its normal; the anchor chain also slides sideways (a
        junction filling in drags its nearest point along), so the arrow takes the
        NORMAL: the anchor's normal, Gaussian-smoothed over iterations, signed by the
        site's change; its rate is the normal speed, smoothed the same way."""
        P = np.zeros((len(sites), TL.N_ITERS + 1, 3))
        N = np.zeros_like(P)
        Cc = np.zeros((len(sites), TL.N_ITERS + 1))
        P[:, -1] = V40[[s_["i"] for s_ in sites]]
        for k in range(TL.N_ITERS - 1, -1, -1):
            P[:, k] = bvh[k].closest(P[:, k + 1], points=True)[3]
        for k in range(TL.N_ITERS + 1):
            V, F = its[k]
            face = bvh[k].closest(P[:, k])[0]
            fn = np.cross(V[F[face, 1]] - V[F[face, 0]], V[F[face, 2]] - V[F[face, 0]])
            N[:, k] = fn / np.linalg.norm(fn, axis=1, keepdims=True)     # outward (MC winding)
        # each site's change so far: its anchor's signed distance from it_0 (h)
        for k in range(TL.N_ITERS + 1):
            _, dk, sk = bvh[0].closest(P[:, k])
            Cc[:, k] = np.where(sk < 0, -dk, dk) / h
        dP = np.diff(P, axis=1)
        Nm = 0.5 * (N[:, 1:] + N[:, :-1])
        Ns = gaussian_filter1d(Nm, SMOOTH_SIGMA, axis=1, mode="nearest")
        Ns /= np.linalg.norm(Ns, axis=2, keepdims=True)
        sg = np.array([s_["sign"] for s_ in sites], float)[:, None]
        for j, s_ in enumerate(sites):
            tot = P[j, -1] - P[j, 0]
            s_.update(anchor0=P[j, 0], anchor40=P[j, -1], move_mm=float(np.linalg.norm(tot) * 1e3),
                      normal_move_mm=float(sg[j, 0] * (dP[j] * Nm[j]).sum() * 1e3),
                      turn_deg_max=float(np.degrees(np.arccos(np.clip(
                          (Ns[j, 1:] * Ns[j, :-1]).sum(1), -1, 1))).max()),
                      turn_deg_total=float(np.degrees(np.arccos(np.clip(Ns[j, 0] @ Ns[j, -1], -1, 1)))))
        return P, N, Ns, Cc

    # -- the coarse sites (largest growth, 15 cm apart) choose the close-up's end -------
    coarse = []
    for i in np.argsort(-sd):
        if sd[i] < SITE_MIN_H or len(coarse) >= SITES_GROW:
            break
        if all(np.linalg.norm(V40[i] - V40[s_["i"]]) > SITE_SEP_M for s_ in coarse):
            coarse.append({"i": int(i), "sign": 1, "change_h": float(sd[i])})
    track(coarse)
    mid_x = 0.5 * (V40[:, 0].min() + V40[:, 0].max())
    ends = {}
    for s_ in coarse:
        ends.setdefault("-x" if s_["anchor40"][0] < mid_x else "+x", []).append(s_)
    totals = {e: sum(abs(s_["change_h"]) for s_ in v) for e, v in ends.items()}
    end = max(totals, key=totals.get)
    focus = np.mean([s_["anchor40"] for s_ in ends[end]], axis=0)
    cl = close_params(bench, focus)
    cl.update(end=end, end_totals_h=totals, end_coarse_sites=len(ends[end]))
    # -- the close-up's arrows: dense sites on that end ----------------------------------
    # The user (2026-11-08): emphasis by DENSITY, not size -- spacing shrinks steeply with
    # the change, arrows nearly uniform; the threshold tied to the colour map (0.5 h, where
    # the lut has clearly left white for orange); any visibly red patch at the end gets an
    # arrow (coverage check below); ONE visibility per arrow, owned by the compose (no
    # geometric growth or gating here: two mechanisms fighting made arrows blink).
    on_end = (V40[:, 0] < mid_x) if end == "-x" else (V40[:, 0] >= mid_x)
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix

    def components(V, F):
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]]])
        return connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])),
                                               shape=(len(V), len(V))), directed=False)

    comp = [components(V, F) for V, F in its]
    csize = [np.bincount(labv[F[:, 0]], minlength=nc) for (nc, labv), (V, F) in zip(comp, its)]
    sd_s = smooth_on_mesh(sd, its[-1][1], CHANGE_SMOOTH_ITERS)   # the change as coloured

    def spacing(c):
        return max(END_SEP_MIN, END_SEP_M0 * (END_MIN_H / abs(c)) ** END_SEP_POW)

    cand = []
    for i in np.argsort(-np.abs(sd_s)):        # biggest change first; its own (smaller)
        if abs(sd_s[i]) < END_MIN_H:           # change sets a candidate's spacing
            break
        if on_end[i] and all(np.linalg.norm(V40[i] - V40[s_["i"]]) > spacing(sd_s[i]) for s_ in cand):
            cand.append({"i": int(i), "sign": 1 if sd_s[i] > 0 else -1, "change_h": float(sd_s[i])})
    frames = [(f, t) for f, t in TL.frame_times(1) if TL.arrows_visible(t)]
    removed_piece = []

    def build(cand):
        """Track every candidate; per frame its arrow (tail, tip), whether it is seen
        (on the close-up's left half, not end-on: projected / full length), its screen
        mid point, its side-on area, its change so far; and the small-piece test."""
        P, N, Ns, Cc = track(cand)
        piece = np.ones((len(cand), TL.N_ITERS + 1))
        for k in range(TL.N_ITERS + 1):
            nc, labv = comp[k]
            face = bvh[k].closest(P[:, k])[0]
            piece[:, k] = csize[k][labv[its[k][1][face, 0]]] / csize[k].max()
        cmax = max(abs(s_["change_h"]) for s_ in cand)
        geo = []
        for f, t in frames:
            c, D, hd, el = bench_cam(bench, cl, t)
            cam = c + D * view_dir_np(hd, el)
            ppm = (TL.H / 2) / (D * math.tan(math.radians(A.VIEW_ANGLE) / 2))
            pos = TL.iter_pos(t)
            k0 = min(int(pos), TL.N_ITERS - 1)
            w = pos - k0
            q = np.clip(pos - 0.5, 0, TL.N_ITERS - 1)
            q0 = min(int(q), TL.N_ITERS - 2)
            row = []
            for j, s_ in enumerate(cand):
                p = (1 - w) * P[j, k0] + w * P[j, k0 + 1]
                dq = (1 - (q - q0)) * Ns[j, q0] + (q - q0) * Ns[j, q0 + 1]
                dirn = s_["sign"] * dq / np.linalg.norm(dq)
                c_now = abs((1 - w) * Cc[j, k0] + w * Cc[j, k0 + 1])
                # nearly uniform: 70 px, up to 1.5x with the change so far (the user)
                L_px = OARROW_BASE_PX * (1 + 0.5 * float(np.clip((c_now - END_MIN_H) / max(cmax - END_MIN_H, 1e-9), 0, 1)))
                L = L_px / ppm
                base = p + OARROW_GAP_PX / ppm * (s_["sign"] * dirn)
                tail, tip = (base, base + L * dirn) if s_["sign"] > 0 else (base - L * dirn, base)
                mid = 0.5 * (tail + tip)
                v = mid - cam
                dv = np.linalg.norm(v)
                sin_t = float(np.linalg.norm(np.cross(dirn, v / dv)))
                px, py = project(mid[None], c, hd, el, D, TL.W, TL.H)
                on_frame = 40 <= px[0] <= TL.BODY_CLEAR_X[0] - 40 and 210 <= py[0] <= TL.H - 30
                hp = OARROW_HEAD_PX
                area = (L_px * OARROW_SHAFT_PX
                        + 0.5 * hp * min(hp * ar_prep.ARROW_HEAD_L / (2 * ar_prep.ARROW_HEAD_R), 0.45 * L_px))
                tx_, ty_ = project(np.array([tail, tip]), c, hd, el, D, TL.W, TL.H)
                row.append({"tail": tail, "tip": tip, "seen": sin_t if on_frame else -1.0,
                            "seg": (float(tx_[0]), float(ty_[0]), float(tx_[1]), float(ty_[1])),
                            "mid": (float(px[0]), float(py[0])), "mid3": mid, "cam": cam,
                            "area": area, "c_now": c_now, "ppm": ppm})
            geo.append(row)
        return P, Cc, piece, geo

    def hysteresis(raw):
        st_ = np.zeros_like(raw)
        for j in range(raw.shape[1]):
            on = False
            for n in range(raw.shape[0]):
                if raw[n, j] < 0.45:
                    on = False
                elif raw[n, j] >= 0.55:
                    on = True
                st_[n, j] = 1.0 if on else 0.0
        return st_

    def seg_dist(a, b):
        """Min distance between screen segments a, b (..., 4) = (x0, y0, x1, y1)."""
        def pt_seg(p, s0, s1):
            d = s1 - s0
            tt = np.clip(((p - s0) * d).sum(-1) / np.maximum((d * d).sum(-1), 1e-9), 0, 1)
            return np.linalg.norm(p - (s0 + tt[..., None] * d), axis=-1)
        a0, a1, b0, b1 = a[..., :2], a[..., 2:], b[..., :2], b[..., 2:]
        dd = np.minimum(np.minimum(pt_seg(a0, b0, b1), pt_seg(a1, b0, b1)),
                        np.minimum(pt_seg(b0, a0, a1), pt_seg(b1, a0, a1)))
        # crossing segments: distance 0
        def cross(o, p, q):
            return (p[..., 0] - o[..., 0]) * (q[..., 1] - o[..., 1]) - (p[..., 1] - o[..., 1]) * (q[..., 0] - o[..., 0])
        x = (np.sign(cross(a0, a1, b0)) != np.sign(cross(a0, a1, b1))) & \
            (np.sign(cross(b0, b1, a0)) != np.sign(cross(b0, b1, a1)))
        return np.where(x, 0.0, dd)

    pause_n = [n for n, (f, t) in enumerate(frames) if TL.ITERS[1] - 1e-6 <= t < TL.SHRINK0]
    # ... and the end fade's first frames (alpha > 0.3): an arrow full at the pause's end
    # cannot fade (1/15 a frame) before the body hides it there, so it must stay in view
    early_n = [n for n, (f, t) in enumerate(frames) if TL.SHRINK0 <= t and TL.arrows_opacity(t) > 0.3]

    def pause_vis(geo, js, ns=9, which=None):
        """(pause frames, len(js)) unoccluded share of each arrow on it_40: ns points along
        tail -> tip, each a ray from the camera against the body's BVH."""
        which = pause_n if which is None else which
        out = np.zeros((len(which), len(js)))
        if not js:
            return out
        w_ = (np.arange(ns) + 0.5) / ns
        for r, n in enumerate(which):
            f, t = frames[n]
            c, D, hd, el = bench_cam(bench, cl, t)
            cam = c + D * view_dir_np(hd, el)
            pts = np.concatenate([(1 - w_)[:, None] * geo[n][j]["tail"] + w_[:, None] * geo[n][j]["tip"]
                                  for j in js])
            d_ = pts - cam
            dist = np.linalg.norm(d_, axis=1)
            hit = bvh[-1].first_hit(np.repeat(cam[None], len(pts), 0), d_ / dist[:, None], 10 * D)
            out[r] = (hit >= dist - 1e-3).reshape(len(js), ns).mean(1)
        return out

    def pause_shown(geo, state, js):
        """Per arrow of js: shown in the end pause (>= PAUSE_MIN_SHARE unoccluded and not
        end-on in >= PAUSE_MIN_FRAC of its frames), and the per-frame shown mask."""
        pv = pause_vis(geo, js)
        okf = (pv >= PAUSE_MIN_SHARE) & (state[pause_n][:, js] > 0)
        shown = okf.mean(0) >= PAUSE_MIN_FRAC if len(pause_n) else np.ones(len(js), bool)
        if early_n and len(js):
            shown &= (pause_vis(geo, js, which=early_n) >= PAUSE_MIN_SHARE).all(0)
        return shown, okf

    def select(cand, geo, piece, forced):
        raw = np.array([[a_["seen"] for a_ in g] for g in geo])
        segs = np.array([[a_["seg"] for a_ in g] for g in geo])     # (frames, cand, 4)
        state = hysteresis(raw)
        it_frames = np.array([TL.ITERS[0] <= t <= TL.ITERS[1] for _, t in frames])
        seen = state[it_frames].mean(0)
        # forced (coverage) groups: of each uncovered patch's vertices, the one whose arrow
        # is seen best over the check frames (none if every one is end-on there)
        chk_ = [n for n, (f, t) in enumerate(frames) if TL.ITERS[1] - 1e-6 <= t < TL.SHRINK0]   # it_40 on screen (the arrows are fading out
                                                            # through it now, but all still on)
        keep = []
        for grp in forced:
            ok_ = [j for j in grp if piece[j].min() >= END_PIECE_MIN]
            if not ok_:
                continue
            # the one shown best through the end pause (unoccluded >= 60 % and not end-on)
            sh_, okf_ = pause_shown(geo, state, ok_)
            score = okf_.mean(0) * sh_ if len(pause_n) else np.zeros(len(ok_))
            best = ok_[int(np.argmax(score))]
            if score.max() >= PAUSE_MIN_FRAC:
                keep.append(best)
        for j in sorted(range(len(cand)), key=lambda j: -abs(cand[j]["change_h"])):
            if j in keep or seen[j] < END_MIN_SEEN or piece[j].min() < END_PIECE_MIN:
                continue
            # no overlap on screen with any kept arrow, whenever both are on (the user:
            # dense, as long as arrows don't overlap): segment-to-segment distance
            if keep:
                kk = np.array(keep)
                both = (state[:, j][:, None] > 0) & (state[:, kk] > 0)          # (frames, K)
                if both.any():
                    dmin = seg_dist(np.broadcast_to(segs[:, j][:, None], segs[:, kk].shape), segs[:, kk])
                    if (dmin[both] < END_MIN_GAP_PX).any():
                        continue
            keep.append(j)
        return keep, state, seen

    def coverage(keep, geo, state, P, Cc):
        """On the last iterate's frame and the end pause: every connected visible patch
        of it_40 with |change| >= the threshold (>= COVER_MIN_VERTS vertices) must hold a
        visible arrow (one whose site lies within COVER_NEAR_M of the patch, on there,
        past the threshold and not behind the body).  Returns the uncovered patches."""
        V, F = its[-1]
        chk = [n for n, (f, t) in enumerate(frames) if TL.ITERS[1] - 1e-6 <= t < TL.SHRINK0]   # it_40 on screen (the arrows are fading out
                                                            # through it now, but all still on)
        nc40, lab40 = comp[-1]
        vsize = csize[-1][lab40] / csize[-1].max()           # per vertex: its piece's share
        hot = (np.abs(sd_s) >= END_MIN_H) & (vsize >= END_PIECE_MIN)   # fragments carry no arrows
        e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        out = []
        shown, okf = pause_shown(geo, state, keep)
        for n in chk:
            f, t = frames[n]
            c, D, hd, el = bench_cam(bench, cl, t)
            cam = c + D * view_dir_np(hd, el)
            idx = np.nonzero(hot)[0]
            x, y = project(V[idx], c, hd, el, D, TL.W, TL.H)
            onf = (x >= 0) & (x <= TL.BODY_CLEAR_X[0]) & (y >= 210) & (y <= TL.H)
            idx = idx[onf]
            if len(idx) == 0:
                continue
            dirs = V[idx] - cam
            dist = np.linalg.norm(dirs, axis=1)
            hit = bvh[-1].first_hit(np.repeat(cam[None], len(idx), 0), dirs / dist[:, None], 10 * D)
            vis_v = np.zeros(len(V), bool)
            vis_v[idx[hit >= dist - 0.003]] = True
            ee = e[vis_v[e[:, 0]] & vis_v[e[:, 1]]]
            nc, lab = connected_components(coo_matrix((np.ones(len(ee)), (ee[:, 0], ee[:, 1])),
                                                      shape=(len(V), len(V))), directed=False)
            # visible arrows: shown in the pause (pause_shown) and, this frame, >= 60 %
            # unoccluded and not end-on (half-hidden ones and head-only stubs don't count)
            arrows_vis = [P[j, -1] for q, j in enumerate(keep)
                          if shown[q] and okf[pause_n.index(n), q] and geo[n][j]["c_now"] >= END_MIN_H]
            arrows_vis = np.array(arrows_vis) if arrows_vis else np.zeros((0, 3))
            for comp_id in np.unique(lab[vis_v]):
                vv = np.nonzero(vis_v & (lab == comp_id))[0]
                if len(vv) < COVER_MIN_VERTS:
                    continue
                ok = len(arrows_vis) and (np.linalg.norm(V[vv][:, None] - arrows_vis[None], axis=2).min() <= COVER_NEAR_M)
                if not ok and os.environ.get("DV_COVER_DEBUG"):
                    dk = [(j, float(np.linalg.norm(V[vv] - P[j, -1], axis=1).min()), float(state[n, j]),
                           round(geo[n][j]["c_now"], 2)) for j in keep]
                    dk.sort(key=lambda x: x[1])
                    print("   uncovered", f, V[vv].mean(0).round(3).tolist(), len(vv),
                          round(float(np.abs(sd_s[vv]).max()), 2), "nearest kept", dk[:3], flush=True)
                if not ok:
                    out.append({"frame": f, "verts": vv, "best": int(vv[np.argmax(np.abs(sd_s[vv]))]),
                                "centre": V[vv].mean(0).round(3).tolist(),
                                "max_h": float(np.abs(sd_s[vv]).max())})
        return out, len(chk)

    forced, exempt = [], []
    for round_ in range(4):
        P, Cc, piece, geo = build(cand)
        keep, state, seen = select(cand, geo, piece, forced)
        uncovered, n_chk = coverage(keep, geo, state, P, Cc)
        print(f"  arrows round {round_}: {len(keep)} of {len(cand)}; uncovered patches "
              f"{len(uncovered)} over {n_chk} check frames", flush=True)
        # a patch already tried (all its vertices forced) and still uncovered faces the
        # camera squarely: every arrow there is end-on (a dot), so it is exempt
        tried = {tuple(sorted(g_)) for g_ in forced}
        new = []
        for u in uncovered:
            grp = []
            for i in u["verts"]:
                if any(s_["i"] == int(i) for s_ in cand):
                    grp.append(next(j for j, s_ in enumerate(cand) if s_["i"] == int(i)))
                else:
                    cand.append({"i": int(i), "sign": 1 if sd_s[i] > 0 else -1,
                                 "change_h": float(sd_s[i]), "forced": True})
                    grp.append(len(cand) - 1)
            if tuple(sorted(grp)) in tried:
                key = tuple(np.round(its[-1][0][u["verts"]].mean(0), 2))
                if key not in [e_["centre"] for e_ in exempt]:
                    exempt.append({"centre": key, "n_verts": int(len(u["verts"])),
                                   "reason": "every arrow there is end-on or mostly hidden in the end pause"})
            elif grp not in new:
                new.append(grp)
        if not new:
            break
        forced += new
    ex_c = {e_["centre"] for e_ in exempt}
    uncovered = [u for u in uncovered if tuple(np.round(its[-1][0][u["verts"]].mean(0), 2)) not in ex_c]
    coverage_result = {"threshold_h": END_MIN_H, "rounds": round_ + 1,
                       "exempt_end_on": [dict(e_, centre=list(e_["centre"])) for e_ in exempt],
                       "uncovered_after": [{"frame": u["frame"], "n_verts": int(len(u["verts"]))} for u in uncovered],
                       "check_frames": n_chk, "min_patch_verts": COVER_MIN_VERTS, "near_m": COVER_NEAR_M}
    print(f"  coverage check: {'PASS' if not uncovered else 'FAIL'} ({len(uncovered)} uncovered patches "
          f"on the last iterate + end-pause frames; exempt (end-on) {[(e_['centre'], e_['n_verts']) for e_ in exempt]})",
          flush=True)
    removed_piece = [{"anchor40": P[j, -1].tolist(), "change_h": cand[j]["change_h"],
                      "min_piece_fraction": float(piece[j].min())}
                     for j in range(len(cand)) if piece[j].min() < END_PIECE_MIN]
    sites = [dict(cand[j], seen_fraction=float(seen[j]), anchor0=P[j, 0], anchor40=P[j, -1])
             for j in keep]

    def region(a):            # for the record: where on the end frame a site sits
        return "legs/feet" if a[1] < 0.42 else ("arm" if a[2] > 0.55 else "junction")
    for s_ in sites:
        s_["region"] = region(s_["anchor40"])
    regions = {r_: sum(s_["region"] == r_ for s_ in sites) for r_ in ("junction", "arm", "legs/feet")}
    count_on = {}
    for k in (0, 10, 20, 40):
        n_ = int(np.argmin([abs(TL.iter_pos(t) - k) for _, t in frames]))
        count_on[f"it_{k}"] = int(sum(1 for j in keep if state[n_, j] > 0 and geo[n_][j]["c_now"] >= END_MIN_H))
    # every kept arrow at full size in every frame (the compose owns its visibility):
    # the colour mesh, its flat ID twin (red channel = m + 1), and per frame its side-on
    # area and whether its site has passed the threshold
    shown_k, _ = pause_shown(geo, state, keep)
    meta = {"t_cross": {}, "nominal": {},
            # the end pause shows these (the compose holds the rest off through it, faded
            # out ahead; dv_compose.arrow_fades)
            "pause_on": {str(m + 1): bool(shown_k[m]) for m in range(len(keep))}}
    print(f"  end pause: {int(shown_k.sum())} of {len(keep)} arrows shown "
          f"(>= {PAUSE_MIN_SHARE:.0%} unoccluded in >= {PAUSE_MIN_FRAC:.0%} of its frames)", flush=True)
    for m, j in enumerate(keep):
        tc = next((t for n, (f, t) in enumerate(frames) if geo[n][j]["c_now"] >= END_MIN_H), None)
        meta["t_cross"][str(m + 1)] = tc
    for n, (f, t) in enumerate(frames):
        vs, fs, cols, ids, off = [], [], [], [], 0
        for m, j in enumerate(keep):
            a_ = geo[n][j]
            meta["nominal"].setdefault(str(f), {})[str(m + 1)] = float(a_["area"])
            v_, f_ = oarrow_mesh(a_["tail"], a_["tip"], a_["ppm"])
            vs.append(v_)
            fs.append(f_ + off)
            off += len(v_)
            cols.append(np.tile(np.array(GROW_RGB if cand[j]["sign"] > 0 else SHRINK_RGB), (len(f_), 1)))
            ids.append(np.tile(np.array([(m + 1) / 255, 0.0, 0.0]), (len(f_), 1)))
        for sub, cl_ in (("oarrows", cols), ("oarrows_id", ids)):
            path = os.path.join(work, sub, f"f{f:04d}.vtk")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if vs:
                ar_prep.write_polydata(path, np.vstack(vs), np.vstack(fs), cell_rgb=np.vstack(cl_))
            else:      # nothing visible: one degenerate (zero-area) triangle
                ar_prep.write_polydata(path, np.zeros((3, 3)), np.array([[0, 1, 2]]),
                                       cell_rgb=np.array([GROW_RGB]))
    json.dump(meta, open(os.path.join(work, "arrow_meta.json"), "w"))
    # -- the optimised bench's |u| (its per-part solve), and the shrink's box --------------
    vtu = os.path.join(ar_prep.REPO, crec["parts_final"]["vtu"])
    fields, err, nn, ne = ar_prep.interpolate_from_vtu(vtu, V40, *ctx)
    inside = err < 1e-6 * bench["extent"]
    assert inside.mean() > 0.95, f"only {inside.mean():.1%} of it_40 inside the parts solve mesh"
    u = fields["displacement"]
    umag = np.linalg.norm(u, axis=1)
    ar_prep.write_polydata(os.path.join(work, "ar_0_optsolve.vtk"), V40, its[-1][1],
                           vectors=u, scalars=umag)
    k = bench["k"]
    xs, ys = [], []
    for t in np.arange(TL.SHRINK0, TL.DURATION, 0.25):
        c, D, hd, el = bench_cam(bench, cl, t, z=0.0)     # the tile's frame: the orbit camera
        for Pts in (V40, V40 + k * u):
            x, y = project(Pts, c, hd, el, D, TL.W, TL.H)
            xs += [x.min(), x.max()]
            ys += [y.min(), y.max()]
    box_close = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
    # how much of the body the beat's close-up puts right of the clear band (faded there)
    xs_ = []
    for t in np.arange(TL.MOVE_IN[1], TL.SHRINK0, 0.1):
        c, D, hd, el = bench_cam(bench, cl, t)
        x, y = project(V40[::5], c, hd, el, D, TL.W, TL.H)
        xs_.append(float(np.mean(x > TL.BODY_CLEAR_X[0])))
    res = json.load(open(os.path.join(DC.RUN, "results.json")))
    rec = {"source": "experiments/geometry_cleanup run.py, re-run by dv_cleanup.py with "
                     "GEOMETRY_CLEANUP_SAVE_EVERY=1",
           "archive": "output/archives/geometry_cleanup.tar.zst (geometry_cleanup/park-bench)",
           "gate_vs_archive": {k_: v for k_, v in crec.items() if k_ != "parts_final"},
           "settings": {"material": "uniform stand-in " + json.dumps(setup["standin_material"]),
                        "h_m": h, "step_h": res["step_h"], "cap": res["cap"],
                        "reinit": res["reinit"], "gd_lr_eff": res["step"]["gd_lr_eff"]},
           "se_norm": se_norm, "se_final_pct": 100 * (se_norm[-1] - 1), "vol_ratio": res["vol_ratio"],
           "change_h": {"grow_max": float(sd.max()), "shrink_max": float(-sd.min()),
                        "p1": float(np.percentile(sd, 1)), "p99": float(np.percentile(sd, 99))},
           "change_colour": {"source": "experiments/geometry_cleanup/render.py lut() (read from source)",
                             "change_max_h": dv_change.CHANGE_MAX, "h_mm": h * 1e3,
                             "per_iterate": change_stats},
           "coarse_sites": coarse, "sites": sites, "candidates": len(cand), "arrows_on": count_on, "pause_shown": int(shown_k.sum()),
           "sites_per_region": regions, "removed_on_small_piece": removed_piece,
           "coverage_check": coverage_result,
           "site_rule": {"min_h": END_MIN_H, "max": END_MAX_SITES,
                         "sep": f"max({END_SEP_MIN}, {END_SEP_M0} (min_h / c)^{END_SEP_POW}) m",
                         "piece_min": END_PIECE_MIN,
                         "min_seen": END_MIN_SEEN, "min_gap_px": END_MIN_GAP_PX},
           "smooth_sigma_iters": SMOOTH_SIGMA,
           "arrow_frames": len(frames),
           "arrow_colours": {"grow": _DC.ARROW_GROW, "shrink": _DC.ARROW_SHRINK, "halo": "#ffffff",
                             "change_map_ends": _DC.ENDS,
                             "cyan_mid_knot": {"h": _DC.CYAN_MID_H, "k": _DC.CYAN_MID_K},
                             "why": "navy = the +1 h stop (9.8 on white), deep teal = the -3 h "
                                    "end (4.5 on white, 2.0 on the 1 h cyan); the 4 px white halo "
                                    "separates them from the coloured surface and the white body"},
           "arrow_px": {"shaft": OARROW_SHAFT_PX, "head": OARROW_HEAD_PX, "base_len": OARROW_BASE_PX, "halo": 4.0},
           "close": cl, "beat_vertex_fraction_right_of_clear_band_max": max(xs_), "box_close_px": box_close,
           "opt_solve": {"vtu": crec["parts_final"]["vtu"], "nodes": nn, "hexes": ne,
                         "umax_surface": float(umag.max()), "umax_session": bench["umax"],
                         "umax_ratio": float(umag.max()) / bench["umax"],
                         "inside_fraction": float(inside.mean()),
                         "coverage": crec["parts_final"]["coverage"],
                         "lut_range": "the session's |u|max, so the stiffer bench reads lighter"}}
    print(f"beat: SE/SE0 {se_norm[-1]:.4f}; end {end} {totals}; {len(sites)} of {len(cand)} sites "
          f"({sum(s_['sign'] > 0 for s_ in sites)} grow, {sum(s_['sign'] < 0 for s_ in sites)} shrink; "
          f"on at it_0/20/40 {count_on}; regions {regions}; small-piece removed "
          f"{[(round(r_['change_h'], 2), [round(x, 3) for x in r_['anchor40']], round(r_['min_piece_fraction'], 4)) for r_ in removed_piece]}; "
          f"change {[round(s_['change_h'], 2) for s_ in sites]} h, max turn per update "
          f"{max(s_['turn_deg_max'] for s_ in sites):.1f} deg); close {cl}; right of band "
          f"{max(xs_):.3f}; |u|max opt {umag.max():.4g} vs {bench['umax']:.4g}; "
          f"shrink box {np.round(box_close, 1).tolist()}")
    return rec


# -- tiles --------------------------------------------------------------------------------
def prep_tile(j, name, work, ctx, recs):
    run = ar_runs.load_run(os.path.join(BATCH, name))
    s = run.session
    fig = recs.get(run_key(s))
    assert fig is not None, f"{name}: no figure record to gate against"
    mesh = ar_prep.load_mesh(os.path.join(s, "geometry", "physical.glb"))
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    L = float(np.ptp(V, axis=0).max())
    tf = os.path.join(s, "geometry", "physical.glb.transform.json")
    mat_p = os.path.join(s, "case", "material.json")
    mat = json.load(open(mat_p)) if os.path.exists(mat_p) else {}
    units = (json.load(open(tf)).get("units") if os.path.exists(tf) else None) or mat.get("unit_system")
    rec = {"notes": [], "unit_system": units, "bc": {"regions": gl_prep.bc_regions(s, mesh)}}
    out = {}
    ar_prep._prep_solve(rec, out, run, s, V, L, mat, ctx)
    k, k_fig = rec["scale"]["factor"], fig["scale"]["factor"]
    um, um_fig = rec["solve"]["umax_surface"], fig["solve"]["umax_surface"]
    assert k == k_fig, f"{name}: x{k:g} here, x{k_fig:g} in {fig['_file']}"
    assert abs(um / um_fig - 1) < 1e-6, f"{name}: |u|max {um:.6g} vs {um_fig:.6g}"
    ar_prep.write_polydata(os.path.join(work, f"ar_t{j}_solve.vtk"), V, F,
                           vectors=out["u"], scalars=out["umag"])
    h0, elev = heading_elev(fig["view"]["dir"])
    w_px, h_px = 2 * TL.TILE_BODY_W, 2 * TL.TILE_BODY_H     # rendered at 2x, downsampled
    target = TL.TILE_FILL * 2 * min(TL.TILE_BODY_W, TL.TILE_BODY_H)
    Vd = V + k * out["u"]
    c, D, box = fit_camera([V, Vd], elev, w_px, h_px, target,
                           measure=lambda b: max(b[2] - b[0], b[3] - b[1]))
    t = {"j": j, "name": name, "session": s, "instruction": run.get("instruction"),
         "label": object_name(run.get("instruction")),
         "wall_clock_s": run.get("wall_clock_s"), "record": fig["_file"], "k": k, "umax": um,
         "extent": L, "heading0": h0, "elev": elev, "centre": c.tolist(), "distance": D,
         "box_px_2x": box, "notes": rec["notes"]}
    print(f"[t{j:2d}] {name[:40]:40s} x{k:<6g} heading {h0:7.1f} elev {elev:4.1f} "
          f"{(run.get('wall_clock_s') or 0) / 60:4.1f} min  box {np.round(box, 1).tolist()}")
    return t


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", default=os.path.join(HERE, "_work"))
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    t0 = time.time()
    sys.path.insert(0, os.path.join(ar_prep.REPO, "src", "warp_wrapper"))
    sys.path.insert(0, ar_prep.REPO)
    import torch
    from core.remeshing import octree_hex_interpolate
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = (torch, octree_hex_interpolate, device)
    recs = figure_records()
    bench = prep_bench(args.work, ctx, recs)
    bench["cleanup"] = prep_cleanup(args.work, bench, ctx)
    tiles = [prep_tile(j, n, args.work, ctx, recs) for j, n in enumerate(TILES)]
    walls = [t["wall_clock_s"] / 60 for t in tiles if t.get("wall_clock_s")]
    # the scene-8 title's "6.5 minutes on average": the mean wall clock (metrics.json) of
    # all 50 Opus runs, as the Opus category figures record it; the same numbers are the
    # Opus columns of agentic_results_compare_time.csv's wall_clock row (mean 6.46 min)
    all50 = [r["meta"]["wall_clock_s"] for r in gl_prep.category_records().values()
             if "/agentic_batch50_opus/" in r["session"]]
    mean50 = sum(all50) / len(all50) / 60
    assert len(all50) == 50 and round(mean50, 1) == TL.MEAN_WALL_MIN, (len(all50), mean50)
    stats = {"bench": bench, "tiles": tiles, "batch_root": BATCH,
             "tile_wall_clock_min": [min(walls), max(walls)] if walls else None,
             "mean_wall_clock_min_opus50": mean50,
             "mean_wall_clock_source": "metrics.json wall_clock_s of the 50 Opus runs, via "
                                       "agentic_results/ar_stats_agentic_results_*_opus.json "
                                       "(= the Opus columns of agentic_results_compare_time.csv, "
                                       "wall_clock row)",
             "prep_seconds": time.time() - t0}
    json.dump(stats, open(os.path.join(args.work, "dv_stats.json"), "w"), indent=1,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
    print(f"wrote {args.work}/dv_stats.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
