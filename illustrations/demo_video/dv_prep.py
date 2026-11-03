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
from dv_camera import project  # noqa: E402

BATCH = os.environ.get("DV_BATCH_ROOT", "/home/xl2254/implicit_optimize/output/agentic_batch50_opus")
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
def figure_records():
    """session folder -> its record: the main figure's (ar_stats.json) first, then the
    Opus category figures' (the heading each figure chose for it)."""
    out = {}
    for r in json.load(open(os.path.join(AR, "ar_stats.json")))["runs"]:
        out[os.path.normpath(r["session"])] = dict(r, _file="ar_stats.json")
    for instr, r in gl_prep.category_records().items():
        out.setdefault(os.path.normpath(r["session"]), r)
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
    fig = recs[os.path.normpath(run0.session)]
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
    span = TL.SPIN_DPS * (TL.DURATION - TL.SPIN_START)
    heads = np.arange(h0, h0 - span - 2.0, -2.0)       # the headings the clip shows
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


# -- tiles --------------------------------------------------------------------------------
def prep_tile(j, name, work, ctx, recs):
    run = ar_runs.load_run(os.path.join(BATCH, name))
    s = run.session
    fig = recs.get(os.path.normpath(s))
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
    json.dump(stats, open(os.path.join(args.work, "dv_stats.json"), "w"), indent=1, default=float)
    print(f"wrote {args.work}/dv_stats.json ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
