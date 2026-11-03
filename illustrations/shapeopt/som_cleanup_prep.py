"""shape_opt_merged row c, stage 1 (conda `mechfoundry-warp117`, CPU + a little warp):
the geometry clean-up runs -> BC-panel surfaces and the row's record.

    python som_cleanup_prep.py [--out DIR] [object ...]   # default: the 17 of OBJECTS

Row c shows the geometry clean-up of experiments/geometry_cleanup (a short,
volume-as-is shape optimization of each agentic-batch body, stopped as soon as the
surface has moved a fixed fraction of a leaf).  Each object has two panels from ONE
camera, the clean-up's own left-hand view (experiments/geometry_cleanup/render.py,
VIEWS[0]: azimuth 35 deg, elevation 25 deg, +y up, framed on the starting surface's
box from <run>/render/camera.json; VIEW_OVERRIDE, the dental crown only, turns it):

  BC      the supports and loads on the starting body, in agentic_results' look:
          the session's geometry/physical.glb (the surface the masks in
          boundary_conditions/resolved.npz are defined on) with Dirichlet teal and
          Neumann orange faces, ar_prep.load_arrows along each traction of
          case/case.json, and ar_prep.support_outline around a support that faces
          away from the camera (feet, rocker bottoms).  Written here as
          <out>/<object>_{bc,arrows,support}.vtk; render_som_cleanup.py draws them.
  change  the clean-up's signed change from the start, in leaves h, on the final
          body: the clean-up's own <run>/render/it_<final>.vtk (render_prep.py's
          "change_h"), drawn by render_som_cleanup.py with geometry_cleanup/render.py's
          change-mode code -- the left-hand change tile of
          output/geometry_cleanup/montage_*.png (it_<final>_v0_change.png), only
          dollied back so no body is cut by the frame.

The clean-up runs are read-only here; nothing is re-run.  som_cleanup.json records,
per object, what the compose prints (updates, SE/SE0, V/V0, stop reason, h) and
where each number came from.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "agentic_results"))
# ar_prep imports ar_figure (matplotlib) before trimesh -- keep that order (its note)
import ar_prep as AP  # noqa: E402
import ar_style as A  # noqa: E402

CLEANUP = "/home/xl2254/implicit_optimize/output/geometry_cleanup"
# the clean-up's left-hand view (experiments/geometry_cleanup/render.py VIEWS[0])
VIEW_AZ, VIEW_EL = 35.0, 25.0

# (folder under CLEANUP, the label printed on the figure)
OBJECTS = [
    ("claw-hammer", "Claw hammer"),
    ("crocodile", "Crocodile"),
    ("elephant", "Elephant"),
    ("horse", "Horse"),
    ("human-femur", "Human femur"),
    ("industrial-robot-arm", "Robot arm"),
    ("natural-stone-arch", "Stone arch"),
    ("pantheon", "Pantheon"),
    ("park-bench", "Park bench"),
    ("school-chair-with-a-steel-frame-and-a-plastic-se", "School chair"),
    ("wooden-rocking-horse", "Rocking horse"),
    ("rhinoceros", "Rhinoceros"),
    ("layered-birthday-cake", "Birthday cake"),
    ("car-suspension-control-arm-with-rubber-bushings", "Control arm"),
    ("dental-crown", "Dental crown"),
    ("pogo-stick", "Pogo stick"),
    ("truck-leaf-spring", "Leaf spring"),
]
# The ONE declared per-object view (the user asked for it, 2026-10-14): (azimuth,
# elevation) replacing VIEW_AZ / VIEW_EL for both of that object's panels.  The dental
# crown's change (9 % of its area beyond +-0.25 h, up to +1.2 h) sits on its +z flank,
# which the left-hand view sees edge-on: scoring every 15 deg azimuth x {10..70 deg}
# elevation by the area-weighted |change| facing the camera, azimuth 35 sees 0.59 of the
# best and azimuth 90 at the same 25 deg elevation 0.91 (the best, 10 deg, loses the
# cusps that make it read as a tooth).
VIEW_OVERRIDE = {"dental-crown": (90.0, 25.0)}


def view_dir(az=VIEW_AZ, el=VIEW_EL):
    """Toward the camera, exactly as geometry_cleanup/render.camera builds it for a
    y-up run: horizontal pair (x, z), x at azimuth 0, z at 90."""
    a, b = math.radians(az), math.radians(el)
    return np.array([math.cos(b) * math.cos(a), math.sin(b), math.cos(b) * math.sin(a)])


def prep(name, label, out):
    rd = os.path.join(CLEANUP, name)
    s = os.path.join(rd, "session")
    res = json.load(open(os.path.join(rd, "results.json")))
    cam = json.load(open(os.path.join(rd, "render", "camera.json")))
    assert cam["up"] == "y", f"{name}: the clean-up camera is not y-up"
    final = cam["its"][-1]
    assert res["final_surface"] == f"surfaces/it_{final}.npz", \
        f"{name}: render/ shows iteration {final}, results.json's final is {res['final_surface']}"
    change_png = os.path.join(rd, "render", f"it_{final}_v0_change.png")
    assert os.path.isfile(change_png), change_png
    assert os.path.isfile(os.path.join(rd, "render", f"it_{final}.vtk"))

    mesh = AP.load_mesh(os.path.join(s, "geometry", "physical.glb"))
    V, F = np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64)
    L = float(np.ptp(V, axis=0).max())
    # the BC body and the clean-up's iteration-0 surface must share a frame: the
    # camera is framed on the latter (marching cubes of the same SDF, within a leaf)
    lo, hi = np.array(cam["lo"]), np.array(cam["hi"])
    off = max(np.abs(V.min(0) - lo).max(), np.abs(V.max(0) - hi).max())
    assert off < 3.0 * cam["h"], f"{name}: physical.glb is {off / cam['h']:.1f} h off it_0"

    masks = np.load(os.path.join(s, "boundary_conditions", "resolved.npz"))
    regions = json.load(open(os.path.join(s, "boundary_conditions", "resolved.json")))["regions"]
    cat = {r["key"]: r["category"] for r in regions if "key" in r}
    nm = {r["key"]: r.get("name", r["key"]) for r in regions if "key" in r}
    case = json.load(open(os.path.join(s, "case", "case.json")))
    tractions = {nb["name"]: nb["traction"]
                 for nb in (case.get("boundary_conditions") or {}).get("neumann") or []
                 if nb.get("traction") is not None}
    az, el = VIEW_OVERRIDE.get(name, (VIEW_AZ, VIEW_EL))
    d = view_dir(az, el)

    rgb = np.tile(np.array(A.CLAY), (len(F), 1))
    bcid = np.zeros(len(F), np.int32)
    summary, supports, arrows = [], [], []
    k = 0
    # Dirichlet first, Neumann over it (ar_prep.prep_run's order)
    for want, colour in (("dirichlet", A.DIRICHLET), ("neumann", A.NEUMANN)):
        for key in masks.files:
            if cat.get(key, key.split("_")[0]) != want:
                continue
            m = np.asarray(masks[key], bool)
            assert len(m) == len(F), f"{name}/{key}: mask over {len(m)} faces, surface {len(F)}"
            k += 1
            rgb[m] = colour
            bcid[m] = k
            g = {"key": key, "name": nm.get(key, key), "category": want, "faces": int(m.sum())}
            if want == "dirichlet" and m.any():
                # ar_prep.prep_run's marker rule: a support facing away from the camera
                # (or almost edge-on) gets its outline, a small gap off the faces
                a = mesh.area_faces[m]
                fn = mesh.face_normals[m]
                facing = float((a * np.clip(fn @ d, 0, None)).sum()
                               / (a * np.abs(fn @ d)).sum().clip(1e-30))
                away = float((a[:, None] * fn).sum(0) @ d) < 0
                g["facing_camera"] = facing
                if facing < AP.MARKER_BELOW or away:
                    sv, sf = AP.support_outline(mesh, m, L)
                    supports.append((sv, sf, k))
                    g["outline"] = True
            if want == "neumann":
                t_ = tractions.get(g["name"])
                if t_ is None or np.linalg.norm(t_) == 0:
                    g["arrows"] = 0
                else:
                    got = AP.load_arrows(mesh, m, t_, L, d)
                    arrows += [(v_, f_) for v_, f_, *_ in got]
                    g["arrows"] = len(got)
                    g["traction"] = list(map(float, t_))
            summary.append(g)
    p = lambda kind: os.path.join(out, f"{name}_{kind}.vtk")  # noqa: E731
    AP.write_polydata(p("bc"), V, F, cell_rgb=rgb, cell_ids=bcid)
    for stale in ("arrows", "support"):
        if os.path.exists(p(stale)):
            os.remove(p(stale))
    if arrows:
        off_, vs, fs = 0, [], []
        for v_, f_ in arrows:
            vs.append(v_)
            fs.append(f_ + off_)
            off_ += len(v_)
        fs = np.vstack(fs)
        AP.write_polydata(p("arrows"), np.vstack(vs), fs,
                          cell_rgb=np.tile(np.array(A.LOAD_ARROW), (len(fs), 1)))
    if supports:
        off_, vs, ss, ids = 0, [], [], []
        for sv, sf, rid in supports:
            vs.append(sv)
            ss.append(sf + off_)
            ids.append(np.full(len(sf), rid, np.int32))
            off_ += len(sv)
        ss = np.vstack(ss)
        AP.write_lines(p("support"), np.vstack(vs), ss,
                       np.tile(np.array(A.DIRICHLET), (len(ss), 1)), np.concatenate(ids))
    acc = np.asarray((case.get("loads") or {}).get("acceleration") or [0, 0, 0], float)
    rec = {"name": name, "label": label, "extent": L, "tube_r": AP.MARKER_TUBE * L,
           "camera": {"lo": cam["lo"], "hi": cam["hi"], "up": "y", "azimuth": az,
                      "elevation": el, "view_override": name in VIEW_OVERRIDE, "source": f"{rd}/render/camera.json"},
           "change_png": change_png, "final_iteration": final,
           "change_vtk": os.path.join(rd, "render", f"it_{final}.vtk"),
           "updates": int(res["updates"]), "stop_reason": res["stop_reason"],
           "se_ratio": float(res["se_ratio"]), "vol_ratio": float(res["vol_ratio"]),
           "d_p99_h": float(res["final_d_h"]), "d_max_h": float(res["final_d_max_h"]),
           "h": float(res["h"]), "standin_material": bool(res["standin"]),
           "self_weight_only": not any(g.get("arrows") for g in summary)
           and bool(np.linalg.norm(acc) > 0),
           "regions": summary, "results": f"{rd}/results.json"}
    print(f"{name}: {k} regions, {len(arrows)} arrows, {len(supports)} support outlines; "
          f"{rec['updates']} updates ({rec['stop_reason']}), SE/SE0 {rec['se_ratio']:.3f}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--out", default=os.path.join(HERE, "_cleanup"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    objs = [o for o in OBJECTS if not a.names or o[0] in a.names]
    recs = [prep(n, lab, a.out) for n, lab in objs]
    json.dump({"objects": recs, "view": {"azimuth": VIEW_AZ, "elevation": VIEW_EL}},
              open(os.path.join(HERE, "som_cleanup.json"), "w"), indent=1)
    print("wrote som_cleanup.json")


if __name__ == "__main__":
    main()
