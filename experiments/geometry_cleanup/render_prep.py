"""Write the before/after surfaces of one or more clean-up runs for render.py (GPU, warp).

    python render_prep.py [--tag TAG] [--its 0,5,10] [object ...]   # default: every run

For each run (<OUT>/<object>[/<tag>]/, read from its surfaces/it_<k>.npz) writes into
<run>/render/:

  it_<k>.vtk     the marching-cubes surface the run solved at iteration k (physical
                 metres), with the point scalar "change_h": the SIGNED distance of
                 each vertex to the starting surface (iteration 0) in leaves h,
                 positive outside the start (material added), negative inside
                 (material removed)
  camera.json    one camera for all of the run's frames: +y up, the starting
                 surface's bounds

Default frames: 0 and the final shape (results.json: final_surface).  Legacy-VTK POLYDATA (binary) because pvbatch reads it
without conda's libraries (headless-paraview note: STL has no point data; PLY from
trimesh loads as points).
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

import common as C

import warp as wp  # noqa: E402

wp.config.quiet = True


@wp.kernel
def _signed(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_d: float,
            out: wp.array(dtype=float)):
    i = wp.tid()
    q = wp.mesh_query_point(mesh, pts[i], max_d)
    if q.result:
        d = wp.length(pts[i] - wp.mesh_eval_position(mesh, q.face, q.u, q.v))
        out[i] = d * q.sign
    else:
        out[i] = max_d


def signed_change(v0, f0, v):
    dev = "cuda:0" if wp.is_cuda_available() else "cpu"
    m = wp.Mesh(points=wp.array(v0, dtype=wp.vec3, device=dev),
                indices=wp.array(f0.astype(np.int32).ravel(), dtype=int, device=dev))
    pts = wp.array(v.astype(np.float32), dtype=wp.vec3, device=dev)
    out = wp.zeros(len(v), dtype=float, device=dev)
    diag = float(np.linalg.norm(v0.max(0) - v0.min(0)))
    wp.launch(_signed, dim=len(v), inputs=[m.id, pts, 2 * diag, out], device=dev)
    return out.numpy()


def write_vtk(path, v, f, scal=None):
    with open(path, "wb") as fh:
        fh.write(b"# vtk DataFile Version 3.0\ncleanup surface\nBINARY\nDATASET POLYDATA\n")
        fh.write(f"POINTS {len(v)} float\n".encode())
        fh.write(np.asarray(v, ">f4").tobytes())
        cells = np.hstack([np.full((len(f), 1), 3), f]).astype(">i4")
        fh.write(f"\nPOLYGONS {len(f)} {cells.size}\n".encode())
        fh.write(cells.tobytes())
        if scal is not None:
            fh.write(f"\nPOINT_DATA {len(v)}\nSCALARS change_h float 1\nLOOKUP_TABLE default\n".encode())
            fh.write(np.asarray(scal, ">f4").tobytes())
        fh.write(b"\n")


def prep(name, tag="", its=None):
    rd = os.path.join(C.obj_dir(name), tag) if tag else C.obj_dir(name)
    files = glob.glob(os.path.join(rd, "surfaces", "it_*.npz"))
    if not files:
        print(f"{name}: no surfaces"); return
    have = sorted(int(re.search(r"it_(\d+)", p).group(1)) for p in files)
    last = have[-1]
    rp = os.path.join(rd, "results.json")
    if os.path.isfile(rp) and json.load(open(rp)).get("final_surface"):
        # the final shape (a restored one when the last solved shape was rejected)
        last = int(re.search(r"it_(\d+)", json.load(open(rp))["final_surface"]).group(1))
    its = [i for i in (its or [0, last]) if i in have]
    h = json.load(open(os.path.join(C.obj_dir(name), "setup.json")))["h"]
    s0 = np.load(os.path.join(rd, "surfaces", "it_0.npz"))
    v0, f0 = s0["v"], s0["f"]
    out = os.path.join(rd, "render")
    os.makedirs(out, exist_ok=True)
    for k in its:
        s = np.load(os.path.join(rd, "surfaces", f"it_{k}.npz"))
        ch = np.zeros(len(s["v"])) if k == 0 else signed_change(v0, f0, s["v"]) / h
        write_vtk(os.path.join(out, f"it_{k}.vtk"), s["v"], s["f"], ch)
    json.dump({"up": "y",   # generators are y-up; every session's gravity is along -y
               "lo": v0.min(0).tolist(), "hi": v0.max(0).tolist(), "its": its, "h": h},
              open(os.path.join(out, "camera.json"), "w"))
    print(f"{name}{'/' + tag if tag else ''}: wrote frames {its}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--tag", default="")
    ap.add_argument("--its", default="")
    a = ap.parse_args()
    its = [int(i) for i in a.its.split(",")] if a.its else None
    for n in a.names or sorted(C.runs()):
        prep(n, a.tag, its)
