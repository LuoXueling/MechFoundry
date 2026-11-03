"""Stage 2 (CPU, env mechfoundry-warp117): the cached assets, made renderable.

    python ov_surfaces.py            # in overview/, after ov_bunny_loop.py

Reads ``assets/`` (every file checked against ``assets/MANIFEST.json``),
``bunny.stl`` and ``ov_loop.json`` (the segmentation and BC parts the loop
used), and writes plain legacy-VTK polydata in the DISPLAY frame — bunny.stl
mirrored in y as pfbunny mirrors it, the frame every surface panel is drawn in:

  ov_s_painted.vtk            the display bunny painted by Hunyuan-Paint (also the
                              Hunyuan3D lane's output: one mesh for every source)
  ov_s_seg_pf.vtk             PartField (the loop's k), one pastel tint per part
  ov_s_bc.vtk        the surface, the loop's BC sets as cell colours on the clay
  ov_s_materials.vtk          per-part Young's modulus (viridis on log10 E)
  ov_s_cad.vtk / _edges.vtk   the CAD bunny: clay, and its B-rep face edges
  ov_surfaces.json            the choices made here, for the record

PART MATERIALS are the user's brief (2026-09-29): "optionally heterogeneous",
shown as one panel in which only the ears are softer.  The loop's ear parts get
"rubber, hard", every other part "ABS" — the middles of the workflow's own
``material_values.md`` rows; densities are handbook values, as that file says.
(The solve itself stays homogeneous; the caption says so.)
"""

import hashlib
import itertools
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import ov_manifest
import ov_style as S
from ov_manifest import asset, has

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = ov_manifest.ASSETS
TO_HUNYUAN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # == ov_assets
MIRROR = np.array(S.MIRROR)
# material_values.md: "ABS" 1.6e9-2.6e9 / 0.33-0.37; "rubber, soft" 1.0e6-1.0e7 /
# 0.48-0.49.  E at the geometric middle of the band, nu at its middle, density a
# handbook value (the table has none).
ABS = {"material": "ABS", "youngs_modulus": float(np.sqrt(1.6e9 * 2.6e9)),
       "poisson_ratio": 0.35, "density": 1050.0}
RUBBER = {"material": "hard rubber", "youngs_modulus": float(np.sqrt(5.0e7 * 2.0e8)),
          "poisson_ratio": 0.48, "density": 1150.0}
SEGMENT_TINTS = S.SEGMENT_TINTS


def write_polydata(path, verts, faces=None, lines=None, point_rgb=None, cell_rgb=None):
    v = np.asarray(verts, dtype=np.float32)
    with open(path, "w") as out:
        out.write(f"# vtk DataFile Version 3.0\n{os.path.basename(path)}\nASCII\n"
                  "DATASET POLYDATA\n")
        out.write(f"POINTS {len(v)} float\n")
        np.savetxt(out, v, fmt="%.7g")
        n_cells = 0
        if faces is not None:
            f = np.asarray(faces, dtype=np.int64)
            n_cells = len(f)
            out.write(f"POLYGONS {len(f)} {4 * len(f)}\n")
            np.savetxt(out, np.column_stack([np.full(len(f), 3), f]), fmt="%d")
        if lines is not None:
            ln = np.asarray(lines, dtype=np.int64)
            n_cells = len(ln)
            out.write(f"LINES {len(ln)} {3 * len(ln)}\n")
            np.savetxt(out, np.column_stack([np.full(len(ln), 2), ln]), fmt="%d")
        if point_rgb is not None:
            out.write(f"POINT_DATA {len(v)}\nCOLOR_SCALARS rgb 3\n")
            np.savetxt(out, np.clip(np.asarray(point_rgb, float), 0, 1), fmt="%.4f")
        if cell_rgb is not None:
            out.write(f"CELL_DATA {n_cells}\nCOLOR_SCALARS rgb 3\n")
            np.savetxt(out, np.clip(np.asarray(cell_rgb, float), 0, 1), fmt="%.4f")
    print(f"wrote {path}")


def cube_rotations():
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = s
            if np.linalg.det(m) > 0:
                out.append(m)
    assert len(out) == 24
    return out


def align(mesh, target, label, n=4000, seed=0):
    """Best uniform-scale rigid fit of ``mesh`` onto ``target`` (no reflection)."""
    src_pts = mesh.sample(n, seed=np.random.default_rng(seed))
    tgt_pts = target.sample(20000, seed=np.random.default_rng(seed + 1))
    c_src, c_tgt = mesh.bounds.mean(0), target.bounds.mean(0)
    s0 = np.linalg.norm(np.ptp(target.vertices, 0)) / np.linalg.norm(np.ptp(mesh.vertices, 0))
    best = None
    for rot in cube_rotations():
        m0 = np.eye(4)
        m0[:3, :3] = s0 * rot
        m0[:3, 3] = c_tgt - s0 * rot @ c_src
        matrix, _, cost = trimesh.registration.icp(
            src_pts, tgt_pts, initial=m0, threshold=1e-7, max_iterations=80,
            reflection=False, translation=True, scale=True)
        if best is None or cost < best[1]:
            best = (matrix, cost)
    matrix = best[0]
    lin = matrix[:3, :3]
    scale = np.cbrt(np.linalg.det(lin))
    assert scale > 0 and np.allclose(lin / scale @ (lin / scale).T, np.eye(3), atol=1e-6), \
        f"{label}: fit is not a rotation + uniform scale"
    d, _ = cKDTree(tgt_pts).query(trimesh.transform_points(src_pts, matrix))
    diag = np.linalg.norm(np.ptp(target.vertices, 0))
    print(f"{label}: ICP mean distance {d.mean() / diag:.3%} of the bunny diagonal")
    assert d.mean() / diag < 0.05, f"{label}: no good rigid fit to the bunny"
    return trimesh.transform_points(mesh.vertices, matrix)


def colours_of(mesh):
    vc = np.asarray(mesh.visual.vertex_colors, dtype=float)[:, :3] / 255.0
    assert vc.shape[0] == len(mesh.vertices)
    return vc


def seams(tm, labels):
    """Mesh edges whose two faces carry different labels, as line segments."""
    adj = tm.face_adjacency
    return tm.face_adjacency_edges[labels[adj[:, 0]] != labels[adj[:, 1]]]


def colour_labels(tm, labels, n_tints):
    """Colour index per label, largest part first (pf_prep's order).  Every part
    gets its own colour when there are enough; otherwise touching parts never
    share one."""
    ids, counts = np.unique(labels, return_counts=True)
    if len(ids) <= n_tints:
        return {int(l): i for i, l in enumerate(ids[np.argsort(-counts)])}
    adj = tm.face_adjacency
    la, lb = labels[adj[:, 0]], labels[adj[:, 1]]
    touching = {int(i): set() for i in ids}
    for a_, b_ in set(zip(la[la != lb].tolist(), lb[la != lb].tolist())):
        touching[a_].add(b_)
        touching[b_].add(a_)
    colour = {}
    for lab in ids[np.argsort(-counts)]:
        used = {colour[n] for n in touching[int(lab)] if n in colour}
        free_ = [c for c in range(n_tints) if c not in used]
        assert free_, f"label {lab} touches every tint"
        colour[int(lab)] = free_[0]
    return colour


# --------------------------------------------------------------------------
raw = trimesh.load(os.path.join(HERE, "bunny.stl"))
V = np.asarray(raw.vertices) * MIRROR           # == ov_assets.bunny()
F = np.asarray(raw.faces)
bunny = trimesh.Trimesh(V, F, process=False)
assert bunny.volume > 0, "the display bunny must be wound outward"
print(f"display bunny: {len(V)} V / {len(F)} F")
record = {}

loop = json.load(open("ov_loop.json"))

# -- TripoSG and Hunyuan3D show the display bunny itself (see ov_assets) --------
# -- Hunyuan-Paint on the display bunny: same vertices, rotated back -----------
p = trimesh.load(asset("paint/painted_yup.glb"), force="mesh", process=False)
back = np.asarray(p.vertices) @ TO_HUNYUAN        # inverse of v @ R.T is v @ R
assert back.shape == V.shape and np.abs(back - V).max() < 1e-4, \
    "painted surface is not the display bunny"
assert np.array_equal(np.asarray(p.faces), F), "painted surface re-ordered its faces"
paint_rgb = colours_of(p)
assert paint_rgb.std(0).max() > 0.02, "the painted surface is one flat colour"
write_polydata("ov_s_painted.vtk", V, F, point_rgb=paint_rgb)

# -- PartField segments, one pastel tint each: touching parts get the most
#    different pair (ov_palette_check.segment_tints, shared with pfbunny)
import ov_palette_check as G  # noqa: E402

tints = np.array([S._hex(c) for c in SEGMENT_TINTS])
k = int(loop["partfield_k"])
pf_labels = np.load(asset(f"seg/pf_k{k}.npz"))["segment_ids"].astype(np.int64)
assert len(pf_labels) == len(F)
colour, _, min_de = G.segment_tints(pf_labels, bunny.face_adjacency, SEGMENT_TINTS)
ids = sorted(colour)
record["segment_tint_min_dE_touching"] = round(float(min_de), 1)
write_polydata("ov_s_seg_pf.vtk", V, F, cell_rgb=tints[np.array([colour[int(l)] for l in pf_labels])])

# -- BC facets: the loop's sets (face order == bunny.stl == the display bunny) ----
masks = np.load("ov_bc_faces.npz")
ear_f, base_f = masks["ears"], masks["base"]
# one surface, the sets as cell colours on the TripoSG panel's clay: split into
# three meshes, each part got its own normals and the seams sparkled
bc_rgb = np.tile(np.array(S.CLAY), (len(F), 1))
bc_rgb[ear_f], bc_rgb[base_f] = S.NEUMANN, S.DIRICHLET
write_polydata("ov_s_bc.vtk", V, F, cell_rgb=bc_rgb)

# -- materials: one field, only the ears softer --------------------------------------
parts = [{"id": l, **(RUBBER if l in loop["neumann_parts"] else ABS)} for l in ids]
record["parts"] = parts
E = {q["id"]: q["youngs_modulus"] for q in parts}
logE = np.log10(np.array([E[int(l)] for l in pf_labels]))
lo, hi = float(logE.min()), float(logE.max())
t = ((logE - lo) / max(hi - lo, 1e-9))[:, None]
write_polydata("ov_s_materials.vtk", V, F, cell_rgb=S.e_rgb(t[:, 0]))   # the shared "steel" E map
print("parts:", [(q["id"], q["material"]) for q in parts])

# -- CAD --------------------------------------------------------------------------
if has("cad/cad_bunny.npz"):
    cad = np.load(asset("cad/cad_bunny.npz"))
    cm = trimesh.Trimesh(cad["vertices"], cad["faces"], process=False)
    moved = align(cm, bunny, "CAD bunny")
    face_ids = cad["face_ids"].astype(np.int64)
    # the TripoSG panel's clay, the B-rep face edges drawn in ink (the CAD look
    # the user chose from a Gemini sketch: one surface, its patches outlined)
    write_polydata("ov_s_cad.vtk", moved, cm.faces,
                   cell_rgb=np.tile(np.array(S.CLAY), (len(cm.faces), 1)))
    write_polydata("ov_s_cad_edges.vtk", moved,
                   lines=seams(trimesh.Trimesh(moved, cm.faces, process=False), face_ids))

json.dump(record, open("ov_surfaces.json", "w"), indent=1)
print("wrote ov_surfaces.json")
