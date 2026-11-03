"""Copy each session's inputs and write its clean-up case (CPU, no GPU).

    python setup.py [object ...]      # conda mechfoundry-warp117; default: all 50

For every object of the batch (common.runs()):

  output/geometry_cleanup/<object>/session/   the session's geometry/, boundary_conditions/,
                                              case/ and materials/parts/materials.json,
                                              copied once and never overwritten (the agent's
                                              run under agentic_batch50_opus stays read-only)
  output/geometry_cleanup/<object>/case.json  the session's case with its paths re-pointed at
                                              the copy, shape_opt = the session's settings +
                                              common.COMMON (validated by ShapeOptSettings),
                                              and, for a per-part session, a uniform stand-in
                                              material
  output/geometry_cleanup/<object>/setup.json what setup decided: the stand-in and its part
                                              volumes, the analysis leaf h

The workflow refuses a shape optimization on a spatial material field (workflow_tool
run_forward: "spatial material fields currently support forward solves only"), so a
"parts" session gets the part-volume-weighted mean E, nu and rho of its parts as a
uniform material.  Part volumes follow the rule the part field itself is built by
(material_parts.field.part_grid: every lattice cell the surface crosses or encloses
takes the part of the nearest surface point), on a 160^3 lattice instead of the
field's 64^3 so thin parts are not over-weighted by the surface shell.
"""

import json
import os
import shutil
import sys

import numpy as np

import common as C

sys.path.insert(0, C.CODE)
from src.agents_wrapper.fem_workflow.models import MaterialParameters, ShapeOptSettings  # noqa: E402

PART_GRID = 160

# Per-object shape_opt overrides, each with the failure that called for it (README).
# jackhammer was tried with {"scale_system_to_unit_diagonal": True}: worse (stalled at
# ||r||/||b|| 2e-4 on the first solve); its fix is a solver backend (run.py --extra).
OVERRIDES = {}


def copy_session(src, dst):
    if os.path.isdir(dst):
        return
    os.makedirs(dst)
    for sub in ("geometry", "boundary_conditions", "case"):
        shutil.copytree(os.path.join(src, sub), os.path.join(dst, sub))
    mj = os.path.join(src, "materials", "parts", "materials.json")
    if os.path.isfile(mj):
        os.makedirs(os.path.join(dst, "materials", "parts"))
        shutil.copy(mj, os.path.join(dst, "materials", "parts", "materials.json"))


def part_standin(sess, case):
    """Part-volume-weighted mean (E, nu, rho) of a per-part session's parts."""
    import trimesh
    from src.material_parts.field import part_grid

    parts = json.load(open(os.path.join(sess, "materials", "parts", "materials.json")))["parts"]
    table = {int(p["id"]): p for p in parts}
    mesh = trimesh.load(case["surface"], force="mesh", process=False)
    labels = np.load(case["segments_npz"])["segment_ids"]
    coords, cell_labels, _, scale = part_grid(mesh, labels, grid_size=PART_GRID)
    cell = scale / PART_GRID
    ids, counts = np.unique(cell_labels, return_counts=True)
    vol = {int(i): float(n) * cell ** 3 for i, n in zip(ids, counts)}
    missing = sorted(set(vol) - set(table))
    if missing:
        raise ValueError(f"no material for parts {missing}")
    w = np.array([vol[i] for i in vol]) / sum(vol.values())
    mean = {k: float(sum(wi * table[i][k] for wi, i in zip(w, vol)))
            for k in ("youngs_modulus", "poisson_ratio", "density")}
    return mean, [{"id": i, "volume_fraction": vol[i] / sum(vol.values()),
                   **{k: table[i][k] for k in ("youngs_modulus", "poisson_ratio", "density")}}
                  for i in sorted(vol)]


def setup(name, run_dir, src):
    od = C.obj_dir(name)
    sess = os.path.join(od, "session")
    copy_session(src, sess)
    case = json.load(open(os.path.join(sess, "case", "case.json")))
    case["surface"] = os.path.join(sess, "geometry", "physical.glb")
    if case.get("segments_npz"):
        case["segments_npz"] = os.path.join(sess, "boundary_conditions", "segments.npz")
    for p in (case["surface"], case.get("segments_npz")):
        assert p is None or os.path.isfile(p), p

    info = {"object": name, "run": os.path.basename(run_dir), "session": src,
            "material_mode": case["material"]["mode"], "standin": False}
    mat = dict(case["material"])
    if mat["mode"] == "parts":
        mean, parts = part_standin(sess, case)
        mat.update(mode="uniform", artifact=None, **mean)
        info.update(standin=True, standin_material=mean, parts=parts)
    case["material"] = MaterialParameters.model_validate(mat).model_dump(mode="json")

    so = dict(case["shape_opt"])
    so.update(C.COMMON)
    so.update(OVERRIDES.get(name, {}))
    so = ShapeOptSettings.model_validate(so).model_dump(mode="json")
    case["shape_opt"] = so
    case["output"] = os.path.join(od, "solve")
    # the analysis leaf, as run_forward derives it
    h = float(case["h_target"]) * 2.0 ** (int(case["octree_depth"]) - so["remesh_octree_depth"])
    info.update(h=h, remesh_octree_depth=so["remesh_octree_depth"],
                overrides=OVERRIDES.get(name, {}),
                physical_extent=case["geometry_transform"]["physical_extent"])
    with open(C.case_path(name), "w") as f:
        json.dump(case, f, indent=2)
    with open(os.path.join(od, "setup.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"{name}: depth {so['remesh_octree_depth']}, h {h:.3e} m, material "
          f"{info['material_mode']}"
          + (" -> uniform stand-in E %.3e nu %.3f rho %.1f" % (
              mat["youngs_modulus"], mat["poisson_ratio"], mat["density"])
             if info["standin"] else ""))


def main():
    allruns = C.runs()
    names = sys.argv[1:] or list(allruns)
    for n in names:
        setup(n, *allruns[n])


if __name__ == "__main__":
    main()
