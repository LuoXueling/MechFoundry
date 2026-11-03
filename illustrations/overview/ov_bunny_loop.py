"""Stage 2 (GPU, env mechfoundry-warp117): the overview's whole loop on the bunny.

    python ov_bunny_loop.py          # in overview/, after prep_bunny_sdf and ov_assets
    OV_GRAD_FD=1 python ov_bunny_loop.py   # + a directional finite-difference check

One problem, computed once, so every panel downstream of the segmentation shows
the same thing:

1. ``nitsche/dn_prep.py`` (runpy) gives the octree (its asserted 45,082 nodes),
   the phase field, the boundary band and the trimming, on the unmirrored
   bunny.stl — the same mesh ``diffused_nitsche`` shows.
2. THE LOADED EARS COME FROM THE SEGMENTATION (the user's rule): the PartField
   parts from ``assets/seg``, at the smallest k that cuts both ears off the rest,
   through the workflow's own ``SurfaceSegmentSelector`` — exactly what a
   reviewed segmentation becomes in ``resolve_bcs``.  (dn_prep's flood-fill ears
   are only the geometric reference that says which parts ARE the ears.)  The
   clamped base is dn_prep's own fixed bottom, the lowest 5 % of the height, as
   ``diffused_nitsche`` has it.
3. ``dn_prep.classify`` intersects those selectors with the boundary band — the
   solver's own ``_classify_bc_elements`` — and the tags are carried onto the
   trimmed mesh.
4. The production ``ShapeOptTrimmed`` on that octree (``remeshing=False``) runs
   one taped forward solve and its adjoint twice: in ``sdf`` mode for dJ/dd, and
   in hidden-phase-field mode (``design_variable="phase_field"``) for dJ/dphi —
   the optional path.  Both gradients live on the untrimmed octree; nothing is
   drawn on the surface.  The forward solution is the figure's Nitsche solve.

The objective is compliance alone (``shape_opt_w_vol = 0``); J = SE / SE_0.
Frozen nodes (``_e_fixed_bool``: every node of a boundary element touching a BC
patch) are never updated, so they are written as masked.

Writes (render_ov_panels.py / ov_surfaces.py read them; no number is printed on
the figure):
  ov_grad_full.vtk   untrimmed octree: nodal sdf, gd / gphi (log |g|/max, -1 masked)
  ov_trim_bc.vtk     trimmed octree: nodal phi, cell tag (0 in, 1 band, 2 N, 3 D)
  ov_bc_{rest,ears,base}.stl   the surface split by the BC sets (bunny.stl frame)
  ov_bc_faces.npz    the two BC facet masks, in bunny.stl's face order
  ov_loop_solve.vtu  the surface with its displacement, in the DISPLAY frame —
                     ov_bunny_solve.vtu's format, so diffused_nitsche's own
                     renderer (render_ov_solve.py) draws it as its panel (f)
  ov_loop.json       k, BC segment ids, gradient cut planes (display frame), stats
"""

import json
import os
import runpy
import shutil
import sys

import numpy as np
import torch
import trimesh
import warp as wp

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "warp_wrapper"))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from core.bc_surface_selectors import SurfaceSegmentSelector  # noqa: E402
from core.remeshing import octree_hex_interpolate  # noqa: E402
from core.shape_opt import ShapeOptTrimmed  # noqa: E402
from core.vtu_writer import VTUSurfaceWriter  # noqa: E402

import ov_manifest  # noqa: E402
import ov_style as S  # noqa: E402

DECADES = 3.0                    # |g| over three decades: the band reads at panel size
# The ears' downward traction.  2.5x ov_bunny_solve's -0.020 (the user asked for
# a larger ear deflection); the figure draws the solve at true scale, no warp factor.
TRACTION_Z = float(os.environ.get("OV_TRACTION_Z", -0.050))
LOAD_STEPS = int(os.environ.get("OV_LOAD_STEPS", 4))
PART_KS = (5, 6, 7, 8, 9, 10)
MIRROR = np.array(S.MIRROR)
FD = os.environ.get("OV_GRAD_FD") == "1"

dn = runpy.run_path(os.path.join(HERE, "..", "nitsche", "dn_prep.py"))
bunny = dn["bunny"]
write_hexes = dn["write_hexes"]
N_FULL, E_FULL = int(dn["nodes"].shape[0]), int(dn["elements"].shape[0])
F = np.asarray(bunny.faces)
ear_ref = dn["ear_mask"].cpu().numpy().astype(bool)
base_ref = dn["bottom_mask"].cpu().numpy().astype(bool)
out = {}

# ---------------------------------------------------------------------------
# 1. the segmentation, and the BC parts it gives
# ---------------------------------------------------------------------------
tm = trimesh.Trimesh(np.asarray(bunny.vertices), F, process=False)
adj = tm.face_adjacency
ear_cc = trimesh.graph.connected_components(adj[ear_ref[adj].all(1)],
                                            nodes=np.where(ear_ref)[0], min_len=100)
ear_cc = sorted(ear_cc, key=len, reverse=True)[:2]
assert len(ear_cc) == 2, "expected two ears in dn_prep's ear selection"


def ears_separate(labels):
    """Each ear is mostly one part, and that part is mostly ear."""
    rows = []
    for comp in ear_cc:
        lab, cnt = np.unique(labels[comp], return_counts=True)
        dom = int(lab[np.argmax(cnt)])
        rows.append((dom, cnt.max() / len(comp), float(ear_ref[labels == dom].mean())))
    return all(c >= 0.75 and p >= 0.6 for _, c, p in rows), rows


labels, k_used = None, None
for k in PART_KS:
    rel = f"seg/pf_k{k}.npz"
    if not ov_manifest.has(rel):
        continue
    lab = np.load(ov_manifest.asset(rel))["segment_ids"].astype(np.int64)
    assert len(lab) == len(F), f"{rel}: {len(lab)} labels for {len(F)} faces"
    ok, rows = ears_separate(lab)
    print(f"PartField k={k}: ears (part, cover, purity) "
          f"{[(d, round(c, 2), round(p, 2)) for d, c, p in rows]} -> "
          f"{'separate' if ok else 'merged'}")
    if ok:
        labels, k_used = lab, k
        break
assert labels is not None, "no PartField k separates both ears from the rest"
ear_ids = sorted(int(l) for l in np.unique(labels) if ear_ref[labels == l].mean() >= 0.5)
print(f"k={k_used}: Neumann (ears) parts {ear_ids}; Dirichlet: dn_prep's lowest 5 %")
out.update(partfield_k=k_used, neumann_parts=ear_ids, dirichlet="dn_prep bottom 5 %")

# ---------------------------------------------------------------------------
# 2. the selectors (the workflow's SurfaceSegmentSelector), resolved as dn_prep
#    resolves its own, and classified against the band by dn_prep's classify
# ---------------------------------------------------------------------------
ear_sel = SurfaceSegmentSelector(ear_ids, labels)
ear_sel.resolve(dn["verts"], dn["faces"], eps_expansion=dn["eps_tol"])
base_sel = dn["bottom_sel"]                      # resolved by dn_prep already
ear_bc, _ = dn["classify"](ear_sel)
base_bc = dn["bottom_bc"]
assert int(ear_bc.sum()) > 0 and int(base_bc.sum()) > 0, "empty BC region"
assert not bool((ear_bc & base_bc).any()), "Dirichlet and Neumann overlap"
band_idx = dn["band_idx"]
tag_full = torch.zeros(E_FULL, dtype=torch.float32, device=band_idx.device)
tag_full[band_idx] = 1.0
tag_full[band_idx[ear_bc]] = 2.0
tag_full[band_idx[base_bc]] = 3.0
out.update(neumann_band_elements=int(ear_bc.sum()), dirichlet_band_elements=int(base_bc.sum()))

keep, trim = dn["keep_elem"], dn["trim_mesh"]
full_c = dn["nodes"][dn["elements"][keep]].mean(1)
trim_c = trim["nodes"].to(torch.float32)[trim["elements"].long()].mean(1).to(full_c.device)
assert torch.allclose(full_c, trim_c, atol=1e-5), \
    "trimmed elements are not full[keep_elem] in order — tags would land on wrong cells"
trim_phi = (0.5 * (1.0 - torch.tanh(trim["sdf"] / dn["eps"]))).clamp(0.0, 1.0)
write_hexes("ov_trim_bc.vtk", trim["nodes"].to(torch.float32), trim["elements"].long(),
            "trimmed octree, phi + region tags", point_scalars={"phi": trim_phi},
            cell_scalar=tag_full[keep])

ear_f = np.isin(labels, ear_ids)
base_f = dn["bottom_mask"].cpu().numpy().astype(bool)
assert not (ear_f & base_f).any(), "a facet is both loaded and clamped"
np.savez("ov_bc_faces.npz", ears=ear_f, base=base_f)      # face order == bunny.stl
for name, mask in (("ears", ear_f), ("base", base_f), ("rest", ~(ear_f | base_f))):
    trimesh.Trimesh(np.asarray(bunny.vertices), F[mask], process=False).export(f"ov_bc_{name}.stl")
print(f"BC facets: ears {int(ear_f.sum())}, base {int(base_f.sum())}")


# ---------------------------------------------------------------------------
# 3. the loop: one taped forward solve + adjoint, in sdf and in phase-field mode
# ---------------------------------------------------------------------------
def owned_octree(sdf=None):
    """The solver mutates what it is given; hand each run its own copy."""
    o = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in dn["octree"].items()}
    if sdf is not None:
        o["sdf"] = sdf.clone()
    o["domain_length"] = dn["domain_length"]
    return o


class Capture(ShapeOptTrimmed):
    """The production optimiser, instrumented to record (never to alter)."""

    snap = None

    def _post_tape_backward(self):
        super()._post_tape_backward()
        inner = self._inner
        u = wp.to_torch(inner.u.dof_values).detach()
        u = u.reshape(inner._prev_octree_nodes.shape[0], -1)
        if getattr(self, "_last_e_grad", None) is None:
            raise RuntimeError("ShapeOptTrimmed set no _last_e_grad")
        phi_g = getattr(self, "_last_e_phi_grad", None)
        self.snap = {
            "g": self._last_e_grad.detach().double().cpu().clone(),
            "gphi": None if phi_g is None else phi_g.detach().double().cpu().clone(),
            "nodes": self._prev_octree_nodes.detach().double().cpu().clone(),
            "elements": self._prev_octree_elements.detach().long().cpu().clone(),
            "fixed": self._e_fixed_bool.detach().bool().cpu().clone(),
            "inner_elements": int(inner._prev_octree_elements.shape[0]),
            "inner_nodes": inner._prev_octree_nodes.detach().double().clone(),
            "inner_elems": inner._prev_octree_elements.detach().long().clone(),
            "u": u.double().clone(),
            "se_raw": float(self._last_se_raw),
        }


def params(root, mode="sdf", sdf=None, precise=False):
    p = {
        "project_name": "ov_bunny_loop", "output_root": root,
        "octree_mesh": owned_octree(sdf), "phase_field_epsilon": dn["eps"],
        "remeshing": False, "reinit_sdf": False, "design_variable": mode,
        "optimizer": "gd", "gd_lr": 1e-6, "max_optimize_iters": 1,
        "output_per_optimize_step": 0, "output_surface_vtu": False,
        "output_surface_glb": False, "output_surface_glb_deformed": False,
        "shape_opt_w_vol": 0.0,
        # a load this large needs continuation: run_forward's recipe — a plain
        # forward() over LOAD_STEPS first, then the taped iteration from there
        "load_steps": LOAD_STEPS,
        # == ov_bunny_solve's solver settings
        "relative_tol": 1e-3, "use_mixed_precision": False, "tangent_f64": False,
        "cg_backend_assembled": "jaxamg", "async_io": False,
        "dirichlet_bcs_dict": {base_sel: 0.0},
        "neumann_bcs_dict": {ear_sel: (0.0, 0.0, TRACTION_Z)},
    }
    if precise:
        p.update(relative_tol=1e-10, absolute_tol=1e-16, max_iter=40,
                 use_mixed_precision=True, tangent_f64=True)
    return p


def run(mode="sdf", sdf=None, precise=False):
    root = os.path.join(os.getcwd(), f"_ov_bunny_loop_{mode}")
    shutil.rmtree(root, ignore_errors=True)
    opt = Capture("configs.config_shape_opt_cantilever",
                  override_params=params(root, mode, sdf, precise))
    if LOAD_STEPS > 1:
        opt.forward()
    opt.optimize()
    shutil.rmtree(root, ignore_errors=True)
    assert opt.snap is not None, "the backward hook never ran"
    return opt.snap


snap = run("sdf")
snap_pf = run("phase_field")
assert snap_pf["gphi"] is not None, "phase-field mode set no _last_e_phi_grad"

nodes, elements = snap["nodes"], snap["elements"]
assert (len(nodes), len(elements)) == (N_FULL, E_FULL), (
    f"gradient mesh {len(nodes)}N/{len(elements)}E is not dn_prep's {N_FULL}N/{E_FULL}E")
assert np.allclose(nodes.numpy(), dn["nodes"].double().cpu().numpy(), atol=1e-6)
assert snap["inner_elements"] == int(trim["elements"].shape[0]), (
    f"optimiser trimmed to {snap['inner_elements']} elements, dn_prep to "
    f"{int(trim['elements'].shape[0])}")
fixed = snap["fixed"].numpy().reshape(-1)
free = ~fixed
assert fixed.any() and free.any(), "frozen mask degenerate"


def log_masked(g):
    g = np.asarray(g).reshape(-1)
    assert np.isfinite(g).all(), "non-finite gradient"
    gf = np.where(free, g, 0.0)
    gmax = float(np.abs(gf).max())
    assert gmax > 0.0, "the gradient is zero on every free node"
    t = (np.log10(np.maximum(np.abs(gf) / gmax, 10.0 ** -DECADES)) + DECADES) / DECADES
    t[fixed] = -1.0
    return t


gd, gphi = log_masked(snap["g"].numpy()), log_masked(snap_pf["gphi"].numpy())


def cut_plane(t):
    """The finest level's cell mid-plane (clean quad cuts, §5 clip gotcha) whose
    neighbouring nodes carry the most of this gradient.  The paper's cut
    (+0.009) runs through the bunny's middle; the sensitivity sits where the
    loaded ears join the head, off that plane."""
    yn = nodes.numpy()[:, 1]
    h = float(dn["h_min"])
    planes = (np.arange(-40, 40) + 0.5) * h
    score = [float(np.sum(np.clip(t[np.abs(yn - yc) <= 0.5 * h + 1e-6], 0, None) ** 2))
             for yc in planes]
    return float(planes[int(np.argmax(score))])


# cube panels are drawn in bunny.stl's own frame, as diffused_nitsche draws them
out["cut_y_gd"] = cut_plane(gd)
out["cut_y_gphi"] = cut_plane(gphi)
print(f"cut planes: dJ/dd {out['cut_y_gd']:+.5f}, dJ/dphi {out['cut_y_gphi']:+.5f}")
tensor = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
write_hexes("ov_grad_full.vtk", tensor(nodes), elements, "untrimmed octree, d + gradients",
            point_scalars={"sdf": tensor(dn["octree"]["sdf"].cpu()), "gd": tensor(gd),
                           "gphi": tensor(gphi)})

# the forward solution on the surface, written as ov_bunny_solve writes its own,
# so render_ov_solve.py (diffused_nitsche's panel (f)) draws it unchanged
surf_v = torch.as_tensor(np.asarray(bunny.vertices), dtype=torch.float64, device="cuda")
su = octree_hex_interpolate(snap["inner_nodes"], snap["inner_elems"], snap["u"],
                            surf_v).cpu().numpy()
umag = np.linalg.norm(su, axis=1)
# in the display frame (mirrored in y, as every overview bunny is), so the panel
# sees the face from the BC panel's camera
_vtu_dir = os.path.abspath("_ov_loop_vtu")
VTUSurfaceWriter(_vtu_dir, grid_shape=None, write_vtu=True, write_glb=False,
                 write_glb_deformed=False, async_io=False).save_from_mesh(
    0, np.asarray(bunny.vertices) * MIRROR, F,
    {"displacement": su * MIRROR, "displacement_magnitude": umag})
os.replace(os.path.join(_vtu_dir, "vtu_surface", "surface_t0000.vtu"), "ov_loop_solve.vtu")
shutil.rmtree(_vtu_dir, ignore_errors=True)
print("wrote ov_loop_solve.vtu")

out.update(traction_z=TRACTION_Z, nodes=N_FULL, elements=E_FULL, inner_elements=snap["inner_elements"],
           frozen_nodes=int(fixed.sum()), se_raw=snap["se_raw"], umax_surface=float(umag.max()),
           w_vol=0.0, decades=DECADES)


# ---------------------------------------------------------------------------
# optional: does the taped adjoint agree with finite differences?  (float64;
# see the note — the ratio carries a fixed discrete step, fitted out as c/h)
# ---------------------------------------------------------------------------
if FD:
    base = run("sdf", precise=True)
    g64 = base["g"].numpy().reshape(-1)
    x = dn["nodes"].double().cpu().numpy()
    centre = x[np.argmax(np.abs(np.where(free, g64, 0.0)))]
    v = np.exp(-(np.linalg.norm(x - centre, axis=1) / (4.0 * float(dn["h_min"]))) ** 2) * free
    sdf0 = dn["octree"]["sdf"].double().cpu()
    predicted = float(g64 @ v) * base["se_raw"]            # dSE = SE_0 * dJ
    rows = []
    for h in (3e-3, 1e-3, 3e-4):
        vp = torch.as_tensor(v * h, dtype=torch.float64)
        dev = dn["octree"]["sdf"].device
        plus = run("sdf", (sdf0 + vp).float().to(dev), precise=True)["se_raw"]
        minus = run("sdf", (sdf0 - vp).float().to(dev), precise=True)["se_raw"]
        rows.append((h, (plus - minus) / (2.0 * h)))
        print(f"FD h={h:g}: {rows[-1][1]:.6e}  adjoint {predicted:.6e}")
    hs, fds = np.array([r[0] for r in rows]), np.array([r[1] for r in rows])
    (a_fit, c_fit), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(hs), 1.0 / hs]),
                                         fds, rcond=None)
    print(f"FD = a + c/h: a/adjoint = {a_fit / predicted:.5f}, step c = {c_fit:.3e}")
    assert abs(a_fit / predicted - 1.0) < 0.02, "adjoint disagrees with the smooth FD slope"
    out.update(fd_fit_over_adjoint=float(a_fit / predicted), fd_step_c=float(c_fit))

json.dump(out, open("ov_loop.json", "w"), indent=1)
print("wrote ov_loop.json")
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)          # Warp's teardown segfaults in this env (see dn_prep)
