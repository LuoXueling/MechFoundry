"""Prep (conda GPU): run the REAL ShapeOptTrimmed pipeline on a cylinder design
domain and capture every forward/backward intermediate for the shape-opt loop.

Example (NOT the cantilever): a solid cylinder — the "blank" for a table / stool.
A downward Neumann load presses the whole top surface; four small Dirichlet feet
on the bottom are fixed; the optimizer targets 50% volume and maximises stiffness
(minimises compliance, shape_opt_w_vol=5), so it carves the cylinder into a top
plate on four legs.  BC regions are selected with production primitives
(SphereSelector for the feet, HighestSelector for the top load) like the bunny
figure, not coordinate lambdas.

The figure's "initial geometry" is not the raw cylinder but the design after a
few optimizer iterations, once the compliance has started to decrease — set via
SO_CAPTURE_ITER.  That geometry is what the forward/backward panels illustrate.

At CAPTURE_ITER, a ShapeOptCapture subclass dumps (to this folder):
  so_design.vtk      octree hex mesh + nodal SDF (panel a: current design)
  so_solve.vtk       octree + |u| displacement magnitude + strain energy (panel b)
  so_grad.vtk        octree + dJ/dSDF sensitivity  (panel d)
  so_updated.vtk     octree + SDF after the GD step, pre-reinit (panel e)
  so_bg.vtk          dense background grid SDF  (panel f, reinit input)
  so_mc_raw.stl      raw marching-cubes surface  (panel f, pre-cleanup)
  so_mc_clean.stl    after drop-non-watertight + close-open-loops (panel f)
  so_stitched.stl    after BC-region restore + zip-stitch (panel g)
  so_new.vtk         new octree hex mesh + nodal SDF (panel h)
Plus so_stats.txt (objective, volume, |grad|, counts, GPU) for the captions.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir("/home/xl2254/implicit_optimize/src/warp_wrapper")
sys.path.insert(0, ".")

import numpy as np
import torch
import trimesh
from skimage import measure

import core.shape_opt as so_mod
from core.shape_opt import ShapeOptTrimmed

CAPTURE_ITER = int(os.environ.get("SO_CAPTURE_ITER", "40"))  # compliance turn (~50% vol)
N_ITERS = int(os.environ.get("SO_N_ITERS", "101"))           # run to objective convergence
# Evolution strip (panel f): dump the design MC surface at these iterations; the
# last one is the converged shape.  Keep CAPTURE_ITER in the list so the strip
# frame and the loop panels show the same iteration.
EVO_ITERS = sorted(set(int(x) for x in
                       os.environ.get("SO_EVO", "0,20,40,60,80,100").split(",")))
# FRAMES mode (SO_FRAMES=1): the loop-figure captures are turned OFF and instead
# EVERY iteration dumps its octree+d (so_fr_<it>.vtk) and design surface
# (so_fr_<it>.stl), plus a cumulative optimiser wall-clock (dump I/O excluded)
# to so_fr_times.txt — the inputs for the design-evolution frame series.  Run
# with SO_OUT_DIR pointing at a scratch folder so the shipped loop cache in
# shapeopt/ is never touched.
FRAMES = os.environ.get("SO_FRAMES", "0") == "1"
_stats = {}
_traj = []          # (iter, se_raw, vol_raw) each iteration -> so_traj.txt


# --------------------------------------------------------------------------
# dump helpers (legacy VTK / STL, physical coords)
# --------------------------------------------------------------------------
def _p(name):
    # SO_OUT_DIR redirects every dump (incl. so_stats/so_traj) somewhere else, so
    # this script can be re-run as a benchmark (overview/ov_bench.py runs 4
    # iterations to time one) without clobbering the shipped figure's artefacts.
    # Unset — the normal build — it is exactly os.path.join(HERE, name).
    return os.path.join(os.environ.get("SO_OUT_DIR", HERE), name)


def write_hex_vtk(path, nodes, elems, fields):
    nodes = np.asarray(nodes, np.float64)
    elems = np.asarray(elems, np.int64)
    N, M = len(nodes), len(elems)
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\noctree hex mesh\nASCII\n"
                "DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {N} float\n")
        np.savetxt(f, nodes, fmt="%.6f")
        f.write(f"\nCELLS {M} {M*9}\n")
        np.savetxt(f, np.hstack([np.full((M, 1), 8, np.int64), elems]), fmt="%d")
        f.write(f"\nCELL_TYPES {M}\n")
        np.savetxt(f, np.full(M, 12, np.int64), fmt="%d")
        f.write(f"\nPOINT_DATA {N}\n")
        for i, (nm, val) in enumerate(fields.items()):
            val = np.asarray(val, np.float64).reshape(-1)
            f.write(f"SCALARS {nm} float 1\nLOOKUP_TABLE default\n")
            np.savetxt(f, val, fmt="%.6g")


def write_structured_points(path, grid, origin, spacing, name="sdf"):
    grid = np.asarray(grid, np.float32)
    nx, ny, nz = grid.shape
    with open(path, "wb") as f:
        hdr = ("# vtk DataFile Version 3.0\nbg grid\nBINARY\n"
               "DATASET STRUCTURED_POINTS\n"
               f"DIMENSIONS {nx} {ny} {nz}\n"
               f"ORIGIN {origin[0]} {origin[1]} {origin[2]}\n"
               f"SPACING {spacing[0]} {spacing[1]} {spacing[2]}\n"
               f"POINT_DATA {nx*ny*nz}\nSCALARS {name} float 1\n"
               "LOOKUP_TABLE default\n")
        f.write(hdr.encode()); f.write(grid.ravel(order="F").astype(">f4").tobytes())


def write_stl(path, verts, faces):
    trimesh.Trimesh(np.asarray(verts, np.float64), np.asarray(faces, np.int64),
                    process=False).export(path)


def _coarse_aligned_clip_bbox(bounds, h_leaf, depth, init_depth, nominal_pad):
    """Expand the nominal clip box to coarse planes without shifting its cube.

    ``clip_octree_to_bbox`` retains cells by centroid.  A box face through a
    mixed-depth cell therefore removes alternating boundary cells, which is
    visible as a missing outer layer in section renders.  Reconstruct the cube
    origin selected for the old nominal box and expand each face to the nearest
    initial-octree plane on that same lattice.
    """
    lo, hi = (np.asarray(v, dtype=np.float64) for v in bounds)
    nominal_lo, nominal_hi = lo - nominal_pad, hi + nominal_pad
    cube_length = h_leaf * (2 ** depth)
    half_length = 0.5 * cube_length
    coarse_h = h_leaf * (2 ** (depth - init_depth))
    half_gap = half_length - 0.5 * (nominal_hi - nominal_lo)
    k = np.maximum(0, np.floor(half_gap / h_leaf + 1.0e-12)).astype(np.int64)
    cube_lo = nominal_lo - k * h_leaf
    lower_index = np.floor((nominal_lo - cube_lo) / coarse_h + 1.0e-12)
    upper_index = np.ceil((nominal_hi - cube_lo) / coarse_h - 1.0e-12)
    box_lo = cube_lo + lower_index * coarse_h
    box_hi = cube_lo + upper_index * coarse_h
    assert np.all(box_lo <= nominal_lo + 1.0e-12)
    assert np.all(box_hi >= nominal_hi - 1.0e-12)

    # Production must infer the same full-cube origin from the expanded box;
    # otherwise SDF samples, marching cubes, and BC zip would all move.
    new_half_gap = half_length - 0.5 * (box_hi - box_lo)
    new_k = np.maximum(0, np.floor(new_half_gap / h_leaf + 1.0e-12)).astype(np.int64)
    new_cube_lo = box_lo - new_k * h_leaf
    assert np.allclose(new_cube_lo, cube_lo, atol=1.0e-12, rtol=0.0), \
        f"coarse bbox would shift cube origin {cube_lo} -> {new_cube_lo}"
    return tuple(box_lo.tolist()), tuple(box_hi.tolist())


def _assert_complete_outer_box_numpy(nodes, elements, tag):
    """Reject a capture whose octree does not tile the whole clip box."""
    corners = np.asarray(nodes, dtype=np.float64)[np.asarray(elements, dtype=np.int64)]
    extents = corners.max(axis=1) - corners.min(axis=1)
    hex_volume = float(np.prod(extents, axis=1).sum())
    box_lo = np.asarray(CLIP_BBOX[0], dtype=np.float64)
    box_hi = np.asarray(CLIP_BBOX[1], dtype=np.float64)
    box_volume = float(np.prod(box_hi - box_lo))
    assert np.allclose(corners.min(axis=(0, 1)), box_lo, atol=1.0e-7, rtol=0.0), \
        f"{tag}: octree minimum does not match clip bbox"
    assert np.allclose(corners.max(axis=(0, 1)), box_hi, atol=1.0e-7, rtol=0.0), \
        f"{tag}: octree maximum does not match clip bbox"
    assert abs(hex_volume - box_volume) <= 1.0e-7 * box_volume, \
        f"{tag}: outer octree covers {hex_volume:.9g}, expected {box_volume:.9g}"
    return box_volume, hex_volume


# The BC-patch grow fix now lives in production (core.remeshing.compute_bc_patch,
# grow_rings, wired through shape_opt via the `bc_zip_patch_grow` config param);
# so_prep just passes SO_PATCH_GROW through that param below.  (Was a monkeypatch.)


# --------------------------------------------------------------------------
# capture subclass — dumps intermediates at CAPTURE_ITER, else vanilla
# --------------------------------------------------------------------------
class ShapeOptCapture(ShapeOptTrimmed):
    _cap_iter = CAPTURE_ITER
    _cur_iter = -1
    _n_step = 0
    _t0 = None            # wall clock of the first step (frames mode)
    _dump_t = 0.0         # cumulative time spent in frame I/O (excluded from the clock)
    _frame_times = None   # [(iter, cumulative_compute_seconds)]

    def _dump_frame(self, iteration):
        """One frame of the evolution series: the post-remesh octree+d (like
        panel h/a) and the design surface (like the evo strip), both the SAME
        state, plus the cumulative optimiser wall-clock with this dump's own cost
        subtracted out so the number reflects the run, not the figure."""
        import warp as wp
        t = time.perf_counter()
        nd = self._prev_octree_nodes.detach().cpu().numpy()
        el = self._prev_octree_elements.detach().cpu().numpy()
        _assert_complete_outer_box_numpy(nd, el, f"frame {iteration}")
        sdf = wp.to_torch(self.e.dof_values).detach().cpu().numpy()
        write_hex_vtk(_p(f"so_fr_{iteration}.vtk"), nd, el, {"sdf": sdf})
        s = self._mc_surface()
        if s is not None:
            s.export(_p(f"so_fr_{iteration}.stl"))
        self._dump_t += time.perf_counter() - t
        if self._frame_times is None:
            self._frame_times = []
        self._frame_times.append(
            (iteration, time.perf_counter() - self._t0 - self._dump_t, len(nd), len(el)))
        print(f"[frame] iter {iteration}: {len(nd)} nodes {len(el)} elems, "
              f"surf={'ok' if s is not None else 'MISSING'}, "
              f"t={self._frame_times[-1][1]:.1f}s")

    def step(self, tape=None):
        if self._t0 is None:
            self._t0 = time.perf_counter()
        it = self._n_step
        if (not FRAMES) and it == self._cap_iter:
            # panel (a): current design octree + SDF, BEFORE this step's update
            import warp as wp
            nd = self._prev_octree_nodes.detach().cpu().numpy()
            el = self._prev_octree_elements.detach().cpu().numpy()
            box_volume, hex_volume = _assert_complete_outer_box_numpy(
                nd, el, f"capture {it} design")
            sdf = wp.to_torch(self.e.dof_values).detach().cpu().numpy()
            write_hex_vtk(_p("so_design.vtk"), nd, el, {"sdf": sdf})
            _stats["design_nodes"] = len(nd); _stats["design_hex"] = len(el)
            _stats["outer_bbox_volume"] = box_volume
            _stats["outer_hex_volume"] = hex_volume
        r = super().step(tape)
        self._n_step += 1
        return r

    def _post_optimizer_step_remesh_impl(self, iteration):
        self._cur_iter = iteration
        _traj.append((iteration,
                      float(getattr(self._inner, "_last_se_raw", float("nan"))),
                      float(getattr(self._inner, "_last_vol_raw", float("nan")))))
        if (not FRAMES) and iteration == self._cap_iter:
            import warp as wp
            nd = self._prev_octree_nodes.detach().cpu().numpy()
            el = self._prev_octree_elements.detach().cpu().numpy()
            # panel (b): the trimmed INNER (material) mesh + |u| (real FEM solve)
            ind = self._inner._prev_octree_nodes.detach().cpu().numpy()
            iel = self._inner._prev_octree_elements.detach().cpu().numpy()
            iu = wp.to_torch(self._inner.u.dof_values).detach().reshape(ind.shape[0], -1)
            iumag = torch.linalg.norm(iu, dim=1).cpu().numpy()
            write_hex_vtk(_p("so_solve.vtk"), ind, iel, {"umag": iumag})
            _stats["umag_max"] = float(iumag.max())
            _stats["solve_nodes"] = len(ind); _stats["solve_hex"] = len(iel)
            _stats["se_raw"] = float(getattr(self._inner, "_last_se_raw", float("nan")))
            _stats["vol_raw"] = float(getattr(self._inner, "_last_vol_raw", float("nan")))
            # panel (d): dJ/dSDF sensitivity on the design octree, with the frozen
            # BC region (feet + top load) masked out (NaN) — the SDF is fixed there
            # so it is not a design region and its raw gradient is not meaningful.
            if hasattr(self, "_last_e_grad"):
                g = self._last_e_grad.detach().cpu().numpy().astype(np.float64)
                try:
                    pts = self.vtu_writer.mesh_info["points"]
                    pts = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
                    fb = self._reinit_fixed_bool
                    fb = fb.detach().cpu().numpy() if hasattr(fb, "detach") else np.asarray(fb)
                    frozen = pts[fb.astype(bool)]
                    from scipy.spatial import cKDTree
                    dmin, _ = cKDTree(frozen).query(nd, k=1)
                    # Neutralise (0), not NaN: NaN contaminates the clip's cross-
                    # section interpolation and greys the whole cut; 0 renders as
                    # the diverging-map centre so the free-region signal stands out.
                    g[dmin < 1.5 * h_target] = 0.0
                    _stats["grad_masked_nodes"] = int((dmin < 1.5 * h_target).sum())
                except Exception as _e:
                    print("grad BC mask skipped:", _e)
                write_hex_vtk(_p("so_grad.vtk"), nd, el, {"grad": g})
                _stats["grad_absmax"] = float(np.max(np.abs(g)))
            # panel (e): SDF after the GD step, before reinit
            sdf_upd = wp.to_torch(self.e.dof_values).detach().cpu().numpy()
            write_hex_vtk(_p("so_updated.vtk"), nd, el, {"sdf": sdf_upd})
        super()._post_optimizer_step_remesh_impl(iteration)
        if FRAMES:
            self._dump_frame(iteration)
            return
        if iteration == self._cap_iter:
            import warp as wp
            nd = self._prev_octree_nodes.detach().cpu().numpy()
            el = self._prev_octree_elements.detach().cpu().numpy()
            _assert_complete_outer_box_numpy(nd, el, f"capture {iteration} remesh")
            sdf_new = wp.to_torch(self.e.dof_values).detach().cpu().numpy()
            write_hex_vtk(_p("so_new.vtk"), nd, el, {"sdf": sdf_new})
            _stats["new_nodes"] = len(nd); _stats["new_hex"] = len(el)
            # resolved Dirichlet/Neumann BC nodes (for verifying the square feet)
            try:
                pts = self.vtu_writer.mesh_info["points"]
                pts = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
                fb = self._reinit_fixed_bool
                fb = fb.detach().cpu().numpy() if hasattr(fb, "detach") else np.asarray(fb)
                np.save(_p("so_bcnodes.npy"), pts[fb])
            except Exception as _e:
                print("bcnodes dump skipped:", _e)
        # panel (f): design-surface evolution strip — dump the current MC surface
        # (full exterior) at the sampled iterations, last == converged shape.
        if iteration in EVO_ITERS:
            _s = self._mc_surface()
            if _s is not None:
                _s.export(_p(f"so_evo_{iteration}.stl"))
                print(f"[evo] wrote so_evo_{iteration}.stl ({len(_s.faces)} faces)")

    def _build_mc_surface_and_bvh(self, grid_sdf_phys, _precomputed=None):
        cap = ((not FRAMES) and self._cur_iter == self._cap_iter
               and hasattr(self, "_bc_zip_patch"))
        if not cap:
            return super()._build_mc_surface_and_bvh(grid_sdf_phys, _precomputed)
        # instrumented copy of the real sequence (raw MC → filter → close → stitch)
        from core.analytical_sdf import extract_surface, build_warp_mesh
        from core.remeshing import (_close_open_loops, filter_non_watertight_components,
                                    stitch_bc_patches)
        grid_norm = (grid_sdf_phys / self._remesh_sdf_scale).reshape(self._sussman_grid_shape)
        # bg grid (panel f input) in physical coords
        gs = self._sussman_grid_shape
        org = [float(self._remesh_geo_min[i]) for i in range(3)]
        sp = [float(self._sussman_grid_spacing[i]) for i in range(3)]
        write_structured_points(_p("so_bg.vtk"),
                                grid_sdf_phys.reshape(gs).detach().cpu().numpy(), org, sp)
        vN, faces, probeN = extract_surface(grid_norm, domain_length=2.0,
                                            device=self._remesh_device)
        vph = self._normalized_to_physical(vN.double()).float().cpu().numpy()
        write_stl(_p("so_mc_raw.stl"), vph, faces.cpu().numpy())
        _stats["mc_raw_faces"] = int(faces.shape[0])
        keep = filter_non_watertight_components(faces, keep_largest=True)
        _stats["mc_dropped_faces"] = int(faces.shape[0] - keep.shape[0])
        faces = faces[keep]
        vN, faces = _close_open_loops(vN, faces)
        _stats["mc_clean_faces"] = int(faces.shape[0])
        vph = self._normalized_to_physical(vN.double()).float().cpu().numpy()
        write_stl(_p("so_mc_clean.stl"), vph, faces.cpu().numpy())
        # BC restore + zip stitch (panel g)
        bc_patches = self._bc_zip_patch.get("components", [])
        vphys_d = self._normalized_to_physical(vN.to(torch.float64))
        out_vphys, faces_s, face_labels = stitch_bc_patches(
            base_verts_phys=vphys_d, base_faces=faces, bc_patches=bc_patches, log=self.log)
        write_stl(_p("so_stitched.stl"), out_vphys.cpu().numpy(), faces_s.cpu().numpy())
        _stats["stitch_faces"] = int(faces_s.shape[0]); _stats["n_bc_patch"] = len(bc_patches)
        # panel (g) colouring: stitch_bc_patches' own per-face labels — 1 = the
        # restored BC-patch faces (the surfaces actually held fixed / frozen in the
        # computation), 0 = free base, 2 = zip bridge.  These are exact face-sets,
        # so the patches are clean AND identical to what the pipeline fixes.
        np.savez(_p("so_bc_masks.npz"),
                 verts=out_vphys.detach().cpu().numpy(),
                 faces=faces_s.detach().cpu().numpy(),
                 labels=face_labels.detach().cpu().numpy())
        # finish exactly like the real method so the pipeline continues correctly
        vN = self._physical_to_normalized(out_vphys).to(vN.dtype); faces = faces_s
        vphys = self._normalized_to_physical(vN.double()).float()
        probe_phys = self._normalized_to_physical(probeN.double().unsqueeze(0)).squeeze(0).float()
        bbox = float(torch.linalg.norm(vphys.amax(0) - vphys.amin(0)))
        wpm, neg, ka = build_warp_mesh(vphys, faces, probe_phys, bbox, device=self._remesh_device)
        self._mc_wp_mesh, self._mc_wp_negate, self._mc_wp_keep_alive = wpm, neg, ka
        self._mc_verts_norm, self._mc_faces, self._mc_bbox_diag_phys = vN, faces, bbox
        return self._mc_geo_func()

    def _mc_surface(self):
        """Current design's MC surface as a trimesh (for the evolution strip)."""
        try:
            v, f = self._get_or_build_mc_surface_physical()
            return trimesh.Trimesh(vertices=v.detach().cpu().numpy(),
                                   faces=f.detach().cpu().numpy(), process=False)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  [evo] MC surface unavailable: {exc}")
            return None


# --------------------------------------------------------------------------
# Design domain: a solid cylinder (the "blank" for a table / stool).  A downward
# load presses the whole top surface; four small feet on the bottom are fixed;
# the optimizer targets 50% volume and maximises stiffness (min compliance), so
# it carves the cylinder into a top plate standing on four legs.  BC regions are
# picked with production primitives (SphereSelector / HighestSelector) — like the
# bunny figure — not coordinate lambdas.
# --------------------------------------------------------------------------
from core.bc_surface_selectors import (BoundingBoxSelector, CylinderSelector,
                                       HighestSelector, LowestSelector)

R_CYL = float(os.environ.get("SO_R", "1.0"))
H_CYL = float(os.environ.get("SO_H", "1.0"))
Z_TOP, Z_BOT = H_CYL / 2.0, -H_CYL / 2.0

# Four short cylindrical FEET, unioned onto the bottom, so the fixed supports are
# real geometry the table stands on (not squares painted on a flat disk).  Each
# foot is a stubby cylinder on the bolt-circle R_FOOT that PROTRUDES below the
# disk (bottom at Z_BOT-FOOT_H) with a small overlap FOOT_OV up into the disk for
# a clean watertight boolean union (needs the manifold3d backend).
R_FOOT = float(os.environ.get("SO_RFOOT", "0.6"))
FOOT_R = float(os.environ.get("SO_FOOTR", "0.24"))   # foot cylinder radius (larger feet)
FOOT_H = float(os.environ.get("SO_FOOT_H", "0.22"))  # protrusion below the disk (TALL)
FOOT_OV = float(os.environ.get("SO_FOOT_OV", "0.06"))  # overlap up into the disk
_base = trimesh.creation.cylinder(radius=R_CYL, height=H_CYL, sections=96)
# Feet on the x/y axes (0/90/180/270 deg) so a centered coordinate-plane cut
# (y=0, like the bunny) slices through two opposite feet — showing two legs in
# the octree profile without a diagonal cut.
_ang = np.deg2rad([0.0, 90.0, 180.0, 270.0])
_foot_meshes = []
for _a in _ang:
    _fm = trimesh.creation.cylinder(radius=FOOT_R, height=FOOT_H + FOOT_OV, sections=48)
    _fm.apply_translation([R_FOOT * np.cos(_a), R_FOOT * np.sin(_a),
                           Z_BOT - FOOT_H + (FOOT_H + FOOT_OV) / 2.0])
    _foot_meshes.append(_fm)
box = trimesh.boolean.union([_base] + _foot_meshes)
trimesh.repair.fix_normals(box)
Z_FOOT_BOT = Z_BOT - FOOT_H                      # true lowest z (foot caps)
# Fix the feet with ONE percentile selector: the protruding feet are the lowest
# slab of the mesh, so LowestSelector("z", fraction) grabs all four at once (and
# keeps grabbing them as the shape evolves, since the frozen feet stay lowest).
# Freeze an EXTREMELY SHORT band: just the four bottom foot surfaces (caps) plus a
# little wall above them (FROZEN_H).  This way the frozen/free boundary — and the
# zip stitch seam that lives on it — sits at the very bottom of the foot, hidden,
# and the optimizer freely builds the leg above (adding the volume it needs) as one
# continuous surface with no mid-leg seam.  The tall foot (FOOT_H) is scaffolding:
# it puts the frozen band + the 3-ring grow (~3h≈0.09) well below the flat disk
# bottom (z=Z_BOT) so the patch never covers the large bottom face; the optimizer
# reshapes the foot above the frozen band into the leg.
FROZEN_H = float(os.environ.get("SO_FROZEN_H", "0.02"))  # frozen band height above the foot cap (~1 cell)
_z_lo = Z_BOT - FOOT_H                               # foot cap = lowest surface z
_z_thr = _z_lo + FROZEN_H                            # freeze top (well below the disk)
FOOT_FRAC = float(os.environ.get("SO_FOOT_FRAC", f"{(_z_thr - _z_lo) / (Z_TOP - _z_lo):.4f}"))
_fix_sel = LowestSelector("z", fraction=FOOT_FRAC)
print(f"[feet] FOOT_H={FOOT_H} FROZEN_H={FROZEN_H} freeze top z={_z_thr:.3f} "
      f"(disk bottom z={Z_BOT}); grow-3 reach ~{_z_thr + 3*(3.84/2**int(os.environ.get('SO_DEPTH','7'))):.3f}")
# Downward load on the top surface.  SO_TOPMODE=full loads the whole top (the
# tabletop stays connected to the legs, so it can't separate); SO_TOPMODE=disk
# loads a central disk of radius R_TOP that stays off the sharp rim.
if os.environ.get("SO_TOPMODE", "full") == "disk":
    R_TOP = float(os.environ.get("SO_RTOP", "0.72"))
    _load_sel = CylinderSelector((0.0, 0.0, Z_TOP - 0.12), (0.0, 0.0, Z_TOP + 0.12), R_TOP)
else:
    _load_sel = HighestSelector("z", fraction=float(os.environ.get("SO_TOP_FRAC", "0.06")))

depth = int(os.environ.get("SO_DEPTH", "7")); init_depth = 4
h_target = 3.84 / (2 ** depth)      # keep the octree cube side ~3.84 units
# The bg grid is snapped so its planes land on `bbox_lo + n*h_target`, and
# bbox_lo = geometry_min - pad.  A pad that is an integer multiple of h_target
# puts geometry_min (here the flat foot caps at z=Z_BOT-FOOT_H) exactly ON a grid
# plane, so those nodes sit at SDF=0 and marching cubes degenerates into sliver
# triangles (the iso-surface is tangent to the grid — the "ugly feet").  Offset
# pad by half a cell so the caps fall mid-cell and MC stays clean.
nominal_pad = 0.3 + 0.5 * h_target
eps = 0.5 * h_target
center = [0.0, 0.0, 0.0]
CLIP_BBOX = _coarse_aligned_clip_bbox(
    box.bounds, h_target, depth, init_depth, nominal_pad
)
print(f"coarse-aligned clip bbox: {CLIP_BBOX[0]} -> {CLIP_BBOX[1]}")
LOAD_TRACTION = (float(os.environ.get("SO_TX", "0.0")),
                 float(os.environ.get("SO_TY", "0.0")),
                 float(os.environ.get("SO_TZ", "-0.02")))

overrides = {
    "phase_field_epsilon": eps, "remeshing": True, "remeshing_method": "marchingcubes",
    "reinit_sdf_method": "marchingcubes",
    "sdf_bc_fix_method": os.environ.get("SO_BCFIX", "zip"),
    "remesh_octree_depth": depth, "remesh_init_octree_depth": init_depth,
    "design_variable": "sdf", "phi_reproject_band_k": 3.0,
    "bc_zip_patch_grow": int(os.environ.get("SO_PATCH_GROW", "0")),
    "output_per_optimize_step": int(os.environ.get("SO_SAVE_EVERY", "1")),
    "max_optimize_iters": N_ITERS,
    "absolute_tol": float(os.environ.get("SO_ABSOLUTE_TOL", "5e-7")),
    "gd_lr": float(os.environ.get("SO_GD_LR", "8.0")),
    "target_volume_fraction": float(os.environ.get("SO_VOLFRAC", "0.5")),
    "shape_opt_w_vol": float(os.environ.get("SO_W_VOL", "5.0")),
    "optimize_relative_tol": 1e-9,
    "remeshing_trimesh": {"mesh": box, "mesh_center": center,
                          "h_target": h_target, "clip_bbox": CLIP_BBOX},
    "dirichlet_bcs_dict": {_fix_sel: 0.0},
    "neumann_bcs_dict": {_load_sel: LOAD_TRACTION},
}
_stats["load_traction"] = LOAD_TRACTION
_stats["h_target"] = h_target
_stats["clip_bbox_lo"] = CLIP_BBOX[0]
_stats["clip_bbox_hi"] = CLIP_BBOX[1]

prob = ShapeOptCapture("configs.config_shape_opt_cantilever", override_params=overrides)
_stats["sdf_truncation"] = float(prob._sdf_truncation)
_stats["sdf_refinement_band"] = float(prob._sdf_refinement_band)
prob.optimize()

# write stats
import warp as wp  # noqa
gpu = torch.cuda.get_device_name(0)
if FRAMES:
    ft = prob._frame_times or []
    with open(_p("so_fr_times.txt"), "w") as fp:
        fp.write("# iter cum_seconds nodes elems\n")
        for it, ts, nn, ne in ft:
            fp.write(f"{it} {ts:.4f} {nn} {ne}\n")
    with open(_p("so_fr_stats.txt"), "w") as fp:
        fp.write(f"gpu={gpu}\n")
        fp.write(f"n_frames={len(ft)}\n")
        fp.write(f"total_seconds={ft[-1][1] if ft else 0.0:.2f}\n")
    print(f"FRAMES: wrote {len(ft)} so_fr_*.vtk/.stl + so_fr_times.txt "
          f"({ft[-1][1] if ft else 0.0:.1f}s optimiser wall clock) to",
          os.environ.get("SO_OUT_DIR", HERE))
    sys.exit(0)
with open(_p("so_stats.txt"), "w") as fp:
    for k, v in _stats.items():
        fp.write(f"{k}={v}\n")
    fp.write(f"gpu={gpu}\n")
    fp.write(f"capture_iter={CAPTURE_ITER}\n")
with open(_p("so_traj.txt"), "w") as fp:
    fp.write("# iter se_raw vol_raw\n")
    for it, se, vol in _traj:
        fp.write(f"{it} {se:.8e} {vol:.8e}\n")
print("STATS", _stats, "gpu", gpu)
print("wrote so_*.vtk / so_*.stl / so_stats.txt / so_traj.txt to", HERE)
