"""Prep step (conda `mechfoundry-warp117`, GPU): everything the diffused-Nitsche figure shows.

Reads the SAME geometry ``bunny_sdf_octree`` shows — ``bunny.stl`` and the cube
half-side in ``bunny_sdf.meta``, both written by ``bunny/prep_bunny_sdf.py`` —
and builds the octree with ``bunny_sdf_octree``'s own generator and arguments,
so the two figures print the same mesh (see WHICH BUNNY below).  On that mesh it
re-derives, with the REAL solver functions, each stage the diffused Nitsche
formulation needs:

    surface d ──► phi = 1/2 (1 - tanh(d/eps))           eps = 1/2 h_min
              ──► boundary region: elements with a node in 1e-3 < phi < 1-1e-3
                  (NitscheNeoHookean._build_adaptive_quadrature)
              ──► BC region: boundary region ∩ resolved SurfaceSelector
                  (NitscheNeoHookean._classify_bc_elements → the selector's own
                   patch-box cull + BVH distance test)
              ──► trimmed mesh: NitscheNeoHookean._trim_exterior_elements

WHICH BUNNY
-----------
NOT the TripoSG one.  This figure used to read ``bc_bunny_geom.npz`` — the same
decode ``bc_region_id`` uses — and that put it beside ``bunny_sdf_octree`` in the
paper showing a visibly different mesh: same source .obj, but one side went
through ``mesh2latent -> latent2octree -> marching cubes`` and the other did not,
and the cubes were padded differently (1.335x the model against 1.16x), which
alone made the cells 15 % coarser here.  Nothing in this figure needs the
decoder — it illustrates the FORMULATION, not the generative path — so it now
takes ``bunny_sdf_octree``'s geometry and its octree call verbatim.  The two
meshes come out identical, and the BSO_NODES/BSO_HEXES assert below is the gate.

This figure now works in ``prep_bunny_sdf``'s frame — z-up — where it used to be
y-up, and ``render_dn`` was moved to bunny_sdf_octree's camera to match.  That
is not cosmetic: a 90 deg rotation of the mesh is exact in the coordinates but
NOT in the distances warp computes from them, and near ``hierarchical_extrac-
tion``'s refinement band a handful of cells flip either way.  Measured: rotating
into the old y-up frame moved the octree by 65 nodes / 42 hexes (45,147/35,484
against 45,082/35,442).  Small, but it is the difference between "the same mesh"
and "almost the same mesh", so the frame stays put and the camera moves.

The nodal SDF is a true, untruncated warp/BVH distance: ``sdf_truncation`` only
sets ``hierarchical_extraction``'s refinement band (0.9 x it), it does not clamp
the values.  That matters here — the decoder's SDF really is clamped at 0.01,
and with eps = h/2 the far field would never reach the 1e-3 that the boundary
test needs, so ``is_boundary`` would flag every element and
``_trim_exterior_elements`` would remove none.

Writes (all consumed by ``render_dn.py``):
  dn_full.vtk         — the FULL octree mesh, nodal ``sdf``/``phi`` + cell ``tag``
  dn_trim.vtk         — the trimmed octree mesh, nodal ``sdf``/``phi``
  dn_surf_{rest,ears,base}.stl — the embedded surface, split into three disjoint
                        pieces for panel (d)'s translucent overlay
  dn_stats.txt        — every number the caption quotes
"""

import math
import os
import sys

import numpy as np
import torch
import trimesh

# Resolve the production package from THIS file, not from the cwd: the figures
# moved into per-figure subfolders (AGENTS.md "LAYOUT"), which silently broke
# the old cwd-relative "../src/warp_wrapper" — it now points inside
# illustrations/ and appends a path that does not exist.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(_REPO, "src", "warp_wrapper"))

from core.bc_surface_selectors import (
    FloodFillSelector,
    GrowSelectionSelector,
    HalfSpaceSelector,
    HighestSelector,
    LowestSelector,
    _build_adj_pairs,
    _cc_by_label_prop,
    _compute_face_attributes,
)
from core.nitsche_field import NitscheNeoHookean
from core.utils import identify_boundary_nodes

# bunny_sdf_octree's octree generator.  It lives under triposg_wrapper but is a
# plain trimesh -> warp-BVH SDF -> sdf2octree utility; no network is involved.
sys.path.insert(0, _REPO)
from src.triposg_wrapper.trimesh_sdf import trimesh_to_octree  # noqa: E402

device = "cuda"

# --- production knobs (see the module docstring for provenance) --------------
INIT_DEPTH = 3                 # == prep_octree_sdf.py / octree_step1's l_init
OCTREE_DEPTH = 6               # == its l_final; the FEM mesh, h_min = L / 2^6
EPS_RATIO = 0.5                # LatentShapeOpt(epsilon_to_size_ratio=0.5)
PHI_LO, PHI_HI = 1e-3, 1.0 - 1e-3          # _build_adaptive_quadrature
TRIM_PHI = 1e-3                            # _trim_exterior_elements default

# Neumann = the two ears: bc_region_id's recipe (AGENTS.md §5b table), with the
# two flood-fill angles re-tuned for THIS surface.  They had to be: bc_region_id
# runs on the TripoSG decode's marching-cubes tessellation, this runs on
# prep_bunny_sdf's Loop-subdivided + Taubin-smoothed scan, and a dihedral
# threshold is a property of the tessellation, not of the shape.  These are
# bcregion/bc_bunny_select.py's tuned values, and they are shared deliberately:
# the two figures draw the same bunny, so "the ears" has to mean one region in
# both.  Each ear takes the LARGEST gate that still stays on it (the front ear
# bursts to the floor at 3.8, the back at 5.0) and enough rings to close what
# the gate cannot cross.  See that file for the sweeps behind the numbers.
SEED_FRACTION = 0.05
FRONT_ANGLE, BACK_ANGLE = 3.4, 4.6
FRONT_RINGS, BACK_RINGS = 14, 14
# Names the two halves only — the ear it calls "front" gets the generous
# FRONT_RINGS patch — but it is read in the geometry's own frame, so it must
# point the way the camera does.  Get the sign wrong and the asymmetric recipe
# lands on the other ear, its 5 deg flood fill leaks down the neck and the patch
# reaches the base, tripping the Dirichlet/Neumann overlap assert below.  This
# is render_dn's VIEW_DIR, which is bunny_sdf_octree's camera.
VIEW_DIR_BC = np.array([1.5, -3.0, 1.25])

# Dirichlet = the clamped base: the bottom BOTTOM_FRACTION of the model's height,
# by percentile, and nothing else.  The image-to-FEM surface-BC example also ANDs a downward
# normal cone, and that does not survive the move to this surface: the decode has
# a flat base (a capped hole), while prep_bunny_sdf caps the scan's hole and then
# Taubin-smooths it, leaving a ROUNDED underside where almost nothing points
# straight down — the production 5 % / 45 deg picks 13 facets out of 161,568.
# Percentile alone needs no tuning to survive that, and at production's own 5 %
# it lands on 26,409 facets against the decode's 24,294.
BOTTOM_FRACTION = 0.05

# prep_bunny_sdf's frame is z-up (its `v[:, [0, 2, 1]]` puts the model's up on
# z), so every selector that names a direction names it here, once.
UP = "z"

# ---------------------------------------------------------------------------
# 1. Geometry + octree — bunny_sdf_octree's, verbatim (see WHICH BUNNY above)
# ---------------------------------------------------------------------------
# H and bunny.stl are prep_bunny_sdf's, produced in THIS directory by build.sh
# exactly as build_sdfoctree does it.
H = float(open("bunny_sdf.meta").readline())         # == prep_octree_sdf.py:26
bunny = trimesh.load("bunny.stl")                    # == prep_octree_sdf.py:27
# NO transform of any kind here — see WHICH BUNNY.  render_dn takes
# bunny_sdf_octree's camera instead, and UP below is this frame's up (+z).

domain_length = 2.0 * H
depth = OCTREE_DEPTH
h_min = domain_length / (2 ** depth)
extent = float(np.max(bunny.vertices.max(0) - bunny.vertices.min(0)))

# prep_octree_sdf.py's `_gen`, argument for argument.  `sdf_truncation` sets the
# refinement band only (0.9 x it); the nodal SDF that comes back is a true warp
# BVH distance, which is what the phase field needs.
octree = trimesh_to_octree(
    bunny,
    bounds=(-H, -H, -H, H, H, H),
    init_octree_depth=INIT_DEPTH,
    octree_depth=depth,
    domain_length=domain_length,
    sdf_truncation=0.01,
)
nodes = octree["nodes"].to(torch.float32).to(device)
elements = octree["elements"].long().to(device)
node_sdf = octree["node_sdf"].to(torch.float32).to(device)
octree = {"nodes": nodes, "elements": elements, "sdf": node_sdf}

# The gate for "the two figures print the same mesh".  These are what
# prep_octree_sdf.py reports at INIT_DEPTH -> OCTREE_DEPTH on the same bunny.stl
# and the same cube, and the figure's whole claim to matching bunny_sdf_octree
# rests on them: same generator, same arguments, same input, same mesh.
BSO_NODES, BSO_HEXES = 45082, 35442
assert (nodes.shape[0], elements.shape[0]) == (BSO_NODES, BSO_HEXES), (
    f"octree is {nodes.shape[0]}N/{elements.shape[0]}H, bunny_sdf_octree is "
    f"{BSO_NODES}N/{BSO_HEXES}H — the two figures no longer show the same mesh")

verts = torch.tensor(np.ascontiguousarray(bunny.vertices), dtype=torch.float32,
                     device=device)
faces = torch.tensor(np.ascontiguousarray(bunny.faces), dtype=torch.long,
                     device=device)
print(f"cube {domain_length:.4f} (H={H:.4f})   model extent {extent:.4f}   "
      f"{extent / h_min:.1f} cells across it")
print(f"surface {verts.shape[0]}V {faces.shape[0]}F   "
      f"h_min={h_min}  depth={depth}  L={domain_length}")
print(f"octree: {nodes.shape[0]} nodes, {elements.shape[0]} hexes, "
      f"nodal d in [{float(node_sdf.min()):.4f}, {float(node_sdf.max()):.4f}]"
      f"   (== bunny_sdf_octree)")

# ---------------------------------------------------------------------------
# 2. Phase field.  eps = 1/2 h_min, exactly as LatentShapeOpt auto-computes it.
# ---------------------------------------------------------------------------
eps = EPS_RATIO * h_min
phi_nodal = (0.5 * (1.0 - torch.tanh(node_sdf / eps))).clamp(0.0, 1.0)
# phi in (PHI_LO, PHI_HI)  <=>  |d| < eps * atanh(1 - 2*PHI_LO)
band_halfwidth = eps * math.atanh(1.0 - 2.0 * PHI_LO)
print(f"eps = {EPS_RATIO} * h_min = {eps:.6f}    "
      f"band |d| < {band_halfwidth:.6f} = {band_halfwidth / h_min:.3f} h_min")

# ---------------------------------------------------------------------------
# 3. Boundary region — NitscheNeoHookean._build_adaptive_quadrature, verbatim
# ---------------------------------------------------------------------------
elem_node_phi = phi_nodal[elements]                       # (E, 8)
is_boundary = ((elem_node_phi > PHI_LO) & (elem_node_phi < PHI_HI)).any(dim=1)
n_band = int(is_boundary.sum())
print(f"boundary region: {n_band} / {elements.shape[0]} elements "
      f"({100.0 * n_band / elements.shape[0]:.1f} %)")
assert 0 < n_band < elements.shape[0], "boundary-region test degenerated"

# Cross-check the element test against the closed-form band width: every band
# element must own a node inside the band, and no non-band element may.
_node_in_band = node_sdf.abs() < band_halfwidth
assert bool((_node_in_band[elements].any(1) == is_boundary).all()), \
    "phi threshold and |d| band disagree"

# ---------------------------------------------------------------------------
# 4. BC selectors — the production recipes, resolved with the solver's tolerances
# ---------------------------------------------------------------------------
# NitscheNeoHookean._bc_match_tolerance: max(1.5*eps, 0.75*h_min), used both
# as the selectors' eps_expansion and to widen torch.isclose in classify().
eps_tol = max(1.5 * eps, 0.75 * h_min)
print(f"eps_expansion = max(1.5*eps, 0.75*h_min) = {eps_tol:.6f}")

centroids, normals, areas = _compute_face_attributes(verts, faces)
adj = _build_adj_pairs(faces)

# --- Neumann: the two ears (bc_region_id's split flood fill) ---
seed_sel = HighestSelector(UP, fraction=SEED_FRACTION)
seed_mask = seed_sel.select(verts, faces, centroids, normals, areas, None,
                            adj_pairs=adj)
seed_comps = sorted(_cc_by_label_prop(seed_mask, adj), key=len, reverse=True)
assert len(seed_comps) == 2, f"expected 2 ear tips, got {len(seed_comps)}"
tip_c = [centroids[c].mean(0).cpu().numpy() for c in seed_comps]
axis = tip_c[0] - tip_c[1]
axis /= np.linalg.norm(axis)
mid = 0.5 * (tip_c[0] + tip_c[1])
near_first = float(tip_c[0] @ VIEW_DIR_BC) > float(tip_c[1] @ VIEW_DIR_BC)
front_half = HalfSpaceSelector(mid, axis if near_first else -axis)
back_half = HalfSpaceSelector(mid, -axis if near_first else axis)


def one_ear(half, angle, rings):
    return GrowSelectionSelector(
        FloodFillSelector(seed_sel & half, angle) & half, num_rings=rings)


ear_sel = (one_ear(front_half, FRONT_ANGLE, FRONT_RINGS)
           | one_ear(back_half, BACK_ANGLE, BACK_RINGS))
ear_mask = ear_sel.select(verts, faces, centroids, normals, areas, None,
                          adj_pairs=adj)

# --- Dirichlet: image-to-FEM surface-BC example's clamped base ---
bottom_sel = LowestSelector(UP, fraction=BOTTOM_FRACTION)
bottom_mask = bottom_sel.select(verts, faces, centroids, normals, areas, None,
                                adj_pairs=adj)
print(f"ears: {int(ear_mask.sum())} facets   "
      f"bottom {BOTTOM_FRACTION:.0%}: {int(bottom_mask.sum())} facets")

for sel in (ear_sel, bottom_sel):
    sel.resolve(verts, faces, eps_expansion=eps_tol)

# ---------------------------------------------------------------------------
# 5. Intersection — NitscheNeoHookean._classify_bc_elements, verbatim
#    (widened torch.isclose + identify_boundary_nodes over the band elements'
#     nodes; the selector itself does patch-box cull → BVH distance <= eps_tol)
# ---------------------------------------------------------------------------
band_idx = torch.where(is_boundary)[0]
band_elem_nodes = elements[band_idx]                       # (B, 8)
flat_node_ids = band_elem_nodes.reshape(-1)
unique_node_ids, flat_to_unique = torch.unique(
    flat_node_ids, sorted=False, return_inverse=True)
unique_positions = nodes[unique_node_ids]

_orig_isclose = torch.isclose


def _widened_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    return _orig_isclose(a, b, rtol=rtol, atol=max(atol, eps_tol),
                         equal_nan=equal_nan)


def classify(selector):
    """Band elements owning at least one node the selector accepts."""
    torch.isclose = _widened_isclose
    try:
        matched = identify_boundary_nodes(selector, unique_positions, 3)
    finally:
        torch.isclose = _orig_isclose
    unique_hit = torch.zeros(
        unique_positions.shape[0], dtype=torch.bool, device=device)
    unique_hit[matched] = True
    hit = unique_hit[flat_to_unique]
    return hit.reshape(-1, 8).any(dim=1), int(unique_hit.sum())


ear_bc, ear_node_hits = classify(ear_sel)
bottom_bc, bottom_node_hits = classify(bottom_sel)
print(f"BC elements — ears: {int(ear_bc.sum())} / {n_band};  "
      f"bottom: {int(bottom_bc.sum())} / {n_band}")
assert int(ear_bc.sum()) > 0 and int(bottom_bc.sum()) > 0, "empty BC region"
assert not bool((ear_bc & bottom_bc).any()), "Dirichlet and Neumann overlap"

# Cull-only counts: the band nodes the selector's patch box alone admits, i.e.
# how many reach its BVH distance test.
band_node_pos = nodes[torch.unique(band_elem_nodes.reshape(-1))]


def in_patch_box(sel):
    lo, hi = sel._patch_lo.to(device), sel._patch_hi.to(device)
    return ((band_node_pos >= lo) & (band_node_pos <= hi)).all(1)


ear_in_box = int(in_patch_box(ear_sel).sum())
bottom_in_box = int(in_patch_box(bottom_sel).sum())
band_nodes_total = int(band_node_pos.shape[0])

# ---------------------------------------------------------------------------
# 6. Trimmed mesh — NitscheNeoHookean._trim_exterior_elements (production)
# ---------------------------------------------------------------------------
trim_mesh = {
    "nodes": octree["nodes"].clone(),
    "elements": octree["elements"].clone(),
    "sdf": octree["sdf"].clone(),
}
for k in ("edge_hanging", "face_hanging"):
    if k in octree:
        trim_mesh[k] = octree[k].clone()
# Same predicate the trimmer uses, kept so the per-element tags can be carried
# onto the compacted mesh (the trimmer only reports the node mapping).
keep_elem = elem_node_phi.max(dim=1).values >= TRIM_PHI
n_before, n_after = NitscheNeoHookean._trim_exterior_elements(
    trim_mesh, eps, device, phi_threshold=TRIM_PHI)
assert int(keep_elem.sum()) == n_after, "keep-mask disagrees with the trimmer"
# The trimmed mesh is exactly interior ⊎ boundary region: every band element
# owns a node with |d| < band_halfwidth, hence max phi >= 1e-3, hence survives.
assert bool((is_boundary <= keep_elem).all()), "trimming dropped band elements"
print(f"trimmed: {n_before} -> {n_after} elements "
      f"(removed {n_before - n_after}, {100.0 * n_after / n_before:.1f} % kept), "
      f"{trim_mesh['nodes'].shape[0]} nodes")
assert n_after < n_before, "trimming removed nothing"

# ---------------------------------------------------------------------------
# 7. Export
# ---------------------------------------------------------------------------
def write_hexes(path, nodes_t, elems_t, name, point_scalars=None,
                cell_scalar=None, cell_name="tag"):
    """Legacy ASCII VTK UNSTRUCTURED_GRID of hexes (sdf2octree corner order
    already matches VTK_HEXAHEDRON, see AGENTS.md §5).

    ``point_scalars`` is a {name: tensor} dict — a legacy POINT_DATA block may
    carry several SCALARS arrays, which is how one mesh serves both the d panel
    and the phi panels."""
    v = nodes_t.detach().cpu().numpy().astype(np.float64)
    e = elems_t.detach().cpu().numpy().astype(np.int64)
    with open(path, "w") as f:
        f.write(f"# vtk DataFile Version 3.0\n{name}\nASCII\n"
                "DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(v)} float\n")
        np.savetxt(f, v, fmt="%.6f")
        f.write(f"\nCELLS {len(e)} {len(e) * 9}\n")
        np.savetxt(f, np.hstack([np.full((len(e), 1), 8, np.int64), e]), fmt="%d")
        f.write(f"\nCELL_TYPES {len(e)}\n")
        np.savetxt(f, np.full(len(e), 12, np.int64), fmt="%d")
        if point_scalars:
            f.write(f"\nPOINT_DATA {len(v)}\n")
            for nm, arr in point_scalars.items():
                f.write(f"SCALARS {nm} float 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, arr.detach().cpu().numpy(), fmt="%.6f")
        if cell_scalar is not None:
            f.write(f"\nCELL_DATA {len(e)}\nSCALARS {cell_name} float 1\n"
                    "LOOKUP_TABLE default\n")
            np.savetxt(f, cell_scalar.detach().cpu().numpy(), fmt="%d")
    print(f"wrote {path} ({len(e)} hexes, {len(v)} nodes)")


# The FULL octree mesh carries phi at its nodes and a per-cell region tag:
#   0 = no Nitsche term                       2 = boundary region ∩ Neumann (ears)
#   1 = boundary region                       3 = boundary region ∩ Dirichlet (base)
# Panels (c) and (d) draw this mesh coloured by phi and overlay a Threshold on
# tag >= 1; (e) draws the trimmed mesh, so the two are directly comparable.
tag_full = torch.zeros(elements.shape[0], dtype=torch.float32, device=device)
tag_full[band_idx] = 1.0
tag_full[band_idx[ear_bc]] = 2.0
tag_full[band_idx[bottom_bc]] = 3.0
write_hexes("dn_full.vtk", nodes, elements,
            "full octree mesh, d + phi + region tags",
            point_scalars={"sdf": node_sdf, "phi": phi_nodal},
            cell_scalar=tag_full)

trim_phi = (0.5 * (1.0 - torch.tanh(trim_mesh["sdf"] / eps))).clamp(0.0, 1.0)
write_hexes("dn_trim.vtk", trim_mesh["nodes"].to(torch.float32),
            trim_mesh["elements"].long(), "trimmed octree mesh, d + phi",
            point_scalars={"sdf": trim_mesh["sdf"], "phi": trim_phi})

# The embedded surface, for panel (d)'s translucent overlay.  Three DISJOINT
# meshes, never body-plus-subset: drawing the selected facets on top of a copy
# of themselves makes them coincident geometry and the z-fight speckles body
# colour through the highlight (AGENTS.md §5b).
vnp = verts.cpu().numpy()
fnp = faces.cpu().numpy()


def export_faces(mask, path):
    sub = fnp[mask.cpu().numpy()]
    trimesh.Trimesh(vertices=vnp, faces=sub, process=False).export(path)
    print(f"wrote {path} ({len(sub)} faces)")


export_faces(~(ear_mask | bottom_mask), "dn_surf_rest.stl")
export_faces(ear_mask, "dn_surf_ears.stl")
export_faces(bottom_mask, "dn_surf_base.stl")

# ---------------------------------------------------------------------------
# 9. Stats — every figure caption number comes from here
# ---------------------------------------------------------------------------
ear_v = verts[faces[ear_mask].reshape(-1)]
bot_v = verts[faces[bottom_mask].reshape(-1)]
with open("dn_stats.txt", "w") as f:
    f.write(f"h_min {h_min:.7f}\n")
    f.write(f"eps {eps:.7f}\n")
    f.write(f"eps_ratio {EPS_RATIO}\n")
    f.write(f"depth {depth}\n")
    f.write(f"phi_lo {PHI_LO}\n")
    f.write(f"band_halfwidth {band_halfwidth:.7f}\n")
    f.write(f"band_h_ratio {band_halfwidth / h_min:.4f}\n")
    f.write(f"band_eps_ratio {band_halfwidth / eps:.4f}\n")
    f.write(f"elements {int(elements.shape[0])}\n")
    f.write(f"nodes {int(nodes.shape[0])}\n")
    f.write(f"band_elements {n_band}\n")
    f.write(f"band_nodes {band_nodes_total}\n")
    f.write(f"init_depth {INIT_DEPTH}\n")
    f.write(f"eps_tol {eps_tol:.7f}\n")
    f.write(f"ear_facets {int(ear_mask.sum())}\n")
    f.write(f"ear_bc_elements {int(ear_bc.sum())}\n")
    f.write(f"ear_in_patch_box_nodes {ear_in_box}\n")
    f.write(f"ear_node_hits {ear_node_hits}\n")
    f.write(f"bottom_facets {int(bottom_mask.sum())}\n")
    f.write(f"bottom_bc_elements {int(bottom_bc.sum())}\n")
    f.write(f"bottom_in_patch_box_nodes {bottom_in_box}\n")
    f.write(f"bottom_node_hits {bottom_node_hits}\n")
    f.write(f"bottom_fraction {BOTTOM_FRACTION}\n")
    f.write(f"seed_fraction {SEED_FRACTION}\n")
    f.write(f"trim_before {n_before}\n")
    f.write(f"trim_after {n_after}\n")
    f.write(f"trim_nodes {int(trim_mesh['nodes'].shape[0])}\n")
    f.write(f"domain_length {domain_length}\n")
    f.write(f"cube_half_side {H}\n")
    f.write(f"surf_extent {extent}\n")
    f.write(f"cells_across_model {extent / h_min:.2f}\n")
    f.write(f"sdf_min {float(node_sdf.min()):.5f}\n")
    f.write(f"sdf_max {float(node_sdf.max()):.5f}\n")
    f.write("ear_lo %.5f %.5f %.5f\n" % tuple(ear_v.amin(0).cpu().numpy()))
    f.write("ear_hi %.5f %.5f %.5f\n" % tuple(ear_v.amax(0).cpu().numpy()))
    f.write("bottom_lo %.5f %.5f %.5f\n" % tuple(bot_v.amin(0).cpu().numpy()))
    f.write("bottom_hi %.5f %.5f %.5f\n" % tuple(bot_v.amax(0).cpu().numpy()))
    f.write("surf_lo %.5f %.5f %.5f\n" % tuple(verts.amin(0).cpu().numpy()))
    f.write("surf_hi %.5f %.5f %.5f\n" % tuple(verts.amax(0).cpu().numpy()))
print("wrote dn_stats.txt")

# Warp's teardown segfaults in this env — exit 139 raised AFTER the line above
# has run and every product is on disk, so it is a shutdown crash, not a failed
# build.  A direct build leaves before interpreter finalization.  The overview
# deliberately executes this file with runpy to reuse the exact mesh and BC
# selector objects for its solve panel, so it must be allowed to continue.
if __name__ == "__main__":
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
