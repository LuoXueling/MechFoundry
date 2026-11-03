"""Prep step 2 (conda `mechfoundry-warp117`, GPU): run the production BC selectors on the bunny.

Reads the SAME geometry ``bunny_sdf_octree`` and ``diffused_nitsche`` show —
``bunny.stl`` and the cube half-side in ``bunny_sdf.meta``, both written by
``bunny/prep_bunny_sdf.py`` — and builds the octree with ``bunny_sdf_octree``'s
own generator and arguments, so all three figures print the same bunny and the
same mesh.  It used to read ``bc_bunny_geom.npz``, the TripoSG decode; nothing
in this figure needs the decoder, which illustrates the SELECTOR, not the
generative path.  See ``nitsche/dn_prep.py``'s WHICH BUNNY for the full
argument, including why the geometry is NOT rotated into some other frame.

On that surface it applies the real ``core.bc_surface_selectors`` primitives —
no re-implementation:

    seed  = HighestSelector('y', 0.05)              # coordinate inquiry: ear tips
    half  = HalfSpaceSelector(mid, axis)            # split the two tips apart
    ear_i = GrowSelectionSelector(                  # per ear, its OWN gate:
        FloodFillSelector(seed & half, angle_i)     #   grow by adjacency until
        & half, num_rings=2)                        #   the normal turns, clip
    ears  = ear_front | ear_back                    #   back, then close fjords

The two ears need different gates AND different ring counts: 3.0 deg / 6 rings on
the near ear, 2.4 deg / 2 rings on the far one — see FRONT_ANGLE below for why
those are not the 5.0 / 4.0 this figure used on the decode.

``ears.resolve(...)`` is the production call: it builds the Warp BVH of the
selected facets that decides the BC region.  That is where this figure stops — no octree, no FEM mesh.  It used to go one step
further and report which octree nodes the resolved selector accepts, the way
``identify_boundary_nodes`` does inside the solver, but nothing rendered those
node clouds and nothing quoted their counts, so the whole mesh went with them.

Writes, for the ParaView renderer:
  * ``bc_full.stl``      — the whole MC surface
  * ``bc_seed.stl``      — the seed facets            (panel a)
  * ``bc_ears.stl``      — the flood-filled facets    (panels b, c)
  * ``bc_stats.txt``     — the numbers quoted in the caption
"""

import os
import sys

import numpy as np
import torch
import trimesh

# Resolve the production package from THIS file, not from the cwd: the figures
# moved into per-figure subfolders (AGENTS.md "LAYOUT"), which silently broke
# the old cwd-relative "../src/warp_wrapper".
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(_REPO, "src", "warp_wrapper"))

from core.bc_surface_selectors import (
    HighestSelector,
    FloodFillSelector,
    GrowSelectionSelector,
    HalfSpaceSelector,
    _build_adj_pairs,
    _cc_by_label_prop,
    _compute_face_attributes,
)

SEED_FRACTION = 0.05        # top 5% of height -> the two ear tips

# The two ears need DIFFERENT flood-fill gates and are therefore handled as two
# separate selectors, split by a half-space through the midpoint of the two seed
# components.  A single global angle cannot serve both.
#
# A dihedral gate is a property of the TESSELLATION, not of the shape, so the
# 5.0 / 4.0 this figure ran on the decode's marching-cubes surface means nothing
# on prep_bunny_sdf's Loop-subdivided, Taubin-smoothed scan.  Each ear is set to
# the LARGEST gate that still stays on it — measured, by sweeping until the fill
# reaches the floor (min z = -0.493).  The front ear holds to 3.4 and bursts at
# 3.8 (11,182 -> 30,888 facets); the back holds to 4.6 and bursts at 5.0
# (12,290 -> 134,748).  Seed size does not move either cliff: at SEED_FRACTION
# 0.05, 0.10 and 0.16 both break at the same angle.
# NOTE the order against dn_prep.py's 3.4 / 4.6.  The gates belong to the two
# PHYSICAL ears, but FRONT/BACK name whichever is nearer the CAMERA, and MIRROR
# above swaps that.  Same two ears, same two gates, labels exchanged; if the
# mirror ever goes, these swap back.  A leak announces itself — the burst ear
# jumps to ~35k facets and the component count goes from 2 to 14.
FRONT_ANGLE = 4.6           # deg, ear nearer the camera
BACK_ANGLE = 3.4            # deg
# Rings then close what the gate cannot cross, and they are what actually fills
# the ears: at the angles above, 6 rings leave 87 % / 78 % of the ear covered,
# 14 rings reach 95 % / 82 %.  Past that they stop helping and start creeping
# down the neck instead — 30 rings buys 2 more points of the front ear while its
# reach drops from z = 0.197 to 0.141, well below the ear base at z ~ 0.21.
FRONT_RINGS = 14
BACK_RINGS = 14
# prep_bunny_sdf's frame, untransformed: z-up, and viewed from where
# bunny_sdf_octree and diffused_nitsche view it, so all three show the bunny at
# the same angle.  VIEW_DIR only names which ear is nearer the camera here; the
# renderer holds the real camera.
UP = "z"
VIEW_DIR = np.array([1.5, -3.0, 1.25])   # == render_bc_bunny / render_dn
# Camera and angle are diffused_nitsche's, but that camera looks at this bunny's
# BACK, and this figure has always shown its front — the ear cups have to face
# the reader or panels (b) and (c) show two blank ear backs.  MIRROR turns the
# animal around under the fixed camera.  It is free here: a reflection changes
# nothing a surface selector measures (heights, dihedral angles, adjacency), and
# unlike diffused_nitsche there is no octree to perturb.  Faces are re-wound
# because a reflection reverses orientation and the selectors read NORMALS.
MIRROR = np.array([1.0, -1.0, 1.0])

device = "cuda"
# The bunny is prep_bunny_sdf's, the one bunny_sdf_octree and diffused_nitsche
# show.  No octree: this figure is about the SELECTOR, and everything it draws —
# seed facets, flood-filled facets, the close-up of them — comes off the
# surface.  The only thing the mesh ever contributed was its element size, and
# that is a scalar, prep_octree_sdf's cube over its 2^6 leaves.
H = float(open("bunny_sdf.meta").readline())         # == prep_octree_sdf.py:26
bunny = trimesh.load("bunny.stl")                    # == prep_octree_sdf.py:27
verts = torch.tensor(np.ascontiguousarray(bunny.vertices * MIRROR),
                     dtype=torch.float32, device=device)
faces = torch.tensor(np.ascontiguousarray(bunny.faces[:, [0, 2, 1]]),
                     dtype=torch.long, device=device)
min_elem = 2.0 * H / 2 ** 6                          # == dn_prep's h_min

eps_tol = 0.75 * min_elem      # solver _bc_match_tolerance, max(1.5 eps, 0.75 h), at eps = h/2
body_diag = float(torch.linalg.norm(verts.amax(0) - verts.amin(0)))
print(f"surface {verts.shape[0]}V {faces.shape[0]}F")
print(f"eps_tol={eps_tol:.5f}  body_diag={body_diag:.3f}")

# ---------------------------------------------------------------------------
# 1. The selector recipe (production primitives)
# ---------------------------------------------------------------------------
seed_sel = HighestSelector(UP, fraction=SEED_FRACTION)

centroids, normals, areas = _compute_face_attributes(verts, faces)
adj = _build_adj_pairs(faces)
seed_mask = seed_sel.select(verts, faces, centroids, normals, areas, None, adj_pairs=adj)

# Split the two ear tips with a half-space through their midpoint, oriented along
# the line joining them — derived from the seed, not hard-coded.
seed_comps = sorted(_cc_by_label_prop(seed_mask, adj), key=len, reverse=True)
assert len(seed_comps) == 2, f"expected 2 ear tips, got {len(seed_comps)}"
tip_c = [centroids[c].mean(0).cpu().numpy() for c in seed_comps]
axis = tip_c[0] - tip_c[1]
axis /= np.linalg.norm(axis)
mid = 0.5 * (tip_c[0] + tip_c[1])
# Name the halves by which ear is nearer the camera
near_first = float(tip_c[0] @ VIEW_DIR) > float(tip_c[1] @ VIEW_DIR)
front_half = HalfSpaceSelector(mid, axis if near_first else -axis)
back_half = HalfSpaceSelector(mid, -axis if near_first else axis)
print(f"ear tips at {np.round(tip_c[0],3)} / {np.round(tip_c[1],3)}; "
      f"split normal {np.round(axis,3)} through {np.round(mid,3)}")


def one_ear(half, angle, rings):
    """Seed on this side, flood-fill at its own gate, clip back to this side, grow."""
    return GrowSelectionSelector(
        FloodFillSelector(seed_sel & half, angle) & half,
        num_rings=rings,
    )


# The 3-4 deg gate sits near this surface's noise floor (the median facet's
# largest dihedral to a neighbour is 2.9 deg), so isolated noisy-normal facets
# block the BFS and leave thin unselected fjords; two rings of growth close them.
front_sel = one_ear(front_half, FRONT_ANGLE, FRONT_RINGS)
back_sel = one_ear(back_half, BACK_ANGLE, BACK_RINGS)
ear_sel = front_sel | back_sel

front_mask = front_sel.select(verts, faces, centroids, normals, areas, None, adj_pairs=adj)
back_mask = back_sel.select(verts, faces, centroids, normals, areas, None, adj_pairs=adj)
ear_mask = ear_sel.select(verts, faces, centroids, normals, areas, None, adj_pairs=adj)
print(f"front ear ({FRONT_ANGLE} deg): {int(front_mask.sum())} facets, "
      f"area {float(areas[front_mask].sum()):.4f}")
print(f"back  ear ({BACK_ANGLE} deg): {int(back_mask.sum())} facets, "
      f"area {float(areas[back_mask].sum()):.4f}")
print(f"seed: {int(seed_mask.sum())} facets, "
      f"{len(_cc_by_label_prop(seed_mask, adj))} components")
print(f"ears: {int(ear_mask.sum())} facets, "
      f"{len(_cc_by_label_prop(ear_mask, adj))} components")

# ---------------------------------------------------------------------------
# 2. Production resolve -> BVH of the selected facets
# ---------------------------------------------------------------------------
ear_sel.resolve(verts, faces, eps_expansion=eps_tol)
print(f"resolved (BVH: {'built' if ear_sel._wp_mesh is not None else 'MISSING'})")

# ---------------------------------------------------------------------------
# 4. Export for ParaView
# ---------------------------------------------------------------------------
vnp = verts.cpu().numpy()
fnp = faces.cpu().numpy()


def export_faces(mask, path):
    sub = fnp if mask is None else fnp[mask.cpu().numpy()]
    trimesh.Trimesh(vertices=vnp, faces=sub, process=False).export(path)
    print(f"wrote {path} ({len(sub)} faces)")


export_faces(None, "bc_full.stl")
export_faces(seed_mask, "bc_seed.stl")
export_faces(ear_mask, "bc_ears.stl")
# Body layers for panels (a)/(b) = the COMPLEMENT of the highlighted set.
# Drawing the whole surface and the selected subset on top makes the selected
# facets coincident geometry, and the z-fighting speckles the body colour
# through the highlight.  Disjoint meshes render cleanly.
export_faces(~seed_mask, "bc_rest_seed.stl")
export_faces(~ear_mask, "bc_rest_ears.stl")

# ---------------------------------------------------------------------------
# 5. Close-up target and camera direction, derived from the selection itself
# ---------------------------------------------------------------------------
# Best viewing direction = area-weighted mean normal of the selected facets, so
# the camera looks straight into the concave inner ear surface.
w = areas[ear_mask].unsqueeze(1)
view_dir = (normals[ear_mask] * w).sum(0)
view_dir = (view_dir / view_dir.norm()).cpu().numpy()

# Close up on the ear NEAREST the camera, so the reader can match it to (a)/(b).
# Use THAT ear's own selector: filtering the combined result by the ear's
# bounding box instead would drag in part of the far ear.
focus_sel, focus_mask = front_sel, front_mask

# Look straight into that ear's selected surface
fw = areas[focus_mask].unsqueeze(1)
focus_dir = (normals[focus_mask] * fw).sum(0)
focus_dir = (focus_dir / focus_dir.norm()).cpu().numpy()
fverts = verts[faces[focus_mask].reshape(-1)]
focus_lo = fverts.amin(0).cpu().numpy()
focus_hi = fverts.amax(0).cpu().numpy()
focus = 0.5 * (focus_lo + focus_hi)
focus_r = float(np.linalg.norm(focus_hi - focus_lo)) * 0.5

export_faces(focus_mask, "bc_ear_focus.stl")

# Local context for the close-up: a thin COLLAR of unselected facets around the
# patch, so any gap inside it is occluded rather than showing background.
# Taking every unselected facet in the ear's bounding box instead drags in a
# strip of head/neck that hangs out of frame as a grey tail with a black
# back-facing edge.
collar = GrowSelectionSelector(focus_sel, num_rings=10).select(
    verts, faces, centroids, normals, areas, None, adj_pairs=adj)
export_faces(collar & ~focus_mask, "bc_ear_context.stl")

# ---------------------------------------------------------------------------
# 6. Panel (c) shows BOTH ears: same collar trick over the full selection, and
#    the framing centre/radius of the two ears together.
# ---------------------------------------------------------------------------
# The collar is tagged with its ring distance from the selection so the renderer
# can fade it out.  A hard-clipped collar ends on a ragged facet boundary that
# reads as a zigzag tear; ramping colour to white and opacity to 0 over the
# outer rings dissolves that edge into the background instead.
from core.bc_surface_selectors import _bfs_expand   # noqa: E402

COLLAR_RINGS = 30
ring_id = torch.zeros(faces.shape[0], dtype=torch.long, device=device)
prev = ear_mask.clone()
for r in range(1, COLLAR_RINGS + 1):
    cur = _bfs_expand(ear_mask, adj, r)
    ring_id[cur & ~prev] = r
    prev = cur
ears_collar = prev & ~ear_mask
fade = (ring_id.float() - 1.0) / max(COLLAR_RINGS - 1, 1)   # 0 at the patch, 1 outside


def write_faces_scalar(mask, values, path):
    """Legacy ASCII VTK POLYDATA of the masked faces + a per-cell scalar."""
    idx = mask.nonzero(as_tuple=True)[0].cpu().numpy()
    tri = fnp[idx]
    val = values.cpu().numpy()[idx]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nbc context\nASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {len(vnp)} float\n")
        for q in vnp:
            f.write(f"{q[0]} {q[1]} {q[2]}\n")
        f.write(f"POLYGONS {len(tri)} {4 * len(tri)}\n")
        for t in tri:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")
        f.write(f"CELL_DATA {len(tri)}\nSCALARS fade float 1\nLOOKUP_TABLE default\n")
        for v in val:
            f.write(f"{v}\n")
    print(f"wrote {path} ({len(tri)} faces, fade 0..1)")


write_faces_scalar(ears_collar, fade, "bc_ears_context.vtk")

everts = verts[faces[ear_mask].reshape(-1)]
ears_lo = everts.amin(0).cpu().numpy()
ears_hi = everts.amax(0).cpu().numpy()
ears_center = 0.5 * (ears_lo + ears_hi)
ears_radius = float(np.linalg.norm(ears_hi - ears_lo)) * 0.5
print(f"both ears: centre {np.round(ears_center,3)}  radius {ears_radius:.3f}")

with open("bc_stats.txt", "w") as f:
    f.write(f"seed_facets {int(seed_mask.sum())}\n")
    f.write(f"seed_components {len(_cc_by_label_prop(seed_mask, adj))}\n")
    f.write(f"ear_facets {int(ear_mask.sum())}\n")
    f.write(f"ear_components {len(_cc_by_label_prop(ear_mask, adj))}\n")
    f.write(f"surface_facets {faces.shape[0]}\n")
    f.write(f"eps_tol {eps_tol:.6f}\n")
    f.write(f"seed_fraction {SEED_FRACTION}\n")
    f.write(f"front_angle {FRONT_ANGLE}\n")
    f.write(f"back_angle {BACK_ANGLE}\n")
    f.write(f"front_rings {FRONT_RINGS}\n")
    f.write(f"back_rings {BACK_RINGS}\n")
    f.write(f"front_facets {int(front_mask.sum())}\n")
    f.write(f"back_facets {int(back_mask.sum())}\n")
    f.write(f"focus {focus[0]:.5f} {focus[1]:.5f} {focus[2]:.5f}\n")
    f.write(f"focus_radius {focus_r:.5f}\n")
    f.write(f"view_dir {view_dir[0]:.5f} {view_dir[1]:.5f} {view_dir[2]:.5f}\n")
    f.write(f"focus_dir {focus_dir[0]:.5f} {focus_dir[1]:.5f} {focus_dir[2]:.5f}\n")
    f.write(f"ears_center {ears_center[0]:.5f} {ears_center[1]:.5f} "
            f"{ears_center[2]:.5f}\n")
    f.write(f"ears_radius {ears_radius:.5f}\n")
print("wrote bc_stats.txt")
