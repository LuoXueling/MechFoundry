"""Write the CAD lane's CadQuery script from the real bunny (env mechfoundry-warp117, CPU).

    python ov_cad_fit.py            # -> assets/cad/bunny_cadcoder.py

The CAD-Coder-style script rebuilds the Stanford bunny the way a CAD modeller
reverse-engineers a scan: lofts through measured sections, in many patches.

- **Body**: the display bunny's interior, voxelised; each voxel labelled with
  the PartField part (the overview's k = 8 run) of its nearest surface facet, and
  the ear parts left out.  Slabs across its length (x, snout to tail): each
  slab's largest connected region, blurred a little and contoured at 0.5, is
  its REAL outline (not a hull: the hull turned head, chest and paws into one
  block), smoothed by keeping its first HARMONICS Fourier terms, resampled into
  N_PTS points from the top, counter-clockwise seen from +x, so neighbouring
  sections correspond and the loft does not twist.  Both ends close through
  shrinking copies on an elliptical profile (no flat caps, no spikes).
- **Ears**: each ear part, sliced along its principal axis; every slice becomes
  the ellipse with its second moments (a loft through the thin hull slices was
  an invalid solid however sampled; ellipses loft cleanly), the first pushed
  into the head so the ear has a root, a last, shrunk one closing the tip.
- **Patches**: body and ears are each ONE smooth loft; flat sheets then split
  the loft's SHELL (a splitter on the surface, no Boolean fuse) into a grid of
  B-spline patches: rings across the body (none through an ear root, where the
  ear's fusion would fail), waterlines, lines along the flanks, rings on the
  ears.  The solid is rebuilt from the split shell and never clean()ed, so the
  grid is the model's B-rep edges.  (Lofting in runs and gluing them left
  cracks wherever one run's cut was refused.)

History: moment ellipsoids per k-means cluster (IoU 0.83) read as a balloon
animal; a loft through convex-hull sections with ellipsoid ears (IoU 0.88)
lost the head, the chin and the ears' shape ("far off").
"""

import os

import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure

import ov_manifest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "cad", "bunny_cadcoder.py")
K = 8                          # the overview's PartField run
EARS = (5, 7)                  # its ear parts (ov_loop.json: neumann_parts)
PITCH = 1.0 / 150.0            # voxel size; the bunny's extent is ~1
# the rings' axis, flank to flank: the flanks face it squarely, so the end rings
# are nearly planar.  (Tilted so both poles faced away from the camera, the
# build broke in every variant tried.)
POLE = np.array([0.0, 1.0, 0.0])
N_RINGS_T = 48                 # latitude rings (the loft's sections), flank to flank
N_PTS = 64                     # points per ring (the same longitudes on every ring)
THETA0 = 0.025                 # the first / last ring's polar angle (rad) off the pole:
                               # the loft's flat end cap there is ~1 px in the figure
                               # (at 0.06-0.22 rad it showed as a disc or, domed, a knob)
R_SMOOTH = 0.5                 # Gaussian blur of the radius grid, in grid steps
N_RINGS = 12                   # patch lines across the body (short ones under the ears)
LEVELS = 4                     # waterlines up its height
SIDE_Y = (-0.05,)              # a line along the rabbit, down its middle (flank lines
                               # ran beside a waterline on the near flank: doubled lines)
EAR_SECTIONS, EAR_RINGS = 9, 3


def resample(poly, n):
    """n points evenly along a closed polygon, by arc length."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    t = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(t, s, closed[:, k]) for k in range(2)], 1)


def outline(mask, origin, n):
    """The mask's outer outline in (y, z): contoured, Fourier-smoothed, n points
    from the top, counter-clockwise (y right, z up); and its area."""
    blur = ndimage.gaussian_filter(mask.astype(float), 1.0)
    contour = max(measure.find_contours(np.pad(blur, 2), 0.5), key=len) - 2.0
    yz = origin + contour * PITCH                    # rows -> y, columns -> z
    area = 0.5 * np.sum(yz[:, 0] * np.roll(yz[:, 1], -1) - np.roll(yz[:, 0], -1) * yz[:, 1])
    if area < 0:
        yz = yz[::-1]
    loop = resample(yz, 256)
    spec = np.fft.fft(loop[:, 0] + 1j * loop[:, 1])
    keep = np.zeros_like(spec)
    keep[:HARMONICS + 1], keep[-HARMONICS:] = spec[:HARMONICS + 1], spec[-HARMONICS:]
    smooth = np.fft.ifft(keep)
    loop = resample(np.c_[smooth.real, smooth.imag], 256)
    loop = np.roll(loop, -int(np.argmax(loop[:, 1])), axis=0)
    return resample(loop, n), abs(area)


bunny = trimesh.load(ov_manifest.asset("bunny_display.stl"))
labels = np.load(ov_manifest.asset(f"seg/pf_k{K}.npz"))["segment_ids"].astype(int)
vox = bunny.voxelized(PITCH).fill()
pts = vox.points
part = labels[cKDTree(bunny.triangles_center).query(pts)[1]]
body = pts[~np.isin(part, EARS)]
ijk = np.round(body / PITCH).astype(int)
lo = ijk.min(0)
grid = np.zeros(ijk.max(0) - lo + 1, bool)
grid[tuple((ijk - lo).T)] = True

# -- body: latitude rings of its radius function ------------------------------------
# The ear-less body is star-shaped from its centroid to 99 % (1 % of rays cross
# it twice; the farthest crossing is taken, bridging those few notches).  Rays
# on a (theta, phi) grid about the y axis give its radius r(theta, phi); each
# theta is a closed ring of sections at the SAME longitudes, so neighbouring
# rings correspond exactly.  The axis runs flank to flank: the flanks face it
# squarely, so the rings there are nearly planar and the loft's end caps (which
# must be planar) cost nothing; head, tail, feet and ear roots all lie between.
G = ndimage.gaussian_filter(grid.astype(float), 1.0)
centre = body.mean(0)
u_ax = np.array([1.0, 0.0, 0.0])                     # perpendicular to POLE
w_ax = np.cross(POLE, u_ax)
theta = np.linspace(THETA0, np.pi - THETA0, N_RINGS_T)
phi = np.linspace(0.0, 2.0 * np.pi, N_PTS, endpoint=False)
T, F = np.meshgrid(theta, phi, indexing="ij")
D = (np.cos(T)[..., None] * POLE
     + np.sin(T)[..., None] * (np.cos(F)[..., None] * u_ax + np.sin(F)[..., None] * w_ax))
steps = np.linspace(0.0, 0.9, 450)
probe = centre + D[..., None, :] * steps[:, None]
occ = ndimage.map_coordinates(G, (probe / PITCH - lo).reshape(-1, 3).T, order=1,
                              mode="constant").reshape(probe.shape[:-1]) >= 0.5
radius = steps[np.where(occ.any(-1), occ.shape[-1] - 1 - np.argmax(occ[..., ::-1], -1), 0)]
radius = ndimage.gaussian_filter(radius, R_SMOOTH, mode=("nearest", "wrap"))
rings = centre + D * radius[..., None]
# the end rings flattened onto their mean planes (the loft caps them flat);
# flattening more rings dented the flank around the cap
for depth, weight in enumerate((1.0,)):
    for k in (depth, -1 - depth):
        h = rings[k] @ POLE
        rings[k] = rings[k] + weight * np.outer(h.mean() - h, POLE)
sections = [rings[k] for k in range(N_RINGS_T)]
# waterlines evenly spaced up the body, none within 0.07 of a pole's height:
# two passing either side of the near pole read as a doubled line by the leg
zs = np.concatenate(sections)[:, 2]
pole_z = [float(sections[k][:, 2].mean()) for k in (0, -1)]
cands = np.linspace(zs.min() + 0.12, zs.max() - 0.2, LEVELS + 2)
LEVEL_Z = tuple(round(float(z), 3) for z in cands if all(abs(z - pz) >= 0.07 for pz in pole_z))
print(f"body: {len(sections)} rings x {N_PTS} points about y, waterlines {LEVEL_Z}")

# -- ears: moment ellipses of slices along each ear's axis -------------------------
ears = []
head_c = pts[part == 1].mean(0)
for ear in EARS:
    ep = pts[part == ear]
    c = ep.mean(0)
    axis = np.linalg.eigh(np.cov((ep - c).T))[1][:, -1]
    if np.dot(axis, c - head_c) < 0:
        axis = -axis
    t = (ep - c) @ axis
    stops = np.linspace(np.percentile(t, 1), np.percentile(t, 99), EAR_SECTIONS)
    half = 0.6 * (stops[1] - stops[0])
    loops, prev = [], None
    for k, t0 in enumerate(stops):
        sl = ep[np.abs(t - t0) <= half]
        sc = sl.mean(0)
        sc = sc - axis * (np.dot(sc - c, axis) - t0)            # onto the station's plane
        d = (sl - sc) - np.outer((sl - sc) @ axis, axis)
        lam, vec = np.linalg.eigh(np.cov(d.T))
        major = vec[:, -1]
        if prev is not None and np.dot(major, prev) < 0:        # no flips between sections
            major = -major
        prev = major
        ra = 2.0 * np.sqrt(lam[-1]) + PITCH / 2
        rb = max(2.0 * np.sqrt(lam[-2]), 0.006) + PITCH / 2
        if k == 0:                                             # root: into the head
            sc = sc - axis * 0.05
        loops.append((sc, axis, major, ra, rb))
    sc, ax, mj, ra, rb = loops[-1]
    loops.append((sc + axis * 0.02, ax, mj, ra * 0.45, rb * 0.6))
    ears.append(loops)
print(f"ears: {len(ears)} x {len(ears[0])} ellipse sections")

# the patch grid: rings across the body, waterlines, lines along it
all_x = np.concatenate(sections)[:, 0]
x_span = (float(all_x.min()) + 0.05, float(all_x.max()) - 0.05)
# every sheet is a FULL plane, crossing the whole surface: a sheet that ended
# inside it (rings stopped under the ear roots, flank and face lines stopped
# short) left sliver faces the mesher could not close
RINGS = tuple(round(float(x), 4) for x in np.linspace(*x_span, N_RINGS))
print(f"{len(RINGS)} rings, {len(LEVEL_Z)} waterlines, {len(SIDE_Y)} lines along")

ear_rings = []
for loops in ears:
    t0, t1 = 1, len(loops) - 2
    ear_rings.append([loops[int(k)][0] for k in np.linspace(t0, t1, EAR_RINGS + 2)[1:-1].round()])


def fmt(v):
    return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"


def table(loops):
    return "\n".join("    [" + ", ".join(fmt(p) for p in loop) + "]," for loop in loops)


ear_rows = []
for loops in ears:
    ear_rows.append("    [\n" + "\n".join(
        f"        ({fmt(sc)}, {fmt(ax)}, {fmt(mj)}, {ra:.4f}, {rb:.4f}),"
        for sc, ax, mj, ra, rb in loops) + "\n    ],")
ear_ring_rows = "\n".join("    [" + ", ".join(fmt(c) for c in rings) + "]," for rings in ear_rings)

script = f'''# A sitting rabbit, written the way CAD-Coder answers: one CadQuery script
# that assigns its model to r (cadquery only; run by the workflow's sandbox,
# src/cad_wrapper/cadcoder_wrapper/sandbox.py).
#
# Reverse-engineered from the Stanford bunny by
# illustrations/overview/ov_cad_fit.py: {len(sections)} body cross-sections
# (snout to tail, {N_PTS} points each, their real outlines, smoothed) and each
# ear as {len(ears[0])} ellipses along its axis.  Body and ears are smooth lofts;
# planes then SPLIT their surfaces into a grid of B-spline patches (rings,
# waterlines and flank lines; rings on the ears), and the ears are fused on.
# Never clean()ed, so the patch grid is the model's B-rep edges.
import cadquery as cq

BODY = [
{table(sections)}
]
RINGS = {RINGS}                # x of each ring across the body
LEVELS = {LEVEL_Z}             # z of each waterline
FLANKS = {SIDE_Y}              # y of each line along the rabbit
# ears: per section (centre, axis, major direction, major radius, minor radius)
EARS = [
{chr(10).join(ear_rows)}
]
EAR_RINGS = [                  # a point on each ring of each ear
{ear_ring_rows}
]


def section(loop):
    edge = cq.Edge.makeSpline([cq.Vector(*p) for p in loop], periodic=True)
    return cq.Wire.assembleEdges([edge])


def ellipse(row):
    centre, axis, major, ra, rb = row
    return cq.Wire.makeEllipse(ra, rb, cq.Vector(*centre), cq.Vector(*axis), cq.Vector(*major))


def sheet(corners):
    """A flat cutting sheet through four corner points."""
    return cq.Face.makeFromWires(cq.Wire.makePolygon([cq.Vector(*c) for c in corners], close=True))


def patched(solid, sheets):
    """The solid, its surface split along the sheets into patches (all in one
    splitter call: one at a time cost three times the CPU)."""
    shell = solid.Shells()[0].split(*sheets).Shells()[0]
    return cq.Solid.makeSolid(shell)


body = cq.Solid.makeLoft([section(loop) for loop in BODY], ruled=False)
sheets = [sheet([(x, -2, -2), (x, 2, -2), (x, 2, 2), (x, -2, 2)]) for x in RINGS]
sheets += [sheet([(-2, -2, z), (2, -2, z), (2, 2, z), (-2, 2, z)]) for z in LEVELS]
sheets += [sheet([(-2, y, -2), (2, y, -2), (2, y, 2), (-2, y, 2)]) for y in FLANKS]
# ears first, onto the smooth unsplit body (a fuse onto patch edges failed
# wherever an edge met an ear's root), then the patch grid on the whole rabbit
for ear in EARS:
    body = body.fuse(cq.Solid.makeLoft([ellipse(row) for row in ear], ruled=False))
ear_cuts = []
for ear, rings in zip(EARS, EAR_RINGS):
    axis = cq.Vector(*ear[0][1])
    ear_cuts += [cq.Face.makePlane(0.3, 0.3, cq.Vector(*c), axis) for c in rings]
rabbit = patched(body, sheets + ear_cuts)
r = rabbit
'''
with open(OUT, "w") as f:
    f.write(script)
print(f"wrote {OUT}")
