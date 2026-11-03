"""Write the exact surface PartField and the renders both consume.

``../bunny/prep_bunny_sdf.py`` emits the Loop-subdivided, Taubin-smoothed
``bunny.stl`` that every bunny figure shows.  ``bcautobunny`` mirrors it and
repairs the winding in memory before selection; PartField instead keys its
feature cache on a *file*, so the mirrored surface has to exist on disk.

STL is a triangle soup -- it carries no shared vertices, so a mesh read back
from one has no facet adjacency and no smooth normals.  The welded topology is
therefore written beside it as ``pf_topology.npz``.  ``Trimesh.export`` emits
triangles in ``mesh.faces`` order, so facet *i* of the STL is facet *i* of the
welded mesh, and PartField's per-face features index straight into it.
"""

import numpy as np
import trimesh

MIRROR = np.array([1.0, -1.0, 1.0])  # identical to bcregion / bcautobunny

bunny = trimesh.load("bunny.stl")
# Reflection reverses orientation; this is exactly bcautobunny's winding repair.
surface = trimesh.Trimesh(
    vertices=np.ascontiguousarray(bunny.vertices * MIRROR),
    faces=np.ascontiguousarray(bunny.faces[:, [0, 2, 1]]),
    process=False,
)
surface.export("pf_surface.stl")
np.savez_compressed(
    "pf_topology.npz",
    vertices=np.asarray(surface.vertices, dtype=np.float64),
    faces=np.asarray(surface.faces, dtype=np.int64),
)

# The correspondence above is load-bearing: PartField reads the STL with
# process=False and returns one feature per soup facet, which pf_prep.py then
# indexes against the welded facets.  Check it rather than trust it.
soup = trimesh.load("pf_surface.stl", force="mesh", process=False)
assert len(soup.faces) == len(surface.faces), "STL round trip changed facet count"
drift = np.abs(soup.triangles.mean(axis=1) - surface.triangles.mean(axis=1)).max()
assert drift < 1e-5, f"STL facet order does not match the welded mesh ({drift:g})"
print(f"wrote pf_surface.stl + pf_topology.npz: {len(surface.vertices)} vertices, "
      f"{len(surface.faces)} faces (facet-order drift {drift:.2g})")
