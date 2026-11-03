"""Mesh post-processing shared by the CAD generators.

A CAD generator ends with a B-rep tessellated face by face.  Everything after
that -- welding the faces into one closed surface, keeping the largest part,
refining slivers to a face budget, normalising into the [-1, 1] box a TripoSG
surface occupies, and saving which B-rep face each triangle came from -- is the
same whatever built the solid, and lives here.
"""

from __future__ import annotations

import math
from pathlib import Path

# Largest half-extent of the normalised surface, inside TripoSG's +-1.005 box.
HALF_EXTENT = 0.95
# Face budget used to bound edge lengths when the caller leaves --faces at -1.
DEFAULT_FACES = 20000
# Decimal places at which tessellated vertices of a unit-sized part are welded.
WELD_DIGITS = 6
# CAD parts are modelled z-up; the workflow's frame is y-up (TripoSG's), and every
# later stage -- "downward is -y" for loads, gravity, which face is the top --
# reads directions in it.  This proper rotation (-90 degrees about x) takes CAD
# +z to +y and CAD +y to -z, keeping the part right-handed and its normals out.
CAD_TO_WORKFLOW = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
# OCC's GeomAbs_SurfaceType, in enum order.
SURFACE_TYPES = (
    "plane", "cylinder", "cone", "sphere", "torus", "bezier", "bspline",
    "revolution", "extrusion", "offset", "other",
)


def face_labels_path(surface: Path) -> Path:
    """Where a generated surface's B-rep face labels are saved, beside it."""
    surface = Path(surface)
    return surface.with_suffix(surface.suffix + ".faces.npz")


def finish_mesh(vertices, triangles, labels, source: str = "CAD model"):
    """Weld per-face tessellations into one closed mesh, keeping face labels.

    ``labels`` is the B-rep face of each triangle.  Returns the mesh, the labels
    of the triangles kept, and how many loose components were dropped.
    """
    import numpy as np
    import trimesh

    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if not len(triangles):
        raise ValueError(f"{source} produced no surface")
    # Each face carries its own copy of the nodes on its edges; weld them at a
    # tolerance relative to the part's size, in coordinates of order one.
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    vertices = (vertices - (lo + hi) / 2.0) / max(float((hi - lo).max()), 1e-12)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    mesh.merge_vertices(digits_vertex=WELD_DIGITS)
    keep = mesh.nondegenerate_faces()
    mesh.update_faces(keep)
    labels = labels[keep]
    # The largest component, chosen by face so the labels follow it.
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1
    )
    discarded = max(len(components) - 1, 0)
    if discarded:
        largest = max(components, key=lambda part: abs(mesh.submesh([part], append=True).volume))
        keep = np.zeros(len(mesh.faces), dtype=bool)
        keep[largest] = True
        mesh.update_faces(keep)
        labels = labels[keep]
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        raise ValueError(f"{source} did not produce a closed solid")
    # Flips triangles in place; their order, and so the labels, is unchanged.
    mesh.fix_normals()
    if mesh.volume <= 0:
        raise ValueError(f"{source} produced an empty solid")
    return mesh, labels, discarded



def refine_to_size(vertices, faces, max_edge: float, max_iter: int = 10):
    """Split edges longer than ``max_edge`` until none are, keeping the mesh closed.

    Each long edge is split once, at a midpoint its two faces share, and every
    face is re-triangulated by how many of its edges were split.  Splitting
    faces independently -- as trimesh's ``subdivide_to_size`` does before
    trimesh 5 -- leaves T-junctions wherever two neighbours disagree, which
    opens the surface.  Returns the vertices, faces, and each new face's parent.
    """
    import numpy as np

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    parents = np.arange(len(faces))
    for _ in range(max_iter):
        edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 3, 2), axis=2)
        unique, inverse = np.unique(edges.reshape(-1, 2), axis=0, return_inverse=True)
        inverse = inverse.reshape(-1, 3)
        lengths = np.linalg.norm(vertices[unique[:, 0]] - vertices[unique[:, 1]], axis=1)
        split = lengths > max_edge
        if not split.any():
            break
        midpoint = np.full(len(unique), -1, dtype=np.int64)
        midpoint[split] = len(vertices) + np.arange(split.sum())
        vertices = np.vstack([vertices, vertices[unique[split]].mean(axis=1)])
        new_faces, new_parents = [], []
        for (a, b, c), (m0, m1, m2), parent in zip(faces, midpoint[inverse], parents):
            # Edges in order: (a, b) -> m0, (b, c) -> m1, (c, a) -> m2.
            count = int(m0 >= 0) + int(m1 >= 0) + int(m2 >= 0)
            if count == 0:
                pieces = [(a, b, c)]
            elif count == 3:
                pieces = [(a, m0, m2), (m0, b, m1), (m2, m1, c), (m0, m1, m2)]
            else:
                # Rotate so the split pattern starts at edge (a, b).
                corners, mids = [a, b, c], [m0, m1, m2]
                while mids[0] < 0 or (count == 2 and mids[2] >= 0):
                    corners, mids = corners[1:] + corners[:1], mids[1:] + mids[:1]
                (a, b, c), (m0, m1, _) = corners, mids
                pieces = ([(a, m0, c), (m0, b, c)] if count == 1
                          else [(m0, b, m1), (a, m0, m1), (a, m1, c)])
            new_faces.extend(pieces)
            new_parents.extend([parent] * len(pieces))
        faces = np.asarray(new_faces, dtype=np.int64)
        parents = np.asarray(new_parents, dtype=np.int64)
    return vertices, faces, parents


def normalize(mesh, faces: int = -1, labels=None):
    """Stand the part up in the workflow's frame, centre and scale it into
    [-HALF_EXTENT, HALF_EXTENT], and bound edge lengths.

    Tessellated CAD faces are long slivers; splitting them keeps segmentation
    rays and SDF queries from seeing a handful of huge triangles.  Per-face
    ``labels`` follow each triangle into the pieces it is split into.
    """
    import numpy as np
    import trimesh

    mesh = mesh.copy()
    mesh.apply_transform(np.asarray(CAD_TO_WORKFLOW))
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2.0)
    mesh.apply_scale(HALF_EXTENT / (float((hi - lo).max()) / 2.0))
    labels = np.zeros(len(mesh.faces), dtype=np.int64) if labels is None else np.asarray(labels)
    budget = DEFAULT_FACES if faces == -1 else faces
    if len(mesh.faces) >= budget:
        return mesh, labels
    # The edge an equilateral mesh of `budget` faces would have.  Slivers split
    # into more pieces than that estimate, so lengthen it until the budget holds.
    edge = math.sqrt(4.0 * mesh.area / (math.sqrt(3.0) * budget))
    for _ in range(12):
        vertices, triangles, parents = refine_to_size(mesh.vertices, mesh.faces, edge)
        if len(triangles) <= budget:
            break
        edge *= 1.25
    else:
        return mesh, labels
    return trimesh.Trimesh(vertices, triangles, process=False), labels[parents]


def write_face_labels(path: Path, labels, surface_types) -> None:
    """Save the B-rep face of each triangle, in the surface's face order."""
    import numpy as np

    np.savez_compressed(
        path, face_ids=np.asarray(labels, dtype=np.int32),
        surface_types=np.asarray(surface_types, dtype=str),
    )
