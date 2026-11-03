"""Shared fixtures for the operator-skill tests: a fast in-process stand-in for the MeshLab remesh.

``drive.build_blank`` always ends in ``drive.remesh_built``, which starts the compute
environment's interpreter on ``remesh_blank.py`` (pymeshlab isotropic remeshing, 12
iterations).  Measured on this suite: 195 tests took 276 s, and the five slowest (59, 49, 43,
23, 21 s) were the ``build_blank`` tests; one 48-sided 0.9 m cylinder alone remeshes in 14-21 s,
most of it MeshLab on the long faces.  Almost no test is about the remesh itself: they read the
built body's watertightness, extents, volume, bores, sections and wall thicknesses, all of which
the body's planar faces fix whatever the triangles on them are.

So every test gets ``_remesh_facets`` in place of ``remesh_built``, with the same contract: the
body and an ``edge`` defaulting to ``drive.blank_edge`` in, a watertight trimesh with faces of
about that edge on the same surface out, ``metadata["edge"]`` set.  Each planar facet (a built
body is all planar facets: box sides, the 48 strips of a cylinder, a profile's caps and side
quads) is re-triangulated in its own plane: every crease edge is sampled at the target spacing
once, so the two facets on it share the points and the body stays watertight; the interior is a
hex lattice of that spacing; a 2-D Delaunay joins them.  Every vertex lies on the original
facets, so extents and volume are exact.  On the bodies the tests build it gives 1.0x the target
median edge, 1st-percentile minimum angles of 21-38 degrees and 1-2x MeshLab's face count (the
cylinders 2x: MeshLab merges across their 7.5-degree strips), in 0.06-0.6 s a body.  Plain
midpoint subdivision was tried first and rejected: it keeps every sliver's shape (minimum angle
0.1-4 degrees) and gives 10-60x MeshLab's face count, so 600k-face bodies.

The tests that check what MeshLab makes (face shape and density) are marked ``real_remesh``,
which opts out of the stand-in, and ``slow``, which the default run skips.  Run them with

    conda run -n mechfoundry-agents python -m pytest -q -p no:cacheprovider -m slow \\
        src/agents_wrapper/fem_workflow/operator_skill/tests

(``-m "slow or not slow"`` runs everything).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

import drive  # noqa: E402


def _remesh_facets(mesh, edge: float | None = None):
    """``remesh_built``'s stand-in: each planar facet re-triangulated at about ``edge``."""
    import trimesh
    from scipy.spatial import Delaunay, cKDTree

    edge = edge or drive.blank_edge(mesh)
    mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=True)
    group = np.full(len(mesh.faces), -1, dtype=np.int64)
    for k, facet in enumerate(mesh.facets):
        group[facet] = k
    normals = list(mesh.facets_normal)
    lone = np.flatnonzero(group < 0)                     # a face coplanar with no neighbour
    group[lone] = len(normals) + np.arange(len(lone))
    normals += list(mesh.face_normals[lone])
    vertices, count = [np.asarray(mesh.vertices, dtype=float)], len(mesh.vertices)
    # Every edge between two facets is a crease, sampled once: both sides use the same points.
    crease = group[mesh.face_adjacency[:, 0]] != group[mesh.face_adjacency[:, 1]]
    samples = {}
    for u, v in np.sort(mesh.face_adjacency_edges[crease], axis=1):
        a, b = mesh.vertices[u], mesh.vertices[v]
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / edge)))
        vertices.append(a + np.arange(1, n)[:, None] / n * (b - a))
        samples[(int(u), int(v))] = [int(u), *range(count, count + n - 1), int(v)]
        count += n - 1
    faces = []
    for g, normal in enumerate(normals):
        own = mesh.faces[group == g]
        normal = np.asarray(normal, dtype=float)
        e1 = np.cross(normal, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 0.5:
            e1 = np.cross(normal, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(normal, e1)                         # (e1, e2, normal) right-handed
        origin = mesh.vertices[own[0, 0]]
        flat = lambda p: np.column_stack([(p - origin) @ e1, (p - origin) @ e2])
        # The facet's outline: its edges used by one of its faces, as their crease samples.
        key = np.sort(np.concatenate([own[:, [0, 1]], own[:, [1, 2]], own[:, [2, 0]]]), axis=1)
        _, inverse, uses = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        ids = sorted({i for u, v in key[uses[inverse.ravel()] == 1] for i in samples[(int(u), int(v))]})
        outline = flat(np.vstack(vertices)[ids])
        triangles = flat(mesh.vertices[own.ravel()]).reshape(-1, 3, 2)
        # The interior: a hex lattice of spacing edge, kept 0.56 edge or more off the outline so
        # each outline segment's diametral circle is empty and Delaunay keeps it as an edge.
        lo, hi = triangles.reshape(-1, 2).min(axis=0), triangles.reshape(-1, 2).max(axis=0)
        rows = [np.column_stack([xs, np.full(len(xs), y)]) for r, y in enumerate(
                    np.arange(lo[1] + edge * 0.433, hi[1], edge * 0.866))
                for xs in [np.arange(lo[0] + (0.5 + 0.5 * (r % 2)) * edge, hi[0], edge)]]
        inner = np.vstack(rows) if rows else np.zeros((0, 2))
        if len(inner):
            inner = inner[cKDTree(outline).query(inner)[0] > 0.75 * edge]
            inner = inner[_inside(inner, triangles)]
            vertices.append(origin + inner[:, :1] * e1 + inner[:, 1:] * e2)
        points = np.vstack([outline, inner])
        index = np.array(ids + list(range(count, count + len(inner))))
        count += len(inner)
        simplices = Delaunay(points).simplices
        corner = points[simplices]
        cross = ((corner[:, 1, 0] - corner[:, 0, 0]) * (corner[:, 2, 1] - corner[:, 0, 1])
                 - (corner[:, 1, 1] - corner[:, 0, 1]) * (corner[:, 2, 0] - corner[:, 0, 0]))
        keep = (np.abs(cross) > 1e-12 * edge * edge) & _inside(corner.mean(axis=1), triangles)
        simplices, cross = simplices[keep], cross[keep]
        simplices[cross < 0] = simplices[cross < 0][:, ::-1]      # counter-clockwise: outward
        faces.append(index[simplices])
    out = trimesh.Trimesh(vertices=np.vstack(vertices), faces=np.vstack(faces), process=True)
    out.metadata["edge"] = edge
    return out


def _inside(points, triangles, chunk=4096):
    """Which 2-D points lie in any of the 2-D triangles (barycentric, edges included)."""
    out = np.zeros(len(points), dtype=bool)
    a, v0, v1 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    for s in range(0, len(points), chunk):
        p = points[s:s + chunk, None, :] - a[None]
        l1 = (p[..., 0] * v1[:, 1] - p[..., 1] * v1[:, 0]) / den
        l2 = (v0[:, 0] * p[..., 1] - v0[:, 1] * p[..., 0]) / den
        out[s:s + chunk] = ((l1 >= -1e-9) & (l2 >= -1e-9) & (l1 + l2 <= 1 + 1e-9)).any(axis=1)
    return out


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a test that takes many seconds; skipped unless -m selects it")
    config.addinivalue_line("markers", "real_remesh: run the real MeshLab remesh, not the in-process stand-in")
    config.addinivalue_line("markers", "web_rule: a built body needs the web photograph tried first, as in a run")


def pytest_collection_modifyitems(config, items):
    # With no -m given the slow tests are skipped; any -m expression decides for itself.
    if config.option.markexpr:
        return
    skip = pytest.mark.skip(reason="slow: run with -m slow")
    for item in items:
        if HERE in Path(str(item.fspath)).resolve().parents and item.get_closest_marker("slow"):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fast_remesh(request, monkeypatch):
    if request.node.get_closest_marker("real_remesh") is None:
        monkeypatch.setattr(drive, "remesh_built", _remesh_facets)


@pytest.fixture(autouse=True)
def _web_photo_tried(request, monkeypatch):
    # A test of something else builds its body with `blank`: the web photograph that comes first is
    # taken as tried.  The tests of that rule take the web_rule marker and meet it.
    if request.node.get_closest_marker("web_rule") is None:
        monkeypatch.setattr(drive, "web_photo_problem", lambda session: None)


# After the remesh, the suite's cost was the surface-pair search (``drive._facing_pairs``): 53 s
# of a 63 s run, 2-3 s a call on a long thin bar, where every sample has thousands of neighbours
# within reach.  ``_planned_resolution`` runs ``thin_gaps`` (30k samples) on every body, and most
# tests that plan a single bar read only the member and section measures beside it: the gap
# measure is computed and never looked at (only ``thin_gaps_problem`` and the report's history
# read it).  Those tests take the fixtures below, which return what the real measure returns on
# a body with no gap (or no wall under four leaves): its share 0 and its leaf, nothing else.  The
# tests of the measures themselves (thin_gaps, thin_walls, the geometry and gap stops) run them.

# A brief written for a test of something else needs the JSON lines G1 reads: a Use table with one
# run row, a Contacts JSON holding it, the Load path, the Real section's member and the Real extents.
USE_TABLE_LINE = ('Use table: {"actions": {"A1": {"load": "top load", "state": "working", "kind": "weight", '
                  '"size": 785, "unit": "N", "source": "80 kg x g x 1", "enters": "body", "held": ["base"]}}, '
                  '"run": "working"}')
CONTACTS_LINE = '{"base": {"transmits": "force and moment", "kind": "fastening"}}'


def tabled(brief: str) -> str:
    """``brief`` with the JSON lines G1 reads added where it lacks them (its own lines kept)."""
    import re
    brief = re.sub(r' ?\{"loads": \{"top load": 1\.0\}, "unit": "Pa", "governing": "top load"\}', "", brief)
    out = []
    for block in re.split(r"(?m)^(?=#+\s)", brief):
        title = block.split("\n", 1)[0].strip("# ").strip().lower()
        body = block.rstrip("\n")
        if title == "use" and "Use table:" not in body:
            body += "\n" + USE_TABLE_LINE
        elif title == "contacts" and "{" not in body:
            body += "\n" + CONTACTS_LINE
        elif title == "size and material":
            if "Load path:" not in body:
                body += '\nLoad path: {"body": "series"}'
            else:
                body = re.sub(r'Load path: \{"([^"]+)"', 'Load path: {"body": "series", "\\1"', body)
            if "Real extents:" not in body:
                body += "\nReal extents: 1 x 1 x 1 m"
            body = body.replace("Real section:", "Real section (body):")
        out.append(body + ("\n" if block.endswith("\n") else ""))
    return "".join(out)


def _nothing_found(mesh, leaf, samples=None):
    return {"share_of_surface": 0.0, "leaf_m": round(leaf, 5)}


@pytest.fixture
def gaps_not_read(monkeypatch):
    """For a test whose bodies have no gap and that never reads ``thin_gaps``."""
    monkeypatch.setattr(drive, "thin_gaps", _nothing_found)


@pytest.fixture
def walls_not_read(monkeypatch):
    """For a test whose bodies have no wall under four leaves and that never reads ``thin_walls``."""
    monkeypatch.setattr(drive, "thin_walls", _nothing_found)
