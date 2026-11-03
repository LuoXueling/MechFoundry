"""A flat base the generator added under a body the brief stands on several supports (flagged at the
geometry stop, listed as a simplification when kept), and a rest held above the ground the body
stands on (refused at the regions approval).  Synthetic placeholder bodies; no GPU, no API."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import drive  # noqa: E402
import sessionlib as lib  # noqa: E402

REST = {"transmits": "force only", "kind": "rest"}


def _cart(slab: bool):
    """A box on four disc wheels (axes along z, their bottoms at y = 0), with a flat plank under the
    box down to the ground when ``slab``."""
    import trimesh
    parts = [trimesh.creation.box(extents=[0.3, 0.06, 0.12], transform=trimesh.transformations.translation_matrix(
        [0.0, 0.08, 0.0]))]
    for x in (-0.1, 0.1):
        for z in (-0.07, 0.07):
            wheel = trimesh.creation.cylinder(radius=0.04, height=0.02, sections=48)
            wheel.apply_translation([x, 0.04, z])
            parts.append(wheel)
    if slab:
        parts.append(trimesh.creation.box(extents=[0.28, 0.01, 0.08], transform=trimesh.transformations.translation_matrix(
            [0.0, 0.006, 0.0])))
    return trimesh.util.concatenate(parts)


def _body_session(tmp_path, monkeypatch, mesh, contacts):
    session = tmp_path / "abc"
    (session / "operator").mkdir(parents=True)
    body = session / "body.glb"
    mesh.export(body)
    monkeypatch.setattr(drive, "body_source", lambda s: body)
    monkeypatch.setattr(drive, "body_provenance", lambda s: "triposg")
    monkeypatch.setattr(drive, "body_digest", lambda s: "body1")
    monkeypatch.setattr(drive, "brief_contacts_v2", lambda operator: contacts)
    return session


def test_supports_the_brief_makes_several_are_read_from_the_contacts():
    assert drive.discrete_rests({"wheel rests": {**REST, "member": "wheels"}}) == ["wheel rests"]
    assert drive.discrete_rests({"feet": REST}) == ["feet"]
    assert drive.discrete_rests({"rocker rest": {**REST, "member": "rockers"}}) == ["rocker rest"]
    assert drive.discrete_rests({"seat front": REST, "seat rear": REST}) == ["seat front", "seat rear"]
    # one support: a plate under stacked layers, a base; a grip or a hang is no rest
    assert drive.discrete_rests({"plate": {**REST, "member": "sponge layers"}}) == []
    assert drive.discrete_rests({"base": REST}) == []
    assert drive.discrete_rests({"handles": {"transmits": "force and moment", "kind": "grip"}}) == []


def test_a_flat_base_under_discrete_supports_is_flagged_and_listed_when_kept(tmp_path, monkeypatch):
    contacts = {"wheel rests": {**REST, "member": "wheels"}}
    session = _body_session(tmp_path, monkeypatch, _cart(slab=True), contacts)
    said = drive.base_slab_warning(session, {})
    assert said and "a flat base under the body that the brief's supports do not have" in said
    assert "may have swallowed the supports (wheels)" in said and "refine first" in said
    assert drive.base_slab_kept(session)
    assert drive.SLAB_SAID in drive.runner_simplifications(session, {}, [])
    # the flag is a reject's check: the supports fused into the base, named either way
    assert drive.reject_problem(session, "the wheels fused into a flat base slab") is None
    # measured once per body (the record), said again at the same body
    assert lib.read_json(session / "operator" / drive.SLAB_RECORD)["flagged"] is True
    assert drive.base_slab_warning(session, {}) == said


def test_discrete_supports_reaching_the_ground_alone_are_not_flagged(tmp_path, monkeypatch):
    session = _body_session(tmp_path, monkeypatch, _cart(slab=False), {"wheel rests": {**REST, "member": "wheels"}})
    assert drive.base_slab_warning(session, {}) is None
    assert not drive.base_slab_kept(session)
    assert drive.SLAB_SAID not in drive.runner_simplifications(session, {}, [])


def test_a_genuine_flat_base_is_never_flagged(tmp_path, monkeypatch):
    import trimesh
    plinth = trimesh.creation.box(extents=[0.3, 0.04, 0.3], transform=trimesh.transformations.translation_matrix(
        [0.0, 0.02, 0.0]))
    column = trimesh.creation.cylinder(radius=0.05, height=0.4, sections=32)
    column.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    column.apply_translation([0.0, 0.24, 0.0])
    body = trimesh.util.concatenate([plinth, column])
    # the base fills the footprint: the measure sees it, the brief's one support keeps it unmeasured
    found = drive.base_slab(body, [0.0, 1.0, 0.0])
    assert found["share"] > 0.9
    session = _body_session(tmp_path, monkeypatch, body, {"base": {**REST, "member": "plinth"}})
    assert drive.base_slab_warning(session, {}) is None
    # a built body is the agent's own: not measured either
    monkeypatch.setattr(drive, "brief_contacts_v2", lambda operator: {"feet": REST})
    monkeypatch.setattr(drive, "body_provenance", lambda s: "built")
    assert drive.base_slab_warning(session, {}) is None


def _two_level_session(tmp_path):
    """A low block on the ground and a bar overhanging at 0.2 m, each underside a held piece of one
    region "under"."""
    import trimesh
    block = trimesh.creation.box(extents=[0.4, 0.05, 0.2], transform=trimesh.transformations.translation_matrix(
        [0.0, 0.025, 0.0]))
    post = trimesh.creation.box(extents=[0.04, 0.3, 0.04], transform=trimesh.transformations.translation_matrix(
        [-0.15, 0.17, 0.0]))
    bar = trimesh.creation.box(extents=[0.6, 0.02, 0.04], transform=trimesh.transformations.translation_matrix(
        [0.15, 0.21, 0.0]))
    body = trimesh.util.concatenate([block, post, bar]).subdivide_to_size(0.02)
    body = trimesh.Trimesh(body.vertices, body.faces, process=False)
    session = tmp_path / "abc"
    for folder in ("geometry", "boundary_conditions", "operator"):
        (session / folder).mkdir(parents=True, exist_ok=True)
    (session / "checkpoints.sqlite").write_text("")
    body.export(session / "geometry" / "physical.glb")
    centres, normals = body.triangles_center, body.face_normals
    under = (normals[:, 1] < -0.9) & (((centres[:, 1] < 0.001) & (np.abs(centres[:, 0]) < 0.05))
                                      | ((np.abs(centres[:, 1] - 0.2) < 0.001) & (centres[:, 0] > 0.35)))
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"category": "dirichlet", "name": "under", "key": "under",
                                 "mean_normal": [0, -1, 0], "bounds": [[-0.05, 0.0, -0.1], [0.45, 0.2, 0.1]]}]})
    lib.write_json(session / "boundary_conditions" / "proposal.json", {"dirichlet": [{"name": "under",
                                                                                    "components": [1]}]})
    np.savez(session / "boundary_conditions" / "resolved.npz", under=under)
    return session


@pytest.mark.parametrize("name,kind,refused", [("wheels", "rest", True), ("floor", "rest", False),
                                                ("hooks", "hang", False)])
def test_a_rest_held_above_the_ground_is_refused(tmp_path, name, kind, refused):
    session = _two_level_session(tmp_path)
    plan = {"contacts": {name: {"transmits": "force only", "regions": ["under"]}}}
    rows = drive.held_regions(session, None, plan["contacts"])
    heights = sorted(r["above_ground_m"] for r in rows if not r.get("sliver"))
    assert len(heights) == 2 and heights[0] == pytest.approx(0.0, abs=1e-6) and heights[1] == pytest.approx(0.2, abs=1e-3)
    said = drive.rest_ground_problem(session, plan, rows, {name: {"transmits": "force only", "kind": kind}})
    if refused:
        # several supports: each must touch the ground; the raised bar is no support of the brief's
        assert said and f"contact {name} (a rest)" in said and "above the lowest point the body reaches" in said
        assert "never another member held in its place" in said
    else:
        # one support reaches the ground with its lowest piece; a hang is not checked
        assert said is None
    # a held region no contact lists yet is the brief's one contact's
    unbound = drive.held_regions(session, None, {})
    assert (drive.rest_ground_problem(session, {}, unbound, {name: {"transmits": "force only", "kind": kind}})
            is None) is not refused


def test_a_flagged_body_is_refused_until_the_ladder_and_the_listed_routes_were_tried(tmp_path, monkeypatch):
    session = _body_session(tmp_path, monkeypatch, _cart(slab=True), {"wheel rests": {**REST, "member": "wheels"}})
    monkeypatch.setattr(drive, "plan_image_key", lambda plan: "img")
    monkeypatch.setattr(drive, "hunyuan_ready", lambda: True)
    monkeypatch.setattr(drive, "generated_bodies", lambda s: drive.GENERATED_BUDGET)
    monkeypatch.setattr(drive, "declared_routes", lambda s: ["triposg_text", "drawing", "hunyuan3d"])
    monkeypatch.setattr(drive, "route_tried", lambda s, route, counts=None: route != "hunyuan3d")
    plan = {"geometry": {"draw": True}}
    assert drive.slab_route_problem(session, plan) is None          # not measured yet: nothing flagged
    drive.base_slab_warning(session, plan)
    said = drive.slab_route_problem(session, plan)
    assert "refine first" in said and "init_octree_depth 6, octree_depth 9" in said
    # the first rung sent (the body at 6/9): the second; both sent: Hunyuan3D from the same image
    lib.write_json(session / "operator" / drive.REFINE_SENT, {"img": ["a"]})
    finer = {"geometry": {"draw": True, "generation_settings": {"init_octree_depth": 6, "octree_depth": 9}}}
    assert "init_octree_depth 5, octree_depth 8" in drive.slab_route_problem(session, finer)
    lib.write_json(session / "operator" / drive.REFINE_SENT, {"img": ["a", "b"]})
    assert "Hunyuan3D from the same image" in drive.slab_route_problem(session, finer)
    # every rung and listed route tried: accepted, a marked simplification
    lib.write_json(session / "operator" / "run_counts.json", {"hunyuan": 1})
    monkeypatch.setattr(drive, "route_tried", lambda s, route, counts=None: True)
    assert drive.slab_route_problem(session, finer) is None
    assert drive.SLAB_SAID in drive.runner_simplifications(session, {}, [])
    # a listed route still untried within the budget is taken before
    monkeypatch.setattr(drive, "generated_bodies", lambda s: 1)
    monkeypatch.setattr(drive, "route_tried", lambda s, route, counts=None: route != "triposg_text")
    assert drive.ROUTE_WORDS["triposg_text"] in drive.slab_route_problem(session, finer)


def _held_mesh_session(tmp_path, body, mask, components):
    session = tmp_path / "held"
    for folder in ("geometry", "boundary_conditions", "operator"):
        (session / folder).mkdir(parents=True, exist_ok=True)
    (session / "checkpoints.sqlite").write_text("")
    body.export(session / "geometry" / "physical.glb")
    lo, hi = body.bounds
    lib.write_json(session / "boundary_conditions" / "resolved.json",
                   {"regions": [{"category": "dirichlet", "name": "held", "key": "held", "mean_normal": [0, -1, 0],
                                 "bounds": [lo.tolist(), hi.tolist()]}]})
    lib.write_json(session / "boundary_conditions" / "proposal.json",
                   {"dirichlet": [{"name": "held", "components": components}]})
    np.savez(session / "boundary_conditions" / "resolved.npz", held=mask)
    return session


def test_a_sideways_rest_is_not_checked_against_the_ground(tmp_path):
    """A rest against a wall (its faces looking along x, held on x) or on a top face (a BC-sweep case)
    is no rest the body stands on: never refused for its height."""
    import trimesh
    body = trimesh.creation.box(extents=[0.2, 0.6, 0.2]).subdivide_to_size(0.03)
    body = trimesh.Trimesh(body.vertices, body.faces, process=False)
    centres, normals = body.triangles_center, body.face_normals
    upper_side = (normals[:, 0] > 0.9) & (centres[:, 1] > 0.15)               # a wall rest, high up
    top = normals[:, 1] > 0.9                                                 # a rest on the top face
    contacts = {"wall stops": {"transmits": "force only", "kind": "rest"}}
    for mask, components in ((upper_side, [0]), (upper_side, [0, 1, 2]), (top, [1])):
        session = _held_mesh_session(tmp_path / str(len(list(tmp_path.iterdir()))), body, mask, components)
        plan = {"contacts": {"wall stops": {"transmits": "force only", "regions": ["held"]}}}
        rows = drive.held_regions(session, None, plan["contacts"], plan)
        assert rows and min(r["above_ground_m"] for r in rows) > 0.1
        assert drive.rest_ground_problem(session, plan, rows, contacts) is None


def test_a_curved_rocker_held_whole_is_refused_and_its_lowest_band_taken(tmp_path):
    """Two curved runners: their whole undersides held rise 0.13 m up the arcs (a rocker touches along
    its lowest line); the faces within the band the message names pass."""
    import trimesh
    angles = np.radians(np.linspace(240.0, 300.0, 61))
    parts = []
    for z in (-0.2, 0.2):
        # an annular sector (radii 0.97-1.0) 0.04 thick: per angle, outer/inner at z -/+ 0.02
        ring = np.asarray([(r * np.cos(a), r * np.sin(a), z + dz) for a in angles for r in (1.0, 0.97)
                           for dz in (-0.02, 0.02)])
        faces, centres = [], []
        for i in range(len(angles) - 1):
            a, b = 4 * i, 4 * (i + 1)
            centre = ring[a:b + 4].mean(axis=0)                  # each segment is convex: outward is away from it
            for p, q in ((0, 1), (1, 3), (3, 2), (2, 0)):         # outer, side, inner, side
                faces += [[a + p, b + p, b + q], [a + p, b + q, a + q]]
                centres += [centre, centre]
        last = 4 * (len(angles) - 1)
        faces += [[0, 2, 3], [0, 3, 1], [last, last + 1, last + 3], [last, last + 3, last + 2]]
        centres += [ring[0:8].mean(axis=0)] * 2 + [ring[last - 4:last + 4].mean(axis=0)] * 2
        faces = np.asarray(faces)
        tri = ring[faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        flip = np.einsum("ij,ij->i", normal, tri.mean(axis=1) - np.asarray(centres)) < 0
        faces[flip] = faces[flip][:, ::-1]
        parts.append(trimesh.Trimesh(ring, faces, process=True))
    seat = trimesh.creation.box(extents=[0.5, 0.05, 0.44], transform=trimesh.transformations.translation_matrix(
        [0.0, -0.7, 0.0]))
    body = trimesh.util.concatenate(parts + [seat]).subdivide_to_size(0.02)
    body = trimesh.Trimesh(body.vertices, body.faces, process=False)
    centres, normals = body.triangles_center, body.face_normals
    radial = np.einsum("ij,ij->i", normals[:, :2], centres[:, :2] / np.linalg.norm(centres[:, :2], axis=1)[:, None])
    bottoms = (radial > 0.9) & (np.linalg.norm(centres[:, :2], axis=1) > 0.99)          # each runner's underside
    contacts = {"rocker rest": {"transmits": "force only", "kind": "rest", "member": "rockers"}}
    plan = {"contacts": {"rocker rest": {"transmits": "force only", "regions": ["held"]}}}
    session = _held_mesh_session(tmp_path / "whole", body, bottoms, [1])
    rows = drive.held_regions(session, None, plan["contacts"], plan)
    said = drive.rest_ground_problem(session, plan, rows, contacts)
    assert said and "contact rocker rest (a rest)" in said and "Narrow the hold to the support's faces" in said
    band = float(said.split("within ")[1].split(" m")[0])
    lowest = body.vertices[:, 1].min()
    near = bottoms & (body.vertices[body.faces][:, :, 1].max(axis=1) <= lowest + 0.9 * band)
    session = _held_mesh_session(tmp_path / "band", body, near, [1])
    rows = drive.held_regions(session, None, plan["contacts"], plan)
    assert len([r for r in rows if not r.get("sliver")]) == 2
    assert drive.rest_ground_problem(session, plan, rows, contacts) is None
