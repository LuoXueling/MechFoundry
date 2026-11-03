#!/usr/bin/env python
"""Union the members of a built frame into one watertight body.

``drive.py blank --bars`` builds each bar, box and joint hub as its own closed
mesh; they overlap at the joints.  Manifold's boolean union (through trimesh,
``engine="manifold"``) returns a closed, oriented 2-manifold of the exact
union, with sharp edges where members meet: a voxel union with marching
cubes rounds every edge to the grid and moves the faces the regions are picked
on.  On a frame of members meeting at one apex under a box, and on one of
members in two planes with a cross bar, it came out watertight and one body,
and the geometry stop's thin-wall and thin-gap measures read nothing at depth
8 after the remesh.

The union is written as Manifold returns it and read back unmerged: it can hold
vertices a few nanometres apart where two round bars touch tangentially, and
merging them pinches the body (``drive._union_groups``).

Run under the compute environment (manifold3d is not in the agents env):

    union_blank.py OUT_MESH IN_MESH [IN_MESH ...]
    union_blank.py --groups GROUPS_JSON     # [{"target": OUT, "sources": [IN, ...]}, ...]

The second form unions several groups in one launch (a failed frame is checked
joint by joint that way).
"""

from __future__ import annotations

import json
import sys


def union(meshes):
    import trimesh
    return trimesh.boolean.union(meshes, engine="manifold")


def main(argv: list[str] | None = None) -> None:
    import trimesh

    args = argv or sys.argv[1:]
    if args[0] == "--groups":
        with open(args[1], encoding="utf-8") as handle:
            jobs = json.load(handle)
    else:
        jobs = [{"target": args[0], "sources": args[1:]}]
    loaded: dict[str, object] = {}      # each member read once, though several groups share it

    def member(source):
        if source not in loaded:
            loaded[source] = trimesh.load(source, force="mesh", process=True)
        return loaded[source]

    for job in jobs:
        meshes = [member(s) for s in job["sources"]]
        (meshes[0] if len(meshes) == 1 else union(meshes)).export(job["target"])


if __name__ == "__main__":
    main()
