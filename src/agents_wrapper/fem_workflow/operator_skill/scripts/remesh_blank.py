#!/usr/bin/env python
"""Remesh a built body into well-shaped triangles of one size, keeping its edges.

A built body comes out of ``drive.py blank`` as few, long triangles (a box face
is two, an ear-clipped profile is a fan of slivers), and midpoint subdivision
keeps their shape: a long face stays a stack of slivers, so a contact a few
leaves wide catches a handful of faces and a held region has gaps between
them.  MeshLab's isotropic explicit remeshing splits, collapses, flips and
smooths edges towards one target length while keeping the creases (dihedral
angle above ``featuredeg``) and reprojecting onto the original surface, so the
planar faces, the edges and the dimensions stay those of the built body.

Run under the compute environment (pymeshlab is not in the agents env):

    remesh_blank.py IN_MESH OUT_MESH TARGET_EDGE
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    import pymeshlab

    source, target, edge = (argv or sys.argv[1:])[:3]
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(source)
    ms.meshing_isotropic_explicit_remeshing(
        iterations=12, targetlen=pymeshlab.PureValue(float(edge)), featuredeg=30.0,
        checksurfdist=True, maxsurfdist=pymeshlab.PureValue(float(edge) * 0.05))
    ms.save_current_mesh(target)


if __name__ == "__main__":
    main()
