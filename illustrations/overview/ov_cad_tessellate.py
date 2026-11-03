"""Tessellate the downloaded CAD bunny with its B-rep face ids (env mechfoundry-cadcoder).

    python ov_cad_tessellate.py assets/cad/<file>.step assets/cad/cad_bunny.npz

The workflow's own CAD path, minus the generator: ``cadcoder_wrapper.sandbox.
tessellate`` meshes the solid ONCE (so faces sharing an edge share its nodes)
and labels every triangle with the B-rep face it came from, and
``cad_common.mesh.finish_mesh`` welds, keeps the largest closed component and
orients it outward — exactly what a Text2CAD / CAD-Coder part goes through
before ``bc_input`` offers its faces as the segmentation (``cad_faces``).

It also reports what the file IS, because the acceptance test for the figure is
"reads as CAD": a solid of tens to a few hundred B-rep faces, mostly curved,
not a mesh exported to STEP with one planar face per triangle.
"""

import collections
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

src, out = Path(sys.argv[1]), Path(sys.argv[2])

import cadquery as cq  # noqa: E402

from src.cad_wrapper.cad_common.mesh import finish_mesh  # noqa: E402
from src.cad_wrapper.cadcoder_wrapper.sandbox import tessellate  # noqa: E402

shape = cq.importers.importStep(str(src))
solid = shape.val() if hasattr(shape, "val") else shape
faces = solid.Faces()
kinds = collections.Counter(f.geomType().lower() for f in faces)
print(f"{src.name}: {len(solid.Solids())} solid(s), {len(faces)} B-rep faces: {dict(kinds)}")
planar = kinds.get("plane", 0) / max(len(faces), 1)
if len(faces) > 2000 or planar > 0.9:
    raise SystemExit(f"{src.name} looks like a mesh exported to STEP ({len(faces)} faces, "
                     f"{planar:.0%} planar) — it will not read as CAD")

vertices, triangles, labels, surface_types = tessellate(solid, relative_tolerance=5e-4)
mesh, labels, dropped = finish_mesh(vertices, triangles, labels, source=src.name)
print(f"tessellated: {len(mesh.vertices)} V / {len(mesh.faces)} F, "
      f"{len(np.unique(labels))} faces kept, {dropped} loose components dropped")
np.savez(out, vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces),
         face_ids=labels.astype(np.int32), surface_types=np.array(surface_types))
print(f"wrote {out}")
