"""Build the CAD lane's CadQuery script as CAD-Coder's output is built (env mechfoundry-cadcoder).

    python ov_cad_build.py assets/cad/bunny_cadcoder.py assets/cad/cad_bunny.npz

The workflow's own functions, minus the language model:
``cadcoder_wrapper.sandbox.check_source`` (the static safety check a generated
script must pass), ``execute`` and ``tessellate`` (the solid meshed face by face
at the default tolerance), and ``cad_common.mesh.finish_mesh`` (welded into one
closed, outward mesh — what ``bc_input`` then offers as ``cad_faces`` segments).

In process, NOT through ``run_sandboxed``: its child is capped at 120 s of CPU,
and OCC runs every Boolean on all 64 cores of this machine, so the 143-patch
bunny (about 30 s of wall time) spends 150-400 CPU-seconds and is killed.  The
script itself passes the same checks.  The npz is ``ov_cad_tessellate.py``'s
format, so ``ov_surfaces.py`` reads either.
"""

import collections
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.cad_wrapper.cad_common.mesh import finish_mesh  # noqa: E402
from src.cad_wrapper.cadcoder_wrapper.sandbox import check_source, execute, tessellate  # noqa: E402

src, out = Path(sys.argv[1]), Path(sys.argv[2])
code = src.read_text()
check_source(code)                                   # what CAD-Coder's output must pass
solid = execute(code)
assert solid.isValid(), f"{src.name} built an invalid solid"
vertices, triangles, labels, surface_types = tessellate(solid)
print(f"{src.name}: {len(surface_types)} B-rep faces "
      f"{dict(collections.Counter(surface_types))}")
mesh, labels, dropped = finish_mesh(vertices, triangles, labels, source=src.name)
print(f"closed: {len(mesh.vertices)} V / {len(mesh.faces)} F, {len(np.unique(labels))} faces, "
      f"{dropped} loose components dropped")
# finish_mesh centres the part in a unit box; the script is in the display
# frame already, so put it back where it was built (ov_surfaces still aligns it)
lo, hi = vertices.min(axis=0), vertices.max(axis=0)
placed = np.asarray(mesh.vertices) * float((hi - lo).max()) + (lo + hi) / 2.0
np.savez(out, vertices=placed, faces=np.asarray(mesh.faces),
         face_ids=labels.astype(np.int32), surface_types=np.array(surface_types))
print(f"wrote {out}")
