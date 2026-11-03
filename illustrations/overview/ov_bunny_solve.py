"""Solve the bunny problem used by diffused_nitsche and export its surface.

Runs ``nitsche/dn_prep.py`` in this directory so geometry, octree, phase field,
and resolved ear/base SurfaceSelectors are literally the ones underlying
``diffused_nitsche.pdf``.  A downward traction is applied to the ears and the
bottom is fixed.  The production ``octree_hex_interpolate`` routine transfers
the converged octree displacement to the original surface mesh; VTUSurfaceWriter
then emits the triangle VTU consumed by ``render_ov_solve.py``.
"""
import os
import runpy
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.getcwd()
PREFIX = os.environ.get("DN_SOLVE_PREFIX", "ov_bunny_solve")
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "warp_wrapper"))
sys.path.insert(0, REPO)

from core.nitsche_field import NitscheNeoHookean
from core.remeshing import octree_hex_interpolate
from core.vtu_writer import VTUSurfaceWriter

# Execute the source figure's prep, rather than maintaining a second copy of
# its selection recipes.  build.sh has already put bunny.stl/.meta in cwd.
dn = runpy.run_path(os.path.join(HERE, "..", "nitsche", "dn_prep.py"))
bunny = dn["bunny"]

# Nitsche mutates its octree when trim_exterior=True, so hand it an owned copy.
octree = {}
for key, val in dn["octree"].items():
    octree[key] = val.clone() if isinstance(val, torch.Tensor) else val
octree["domain_length"] = dn["domain_length"]

out_root = os.path.join(OUT, f"_{PREFIX}")
shutil.rmtree(out_root, ignore_errors=True)
# Four times the former small-strain load: the physical deformation should be
# close to the old solution displayed with an artificial 4x warp.
TRACTION_Z = -0.020

params = dict(
    project_name="bunny_nitsche",
    output_root=out_root,
    octree_mesh=octree,
    phase_field_epsilon=dn["eps"],
    # Match the accuracy-relevant settings used by nitsche_convergence.  Its
    # 20-step continuation is specific to the large prescribed twist; this
    # small traction problem converges directly in one (the base default).
    # This small-load f32 case reaches a nonlinear residual floor around 8e-4.
    relative_tol=1e-3,
    use_mixed_precision=False,
    tangent_f64=False,
    cg_backend_assembled="jaxamg",
    async_io=True,
    trim_exterior=True,
    # No geometric_func: this explicitly selects the solver's standard
    # 2x2x2 Gauss rule on every retained hex.  Merely setting refinement depth
    # to zero still enters the adaptive-quadrature machinery.
    dirichlet_bcs_dict={dn["bottom_sel"]: 0.0},
    neumann_bcs_dict={dn["ear_sel"]: (0.0, 0.0, TRACTION_Z)},
    output_per_step=0,
)

solver = NitscheNeoHookean(
    "src.warp_wrapper.configs.config_nitsche_example",
    override_params=params)
t0 = time.perf_counter()
solver.forward()
solve_seconds = time.perf_counter() - t0

mesh = solver.params.octree_mesh
nodes = torch.as_tensor(mesh["nodes"], dtype=torch.float64, device="cuda")
elems = torch.as_tensor(mesh["elements"], dtype=torch.long, device="cuda")
u = torch.as_tensor(solver.get_displacement(), dtype=torch.float64, device="cuda")
surf_v = torch.as_tensor(np.asarray(bunny.vertices), dtype=torch.float64,
                         device="cuda")
surf_u = octree_hex_interpolate(nodes, elems, u, surf_v).cpu().numpy()

writer = VTUSurfaceWriter(OUT, grid_shape=None, write_vtu=True,
                          write_glb=False, write_glb_deformed=False,
                          async_io=False)
writer.save_from_mesh(0, np.asarray(bunny.vertices), np.asarray(bunny.faces),
                      {"displacement": surf_u,
                       "displacement_magnitude": np.linalg.norm(surf_u, axis=1)})
src = os.path.join(OUT, "vtu_surface", "surface_t0000.vtu")
os.replace(src, os.path.join(OUT, f"{PREFIX}.vtu"))
shutil.rmtree(os.path.join(OUT, "vtu_surface"), ignore_errors=True)
shutil.rmtree(out_root, ignore_errors=True)

umax = float(np.linalg.norm(surf_u, axis=1).max())
with open(f"{PREFIX}_stats.txt", "w") as f:
    f.write(f"surface_vertices {len(bunny.vertices)}\n")
    f.write(f"surface_faces {len(bunny.faces)}\n")
    f.write(f"solve_nodes {len(nodes)}\n")
    f.write(f"solve_elements {len(elems)}\n")
    f.write(f"traction_z {TRACTION_Z}\n")
    f.write(f"umax {umax:.8e}\n")
    f.write(f"newton_iterations {solver.newton_iterations}\n")
    f.write(f"solve_seconds {solve_seconds:.3f}\n")
print(f"wrote {PREFIX}.vtu: umax={umax:.4e}, "
      f"Newton={solver.newton_iterations}, solve={solve_seconds:.2f} s")
