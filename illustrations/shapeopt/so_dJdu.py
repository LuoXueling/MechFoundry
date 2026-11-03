"""dJ/du of the captured iteration, on the CPU, from the cached solve (pvpython).

    pvpython so_dJdu.py RUN_VTU            # -> so_dJdu.vtk  (+ prints a check)

The adjoint right-hand side of ShapeOpt's objective.  J = SE/SE_0 + w_vol * P(V); the
volume term does not depend on u, so dJ/du = (1/SE_0) dSE/du, and SE's standard-path
integrand (shape_opt._make_se_standard_integrand_phi_direct) is

    phi_w W(F),  phi_w = (1 - phi_floor) phi + phi_floor,
    W = mu/2 (J^-2/3 I1 - 3) + kappa/2 (ln J)^2,  F = I + grad u,

so dSE/du_a = sum_e sum_qp  phi_w  P(F) . grad N_a  w_qp |J_e|  with
P = mu J^-2/3 (F - I1/3 F^-T) + kappa ln J F^-T -- the internal-force vector.

What is ported, and what is not:
  * the trimmed solve mesh, its nodal displacement and nodal phi are read from the run's
    own VTU of that iteration (solution_t0040.vtu: 78 223 nodes, max |u| asserted equal
    to so_stats.txt's umag_max);
  * E = 10, nu = 0.3, phi_floor = 1e-4 (configs/config_shape_opt_cantilever.py,
    core/default_config.py); 2 x 2 x 2 Gauss on every axis-aligned hex;
  * hanging nodes: each slave's entry is handed to its edge (1/2) or face (1/4)
    masters, as C^T does (the solver's constrained residual);
  * the strong Dirichlet nodes (bc_dirichlet_0_region) are zeroed, as the adjoint
    masks them (abstract_field.adjoint_fn, step 4);
  * NOT ported: the adaptive Nitsche quadrature the solver uses in boundary elements
    (here every element gets the standard phi-weighted rule) and the 1/SE_0 scale
    (the figure normalises by the field's own 99th percentile anyway).
"""
import sys

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

E, NU, PHI_FLOOR = 10.0, 0.3, 1e-4
MU = E / (2.0 * (1.0 + NU))
KAPPA = E / (3.0 * (1.0 - 2.0 * NU))

r = vtk.vtkXMLUnstructuredGridReader()
r.SetFileName(sys.argv[1])
r.Update()
g = r.GetOutput()
X = vtk_to_numpy(g.GetPoints().GetData()).astype(np.float64)
pd = g.GetPointData()
U = vtk_to_numpy(pd.GetArray("displacement")).astype(np.float64)
PHI = vtk_to_numpy(pd.GetArray("phi")).astype(np.float64)
DIR = vtk_to_numpy(pd.GetArray("bc_dirichlet_0_region")) > 0.5
conn = vtk_to_numpy(g.GetCells().GetConnectivityArray()).reshape(-1, 8).astype(np.int64)
stats = dict(l.strip().split("=", 1) for l in open("so_stats.txt") if "=" in l)
assert len(X) == int(stats["solve_nodes"]) and len(conn) == int(stats["solve_hex"])
assert abs(np.linalg.norm(U, axis=1).max() - float(stats["umag_max"])) < 1e-6

# element boxes, and each corner's reference coordinates (0/1 per axis)
xe = X[conn]                                            # (ne, 8, 3)
lo, hi = xe.min(1), xe.max(1)
h = hi - lo
ref = np.rint((xe - lo[:, None, :]) / h[:, None, :]).astype(np.int64)   # (ne, 8, 3)
gp = np.array([0.5 - 0.5 / np.sqrt(3.0), 0.5 + 0.5 / np.sqrt(3.0)])
f = np.zeros_like(U)
I3 = np.eye(3)
for qx in gp:
    for qy in gp:
        for qz in gp:
            q = np.array([qx, qy, qz])
            # trilinear N_a and dN_a/dx at the point, per element
            fac = np.where(ref == 1, q, 1.0 - q)                         # (ne, 8, 3)
            dfac = np.where(ref == 1, 1.0, -1.0)
            N = fac.prod(-1)
            dN = np.stack([dfac[..., 0] * fac[..., 1] * fac[..., 2],
                           fac[..., 0] * dfac[..., 1] * fac[..., 2],
                           fac[..., 0] * fac[..., 1] * dfac[..., 2]], -1) / h[:, None, :]
            F = I3 + np.einsum("eai,eaj->eij", U[conn], dN)
            J = np.linalg.det(F)
            Finv_T = np.transpose(np.linalg.inv(F), (0, 2, 1))
            I1 = np.einsum("eij,eij->e", F, F)
            P = MU * J[:, None, None] ** (-2.0 / 3.0) * (F - I1[:, None, None] / 3.0 * Finv_T) \
                + KAPPA * np.log(J)[:, None, None] * Finv_T
            phi_w = (1.0 - PHI_FLOOR) * (N * PHI[conn]).sum(1) + PHI_FLOOR
            w = phi_w * h.prod(1) / 8.0
            fe = np.einsum("eij,eaj->eai", P, dN) * w[:, None, None]
            np.add.at(f, conn, fe)

# hanging nodes: a node at an element's edge midpoint / face centre that is not one of
# its corners is a slave of that edge's / face's corners
key = lambda p: tuple(np.rint(p * 1e5).astype(np.int64))           # noqa: E731
pos = {key(p): i for i, p in enumerate(X)}
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
FACES = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
masters = {}
for e, c in enumerate(conn):
    cs = set(c.tolist())
    for grp in EDGES + FACES:
        i = pos.get(key(X[c[list(grp)]].mean(0)))
        if i is not None and i not in cs:
            masters.setdefault(i, set(c[list(grp)].tolist()))
for _ in range(4):                                      # masters may be slaves themselves
    moved = 0
    for s, ms in masters.items():
        if np.any(f[s]):
            for m in ms:
                f[m] += f[s] / len(ms)
            f[s] = 0.0
            moved += 1
    if not moved:
        break
f[DIR] = 0.0
mag = np.linalg.norm(f, axis=1)
deep = PHI > 0.999
top = vtk_to_numpy(pd.GetArray("bc_neumann_0_region")) > 0.5
print(f"slaves {len(masters)}; |dSE/du| p99 {np.percentile(mag[mag > 0], 99):.3e}; "
      f"median on the loaded top {np.median(mag[top]):.3e}; "
      f"median deep interior {np.median(mag[deep & ~top]):.3e}")

out = vtk.vtkUnstructuredGrid()
out.ShallowCopy(g)
opd = out.GetPointData()
for i in reversed(range(opd.GetNumberOfArrays())):
    opd.RemoveArray(i)
a = numpy_to_vtk(mag.astype(np.float32), deep=1)
a.SetName("gu")
opd.AddArray(a)
s = numpy_to_vtk(vtk_to_numpy(pd.GetArray("sdf")).astype(np.float32), deep=1)
s.SetName("sdf")
opd.AddArray(s)
w = vtk.vtkUnstructuredGridWriter()
w.SetFileName("so_dJdu.vtk")
w.SetInputData(out)
w.SetFileTypeToBinary()
w.Write()
print("wrote so_dJdu.vtk")
