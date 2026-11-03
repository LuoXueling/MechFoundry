from core.utils import dotdict
import torch

axial_strain = -0.009


def ux(x, y, z):
    return axial_strain * x


def uy(x, y, z):
    return 0.0


def uz(x, y, z):
    return 0.0


params = dotdict(
    project_name="three_d_affine_compression",
    output_root="examples_output",
    dim=3,
    q_deg=2,
    elem_order=1,
    width=0.85,
    height=0.35,
    depth=0.3,
    nx=3,
    ny=2,
    nz=2,
    relative_tol=1e-5,
    absolute_tol=2e-8,
    max_iter=25,
    E=32.0,
    nu=0.3,
    dirichlet_bcs_dict={
        lambda x, y, z: torch.isclose(x, torch.zeros_like(x)): (ux, uy, uz),
        lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.85)): (ux, uy, uz),
        lambda x, y, z: torch.isclose(y, torch.zeros_like(y)): (ux, uy, uz),
        lambda x, y, z: torch.isclose(y, torch.full_like(y, 0.35)): (ux, uy, uz),
        lambda x, y, z: torch.isclose(z, torch.zeros_like(z)): (ux, uy, uz),
        lambda x, y, z: torch.isclose(z, torch.full_like(z, 0.3)): (ux, uy, uz),
    },
    neumann_bcs_dict={},
    B=(0.0, 0.0, 0.0),
)
