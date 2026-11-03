from core.utils import dotdict
import torch

axial_strain = -0.012


def ux(x, y):
    return axial_strain * x


def uy(x, y):
    return 0.0


params = dotdict(
    project_name="two_d_affine_compression",
    output_root="examples_output",
    dim=2,
    q_deg=2,
    elem_order=1,
    width=0.9,
    height=0.4,
    nx=4,
    ny=3,
    relative_tol=1e-5,
    absolute_tol=2e-8,
    max_iter=20,
    E=18.0,
    nu=0.29,
    dirichlet_bcs_dict={
        lambda x, y: torch.isclose(x, torch.zeros_like(x)): (ux, uy),
        lambda x, y: torch.isclose(x, torch.full_like(x, 0.9)): (ux, uy),
        lambda x, y: torch.isclose(y, torch.zeros_like(y)): (ux, uy),
        lambda x, y: torch.isclose(y, torch.full_like(y, 0.4)): (ux, uy),
    },
    neumann_bcs_dict={},
    B=(0.0, 0.0),
)
