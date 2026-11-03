from core.utils import dotdict
import torch

shear = 0.02


def ux(x, y):
    return shear * y


def uy(x, y):
    return 0.0


params = dotdict(
    project_name="two_d_affine_shear",
    output_root="examples_output",
    dim=2,
    q_deg=2,
    elem_order=1,
    width=1.0,
    height=0.5,
    nx=20,
    ny=10,
    relative_tol=1e-5,
    absolute_tol=2e-8,
    max_iter=20,
    E=20.0,
    nu=0.3,
    dirichlet_bcs_dict={
        lambda x, y: torch.isclose(x, torch.zeros_like(x)): (ux, uy),
        lambda x, y: torch.isclose(x, torch.full_like(x, 1.0)): (ux, uy),
        lambda x, y: torch.isclose(y, torch.zeros_like(y)): (ux, uy),
        lambda x, y: torch.isclose(y, torch.full_like(y, 0.5)): (ux, uy),
    },
    neumann_bcs_dict={},
    B=(0.0, 0.0),
)
