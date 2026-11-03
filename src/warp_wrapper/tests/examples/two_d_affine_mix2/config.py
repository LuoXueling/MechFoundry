from core.utils import dotdict
import torch

grad_xx = -0.003

grad_xy = 0.008

grad_yx = -0.002

grad_yy = 0.005


def ux(x, y):
    return grad_xx * x + grad_xy * y


def uy(x, y):
    return grad_yx * x + grad_yy * y


params = dotdict(
    project_name="two_d_affine_mix2",
    output_root="examples_output",
    dim=2,
    q_deg=2,
    elem_order=1,
    width=0.85,
    height=0.5,
    nx=5,
    ny=4,
    relative_tol=1e-5,
    absolute_tol=2e-8,
    max_iter=25,
    E=26.0,
    nu=0.31,
    dirichlet_bcs_dict={
        lambda x, y: torch.isclose(x, torch.zeros_like(x)): (ux, uy),
        lambda x, y: torch.isclose(x, torch.full_like(x, 0.85)): (ux, uy),
        lambda x, y: torch.isclose(y, torch.zeros_like(y)): (ux, uy),
        lambda x, y: torch.isclose(y, torch.full_like(y, 0.5)): (ux, uy),
    },
    neumann_bcs_dict={},
    B=(0.0, 0.0),
)
