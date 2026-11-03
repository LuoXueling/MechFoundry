from core.utils import dotdict
import torch

WIDTH = 4.0
HEIGHT = 1.0
HALF_HEIGHT = HEIGHT * 0.5


def right_lower_half(x, y):
    return (x >= WIDTH - 1e-6) & (y <= HALF_HEIGHT + 1e-6)


params = dotdict(
    project_name="two_d_right_boundary_force",
    output_root="examples_output",
    dim=2,
    q_deg=2,
    elem_order=1,
    width=WIDTH,
    height=HEIGHT,
    nx=32,
    ny=12,
    relative_tol=5e-4,
    max_iter=15,
    line_search_max_iters=20,
    line_search_enabled=False,
    E=25.0,
    nu=0.3,
    update_damping_limit=0.0,
    ew_enabled=False,
    linear_solver="cg",
    dirichlet_bcs_dict={
        lambda x, y: torch.isclose(x, torch.zeros_like(x)): 0.0,
    },
    neumann_bcs_dict={
        right_lower_half: (0.0, -0.5),
    },
    B=(0.0, 0.0),
)
