from core.utils import dotdict, make_left_edge, make_right_edge

WIDTH = 4.0
HEIGHT = 1.0
HALF_HEIGHT = HEIGHT * 0.5


def right_lower_half(x, y):
    return (x >= WIDTH - 1e-6) and (y <= HALF_HEIGHT + 1e-6)


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
    relative_tol=1e-5,
    max_iter=60,
    line_search_max_iters=20,
    line_search_enabled=False,
    E=25.0,
    nu=0.3,
    update_damping_limit=0.0,
    dirichlet_bcs_dict={
        make_left_edge(): 0.0,
    },
    neumann_bcs_dict={
        make_right_edge(WIDTH): (0.05, 0.0),
    },
    B=(0.0, 0.0),
)
