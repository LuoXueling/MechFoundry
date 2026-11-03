from core.utils import dotdict

grad = (
    (0.004, -0.002, 0.0015),
    (0.0035, 0.0025, -0.002),
    (-0.001, 0.002, 0.003),
)


def ux(x, y, z):
    return grad[0][0] * x + grad[0][1] * y + grad[0][2] * z


def uy(x, y, z):
    return grad[1][0] * x + grad[1][1] * y + grad[1][2] * z


def uz(x, y, z):
    return grad[2][0] * x + grad[2][1] * y + grad[2][2] * z


params = dotdict(
    project_name="three_d_affine_mix1",
    output_root="examples_output",
    dim=3,
    q_deg=2,
    elem_order=1,
    width=0.8,
    height=0.28,
    depth=0.32,
    nx=3,
    ny=2,
    nz=3,
    relative_tol=1e-5,
    absolute_tol=2e-8,
    max_iter=30,
    E=26.0,
    nu=0.29,
    dirichlet_bcs_dict={
        lambda x, y, z: x < 0.01: (ux, uy, uz),
        lambda x, y, z: x > 0.8 - 0.01: (ux, uy, uz),
        lambda x, y, z: y < 0.01: (ux, uy, uz),
        lambda x, y, z: y > 0.28 - 0.01: (ux, uy, uz),
        lambda x, y, z: z < 0.01: (ux, uy, uz),
        lambda x, y, z: z > 0.32 - 0.01: (ux, uy, uz),
    },
    neumann_bcs_dict={},
    B=(0.0, 0.0, 0.0),
)
