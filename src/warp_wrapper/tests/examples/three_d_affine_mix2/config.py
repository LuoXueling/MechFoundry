from core.utils import dotdict

grad = (
    (-0.003, 0.0045, -0.002),
    (0.0025, 0.0015, 0.003),
    (0.001, -0.0025, 0.002),
)


def ux(x, y, z):
    return grad[0][0] * x + grad[0][1] * y + grad[0][2] * z


def uy(x, y, z):
    return grad[1][0] * x + grad[1][1] * y + grad[1][2] * z


def uz(x, y, z):
    return grad[2][0] * x + grad[2][1] * y + grad[2][2] * z


params = dotdict(
    project_name="three_d_affine_mix2",
    output_root="examples_output",
    dim=3,
    q_deg=2,
    elem_order=1,
    width=0.82,
    height=0.35,
    depth=0.34,
    nx=3,
    ny=3,
    nz=3,
    relative_tol=1e-5,
    absolute_tol=2e-8,
    max_iter=30,
    E=27.0,
    nu=0.3,
    dirichlet_bcs_dict={
        lambda x, y, z: x < 0.01: (ux, uy, uz),
        lambda x, y, z: x > 0.82 - 0.01: (ux, uy, uz),
        lambda x, y, z: y < 0.01: (ux, uy, uz),
        lambda x, y, z: y > 0.35 - 0.01: (ux, uy, uz),
        lambda x, y, z: z < 0.01: (ux, uy, uz),
        lambda x, y, z: z > 0.34 - 0.01: (ux, uy, uz),
    },
    neumann_bcs_dict={},
    B=(0.0, 0.0, 0.0),
)
