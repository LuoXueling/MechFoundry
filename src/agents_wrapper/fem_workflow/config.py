"""Base configuration owned by the API-assisted FEM example."""

try:
    from core.utils import dotdict
except ImportError:
    from src.warp_wrapper.core.utils import dotdict

params = dotdict(
    project_name="api_fem_forward",
    output_root="output",
    async_io=True,
    dim=3,
    q_deg=2,
    elem_order=1,
    device="cuda",
    output_per_step=0,
    relative_tol=1.0e-6,
    absolute_tol=1.0e-8,
    max_iter=50,
    line_search_enabled=False,
    update_damping_limit=0.0,
    load_steps=1,
    linear_solver="cg",
    ew_enabled=False,
    cg_backend_assembled="pyamgx",
    E=1.0,
    nu=0.3,
    B=(0.0, 0.0, 0.0),
    dirichlet_bcs_dict={},
    neumann_bcs_dict={},
    phase_field_epsilon=0.01,
    update_beta_per_newton=False,
    sharp_volumetric=False,
    use_mixed_precision=True,
    tangent_f64=False,
    optimizer="gd",
    gd_lr=1.0,
    max_optimize_iters=1,
    output_per_optimize_step=1,
    optimize_relative_tol=2.0e-3,
    target_volume_fraction=0.5,
    shape_opt_w_vol=5.0,
    reinit_sdf=False,
    reinit_every=1,
    output_surface_vtu=True,
    output_surface_glb=True,
    output_surface_glb_deformed=True,
)
