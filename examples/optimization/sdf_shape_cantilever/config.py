"""
Configuration for shape optimization of a 3D cantilever beam
using the diffused Nitsche method with Adam optimizer.

Geometry: [0, 4] x [0, 1] x [0, 1] box embedded in a background mesh
          with padding for SDF evolution.
BCs:      Clamped at x=0 face, downward traction at x=4 lower edge.
Objective: SE/SE_0 + w_vol*(V-V_target)/norm + w_eik*eikonal/norm
"""

import torch

try:
    from core.utils import dotdict
except ImportError:
    from src.warp_wrapper.core.utils import dotdict

params = dotdict(
    # ========== Project Settings ==========
    project_name="shape_opt_cantilever",
    output_root="output",
    async_io=True,
    # ========== FEM Settings ==========
    dim=3,
    q_deg=2,
    elem_order=1,
    # ========== Solver Settings ==========
    device="cuda",
    output_per_step=0,
    # Preserve the reference relative tolerance. On the documented depth-6
    # and default depth-7 meshes, native-f32 Newton reaches a sub-4e-7
    # absolute residual before its relative residual plateaus, so use an
    # absolute floor appropriate to that precision.
    relative_tol=1e-3,
    absolute_tol=5e-7,
    max_iter=20,
    line_search_enabled=False,
    update_damping_limit=0.0,
    load_steps=1,
    linear_solver="cg",
    ew_enabled=False,
    cg_backend_assembled="amgx",
    # ========== Material Properties ==========
    E=10.0,
    nu=0.3,
    # ========== Body Force ==========
    B=(0.0, 0.0, 0.0),
    # ========== Boundary Conditions ==========
    # Clamped at x=0
    dirichlet_bcs_dict={
        lambda x, y, z: torch.isclose(x, torch.zeros_like(x))
        & (y < 1.05)
        & (y > -0.05)
        & (z < 1.05)
        & (z > -0.05): 0.0,
    },
    # Downward traction at the lower edge of the x=4 face
    neumann_bcs_dict={
        lambda x, y, z: torch.isclose(x, torch.full_like(x, 4.0))
        & (y <= 0.15): (0.0, -0.0025, 0.0),
    },
    # ========== Nitsche Parameters ==========
    phase_field_epsilon=None,
    update_beta_per_newton=False,
    sharp_volumetric=False,
    # ========== Precision Settings ==========
    # Native f32 ShapeOpt path, including SDF/phase-field objective fields.
    use_mixed_precision=False,
    tangent_f64=False,
    # ========== Optimization Settings ==========
    optimizer="gd",
    gd_lr=1.0,
    max_optimize_iters=2000,
    output_per_optimize_step=10,
    optimize_relative_tol=2e-3,
    # ========== Shape Optimization ==========
    target_volume_fraction=0.5,
    shape_opt_w_vol=5.0,
    # ----- SDF Reinitialization -----
    reinit_sdf=True,
    reinit_every=1,
    reinit_n_iters=20,
    # ----- Surface Output -----
    output_surface_vtu=True,
    output_surface_glb=True,
    output_surface_glb_deformed=True,
)
