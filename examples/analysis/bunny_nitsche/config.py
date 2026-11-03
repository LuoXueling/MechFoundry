"""
Example configuration for the diffused Nitsche method on octree mesh.

This config is a template — octree_mesh, quad_octree_mesh, and quad_sdf
must be provided via override_params at runtime (they come from sdf2octree).
"""

import torch

try:
    from core.utils import dotdict
except ImportError:
    from src.warp_wrapper.core.utils import dotdict

params = dotdict(
    # ========== Project Settings ==========
    project_name="nitsche_example",
    output_root="output",
    async_io=False,
    # ========== FEM Settings ==========
    dim=3,
    q_deg=2,
    elem_order=1,
    # ========== Solver Settings ==========
    device="cuda",
    output_per_step=1,
    relative_tol=1e-8,
    absolute_tol=1e-10,
    max_iter=30,
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
    # These will be enforced via Nitsche method on the embedded geometry:
    dirichlet_bcs_dict={
        lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
    },
    neumann_bcs_dict={
        lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.5)): (0.01, 0.0, 0.0),
    },
    # ========== Nitsche Parameters ==========
    phase_field_epsilon=0.001,  # Must be <= min element size; typically ~1/10 of it
    update_beta_per_newton=False,
    # ========== Precision Settings ==========
    use_mixed_precision=False,
)
