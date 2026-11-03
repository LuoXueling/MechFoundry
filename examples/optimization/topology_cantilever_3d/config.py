"""
Configuration file for Neo-Hookean elasticity solver using Warp

This configuration replaces the original FEniCS-based config with pure Warp parameters.
"""

import warp as wp
from core.utils import dotdict
import torch

params = dotdict(
    # ========== Project Settings ==========
    project_name="beam_rightlowercorner_load",
    output_root="output",
    async_io=True,  # Whether to use asynchronous I/O for output saving
    # ========== FEM Settings ==========
    dim=3,  # Problem dimension (2 or 3)
    q_deg=2,  # Quadrature degree
    elem_order=1,  # Element polynomial degree (1=linear, 2=quadratic)
    # ========== Geometry Settings ==========
    width=4.0,  # Domain width
    height=1.0,  # Domain height
    depth=1.0,
    nx=100,  # Number of elements in x-direction
    ny=25,  # Number of elements in y-direction
    nz=25,
    # For 3D problems, also specify:
    # depth=1.0,
    # nz=25,
    # ========== Solver Settings ==========
    # Device selection
    device="cuda",  # "cuda" or "cpu"
    # Solver (nonlinear)
    output_per_step=1,  # Save output every N steps
    relative_tol=1e-10,  # Relative convergence tolerance for Newton solver
    absolute_tol=1e-11,
    max_iter=10,  # Maximum Newton iterations
    line_search_enabled=False,
    update_damping_limit=0.0,
    load_steps=1,
    # debug_dump_linear_system=False,
    # debug_dump_iteration=None,
    linear_solver="cg",  # "cg", "matrix_free_cg", or "autodiff_matrix_free_cg"
    ew_enabled=False,
    cg_backend_assembled="amgx",
    # ========== Material Properties ==========
    E=10.0,  # Young's modulus
    nu=0.3,  # Poisson's ratio
    # ========== Body Force ==========
    B=(0.0, 0.0, 0.0),  # Body force per unit volume (x, y) or (x, y, z) for 3D
    # ========== Boundary Conditions ==========
    dirichlet_bcs_dict={
        lambda x, y, z: torch.isclose(x, torch.zeros_like(x)): 0.0,
        # lambda x, y: x > 4.0 - 0.0001: (0.0, -0.001),
    },
    neumann_bcs_dict={
        lambda x, y, z: (x >= 4.0 - 0.0001) & (y <= 0.04 * 1.1): (0.0, -0.01, 0.0)
    },
    # ========== Optimization Settings ==========
    e_fixed_region=None,  # lambda(x,y[,z]) -> bool; True = e is fixed there
    e_init=1.0,  # initial value for e field (scalar); None = use E from material
    # ========== Topology Optimization ==========
    density_filter_radius=0.01,  # Helmholtz PDE filter radius
    target_volume_fraction=0.5,  # Volume constraint target
    max_optimize_iters=1000,  # Maximum optimization iterations
    optimize_relative_tol=1e-4,  # Convergence tolerance for objective change
    mma_move_limit=0.02,  # MMA move limit
    gcmma_enabled=False,  # Enable GCMMA conservatism check; False = pure MMA
    gcmma_max_inner=15,  # Max inner iterations for conservatism check
    mma_gpu_dual=True,  # Keep MMA dual-solve data on GPU (only lambda transfers to CPU)
    # ========== Precision Settings ==========
    use_mixed_precision=True,  # True = float64 displacement solver vectors, float32 tangent K
)
