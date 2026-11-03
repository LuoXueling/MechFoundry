"""
Default configuration parameters for Neo-Hookean solver (Warp version)

These are fallback defaults used if not specified in a user config.

Boundary conditions are provided directly as Python lambdas. For boundary
predicates, prefer torch.isclose(...) over ad-hoc tolerances to avoid accidental
spill to nearby nodes. Value specifications support scalars, tuples (per
component), callable functions, or component lists paired with a scalar or
callable.
"""

from .utils import dotdict

params = dotdict(
    # ========== Project Settings ==========
    project_name="default_simulation",
    output_root="output",
    async_io=False,  # Whether to use asynchronous I/O for output saving
    # ========== FEM Settings ==========
    dim=2,  # Problem dimension (2 or 3)
    q_deg=2,  # Quadrature degree
    elem_order=1,  # Element polynomial degree (1=linear, 2=quadratic)
    # ========== Geometry Settings ==========
    width=4.0,  # Domain width
    height=1.0,  # Domain height
    depth=1.0,  # Domain depth (3D only)
    nx=100,  # Number of elements in x-direction
    ny=25,  # Number of elements in y-direction
    nz=25,  # Number of elements in z-direction (3D only)
    # ========== Solver Settings ==========
    # Device selection
    device="cuda",  # "cuda" or "cpu"
    # Solver (nonlinear)
    output_per_step=1,  # Save output every N steps
    relative_tol=1e-5,  # Tight tolerance required for validation-quality convergence
    absolute_tol=2e-8,
    max_iter=150,  # Maximum Newton iterations
    line_search_enabled=True,
    debug_line_search=False,  # Print line search details
    update_damping_limit=0.0,
    load_steps=1,
    debug_dump_linear_system=False,
    debug_dump_iteration=None,
    linear_solver="cg",  # "cg", "matrix_free_cg", or "autodiff_matrix_free_cg"
    # Eisenstat-Walker adaptive CG tolerance
    ew_enabled=True,
    ew_gamma=0.9,
    ew_alpha=1.618,  # (1+sqrt(5))/2, golden ratio
    ew_eta_max=0.9,
    # Assembled CG backend selection
    # (only affects linear_solver="cg"; matrix-free paths always use warp CG)
    cg_backend_assembled="amgx",  # "amgx" prefers JAX-AMG then pyamgx; or force "jaxamg", "pyamgx", "native"
    amgx_rebuild_every_iter=False,  # Rebuild AMGX operator every Newton iteration
    # ========== Material Properties ==========
    E=10.0,  # Young's modulus
    nu=0.3,  # Poisson's ratio
    # ========== Body Force ==========
    # Body force per unit volume (x, y) or (x, y, z) for 3D
    B=(0.0, 0.0),
    # ========== Boundary Conditions ==========
    dirichlet_bcs_dict={},
    neumann_bcs_dict={},
    # ========== Nitsche Parameters ==========
    phase_field_epsilon=0.01,  # Phase-field half-width; typically ~h/2 where h is element size
    phi_floor=1e-4,  # Minimum phi weight to avoid zero stiffness outside domain
    sharp_volumetric=False,  # Use sharp (Heaviside) volume integration instead of diffuse
    trim_exterior=False,  # Remove elements fully outside the embedded domain
    trim_phi_threshold=1e-3,  # phi threshold for trim_exterior classification
    # nitsche_phi_threshold: phi threshold for Nitsche BC nodes; defaults to phi_floor
    # Scale each Neumann region's traction so its discrete load equals traction x
    # loaded area.  The BC band keeps only tanh(tol/eps) of the traction kernel and
    # the 2x2x2 Gauss sum depends on alignment: a rotated bar got 87.7 % of its
    # load at tol = 1.5 eps without it.
    nitsche_neumann_rescale=True,
    # bc_reference_surface: {"vertices": (V,3), "faces": (F,3)} physical surface on
    # which Neumann SurfaceSelector regions are measured for the rescale (their MC-
    # surface patch can miss a face's rim).  ShapeOptTrimmed sets it from its
    # remeshing_trimesh mesh; omit to measure on the resolved patch.
    update_beta_per_newton=False,  # Recompute beta stabilization each Newton iteration
    heuristic_beta=False,  # Use heuristic (element-wise) beta instead of eigenvalue-based
    # Safety factor multiplier for beta stabilization.  Beta also sets how hard the
    # penalty clamps u = u_bar across the diffuse Dirichlet band (which moves the
    # face ~eps inward): twist cube, depth 6, float32, rel. L2 aligned / rotated
    # 5.4e-3 / 1.10e-2 at 10, 4.1e-3 / 8.9e-3 at 5; at 3 the rotated Newton solve
    # stalls.
    beta_safety_factor=5.0,
    # beta_max: optional cap on beta values; omit for no cap
    tangent_f64=False,  # Accumulate tangent in float64 for precision
    # Symmetric Jacobi scaling of the assembled system before the Krylov solve.
    # Nitsche penalties on sliver boundary elements push row magnitudes thousands
    # of times above the elastic terms, which stalls CG in single precision; a
    # unit diagonal removes that spread.  Exact, not an approximation.
    scale_system_to_unit_diagonal=False,
    # jaxamg_max_iters: JAX-AMG Krylov iteration cap; omit for 1000.  Raise it when
    # a solve fails "still converging ... ran out of iterations".
    skip_d2P=True,  # Skip second-order stress derivative in tangent (geometric stiffness)
    # ========== Adaptive Quadrature ==========
    # geometric_func: callable(x) -> SDF for refined quadrature; omit for no refinement
    # quad_refinement_depth: subdivision depth for boundary elements; omit for auto from epsilon/h
    # quad_refinement_depth_interior: subdivision depth for non-BC boundary elements; omit = same as quad_refinement_depth
    # octree_mesh: pre-built octree mesh dict {nodes, elements, sdf}; omit = build from geometry
    # ========== Optimization Settings ==========
    optimizer="mma",  # "mma" (GCMMA/CCSA) or "gd" (gradient descent)
    gd_lr=1e-3,  # Learning rate for gradient descent optimizer
    # Automatic gd_lr (gradient descent only; off by default, and with it off the
    # loop is unchanged).  The SDF is a length and dJ/ds scales with the body, so
    # one gd_lr cannot serve parts of every size.  With it on, EVERY update reads
    # the largest |dJ/ds| over the free design nodes at that iteration and steps at
    #     gd_lr_eff(k) = gd_lr_auto_fraction * L / max|dJ/ds|_k,
    # so each step moves the SDF by exactly that fraction of L at its largest node
    # (the geometry clean-up's rule, experiments/geometry_cleanup, applied every
    # iteration).  gd_lr is then not used for the SDF step: it is only the
    # fallback for a zero or non-finite gradient (logged; a later one keeps the
    # previous rate).  pf_phi_lr's fallback still reads gd_lr.
    # This is normalized gradient descent: the step does not shrink as the
    # gradient does, so near an optimum the design keeps moving by fraction * L
    # and the objective oscillates instead of settling.  The relative-tolerance
    # test (optimize_relative_tol, from iteration 20) may then never pass, and the
    # run ends at max_optimize_iters (or, past 500 iterations, after 50 without a
    # new best): set max_optimize_iters to the run you want, and lower the
    # fraction for a finer final shape.
    gd_lr_auto=False,
    gd_lr_auto_fraction=0.5,  # every step's largest move, as a fraction of L
    # gd_lr_auto_length: the reference length L; omit for the diffuse width
    # phase_field_epsilon (a field without one must pass it)
    pf_phi_lr=None,  # Phase-field mode hidden-phi LR; None -> fallback to gd_lr
    pf_phi_start_iter=0,  # First iteration eligible for deep-interior hidden-phi updates
    e_fixed_region=None,  # lambda(x,y[,z]) -> bool; True = e is fixed there
    e_init=None,  # initial value for e field (scalar); None = use E from material
    lr=1e-3,  # learning rate for optimization step
    # ========== Topology Optimization ==========
    density_filter_radius=0.01,  # Helmholtz PDE filter radius
    simp_beta=3.0,  # SIMP projection sharpness parameter
    simp_eta=0.5,  # SIMP projection threshold
    target_volume_fraction=0.5,  # Volume constraint target
    volume_as_is=False,  # If True, penalize volume above initial (target_volume_fraction ignored)
    max_optimize_iters=1000,  # Maximum optimization iterations
    optimize_relative_tol=1e-4,  # Convergence tolerance for objective change
    output_per_optimize_step=1,  # Save output every N optimization steps
    mma_move_limit=0.2,  # MMA move limit
    gcmma_enabled=True,  # Enable GCMMA conservatism check; False = pure MMA
    gcmma_max_inner=15,  # Max inner iterations for conservatism check
    mma_gpu_dual=False,  # Keep MMA dual-solve data on GPU (only lambda transfers to CPU)
    # ========== Shape Optimization ==========
    warm_start_u=True,  # Interpolate previous displacement as Newton initial guess on remesh
    design_variable="sdf",  # "sdf" (default) or "phase_field" (hidden-phi interior carving)
    phi_reproject_band_k=3.0,  # Phase-field: boundary-tracking band width in units of epsilon
    shape_opt_w_vol=10.0,  # Volume penalty weight in objective
    latent_weight_decay=0.0,  # Adam weight decay for latent shape optimization; ignored by "rmsprop"
    latent_optimizer="adam",  # Latent optimizer: "adam" (AdamW), "rmsprop" (RMSProp, no weight decay), or "sgd" (SGD+momentum, L2-normalized grad)
    # ----- Latent pull-back-metric correction (LatentShapeOpt precond dict) -----
    # Defaults for the pull-back CG that corrects the latent gradient by solving
    # (J^T J + λI) x = g_raw (J = decoder Jacobian).  Overridden per-key by the
    # ``precond`` dict passed to LatentShapeOpt (see examples/optimization/latent_shape_opt).
    latent_precond_enabled=True,  # Apply the pull-back CG correction; False = raw VJP gradient, L2-rescaled to sqrt(N)
    latent_precond_damping=1e-2,  # Tikhonov damping ratio × Rayleigh-quotient curvature scale for the CG solve
    latent_precond_cg_iters=20,  # Max matrix-free CG iterations
    latent_precond_cg_rtol=1e-2,  # CG relative residual tolerance (truncated/early-stopped solve)
    latent_precond_cg_warm_start=True,  # Warm-start CG from the previous iteration's solution
    latent_precond_cg_matvec="jvp",  # J^T J matvec: "jvp" (forward-JVP+VJP) or "reverse" (double-backward, ~40GB VRAM, opt-in)
    vol_softplus_sharpness=500.0,  # Softplus sharpness for one-sided volume penalty
    # ----- SDF Reinitialization -----
    reinit_sdf=True,  # Reinitialize SDF to enforce |∇φ|=1 after each optimizer step
    reinit_every=1,  # Reinitialize SDF every N optimizer steps
    reinit_sdf_method="sussman",  # "sussman" (eikonal PDE) or "marchingcubes" (BVH query)
    # reinit_n_iters: Sussman reinit iterations; omit = max(grid_shape)
    # ----- Remeshing (Adaptive Octree) -----
    remeshing=False,  # Master switch: rebuild outer octree from cleaned SDF each iteration
    remeshing_method="interpolate",  # Octree geo_func: "interpolate" (trilinear), "marchingcubes" (BVH), or "decoder_marchingcubes" (decoder grid re-query)
    remesh_octree_depth=7,  # Target adaptive octree depth; leaf size ≈ domain / 2^depth
    remesh_init_octree_depth=4,  # Initial coarse octree depth (set == remesh_octree_depth for uniform hex)
    sdf_bc_fix_method="mask",  # BC region preservation: "mask" (freeze), "graft" (snap to init surface), or "zip" (exact patch)
    # remesh_clip_bbox: ((xlo,ylo,zlo),(xhi,yhi,zhi)) — physical bbox to clip outer octree; omit for no clip
    # remeshing_trimesh: path-A geometry dict {mesh, mesh_center, h_target[, clip_bbox]}
    # remeshing_geo_func: path-B geometry dict {geo_func, physical_extents or (axis_scale+geo_min+h_target)[, filter_sdf_func, clip_bbox]}
    # ----- Surface Output (Shape Optimization) -----
    output_surface_vtu=False,  # Marching-cubes surface mesh as VTU (ParaView)
    output_surface_glb=False,  # Undeformed surface mesh as GLB
    output_surface_glb_deformed=False,  # Deformed surface mesh as GLB
    surface_simplify_n_faces=-1,  # Simplify surface to N faces (-1 = no simplification)
    # ----- Analytical SDF (LatentShapeOpt) -----
    analytical_sdf_full=False,  # True = analytical SDF at every octree node; False = only nodes within 4*phase_field_epsilon of the surface
    # ----- Precision Settings -----
    # True selects Warp's native float64 FEM geometry, fields, and integration.
    use_mixed_precision=False,  # False gives an all-f32 JAX-AMG solve; pyamgx remains dDDI
)

# =============================================================================
# BOUNDARY CONDITION EXAMPLES
# =============================================================================
# Boundaries may be vectorized lambdas or SurfaceSelector objects.  For an
# automatically expanded 3-D part, use ShapeDiameterSegmentSelector(seed),
# where seed is a point/list of points or a simple SurfaceSelector.
# Prefer torch.isclose(...) for coordinate-aligned boundaries.
# Use & instead of Python `and` for combining conditions (torch tensors).
#
# BOUNDARY PREDICATES (dict keys) — examples:
# import torch
# w, h = 4.0, 1.0
# - Left edge:   lambda x, y: torch.isclose(x, torch.zeros_like(x))
# - Right edge:  lambda x, y: torch.isclose(x, torch.full_like(x, w))
# - Bottom edge: lambda x, y: torch.isclose(y, torch.zeros_like(y))
# - Top edge:    lambda x, y: torch.isclose(y, torch.full_like(y, h))
# - Circular:    lambda x, y: (x-2.0)**2 + (y-0.5)**2 < 0.3**2
# - Strip:       lambda x, y: (y >= 0.4) & (y <= 0.6)
# 3D planes use (x, y, z) with torch.isclose on desired coordinate.
#
# VALUE SPECIFICATIONS (dict values):
# - Scalar: 0.0            # all components
# - Tuple: (0.05, 0.0)     # per-component
# - Callable: lambda x, y: 0.01*y
# - Tuple of callables: (lambda x, y: 0.01*y, lambda x, y: 0.0)
# - Component list + scalar: ([0], 0.02)      # x-component only
# - Component list + callable: ([0], lambda x, y: 0.02*y)
#   For 3D, use 3-tuples and component indices in {0,1,2}.
#
# DIRICHLET BC EXAMPLES (displacement constraints):
# import torch
# w, h = 4.0, 1.0
# dirichlet_bcs_dict={
#     lambda x, y: torch.isclose(x, torch.zeros_like(x)): 0.0,                      # Fixed left edge (scalar)
#     lambda x, y: torch.isclose(x, torch.full_like(x, w)): 0.05,                   # Constant displacement (scalar)
#     lambda x, y: torch.isclose(x, torch.full_like(x, w)): (0.05, 0.0),            # Per-component displacement (tuple)
#     lambda x, y: torch.isclose(x, torch.full_like(x, w)): lambda x, y: 0.01*y,    # Spatially varying (single lambda)
#     lambda x, y: torch.isclose(x, torch.full_like(x, w)): (lambda x, y: 0.01*y*(h-y), lambda x, y: 0.0),# Parabolic profile
#     lambda x, y: torch.isclose(y, torch.full_like(y, h)): ([0], 0.02),            # Constrain x-component on top
#     lambda x, y: (x-2.0)**2 + (y-0.5)**2 < 0.3**2: (0.1, 0.0),                   # Circular region
# }
#
# NEUMANN BC EXAMPLES (traction/force boundary conditions):
# import torch
# w, h = 4.0, 1.0
# neumann_bcs_dict={
#     lambda x, y: torch.isclose(x, torch.full_like(x, w)): (1.0, 0.0),             # Constant traction (scalars)
#     lambda x, y: torch.isclose(y, torch.full_like(y, h)): (0.0, -0.5),            # Pressure on top edge
#     lambda x, y: torch.isclose(x, torch.full_like(x, w)): (lambda x, y: 0.1*y, lambda x, y: 0.0),  # Spatially varying
# }
