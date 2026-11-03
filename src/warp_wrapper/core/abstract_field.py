"""
Neo-Hookean Hyperelasticity Solver using NVIDIA Warp

This is a complete replacement of the FEniCS implementation with pure Warp code.
All functionality is self-contained with inlined methods from Warp modules.
"""

import warp as wp
import warp.fem as fem
import numpy as np
import torch
import json
import os
import importlib
import shutil
from functools import partial
from datetime import datetime
import pickle
import pandas as pd
import math
from typing import Union, Callable, List, Optional, Any
from .utils import (
    log,
    FlexibleBC,
    process_flexible_bc,
    build_bc_projector,
    _scale_inplace,
    masked_norm,
    check_finite,
    max_node_norm,
    _copy_masked,
    _matvec_bc_combine,
    _mask_inplace,
    _apply_bc_to_field,
    lump_boundary_tractions_2d,
    lump_boundary_tractions_3d,
    solve_linear_system,
    backtracking_line_search,
    _prolongate_edge_hanging,
    _prolongate_face_hanging,
    _restrict_edge_hanging,
    _restrict_face_hanging,
    _zero_slave_entries,
    _enforce_edge_hanging_3d,
    _enforce_face_hanging_3d,
    _gradient_descent_step,
    _clamp_inplace,
    _simp_projection_kernel,
    _axpy_f32,
)
from .default_config import params
from .vtu_writer import (
    VTUWriter,
    extract_mesh_info_from_warp,
    compute_node_gradients,
    compute_strain_tensor,
    compute_stress_tensor,
)


class AbstractField:
    """
    Abstract field class using NVIDIA Warp

    Completely self-contained implementation with:
    - Mesh generation inlined from geometry_warp
    - Function spaces inlined from function_space_warp
    - Material model inlined from material_warp
    - Boundary conditions inlined from boundary_conditions_warp
    - Newton solver inlined from solver_warp
    - VTK export for ParaView visualization
    """

    def __init__(self, config_path, override_params=None):
        """
        Initialize AbstractField

        Args:
            config_path: Path to configuration file (without .py)
            override_params: Dictionary of parameters to override
        """
        # Warp init + config is done in src/__init__.py at import time

        # Normalize config module path (supports dotted or file-style inputs)
        module_path = config_path.replace("/", ".")
        if module_path.endswith(".py"):
            module_path = module_path[:-3]

        config_module = importlib.import_module(module_path)
        params_updated = config_module.params
        self.params = params.copy()
        self.params.update(params_updated)
        if override_params is not None:
            self.params.update(override_params)

        # Device selection (after params loaded)
        self.device = str(self.params.get("device", "cuda"))
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"Unsupported device: {self.device}")
        try:
            wp.set_device(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to set Warp device '{self.device}': {e}")

        # Initialize output directory
        self.path = os.path.join(
            self.params.output_root,
            f"{self.params.project_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        )
        os.makedirs(self.path, exist_ok=True)
        self.log = partial(log, output_file=os.path.join(self.path, "log.txt"))
        config_file_src = getattr(config_module, "__file__", None)
        if config_file_src is None:
            raise FileNotFoundError(
                f"Could not locate config file for module {module_path}"
            )
        shutil.copy2(config_file_src, os.path.join(self.path, "config.py"))

        if override_params is not None:
            # Strip heavy / non-pickleable entries (callables, tensors, arrays,
            # dicts/lists containing them).  Bound-method overrides like
            # ``geometric_func`` would otherwise drag self into the pickle and
            # blow the file up to GBs; ``octree_mesh`` carries large tensors.
            def _is_lightweight(v):
                if callable(v):
                    return False
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    return False
                if isinstance(v, dict):
                    return all(_is_lightweight(x) for x in v.values())
                if isinstance(v, (list, tuple)):
                    return all(_is_lightweight(x) for x in v)
                return True

            slim = {k: v for k, v in override_params.items() if _is_lightweight(v)}
            try:
                with open(os.path.join(self.path, "config_override.pkl"), "wb") as file:
                    pickle.dump(slim, file)
            except Exception:
                pass

        # Dimension and dtype
        self.dim = self.params.dim
        self.use_mixed_precision = bool(self.params.get("use_mixed_precision", False))
        _mixed = self.use_mixed_precision
        if self.dim == 2:
            self.dtype = wp.vec2d if _mixed else wp.vec2
        elif self.dim == 3:
            self.dtype = wp.vec3d if _mixed else wp.vec3
        else:
            raise ValueError(f"Unsupported dimension: {self.dim}")

        if 1 <= self._design_dim <= 4:
            self.design_dtype = {1: float, 2: wp.vec2, 3: wp.vec3, 4: wp.vec4}[
                self._design_dim
            ]
        else:
            raise ValueError(f"Unsupported design dimension: {self._design_dim}")
        # In mixed precision, density DOF is float64 so all FEM fields on the
        # float64 geometry share the same scalar type (avoids weight/DOF mismatches).
        if _mixed and self.dim in (2, 3) and self._design_dim == 1:
            self.design_dtype = wp.float64

        # Material parameters
        self._has_spatial_material = bool(self.params.get("material_artifact", None))
        self.mu = float(self.params.E / (2.0 * (1.0 + self.params.nu)))
        self.lmbda = float(
            self.params.E
            * self.params.nu
            / ((1.0 + self.params.nu) * (1.0 - 2.0 * self.params.nu))
        )
        self.kappa = float(self.params.E / (3.0 * (1.0 - 2.0 * self.params.nu)))

        # Per-unit-E Lame parameters (for future spatially-varying E via self.e)
        self.mu_per_E = float(1.0 / (2.0 * (1.0 + self.params.nu)))
        self.kappa_per_E = float(1.0 / (3.0 * (1.0 - 2.0 * self.params.nu)))

        # Prebuild integrands specialized to dimension
        _scalar_dtype = wp.float64 if _mixed else wp.float32
        self._integrand_scalar_dtype = _scalar_dtype
        _material_factory_args = (
            {"spatial_material": True} if self._has_spatial_material else {}
        )
        self._stress_integrand = self._make_stress_integrand(
            self.dim, scalar_dtype=_scalar_dtype, **_material_factory_args
        )
        try:
            self._tangent_integrand = self._make_tangent_integrand(
                self.dim, scalar_dtype=_scalar_dtype,
                **_material_factory_args,
            )
        except NotImplementedError:
            self._tangent_integrand = None
        _body_kwargs = {}
        if self._has_spatial_material:
            _body_kwargs = {
                "spatial_material": True,
                "acceleration": tuple(self.params.get("material_acceleration", (0.0,) * self.dim)),
            }
        self._body_force_integrand = self._make_body_force_integrand(
            self.dim,
            tuple(self.params.get("B", (0.0,) * (3 if self.dim == 3 else 2))),
            scalar_dtype=_scalar_dtype, **_body_kwargs,
        )

        self.records = pd.DataFrame()
        self.records.index.name = "Step"

        # Build all mesh-dependent state
        self._build_mesh_state()

        self.log(f"Finish initialization")

    def _build_mesh_state(self):
        """(Re)build all mesh-dependent state from current self.params.

        Creates geometry, function space, displacement field, boundary
        conditions, VTU writer, and resets solver tracking state.
        Called from __init__.
        """
        scalar_type = wp.float64 if self.use_mixed_precision else wp.float32
        if self.dim == 2:
            vec_type = wp.vec2d if self.use_mixed_precision else wp.vec2
            self.geo = fem.Grid2D(
                res=wp.vec2i(self.params.nx, self.params.ny),
                bounds_lo=vec_type(0.0, 0.0),
                bounds_hi=vec_type(self.params.width, self.params.height),
                scalar_type=scalar_type,
            )
        elif self.dim == 3:
            vec_type = wp.vec3d if self.use_mixed_precision else wp.vec3
            self.geo = fem.Grid3D(
                res=wp.vec3i(self.params.nx, self.params.ny, self.params.nz),
                bounds_lo=vec_type(0.0, 0.0, 0.0),
                bounds_hi=vec_type(
                    self.params.width, self.params.height, self.params.depth
                ),
                scalar_type=scalar_type,
            )

        self.space = fem.make_polynomial_space(
            geo=self.geo, degree=self.params.elem_order, dtype=self.dtype
        )
        self.u = self.space.make_field()

        # Scalar optimization field (same geometry/degree, scalar dtype)
        self.e_space = fem.make_polynomial_space(
            geo=self.geo, degree=1, dtype=self.design_dtype
        )
        self.e = self.e_space.make_field()
        self._init_e_field()
        self._build_e_fixed_mask()
        if self._has_spatial_material:
            self._init_material_field()

        self.dirichlet_bcs = []
        self.neumann_bcs = []
        self.affine_constraints = []
        self.load_boundary_conditions()

        mesh_info = extract_mesh_info_from_warp(self.geo, self.space, self.dim)
        async_io = getattr(self.params, "async_io", False)
        self.vtu_writer = VTUWriter(
            self.path, mesh_info, async_io=async_io, log=self.log
        )
        self._mesh_cache = {}

        self.residual_history = []
        self.relative_residual_history = []
        self.newton_iterations = 0
        self.initial_residual = None
        self.final_residual = None
        self.final_relative_residual = None
        self.line_search_alphas = []
        self.current_load_scale = 1.0

        # Initialize solver cache into _mesh_cache
        self._init_solver_cache()

    def _init_solver_cache(self):
        """Initialize solver infrastructure into ``_mesh_cache``.

        Called from _build_mesh_state(). Safe to call multiple times (idempotent).
        Adds solver keys to the existing ``_mesh_cache`` dict (which may already
        contain "points"/"cells" set by the subclass).
        """
        n_nodes = self.space.node_count()
        _scalar = wp.float64 if self.use_mixed_precision else wp.float32
        vecd_type = wp.types.vector(length=self.dim, dtype=_scalar)
        mc = self._mesh_cache

        # Domains and test/trial functions
        mc["domain"] = fem.Cells(geometry=self.geo)
        mc["boundary"] = fem.BoundarySides(self.geo)
        mc["test"] = fem.make_test(space=self.space, domain=mc["domain"])
        mc["trial"] = fem.make_trial(space=self.space, domain=mc["domain"])
        mc["bd_test"] = fem.make_test(space=self.space, domain=mc["boundary"])
        mc["bd_trial"] = fem.make_trial(space=self.space, domain=mc["boundary"])

        # Solver config
        mc["solver_type"] = str(self.params.get("linear_solver", "cg")).lower()
        mc["cg_backend"] = str(self.params.get("cg_backend_assembled", "amgx")).lower()

        # Common reusable buffers
        mc["du_update"] = wp.zeros(n_nodes, dtype=self.dtype, device=self.device)
        mc["scalar_buf"] = wp.zeros(1, dtype=wp.float64, device=self.device)

        # Solver-specific buffers
        if mc["solver_type"] == "matrix_free_cg":
            mc["du_scratch"] = self.space.make_field()
            mc["x_masked_buf"] = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)
            mc["Kx_buf"] = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)
            mc["tmp_store"] = fem.TemporaryStore()

        # Scratch buffers for constraint restriction / RHS masking
        mc["R_restrict_buf"] = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)
        mc["b_fixed_buf"] = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)

        # BSR matrix infrastructure (topology-dependent, reused across load steps)
        mc["K"] = None
        mc["K_extended"] = None
        mc["scatter_mapping"] = None
        mc["actual_nnz_std"] = 0
        mc["extended_differs"] = False
        # Unlike the per-step scratch cache, this survives load/time steps on
        # the same mesh.  JAX-AMG needs it to retain the expanded scalar-CSR
        # pattern and its hierarchy; pyamgx deliberately continues to use the
        # per-step cache below.
        # A matching pattern may already exist in JAX-AMG's process-global
        # cache from a different field/problem.  Reset it once with this
        # problem's coefficients; genuine pattern misses are cold setups
        # already, so the flag adds no work after a topology-changing remesh.
        mc["jaxamg_solve_cache"] = {"jaxamg_force_rebuild": True}

        # Constraint condensation state
        mc["has_cond"] = self._has_affine_constraints
        if mc["has_cond"]:
            mc["cond_buf_a"] = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)
        else:
            mc["cond_buf_a"] = None

        # Assemble external loads (constant across load steps;
        # scaled by current_load_scale at use-time in _evaluate_residual).
        if len(self.neumann_bcs) > 0:
            mc["neumann"] = self._assemble_neumann_bcs()
        else:
            mc["neumann"] = None
        self._refresh_body_force_cache()

        # _step_cache is populated fresh in step(); just ensure attr exists.
        self._step_cache = {}

    def _body_force_cache_ready(self):
        """Whether all fields/quadratures needed by body assembly are current.

        The base discretization is complete when ``_init_solver_cache`` runs.
        Derived formulations whose load depends on later-built mesh state can
        override this hook and refresh the cache once that state is ready.
        """
        return True

    def _refresh_body_force_cache(self):
        """Assemble the constant forward body load from current mesh objects."""
        mc = self._mesh_cache
        has_body = (
            hasattr(self.params, "B")
            and self.params.B is not None
            and any(b != 0.0 for b in self.params.B)
        ) or (
            self._has_spatial_material
            and any(a != 0.0 for a in self.params.get("material_acceleration", ()))
        )
        if has_body and self._body_force_cache_ready():
            mc["body"] = self._assemble_body_forces(mc["domain"], mc["test"])
        else:
            # Never retain a vector assembled on an earlier mesh while a
            # derived class is rebuilding its design-dependent load state.
            mc["body"] = None

    def load_boundary_conditions(self):
        """
        Load boundary conditions from configuration

        INLINE from boundary_conditions_warp and geometry_warp

        Supports lambda function boundaries only:
        - Function predicates: lambda x, y: x < 0.1
        - BC values:
          - Scalars: 0.0 (applies to all components)
          - Tuples: (0.0, 1.0) for component-wise values
          - Direction + value: ([0], 1.0) constrains only x-component
          - Functions: lambda x, y: x**2 for spatially varying BCs
          - Tuple of functions: (lambda x,y: 0, lambda x,y: x**2)
        """

        # Process Dirichlet BCs from dict
        # Keys MUST be lambda functions (boundary predicates)
        # Values can be: scalars, tuples, OR lambda functions
        node_positions = (
            self._node_positions_torch()
        )  # zero-copy GPU torch tensor, reused across BCs
        if (
            hasattr(self.params, "dirichlet_bcs_dict")
            and self.params.dirichlet_bcs_dict
        ):
            for boundary_spec, value in self.params.dirichlet_bcs_dict.items():
                # Boundary must be a callable (lambda function)
                if not callable(boundary_spec):
                    raise ValueError(
                        f"Boundary key must be a callable (lambda function). "
                        f"Got: {type(boundary_spec).__name__}. "
                        f"Use lambda functions like: lambda x, y: x < 0.1"
                    )

                # Use add_flexible_bc for all boundaries
                self.add_flexible_bc(
                    boundary_spec,
                    value,
                    bc_type="dirichlet",
                    _node_positions=node_positions,
                )

        # Process Neumann BCs
        if hasattr(self.params, "neumann_bcs_dict") and self.params.neumann_bcs_dict:
            for boundary_spec, traction in self.params.neumann_bcs_dict.items():
                # Boundary must be a callable (lambda function)
                if not callable(boundary_spec):
                    raise ValueError(
                        f"Neumann boundary key must be a callable (lambda function). "
                        f"Got: {type(boundary_spec).__name__}"
                    )

                # Traction can be: tuple of scalars or tuple of functions
                # Type check
                if isinstance(traction, tuple):
                    # Check length matches dimension
                    if len(traction) != self.dim:
                        raise ValueError(
                            f"Neumann traction tuple length {len(traction)} must match dimension {self.dim}"
                        )
                    # All elements must be either scalars or functions
                    if all(isinstance(t, (int, float)) for t in traction):
                        # Tuple of scalars - supported
                        pass
                    elif all(callable(t) for t in traction):
                        # Tuple of functions - supported
                        pass
                    else:
                        raise ValueError(
                            f"Neumann traction tuple must contain all scalars or all functions, got mixed types"
                        )
                else:
                    raise ValueError(
                        f"Neumann traction must be a tuple. Got: {type(traction)}"
                    )

                # Use add_flexible_bc for Neumann BCs
                self.add_flexible_bc(
                    boundary_spec,
                    traction,
                    bc_type="neumann",
                    _node_positions=node_positions,
                )

        # Pre-compute GPU BC arrays (reused by forward() and load_octree_mesh)
        bc_mask, bc_values = self._get_bc_mask_and_values()  # torch on GPU
        n_nodes = self.space.node_count()

        self.bc_mask = bc_mask  # torch.BoolTensor on GPU
        self._bc_values_full = bc_values.clone()  # torch.float64 on GPU

        _scalar = getattr(self.dtype, "_wp_scalar_type_", self.dtype)
        _torch_dtype = torch.float64 if _scalar == wp.float64 else torch.float32
        self.bc_values_f32 = wp.from_torch(
            bc_values.to(dtype=_torch_dtype).contiguous(), dtype=self.dtype
        )

        _mask_vec_dtype = self.dtype
        bc_float = bc_mask.to(dtype=_torch_dtype)
        free_float = 1.0 - bc_float
        self.free_mask_wp = wp.from_torch(
            free_float.contiguous(), dtype=_mask_vec_dtype
        )
        self.bc_mask_wp = wp.from_torch(bc_float.contiguous(), dtype=_mask_vec_dtype)

        self.free_mask_f32 = wp.zeros(n_nodes, dtype=self.dtype, device=self.device)
        self.bc_mask_f32 = wp.zeros(n_nodes, dtype=self.dtype, device=self.device)
        wp.utils.array_cast(in_array=self.free_mask_wp, out_array=self.free_mask_f32)
        wp.utils.array_cast(in_array=self.bc_mask_wp, out_array=self.bc_mask_f32)

        # bd_matrix must match K's scalar type for project_system_matrix.
        _k_scalar = self._tangent_accumulate_dtype()
        self.bd_matrix = build_bc_projector(
            self.space, self.bc_mask, self.device, scalar_type=_k_scalar
        )

        # Build Dirichlet-only mask (excludes hanging node slaves).
        # Used by the adjoint tape pass to mask only Dirichlet BC DOFs.
        dirichlet_only_mask = torch.zeros(
            n_nodes, self.dim, dtype=torch.bool, device=self.device
        )
        for nodes, value, components in self.dirichlet_bcs:
            if components is None:
                dirichlet_only_mask[nodes, :] = True
            else:
                for comp in components:
                    dirichlet_only_mask[nodes, comp] = True
        dirichlet_free_float = 1.0 - dirichlet_only_mask.to(dtype=_torch_dtype)
        self.dirichlet_free_mask_wp = wp.from_torch(
            dirichlet_free_float.contiguous(), dtype=_mask_vec_dtype
        )

        self._build_affine_constraint_gpu_data()

    def add_flexible_bc(
        self,
        boundary_spec: Callable,
        value_spec: Union[float, tuple, Callable, List],
        components: Optional[List[int]] = None,
        bc_type: str = "dirichlet",
        _node_positions=None,
    ):
        """
        Add a flexible boundary condition

        Args:
            boundary_spec: Function predicate (lambda) defining the boundary
                          Example: lambda x, y: x**2 + y**2 < 1.0
            value_spec: BC value specification:
                       - Scalar: 0.0 (all components)
                       - Tuple: (1.0, 0.0) for component values
                       - Direction + value: ([0], 1.0) for x-component only
                       - Function: lambda x, y: x**2
                       - Tuple of functions: (lambda x,y: 0, lambda x,y: x**2)
            components: Optional list of components to constrain
            bc_type: "dirichlet" or "neumann"

        Examples:
            # Fixed left edge
            solver.add_flexible_bc(lambda x, y: x < 0.01, 0.0)

            # Circular region with parabolic BC
            solver.add_flexible_bc(
                lambda x, y: x**2 + y**2 < 0.5**2,
                lambda x, y: x**2
            )

            # Constrain only x-component on top
            h = solver.params.height
            solver.add_flexible_bc(lambda x, y: y > h - 0.01, ([0], 0.1))

            # Spatially varying BC
            w = solver.params.width
            solver.add_flexible_bc(
                lambda x, y: x > w - 0.01,
                (lambda x, y: 0.01*y, lambda x, y: 0)
            )
        """
        if not callable(boundary_spec):
            raise TypeError(
                f"boundary_spec must be callable (lambda function), got {type(boundary_spec).__name__}"
            )
        node_positions = (
            _node_positions
            if _node_positions is not None
            else self._node_positions_torch()
        )
        if bc_type.lower() == "dirichlet":
            # Process the flexible BC
            nodes, values, comp_list = process_flexible_bc(
                FlexibleBC(boundary_spec, value_spec, components),
                self.dim,
                self.params,
                node_positions,
            )

            # Add to dirichlet_bcs list
            # Store values per node
            self.dirichlet_bcs.append((nodes, values, comp_list))

            self.log(
                f"Added flexible Dirichlet BC: {len(nodes)} nodes, "
                f"components={comp_list if comp_list else 'all'}"
            )

        elif bc_type.lower() == "neumann":
            # For Neumann BCs, value_spec should be traction tuple
            # value_spec can be: tuple of scalars or tuple of functions

            # Process the flexible BC to identify nodes and evaluate traction values
            nodes, traction_values, comp_list = process_flexible_bc(
                FlexibleBC(boundary_spec, value_spec, components),
                self.dim,
                self.params,
                node_positions,
            )

            # Store as (nodes, traction_values) tuple
            # Note: Unlike Dirichlet BCs, Neumann BCs don't constrain specific components
            # The traction is applied to the boundary
            self.neumann_bcs.append((nodes, traction_values))
            self.log(
                f"Added flexible Neumann BC: {len(nodes)} nodes, "
                f"traction shape={traction_values.shape}"
            )

        else:
            raise ValueError(
                f"Unknown bc_type: {bc_type}. Use 'dirichlet' or 'neumann'."
            )

    def _initialize_fields(self):
        """Reset displacement field"""
        self.u = self.space.make_field()

    def _node_positions_torch(self):
        """Return node positions as (N, dim) torch tensor on GPU (zero-copy from warp)."""
        _geo_positions = getattr(self.geo, "positions", None)
        _pos_dtype_name = getattr(
            getattr(_geo_positions, "dtype", None), "__name__", ""
        )
        if _geo_positions is not None and _pos_dtype_name in ("vec2d", "vec3d"):
            return wp.to_torch(_geo_positions).to(dtype=torch.float64)
        pos_wp = self.space.node_positions()
        return wp.to_torch(pos_wp).to(dtype=torch.float64)

    @property
    def _design_dim(self):
        return 1  # Design variable dimension (scalar field e)

    def _init_e_field(self):
        """Initialize optimization field e with params.e_init."""
        e_init = self.params.get("e_init", None)
        if e_init is None:
            e_init = 1.0
        n = self.e_space.node_count()
        if self.design_dtype == wp.float64:
            e_np = np.full(n, e_init, dtype=np.float64)
            self.e.dof_values = wp.array(e_np, dtype=wp.float64, device=self.device)
        else:
            e_np = np.full(n, e_init, dtype=np.float32)
            self.e.dof_values = wp.array(e_np, dtype=wp.float32, device=self.device)

    def _init_material_field(self):
        """Sample a VoMP grid onto mesh nodes in one batched device gather."""
        from .material_field import sample_material_artifact

        positions = self._node_positions_torch()
        sampled = sample_material_artifact(self.params.material_artifact, positions)
        material_dtype = self.dtype
        torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        values = sampled.values.to(dtype=torch_dtype).contiguous()
        self.material_space = fem.make_polynomial_space(
            geo=self.geo, degree=1, dtype=material_dtype
        )
        self.material_field = self.material_space.make_field()
        self.material_field.dof_values = wp.from_torch(values, dtype=material_dtype)
        self._material_values_torch = sampled.values
        self._material_manifest = sampled.manifest

    def _build_e_fixed_mask(self):
        """Build mask for fixed region of optimization field e.

        Nodes where e_fixed_region(x, y[, z]) is True have their value
        frozen and gradients zeroed during optimization.
        """
        n_e = self.e_space.node_count()
        fixed_fn = self.params.get("e_fixed_region", None)
        if fixed_fn is not None and callable(fixed_fn):
            node_positions = self._node_positions_torch()
            try:
                if self.dim == 2:
                    mask = fixed_fn(node_positions[:n_e, 0], node_positions[:n_e, 1])
                else:
                    mask = fixed_fn(
                        node_positions[:n_e, 0],
                        node_positions[:n_e, 1],
                        node_positions[:n_e, 2],
                    )
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, device=node_positions.device)
                fixed_mask = mask.bool().view(-1)[:n_e]
            except Exception:
                pos_np = node_positions[:n_e].cpu().numpy()
                mask_np = np.zeros(n_e, dtype=bool)
                for i in range(n_e):
                    pos = pos_np[i]
                    if self.dim == 2:
                        mask_np[i] = bool(fixed_fn(pos[0], pos[1]))
                    else:
                        mask_np[i] = bool(fixed_fn(pos[0], pos[1], pos[2]))
                fixed_mask = torch.from_numpy(mask_np).to(node_positions.device)
            _design_scalar = (
                self.design_dtype._wp_scalar_type_
                if hasattr(self.design_dtype, "_wp_scalar_type_")
                else self.design_dtype
            )
            _design_torch = (
                torch.float64 if _design_scalar == wp.float64 else torch.float32
            )
            _design_vec = wp.types.vector(length=self._design_dim, dtype=_design_scalar)
            e_free = (~fixed_mask).to(dtype=_design_torch).unsqueeze(-1)
            self._e_free_mask = wp.from_torch(
                e_free.contiguous(),
                dtype=_design_vec,
            )
        else:
            _design_scalar = (
                self.design_dtype._wp_scalar_type_
                if hasattr(self.design_dtype, "_wp_scalar_type_")
                else self.design_dtype
            )
            self._e_free_mask = wp.ones(
                n_e,
                dtype=wp.types.vector(length=self._design_dim, dtype=_design_scalar),
                device=self.device,
            )

    def _get_bc_mask_and_values(self):
        """Get BC mask and values as torch tensors on GPU."""
        n_nodes = self.space.node_count()
        bc_mask = torch.zeros(n_nodes, self.dim, dtype=torch.bool, device=self.device)
        bc_values = torch.zeros(
            n_nodes, self.dim, dtype=torch.float64, device=self.device
        )

        for nodes, value, components in self.dirichlet_bcs:
            if components is None:
                bc_mask[nodes, :] = True
                if isinstance(value, (int, float)):
                    bc_values[nodes, :] = value
                elif isinstance(value, torch.Tensor):
                    if value.shape[0] == nodes.shape[0]:
                        bc_values[nodes, :] = value
                    else:
                        raise ValueError(
                            f"Value tensor size mismatch: {value.shape} vs {nodes.shape[0]} nodes"
                        )
                elif isinstance(value, (tuple, list)):
                    if len(value) == self.dim:
                        bc_values[nodes, :] = torch.tensor(
                            value, dtype=torch.float64, device=self.device
                        )
                    else:
                        raise ValueError(
                            f"Value tuple length {len(value)} != dim {self.dim}"
                        )
                else:
                    bc_values[nodes, :] = value
            else:
                for comp in components:
                    bc_mask[nodes, comp] = True
                    if isinstance(value, (int, float)):
                        bc_values[nodes, comp] = value
                    elif isinstance(value, torch.Tensor):
                        if value.shape[0] == nodes.shape[0]:
                            bc_values[nodes, comp] = value[:, comp]
                        else:
                            raise ValueError(f"Value tensor size mismatch")
                    elif isinstance(value, (tuple, list)):
                        bc_values[nodes, comp] = value[comp]
                    else:
                        bc_values[nodes, comp] = value

        # Mark slave DOFs from affine constraints as constrained
        for slaves, masters, weight in self.affine_constraints:
            if slaves.shape[0] > 0:
                bc_mask[slaves.long(), :] = True

        return bc_mask, bc_values

    def _scale_dirichlet_bc_values(self, scale):
        """Rebuild bc_values_f32 with Dirichlet values scaled by `scale`."""
        scaled = self._bc_values_full * scale  # torch * scalar, stays on GPU
        _scalar = getattr(self.dtype, "_wp_scalar_type_", self.dtype)
        _torch_dtype = torch.float64 if _scalar == wp.float64 else torch.float32
        self.bc_values_f32 = wp.from_torch(
            scaled.to(dtype=_torch_dtype).contiguous(), dtype=self.dtype
        )

    def _build_affine_constraint_gpu_data(self):
        """Convert affine_constraints list to GPU arrays and select kernels.

        Populates self._constraint_groups, self._all_slaves_wp, self._n_all_slaves.
        """
        self._constraint_groups = []
        all_slave_parts = []
        for slaves_t, masters_t, weight in self.affine_constraints:
            n = slaves_t.shape[0]
            if n == 0:
                continue
            k = masters_t.shape[1]
            slaves_wp = wp.from_torch(
                slaves_t.to(dtype=torch.int32).contiguous(), dtype=wp.int32
            )
            master_arrays = [
                wp.from_torch(
                    masters_t[:, col].to(dtype=torch.int32).contiguous(), dtype=wp.int32
                )
                for col in range(k)
            ]
            if k == 2:
                ek = _enforce_edge_hanging_3d
                pk = _prolongate_edge_hanging
                rk = _restrict_edge_hanging
            elif k == 4:
                ek = _enforce_face_hanging_3d
                pk = _prolongate_face_hanging
                rk = _restrict_face_hanging
            else:
                raise ValueError(f"Unsupported master count {k} in affine_constraints")
            self._constraint_groups.append(
                {
                    "n": n,
                    "k": k,
                    "weight": weight,
                    "slave": slaves_wp,
                    "masters": master_arrays,
                    "enforce_kernel": ek,
                    "prolong_kernel": pk,
                    "restrict_kernel": rk,
                }
            )
            all_slave_parts.append(slaves_t.to(dtype=torch.int32))
        if all_slave_parts:
            all_t = torch.cat(all_slave_parts)
            self._all_slaves_wp = wp.from_torch(all_t.contiguous(), dtype=wp.int32)
            self._n_all_slaves = all_t.shape[0]
        else:
            self._all_slaves_wp = None
            self._n_all_slaves = 0

    @property
    def _has_affine_constraints(self):
        """Whether affine constraints (e.g. hanging nodes) are active."""
        return len(self._constraint_groups) > 0

    def _enforce_affine_constraints(self, bc_values_f32):
        """Set u[slave] = sum(w * u[master]) and bc_vals[slave] likewise."""
        _u_scalar = (
            self.u.dof_values.dtype._wp_scalar_type_
            if hasattr(self.u.dof_values.dtype, "_wp_scalar_type_")
            else self.u.dof_values.dtype
        )
        for g in self._constraint_groups:
            inputs = (
                [self.u.dof_values, bc_values_f32, g["slave"]]
                + g["masters"]
                + [_u_scalar(g["weight"])]
            )
            wp.launch(
                g["enforce_kernel"], dim=g["n"], inputs=inputs, device=self.device
            )

    def _prolongate_vector(self, v):
        """C*v: set v[slave] = sum(w * v[master]) for each constraint group."""
        # Weight scalar must match v's scalar type (v may be float32 or float64).
        _v_scalar = (
            v.dtype._wp_scalar_type_
            if hasattr(v.dtype, "_wp_scalar_type_")
            else v.dtype
        )
        for g in self._constraint_groups:
            inputs = [v, g["slave"]] + g["masters"] + [_v_scalar(g["weight"])]
            wp.launch(
                g["prolong_kernel"], dim=g["n"], inputs=inputs, device=self.device
            )

    def _restrict_into(self, v_in, v_out):
        """C^T*v: distribute slave to masters, zero slaves. v_out pre-copied from v_in."""
        _v_scalar = (
            v_in.dtype._wp_scalar_type_
            if hasattr(v_in.dtype, "_wp_scalar_type_")
            else v_in.dtype
        )
        for g in self._constraint_groups:
            inputs = [v_in, v_out, g["slave"]] + g["masters"] + [_v_scalar(g["weight"])]
            wp.launch(
                g["restrict_kernel"], dim=g["n"], inputs=inputs, device=self.device
            )
        if self._all_slaves_wp is not None:
            wp.launch(
                _zero_slave_entries,
                dim=self._n_all_slaves,
                inputs=[v_out, self._all_slaves_wp],
                device=self.device,
            )

    # Removed unused BC projector and normal-weight helpers to reduce clutter

    # _create_bc_value_dict removed (unused)

    # Detailed debug snapshot removed to simplify logs

    def _preprocess_optimized(self, tape=None):
        """Preprocess self.e into a ready-to-use field.

        Default: identity (return self.e unchanged).
        Subclasses (e.g. TopoOpt) override to apply filtering + projection.
        When *tape* is not None, the preprocessing must be recorded so that
        gradients flow back through the chain.
        """
        return self.e

    def step(self, tape=None):
        """Perform one Newton-Raphson solve at the current self.current_load_scale.

        Can be called standalone after initialization. Updates tracking variables
        (residual_history, newton_iterations, etc.) cumulatively.

        When *tape* is not None (a wp.Tape), a single post-convergence tape pass
        records the implicit function for adjoint differentiation:
        p -> fem.integrate(stress) -> R_tape (on tape), then K assembly + solve
        (off tape, reusing existing solver branch), and tape.record_func to
        register the adjoint that maps dL/du -> dR/dp via K^{-1}.

        Raises RuntimeError if Newton solver does not converge.
        """
        mc = self._mesh_cache
        self._step_cache = {
            "solve_cache": {},
            "tangent_tmp_store": fem.TemporaryStore(),
        }
        sc = self._step_cache
        n_nodes = self.space.node_count()
        _step_residual_norm_0 = None
        step_converged = False

        if tape is not None:
            self.u.dof_values.requires_grad = True
            self.e.dof_values.requires_grad = True

        # Preprocess optimization field (identity by default; filtering+projection in TopoOpt)
        self._current_preprocessed_e = self._preprocess_optimized(tape=tape)

        # Scale Dirichlet BCs by current load
        self._scale_dirichlet_bc_values(self.current_load_scale)

        wp.launch(
            _apply_bc_to_field,
            dim=n_nodes,
            inputs=[
                self.u.dof_values,
                self.bc_values_f32,
                self.free_mask_f32,
                self.bc_mask_f32,
            ],
            device=self.device,
        )
        if mc["has_cond"]:
            self._enforce_affine_constraints(self.bc_values_f32)

        _ew_prev_residual_norm = None
        _ew_prev_eta = None
        # apply_step() evaluates the residual at the accepted updated state.
        # Carry that vector into the next Newton iteration instead of assembling
        # the identical residual again at the top of the loop.
        _cached_residual = None

        for iteration in range(self.params.max_iter):
            # Hook for per-iteration pre-processing (e.g., Nitsche beta recompute)
            self._pre_newton_iteration(iteration)
            # Enforce constraints before residual evaluation
            if mc["has_cond"]:
                self._enforce_affine_constraints(self.bc_values_f32)
            residual_is_condensed = False
            if _cached_residual is None:
                R = self._evaluate_residual(
                    mc["domain"],
                    mc["test"],
                    mc["boundary"],
                    mc["bd_test"],
                    neumann_load=mc["neumann"],
                    body_force=mc["body"],
                )
            else:
                R, residual_is_condensed = _cached_residual
                _cached_residual = None

            # Condense RHS for affine constraints: R <- C^T * R
            # (needed for convergence check and for b = -R in solver paths)
            if mc["has_cond"] and not residual_is_condensed:
                R_raw = R
                wp.copy(src=R_raw, dest=mc["R_restrict_buf"])
                R = mc["R_restrict_buf"]
                self._restrict_into(R_raw, R)

            # GPU masked norm (single scalar transfer)
            residual_norm = masked_norm(
                R,
                self.free_mask_wp,
                n_nodes,
                self.device,
                result_buf=mc["scalar_buf"],
            )

            if _step_residual_norm_0 is None:
                if self.initial_residual is None:
                    self.initial_residual = residual_norm
                _step_residual_norm_0 = residual_norm if residual_norm > 0.0 else 1.0
                relative_residual = 1.0 if residual_norm > 0.0 else 0.0
            else:
                relative_residual = (
                    residual_norm / _step_residual_norm_0
                    if _step_residual_norm_0 > 0.0
                    else 0.0
                )

            abs_tol = self.params.get("absolute_tol", 1e-8)
            self.log(
                f"  scale={self.current_load_scale:.4f} Iter {iteration}: "
                f"||R|| = {residual_norm:.3e}, rel = {relative_residual:.3e}"
            )
            self.residual_history.append(residual_norm)
            self.relative_residual_history.append(relative_residual)

            if math.isnan(residual_norm):
                raise RuntimeError(
                    f"Newton solver: NaN residual at iteration {iteration} "
                    f"(load scale {self.current_load_scale:.4f}). "
                )

            if iteration > 0 and (
                residual_norm < abs_tol or relative_residual < self.params.relative_tol
            ):
                step_converged = True
                self.newton_iterations += iteration + 1
                self.final_residual = residual_norm
                self.final_relative_residual = relative_residual
                if tape is None:
                    break
                # ---- Tape pass: re-evaluate R on tape, then fall through ----
                self._pre_tape_residual(tape)
                if mc["has_cond"]:
                    self._enforce_affine_constraints(self.bc_values_f32)
                R_tape = self._evaluate_residual(
                    mc["domain"],
                    mc["test"],
                    mc["boundary"],
                    mc["bd_test"],
                    neumann_load=mc["neumann"],
                    body_force=mc["body"],
                    tape=tape,
                )
                if mc["has_cond"]:
                    R_raw = R_tape
                    wp.copy(src=R_raw, dest=mc["R_restrict_buf"])
                    R = mc["R_restrict_buf"]
                    self._restrict_into(R_raw, R)
                else:
                    R = R_tape
                # Falls through to K/matvec assembly (for adjoint); solve is skipped.

            if not step_converged:
                # Eisenstat-Walker adaptive CG tolerance.
                # Active for wp_cg paths (matrix-free and native assembled).
                # AMGX backends use their own fixed, backend-specific tolerance.
                uses_wp_cg = mc["solver_type"] != "cg" or mc["cg_backend"] == "native"
                ew_enabled = bool(self.params.get("ew_enabled", True))
                if ew_enabled and uses_wp_cg:
                    ew_eta_max = float(self.params.get("ew_eta_max", 0.9))
                    if (
                        _ew_prev_residual_norm is not None
                        and _ew_prev_residual_norm > 0.0
                    ):
                        ew_gamma = float(self.params.get("ew_gamma", 0.9))
                        ew_alpha = float(self.params.get("ew_alpha", 1.618))
                        eta = (
                            ew_gamma
                            * (residual_norm / _ew_prev_residual_norm) ** ew_alpha
                        )
                        if _ew_prev_eta is not None:
                            eta = max(eta, ew_gamma * _ew_prev_eta**ew_alpha)
                        eta = min(eta, ew_eta_max)
                        _ew_prev_eta = eta
                    else:
                        eta = ew_eta_max
                        _ew_prev_eta = eta
                    _ew_prev_residual_norm = residual_norm
                    rel_tol = float(self.params.relative_tol)
                    eta = min(
                        eta,
                        max(
                            rel_tol * relative_residual,
                            abs_tol / max(residual_norm, 1e-30),
                        ),
                    )
                    cg_atol = eta * residual_norm
                    self.log(f"  EW: eta = {eta:.3e}, atol = {cg_atol:.3e}")
                else:
                    cg_atol = None
            else:
                cg_atol = None

            if mc["solver_type"] == "matrix_free_cg":
                tangent_int = self._tangent_integrand

                # Define matvec closure (needed by adjoint even when converged)
                def matvec(x, y, z, alpha, beta):
                    _x_scalar = getattr(x.dtype, "_wp_scalar_type_", x.dtype)
                    alpha = _x_scalar(alpha)
                    beta = _x_scalar(beta)
                    if mc["has_cond"]:
                        wp.copy(src=x, dest=mc["x_masked_buf"])
                        self._prolongate_vector(mc["x_masked_buf"])
                    else:
                        # Mask constrained DOFs (float64)
                        wp.launch(
                            _copy_masked,
                            dim=n_nodes,
                            inputs=[
                                x,
                                mc["x_masked_buf"],
                                self.free_mask_wp,
                            ],
                            device=self.device,
                        )
                    # Cast float64 -> float32 into scratch field
                    wp.utils.array_cast(
                        in_array=mc["x_masked_buf"],
                        out_array=mc["du_scratch"].dof_values,
                    )
                    # JVP: du is now the seed of CG instead of trial function
                    mc["Kx_buf"].zero_()
                    _mf_values = {
                        "mu": self._integrand_scalar_dtype(self.mu),
                        "kappa": self._integrand_scalar_dtype(self.kappa),
                    }
                    _mf_values.update(self._extra_integrate_values())
                    _mf_kwargs = dict(
                        fields={
                            "u": self.u,
                            "du": mc["du_scratch"],
                            "v": mc["test"],
                            "e": self._current_preprocessed_e,
                            **({"material": self.material_field}
                               if self._has_spatial_material else {}),
                        },
                        values=_mf_values,
                        output=mc["Kx_buf"],
                        temporary_store=mc["tmp_store"],
                        accumulate_dtype=(
                            wp.float64 if self.use_mixed_precision else wp.float32
                        ),
                    )
                    _mf_kwargs.update(self._extra_integrate_kwargs())
                    fem.integrate(tangent_int, **_mf_kwargs)
                    if mc["has_cond"]:
                        wp.copy(src=mc["Kx_buf"], dest=mc["cond_buf_a"])
                        self._restrict_into(mc["Kx_buf"], mc["cond_buf_a"])
                        wp.launch(
                            _matvec_bc_combine,
                            dim=n_nodes,
                            inputs=[
                                mc["cond_buf_a"],
                                x,
                                y,
                                z,
                                self.free_mask_wp,
                                self.bc_mask_wp,
                                alpha,
                                beta,
                            ],
                            device=self.device,
                        )
                    else:
                        # BC combine: free -> Kx, constrained -> identity
                        wp.launch(
                            _matvec_bc_combine,
                            dim=n_nodes,
                            inputs=[
                                mc["Kx_buf"],
                                x,
                                y,
                                z,
                                self.free_mask_wp,
                                self.bc_mask_wp,
                                alpha,
                                beta,
                            ],
                            device=self.device,
                        )

                if not step_converged:
                    b = -R
                    try:
                        wp.copy(src=b, dest=mc["b_fixed_buf"])
                        b_fixed = mc["b_fixed_buf"]
                        wp.launch(
                            _mask_inplace,
                            dim=n_nodes,
                            inputs=[b_fixed, self.free_mask_wp],
                            device=self.device,
                        )

                        du = solve_linear_system(
                            K=None,
                            b=b_fixed,
                            method="cg",
                            cg_atol=cg_atol,
                            device=self.device,
                            dim=self.dim,
                            mv_routine=matvec,
                            _cache=sc["solve_cache"],
                        )
                    except Exception as e:
                        self.log(
                            f"ERROR: Matrix-free CG failed at iteration {iteration}: {e}"
                        )
                        raise RuntimeError(
                            f"Matrix-free CG failed at Newton iteration {iteration}: {e}"
                        )

            else:
                # Assembled path: tangent matrix → AMGX full solve
                tangent_int = self._tangent_integrand

                # Warp propagates geometry precision through dispatch assembly.
                _k_integrate_kwargs = {
                    "accumulate_dtype": self._tangent_accumulate_dtype(),
                    "output_dtype": self._tangent_accumulate_dtype(),
                    "quadrature": self._default_quadrature(),
                }

                # Build common tangent integrate kwargs with extra values
                _sd = self._integrand_scalar_dtype
                _tangent_values = {"mu": _sd(self.mu), "kappa": _sd(self.kappa)}
                _tangent_values.update(self._extra_integrate_values())
                _tangent_fields = {
                    "u": self.u,
                    "du": mc["trial"],
                    "v": mc["test"],
                    "e": self._current_preprocessed_e,
                }
                if self._has_spatial_material:
                    _tangent_fields["material"] = self.material_field
                _tangent_common = dict(
                    fields=_tangent_fields,
                    values=_tangent_values,
                    temporary_store=sc["tangent_tmp_store"],
                    **_k_integrate_kwargs,
                )
                _tangent_common.update(self._extra_integrate_kwargs())

                if (
                    mc["has_cond"]
                    and mc["K_extended"] is not None
                    and mc["extended_differs"]
                ):
                    # Subsequent iterations: assemble into original K,
                    # then scatter values into the extended BSR.
                    mc["K"].values.zero_()
                    K_std = fem.integrate(
                        tangent_int,
                        output=mc["K"],
                        **_tangent_common,
                    )
                    mc["K"] = K_std
                    K_std = self._post_tangent_assembly(K_std)
                    from .condensation import scatter_to_extended

                    scatter_to_extended(
                        K_std,
                        mc["K_extended"],
                        mc["scatter_mapping"],
                        mc["actual_nnz_std"],
                        self.device,
                    )
                    K = mc["K_extended"]
                else:
                    # First iteration (or no extension needed)
                    if mc["K"] is not None:
                        mc["K"].values.zero_()
                    K = fem.integrate(
                        tangent_int,
                        output=mc["K"],
                        **_tangent_common,
                    )
                    mc["K"] = K
                    K = self._post_tangent_assembly(K)

                    if mc["has_cond"] and mc["K_extended"] is None:
                        _block_type = K.values.dtype
                        from .condensation import extend_bsr_pattern

                        (
                            mc["K_extended"],
                            mc["scatter_mapping"],
                            mc["actual_nnz_std"],
                        ) = extend_bsr_pattern(
                            K,
                            self._constraint_groups,
                            _block_type,
                            self.device,
                        )
                        mc["extended_differs"] = mc["scatter_mapping"] is not None
                        if mc["extended_differs"]:
                            K = mc["K_extended"]

                # Hook for auxiliary tangent contributions (e.g. Nitsche boundary)
                self._add_to_tangent(K)

                b = -R

                if mc["has_cond"]:
                    from .condensation import condense_bsr_inplace

                    condense_bsr_inplace(K, self._constraint_groups, self.device)

                # K and projector must share scalar type. b is float64 —
                # project_linear_system would call bsr_mv(K, b_f64) → type mismatch
                # if K is float32. Split into matrix projection and manual RHS masking.
                fem.normalize_dirichlet_projector(self.bd_matrix)
                fem.project_system_matrix(K, self.bd_matrix)
                wp.launch(
                    _mask_inplace,
                    dim=n_nodes,
                    inputs=[b, self.free_mask_wp],
                    device=self.device,
                )

                if not step_converged:
                    try:
                        solve_cache = sc["solve_cache"]
                        if mc["cg_backend"] in {"amgx", "jaxamg"}:
                            from .jax_amg_solver import resolve_amgx_backend

                            if resolve_amgx_backend(mc["cg_backend"]) == "jaxamg":
                                solve_cache = mc["jaxamg_solve_cache"]
                        du = solve_linear_system(
                            K,
                            b,
                            method="cg",
                            cg_atol=cg_atol,
                            device=self.device,
                            dim=self.dim,
                            cg_backend=mc["cg_backend"],
                            _cache=solve_cache,
                            amgx_rebuild_every_iter=bool(
                                self.params.get("amgx_rebuild_every_iter", False)
                            ),
                            use_mixed_precision=self.use_mixed_precision,
                            scale_system_to_unit_diagonal=self.params.get(
                                "scale_system_to_unit_diagonal", False
                            ),
                            jaxamg_max_iters=self.params.get("jaxamg_max_iters", None),
                        )
                    except Exception as e:
                        self.log(
                            f"ERROR: Linear solver failed at iteration {iteration}: {e}"
                        )
                        raise RuntimeError(
                            f"Linear solver failed at Newton iteration {iteration}: {e}"
                        )

            # Recover slave du from master updates (C * du)
            if not step_converged and mc["has_cond"]:
                self._prolongate_vector(du)

            # ---- Tape: register adjoint and exit loop ----
            if step_converged:
                vecd_type = self.dtype
                free_mask = self.dirichlet_free_mask_wp
                u_dof = self.u.dof_values
                solve_cache = sc["solve_cache"]
                device = self.device
                dim = self.dim
                use_mixed_precision = self.use_mixed_precision

                # Build adjoint solver closure (captures K or matvec from above)
                if mc["solver_type"] == "matrix_free_cg":
                    _adj_matvec = matvec  # defined in solver branch above

                    def _adjoint_solve(b):
                        return solve_linear_system(
                            K=None,
                            b=b,
                            method="cg",
                            device=device,
                            dim=dim,
                            mv_routine=_adj_matvec,
                            _cache=solve_cache,
                        )

                else:
                    _adj_K = K  # defined in solver branch above

                    def _adjoint_solve(b):
                        self.log(f"Solving adjoint system")
                        return solve_linear_system(
                            _adj_K,
                            b,
                            method="cg",
                            device=device,
                            dim=dim,
                            cg_backend=mc["cg_backend"],
                            _cache=solve_cache,
                            use_mixed_precision=use_mixed_precision,
                            scale_system_to_unit_diagonal=self.params.get(
                                "scale_system_to_unit_diagonal", False
                            ),
                            jaxamg_max_iters=self.params.get("jaxamg_max_iters", None),
                        )

                def adjoint_fn():
                    nn = n_nodes
                    # 1) Cast dL/du to solver scalar type
                    adj_b = wp.zeros(nn, dtype=vecd_type, device=device)
                    wp.utils.array_cast(in_array=u_dof.grad, out_array=adj_b)
                    # 2) Negate: solve K·x = -(dL/du) so x = -λ = R.grad
                    wp.launch(
                        _scale_inplace,
                        dim=nn,
                        inputs=[
                            adj_b,
                            getattr(vecd_type, "_wp_scalar_type_", wp.float32)(-1.0),
                        ],
                        device=device,
                    )
                    # 3) Condense RHS (mirrors forward restrict)
                    if mc["has_cond"]:
                        adj_b_raw = adj_b
                        adj_b = wp.zeros(nn, dtype=vecd_type, device=device)
                        wp.copy(src=adj_b_raw, dest=adj_b)
                        self._restrict_into(adj_b_raw, adj_b)
                    # 4) Mask Dirichlet BCs only (NOT hanging nodes)
                    wp.launch(
                        _mask_inplace,
                        dim=nn,
                        inputs=[adj_b, free_mask],
                        device=device,
                    )
                    # 5) Solve adjoint system
                    lam = _adjoint_solve(adj_b)
                    # 6) Prolongate (mirrors forward)
                    if mc["has_cond"]:
                        self._prolongate_vector(lam)
                    # 7) Write to R_tape.grad (cast if lam dtype differs, e.g. f64→f32)
                    wp.utils.array_cast(in_array=lam, out_array=R_tape.grad)
                    # 8) Mask Dirichlet in R_tape.grad
                    wp.launch(
                        _mask_inplace,
                        dim=nn,
                        inputs=[R_tape.grad, free_mask],
                        device=device,
                    )
                    # 9) Zero u.grad (u is implicit, not an input)
                    u_dof.grad.zero_()

                self._tape_residual = R_tape
                tape.record_func(adjoint_fn, arrays=[R_tape, self.u.dof_values])
                break

            mc["scalar_buf"].zero_()
            if not check_finite(du, n_nodes, self.device, result_buf=mc["scalar_buf"]):
                self.log(
                    f"ERROR: Linear solver produced NaN/Inf at iteration {iteration}"
                )
                raise RuntimeError(
                    f"Linear solver produced invalid solution at iteration {iteration}. "
                    f"Check for singular matrix or ill-conditioning."
                )

            # Damping (GPU, single scalar transfer for max norm)
            damping_limit = float(self.params.get("update_damping_limit", 0.0))
            if damping_limit > 0.0:
                max_du = max_node_norm(
                    du, n_nodes, self.device, result_buf=mc["scalar_buf"]
                )
                if max_du > damping_limit and max_du > 0.0:
                    scale = damping_limit / max_du
                    if self.params.get("debug_line_search", False):
                        self.log(
                            f"    damping update: max ||du||={max_du:.3e}, applying scale {scale:.3e}"
                        )
                    _du_scalar = (
                        du.dtype._wp_scalar_type_
                        if hasattr(du.dtype, "_wp_scalar_type_")
                        else du.dtype
                    )
                    wp.launch(
                        _scale_inplace,
                        dim=n_nodes,
                        inputs=[du, _du_scalar(scale)],
                        device=self.device,
                    )

            # Cast solver dtype → float32 for field update (stays on GPU)
            wp.utils.array_cast(in_array=du, out_array=mc["du_update"])

            debug_fn = self.log if self.params.get("debug_line_search", False) else None

            def apply_step(alpha_trial):
                nonlocal _cached_residual
                fem.linalg.array_axpy(
                    x=mc["du_update"],
                    y=self.u.dof_values,
                    alpha=alpha_trial,
                )
                if mc["has_cond"]:
                    self._enforce_affine_constraints(self.bc_values_f32)
                R_trial = self._evaluate_residual(
                    mc["domain"],
                    mc["test"],
                    mc["boundary"],
                    mc["bd_test"],
                    neumann_load=mc["neumann"],
                    body_force=mc["body"],
                )
                if mc["has_cond"]:
                    R_trial_raw = R_trial
                    wp.copy(
                        src=R_trial_raw,
                        dest=mc["R_restrict_buf"],
                    )
                    R_trial = mc["R_restrict_buf"]
                    self._restrict_into(R_trial_raw, R_trial)
                    _cached_residual = (R_trial, True)
                else:
                    _cached_residual = (R_trial, False)
                # GPU masked norm (single scalar transfer)
                residual_trial = masked_norm(
                    R_trial,
                    self.free_mask_wp,
                    n_nodes,
                    self.device,
                    result_buf=mc["scalar_buf"],
                )
                relative_trial = (
                    residual_trial / _step_residual_norm_0
                    if _step_residual_norm_0 > 0.0
                    else residual_trial
                )
                return residual_trial, relative_trial

            def revert_step(alpha_trial):
                fem.linalg.array_axpy(
                    x=mc["du_update"],
                    y=self.u.dof_values,
                    alpha=-alpha_trial,
                )
                if mc["has_cond"]:
                    self._enforce_affine_constraints(self.bc_values_f32)

            line_search_enabled = bool(self.params.get("line_search_enabled", True))
            if line_search_enabled:
                (
                    accepted,
                    residual_after_step,
                    relative_after_step,
                    alpha_used,
                ) = backtracking_line_search(
                    residual_norm,
                    relative_residual,
                    abs_tol,
                    self.params,
                    apply_step,
                    revert_step,
                    _step_residual_norm_0,
                    debug_log=debug_fn,
                )

                if not accepted:
                    raise RuntimeError(
                        "Line search failed to reduce residual; Newton step rejected."
                    )

                residual_norm = residual_after_step
                relative_residual = relative_after_step
            else:
                alpha_used = 1.0
                residual_norm, relative_residual = apply_step(alpha_used)

            self.line_search_alphas.append(float(alpha_used))

        if not step_converged:
            raise RuntimeError(
                f"Newton solver did not converge in {self.params.max_iter} iterations. "
                f"Final residual: {residual_norm:.3e}"
            )

        self._last_K = mc["K"]

        # Snapshot mutable GPU arrays immediately after Newton convergence.
        # At this point the GPU has no pending work beyond the forward solve;
        # tape.backward() hasn't been called yet.  Each .numpy() call here is a
        # ~1 ms device-to-host copy, vs ~730 ms if deferred to save() where they
        # would sync the entire constraint backward pass (tape replay of hundreds
        # of Helmholtz CG kernel launches).
        self._u_snapshot_np = self.u.dof_values.numpy().copy()
        self._e_snapshot_np = self.e.dof_values.numpy().copy()
        _pre = getattr(self, "_current_preprocessed_e", None)
        if _pre is not None and _pre is not self.e:
            self._pre_snapshot_np = _pre.dof_values.numpy().copy()
        else:
            self._pre_snapshot_np = None

    def forward(self, tape=None, on_load_step=None):
        """Solve nonlinear elasticity problem using Newton-Raphson with load stepping.

        Thin wrapper that resets tracking variables, iterates through load steps
        setting current_load_scale, and calls step() for each.

        When *tape* is not None, it is passed to the **last** load step only,
        recording the implicit function for adjoint differentiation.

        *on_load_step*, when given, is called as ``(index, total)`` after each
        step has converged.  A ramp otherwise leaves nothing behind but its end
        state, so there is no way to look at how the part got there.
        """
        self.log("Initializing forward solve")

        load_steps = max(1, int(self.params.get("load_steps", 1)))

        # Reset all tracking variables
        self.residual_history = []
        self.relative_residual_history = []
        self.newton_iterations = 0
        self.initial_residual = None
        self.final_residual = None
        self.final_relative_residual = None
        self.line_search_alphas = []
        self.current_load_scale = 1.0

        try:
            for load_step_idx in range(load_steps):
                self.current_load_scale = float(load_step_idx + 1) / float(load_steps)
                if load_steps > 1:
                    self.log(
                        f"===== Load step {load_step_idx + 1}/{load_steps}: "
                        f"scale={self.current_load_scale:.6f} ====="
                    )
                step_tape = tape if (load_step_idx == load_steps - 1) else None
                self.step(tape=step_tape)
                if on_load_step is not None:
                    on_load_step(load_step_idx, load_steps)
        finally:
            # Outside a ramp the load is whole, so a run that stops partway does
            # not leave a later solve quietly under-loaded.
            self.current_load_scale = 1.0

        mc = self._mesh_cache
        self._last_K = mc["K"]
        return

    def _normalized_scalar(self, value, name):
        if getattr(self, "_initial_objective_values", None) is None:
            self._initial_objective_values = {}
        if name not in self._initial_objective_values.keys():
            # Evaluate as a Python float so the division is NOT recorded on
            # the tape.  Computing  scale = 1.0 / value  on tape
            # creates  normalized = value * (1/value)  ≡ 1,  whose derivative
            # w.r.t. value is identically 0 — killing the objective gradient
            # at iteration 0.
            val = float(value.numpy()[0])
            self._initial_objective_values[name] = wp.array(
                [1.0 / val],
                dtype=value.dtype,
                device=value.device,
                requires_grad=False,
            )
        normalized_scalar = value * self._initial_objective_values[name]
        return normalized_scalar

    def optimize(self):
        """Run multi-iteration optimization loop.

        Supports two optimizer backends controlled by
        ``self.params.get("optimizer", "mma")``:

        - ``"mma"`` (default): GCMMA / CCSA-MMA with explicit constraints.
        - ``"gd"``: Gradient descent for unconstrained problems where
          constraints are baked into the objective as soft penalties.

        Each iteration: forward solve (with tape) -> objective + backward ->
        [constraint eval + backward (MMA only)] -> update -> clamp.

        Convergence criteria (both backends):
        - Skip checking in the first 20 iterations.
        - Skip when relative reduction is increasing.
        - Converge when relative reduction < tol for last 2 iterations.
        - Converge if >500 iters with no improvement in last 50.
        """
        optimizer_type = self.params.get("optimizer", "mma")
        lo, hi = self._bounds()
        max_iters = int(self.params.get("max_optimize_iters", 1000))
        rel_tol = float(self.params.get("optimize_relative_tol", 1e-4))
        n_e = self.e_space.node_count()
        n_nodes = self.space.node_count()
        _design_scalar = (
            self.design_dtype._wp_scalar_type_
            if hasattr(self.design_dtype, "_wp_scalar_type_")
            else self.design_dtype
        )
        vecd_type = wp.types.vector(length=self._design_dim, dtype=_design_scalar)
        save_freq = int(self.params.get("output_per_optimize_step", 1))

        if optimizer_type == "gd":
            self.log("Using gradient descent optimizer (unconstrained)")
            gd_lr = wp.float64(self.params.get("gd_lr", 1e-3))
            gd_lr_auto = self._gd_lr_auto_setup()
        else:
            from .mma_solver import CCSAMMASolver

            constraint_fns = self._constraints()
            n_constraints = len(constraint_fns)

            gcmma_enabled = bool(self.params.get("gcmma_enabled", True))
            max_inner = int(self.params.get("gcmma_max_inner", 15))

            mma = CCSAMMASolver(
                n_vars=n_e,
                n_constraints=n_constraints,
                xmin=lo,
                xmax=hi,
                move_limit=float(self.params.get("mma_move_limit", 0.2)),
                rel_tolerance=rel_tol,
                max_inner=max_inner,
                device=self.device,
                dtype=wp.float64 if self.use_mixed_precision else wp.float32,
                gpu_dual=bool(self.params.get("mma_gpu_dual", False)),
            )

            # Build eval_func callback for GCMMA conservatism loop
            if gcmma_enabled:
                e_backup = wp.zeros_like(self.e.dof_values)
                u_backup = wp.zeros_like(self.u.dof_values)

                def eval_func(x_trial):
                    wp.copy(src=self.e.dof_values, dest=e_backup)
                    wp.copy(src=self.u.dof_values, dest=u_backup)

                    wp.copy(src=x_trial, dest=self.e.dof_values)
                    try:
                        self.step(tape=None)
                    except RuntimeError:
                        wp.copy(src=e_backup, dest=self.e.dof_values)
                        wp.copy(src=u_backup, dest=self.u.dof_values)
                        return float("inf"), [float("inf")] * n_constraints

                    obj_val = float(self._objective_function(None).numpy()[0])
                    c_vals = [float(cfn(None).numpy()[0]) for cfn in constraint_fns]

                    self.log(
                        f"    GCMMA eval_func: obj={obj_val:.6e}, "
                        f"constraints={[f'{v:.3e}' for v in c_vals]}"
                    )

                    wp.copy(src=e_backup, dest=self.e.dof_values)
                    wp.copy(src=u_backup, dest=self.u.dof_values)
                    return obj_val, c_vals

            else:
                eval_func = None

        obj_history = []
        best_obj = float("inf")
        best_obj_iter = 0

        # Where the loop has got to, for a caller handed an exception: the
        # iteration in progress, and whether its own solve failed -- in which
        # case the save below writes that failed state as the newest output.
        # The workflow (run_forward) reads these to keep its viewer on the last
        # iteration that solved; nothing in the loop depends on them.
        self._optimize_iteration = None
        self._optimize_failed_iteration = None
        # The size of each gradient-descent step, for the same caller: the
        # largest |dJ/ds| over the free design nodes (the masked gradient the
        # step is taken along) and gd_lr times it, the largest nodal SDF move.
        # run_forward sizes the learning rate it recommends after a failure
        # from these.  Bookkeeping only; the step itself is unchanged.
        self._optimize_steps = []
        # What gd_lr_auto chose at each update (None while off, or before its
        # first update): see _gd_lr_auto_choose.  Read by run_forward for its
        # status file.
        self._gd_lr_auto = None
        for it in range(max_iters):
            self._optimize_iteration = it
            self.log(f"===== Optimization Iteration {it} =====")
            # 1. Forward + tape + backward for objective
            tape = wp.Tape()
            self.e.dof_values.requires_grad = True
            try:
                self.step(tape=tape)
            except Exception as e:
                self._optimize_failed_iteration = it
                self.save(time=it)
                self.finalize()
                import time

                time.sleep(1)
                raise e
            # Refresh node counts AFTER step() in case a deferred
            # remesh swapped in a different-sized mesh at the start
            # of step() (e.g. ShapeOptTrimmed adaptive remeshing).
            # No-op for fixed-topology subclasses.
            n_e = self.e_space.node_count()
            n_nodes = self.space.node_count()

            # 2. Record objective on the tape
            with tape:
                obj = self._objective_function(tape)

            # 3. Backward for objective
            self.log(f"Backward for objective")
            tape.zero()
            self.e.dof_values.grad.zero_()
            tape.backward(loss=obj)
            self._post_tape_backward()
            obj_val = float(obj.numpy()[0])
            obj_grad = wp.clone(self.e.dof_values.grad)

            if optimizer_type == "gd":
                # Mask gradient by _e_free_mask
                _obj_grad = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)
                wp.utils.array_cast(in_array=obj_grad, out_array=_obj_grad)
                wp.launch(
                    _mask_inplace,
                    dim=n_e,
                    inputs=[_obj_grad, self._e_free_mask],
                    device=self.device,
                )
                wp.utils.array_cast(in_array=_obj_grad, out_array=obj_grad)
                max_abs_grad = float(wp.to_torch(obj_grad).abs().max()) if n_e else 0.0
                if gd_lr_auto is not None:
                    # Every update: the rate is sized from this iteration's
                    # gradient (the previous rate is the fallback).
                    gd_lr = wp.float64(self._gd_lr_auto_choose(
                        it, float(gd_lr), max_abs_grad, gd_lr_auto))
                self._optimize_steps.append({
                    "iteration": it, "max_abs_grad": max_abs_grad,
                    "max_move": float(gd_lr) * max_abs_grad,
                    # The rate is only worth recording when it can change.
                    **({"gd_lr": float(gd_lr)} if gd_lr_auto is not None else {}),
                })

                self._pre_optimizer_step(it)

                # Simple gradient descent: e -= lr * grad
                fem.linalg.array_axpy(
                    x=obj_grad, y=self.e.dof_values, alpha=-float(gd_lr)
                )

                self._post_optimizer_step(it)
            else:
                # MMA path: also backward for constraints
                c_vals = []
                c_grads = []
                for i, c_fn in enumerate(constraint_fns):
                    tape.zero()
                    self.e.dof_values.grad.zero_()
                    self.log(f"Backward for constraint {i}")
                    with tape:
                        c_val = c_fn(tape)
                    tape.backward(loss=c_val)
                    c_vals.append(float(c_val.numpy()[0]))
                    c_grads.append(wp.clone(self.e.dof_values.grad))

                # Mask gradients by _e_free_mask
                _obj_grad = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)
                wp.utils.array_cast(in_array=obj_grad, out_array=_obj_grad)
                wp.launch(
                    _mask_inplace,
                    dim=n_e,
                    inputs=[_obj_grad, self._e_free_mask],
                    device=self.device,
                )
                wp.utils.array_cast(in_array=_obj_grad, out_array=obj_grad)
                for cg in c_grads:
                    _c_grad = wp.zeros(n_nodes, dtype=vecd_type, device=self.device)
                    wp.utils.array_cast(in_array=cg, out_array=_c_grad)
                    wp.launch(
                        _mask_inplace,
                        dim=n_e,
                        inputs=[_c_grad, self._e_free_mask],
                        device=self.device,
                    )
                    wp.utils.array_cast(in_array=_c_grad, out_array=cg)

                # MMA step
                self.log("Performing MMA step")
                x_new = mma.step(
                    x=self.e.dof_values,
                    f0=obj_val,
                    grad_f0=obj_grad,
                    f_i=c_vals if c_vals else None,
                    grad_f_i=c_grads if c_grads else None,
                    eval_func=eval_func,
                )
                wp.copy(src=x_new, dest=self.e.dof_values)

            # 5. Clamp by bounds
            _bounds_dtype = (
                wp.float64 if self.design_dtype == wp.float64 else wp.float32
            )
            wp.launch(
                _clamp_inplace,
                dim=n_e,
                inputs=[self.e.dof_values, _bounds_dtype(lo), _bounds_dtype(hi)],
                device=self.device,
            )

            # 6. Save if requested
            if save_freq > 0 and it % save_freq == 0:
                self.log("Start saving outputs")
                self.save(time=it)

            # 7. Convergence check
            obj_history.append(obj_val)
            if obj_val < best_obj:
                best_obj = obj_val
                best_obj_iter = it

            if optimizer_type == "mma":
                c_str = ", ".join(f"{v:.3e}" for v in c_vals) if c_vals else "none"
            else:
                c_str = "none (gd)"

            verdict = self._check_convergence(
                it,
                obj_history,
                best_obj_iter,
                rel_tol,
                constraint_str=c_str,
            )
            if verdict == "skip":
                continue
            if verdict == "converged":
                break
        else:
            self.log(f"  Did not converge in {max_iters} iterations")

        self.finalize()

        self._optimize_history = obj_history

    @staticmethod
    def _make_stress_integrand(n: int, scalar_dtype=wp.float32,
                               spatial_material: bool = False):
        raise NotImplementedError()

    @staticmethod
    def _make_tangent_integrand(n: int, scalar_dtype=wp.float32,
                                spatial_material: bool = False):
        raise NotImplementedError()

    @staticmethod
    def _make_body_force_integrand(
        n: int, B_vals: tuple[float, ...], scalar_dtype=wp.float32,
        spatial_material: bool = False,
        acceleration: tuple[float, ...] = (),
    ):
        """b . v, with the body force baked in as a Warp constant.

        A plain Python tuple cannot be closed over here: Warp's codegen resolves
        names referenced inside an integrand and rejects anything that is not
        one of its own types, so indexing a tuple raised "Invalid external
        reference type: <class 'tuple'>" the first time a nonzero body force
        reached this.  Zero body forces never assemble -- see `has_body` -- so
        the failure waited for the first run that actually used one.

        The constant is built in the field's own scalar type, the way every
        other integrand here builds its own -- scalars through `scalar_dtype`,
        matrices through `wp.identity(dtype=scalar_dtype)`.  Spelling it
        `wp.vec3` instead pins it to float32, and a run in double precision then
        failed to compile with "Input types must be exactly the same, got
        ['vec3d', 'vec3f']".  `use_mixed_precision` selects float64 here, so
        that is the setting's on position, not its off one.
        """
        components = [float(B_vals[i]) if i < len(B_vals) else 0.0 for i in range(n)]
        vec_type = wp.types.vector(length=n, dtype=scalar_dtype)
        B_const = wp.constant(vec_type(*components))

        if spatial_material:
            a_components = [
                float(acceleration[i]) if i < len(acceleration) else 0.0
                for i in range(n)
            ]
            a_const = wp.constant(vec_type(*a_components))

            @fem.integrand
            def body_force(s: fem.Sample, v: fem.Field, material: fem.Field):
                rho = scalar_dtype(fem.inner(material, s)[2])
                return wp.dot(v(s), B_const + rho * a_const)
        else:
            @fem.integrand
            def body_force(s: fem.Sample, v: fem.Field):
                return wp.dot(v(s), B_const)

        return body_force

    def _assemble_neumann_bcs(self):
        """Assemble Neumann (traction) boundary loads as a nodal vector.

        High-level: assemble t·v on boundary and lump to nodes consistently.
        Uses edge/face segment lumping to avoid spill across partial boundaries
        and to preserve the resultant for piecewise-constant tractions.
        All operations stay on GPU except scipy.Delaunay (tiny CPU detour).
        """
        _scalar = wp.float64 if self.use_mixed_precision else wp.float32
        _torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        _vecd = wp.types.vector(length=self.dim, dtype=_scalar)
        if not self.neumann_bcs:
            return wp.zeros(self.space.node_count(), dtype=_vecd, device=self.device)

        n_nodes = self.space.node_count()
        traction = torch.zeros(
            n_nodes, self.dim, dtype=_torch_dtype, device=self.device
        )
        node_positions = self._node_positions_torch()

        for nodes_t, traction_values_t in self.neumann_bcs:
            if nodes_t.shape[0] == 0:
                continue

            coords = node_positions[nodes_t.long()].to(dtype=_torch_dtype)
            values = traction_values_t.to(dtype=_torch_dtype)

            if self.dim == 2:
                lump_boundary_tractions_2d(traction, nodes_t, coords, values)
            else:
                # PCA + Delaunay triangulation for 3D boundary
                centered = coords - coords.mean(dim=0)
                _, _, vh = torch.linalg.svd(centered, full_matrices=False)
                coords_2d = centered @ vh[:2].T

                # Delaunay: tiny CPU detour (boundary nodes only)
                from scipy.spatial import Delaunay as _Delaunay

                tri = _Delaunay(coords_2d.cpu().numpy())
                global_tri = nodes_t[torch.from_numpy(tri.simplices).to(nodes_t.device)]
                lump_boundary_tractions_3d(
                    traction,
                    nodes_t,
                    node_positions.to(dtype=_torch_dtype),
                    values,
                    global_tri,
                )

        if not traction.any():
            return wp.zeros(self.space.node_count(), dtype=_vecd, device=self.device)

        return wp.from_torch(traction.contiguous(), dtype=_vecd)

    # traction integrands removed (unused); Neumann loads handled via nodal lumping

    def _body_force_depends_on_design(self):
        """Whether a tape pass must rebuild the body load for its design VJP."""
        return False

    def _assemble_body_forces(self, domain, test, tape=None):
        """Assemble body force contributions using a dimension-generic integrand."""
        result = fem.integrate(
            self._body_force_integrand,
            fields={
                "v": test,
                **({"material": self.material_field} if self._has_spatial_material else {}),
            },
            domain=domain,
            output_dtype=self.dtype,
        )
        return result

    def save(self, main_field=None, time=0):
        """
        Save displacement and computed fields.

        When async_io is enabled the heavy field computation (gradient, strain, stress,
        energy) runs on the VTUWriter background thread.  Only cheap GPU→CPU copies
        of the mutable DOF arrays are done on the main thread before queuing.
        """
        u = main_field if main_field is not None else self.u

        # --- Snapshot mutable GPU arrays cheaply on the main thread ---
        # .numpy() syncs the CUDA stream for just this array, which is much cheaper
        # than the full gradient/stress/strain computation done previously here.
        u_snap = getattr(self, "_u_snapshot_np", None)
        if u_snap is not None and main_field is None:
            u_np = u_snap  # already synced in step() — essentially free
            self._u_snapshot_np = None  # consume; re-captured on next step()
        else:
            u_np = u.dof_values.numpy().copy()  # (n_nodes, dim) float64

        preprocessed_e = getattr(self, "_current_preprocessed_e", None)
        include_raw = preprocessed_e is not None and preprocessed_e is not self.e
        if include_raw:
            e_snap = getattr(self, "_e_snapshot_np", None)
            pre_snap = getattr(self, "_pre_snapshot_np", None)
            if e_snap is not None and main_field is None:
                e_np = e_snap
                pre_np = pre_snap
                self._e_snapshot_np = None
                self._pre_snapshot_np = None
            else:
                e_np = self.e.dof_values.numpy().copy()
                pre_np = preprocessed_e.dof_values.numpy().copy()
        else:
            e_np = None
            pre_np = None

        # --- Snapshot any subclass GPU arrays on the main thread ---
        extra_snap = self._snapshot_extra()

        # --- Snapshot CPU mesh tensors NOW, on the main thread ---
        # Close over the snapshot tuple only, not self._mesh_cache.  This keeps
        # the closure decoupled from any later ``self._mesh_cache`` reassignment
        # (e.g. ShapeOptTrimmed remeshing) and frees the large warp arrays in
        # the old mesh_cache as soon as the main thread moves on.
        mesh_cache = self._mesh_cache
        if "points_cpu" not in mesh_cache:
            pts = mesh_cache.get("points")
            if pts is not None:
                mesh_cache["points_cpu"] = pts.cpu()
                mesh_cache["cells_cpu"] = mesh_cache["cells"].cpu()
            else:
                mesh_info_init = extract_mesh_info_from_warp(
                    self.geo, self.space, self.dim
                )
                import torch as _torch

                mesh_cache["points_cpu"] = _torch.as_tensor(
                    np.asarray(mesh_info_init["points"], dtype=np.float64),
                    dtype=_torch.float64,
                )
                mesh_cache["cells_cpu"] = _torch.as_tensor(
                    np.asarray(mesh_info_init["cells"], dtype=np.int64),
                    dtype=_torch.long,
                )
        points_cpu_snap = mesh_cache["points_cpu"]
        cells_cpu_snap = mesh_cache["cells_cpu"]

        # --- Capture immutable references (safe to close over) ---
        dim = self.dim
        mu = self.mu
        kappa = self.kappa
        material_np = None
        if self._has_spatial_material:
            material_np = np.column_stack((
                extra_snap["material_E"], extra_snap["material_nu"],
                extra_snap["material_rho"],
            ))
        energy_fn = (
            self._strain_energy_density
        )  # bound method; uses self.mu/kappa/dim only

        # Capture geo/space for first-call mesh-cache population (read-only refs).
        geo = self.geo
        space = self.space

        # Capture extension hooks as bound method refs (avoids self access on bg thread)
        get_point_data = self._get_point_data
        get_cell_data = self._get_cell_data

        def _fields_fn(
            u_np=u_np,
            e_np=e_np,
            pre_np=pre_np,
            dim=dim,
            mu=mu,
            kappa=kappa,
            points_cpu=points_cpu_snap,
            cells_cpu=cells_cpu_snap,
            energy_fn=energy_fn,
            geo=geo,
            space=space,
            extra_snap=extra_snap,
            get_point_data=get_point_data,
            get_cell_data=get_cell_data,
            material_np=material_np,
        ):
            import torch

            # Snapshotted CPU mesh tensors — no dependency on any live
            # mesh_cache dict, so save() decouples cleanly from remeshing.
            cpu_cache = {"points": points_cpu, "cells": cells_cpu}
            u_t = torch.from_numpy(u_np).to(dtype=torch.float64)  # CPU tensor

            gradients = compute_node_gradients(
                None, geo, space, dim, mesh_cache=cpu_cache, disp_tensor=u_t
            )
            strain = compute_strain_tensor(None, None, None, dim, gradients=gradients)
            if material_np is None:
                stress_mu, stress_kappa = mu, kappa
            else:
                material_t = torch.from_numpy(material_np).to(
                    device=gradients.device, dtype=gradients.dtype
                )
                stress_mu = material_t[:, 0] / (2.0 * (1.0 + material_t[:, 1]))
                stress_kappa = material_t[:, 0] / (3.0 * (1.0 - 2.0 * material_t[:, 1]))
            stress = compute_stress_tensor(
                None, None, None, dim, stress_mu, stress_kappa, gradients=gradients
            )
            energy = energy_fn(gradients=gradients, material_values=material_np)

            if dim == 2:
                displacement_field = np.column_stack(
                    (u_np, np.zeros((u_np.shape[0], 1), dtype=u_np.dtype))
                )
            else:
                displacement_field = u_np

            point_data = {
                "displacement": displacement_field,
                "strain": strain,
                "stress": stress,
                "strain_energy_density": energy.detach().cpu().numpy(),
            }
            if include_raw:
                point_data["Raw_optimized"] = e_np.copy()
                point_data["Preprocessed_optimized"] = pre_np.copy()

            cells_t = cpu_cache["cells"]
            cell_energy = energy[cells_t].detach().mean(dim=1).cpu().numpy()
            cell_data = {"strain_energy_density": cell_energy}

            point_data = get_point_data(point_data, extra_snap)
            cell_data = get_cell_data(cell_data, extra_snap)

            return point_data, cell_data

        self.vtu_writer.save_timestep(time, _fields_fn)

    def _save_numpy(self, main_field, time):
        """Fallback: Save raw numpy arrays"""
        np.save(
            os.path.join(self.path, f"displacement_t{time}.npy"),
            main_field.dof_values.numpy(),
        )
        self.log(f"Saved numpy: displacement_t{time}.npy")

    def _evaluate_residual(
        self,
        domain,
        test,
        boundary,
        bd_test,
        neumann_load=None,
        body_force=None,
        tape=None,
    ):
        """Assemble internal residual vector for the current displacement.

        *neumann_load* and *body_force* are optional pre-computed GPU vectors
        that avoid recomputing constant external loads every Newton iteration.

        When *tape* is not None, pre-allocates R with requires_grad=True and
        wraps the stress fem.integrate inside ``with tape:`` so the computation
        graph from optimized fields to R is recorded.  Constant external loads
        stay outside the tape; a design-dependent body load is reassembled and
        subtracted on tape in both cached and uncached call modes.
        """
        stress_int = self._stress_integrand
        e_field = getattr(self, "_current_preprocessed_e", self.e)
        _scalar_dtype = wp.float64 if self.use_mixed_precision else wp.float32
        vecd_type = wp.types.vector(length=self.dim, dtype=_scalar_dtype)
        _assembly = None  # auto-detect (dispatch when possible)
        _accum_dtype = _scalar_dtype
        # Explicit quadrature ensures R and K use the same integration rule.
        # Without this, dispatch auto-selects quadrature order from field
        # degrees: R has 3 fields (order=3) and K has 4 (order=4), which
        # breaks Newton consistency (K != dR/du) and causes linear convergence.
        _quadrature = self._default_quadrature()
        _sd = self._integrand_scalar_dtype
        _values = {"mu": _sd(self.mu), "kappa": _sd(self.kappa)}
        _values.update(self._extra_integrate_values())
        _integrate_kwargs = dict(
            fields={"u": self.u, "v": test, "e": e_field},
            values=_values,
            assembly=_assembly,
            accumulate_dtype=_accum_dtype,
        )
        if self._has_spatial_material:
            _integrate_kwargs["fields"]["material"] = self.material_field
        if _quadrature is not None:
            _integrate_kwargs["quadrature"] = _quadrature
        _integrate_kwargs.update(self._extra_integrate_kwargs())
        if tape is not None:
            R = wp.zeros(
                self.space.node_count(),
                dtype=vecd_type,
                requires_grad=True,
                device=self.device,
            )
            with tape:
                fem.integrate(stress_int, output=R, **_integrate_kwargs)
        else:
            R = fem.integrate(stress_int, output_dtype=vecd_type, **_integrate_kwargs)

        if neumann_load is not None:
            fem.linalg.array_axpy(x=neumann_load, y=R, alpha=-self.current_load_scale)
        elif len(self.neumann_bcs) > 0:
            R_neumann = self._assemble_neumann_bcs()
            fem.linalg.array_axpy(x=R_neumann, y=R, alpha=-self.current_load_scale)

        has_configured_body = (
            hasattr(self.params, "B")
            and self.params.B is not None
            and any(b != 0.0 for b in self.params.B)
        ) or (
            self._has_spatial_material
            and any(a != 0.0 for a in self.params.get("material_acceleration", ()))
        )
        if body_force is not None or has_configured_body:
            # A cached load is valid for the ordinary forward solve.  A
            # design-dependent tape pass must reassemble it on tape even when
            # the caller omitted the optional cache, otherwise the body-load
            # VJP silently disappears on the uncached helper path.
            if tape is not None and self._body_force_depends_on_design():
                R_body = self._assemble_body_forces(domain, test, tape=tape)
                with tape:
                    fem.linalg.array_axpy(
                        x=R_body, y=R, alpha=-self.current_load_scale
                    )
            else:
                R_body = body_force
                if R_body is None:
                    R_body = self._assemble_body_forces(domain, test)
                fem.linalg.array_axpy(
                    x=R_body, y=R, alpha=-self.current_load_scale
                )

        # Hook for additional residual contributions (e.g. Nitsche boundary terms)
        self._add_to_residual(R, tape=tape)

        return R

    def _strain_energy_density(self, gradients=None, material_values=None):
        raise NotImplementedError()

    def _bounds(self):
        """Return (lower, upper) bounds for e.dof_values.

        Both are scalars or numpy arrays of shape (n_e,).
        """
        raise NotImplementedError

    def _objective_function(self, tape):
        """Compute scalar objective from self.e and self.u on tape.

        Must be called inside ``with tape:`` context.
        Returns a wp.array of shape (1,) with requires_grad=True.
        """
        raise NotImplementedError

    def _constraints(self):
        """Return list of constraint callables.

        Each callable takes (tape) as argument, is called inside ``with tape:``,
        and returns a wp.array of shape (1,) with requires_grad=True.
        """
        raise NotImplementedError

    # _potential_energy removed (unused)

    def _snapshot_extra(self) -> dict:
        """Main-thread hook: snapshot any extra GPU arrays needed for VTU output.

        Called by ``save()`` on the main thread before the background worker is
        queued.  Subclasses should override this to copy GPU arrays to numpy so
        they can be safely consumed by ``_get_point_data`` / ``_get_cell_data``
        on the background thread.

        Returns:
            dict mapping field names (or arbitrary keys) to numpy arrays.
        """
        if not self._has_spatial_material:
            return {}
        material = self._material_values_torch.detach().cpu().numpy()
        return {
            "material_E": material[:, 0],
            "material_nu": material[:, 1],
            "material_rho": material[:, 2],
            "material_mu": material[:, 0] / (2.0 * (1.0 + material[:, 1])),
            "material_kappa": material[:, 0] / (3.0 * (1.0 - 2.0 * material[:, 1])),
        }

    def _get_point_data(self, point_data: dict, extra: dict) -> dict:
        """Background-thread hook: add extra point-based VTU fields.

        Called after the standard fields (displacement, strain, stress,
        strain_energy_density, Raw_optimized, Preprocessed_optimized) have been
        assembled into *point_data*.  Subclasses may mutate *point_data* in-place
        or return a new dict.

        Args:
            point_data: dict already containing the standard point fields.
            extra: dict returned by ``_snapshot_extra()`` on the main thread.

        Returns:
            Updated *point_data* dict.
        """
        for source, target in (
            ("material_E", "youngs_modulus"),
            ("material_nu", "poisson_ratio"),
            ("material_rho", "density"),
            ("material_mu", "mu"),
            ("material_kappa", "kappa"),
        ):
            if source in extra:
                point_data[target] = extra[source]
        return point_data

    def _get_cell_data(self, cell_data: dict, extra: dict) -> dict:
        """Background-thread hook: add extra cell-based VTU fields.

        Args:
            cell_data: dict already containing ``strain_energy_density``.
            extra: dict returned by ``_snapshot_extra()`` on the main thread.

        Returns:
            Updated *cell_data* dict.
        """
        return cell_data

    def _extra_integrate_values(self):
        """Extra entries for the ``values`` dict passed to fem.integrate.

        Subclasses (e.g. NitscheNeoHookean) override this to inject
        additional per-QP arrays (phi, grad_phi, beta, ...) into the
        stress and tangent integrands.
        """
        return {}

    def _extra_integrate_kwargs(self):
        """Extra keyword arguments for fem.integrate calls.

        Subclasses override to inject quadrature, assembly mode, etc.
        """
        return {}

    def _default_quadrature(self):
        """Return the quadrature for R/K integration.

        For mixed precision (float64), ensures both residual and tangent use
        the same quadrature rule — essential for Newton consistency (K = dR/du).
        Without this, dispatch auto-selects order from field degrees: R has 3
        fields (order=3) and K has 4 (order=4), breaking consistency.

        For non-mixed (float32), returns None to let dispatch auto-select.
        The quadrature mismatch is masked by float32 precision.
        """
        if self.use_mixed_precision:
            mc = self._mesh_cache
            return fem.RegularQuadrature(mc["domain"], order=2)
        return None

    def _tangent_accumulate_dtype(self):
        """Return the accumulate_dtype for tangent assembly.

        Default: wp.float32 (standard). Subclasses (e.g. NitscheNeoHookean)
        may override to use wp.float64 for precision, then cast back in
        _post_tangent_assembly.
        """
        return wp.float32

    def _post_tangent_assembly(self, K):
        """Hook called after tangent K assembly, before condensation/solve.

        Subclasses may override to cast K block types or modify entries.
        Must return the K to use for the rest of the pipeline.
        """
        return K

    def _pre_optimizer_step(self, iteration):
        """Hook: called before the optimizer update in optimize().

        Subclasses can override to save state needed for post-step processing
        (e.g. snapshot of design variables for monotonicity constraints).
        """
        pass

    def _post_optimizer_step(self, iteration):
        """Hook: called after the optimizer update in optimize().

        Subclasses can override to apply constraints on the design variable
        update (e.g. monotonic SDF increase for compliance optimization).
        """
        pass

    def _pre_tape_residual(self, tape):
        """Hook: called before the tape-recorded residual evaluation in step().

        Subclasses can override to perform tape-recorded computations that
        must precede the residual evaluation (e.g. recomputing phi/grad_phi
        on tape so backward flows to design variables).
        """
        pass

    def _post_tape_backward(self):
        """Hook: called after tape.backward() in optimize().

        Subclasses can override to accumulate additional gradients that are
        not captured by the tape (e.g. scattering child SDF gradients to
        parent nodes when geometric_func provides QP SDF directly).
        """
        pass

    def _gd_lr_auto_setup(self):
        """``(fraction, length, length_name)`` when gd_lr_auto is on, else None.

        The reference length is ``gd_lr_auto_length`` when given, else the
        diffuse width epsilon (``self._epsilon``, set from phase_field_epsilon by
        the Nitsche fields; the workflow derives it as the epsilon factor times
        the analysis leaf).  Checked before the first solve, so a bad setting
        fails at once instead of after an iteration's work.
        """
        if not bool(self.params.get("gd_lr_auto", False)):
            return None
        fraction = float(self.params.get("gd_lr_auto_fraction", 0.5))
        length = self.params.get("gd_lr_auto_length", None)
        name = "gd_lr_auto_length"
        if length is None:
            length, name = getattr(self, "_epsilon", None), "epsilon"
        if length is None:
            raise ValueError("gd_lr_auto needs gd_lr_auto_length: this field has no "
                             "phase-field epsilon to scale the first step to")
        length = float(length)
        if not (np.isfinite(fraction) and fraction > 0):
            raise ValueError(f"gd_lr_auto_fraction must be positive, got {fraction}")
        if not (np.isfinite(length) and length > 0):
            raise ValueError(f"gd_lr_auto reference length must be positive, got {length}")
        return fraction, length, name

    def _gd_lr_auto_choose(self, iteration, gd_lr, max_abs_grad, setup):
        """The rate gd_lr_auto steps at this iteration: fraction * length / max|g|.

        Called at every gradient-descent update.  ``max_abs_grad`` is this
        iteration's largest |dJ/ds| over the free design nodes (the masked
        gradient the step is taken along, the same number ``_optimize_steps``
        records), so every step's largest nodal move is exactly
        ``fraction * length``.  That makes it normalized gradient descent: the
        step no longer shrinks as the gradient does, so near an optimum the
        design keeps moving by that much and the objective oscillates rather
        than settling; ``optimize_relative_tol`` may then never trigger and the
        run ends at ``max_optimize_iters`` (see default_config).  The rule is
        the geometry clean-up's (experiments/geometry_cleanup/run.py), which
        applied it once, at iteration 0, with h for epsilon, and the one
        run_forward's failure advice sizes its recommended gd_lr by.

        A zero or non-finite gradient has nothing to scale: ``gd_lr`` (the
        previous iteration's rate, or the configured one at the first update)
        is kept and logged.  Every choice is appended to
        ``self._gd_lr_auto["steps"]``; the record's top level holds the latest.
        """
        fraction, length, name = setup
        record = self._gd_lr_auto
        if record is None:
            record = self._gd_lr_auto = {
                "gd_lr_configured": float(gd_lr), "fraction": fraction,
                "length": length, "length_name": name, "steps": []}
        applied = bool(np.isfinite(max_abs_grad) and max_abs_grad > 0)
        rate = fraction * length / max_abs_grad if applied else float(gd_lr)
        step = {"iteration": int(iteration), "max_abs_grad": float(max_abs_grad),
                "gd_lr": rate, "applied": applied}
        record["steps"].append(step)
        record.update(iteration=step["iteration"], max_abs_grad=step["max_abs_grad"],
                      gd_lr_effective=rate, applied=applied)
        if applied:
            self.log(f"gd_lr_auto: iteration {iteration}: gd_lr {rate:.4e} = {fraction:g} * "
                     f"{name} {length:.4e} / max|dJ/ds| {max_abs_grad:.4e}")
        else:
            self.log(f"gd_lr_auto: iteration {iteration}: max|dJ/ds| is {max_abs_grad:g}; "
                     f"nothing to scale, keeping gd_lr {rate:g}")
        return rate

    def _check_convergence(
        self, it, obj_history, best_obj_iter, rel_tol, constraint_str="none"
    ):
        """Check convergence criteria for the optimization loop.

        Returns ``"converged"`` if the loop should break, ``"skip"`` if
        convergence checking should be skipped this iteration, or
        ``"continue"`` otherwise.
        """
        obj_val = obj_history[-1]
        if it > 0:
            rel_reduction = (1 - obj_history[-1] / obj_history[-2]) * np.sign(
                obj_history[-1]
            )
        else:
            rel_reduction = float("inf")

        self.log(
            f"  Opt iter {it}: obj={obj_val:.6e}, "
            f"constraints=[{constraint_str}], rel_red={rel_reduction:.6e}"
        )

        if it < 20:
            return "skip"

        if it >= 2:
            prev_rel = (1 - obj_history[-2] / obj_history[-3]) * np.sign(
                obj_history[-2]
            )
            if rel_reduction > prev_rel > 0:
                self.log("Reduction increases. Ignoring convergence criterion.")
                return "skip"

        if it >= 2:
            prev_rel = (1 - obj_history[-2] / obj_history[-3]) * np.sign(
                obj_history[-2]
            )
            if 0 < rel_reduction < rel_tol and 0 < prev_rel < rel_tol:
                self.log("  Converged: rel_reduction < tol for last 2 iterations")
                return "converged"

        if it > 500 and (it - best_obj_iter) > 50:
            self.log(
                f"  Converged: no improvement in last 50 iters "
                f"(best at iter {best_obj_iter})"
            )
            return "converged"

        return "continue"

    def _add_to_residual(self, R, tape=None):
        """Hook: add extra contributions to the residual vector in-place.

        Called after the standard residual integration and Neumann/body forces.
        Override to add auxiliary integrations (e.g. Nitsche boundary terms).

        When *tape* is not None, subclasses should wrap their fem.integrate
        calls inside ``with tape:`` so the computation graph is recorded for
        backward differentiation.
        """
        pass

    def _add_to_tangent(self, K):
        """Hook: add extra contributions to the tangent BSR matrix in-place.

        Called after the standard tangent integration but before condensation.
        Override for auxiliary tangent terms (e.g. Nitsche boundary stiffness).
        """
        pass

    def _pre_newton_iteration(self, iteration):
        """Hook called at the start of each Newton iteration in step().

        Override for per-iteration pre-processing (e.g. Nitsche beta recompute).
        """
        pass

    def compute_traction_dirichlet(self, type="PK1"):
        """
        Compute reaction forces at Dirichlet boundaries

        Placeholder - requires boundary integration
        """
        if type != "PK1":
            self.log(
                f"Warning: traction type {type} not supported; using PK1 reactions"
            )

        domain = fem.Cells(self.geo)
        boundary = fem.BoundarySides(self.geo)
        test = fem.make_test(space=self.space, domain=domain)
        bd_test = fem.make_test(space=self.space, domain=boundary)

        residual = self._evaluate_residual(domain, test, boundary, bd_test)
        residual_np = residual.numpy().reshape(-1, self.dim)
        bc_mask_np = (
            self.bc_mask.cpu().numpy()
        )  # small one-time copy for reaction computation

        reactions = {}
        for idx, (nodes, _, components) in enumerate(self.dirichlet_bcs):
            key = f"dirichlet_{idx}"
            nodes_np = nodes.cpu().numpy() if isinstance(nodes, torch.Tensor) else nodes
            reaction = -residual_np[nodes_np].copy()
            if components is not None:
                mask = np.zeros_like(reaction)
                for comp in components:
                    mask[:, comp] = 1.0
                reaction *= mask
            else:
                mask = bc_mask_np[nodes_np]
                reaction *= mask
            reactions[key] = reaction

        return reactions

    def record_traction_dirichlet(self, type="PK1", step=0):
        """Record traction at Dirichlet boundaries"""
        tractions = self.compute_traction_dirichlet(type=type)
        dirs = ["x", "y"] + (["z"] if self.dim == 3 else [])
        for name, values in tractions.items():
            summed = np.sum(values, axis=0)
            for comp, dir_label in enumerate(dirs):
                self.records.loc[step, f"{name}_{dir_label}"] = summed[comp]

    def save_records(self):
        self.records.to_csv(os.path.join(self.path, "records.csv"), index=False)

    def finalize(self):
        """
        Finalize simulation - write PVD collection file

        Call this at the end of your simulation to create the ParaView
        collection file linking all timesteps.
        """
        pvd_path = self.vtu_writer.write_pvd()
        self.vtu_writer.shutdown()
        self.log(f"Simulation finalized. Open {pvd_path} in ParaView to view results.")

    def get_displacement(self):
        """Get displacement field as numpy array (N, dim)."""
        return self.u.dof_values.numpy().copy()


class OctreeField(AbstractField):
    """AbstractField subclass for octree hex meshes with hanging nodes.

    Accepts octree mesh data (nodes, elements, edge/face hanging
    constraints) from sdf2octree and populates affine_constraints
    for the base class to handle constraint enforcement.
    """

    def _build_mesh_state(self):
        """Build mesh from octree_mesh param, or fall back to grid."""
        octree = self.params.get("octree_mesh", None)
        if octree is not None:
            # --- Octree hex mesh path ---
            # Accept torch tensors directly from sdf2octree; keep on GPU.
            nodes_t = torch.as_tensor(
                octree["nodes"], dtype=torch.float32, device=self.device
            )
            elements_t = torch.as_tensor(
                octree["elements"], dtype=torch.int32, device=self.device
            ).contiguous()

            hex_indices = wp.from_torch(elements_t, dtype=wp.int32)

            if self.use_mixed_precision:
                nodes_f64_t = nodes_t.to(dtype=torch.float64)
                positions_f64 = wp.from_torch(nodes_f64_t, dtype=wp.vec3d)
                self.geo = fem.Hexmesh(
                    hex_vertex_indices=hex_indices,
                    positions=positions_f64,
                )
            else:
                positions = wp.from_torch(nodes_t, dtype=wp.vec3)
                self.geo = fem.Hexmesh(
                    hex_vertex_indices=hex_indices,
                    positions=positions,
                )

            # Populate affine_constraints from hanging node data (torch int32 on GPU)
            self.affine_constraints = []
            edge_h = octree.get("edge_hanging", None)
            if edge_h is not None and len(edge_h) > 0:
                edge_data = torch.as_tensor(
                    edge_h.cpu() if hasattr(edge_h, "cpu") else edge_h,
                    dtype=torch.int32,
                ).to(
                    self.device
                )  # (H_e, 3)
                self.affine_constraints.append((edge_data[:, 0], edge_data[:, 1:], 0.5))
            face_h = octree.get("face_hanging", None)
            if face_h is not None and len(face_h) > 0:
                face_data = torch.as_tensor(
                    face_h.cpu() if hasattr(face_h, "cpu") else face_h,
                    dtype=torch.int32,
                ).to(
                    self.device
                )  # (H_f, 5)
                self.affine_constraints.append(
                    (face_data[:, 0], face_data[:, 1:], 0.25)
                )

            # Create function space and field
            self.space = fem.make_polynomial_space(
                geo=self.geo, degree=self.params.elem_order, dtype=self.dtype
            )
            self.u = self.space.make_field()

            # Scalar optimization field
            self.e_space = fem.make_polynomial_space(
                geo=self.geo, degree=self.params.elem_order, dtype=self.design_dtype
            )
            self.e = self.e_space.make_field()
            self._init_e_field()
            self._build_e_fixed_mask()
            if self._has_spatial_material:
                self._init_material_field()

            # Load boundary conditions (also builds GPU constraint data)
            self.dirichlet_bcs = []
            self.neumann_bcs = []
            self.load_boundary_conditions()

            # VTU writer (bypass extract_mesh_info_from_warp which only handles grids)
            mesh_info = {
                "dim": 3,
                "points": nodes_t.cpu(),
                "cells": elements_t.cpu(),
                "cell_types": np.full(elements_t.shape[0], 12, dtype=np.uint8),
            }
            if hasattr(self, "vtu_writer") and self.vtu_writer is not None:
                self.vtu_writer.reinit(mesh_info)
            else:
                async_io = getattr(self.params, "async_io", False)
                self.vtu_writer = VTUWriter(
                    self.path, mesh_info, async_io=async_io, log=self.log
                )

            # Pre-populate mesh_cache so compute_node_gradients works
            self._mesh_cache = {
                "points": nodes_t.to(dtype=torch.float64),
                "cells": elements_t.to(dtype=torch.long),
            }

            # Store nodes/elements as GPU tensors for interpolation in load_octree_mesh
            self._prev_octree_nodes = nodes_t
            self._prev_octree_elements = elements_t
            self._prev_octree_npa = self._infer_full_grid_size(nodes_t)
            self._domain_lo = nodes_t.min(dim=0).values
            self._domain_span = nodes_t.max(dim=0).values - self._domain_lo

            # Reset solver tracking state (matches parent _build_mesh_state)
            self.residual_history = []
            self.relative_residual_history = []
            self.newton_iterations = 0
            self.initial_residual = None
            self.final_residual = None
            self.final_relative_residual = None
            self.line_search_alphas = []
            # Initialised here, never reset.  This is the load ramp's position,
            # owned by whoever drives it, not state derived from the mesh: step()
            # rebuilds this mesh, so overwriting it put every load step back at
            # full load before it solved.  Dropping the line outright is not the
            # answer either -- objects that reach here before any ramp then have
            # no attribute at all.
            self.current_load_scale = getattr(self, "current_load_scale", 1.0)
            self._initial_objective_values = {}

            # Initialize solver cache into _mesh_cache
            self._init_solver_cache()
        else:
            # Fall back to grid-based parent (no affine constraints)
            self.affine_constraints = []
            super()._build_mesh_state()

    @staticmethod
    def _infer_full_grid_size(nodes_t):
        """Infer per-axis full grid dimensions from (possibly sparse) node coords.

        For octree meshes, only active element corners are present (a subset of
        the full (2^depth+1)^3 grid).  We recover the full grid size by finding
        the minimum nonzero spacing along each axis.

        Returns (npa_x, npa_y, npa_z) as a Python tuple of ints.
        """
        lo = nodes_t.min(dim=0).values
        hi = nodes_t.max(dim=0).values
        span = hi - lo
        npas = []
        for ax in range(3):
            ux = torch.unique(nodes_t[:, ax])
            if len(ux) < 2:
                npas.append(2)
                continue
            diffs = ux[1:] - ux[:-1]
            min_sp = diffs.min()
            npas.append(torch.round(span[ax] / min_sp).long().item() + 1)
        return tuple(npas)

    def _interpolate_displacement(self, old_nodes, old_disp, old_elems):
        """Interpolate displacement from an old octree mesh onto the current mesh.

        Three-phase strategy optimized for octree remeshing where most nodes
        are shared between old and new meshes:

        1. **Exact match** — grid-index lookup copies displacement directly
           for nodes present in both meshes (O(V log V), handles majority).
        2. **Trilinear interpolation** — for remaining nodes inside old
           elements, evaluate trilinear shape functions (only the small
           unmatched subset).
        3. **Nearest-neighbor extrapolation** — for nodes outside all old
           elements, copy displacement from the nearest already-assigned node.

        After all phases, applies Dirichlet BCs and hanging-node constraints.

        Args:
            old_nodes: (V_old, 3) float32 torch tensor of old node positions.
            old_disp:  (V_old, D) torch tensor of old displacement values.
            old_elems: (M_old, 8) torch tensor of old element connectivity.
        """
        new_nodes = self._prev_octree_nodes  # set by _build_mesh_state
        device = new_nodes.device
        V_new = len(new_nodes)
        D = old_disp.shape[1]
        new_u = torch.zeros(V_new, D, device=device, dtype=old_disp.dtype)

        # ---- Phase 1: exact grid-index matching ----
        # Compute minimum grid spacing across both meshes (handles depth changes)
        dx = torch.zeros(3, device=device)
        for ax in range(3):
            spacings = []
            for ns in (old_nodes, new_nodes):
                ux = torch.unique(ns[:, ax])
                if len(ux) >= 2:
                    spacings.append((ux[1:] - ux[:-1]).min())
            dx[ax] = min(spacings) if spacings else 1.0
        origin = torch.minimum(old_nodes.min(dim=0).values, new_nodes.min(dim=0).values)

        old_gi = torch.round((old_nodes - origin) / dx).long()
        new_gi = torch.round((new_nodes - origin) / dx).long()

        # Flatten to scalar keys for searchsorted matching
        all_gi = torch.cat([old_gi, new_gi], dim=0)
        npa_max = all_gi.max(dim=0).values + 1
        stride = torch.tensor(
            [npa_max[1] * npa_max[2], npa_max[2], 1], device=device, dtype=torch.long
        )
        old_keys = (old_gi * stride).sum(dim=1)
        new_keys = (new_gi * stride).sum(dim=1)

        old_keys_sorted, old_sort_idx = old_keys.sort()
        positions = torch.searchsorted(old_keys_sorted, new_keys)
        positions_clamped = positions.clamp(max=len(old_keys_sorted) - 1)
        matched = (positions < len(old_keys_sorted)) & (
            old_keys_sorted[positions_clamped] == new_keys
        )
        matched_old_idx = old_sort_idx[positions_clamped]
        new_u[matched] = old_disp[matched_old_idx[matched]]

        n_matched = int(matched.sum())
        remaining = ~matched
        remaining_idx = torch.where(remaining)[0]

        # ---- Phase 2: trilinear interpolation for nodes inside old elements ----
        n_interp = 0
        n_extrap = 0
        if len(remaining_idx) > 0:
            rem_nodes = new_nodes[remaining_idx]

            # Element bounding boxes from old mesh
            elem_pos = old_nodes[old_elems.long()]  # (M, 8, 3)
            elem_lo = elem_pos.min(dim=1).values  # (M, 3)
            elem_hi = elem_pos.max(dim=1).values  # (M, 3)
            elem_size = elem_hi - elem_lo  # (M, 3)

            tol = elem_size.min().item() * 1e-6
            inside = (
                (rem_nodes.unsqueeze(1) >= elem_lo.unsqueeze(0) - tol)
                & (rem_nodes.unsqueeze(1) <= elem_hi.unsqueeze(0) + tol)
            ).all(
                dim=2
            )  # (V_rem, M)
            has_elem = inside.any(dim=1)
            elem_idx = inside.float().argmax(dim=1)

            # Interpolate for nodes inside an old element
            inside_mask = has_elem
            inside_idx = remaining_idx[inside_mask]
            if len(inside_idx) > 0:
                ie = elem_idx[inside_mask]
                local = (new_nodes[inside_idx] - elem_lo[ie]) / elem_size[ie].clamp(
                    min=1e-12
                )
                local.clamp_(0.0, 1.0)
                xi, eta, zeta = local[:, 0], local[:, 1], local[:, 2]

                N = torch.stack(
                    [
                        (1 - xi) * (1 - eta) * (1 - zeta),
                        xi * (1 - eta) * (1 - zeta),
                        xi * eta * (1 - zeta),
                        (1 - xi) * eta * (1 - zeta),
                        (1 - xi) * (1 - eta) * zeta,
                        xi * (1 - eta) * zeta,
                        xi * eta * zeta,
                        (1 - xi) * eta * zeta,
                    ],
                    dim=1,
                )  # (V_inside, 8)

                corner_disp = old_disp[old_elems[ie].long()]  # (V_inside, 8, D)
                new_u[inside_idx] = (N.unsqueeze(-1) * corner_disp).sum(dim=1)
                n_interp = len(inside_idx)

            # ---- Phase 3: nearest-neighbor extrapolation for remaining ----
            outside_mask = ~has_elem
            outside_idx = remaining_idx[outside_mask]
            if len(outside_idx) > 0:
                assigned_idx = (
                    torch.cat([torch.where(matched)[0], inside_idx])
                    if len(inside_idx) > 0
                    else torch.where(matched)[0]
                )
                dists = torch.cdist(
                    new_nodes[outside_idx].float(),
                    new_nodes[assigned_idx].float(),
                )
                nearest = dists.argmin(dim=1)
                new_u[outside_idx] = new_u[assigned_idx[nearest]]
                n_extrap = len(outside_idx)

        self.log(
            f"  Displacement warm-start: {n_matched} matched, "
            f"{n_interp} interpolated, {n_extrap} extrapolated"
        )

        # Write to warp field and apply constraints.  ``new_u`` is a scratch
        # buffer assembled from indexed copies and is non-leaf; detach so
        # ``wp.from_torch`` does not trigger the ".grad on non-leaf" warning.
        _scalar = getattr(self.dtype, "_wp_scalar_type_", self.dtype)
        _torch_dtype = torch.float64 if _scalar == wp.float64 else torch.float32
        self.u.dof_values = wp.from_torch(
            new_u.detach().to(dtype=_torch_dtype).contiguous(),
            dtype=self.dtype,
        )

        n_nodes = self.space.node_count()
        wp.launch(
            _apply_bc_to_field,
            dim=n_nodes,
            inputs=[
                self.u.dof_values,
                self.bc_values_f32,
                self.free_mask_f32,
                self.bc_mask_f32,
            ],
            device=self.device,
        )
        if self._has_affine_constraints:
            self._enforce_affine_constraints(self.bc_values_f32)

    def load_octree_mesh(
        self,
        nodes,
        elements,
        edge_hanging=None,
        face_hanging=None,
        interpolate=True,
    ):
        """Load an octree hex mesh and rebuild solver state.

        Args:
            nodes: (V, 3) node coordinates on a regular cubic grid
                with shape (2**depth+1)^3 dividing the domain.
            elements: (M, 8) hex element connectivity.
            edge_hanging: (H_e, 3) [slave, m1, m2] constraints.
            face_hanging: (H_f, 5) [slave, m1, m2, m3, m4] constraints.
            interpolate: If True and a previous octree mesh exists, interpolate
                the current displacement onto the new mesh as initial guess.
        """
        # Ensure nodes is a GPU torch tensor
        nodes_t = torch.as_tensor(nodes, dtype=torch.float32, device=self.device)

        # Capture old state before rebuild (for interpolation)
        old_state = None
        old_nodes = getattr(self, "_prev_octree_nodes", None)
        old_elems = getattr(self, "_prev_octree_elements", None)
        if (
            interpolate
            and old_nodes is not None
            and old_elems is not None
            and hasattr(self, "u")
        ):
            old_disp = wp.to_torch(self.u.dof_values).clone()  # (V_old, 3) f32
            old_state = (old_nodes, old_disp, old_elems)

        octree_mesh = {
            "nodes": nodes_t,
            "elements": elements,
        }
        if edge_hanging is not None:
            octree_mesh["edge_hanging"] = edge_hanging
        if face_hanging is not None:
            octree_mesh["face_hanging"] = face_hanging
        self.params.update({"octree_mesh": octree_mesh})
        self._build_mesh_state()

        if old_state is not None:
            self._interpolate_displacement(*old_state)
