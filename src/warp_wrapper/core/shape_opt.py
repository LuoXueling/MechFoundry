"""Shape optimization via differentiable Nitsche method.

Optimizes the nodal SDF field (stored in self.e.dof_values) to minimize
strain energy subject to a soft volume penalty.

Objective: SE/SE_0 + w_vol * softplus((V-V_target)/|V_0-V_target|)

The volume term is a one-sided CAP at V_target = f V_0 for every target fraction
f: the denominator is the magnitude |V_0 - V_target| = |1 - f| V_0, so a target
above the start (f > 1) leaves the volume free to grow up to f V_0 and penalises
it beyond, exactly as f < 1 does below the start.

The SE and volume integrands explicitly evaluate phi(e(s)) on tape so that
warp's autodiff traces through:  e(s) -> tanh -> phi -> W(F) -> objective.
This mirrors the topo_opt.py pattern where rho = e_proc(s) appears in the
integrand, giving correct direct + adjoint gradients.

Two integration zones (same as the Nitsche residual):
  - Standard path: phi(e(s)) computed in integrand, regular Gauss quadrature
    (zeroed at boundary elements to avoid double-counting)
  - Boundary path: precomputed phi_arr from _pre_tape_residual, adaptive
    Nitsche quadrature
"""

import copy
import os

import numpy as np
import torch
import warp as wp
import warp.fem as fem
from warp.fem import Coords

from .nitsche_field import NitscheNeoHookean, _STD_GAUSS_PTS, _STD_GAUSS_W
from .nitsche_kernels import sdf_to_phi_f32, sdf_to_phi_f64
from .sdf_reinit import reinitialize_sdf_sussman
from .vtu_writer import VTUSurfaceWriter

# ---------------------------------------------------------------------------
# φ ↔ SDF converters (used by phase_field design mode only)
# ---------------------------------------------------------------------------

_PHI_CLAMP_EPS = 1e-6


def _phi_to_sdf(phi: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Inverse of φ = ½(1 − tanh(d/ε)):  d = ε·atanh(1 − 2·clamp(φ))."""
    p = phi.clamp(_PHI_CLAMP_EPS, 1.0 - _PHI_CLAMP_EPS)
    return epsilon * torch.atanh(1.0 - 2.0 * p)


def _sdf_to_phi(sdf: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Forward map d → φ = ½(1 − tanh(d/ε)), clamped to [0,1]."""
    return (0.5 * (1.0 - torch.tanh(sdf / epsilon))).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Integrands — standard path (φ-direct on a nodal e_phi field)
# ---------------------------------------------------------------------------
#
# Both SDF and PF design-variable modes consume the same phi-direct integrand.
# ``e_phi.dof_values`` is repopulated each forward (on-tape via
# ``sdf_to_phi_f64``) with ``tanh(SDF/ε)`` — the integrand always sees the
# "real" φ derived from SDF.  Boundary elements have their standard-quadrature
# weight zeroed (Nitsche path handles them); non-boundary elements have
# uniform nodal φ (≈ 0 or ≈ 1), so this formulation matches the original
# inline-tanh integrand to machine precision on the volume/SE values.


def _make_hidden_phi_residual_integrand(
    n: int, scalar_dtype=wp.float64, phi_floor=1e-4
):
    """Interior residual with a differentiable SIMP-style φ weight.

    Hidden-φ optimization passes ``e_phi`` containing exactly one to this
    integrand. Consequently its forward value is the ordinary neo-Hookean
    residual, but its φ derivative contributes the missing implicit/adjoint
    design sensitivity.
    """
    _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
    _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
    _one_third = scalar_dtype(1.0 / 3.0)
    _one = scalar_dtype(1.0)
    _phi_floor = scalar_dtype(phi_floor)

    @fem.integrand
    def hidden_phi_residual(
        s: fem.Sample,
        v: fem.Field,
        u: fem.Field,
        e: fem.Field,
        mu: scalar_dtype,
        kappa: scalar_dtype,
    ):
        F = wp.identity(n=n, dtype=scalar_dtype) + fem.grad(u, s)
        J = wp.determinant(F)
        F_inv_T = wp.transpose(wp.inverse(F))
        I1 = wp.ddot(F, F) + _dim_offset

        iso = mu * wp.pow(J, _neg_two_thirds)
        P = iso * F + (kappa * wp.log(J) - iso * I1 * _one_third) * F_inv_T

        phi = e(s)
        phi_weight = (_one - _phi_floor) * phi + _phi_floor
        return phi_weight * wp.ddot(P, fem.grad(v, s))

    return hidden_phi_residual


def _make_se_standard_integrand_phi_direct(
    n: int, scalar_dtype=wp.float64, sharp_volumetric=False
):
    """SE integrand consuming φ directly at QPs (no inline tanh).

    Used in PF mode for the **standard** (non-boundary) quadrature path so
    that φ_at_qp = trilinear(φ_node) is linear in the nodal field — giving
    a non-zero ∂J/∂φ at deep interior nodes (vs the saturated ~0 the
    SDF-based integrand produces there).  Boundary path is unchanged
    (Nitsche QP kernel still derives φ from SDF via tanh + chain rule).
    """
    _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
    _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
    _half = scalar_dtype(0.5)
    _three = scalar_dtype(3.0)
    _one = scalar_dtype(1.0)
    _sharp_vol = sharp_volumetric
    _sharp_k = scalar_dtype(100.0)

    @fem.integrand
    def se_standard_phi(
        s: fem.Sample,
        u: fem.Field,
        e: fem.Field,
        mu: scalar_dtype,
        kappa: scalar_dtype,
        phi_floor: scalar_dtype,
    ):
        phi = e(s)
        if _sharp_vol:
            _hs = _half * (_one + wp.tanh(_sharp_k * (phi - _half)))
            phi_w = (_one - phi_floor) * _hs * phi + phi_floor
        else:
            phi_w = (_one - phi_floor) * phi + phi_floor

        F = wp.identity(n=n, dtype=scalar_dtype) + fem.grad(u, s)
        J = wp.determinant(F)
        I1 = wp.ddot(F, F) + _dim_offset
        J_m23 = wp.pow(J, _neg_two_thirds)
        lnJ = wp.log(J)
        W = _half * mu * (J_m23 * I1 - _three) + _half * kappa * lnJ * lnJ
        return phi_w * W

    return se_standard_phi


def _make_vol_standard_integrand_phi_direct(
    scalar_dtype=wp.float64, sharp_volumetric=False
):
    """Volume integrand consuming φ directly (PF mode standard path)."""
    _half = scalar_dtype(0.5)
    _one = scalar_dtype(1.0)
    _sharp_vol = sharp_volumetric
    _sharp_k = scalar_dtype(100.0)

    @fem.integrand
    def vol_standard_phi(s: fem.Sample, u: fem.Field, e: fem.Field):
        phi = e(s)
        if _sharp_vol:
            _hs = _half * (_one + wp.tanh(_sharp_k * (phi - _half)))
            return _hs * phi
        return phi

    return vol_standard_phi


# ---------------------------------------------------------------------------
# Integrands — boundary path (precomputed phi_arr at Nitsche QPs, on tape)
# ---------------------------------------------------------------------------


def _make_se_boundary_integrand(
    n: int, scalar_dtype=wp.float64, sharp_volumetric=False
):
    """Strain energy at boundary QPs: phi_weight(phi_arr[qp]) * W(F).

    phi_arr is from _pre_tape_residual — already on the warp tape.
    When sharp_volumetric=True, uses H(phi-0.5) weighting.
    """
    _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
    _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
    _half = scalar_dtype(0.5)
    _three = scalar_dtype(3.0)
    _zero = scalar_dtype(0.0)
    _one = scalar_dtype(1.0)
    _sharp_vol = sharp_volumetric
    _sharp_k = scalar_dtype(100.0)

    @fem.integrand
    def se_boundary(
        s: fem.Sample,
        u: fem.Field,
        e: fem.Field,
        mu: scalar_dtype,
        kappa: scalar_dtype,
        phi_arr: wp.array(dtype=wp.float32),
        qp_active_arr: wp.array(dtype=wp.int32),
        phi_floor: scalar_dtype,
    ):
        qp = s.qp_index
        if qp_active_arr[qp] == 0:
            return _zero
        phi = scalar_dtype(phi_arr[qp])
        if _sharp_vol:
            _hs = _half * (_one + wp.tanh(_sharp_k * (phi - _half)))
            phi_w = (_one - phi_floor) * _hs * phi + phi_floor
        else:
            phi_w = (_one - phi_floor) * phi + phi_floor

        F = wp.identity(n=n, dtype=scalar_dtype) + fem.grad(u, s)
        J = wp.determinant(F)
        I1 = wp.ddot(F, F) + _dim_offset
        J_m23 = wp.pow(J, _neg_two_thirds)
        lnJ = wp.log(J)
        W = _half * mu * (J_m23 * I1 - _three) + _half * kappa * lnJ * lnJ

        return phi_w * W

    return se_boundary


def _make_boundary_volume_integrand(scalar_dtype=wp.float64, sharp_volumetric=False):
    """phi at boundary QPs (volume contribution from diffuse boundary).

    When sharp_volumetric=True, uses H(phi-0.5) (step function).
    """
    _half = scalar_dtype(0.5)
    _one = scalar_dtype(1.0)
    _zero = scalar_dtype(0.0)
    _sharp_vol = sharp_volumetric
    _sharp_k = scalar_dtype(100.0)

    @fem.integrand
    def boundary_volume(
        s: fem.Sample,
        u: fem.Field,
        e: fem.Field,
        phi_arr: wp.array(dtype=wp.float32),
        qp_active_arr: wp.array(dtype=wp.int32),
    ):
        qp = s.qp_index
        if qp_active_arr[qp] == 0:
            return _zero
        phi = scalar_dtype(phi_arr[qp])
        if _sharp_vol:
            _hs = _half * (_one + wp.tanh(_sharp_k * (phi - _half)))
            return _hs * phi
        return phi

    return boundary_volume


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _volume_diff_kernel(
    vol: wp.array(dtype=wp.float64),
    target: wp.float64,
    result: wp.array(dtype=wp.float64),
):
    """result = vol - target."""
    result[0] = vol[0] - target


@wp.kernel
def _softplus_scalar(
    x: wp.array(dtype=wp.float64),
    sharpness: wp.float64,
    out: wp.array(dtype=wp.float64),
):
    """out = softplus(x, sharpness) = log(1 + exp(sharpness * x)) / sharpness.

    One-sided: penalizes x > 0 (volume above target), zero for x < 0.
    """
    sx = sharpness * x[0]
    _one = wp.float64(1.0)
    if sx > wp.float64(20.0):
        out[0] = x[0]
    elif sx < wp.float64(-20.0):
        out[0] = wp.float64(0.0)
    else:
        out[0] = wp.log(_one + wp.exp(sx)) / sharpness


@wp.kernel
def _combine_losses(
    se_norm: wp.array(dtype=wp.float64),
    vol_penalty: wp.array(dtype=wp.float64),
    w_vol: wp.float64,
    total: wp.array(dtype=wp.float64),
):
    """total = se_norm + w_vol * vol_penalty."""
    total[0] = se_norm[0] + w_vol * vol_penalty[0]


@wp.kernel
def _volume_diff_kernel_f32(
    vol: wp.array(dtype=wp.float32),
    target: wp.float32,
    result: wp.array(dtype=wp.float32),
):
    result[0] = vol[0] - target


@wp.kernel
def _softplus_scalar_f32(
    x: wp.array(dtype=wp.float32),
    sharpness: wp.float32,
    out: wp.array(dtype=wp.float32),
):
    sx = sharpness * x[0]
    one = wp.float32(1.0)
    if sx > wp.float32(20.0):
        out[0] = x[0]
    elif sx < wp.float32(-20.0):
        out[0] = wp.float32(0.0)
    else:
        out[0] = wp.log(one + wp.exp(sx)) / sharpness


@wp.kernel
def _combine_losses_f32(
    se_norm: wp.array(dtype=wp.float32),
    vol_penalty: wp.array(dtype=wp.float32),
    w_vol: wp.float32,
    total: wp.array(dtype=wp.float32),
):
    total[0] = se_norm[0] + w_vol * vol_penalty[0]


# ---------------------------------------------------------------------------
# ShapeOpt class
# ---------------------------------------------------------------------------


class ShapeOpt(NitscheNeoHookean):
    """Shape optimization on Nitsche-embedded geometry.

    Design variable: nodal SDF (self.e.dof_values).
    Objective: SE/SE_0 + w_vol * softplus((V-V_target)/|V_0-V_target|).
    SE = integral phi(e(s)) * W(F) dx.  Volume = integral phi(e(s)) dx.
    Volume: one-sided softplus penalty (active above target), a cap at
    V_target = f V_0 whether f is below or above 1.
    SDF reinitialization enforces |∇φ| = 1 after each optimizer step.
    """

    def _design_wp_dtype(self):
        return wp.float64 if self.design_dtype == wp.float64 else wp.float32

    def _design_torch_dtype(self):
        return torch.float64 if self._design_wp_dtype() == wp.float64 else torch.float32

    def _copy_design_tensor(self, tensor, dest):
        """Copy a design-field tensor without widening an f32 optimization."""
        dtype = self._design_wp_dtype()
        source = tensor.to(dtype=self._design_torch_dtype(), device=self.device)
        wp.copy(src=wp.from_torch(source.contiguous(), dtype=dtype), dest=dest)

    def _build_e_fixed_mask(self):
        """Freeze SDF at boundary element nodes where BCs are applied.

        Only boundary elements that contain at least one Dirichlet or Neumann
        BC node have their nodes frozen.  The rest of the interface is free
        to evolve during optimization.

        Called during base class init (before Nitsche data exist) — defers
        to base class in that case. Rebuilt in ShapeOpt.__init__ after init.
        """
        if not hasattr(self, "_boundary_elem_mask"):
            super()._build_e_fixed_mask()
            return

        n_e = self.e_space.node_count()
        elements = self._prev_octree_elements.long()[:, :8]

        # Identify boundary elements that touch a BC node
        psi_D = self._psi_D_nodal  # (n_nodes,) 1.0 at Dirichlet nodes
        psi_N = self._psi_N_nodal  # (n_nodes,) 1.0 at Neumann nodes
        bc_nodal = (psi_D + psi_N).clamp(max=1.0)  # 1.0 at any BC node

        boundary_elems = elements[self._boundary_elem_mask]  # (n_bdry, 8)
        elem_bc = bc_nodal[boundary_elems]  # (n_bdry, 8)
        has_bc = elem_bc.any(dim=1)  # (n_bdry,) True if element touches BC

        # Freeze only nodes of BC-touching boundary elements
        fixed = torch.zeros(n_e, dtype=torch.bool, device=self.device)
        bc_elems = boundary_elems[has_bc]
        fixed[bc_elems.reshape(-1)] = True

        n_fixed = int(fixed.sum().item())
        self.log(f"ShapeOpt: freezing SDF at {n_fixed}/{n_e} BC boundary element nodes")

        self._e_fixed_bool = fixed  # keep for reinit
        # Match the optimizer mask to the design field precision.
        _mask_torch_dtype = self._design_torch_dtype()
        _mask_wp_dtype = wp.types.vector(
            length=self._design_dim, dtype=self._design_wp_dtype()
        )
        e_free = (~fixed).to(dtype=_mask_torch_dtype).unsqueeze(-1)
        self._e_free_mask = wp.from_torch(e_free.contiguous(), dtype=_mask_wp_dtype)

    def _use_phase_field_design(self):
        """Whether the phase-field design-variable hybrid is enabled."""
        return self._design_variable == "phase_field"

    def __init__(self, config_module, override_params=None):
        if override_params is None:
            override_params = {}
        # Default to GD optimizer
        override_params.setdefault("optimizer", "gd")
        # Design-variable mode.  In ``phase_field`` mode, the SDF still lives
        # in ``e.dof_values`` and drives the Nitsche boundary path unchanged;
        # a parallel ``e_phi`` field feeds the standard objective integrand
        # so backward yields a non-zero ∂J/∂φ at deep interior nodes
        # (interior carving signal — saturated to zero on the SDF path).
        self._design_variable = str(override_params.get("design_variable", "sdf"))
        super().__init__(config_module, override_params=override_params)

        # Rebuild e_fixed_mask now that _psi_D_nodal/_psi_N_nodal exist
        self._build_e_fixed_mask()

        # Copy initial SDF to self.e.dof_values
        self._copy_design_tensor(self._sdf_nodal, self.e.dof_values)

        # Objective weights
        self._w_vol = float(self.params.get("shape_opt_w_vol", 10.0))

        # Build integrands
        _scalar_dtype = wp.float64 if self.use_mixed_precision else wp.float32
        self._scalar_dtype = _scalar_dtype
        self._sdf_to_phi_kernel = (
            sdf_to_phi_f64 if _scalar_dtype == wp.float64 else sdf_to_phi_f32
        )
        self._volume_diff_kernel = (
            _volume_diff_kernel
            if _scalar_dtype == wp.float64
            else _volume_diff_kernel_f32
        )
        self._softplus_kernel = (
            _softplus_scalar if _scalar_dtype == wp.float64 else _softplus_scalar_f32
        )
        self._combine_losses_kernel = (
            _combine_losses if _scalar_dtype == wp.float64 else _combine_losses_f32
        )

        _sharp_vol = bool(self.params.get("sharp_volumetric", False))
        # Single integrand flavor: phi-direct on the nodal ``e_phi`` field.
        # ``e_phi.dof_values`` is repopulated ON TAPE each forward via the
        # Native f32/f64 SDF→φ kernel (= tanh(SDF/ε)) — matching the field.
        # modes, so the integrand always sees the "real" φ derived from SDF.
        # tape.backward(J) then yields ∂J/∂e_phi at e_phi.dof_values.grad
        # (intermediate from the kernel adjoint) AND ∂J/∂SDF (after chain
        # rule + the Nitsche boundary-path contribution).
        #
        # PF mode additionally extracts ∂J/∂e_phi and applies it OFF-TAPE
        # to a separate persistent ``_hidden_phi`` torch tensor (managed
        # by ``_pf_update_hidden_phi``) that drives the post-opt bg-grid
        # scatter source ``atanh(_hidden_phi)`` — once an interior node's
        # hidden φ dips below 0.5, MC + winding-number reveals a new void
        # surface in the SDF.  ``_hidden_phi`` never enters the tape.
        self._se_standard_integrand = _make_se_standard_integrand_phi_direct(
            self.dim, scalar_dtype=_scalar_dtype, sharp_volumetric=_sharp_vol
        )
        self._vol_standard_integrand = _make_vol_standard_integrand_phi_direct(
            scalar_dtype=_scalar_dtype, sharp_volumetric=_sharp_vol
        )
        self._se_boundary_integrand = _make_se_boundary_integrand(
            self.dim, scalar_dtype=_scalar_dtype, sharp_volumetric=_sharp_vol
        )
        self._boundary_volume_integrand = _make_boundary_volume_integrand(
            scalar_dtype=_scalar_dtype, sharp_volumetric=_sharp_vol
        )

        self.e_phi = self.e_space.make_field()
        self._copy_design_tensor(self._phi_nodal, self.e_phi.dof_values)

        # Hidden-φ mode also feeds ``e_phi`` to the interior residual so the
        # implicit solve contributes dJ/du * du/dφ through the adjoint.  It is
        # reset to one before every solve and remains one through the interior
        # objective, preserving the full-material equilibrium while both terms
        # differentiate with respect to the same field.
        if self._use_phase_field_design():
            self._stress_integrand = _make_hidden_phi_residual_integrand(
                self.dim,
                scalar_dtype=_scalar_dtype,
                phi_floor=self._phi_floor,
            )
            if not bool(getattr(self, "_skip_nitsche_assembly", False)):
                # Nitsche was initially built before e_phi existed.  Rebuild
                # the forward load with the same current-mesh unit probe used
                # by equilibrium and taped residual evaluations, then restore
                # the documented SDF-projected initialization value.  Solve
                # preprocessing resets the probe to one before it is consumed.
                initial_phi = wp.to_torch(self.e_phi.dof_values).detach().clone()
                self._set_e_phi_to_one()
                self._nitsche_body_state_ready = True
                self._refresh_body_force_cache()
                self._copy_design_tensor(initial_phi, self.e_phi.dof_values)

        # Material constants for integrands
        self._eps_inv = _scalar_dtype(1.0 / self._epsilon)
        self._phi_floor_val = _scalar_dtype(self._phi_floor)

        # Build objective quadrature (regular Gauss weights, no phi_weight)
        self._build_objective_quadrature()

        # Grid info for SDF reinitialization (structured grid only)
        self._grid_shape = self.params.get("grid_shape", None)
        self._grid_spacing = self.params.get("grid_spacing", None)
        self._reinit_sdf = bool(self.params.get("reinit_sdf", True))
        self._reinit_every = int(self.params.get("reinit_every", 1))
        self._reinit_n_iters = self.params.get("reinit_n_iters", None)

    def _set_e_phi_to_one(self):
        """Reset ``e_phi`` to unit density for the hidden-φ equilibrium."""
        if not self._use_phase_field_design() or self.e_phi is None:
            return
        ones = torch.ones(
            self.e_space.node_count(),
            dtype=self._design_torch_dtype(),
            device=self.device,
        )
        self._copy_design_tensor(ones, self.e_phi.dof_values)

    def reinit_with_sdf(self, new_octree_mesh):
        """Reinitialize mesh-dependent state from a new (untrimmed) octree mesh.

        Rebuilds geometry, function spaces, FEM fields, Nitsche state, BCs,
        and objective quadrature.  Reuses integrands, material parameters,
        config, and VTU writer (reinit'd, not recreated).
        """
        self.params["octree_mesh"] = new_octree_mesh

        # Clear stale boundary_elem_mask so _build_e_fixed_mask defers to
        # the base class during _build_mesh_state (proper mask is rebuilt after).
        if hasattr(self, "_boundary_elem_mask"):
            del self._boundary_elem_mask

        # Restore original BC dicts — _build_mesh_state replaces them with
        # Nitsche-internal versions, so on reinit we need the user's originals.
        self.params["dirichlet_bcs_dict"] = self._original_dirichlet_bcs_dict
        self.params["neumann_bcs_dict"] = self._original_neumann_bcs_dict

        # Preserve normalization references across reinits
        saved_obj_refs = getattr(self, "_initial_objective_values", None)

        # _build_mesh_state rebuilds: trim → geometry → spaces → u/e fields →
        # BCs → phase field → adaptive quad → QP fields → beta → correction.
        # Defer its design-dependent body cache because e_phi still belongs to
        # the previous function space until the replacement below.
        self._defer_body_cache_until_design_field = True
        self._build_mesh_state()

        # Restore normalization references
        if saved_obj_refs is not None:
            self._initial_objective_values = saved_obj_refs

        # Rebuild e_fixed_mask (needs _psi_D_nodal/_psi_N_nodal from _build_mesh_state)
        self._build_e_fixed_mask()

        # Copy SDF to e.dof_values
        self._copy_design_tensor(self._sdf_nodal, self.e.dof_values)

        # Re-allocate e_phi (function-space changed).  Default-init from
        # current φ = tanh(SDF/ε).  PF mode's ShapeOptTrimmed.step() then
        # overwrites with values interpolated from a stashed bg-grid
        # representation of the previous iteration's e_phi (so interior
        # progress survives adaptive remeshes).
        self.e_phi = self.e_space.make_field()
        self._copy_design_tensor(self._phi_nodal, self.e_phi.dof_values)
        if self._use_phase_field_design():
            self._set_e_phi_to_one()

        # Every dependency now belongs to the rebuilt mesh.  Assemble the
        # forward load only here; in joint mode this uses the unit probe, just
        # like the subsequent forward and taped residual paths.
        self._defer_body_cache_until_design_field = False
        if not bool(getattr(self, "_skip_nitsche_assembly", False)):
            self._nitsche_body_state_ready = True
            self._refresh_body_force_cache()

        # Rebuild objective quadrature (depends on _boundary_elem_mask)
        self._build_objective_quadrature()

        # Skip redundant _preprocess_optimized on next step() — all Nitsche
        # state was just rebuilt by _build_mesh_state above.
        self._skip_next_preprocess = True

        # _current_preprocessed_e (set during the previous step()) now points
        # at the OLD self.e field that _build_mesh_state just replaced.  Any
        # save() called between this reinit and the next step() would see
        # ``_current_preprocessed_e is not self.e`` and try to read a
        # consumed pre-snapshot — clear it so save() falls through cleanly.
        self._current_preprocessed_e = None

    # ------------------------------------------------------------------
    # Objective quadrature (regular Gauss weights, no phi_weight)
    # ------------------------------------------------------------------

    def _build_objective_quadrature(self):
        """Build quadrature for objective integration.

        Regular Gauss weights (NO phi_weight baked in — phi is in the
        integrand). Boundary element weights are zeroed to avoid double-
        counting with the Nitsche boundary path.
        """
        n_elems = self._prev_octree_elements.shape[0]
        std_gauss_pts = torch.tensor(
            _STD_GAUSS_PTS, dtype=torch.float32, device=self.device
        )
        pts = std_gauss_pts.unsqueeze(0).expand(n_elems, -1, -1).contiguous()

        # Regular Gauss weights — same for all elements, no phi_weight
        wts = torch.full(
            (n_elems, 8), _STD_GAUSS_W, dtype=torch.float32, device=self.device
        )
        # Zero boundary elements (handled by Nitsche boundary path)
        _EPS_WEIGHT = 1e-10
        wts[self._boundary_elem_mask] = _EPS_WEIGHT
        wts = wts.contiguous()

        q_torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        q_coord_dtype = wp.vec3d if self.use_mixed_precision else wp.vec3
        q_scalar_dtype = wp.float64 if self.use_mixed_precision else wp.float32
        pts_wp = wp.from_torch(pts.to(dtype=q_torch_dtype), dtype=q_coord_dtype)
        wts_wp = wp.from_torch(wts.to(dtype=q_torch_dtype), dtype=q_scalar_dtype)
        domain = self._mesh_cache["domain"]
        self._objective_quadrature = fem.ExplicitQuadrature(domain, pts_wp, wts_wp)

    # ------------------------------------------------------------------
    # Volume computation (off-tape, for initialization)
    # ------------------------------------------------------------------

    def _compute_volume_fem(self):
        """Compute volume using the same two-zone integration as _objective_function."""
        _sd = self._scalar_dtype
        vol = wp.zeros(1, dtype=_sd, device=self.device)

        # Off-tape refresh of e_phi from current SDF (initialization-time call;
        # not on tape).  Same kernel as _objective_function's on-tape launch.
        wp.launch(
            self._sdf_to_phi_kernel,
            dim=self.e.dof_values.shape[0],
            inputs=[self.e.dof_values, float(self._epsilon)],
            outputs=[self.e_phi.dof_values],
            device=self.device,
        )

        # Standard path: phi(e(s)) with objective quadrature
        _vol_kw = dict(
            fields={"u": self.u, "e": self.e_phi},
            values={},
            output=vol,
            quadrature=self._objective_quadrature,
        )
        if _sd is wp.float64:
            _vol_kw["accumulate_dtype"] = wp.float64
        fem.integrate(self._vol_standard_integrand, **_vol_kw)

        # Boundary path: phi_arr at Nitsche QPs
        self._integrate_boundary(self._boundary_volume_integrand, vol, add=True)

        return float(vol.numpy()[0])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _boundary_obj_values(self, chunk=None):
        """Values dict for boundary objective integrands (volume).

        Returns phi_arr and qp_active_arr from the chunk (or flat arrays).
        """
        if chunk is not None:
            return {
                "phi_arr": chunk.phi_qp,
                "qp_active_arr": chunk.qp_active_wp,
            }
        return {
            "phi_arr": self._phi_qp,
            "qp_active_arr": self._qp_active_wp,
        }

    def _boundary_se_values(self, chunk=None):
        """Values dict for boundary SE integrand.

        Returns phi_arr, qp_active_arr, phi_floor, mu, kappa.
        """
        _sd = self._scalar_dtype
        vals = self._boundary_obj_values(chunk)
        vals.update(
            {
                "phi_floor": self._phi_floor_val,
                "mu": _sd(self.mu),
                "kappa": _sd(self.kappa),
            }
        )
        return vals

    def _integrate_boundary(self, integrand, output, add=True, extra_values=None):
        """Integrate an integrand over boundary QPs (chunked or flat).

        The integrand must accept phi_arr and qp_active_arr as values.
        """
        _sd = self._scalar_dtype
        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            for chunk in self._nitsche_chunks:
                vals = self._boundary_obj_values(chunk)
                if extra_values:
                    vals.update(extra_values)
                kw = dict(
                    output=output,
                    add=add,
                    fields={"u": self.u, "e": self.e},
                    values=vals,
                    quadrature=chunk.quadrature,
                )
                if _sd is wp.float64:
                    kw["accumulate_dtype"] = wp.float64
                fem.integrate(integrand, **kw)
        else:
            vals = self._boundary_obj_values()
            if extra_values:
                vals.update(extra_values)
            kw = dict(
                output=output,
                add=add,
                fields={"u": self.u, "e": self.e},
                values=vals,
                quadrature=self._correction_quadrature,
            )
            if _sd is wp.float64:
                kw["accumulate_dtype"] = wp.float64
            fem.integrate(integrand, **kw)

    def _integrate_boundary_se(self, output, add=True):
        """Integrate SE at boundary QPs (chunked or flat)."""
        _sd = self._scalar_dtype
        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            for chunk in self._nitsche_chunks:
                vals = self._boundary_se_values(chunk)
                kw = dict(
                    output=output,
                    add=add,
                    fields={"u": self.u, "e": self.e},
                    values=vals,
                    quadrature=chunk.quadrature,
                )
                if _sd is wp.float64:
                    kw["accumulate_dtype"] = wp.float64
                fem.integrate(self._se_boundary_integrand, **kw)
        else:
            vals = self._boundary_se_values()
            kw = dict(
                output=output,
                add=add,
                fields={"u": self.u, "e": self.e},
                values=vals,
                quadrature=self._correction_quadrature,
            )
            if _sd is wp.float64:
                kw["accumulate_dtype"] = wp.float64
            fem.integrate(self._se_boundary_integrand, **kw)

    def _post_tape_backward(self):
        super()._post_tape_backward()

        # Apply hanging-node prolongation to SDF gradient.
        # Raw gradient has inconsistent support length across refinement boundaries;
        # prolongate to interpolate each slave from its masters.
        # This leaves masters untouched and smooths the field.
        if self.e.dof_values.grad is not None:
            sdf_grad = wp.to_torch(self.e.dof_values.grad).clone()
            if self._has_affine_constraints:
                g_wp = wp.from_torch(
                    sdf_grad.contiguous(), dtype=self._design_wp_dtype()
                )
                self._prolongate_vector(g_wp)
                sdf_grad = wp.to_torch(g_wp).clone()
            # Write prolongated gradient back to warp field so ShapeOptTrimmed sees it
            wp.copy(
                src=wp.from_torch(sdf_grad.contiguous(), dtype=self._design_wp_dtype()),
                dest=self.e.dof_values.grad,
            )
            self._last_e_grad = sdf_grad

        # PF mode: e_phi receives both the direct objective derivative and the
        # implicit residual derivative.  The latter supplies the adjoint term
        # missing when the interior residual uses the plain neo-Hookean
        # integrand. Apply the same prolongation-only rule (no restriction).
        if (
            self._use_phase_field_design()
            and self.e_phi is not None
            and self.e_phi.dof_values.grad is not None
        ):
            phi_grad = wp.to_torch(self.e_phi.dof_values.grad).clone()
            if self._has_affine_constraints:
                g_wp = wp.from_torch(
                    phi_grad.contiguous(), dtype=self._design_wp_dtype()
                )
                self._prolongate_vector(g_wp)
                phi_grad = wp.to_torch(g_wp).clone()
            # Write prolongated gradient back to warp field for consistency
            wp.copy(
                src=wp.from_torch(phi_grad.contiguous(), dtype=self._design_wp_dtype()),
                dest=self.e_phi.dof_values.grad,
            )
            self._last_e_phi_grad = phi_grad

    def _snapshot_extra(self) -> dict:
        extra = super()._snapshot_extra()
        if hasattr(self, "_last_e_grad"):
            extra["sdf_grad"] = self._last_e_grad.cpu().numpy()
            del self._last_e_grad
        if hasattr(self, "_last_e_phi_grad"):
            extra["phi_grad"] = self._last_e_phi_grad.cpu().numpy()
            del self._last_e_phi_grad
        return extra

    def _get_point_data(self, point_data: dict, extra: dict) -> dict:
        point_data = super()._get_point_data(point_data, extra)
        if "sdf_grad" in extra:
            point_data["sdf_grad"] = extra["sdf_grad"]
        return point_data

    def _bounds(self):
        return (-5.0, 5.0)

    def _constraints(self):
        return []

    def _post_optimizer_step(self, iteration):
        """Reinitialize SDF to enforce |∇φ| = 1 after each optimizer update."""
        if not self._reinit_sdf or self._grid_shape is None:
            return
        if (iteration + 1) % self._reinit_every != 0:
            return

        shape = self._grid_shape  # (Nx, Ny, Nz)
        dx, dy, dz = self._grid_spacing

        # Read current SDF from self.e.dof_values as a GPU tensor
        sdf_flat = wp.to_torch(self.e.dof_values).to(dtype=torch.float64)
        n_expected = shape[0] * shape[1] * shape[2]
        if sdf_flat.shape[0] != n_expected:
            self.log(
                f"ShapeOpt reinit: node count mismatch "
                f"({sdf_flat.shape[0]} vs grid {n_expected}), skipping"
            )
            return

        # Save frozen node values before reinit
        fixed_mask = self._e_fixed_bool
        sdf_frozen = sdf_flat[fixed_mask].clone()

        sdf_3d = sdf_flat.reshape(shape)

        # Measure eikonal violation before
        grad_mag_before = self._eikonal_grad_mag(sdf_3d, dx, dy, dz)

        # Reinitialize (on GPU)
        sdf_reinit_3d = reinitialize_sdf_sussman(
            sdf_3d, dx, dy, dz, n_iters=self._reinit_n_iters
        )
        sdf_reinit_flat = sdf_reinit_3d.reshape(-1)

        # Restore frozen node values
        sdf_reinit_flat[fixed_mask] = sdf_frozen

        # Measure eikonal violation after
        grad_mag_after = self._eikonal_grad_mag(
            sdf_reinit_flat.reshape(shape), dx, dy, dz
        )

        # Write back
        self._copy_design_tensor(sdf_reinit_flat, self.e.dof_values)

        self.log(
            f"  SDF reinit: |∇φ| mean {grad_mag_before:.4f}→{grad_mag_after:.4f}, "
            f"|∇φ|-1 L∞ {abs(grad_mag_before-1):.4f}→{abs(grad_mag_after-1):.4f}"
        )

    @staticmethod
    def _eikonal_grad_mag(sdf_3d, dx, dy, dz):
        """Mean |∇SDF| via central differences (interior nodes only)."""
        gx = (sdf_3d[2:, :, :] - sdf_3d[:-2, :, :]) / (2 * dx)
        gy = (sdf_3d[:, 2:, :] - sdf_3d[:, :-2, :]) / (2 * dy)
        gz = (sdf_3d[:, :, 2:] - sdf_3d[:, :, :-2]) / (2 * dz)
        # Trim to common interior
        gx = gx[:, 1:-1, 1:-1]
        gy = gy[1:-1, :, 1:-1]
        gz = gz[1:-1, 1:-1, :]
        mag = torch.sqrt(gx**2 + gy**2 + gz**2)
        return float(mag.mean().item())

    def _preprocess_optimized(self, tape=None):
        """Sync SDF from self.e, rebuild all Nitsche state for new geometry."""
        if self._use_phase_field_design() and self.e_phi is not None:
            # e_phi is not optimizer-stepped. It remains one through the
            # equilibrium and interior-objective tape operations.
            self._set_e_phi_to_one()
            if tape is not None:
                self.e_phi.dof_values.requires_grad = True

        # After reinit_with_sdf, all Nitsche state is already fresh — skip.
        if getattr(self, "_skip_next_preprocess", False):
            self._skip_next_preprocess = False
            return self.e_phi if self._use_phase_field_design() else self.e

        # Update _sdf_nodal from current self.e.dof_values
        sdf_np = self.e.dof_values.numpy()
        self._sdf_nodal = torch.from_numpy(sdf_np).to(
            dtype=torch.float32, device=self.device
        )

        # Recompute phi_nodal (for boundary element detection + quadrature weights)
        self._phi_nodal = (
            0.5 * (1.0 - torch.tanh(self._sdf_nodal / self._epsilon))
        ).clamp(0.0, 1.0)

        # Full Nitsche state rebuild: adaptive quadrature, standard quadrature,
        # QP fields, BC masks, beta, correction domains
        self._rebuild_nitsche_state()

        # Rebuild objective quadrature (boundary mask may have changed)
        self._build_objective_quadrature()

        # Hidden-φ residuals consume the unit-valued e_phi field. Standard
        # SDF mode retains the original identity preprocessing.
        return self.e_phi if self._use_phase_field_design() else self.e

    # ------------------------------------------------------------------
    # Objective function
    # ------------------------------------------------------------------

    def _objective_function(self, tape):
        """Compute SE/SE_0 + w_vol * softplus(vol_norm).

        ``vol_norm = (V/V_0 - f) / |1 - f|`` with ``f = target_volume_fraction``
        (``V/V_0 - 1`` with ``volume_as_is``).  The denominator is a magnitude,
        taken off the tape: normalising by the SIGNED initial value ``1 - f``
        (what ``_normalized_scalar`` would do) flips the hinge for ``f > 1`` into
        a lower bound that pushes the volume up and never caps it.

        All integrations are ON TAPE (called inside ``with tape:`` from the
        optimize loop). Standard-path SE and volume consume ``e_phi`` directly;
        SDF mode populates it through the tanh projection, while hidden-φ mode
        retains its unit forward value and differentiates with respect to it.
        """
        _sd = self._scalar_dtype

        # SDF mode records SDF → e_phi → objective on tape. Hidden-φ mode
        # keeps the unit e_phi supplied to the interior residual, so direct SE
        # and residual-adjoint derivatives accumulate on the same array without
        # an intervening write. Boundary terms remain SDF-derived in both modes.
        self.e_phi.dof_values.requires_grad = True
        # Warp 1.17 clears intermediate gradients after propagating them unless
        # explicitly retained; phase-field mode consumes this one post-backward.
        self.e_phi.dof_values.retain_grad = True
        if not self._use_phase_field_design():
            wp.launch(
                self._sdf_to_phi_kernel,
                dim=self.e.dof_values.shape[0],
                inputs=[self.e.dof_values, float(self._epsilon)],
                outputs=[self.e_phi.dof_values],
                device=self.device,
            )

        # ---- 1. Strain energy (two-zone) ----
        se = wp.zeros(1, dtype=_sd, requires_grad=True, device=self.device)
        _se_std_kw = dict(
            fields={"u": self.u, "e": self.e_phi},
            values={
                "mu": _sd(self.mu),
                "kappa": _sd(self.kappa),
                "phi_floor": self._phi_floor_val,
            },
            output=se,
            quadrature=self._objective_quadrature,
        )
        if _sd is wp.float64:
            _se_std_kw["accumulate_dtype"] = wp.float64
        fem.integrate(self._se_standard_integrand, **_se_std_kw)

        # Boundary path: phi_arr[qp] * W(F) — SDF-based, unchanged.
        self._integrate_boundary_se(se, add=True)

        se_norm = self._normalized_scalar(se, "strain_energy")

        # ---- 2. Volume (soft penalty) ----
        vol = wp.zeros(1, dtype=_sd, requires_grad=True, device=self.device)
        _vol_std_kw = dict(
            fields={"u": self.u, "e": self.e_phi},
            values={},
            output=vol,
            quadrature=self._objective_quadrature,
        )
        if _sd is wp.float64:
            _vol_std_kw["accumulate_dtype"] = wp.float64
        fem.integrate(self._vol_standard_integrand, **_vol_std_kw)

        # Boundary path: phi_arr[qp]
        self._integrate_boundary(self._boundary_volume_integrand, vol, add=True)

        # Save for logging
        self._current_volume = float(vol.numpy()[0])

        vol_1 = self._normalized_scalar(vol, "volume_raw")
        volume_as_is = bool(self.params.get("volume_as_is", False))

        if volume_as_is:
            # Penalty = 0 when V <= V_0, penalty = 1 when V = 2*V_0.
            # vol_1 - 1 is 0 at initial volume and 1 at twice initial volume.
            target_frac = _sd(1.0)
            self._target_volume = (
                1.0
                / wp.to_torch(self._initial_objective_values["volume_raw"])[0].item()
            )
            vol_diff = wp.zeros(1, dtype=_sd, requires_grad=True, device=self.device)
            wp.launch(
                self._volume_diff_kernel,
                dim=1,
                inputs=[vol_1, target_frac, vol_diff],
                device=self.device,
            )
            vol_norm = vol_diff  # already in units of V_0; no second normalization
        else:
            target_frac = _sd(self.params.get("target_volume_fraction", 0.5))
            self._target_volume = (
                self.params.get("target_volume_fraction", 0.5)
                / wp.to_torch(self._initial_objective_values["volume_raw"])[0].item()
            )
            vol_diff = wp.zeros(1, dtype=_sd, requires_grad=True, device=self.device)
            wp.launch(
                self._volume_diff_kernel,
                dim=1,
                inputs=[vol_1, target_frac, vol_diff],
                device=self.device,
            )
            # Normalise by |1 - f|, NOT by vol_diff's signed iteration-0 value
            # (1 - f): for f > 1 that value is negative and would turn the
            # one-sided softplus into a lower bound (penalising V < f V_0, free
            # above it -- unbounded growth).  The magnitude keeps it a cap at
            # f V_0 for every f; for f < 1 the two are identical.  A Python float
            # held in a constant array, so the scale is off the tape (like
            # _normalized_scalar's).  f = 1 has no natural scale; use V_0, i.e.
            # volume_as_is's normalisation.
            _gap = abs(1.0 - float(self.params.get("target_volume_fraction", 0.5)))
            if _gap < 1.0e-12:
                _gap = 1.0
            if getattr(self, "_initial_objective_values", None) is None:
                self._initial_objective_values = {}
            if "volume_diff" not in self._initial_objective_values:
                self._initial_objective_values["volume_diff"] = wp.array(
                    [1.0 / _gap], dtype=vol_diff.dtype, device=vol_diff.device,
                    requires_grad=False,
                )
            vol_norm = vol_diff * self._initial_objective_values["volume_diff"]

        # One-sided volume penalty: softplus(vol_norm, k)
        vol_penalty = wp.zeros(1, dtype=_sd, requires_grad=True, device=self.device)
        _vol_sharpness = _sd(float(self.params.get("vol_softplus_sharpness", 500.0)))
        wp.launch(
            self._softplus_kernel,
            dim=1,
            inputs=[vol_norm, _vol_sharpness],
            outputs=[vol_penalty],
            device=self.device,
        )

        # ---- Combine ----
        total = wp.zeros(1, dtype=_sd, requires_grad=True, device=self.device)
        wp.launch(
            self._combine_losses_kernel,
            dim=1,
            inputs=[
                se_norm,
                vol_penalty,
                _sd(self._w_vol),
                total,
            ],
            device=self.device,
        )

        # Store raw values for programmatic access (e.g. tests)
        self._last_se_raw = float(se.numpy()[0])
        self._last_vol_raw = float(vol.numpy()[0])

        # Log individual terms
        se_n = float(se_norm.numpy()[0])
        vol_n = float(vol_norm.numpy()[0])
        vol_p = float(vol_penalty.numpy()[0])
        self.log(
            f"  SE={se_n:.4f} (raw={self._last_se_raw:.6e}), "
            f"Vol={self._current_volume:.4f} (raw={self._last_vol_raw:.6e}, "
            f"target={self._target_volume:.4f}, "
            f"norm={vol_n:.4f}, penalty={vol_p:.4f}), "
            f"Total={float(total.numpy()[0]):.6e}"
        )

        return total


# ---------------------------------------------------------------------------
# ShapeOptTrimmed — full-mesh design variable, trimmed-mesh inner solver
# ---------------------------------------------------------------------------


class ShapeOptTrimmed(ShapeOpt):
    """Shape optimization with a full-mesh SDF design variable and a
    trimmed-mesh inner FEM solver.

    ----------------------------------------------------------------
    1. Purpose
    ----------------------------------------------------------------
    ``ShapeOptTrimmed`` is the top-level driver for SDF-based topology /
    shape optimization of a solid body embedded in a background Cartesian
    mesh.  The key architectural split is:

    * **Outer mesh** (``self.e``): a full, *un*trimmed hex (or adaptive
      octree) mesh that spans the entire bounding box including padding.
      The signed-distance function (SDF) lives on this mesh as
      ``self.e.dof_values``.  The outer mesh is **never** used for FEM
      assembly — it exists solely to store and evolve the design variable.

    * **Inner mesh** (``self._inner``, a ``ShapeOpt`` instance): a trimmed
      version of the same mesh, where elements outside the body have been
      removed via the Nitsche trimming criterion.  All forward solves (stiffness
      assembly, load application, Nitsche interface terms) and the adjoint
      backward pass run on this inner instance.  Its node set is a strict
      subset of the outer mesh's nodes.

    The optimizer (gradient descent or MMA) acts on ``self.e.dof_values``.
    After each gradient step the outer SDF is transferred to the inner mesh,
    which is rebuilt (trimmed) from scratch.  This cleanly separates the
    *design representation* (outer SDF) from the *analysis model* (inner FEM).

    ----------------------------------------------------------------
    2. How to Use
    ----------------------------------------------------------------
    Instantiate with a config module path and an ``override_params`` dict::

        from core.shape_opt import ShapeOptTrimmed
        problem = ShapeOptTrimmed(
            "configs.config_shape_opt_cantilever",
            override_params={
                "phase_field_epsilon": 0.02,
                "remeshing": True,
                "remeshing_method": "marchingcubes",
                "reinit_sdf_method": "marchingcubes",
                "sdf_bc_fix_method": "mask",
                "remesh_octree_depth": 7,
                "remesh_init_octree_depth": 4,
                "remeshing_trimesh": {
                    "mesh": box_trimesh,
                    "mesh_center": [2.0, 0.5, 0.5],
                    "h_target": 0.03,
                    "clip_bbox": ((-0.4,-0.4,-0.4),(4.4,1.4,1.4)),
                },
                "max_optimize_iters": 200,
                "shape_opt_w_vol": 0.1,
                "target_volume_fraction": 0.5,
            },
        )
        problem.optimize()

    See ``examples/optimization/sdf_shape_cantilever/main_shape_opt_cantilever.py``
    for a complete CLI driver. See
    ``examples/optimization/latent_shape_opt/main_latent_shape_opt.py`` for
    latent-space optimization using
    ``LatentShapeOpt`` (which reuses this class's remeshing infrastructure).

    **Defining the initial geometry — two paths**

    Exactly one of the following keys must be present in ``override_params``
    when ``remeshing=True``:

    **Path A — explicit surface mesh** (``remeshing_trimesh`` key)
        A dict with keys:

        ``"mesh"`` : trimesh.Trimesh **or** warp.Mesh
            Closed watertight surface mesh in physical coordinates.

            * ``trimesh.Trimesh``: converted internally to a warp BVH via
              :func:`~core.analytical_sdf.build_warp_mesh`
              (``mesh_query_point_sign_winding_number``).  Typical workflow:
              create a primitive (``trimesh.creation.box``, sphere, etc.),
              translate to physical space, then subdivide 1–2 times so
              triangle size roughly matches the target octree leaf size.
            * ``warp.Mesh``: used directly (sign convention auto-detected by
              probing the bounding-box centroid).

            Both inputs are queried via ``hierarchical_extraction`` (dense at
            ``init_octree_depth``, band-only up to ``octree_depth``) to
            produce the raw background-grid SDF and cell-refinement anchors.
            In adaptive mode, every finest cell containing an input vertex is
            forced to ``octree_depth`` through the extractor's must-query
            mechanism.  Later MC-backed remeshes repeat this guarantee around
            the current filtered/restored surface.

        ``"mesh_center"`` : list[float] (len 3)
            Physical centroid of the cube embedding; controls where the
            isotropic cube ``[-1,1]^3`` maps to in physical space.
        ``"h_target"`` : float
            Physical leaf spacing of the background grid.  Sets the cube
            half-extent as ``L/2 = h_target * 2^depth / 2``.
        ``"clip_bbox"`` : ((xlo,ylo,zlo),(xhi,yhi,zhi))
            Physical bounding box for clipping the octree.  Should be
            slightly larger than the body to include padding.

    **Path B — implicit geometry function** (``remeshing_geo_func`` key)
        A dict with keys:

        ``"geo_func"`` : callable ``(1, N, 3) float16 → (1, N, 1) float``
            Signed-distance function in physical units (negative inside).
            Takes cube-normalised query points ``(1, N, 3)`` in
            ``[-1,1]^3`` and returns physical SDF ``(1, N, 1)``.
            A previously initialized solver's ``_mc_geo_func()`` is a
            valid drop-in (the BVH is captured as a closure).
        ``"axis_scale"`` : list[float] (len 3)
            Per-axis cube→physical scale factor.  Copy from the source
            solver's ``_remesh_axis_scale``.
        ``"geo_min"`` : list[float] (len 3)
            Per-axis cube origin offset.  Copy from the source solver's
            ``_remesh_geo_min``.
        ``"h_target"`` : float
            Physical leaf spacing (sets ``grid_spacing``).
        ``"clip_bbox"`` : ((xlo,ylo,zlo),(xhi,yhi,zhi)) — optional.

        ``geo_func`` is queried via ``hierarchical_extraction`` (dense at
        ``init_octree_depth``, band-only up to ``octree_depth``); the
        resulting anchors are reused directly to build the initial octree.
        ``remeshing=True`` is always required when this key is present.

        For latent-space workflows (``LatentShapeOpt``), Path B is the
        expected bootstrap path and ``remeshing_method`` is enforced as
        ``"decoder_marchingcubes"``.

    When ``remeshing=False`` the initial mesh is built directly from the
    config-supplied ``octree_mesh`` (or from a background grid slice) and
    neither key is required.

    **All configurable options**

    ``remeshing`` : bool, default ``False``
        Master switch.  When ``False``, the outer mesh never changes and
        all octree / MC / reinit machinery is bypassed; the class behaves
        like a straightforward SDF-on-hex-mesh optimizer.  When ``True``,
        the outer mesh is rebuilt (or updated) from the current SDF each
        iteration.

    ``remesh_octree_depth`` : int, default ``7``
        Target depth of the adaptive octree used as the outer mesh.  Leaf
        size ≈ ``(domain_size) / 2^depth``.  Higher depth = finer
        resolution near the interface; costs more memory and assembly time.

    ``remesh_init_octree_depth`` : int, default ``4``
        Depth at which the *initial* octree is built from the input
        trimesh.  When ``remesh_init_octree_depth == remesh_octree_depth``
        (the "uniform-hex baseline"), a regular Cartesian hex mesh aligned
        to the background grid is built once and never rebuilt; per-iter
        updates then write the new SDF directly into ``e.dof_values`` via
        a background-grid slice (fast, O(1)).  When
        ``remesh_init_octree_depth < remesh_octree_depth``, the octree
        adapts each iteration (see ``remeshing_method``).

    ``remeshing_method`` : str, ``"interpolate"`` | ``"marchingcubes"`` | ``"decoder_marchingcubes"``
        Controls how the background grid SDF is obtained each iteration and
        how the octree geometry function (``geo_func``) is constructed for
        ``remesh_to_octree``:

        ``"interpolate"``
            Tri-linear interpolation of the cleaned background-grid SDF
            at octree node positions.  Fast, no extra BVH; naturally
            smooth but limited by background-grid resolution.

        ``"marchingcubes"``
            Marching cubes is run on the cleaned background-grid SDF to
            extract an explicit triangle mesh of the zero level set.  A
            warp BVH is built from this surface and ``geo_func`` evaluates
            the analytical signed distance via ``mesh_query_point_sign_winding_number``.
            More accurate near the interface; required by
            ``reinit_sdf_method="marchingcubes"`` and the BC-fix methods
            ``"graft"`` and ``"zip"``.

        ``"decoder_marchingcubes"``
            Calls ``_get_decoder_grid_sdf()`` (must be overridden in a
            subclass) to obtain a fresh ``(n,n,n)`` background-grid SDF,
            bypassing the scatter step from old octree node positions.
            Designed for generative-model optimizers that can re-evaluate
            their decoder over the full grid each iteration (e.g.
            ``LatentShapeOpt`` using ``hierarchical_extraction``).  Use
            with ``reinit_sdf_method="marchingcubes"`` for best surface
            accuracy.  Octree rebuild proceeds via the ``"marchingcubes"``
            path after reinit.

            Incompatible with ``design_variable="phase_field"``.  This
            combination raises ``ValueError`` during initialization.

    ``design_variable`` : str, ``"sdf"`` | ``"phase_field"``
        Chooses the design representation:

        ``"sdf"``
            Standard mode. Gradients flow through the SDF field only.

        ``"phase_field"``
            Hybrid mode introduced after commit ``4328e526``.  Keeps SDF
            as the geometric variable but adds a persistent hidden ``φ``
            interior field so optimization can carve interior voids where
            direct ``∂J/∂SDF`` saturates.

            When ``"phase_field"`` is used, the following parameters are available:
            ``pf_phi_lr`` : float | None, default ``None``
                Learning rate for hidden-``φ`` updates in ``phase_field`` mode.
                ``None`` means fallback to ``gd_lr``.

            ``pf_phi_start_iter`` : int, default ``0``
                First optimizer iteration eligible to apply the deep-interior
                hidden-``φ`` gradient. Lazy initialization takes precedence if
                the field does not exist yet; boundary tracking is never delayed.

            ``phi_reproject_band_k`` : float, default ``3.0``
                Width of the SDF band (in units of ``ε``) where hidden ``φ`` is
                boundary-tracked to ``tanh(SDF/ε)``; outside this band, hidden ``φ``
                evolves via optimization gradients.

    ``reinit_sdf_method`` : str, ``"sussman"`` | ``"marchingcubes"``
        Controls how the background-grid SDF is reinitialized (i.e. made
        to satisfy |∇φ| ≈ 1) after each gradient step:

        ``"sussman"``
            PDE-based iterative eikonal solver.  Restores the eikonal
            condition but *shifts* the zero level set inward by a
            sub-cell amount each iteration.  Over many iterations this
            accumulates a systematic volume loss (~44% of total vol
            reduction over 50 iters in the cantilever benchmark).  Fast
            (GPU kernel, no mesh construction).

        ``"marchingcubes"``
            MC surface extraction + BVH query at every background-grid
            node.  Computes the exact analytical distance to the current
            zero level set; the zero level set itself is not moved, so
            Δvol_proxy ≈ 0 per step.  Eliminates the Sussman drift
            artifact at the cost of one BVH build per iteration.
            Requires ``remeshing_method="marchingcubes"``.

    ``preserve_bc_regions`` : bool, default ``True``
        Whether to build the BC-preservation state at all.  Preservation
        exists to carry the boundary regions across a design update: the
        frozen background mask is read by the reinit that follows a remesh,
        and the graft/zip patch by the stitch beside it.  A run with no
        design update in it -- ``forward()`` without ``optimize()`` -- reaches
        neither, so it pays for a ``bg_n**3`` mask and, under ``"graft"`` or
        ``"zip"``, for a patch cut from the initial surface that nothing then
        reads.  Set ``False`` there.  Leave it ``True`` for any run that
        optimizes: without the frozen mask the reinit lets the BC region
        drift, which is the artifact ``sdf_bc_fix_method`` exists to stop.

    ``sdf_bc_fix_method`` : str, ``"mask"`` | ``"graft"`` | ``"zip"``
        Controls how the boundary-condition (BC) region of the SDF is
        preserved across reinit / remesh steps:

        ``"mask"``
            Cells in the BC region (nodes with ``_reinit_fixed_bool=True``)
            are frozen in the background grid before reinit and restored
            from ``_bg_frozen_values`` afterwards.  Simple and robust;
            may leave a small discontinuity at the BC/free interface after
            Sussman reinit.

        ``"graft"``
            After MC extraction, each vertex in the BC region (within
            ``2 * h_min`` physical distance of a fixed node) is snapped
            to its closest point on the *initial* surface BVH (built once
            at iter 0 from the MC mesh of the initial geometry).  The BVH
            is queried via ``_pin_bc_verts_to_initial_surface``.  Keeps
            the BC surface topology and position frozen without cutting
            holes; topology-preserving and watertight.  Requires
            ``remeshing_method="marchingcubes"``.

        ``"zip"``
            Stores the *exact* BC patch (vertices + faces) from the iter-0
            MC surface.  On each subsequent iter, the current MC mesh is
            cut open in the BC region, the stored patch is inserted, and
            the hole boundary is re-stitched with a Fuchs-Kedem-Uselton
            dynamic-programming zipper (``zipper_stitch`` in
            ``remeshing.py``).  Produces results essentially identical to
            ``"graft"`` (<0.3% difference over 50 iters) but guarantees
            the BC surface is the exact iter-0 mesh rather than a
            closest-point projection.  Requires
            ``remeshing_method="marchingcubes"`` and
            ``reinit_sdf_method="marchingcubes"``.

    ``warm_start_u`` : bool, default ``True``
        When ``True``, the displacement field from the previous iteration's
        inner mesh is interpolated onto the new inner mesh via
        ``_interpolate_displacement`` (exact position matching for shared
        nodes, trilinear fallback for newly introduced nodes).  Reduces the
        number of Newton iterations needed to converge the forward solve at
        each step.

    **Common optimization knobs exposed by CLI drivers**

    These are passed through ``override_params`` by
    the SDF- and latent-shape examples under ``examples/optimization``:

    ``max_optimize_iters``, ``shape_opt_w_vol``, ``target_volume_fraction``,
    ``volume_as_is``, ``optimize_relative_tol``, ``optimizer``, ``gd_lr``,
    ``latent_weight_decay``, ``latent_optimizer``, ``output_root``.

    Latent-driver constructor arguments such as ``latent_lr``,
    ``remesh_every``, ``n_grad_points``, and ``decoder_chunk_size`` are
    handled by ``LatentShapeOpt`` itself (not by this class's
    ``override_params`` parser).

    ----------------------------------------------------------------
    3. Detailed per-iteration workflow
    ----------------------------------------------------------------
    The ``optimize()`` loop (inherited from ``ShapeOpt``) calls the
    following sequence each iteration:

    **A. Apply deferred outer remesh** (``step()``, remeshing path only)
        If ``_pending_remesh_octree`` was set by the *previous* iteration's
        ``_post_optimizer_step_remesh_impl``, ``reinit_with_sdf`` is called
        on ``self.e`` (the outer mesh) with the new octree node positions and
        elements.  This atomically swaps the outer mesh topology without
        touching the inner.  The ``save()`` snapshot taken *before* this
        swap captures the consistent (outer SDF, inner displacement) state
        for VTU/GLB output.

    **B. Sync outer SDF → inner** (``step()``)
        ``self.e.dof_values`` (the outer SDF) is sliced to the inner mesh's
        node index set and written into ``self._inner.e.dof_values``.

    **C. Rebuild inner trimmed mesh** (``step()`` → ``_inner.reinit_with_sdf()``)
        The inner ``ShapeOpt`` re-runs Nitsche trimming: elements with
        ``SDF > ε_trim`` are removed, giving a new trimmed hex mesh.  The
        trimming boundary (cut cells) and Nitsche interface terms are
        recomputed.

    **D. Warm-start displacement** (``step()``, if ``warm_start_u=True``)
        The previous inner displacement is interpolated from
        ``(_prev_octree_nodes, u_old, _prev_octree_elements)`` onto the
        new inner node set.  Nodes present in both old and new meshes are
        copied exactly; new nodes use trilinear interpolation from the old
        element that contains them.

    **E. Forward solve** (``step()`` → ``_inner.step()``)
        The Nitsche FEM linear system is assembled and solved on the
        trimmed inner mesh.  This computes the displacement field ``u``
        and accumulates the strain energy (SE).

    **F. Objective evaluation** (``_objective_function()``)
        ``SE/SE_0 + w_vol * vol_penalty`` is computed on the inner trimmed mesh,
        with ``V = sum(elem_volumes * phase_field_weights)`` measured against
        the starting volume ``V_0``, not a bounding volume.
        ``vol_penalty = softplus_k((V/V_0 - f) / |1 - f|)``, i.e.
        ``softplus_k((V - V_target)/|V_0 - V_target|)`` with ``V_target = f V_0``
        and ``f = target_volume_fraction`` -- a cap at ``f V_0`` for ``f`` below
        or above 1 (above 1 the volume may grow up to it); with ``volume_as_is`` it is
        ``softplus_k(V/V_0 - 1)``.  ``softplus_k(x) = log(1 + exp(k x)) / k``,
        ``k = vol_softplus_sharpness`` (500): one-sided and nearly a hinge, so
        it is about zero below the target and grows linearly above it -- which
        is why the volume, once there, alternates across the target.

    **G. Adjoint / backward** (tape backward → ``_post_tape_backward()``)
        Warp's tape records the forward solve; ``tape.backward()`` accumulates
        ``∂L/∂SDF`` at inner nodes.  ``_post_tape_backward`` then scatters
        the inner-mesh gradient to the full outer mesh (zero for nodes not
        in the trimmed inner mesh).

    **H. Optimizer step** (GD or MMA)
        The outer SDF is updated by the optimizer (e.g.
        ``e.dof_values -= lr * grad``).  The free-node mask ensures BC
        nodes (the fixed wall) remain at their initial SDF values.

    **I. Post-optimizer remesh step** (``_post_optimizer_step_remesh_impl()``,
        remeshing path only)

        *I.1 Scatter to background grid*
            ``scatter_octree_to_cube_grid`` maps ``e.dof_values`` (on the
            outer octree nodes) to a uniform background cube grid of shape
            ``(n, n, n)`` via nearest-voxel assignment.  The background
            grid spans the clip bbox and is aligned to the octree lattice.

        *I.2 Post-scatter hook* (``_post_scatter_bg_grid_hook``)
            No-op in the base class; subclasses override to filter the
            scattered bg grid before BC freeze (e.g. ``LatentShapeOpt``
            removes disconnected interior islands via
            ``filter_largest_interior_component``).

        *I.3 Freeze BC region (pre-reinit)*
            Background-grid cells corresponding to boundary-condition nodes
            (``_reinit_fixed_bool``, captured once as ``_bg_frozen_mask``)
            are overwritten with their iter-0 values (``_bg_frozen_values``)
            **before** reinit runs.  This ensures the eikonal solver or MC
            BVH query sees the correct BC SDF as a boundary condition and
            does not propagate drifted values inward.

        *I.4 SDF reinitialization (with BC fix)*
            Dispatched on ``reinit_sdf_method``:

            - **Sussman**: ``_reinit_sdf_sussman`` runs the iterative eikonal
              PDE on the background grid; logs ``|∇φ|`` stats and
              vol_proxy before/after (Δvol_proxy is typically −0.001 to
              −0.005 per iter due to zero level set inward shift).
              ``_bg_frozen_values`` are re-applied after Sussman because
              the PDE can drift BC cells.
            - **marchingcubes**: ``_mc_reinit_bg_grid`` extracts the MC
              surface from the (already pre-frozen) background grid, then
              dispatches ``sdf_bc_fix_method`` **inside**
              ``_build_mc_surface_and_bvh`` before BVH construction:

              - **mask**: pre-freeze (I.3) is sufficient; no further action.
              - **graft**: BC-region MC vertices are snapped to the
                initial-surface BVH (``_pin_bc_verts_to_initial_surface``).
              - **zip**: BC region is hole-cut, stored iter-0 patch is
                inserted, and the hole boundary is stitched by the
                Fuchs-Kedem-Uselton zipper (``zipper_stitch``).

              After BC fix, analytical SDF is queried at every
              background-grid node via ``compute_nodal_sdf``; Δvol_proxy ≈ 0.

        *I.5 Uniform-hex fast path* (``init_depth == final_depth``)
            The background-grid slice that corresponds to the outer hex
            mesh nodes is extracted and copied directly into
            ``e.dof_values`` via a dtype-matched ``wp.copy``.  No octree
            rebuild is triggered; the outer mesh topology is fixed for
            the entire run.

        *I.6 Adaptive remesh* (``init_depth < final_depth``)
            ``geo_func`` is constructed (trilinear or BVH, dispatched on
            ``remeshing_method``) and ``remesh_to_octree`` is called to
            build a new adaptive octree.  Nodes are scaled to physical
            space via ``scale_octree_nodes_to_physical`` and clipped to
            the ``clip_bbox``.  Nodal SDF on the new octree is assigned
            via trilinear interpolation of ``cleaned`` (not from the BVH
            geo_func, which returns exact distances).

            After building the new octree, ``e.dof_values`` on the
            *current* outer octree is also overwritten with
            trilinear(``cleaned``) so that ``save()`` visualizes the
            reinitialized SDF rather than stale pre-reinit values.

            The new octree is stored as ``_pending_remesh_octree`` and
            applied at the *start* of the next iteration (step A), at
            which point ``e.dof_values`` is overwritten again from the
            pending octree's SDF.

    **J. Snapshot / output** (``save()``)
        Called just before step A of the next iteration.  Writes VTU
        files for the outer SDF (``vtu_sdf/``), the inner solution
        (``vtu/``), and optionally surface meshes (``surface_vtu/``,
        ``surface_glb/``) extracted via MC from the cleaned background
        grid.  Log entries record SE, Vol, Total loss, and timing.
    """

    def __init__(self, config_module, override_params=None):
        if override_params is None:
            override_params = {}

        # Set ``_design_variable`` early so ``_use_phase_field_design()``
        # (called from ``_build_mc_surface_and_bvh`` via the bootstrap
        # ``_init_remeshing_state`` → ``_mc_reinit_bg_grid`` chain below)
        # has a valid attribute to check.  ``ShapeOpt.__init__`` will
        # re-set it later, but with the same value, so this is idempotent.
        self._design_variable = str(override_params.get("design_variable", "sdf"))

        # ----- New optional remeshing/reinit features (default: disabled) -----
        # When _remeshing_enabled is False, every other branch added below is
        # skipped and the existing uniform-mesh code path runs unchanged.
        self._remeshing_enabled = bool(override_params.get("remeshing", False))
        self._remeshing_method = override_params.get("remeshing_method", "interpolate")
        self._reinit_sdf_method = override_params.get("reinit_sdf_method", "sussman")
        self._sdf_bc_fix_method = override_params.get("sdf_bc_fix_method", "mask")
        self._preserve_bc_regions = bool(
            override_params.get("preserve_bc_regions", True)
        )
        self._remesh_octree_depth = int(override_params.get("remesh_octree_depth", 7))
        self._remesh_init_octree_depth = int(
            override_params.get("remesh_init_octree_depth", 4)
        )

        if (
            self._design_variable == "phase_field"
            and self._remeshing_method == "decoder_marchingcubes"
        ):
            raise ValueError(
                "design_variable='phase_field' is incompatible with "
                "remeshing_method='decoder_marchingcubes'."
            )

        if self._remeshing_enabled:
            # Bootstrap the first outer octree from the caller's uniform SDF.
            # Mutates ``override_params`` in place: replaces ``octree_mesh``
            # with the freshly built physical-space octree, and rebinds
            # ``grid_shape`` / ``grid_spacing`` to the background cube grid
            # used by Sussman from now on.
            self._init_remeshing_state(override_params)
            _mc_active = (
                self._remeshing_method == "marchingcubes"
                or self._reinit_sdf_method == "marchingcubes"
            )
            if _mc_active:
                # MC path: we have a cached surface mesh after each iter —
                # enable surface VTU + GLB by default (save_from_mesh path).
                override_params.setdefault("output_surface_vtu", True)
                override_params.setdefault("output_surface_glb", True)
                override_params.setdefault("output_surface_glb_deformed", True)
            else:
                # Interpolate path: no MC mesh, structured-grid writer won't
                # work on the adaptive outer octree.
                override_params["output_surface_vtu"] = False
                override_params["output_surface_glb"] = False
                override_params["output_surface_glb_deformed"] = False

        self._config_module = config_module
        self._override_params = override_params
        # Create inner ShapeOpt on trimmed mesh
        self._inner_override = dict(self._override_params)
        self._inner_override["trim_exterior"] = True

        # Initialize outer on full (untrimmed) mesh
        outer_override = dict(override_params)
        outer_override["trim_exterior"] = False
        outer_override["sharp_volumetric"] = False  # prevent auto-trim

        # The outer ShapeOptTrimmed is a thin wrapper that never runs a
        # Nitsche assembly itself — step() and _objective_function() both
        # delegate to self._inner.  We can therefore skip the assembly-only
        # phases of _build_mesh_state (beta, correction_dom, save_quad_vtu),
        # which on a 129^3 cube / 243k-element mesh cost ~700 ms per call.
        # This applies every time _build_mesh_state runs on the outer —
        # both at __init__ and on every remeshing-mode reinit_with_sdf.
        self._skip_nitsche_assembly = True

        super().__init__(config_module, override_params=outer_override)

        self._warm_start_u = bool(self.params.get("warm_start_u", True))

        # Redirect outer VTU writer to vtu_sdf/ (SDF-only on full mesh)
        # Inner's writer will use vtu/ (full solution on trimmed mesh)
        sdf_vtu_dir = os.path.join(self.path, "vtu_sdf")
        os.makedirs(sdf_vtu_dir, exist_ok=True)
        self.vtu_writer.vtu_dir = sdf_vtu_dir

        # Surface writer: marching cubes → VTU + GLB (gated by config).
        # Subclasses without a structured grid (e.g. LatentShapeOpt, which
        # builds an octree directly from a decoder) pass grid_shape=None and
        # drive the writer exclusively through save_from_mesh(), using
        # their own pre-extracted surface mesh.
        _surf_vtu = bool(self.params.get("output_surface_vtu", False))
        _surf_glb = bool(self.params.get("output_surface_glb", False))
        _surf_glb_def = bool(self.params.get("output_surface_glb_deformed", False))
        if _surf_vtu or _surf_glb or _surf_glb_def:
            if (
                self._grid_shape is not None
                and self._grid_spacing is not None
                and not getattr(self, "_remeshing_enabled", False)
            ):
                pts = self.vtu_writer.mesh_info["points"]
                if isinstance(pts, torch.Tensor):
                    pts = pts.detach().cpu().numpy()
                else:
                    pts = np.asarray(pts)
                bounds_lo = pts.min(axis=0)
                bounds_hi = pts.max(axis=0)
                grid_shape = self._grid_shape
            else:
                bounds_lo = None
                bounds_hi = None
                grid_shape = None
            async_io = getattr(self.params, "async_io", False)
            self._surface_writer = VTUSurfaceWriter(
                self.path,
                grid_shape=grid_shape,
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
                write_vtu=_surf_vtu,
                write_glb=_surf_glb,
                write_glb_deformed=_surf_glb_def,
                simplify_n_faces=int(self.params.get("surface_simplify_n_faces", -1)),
                async_io=async_io,
                log=self.log,
            )

    # ------------------------------------------------------------------
    # Delegate forward/backward to the trimmed inner instance
    # ------------------------------------------------------------------

    def step(self, tape=None):
        """Sync outer SDF to inner, then solve on trimmed mesh."""
        # Apply any deferred outer remesh from the previous iter's
        # _post_optimizer_step_remesh_impl.  Doing it here (not in post_opt)
        # lets save() run between iters on a consistent OLD outer+inner
        # snapshot, and avoids the redundant inner reinit that
        # _sync_inner_after_remesh used to do — the normal inner rebuild
        # below already picks up the new outer via orig_mesh.
        pending = getattr(self, "_pending_remesh_octree", None)
        if pending is not None:
            self._pending_remesh_octree = None
            self.reinit_with_sdf(pending)
            # reinit_with_sdf re-allocated self.e (fresh Warp array); the
            # optimize loop's line ``self.e.dof_values.requires_grad = True``
            # already fired BEFORE this step() was called, so the new array
            # needs it set again or the next ``.grad.zero_()`` hits None.
            self.e.dof_values.requires_grad = True
            n_new = int(self.e_space.node_count())
            n_old = int(self._pending_remesh_n_old)
            self.log(
                f"  Remesh: outer octree {n_old} → {n_new} nodes "
                f"(applied deferred remesh at step() start)"
            )
            # PF mode: carry ``_hidden_phi`` from the OLD outer mesh onto
            # the NEW outer mesh using a minimal-change policy:
            #   • Nodes that exist in both meshes (same bg-lattice index)
            #     copy the old ``_hidden_phi`` verbatim — no interp loss
            #     for carried-over dips.
            #   • Nodes that are new (refined positions not present in
            #     old octree) get parent-element trilinear interpolation
            #     from the OLD element containing them.  This keeps
            #     refined boundary/void-wall nodes aligned with the old
            #     element's nodal phi rather than washing them out
            #     against a dense bg-grid baseline.
            stash = getattr(self, "_pending_old_hidden_phi_stash", None)
            if stash is not None and self._use_phase_field_design():
                self._carry_over_hidden_phi_parent_interp(stash)
                self._pending_old_hidden_phi_stash = None

        # Snapshot the shape this step() solves on, for save().  Between
        # iterations _post_optimizer_step rebuilds the outputs for the NEXT shape
        # (it overwrites _mc_verts_norm with the next surface and copies the
        # reinitialized next-iter SDF into e.dof_values) before save() runs.
        # Capturing them here -- after any deferred remesh, before the FEM solve
        # below -- makes every output at iter i show iter i's shape (the one it
        # actually computed on), consistent with the OLD octree + displacement
        # save() interpolates.  e.dof_values is overwritten in place so it must
        # be copied; _mc_verts_norm is reassigned so a reference suffices.
        if getattr(self, "_mc_verts_norm", None) is not None:
            self._save_mc_verts_norm = self._mc_verts_norm
            self._save_mc_faces = self._mc_faces
        if getattr(self, "e", None) is not None:
            self._save_sdf = self.e.dof_values.numpy().copy()
        # PF mode: hidden φ used this iter; _pf_update_hidden_phi (in
        # _post_optimizer_step) GD-updates it to the next iter before save().
        # Cloned -- some paths mutate _hidden_phi in place via a shared wp view.
        self._save_hidden_phi = (
            self._hidden_phi.detach().clone()
            if getattr(self, "_hidden_phi", None) is not None
            else None
        )

        # Build a fresh octree_mesh dict with the current outer SDF
        orig_mesh = self.params.get("octree_mesh")
        _inner_mesh = {
            k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v))
            for k, v in orig_mesh.items()
        }
        sdf_key = "sdf" if "sdf" in _inner_mesh else "node_sdf"
        _inner_mesh[sdf_key] = wp.to_torch(self.e.dof_values).detach().to(torch.float32)

        if getattr(self, "_inner", None) is None:
            # First call: create the inner solver
            inner_override = dict(self._inner_override)
            inner_override["octree_mesh"] = _inner_mesh
            self._inner = ShapeOpt(self._config_module, override_params=inner_override)
            # Share output directory and logger with outer
            import shutil

            inner_old_path = self._inner.path
            self._inner.path = self.path
            self._inner.log = self.log
            # Redirect VTU writer to outer's directory (inner has trimmed mesh)
            self._inner.vtu_writer.output_dir = self.path
            self._inner.vtu_writer.vtu_dir = os.path.join(self.path, "vtu")
            self._inner.vtu_writer._log = self.log
            if inner_old_path != self.path:
                shutil.rmtree(inner_old_path, ignore_errors=True)
            # Skip redundant _preprocess_optimized — __init__ just built
            # all Nitsche state via _build_mesh_state.
            self._inner._skip_next_preprocess = True
        else:
            # Capture warm-start state before reinit zeros the displacement.
            old_inner_state = None
            if self._warm_start_u:
                old_inner_state = (
                    self._inner._prev_octree_nodes.clone(),
                    wp.to_torch(self._inner.u.dof_values).clone(),
                    self._inner._prev_octree_elements.clone(),
                )

            # Subsequent calls: reinit in-place (no new allocation)
            # (reinit_with_sdf sets _skip_next_preprocess = True)
            self._inner.reinit_with_sdf(_inner_mesh)

            # Warm-start: interpolate displacement onto new inner mesh
            if old_inner_state is not None:
                self._inner._interpolate_displacement(*old_inner_state)

        self._trim_active_indices = _inner_mesh["trim_active_indices"]

        # No need to sync outer e_phi → inner e_phi: in both modes the
        # inner's _objective_function repopulates inner.e_phi on tape
        # each forward via the sdf_to_phi_f64 kernel.

        # Rebuild outer free mask from inner's BC freeze mask
        self._rebuild_outer_free_mask()

        # First call only: initialize BC frozen state from the clean initial SDF.
        # _reinit_fixed_bool is now available; _sussman_grid_sdf is the
        # MC-reinitialized clean SDF from _init_remeshing_state, so the frozen
        # values are consistent with what the per-iter reinit produces.
        # Only for a run that will update the design: both the mask and the
        # graft/zip patch are read from the remesh path alone, so a forward
        # run builds them for nothing -- the mask is bg_n**3 wide, and "zip"
        # additionally validates the boundary loops of a patch it cuts off the
        # initial surface, and raises when it cannot resolve them.
        if (self._remeshing_enabled and self._preserve_bc_regions
                and not hasattr(self, "_bg_frozen_mask")):
            self._init_bg_frozen_mask()
            if self._sdf_bc_fix_method in ("graft", "zip"):
                self._init_bc_graft()

        self.log(
            f"ShapeOptTrimmed: outer {self.e_space.node_count()} nodes, "
            f"inner {self._inner.e_space.node_count()} nodes "
            f"(trimmed {self.e_space.node_count() - self._inner.e_space.node_count()})"
        )
        self._inner.current_load_scale = self.current_load_scale
        self._inner.step(tape=tape)

    def _rebuild_outer_free_mask(self):
        """Map inner's _e_fixed_bool to outer mesh and rebuild outer mask."""
        n_outer = self.e_space.node_count()
        active = self._trim_active_indices.long()

        # --- Optimizer mask: freeze ALL exterior + BC-adjacent active nodes ---
        fixed = torch.ones(n_outer, dtype=torch.bool, device=self.device)
        # Unfreeze active nodes that the inner considers free
        inner_free = ~self._inner._e_fixed_bool
        fixed[active[inner_free]] = False

        self._e_fixed_bool = fixed
        _mask_torch_dtype = self._design_torch_dtype()
        _mask_wp_dtype = wp.types.vector(
            length=self._design_dim, dtype=self._design_wp_dtype()
        )
        e_free = (~fixed).to(dtype=_mask_torch_dtype).unsqueeze(-1)
        self._e_free_mask = wp.from_torch(e_free.contiguous(), dtype=_mask_wp_dtype)

        # --- Reinit mask: freeze ONLY BC-adjacent nodes (exterior is free) ---
        reinit_fixed = torch.zeros(n_outer, dtype=torch.bool, device=self.device)
        inner_fixed = self._inner._e_fixed_bool
        reinit_fixed[active[inner_fixed]] = True
        self._reinit_fixed_bool = reinit_fixed

    def _post_optimizer_step(self, iteration):
        """SDF reinit using BC-only mask, not the optimizer's all-exterior mask.

        The optimizer mask (``_e_fixed_bool``) freezes all exterior nodes so
        they receive no gradient.  But for Sussman reinit, exterior nodes must
        participate so that |nabla phi|=1 propagates across the full grid.
        We temporarily swap in ``_reinit_fixed_bool`` (BC-adjacent only).

        When ``remeshing=True`` we run an entirely different path: project
        the octree SDF onto the background cube grid, Sussman it there,
        rebuild the outer octree from the cleaned grid, and swap it in.
        """
        if self._remeshing_enabled:
            self._post_optimizer_step_remesh(iteration)
            return

        saved = self._e_fixed_bool
        self._e_fixed_bool = self._reinit_fixed_bool
        try:
            super()._post_optimizer_step(iteration)
        finally:
            self._e_fixed_bool = saved

    # ------------------------------------------------------------------
    # Adaptive octree remeshing path (only used when remeshing=True)
    # ------------------------------------------------------------------

    @staticmethod
    def _axis_scale_from_verts_and_extents(
        verts_norm, physical_extents, snap_to_grid=None
    ):
        """Compute (axis_scale, geo_min) from MC surface verts bbox + physical_extents.

        Parameters
        ----------
        verts_norm : torch.Tensor
            ``(V, 3)`` surface vertices in normalized (cube) space.
        physical_extents : tuple
            Either ``(value, axis_name)`` for isotropic proportional scaling
            (e.g. ``(4.0, "x")`` → object spans 4 m along x, other axes scaled
            proportionally), or a 3-tuple ``(lx, ly, lz)`` for per-axis lengths.
        snap_to_grid : float or None
            If given, snap geo_min and the effective geo_max to the nearest
            multiple of this spacing.  Set to ``2.0 / 2**octree_depth`` to
            ensure that boundary lattice nodes (at exact grid positions) map
            to exact physical coordinates — critical for BC predicate matching
            (e.g. ``torch.isclose(x, 4.0)``).

        Returns
        -------
        axis_scale : list[float]
            Per-axis scale: ``physical = (normalized - geo_min) * axis_scale``.
        geo_min : list[float]
            Per-axis offset (min of surface bbox in normalized space).
        """
        v = verts_norm.to(dtype=torch.float64)
        geo_min_t = v.amin(dim=0)
        geo_max_t = v.amax(dim=0)
        geo_extent = geo_max_t - geo_min_t

        for i, label in enumerate("xyz"):
            if float(geo_extent[i]) < 1e-6:
                raise RuntimeError(
                    f"Degenerate geometry: {label}-extent={float(geo_extent[i]):.6e}"
                )

        is_proportional = len(physical_extents) == 2 and isinstance(
            physical_extents[1], str
        )

        if snap_to_grid is not None:
            h = float(snap_to_grid)
            # Snap geo_min DOWN to nearest lattice point (round to nearest).
            geo_min_snapped = [round(float(geo_min_t[i]) / h) * h for i in range(3)]
            if is_proportional:
                value, axis_name = physical_extents
                axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name.lower()]
                geo_max_ref_snapped = round(float(geo_max_t[axis_idx]) / h) * h
                geo_extent_ref = geo_max_ref_snapped - geo_min_snapped[axis_idx]
                scale = float(value) / geo_extent_ref
                axis_scale = [scale, scale, scale]
            else:
                axis_scale = []
                for i in range(3):
                    geo_max_i_snapped = round(float(geo_max_t[i]) / h) * h
                    geo_extent_i = geo_max_i_snapped - geo_min_snapped[i]
                    axis_scale.append(float(physical_extents[i]) / geo_extent_i)
            geo_min = geo_min_snapped
        else:
            if is_proportional:
                value, axis_name = physical_extents
                axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name.lower()]
                scale = float(value) / float(geo_extent[axis_idx])
                axis_scale = [scale, scale, scale]
            else:
                axis_scale = [
                    float(physical_extents[i]) / float(geo_extent[i]) for i in range(3)
                ]
            geo_min = [float(geo_min_t[i]) for i in range(3)]
        return axis_scale, geo_min

    def _init_remeshing_state(self, override_params):
        """Set up remeshing state and build the initial clean octree.

        Runs **before** ``super().__init__``.  Caller must supply exactly one
        of two geometry sources in ``override_params``:

        **Path A — explicit surface mesh** (``remeshing_trimesh`` key):

            ``mesh``         — ``trimesh.Trimesh`` **or** ``warp.Mesh`` in
                               physical coordinates.
                               *trimesh*: converted to a warp BVH internally
                               via ``build_warp_mesh`` + ``compute_nodal_sdf``.
                               *warp.Mesh*: used directly; sign convention
                               auto-detected from the bounding-box centroid.
            ``mesh_center``  — length-3 physical centre for the cube embedding
            ``h_target``     — physical leaf spacing (baseline element size)
            ``clip_bbox``    — optional ``(lo, hi)`` pair in physical coords

            The trimesh BVH is queried via ``hierarchical_extraction`` (dense at
            ``init_octree_depth``, band-only up to ``octree_depth``) to produce
            the raw SDF grid.  The resulting ``grid_sdf`` and ``anchors`` are
            stashed in the return dict of ``build_initial_octree_from_trimesh``
            for reuse in the octree-build step below.  Deduplicated final-parent
            must-query centres additionally force every finest cell containing
            a supplied surface vertex to ``octree_depth``; the extractor builds
            their coarser ancestry.  Optimization remeshes apply the same rule
            to each explicit MC surface already produced by the configured
            remeshing/reinitialization path.

        **Path B — implicit geometry function** (``remeshing_geo_func`` key):

            ``geo_func``     — callable ``(1, N, 3) float16 → (1, N, 1) float``
                               Takes cube-normalised query pts in ``[-1,1]^3``
                               and returns signed distance in **physical units**
                               (negative inside).  Matches the convention of
                               ``_mc_geo_func()``, so an existing solver's
                               ``_mc_geo_func()`` can be passed directly.
            ``axis_scale``   — length-3 list; per-axis cube→physical scale
                               (``axis_scale[i] = L/2`` for isotropic cube).
                               Copy from the source solver's ``_remesh_axis_scale``.
            ``geo_min``      — length-3 list; per-axis cube origin offset.
                               Copy from the source solver's ``_remesh_geo_min``.
            ``h_target``     — physical leaf spacing (sets ``grid_spacing``).
            ``clip_bbox``    — optional ``(lo, hi)`` pair in physical coords.

            When ``axis_scale`` is absent the sub-branch auto-computes it from
            the MC surface bbox + ``physical_extents``.  An optional
            ``filter_sdf_func`` (e.g. ``filter_largest_interior_component``) is
            applied to the raw grid before MC surface extraction.

            ``geo_func`` is evaluated via the same ``hierarchical_extraction``
            call (dense at ``init_octree_depth``, band-only up to
            ``octree_depth``); ``remeshing=True`` is always required for Path B
            because there is no pre-built mesh.

        **Pipeline** (both paths, always MC+MC regardless of
        ``reinit_sdf_method``):

          1. Run ``hierarchical_extraction`` on the geometry source → raw
             ``grid_sdf`` (float16) + cell-refinement ``anchors``.
          2. MC surface extraction + BVH query at all bg-grid pts → clean
             signed-distance SDF with ``|∇φ| ≈ 1``.
          3. Call ``sdf2octree(anchors, grid_sdf)`` to build the initial octree
             structure, reusing the anchors from step 1 — no second
             ``hierarchical_extraction`` pass.
          4. Override nodal SDF with trilinear interp of the clean grid.
          5. Scale nodes to physical space; optionally clip to ``clip_bbox``.
          6. Set ``override_params["octree_mesh"]`` for parent ``__init__``.

        After ``super().__init__`` the caller is responsible for calling
        ``_init_bg_frozen_mask()`` and (for graft/zip) ``_init_bc_graft()``
        so those snapshots come from the clean SDF.
        """
        from .remeshing import (
            build_initial_octree_from_trimesh,
            clip_octree_to_bbox,
            make_trilinear_geo_func,
            resolve_explicit_surface_sdf_truncation,
            scale_octree_nodes_to_physical,
        )

        has_trimesh = "remeshing_trimesh" in override_params
        has_geo_func = "remeshing_geo_func" in override_params
        # Only an explicitly supplied surface opts into persistent
        # vertex-driven refinement.  Geo-function/latent bootstraps retain
        # their existing band-driven behavior.
        self._refine_explicit_surfaces = has_trimesh
        if not has_trimesh and not has_geo_func:
            raise RuntimeError(
                "ShapeOptTrimmed remeshing=True requires either "
                "override_params['remeshing_trimesh'] (path A: explicit surface mesh) "
                "or override_params['remeshing_geo_func'] (path B: implicit geo_func)."
            )
        if has_trimesh and has_geo_func:
            raise RuntimeError(
                "Provide exactly one of 'remeshing_trimesh' or 'remeshing_geo_func', not both."
            )

        device = str(override_params.get("device", "cuda"))

        # Provide a temporary logger (parent __init__ will overwrite with proper one).
        if not hasattr(self, "log"):
            self.log = print

        # ------------------------------------------------------------------
        # Path A: build raw bg-grid SDF from trimesh BVH
        # ------------------------------------------------------------------
        if has_trimesh:
            trimesh_cfg = override_params.pop("remeshing_trimesh")
            # The explicit surface is also the one BC selector regions were
            # approved on.  NitscheNeoHookean measures each Neumann selector's
            # loaded area there rather than on its own MC surface (see
            # _neumann_region_area).  Copied into the inner solver's overrides.
            # ``mesh`` is a trimesh.Trimesh or a warp.Mesh (points / indices).
            source = trimesh_cfg["mesh"]
            if isinstance(source, wp.Mesh):
                ref_vertices = source.points.numpy()
                ref_faces = source.indices.numpy().reshape(-1, 3)
            else:
                ref_vertices, ref_faces = source.vertices, source.faces
            override_params.setdefault("bc_reference_surface", {
                "vertices": np.asarray(ref_vertices, dtype=np.float32),
                "faces": np.asarray(ref_faces, dtype=np.int64),
            })

            # Explicit-surface SDF values are physical distances. Default the
            # extraction width to one finest leaf; explicit overrides are also
            # interpreted as physical lengths.
            self._sdf_truncation = resolve_explicit_surface_sdf_truncation(
                trimesh_cfg["h_target"], trimesh_cfg.get("sdf_truncation")
            )
            self._sdf_refinement_band = 0.9 * self._sdf_truncation
            self.log(
                "ShapeOptTrimmed: explicit-surface SDF truncation "
                f"{self._sdf_truncation:.6g}, refinement band "
                f"{self._sdf_refinement_band:.6g} "
                f"(h_target={float(trimesh_cfg['h_target']):.6g})."
            )

            # build_octree=False: skip sdf2octree — we build from clean SDF below.
            init = build_initial_octree_from_trimesh(
                mesh=trimesh_cfg["mesh"],
                mesh_center=trimesh_cfg["mesh_center"],
                h_target=trimesh_cfg["h_target"],
                octree_depth=self._remesh_octree_depth,
                init_octree_depth=self._remesh_init_octree_depth,
                device=device,
                clip_bbox=trimesh_cfg.get("clip_bbox"),
                build_octree=False,
                sdf_truncation=self._sdf_truncation,
            )

            grid_shape = tuple(init["grid_shape"])
            bg_n = grid_shape[0]
            axis_scale = list(init["axis_scale"])
            geo_min = list(init["geo_min"])
            grid_spacing = tuple(init["grid_spacing"])
            raw_sdf = init["bg_grid_sdf"].to(device=device, dtype=torch.float32)
            _init_grid_h16 = init["init_grid_h16"]
            _init_anchors = init["init_anchors"]
            clip_bbox_raw = init.get("clip_bbox", None)
            init_label = "trimesh"

        # ------------------------------------------------------------------
        # Path B: hierarchically extract bg-grid SDF → raw physical SDF
        # ------------------------------------------------------------------
        else:
            try:
                from src.triposg_wrapper.hierarchical_extraction import (
                    hierarchical_extraction as _hierarchical_extraction,
                )
            except ImportError:
                from triposg_wrapper.hierarchical_extraction import (
                    hierarchical_extraction as _hierarchical_extraction,
                )

            geo_func_cfg = override_params.pop("remeshing_geo_func")
            geo_func_init = geo_func_cfg["geo_func"]
            clip_bbox_raw = geo_func_cfg.get("clip_bbox", None)
            # SDF truncation bound of the geo_func's decoder (rides in the cfg
            # dict alongside the geo_func); neutral 0.01 default if unspecified.
            if "sdf_truncation" not in geo_func_cfg:
                self.log(
                    "ShapeOptTrimmed: remeshing_geo_func did not specify 'sdf_truncation'; "
                    "defaulting to 0.01 (neutral band width)."
                )
            self._sdf_truncation = geo_func_cfg.get("sdf_truncation", 0.01)
            self._sdf_refinement_band = 0.9 * self._sdf_truncation

            bg_n = 2**self._remesh_octree_depth + 1
            grid_shape = (bg_n, bg_n, bg_n)

            # Hierarchically query geo_func: dense at init_octree_depth, then
            # band-only refinement up to octree_depth.  Returns a fully-populated
            # (bg_n, bg_n, bg_n) grid (non-band cells are trilinearly upsampled
            # from the coarse level) plus the cell-refinement anchor tensor.
            with torch.no_grad():
                _init_grid_h16, _init_anchors = _hierarchical_extraction(
                    geometric_func=geo_func_init,
                    bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
                    init_octree_depth=self._remesh_init_octree_depth,
                    octree_depth=self._remesh_octree_depth,
                    device=device,
                    sdf_truncation=self._sdf_truncation,
                )
            raw_sdf_eval = _init_grid_h16.float()  # (bg_n, bg_n, bg_n)

            if "axis_scale" in geo_func_cfg:
                # Sub-branch A: axis_scale/geo_min/h_target provided explicitly.
                # geo_func already returns physical-unit SDF.
                axis_scale = list(geo_func_cfg["axis_scale"])
                geo_min = list(geo_func_cfg["geo_min"])
                h_target = float(geo_func_cfg["h_target"])
                raw_sdf = raw_sdf_eval
            else:
                # Sub-branch B: auto-compute axis_scale from MC surface + physical_extents.
                # geo_func returns normalized-space SDF; scale to physical after.
                physical_extents = geo_func_cfg["physical_extents"]
                # Optional post-eval filter (e.g. filter_largest_interior_component).
                # Removes disconnected interior blobs that corrupt the surface bbox.
                filter_func = geo_func_cfg.get("filter_sdf_func", None)
                if filter_func is not None:
                    raw_sdf_eval = filter_func(raw_sdf_eval)
                from .analytical_sdf import extract_surface

                verts_norm_mc, _faces_mc, _probe_mc = extract_surface(
                    raw_sdf_eval, domain_length=2.0, device=device
                )
                # Stash so _mc_reinit_bg_grid can reuse this surface instead
                # of re-running MC on the same (scaled) data.
                self._mc_preseed = (verts_norm_mc, _faces_mc, _probe_mc)
                # Snap geo_min/geo_max to the octree lattice so that boundary
                # nodes land at exact physical coordinates for BC matching.
                h_norm = 2.0 / (2**self._remesh_octree_depth)
                axis_scale, geo_min = self._axis_scale_from_verts_and_extents(
                    verts_norm_mc, physical_extents, snap_to_grid=h_norm
                )
                h_target = h_norm * float(axis_scale[0])
                raw_sdf = raw_sdf_eval * float(axis_scale[0])
                print(
                    f"  Remesh init (geo_func, physical_extents={physical_extents}): "
                    f"axis_scale={[f'{s:.4f}' for s in axis_scale]}, "
                    f"geo_min={[f'{g:.4f}' for g in geo_min]}, "
                    f"h_target={h_target:.6f}"
                )

            grid_spacing = (h_target, h_target, h_target)
            init_label = "geo_func"

        # ------------------------------------------------------------------
        # Validation common to both paths
        # ------------------------------------------------------------------
        if grid_shape[0] != grid_shape[1] or grid_shape[1] != grid_shape[2]:
            raise RuntimeError(
                f"Remeshing bg grid must be cubic; got grid_shape={grid_shape}"
            )
        if bg_n != 2**self._remesh_octree_depth + 1:
            raise RuntimeError(
                f"Remeshing bg grid size {bg_n} != 2^{self._remesh_octree_depth}+1"
            )

        # ------------------------------------------------------------------
        # Common state setup
        # ------------------------------------------------------------------
        self._remesh_axis_scale = axis_scale
        self._remesh_geo_min = geo_min
        self._remesh_axis_scale_t = torch.tensor(
            axis_scale, dtype=torch.float64, device=device
        )
        self._remesh_geo_min_t = torch.tensor(
            geo_min, dtype=torch.float64, device=device
        )
        self._remesh_sdf_scale = float(axis_scale[0])
        self._remesh_device = device
        self._sussman_grid_shape = grid_shape
        self._sussman_grid_spacing = grid_spacing

        self._remesh_clip_bbox = (
            (
                tuple(float(x) for x in clip_bbox_raw[0]),
                tuple(float(x) for x in clip_bbox_raw[1]),
            )
            if clip_bbox_raw is not None
            else None
        )

        # ------------------------------------------------------------------
        # Step 2. MC reinit: extract surface → BVH → clean SDF at every pt.
        #    At this point _reinit_fixed_bool does not exist yet (parent not
        #    inited), so _build_mc_surface_and_bvh skips the graft/zip pin.
        # ------------------------------------------------------------------
        self._sussman_grid_sdf = raw_sdf  # needed by _mc_reinit_bg_grid for sentinels
        print(
            f"  Remesh init ({init_label}): depth={self._remesh_octree_depth}, "
            f"axis_scale={[f'{s:.4f}' for s in axis_scale]}, "
            f"bg grid {grid_shape} — running MC reinit..."
        )
        clean_sdf = self._mc_reinit_bg_grid(raw_sdf)
        self._sussman_grid_sdf = clean_sdf

        # Path B only: the decoder output has gradient ~0.1/cell while the
        # MC-reinited SDF has gradient ~1/axis_scale per cell (~0.4/cell).
        # Per-iter remesh always feeds clean_sdf back through MC (grid_norm =
        # clean_sdf/axis_scale), so sub-cell zero-crossing interpolation uses
        # the proper-SDF gradient.  Run a second pass here so the initial
        # octree is built from the same field type, making iter-0 and iter-1
        # geometry almost consistent.
        if not has_trimesh and "axis_scale" not in geo_func_cfg:
            clean_sdf = self._mc_reinit_bg_grid(clean_sdf)
            self._sussman_grid_sdf = clean_sdf

        # ------------------------------------------------------------------
        # Step 3. Build initial mesh from the CLEAN SDF.
        #    When init_depth == final_depth we skip the octree machinery
        #    entirely: build a uniform hex mesh from bg-grid lattice points
        #    (aligned → exact per-iter SDF lookup, no remeshing ever needed).
        #    Otherwise use the adaptive octree path.
        # ------------------------------------------------------------------
        if self._remesh_init_octree_depth == self._remesh_octree_depth:
            from .remeshing import build_bg_aligned_hex_mesh

            octree_phys = build_bg_aligned_hex_mesh(
                clean_sdf,
                self._remesh_axis_scale,
                self._remesh_geo_min,
                clip_bbox=self._remesh_clip_bbox,
            )
        else:
            # Both Path A and Path B: reuse anchors from the initial
            # hierarchical extraction — no second extraction needed.
            try:
                from src.triposg_wrapper.utils import sdf2octree as _sdf2octree
            except ImportError:
                from triposg_wrapper.utils import sdf2octree as _sdf2octree

            trilinear_func = make_trilinear_geo_func(clean_sdf)
            nodes, _, elements, edge_h, face_h = _sdf2octree(
                cell_tensor=_init_anchors,
                sdf_tensor=_init_grid_h16,
                domain_length=2.0,
            )
            nodes_f16 = nodes.to(torch.float16).unsqueeze(0)
            octree_norm = {
                "nodes": nodes,
                "elements": elements,
                "sdf": trilinear_func(nodes_f16).squeeze(0).squeeze(-1),
                "grid_sdf": _init_grid_h16,
                "existing_anchors": _init_anchors,
            }
            if edge_h is not None and len(edge_h) > 0:
                octree_norm["edge_hanging"] = edge_h
            if face_h is not None and len(face_h) > 0:
                octree_norm["face_hanging"] = face_h

            octree_phys = scale_octree_nodes_to_physical(
                octree_norm, self._remesh_axis_scale_t, self._remesh_geo_min_t
            )
            if self._remesh_clip_bbox is not None:
                n_before = int(octree_phys["nodes"].shape[0])
                octree_phys = clip_octree_to_bbox(octree_phys, *self._remesh_clip_bbox)
                n_after = int(octree_phys["nodes"].shape[0])
                print(f"  Remesh init clip: {n_before} → {n_after} nodes")

        # Cache bg-index ranges for exact per-iter SDF lookup (uniform hex path).
        if "_bg_ix" in octree_phys:
            self._hex_bg_ix = octree_phys.pop("_bg_ix")
            self._hex_bg_iy = octree_phys.pop("_bg_iy")
            self._hex_bg_iz = octree_phys.pop("_bg_iz")

        # The MC BVH built by _mc_reinit_bg_grid above is left in place so
        # that callers can call _mc_geo_func() after __init__ (e.g. to seed
        # a path-B geo_func).  _post_optimizer_step_remesh_impl clears and
        # rebuilds the cache at the start of each optimization iteration, so
        # this initial BVH is never incorrectly reused during optimization.

        # Step 4. Hand off to parent __init__ via override_params.
        override_params["octree_mesh"] = octree_phys
        override_params["grid_shape"] = grid_shape
        override_params["grid_spacing"] = self._sussman_grid_spacing
        print(
            f"  Remesh init: clean octree "
            f"{octree_phys['nodes'].shape[0]} nodes, "
            f"{octree_phys['elements'].shape[0]} elements"
        )

    # ------------------------------------------------------------------
    # Coordinate-transform helpers (cube ↔ physical)
    # ------------------------------------------------------------------

    def _normalized_to_physical(self, pts_norm):
        """Cube-space ``[-1,1]^3`` → physical coordinates."""
        return (
            pts_norm.to(torch.float64) - self._remesh_geo_min_t
        ) * self._remesh_axis_scale_t

    def _physical_to_normalized(self, pts_phys):
        """Physical coordinates → cube-space ``[-1,1]^3``."""
        return (
            pts_phys.to(torch.float64) / self._remesh_axis_scale_t
            + self._remesh_geo_min_t
        )

    # ------------------------------------------------------------------
    # Marching-cubes BVH helpers
    # ------------------------------------------------------------------

    def _init_bc_graft(self):
        """Cache initial-surface state for BC fix methods 'graft' and 'zip'.

        Both methods share the initial MC surface extraction and frozen-node
        setup.  'graft' additionally builds a BVH for closest-point snapping;
        'zip' additionally computes the BC patch for zipper stitching.

        Called once (lazily) from step() on the first iteration.
        """
        from .analytical_sdf import extract_surface, build_warp_mesh

        # Build initial surface from the clean bg grid (pre-GD snapshot).
        init_norm = (self._sussman_grid_sdf / self._remesh_sdf_scale).reshape(
            self._sussman_grid_shape
        )
        init_verts_norm, init_faces, init_probe_norm = extract_surface(
            init_norm, domain_length=2.0, device=self._remesh_device
        )
        init_verts_phys = self._normalized_to_physical(init_verts_norm.double()).float()

        # Frozen BC node positions (physical) + proximity radius (shared by both).
        pts = self.vtu_writer.mesh_info["points"]
        node_positions = (
            pts.to(dtype=torch.float64, device=self.device)
            if isinstance(pts, torch.Tensor)
            else torch.tensor(
                __import__("numpy").asarray(pts),
                dtype=torch.float64,
                device=self.device,
            )
        )
        frozen_phys = node_positions[self._reinit_fixed_bool].to(torch.float64)
        radius_phys = 2.0 * float(self._sussman_grid_spacing[0])

        if self._sdf_bc_fix_method == "graft":
            init_probe_phys = (
                self._normalized_to_physical(init_probe_norm.double().unsqueeze(0))
                .squeeze(0)
                .float()
            )
            bbox_diag = float(
                torch.linalg.norm(init_verts_phys.amax(0) - init_verts_phys.amin(0))
            )
            wp_mesh, negate, keep_alive = build_warp_mesh(
                init_verts_phys,
                init_faces,
                init_probe_phys,
                bbox_diag,
                device=self._remesh_device,
            )
            self._bc_init_wp_mesh = wp_mesh
            self._bc_init_wp_negate = negate
            self._bc_init_wp_keep_alive = keep_alive
            self._bc_init_bbox_diag = bbox_diag
            self._bc_frozen_phys = frozen_phys
            self._bc_pin_radius_phys = radius_phys
            frozen_f = frozen_phys.float()
            self._bc_frozen_aabb_min = (frozen_f.min(0).values - radius_phys).cpu()
            self._bc_frozen_aabb_max = (frozen_f.max(0).values + radius_phys).cpu()
            self.log(
                f"  BC graft init: initial surface {int(init_verts_norm.shape[0])} verts, "
                f"radius={radius_phys:.4f}"
            )

        elif self._sdf_bc_fix_method == "zip":
            from .remeshing import compute_bc_patch

            patch = compute_bc_patch(
                init_verts_phys.double(),
                init_faces,
                frozen_phys,
                radius_phys,
                grow_rings=int(self.params.get("bc_zip_patch_grow", 2)),
            )
            if patch is None:
                raise RuntimeError(
                    f"BC zip init failed: no surface verts within "
                    f"{radius_phys:.4e} of any frozen BC node"
                )
            self._bc_zip_patch = patch
            self._bc_zip_frozen_phys = frozen_phys
            self.log(
                f"  BC zip init: {patch['n_verts']} patch verts, "
                f"radius={radius_phys:.4f}"
            )

    def _pin_bc_verts_to_initial_surface(self, verts_norm):
        """Move BC-region MC vertices to their nearest point on the initial surface.

        Preserves the mesh topology (same faces) — no hole-cutting, no patch
        insertion, no zipper stitching.  The BVH built from the pinned mesh
        drives the octree refinement near BC and the MC-reinit SDF, naturally
        giving SDF values close to the initial at BC bg-grid cells.

        Parameters
        ----------
        verts_norm : torch.Tensor  ``(V, 3)`` current MC vertices in [-1,1]^3.

        Returns
        -------
        torch.Tensor  ``(V, 3)`` with BC-region vertices snapped to initial surface.
        """
        from .analytical_sdf import compute_closest_points

        device = self._remesh_device
        verts_phys = self._normalized_to_physical(verts_norm.to(torch.float64)).float()

        # Fast AABB pre-filter → only cdist on the small BC-adjacent subset.
        aabb_min = self._bc_frozen_aabb_min.to(device)
        aabb_max = self._bc_frozen_aabb_max.to(device)
        candidates = ((verts_phys >= aabb_min) & (verts_phys <= aabb_max)).all(1)
        bc_mask = torch.zeros(verts_phys.shape[0], dtype=torch.bool, device=device)
        if candidates.any():
            cand_dists = torch.cdist(
                verts_phys[candidates],
                self._bc_frozen_phys.float().to(device),
            )
            bc_mask[candidates] = cand_dists.min(1).values < self._bc_pin_radius_phys

        if not bc_mask.any():
            return verts_norm

        bc_verts_phys = verts_phys[bc_mask]
        closest_phys = compute_closest_points(
            self._bc_init_wp_mesh,
            bc_verts_phys,
            max_dist=10.0 * self._bc_init_bbox_diag,
            device=device,
        )

        verts_phys_new = verts_phys.clone()
        verts_phys_new[bc_mask] = closest_phys
        verts_norm_new = self._physical_to_normalized(
            verts_phys_new.to(torch.float64)
        ).to(verts_norm.dtype)

        self.log(f"  BC graft: pinned {int(bc_mask.sum())} verts to initial surface")
        return verts_norm_new

    def _build_mc_surface_and_bvh(self, grid_sdf_phys, _precomputed=None):
        """MC-extract zero-level-set of bg grid, build physical-space BVH.

        Caches ``_mc_wp_mesh``, ``_mc_wp_negate``, ``_mc_wp_keep_alive``,
        ``_mc_verts_norm``, ``_mc_faces``, ``_mc_bbox_diag_phys`` on ``self``
        for reuse within the same iteration (reinit → remesh share one BVH).
        Returns a geo_func suitable for ``remesh_to_octree``.

        ``_precomputed``: optional ``(verts_norm, faces, probe_norm)`` from a
        prior ``extract_surface`` call on the same SDF.  When provided the MC
        extraction step is skipped; filter/close-loop/BVH steps still run.
        """
        from .analytical_sdf import extract_surface, build_warp_mesh

        if _precomputed is None:
            grid_norm = (grid_sdf_phys / self._remesh_sdf_scale).reshape(
                self._sussman_grid_shape
            )
            verts_norm, faces, probe_norm = extract_surface(
                grid_norm, domain_length=2.0, device=self._remesh_device
            )
        else:
            verts_norm, faces, probe_norm = _precomputed

        # Drop non-watertight non-largest CCs BEFORE ``_close_open_loops``.
        # MC can produce small spurious open-boundary components from
        # SDF noise (sub-bg-cell zero crossings, sharp scatter-clamped
        # transitions in PF mode, etc.).  ``_close_open_loops`` would
        # otherwise close those with centroid fans, making them appear
        # watertight; the winding-number reinit would then classify them
        # as ghost void/island regions (positive SDF inside) which the
        # inner-mesh trimming removes from the body — leaving empty
        # regions in the body interior.  Filtering FIRST drops the
        # genuine SDF noise; then ``_close_open_loops`` only closes
        # actual sliver edges on the body surface.  ``stitch_bc_patches``
        # downstream then receives an all-watertight mesh and preserves
        # legitimate non-largest CCs (e.g. PF-mode interior voids).
        from .remeshing import _close_open_loops, filter_non_watertight_components

        keep = filter_non_watertight_components(faces, keep_largest=True)
        n_drop = int(faces.shape[0] - keep.shape[0])
        if n_drop > 0:
            self.log(f"  MC filter: dropped {n_drop} non-watertight CC face(s)")
            faces = faces[keep]
        verts_norm, faces = _close_open_loops(verts_norm, faces)

        # Apply BC fix before building BVH.  Both methods are skipped during
        # _init_remeshing_state (before parent __init__ / _init_bc_graft).
        if self._sdf_bc_fix_method == "graft" and hasattr(self, "_bc_init_wp_mesh"):
            # Snap BC-region vertices to nearest point on initial surface.
            self.log("  BC graft: snapping BC-region vertices to initial surface")
            verts_norm = self._pin_bc_verts_to_initial_surface(verts_norm)
        elif self._sdf_bc_fix_method == "zip" and hasattr(self, "_bc_zip_patch"):
            # Neighborhood-expansion stitch: for each cached BC patch, select
            # the active base-mesh region by scaled-AABB, validate (expand if
            # needed), remove, and stitch via FKU zipper DP.
            from .remeshing import stitch_bc_patches

            bc_patches = self._bc_zip_patch.get("components", [])
            if bc_patches:
                self.log(
                    f"  BC zip: stitching {len(bc_patches)} cached patch(es) "
                    f"into MC surface"
                )
                verts_phys_d = self._normalized_to_physical(
                    verts_norm.to(torch.float64)
                )
                out_verts_phys, faces, _ = stitch_bc_patches(
                    base_verts_phys=verts_phys_d,
                    base_faces=faces,
                    bc_patches=bc_patches,
                    # Taubin (order 1) is the default: it is a membrane energy
                    # pinned on one ring, so the strip stays only G0 against the
                    # surfaces it joins, but it is cheap and unconditionally
                    # stable.  ``bc_zip_smooth_order=2`` swaps in thin-plate
                    # fairing (G1) for figures where the seam has to read clean.
                    smooth_order=int(self.params.get("bc_zip_smooth_order", 1)),
                    subdivision_size_ratio=float(
                        self.params.get("bc_zip_subdivision_ratio", 3.0)
                    ),
                    log=self.log,
                )
                verts_norm = self._physical_to_normalized(out_verts_phys).to(
                    verts_norm.dtype
                )

        # Build BVH in physical space (distances are physical units → physical SDF).
        verts_phys = self._normalized_to_physical(verts_norm.double()).float()
        probe_phys = (
            self._normalized_to_physical(probe_norm.double().unsqueeze(0))
            .squeeze(0)
            .float()
        )
        bbox_diag = float(torch.linalg.norm(verts_phys.amax(0) - verts_phys.amin(0)))
        wp_mesh, negate, keep_alive = build_warp_mesh(
            verts_phys, faces, probe_phys, bbox_diag, device=self._remesh_device
        )
        self._mc_wp_mesh = wp_mesh
        self._mc_wp_negate = negate
        self._mc_wp_keep_alive = keep_alive
        self._mc_verts_norm = verts_norm
        self._mc_faces = faces
        self._mc_bbox_diag_phys = bbox_diag
        return self._mc_geo_func()

    def _mc_reinit_bg_grid(self, grid_sdf_phys):
        """Replace Sussman: build BVH from MC surface, query analytical SDF at all bg-grid pts.

        Also caches the MC BVH (via ``_build_mc_surface_and_bvh``) so the
        subsequent remesh step can reuse it when ``remeshing_method='marchingcubes'``.
        """
        from .analytical_sdf import compute_nodal_sdf
        from .remeshing import cube_grid_points_normalized

        # Build BVH; consume preseed if available (avoids duplicate MC extraction
        # when sub-branch B already extracted the surface for axis_scale computation).
        _preseed = getattr(self, "_mc_preseed", None)
        if _preseed is not None:
            del self._mc_preseed
        self._build_mc_surface_and_bvh(grid_sdf_phys, _precomputed=_preseed)

        n = self._sussman_grid_shape[0]
        # Query at all bg-grid points in physical space.
        grid_pts_norm = cube_grid_points_normalized(
            n, device=self._remesh_device, dtype=torch.float64
        )
        grid_pts_phys = self._normalized_to_physical(grid_pts_norm).float()
        cleaned_phys = compute_nodal_sdf(
            self._mc_wp_mesh,
            grid_pts_phys,
            self._mc_wp_negate,
            max_dist=10.0 * self._mc_bbox_diag_phys,
            device=self._remesh_device,
        )
        # Sentinel handling: for points too far from the surface (|sdf| ≥ 9.9*max_dist),
        # keep the old sign from the bg grid but cap magnitude.
        sentinel = cleaned_phys.abs() >= 9.9 * self._mc_bbox_diag_phys
        if sentinel.any():
            old_sign = torch.sign(grid_sdf_phys.reshape(-1))
            old_sign[old_sign == 0] = 1.0
            cleaned_phys = torch.where(
                sentinel, old_sign * self._mc_bbox_diag_phys, cleaned_phys
            )
        n_sent = int(sentinel.sum().item())
        cleaned_grid = cleaned_phys.reshape(self._sussman_grid_shape)
        dx, dy, dz = self._sussman_grid_spacing
        grad_before = ShapeOpt._eikonal_grad_mag(
            grid_sdf_phys.reshape(self._sussman_grid_shape),
            dx,
            dy,
            dz,
        )
        grad_after = ShapeOpt._eikonal_grad_mag(cleaned_grid, dx, dy, dz)
        self.log(
            f"  SDF reinit (MC): {n}^3 bg grid, {n_sent} sentinels clamped, "
            f"|∇φ| mean {grad_before:.4f}→{grad_after:.4f}, "
            f"|∇φ|-1 L∞ {abs(grad_before-1):.4f}→{abs(grad_after-1):.4f}"
        )
        return cleaned_grid

    def _get_or_build_mc_surface_physical(self):
        """Return (verts_phys (V,3), faces (F,3)) for SurfaceSelector resolution.

        Reuses the MC surface cached by ``_init_remeshing_state`` at zero extra
        cost.  Falls back to building from ``_sussman_grid_sdf`` if the cache is
        absent, and finally to the base-class scatter path.
        """
        if getattr(self, "_mc_verts_norm", None) is not None:
            verts_phys = self._normalized_to_physical(
                self._mc_verts_norm.to(torch.float64)
            ).float()
            return verts_phys, self._mc_faces

        if hasattr(self, "_sussman_grid_sdf"):
            self.log("  BC selector: building MC surface from bg SDF grid")
            self._build_mc_surface_and_bvh(self._sussman_grid_sdf)
            verts_phys = self._normalized_to_physical(
                self._mc_verts_norm.to(torch.float64)
            ).float()
            return verts_phys, self._mc_faces

        return super()._get_or_build_mc_surface_physical()

    def _mc_geo_func(self):
        """Build a geo_func from the cached MC BVH (call _build_mc_surface_and_bvh first).

        geo_func maps cube-space query points → physical-unit SDF, matching
        the convention of ``make_trilinear_geo_func`` so both feed
        ``remesh_to_octree`` identically.
        """
        from .analytical_sdf import compute_nodal_sdf

        wp_mesh = self._mc_wp_mesh
        negate = self._mc_wp_negate
        bbox_diag = self._mc_bbox_diag_phys
        axis_scale_t = self._remesh_axis_scale_t
        geo_min_t = self._remesh_geo_min_t
        device = self._remesh_device

        def geo_func(pts_cube_f16):
            pts_phys = (
                (pts_cube_f16.squeeze(0).to(torch.float64) - geo_min_t) * axis_scale_t
            ).float()
            return (
                compute_nodal_sdf(
                    wp_mesh, pts_phys, negate, max_dist=10.0 * bbox_diag, device=device
                )
                .unsqueeze(0)
                .unsqueeze(-1)
            )

        return geo_func

    def _post_optimizer_step_remesh(self, iteration):
        """Remeshing-mode reinit: project → Sussman → rebuild outer octree.

        Runs entirely under ``torch.no_grad()``.  ``self.e.dof_values`` carries
        ``requires_grad=True`` during the optimize loop so the backward pass
        can compute gradients w.r.t. the design field; that flag propagates
        through ``wp.to_torch`` into every torch tensor we construct here.
        Without ``no_grad``, Sussman's ~130 iteration inner loop retains every
        intermediate tensor in the autograd graph, leaking ~11 GB per outer
        iteration on a 129³ background grid.
        """
        with torch.no_grad():
            self._post_optimizer_step_remesh_impl(iteration)

    def _post_scatter_bg_grid_hook(self, grid):
        """Hook called on the bg grid immediately after scatter, before BC freeze.

        Override in subclasses to filter the scattered bg grid (e.g. remove
        disconnected interior islands) before the reinit step.
        Default: identity.
        """
        return grid

    def _pf_update_hidden_phi(self):
        """Update the persistent hidden φ field (PF mode only).

        ``self._hidden_phi`` is a torch tensor on the outer mesh that lives
        OUTSIDE the tape.  It accumulates per-iter updates so the bg-grid
        scatter source ``atanh(_hidden_phi)`` evolves topology when interior
        nodes dip below 0.5.  Two updates per iter, elementwise nodal:

          (a) **Boundary band** (|SDF| < band·ε):
              ``_hidden_phi ← tanh(SDF/ε)`` — keep hidden φ consistent with
              the freshly GD-updated SDF on the interface band.
          (b) **Interior** (|SDF| ≥ band·ε):
              ``_hidden_phi -= lr_φ · g_phi`` where
              ``g_phi = ∂J/∂e_phi = self._last_e_phi_grad`` is the
              intermediate φ-gradient extracted from the primary tape
              (non-zero at deep interior, where ∂J/∂SDF saturates to 0).

        Interior dips accumulate across iterations because (a) does NOT
        touch the interior region.
        """
        eps = float(self._epsilon)
        field_torch_dtype = self._design_torch_dtype()
        field_wp_dtype = self._design_wp_dtype()
        sdf_now = wp.to_torch(self.e.dof_values).detach().to(field_torch_dtype)
        n_outer = sdf_now.shape[0]

        # Lazy init: hidden φ starts as tanh(SDF/ε) at iter 0.
        if (
            getattr(self, "_hidden_phi", None) is None
            or self._hidden_phi.shape[0] != n_outer
        ):
            self._hidden_phi = _sdf_to_phi(sdf_now, eps).clone()
            return

        band_k = float(self.params.get("phi_reproject_band_k", 3.0))
        boundary_mask = sdf_now.abs() < band_k * eps

        # (a) Boundary band tracks the freshly updated SDF.
        phi_from_sdf = _sdf_to_phi(sdf_now, eps)
        phi_new = torch.where(boundary_mask, phi_from_sdf, self._hidden_phi)

        # (b) Interior receives the intermediate φ-gradient step.  The
        # gradient ``_last_e_phi_grad`` is already prolongated (``C·g``) by
        # ``_post_tape_backward``. A direct GD step is followed by another
        # prolongation below so slave values remain exactly master-implied.
        g = getattr(self, "_last_e_phi_grad", None)
        if g is not None and g.shape[0] == n_outer:
            lr_phi = self.params.get("pf_phi_lr", None)
            if lr_phi is None:
                lr_phi = float(self.params.get("gd_lr", 1e-3))
            iteration = getattr(self, "_pf_update_iteration", None)
            if iteration is not None and iteration < int(
                self.params.get("pf_phi_start_iter", 0)
            ):
                lr_phi = 0.0
            interior = (~boundary_mask).to(field_torch_dtype)
            phi_new = phi_new - lr_phi * g.to(field_torch_dtype) * interior

        phi_new = phi_new.clamp(0.0, 1.0).contiguous()

        # (c) Re-prolongate masters → slaves so slave ``_hidden_phi``
        # exactly equals the constraint-implied combination of master
        # values. The direct update above touches the vectorized nodal field;
        # this final prolongation restores exact hanging-node consistency.
        # (Thin-feature filter is applied separately in
        # ``_post_optimizer_step_remesh_impl`` so its output can be
        # rolled back when filtering would eliminate every viable void
        # — letting the dips persist and accumulate further.)
        if self._has_affine_constraints:
            phi_wp = wp.from_torch(phi_new, dtype=field_wp_dtype)
            self._prolongate_vector(phi_wp)
            phi_new = wp.to_torch(phi_wp).clone()

        self._hidden_phi = phi_new

    def _pf_filter_thin_features(self, phi):
        """Force ``hidden_phi`` of dipping nodes whose void is sub-2ε
        thin (in any axis direction) back to 0.51.

        For each node, compute the below-φ-0.5 segment length along the
        line ``minus_neighbor → self → plus_neighbor`` for x, y, z
        independently, using the immediate axis-aligned hex-edge
        neighbors.  If the SHORTEST below-length across the 3 axes is
        less than ``2·epsilon``, the node is considered sub-resolution
        and its φ is clamped to 0.51 (just above 0.5, so it stops
        triggering the dipping checks downstream).

        Vectorisation: axis-neighbour discovery uses the 12 axis-aligned
        edges of every hex element via VTK ordering; per axis we
        scatter-min to pick the SHORTEST edge per node (so depth-mixed
        meshes use the finer, more conservative neighbour).  The
        below-segment length is a closed-form linear-interp expression
        evaluated tensor-wise on (V,) tensors.
        """
        if phi is None or phi.numel() == 0:
            return phi
        nodes = self._prev_octree_nodes
        elements = self._prev_octree_elements
        if nodes is None or elements is None or elements.numel() == 0:
            return phi
        device = phi.device
        eps = float(self._epsilon)
        threshold = 2.0 * eps
        V = phi.shape[0]
        elements_l = elements.long().to(device=device)
        nodes_d = nodes.to(device=device, dtype=torch.float64)

        # VTK hex axis-aligned edge pairs (low-axis idx → high-axis idx).
        edge_pair_table = {
            0: torch.tensor(
                [[0, 1], [3, 2], [4, 5], [7, 6]], dtype=torch.long, device=device
            ),  # x
            1: torch.tensor(
                [[0, 3], [1, 2], [4, 7], [5, 6]], dtype=torch.long, device=device
            ),  # y
            2: torch.tensor(
                [[0, 4], [1, 5], [2, 6], [3, 7]], dtype=torch.long, device=device
            ),  # z
        }

        def _shortest_axis_neighbours(axis):
            pairs = edge_pair_table[axis]  # (4, 2)
            edges = elements_l[:, pairs]  # (E, 4, 2)
            low = edges[..., 0].reshape(-1)
            high = edges[..., 1].reshape(-1)
            edge_len = (nodes_d[high, axis] - nodes_d[low, axis]).abs()  # (E*4,) >0

            # Scatter-min floating edge lengths on GPU. A second scatter-min
            # selects the smallest node id only among equal-length winners,
            # providing a deterministic tie-break without unit-dependent
            # integer composite keys.
            big = float("inf")

            def _scatter_shortest(group, neighbor):
                min_len = torch.full(
                    (V,), big, dtype=torch.float64, device=device
                )
                min_len.scatter_reduce_(
                    0, group, edge_len, reduce="amin", include_self=True
                )
                sentinel = torch.full_like(neighbor, V)
                candidates = torch.where(
                    edge_len == min_len[group], neighbor, sentinel
                )
                selected = torch.full(
                    (V,), V, dtype=torch.long, device=device
                )
                selected.scatter_reduce_(
                    0, group, candidates, reduce="amin", include_self=True
                )
                selected = torch.where(
                    selected == V, torch.full_like(selected, -1), selected
                )
                return selected, min_len

            # +axis neighbour: per-low, smallest edge.
            plus_n, plus_h = _scatter_shortest(low, high)

            # -axis neighbour: per-high, smallest edge.
            minus_n, minus_h = _scatter_shortest(high, low)
            return minus_n, plus_n, minus_h, plus_h

        def _below_segment(va, vb, L):
            """Tensor-wise length of the φ<0.5 portion of the linear
            segment from value ``va`` (x=0) to ``vb`` (x=L)."""
            a_below = (va < 0.5).to(torch.float64)
            b_below = (vb < 0.5).to(torch.float64)
            both_below = a_below * b_below
            both_above = (1.0 - a_below) * (1.0 - b_below)
            crossing_mask = 1.0 - both_below - both_above
            # Crossing position fraction t ∈ (0, 1): φ(t) = va + (vb-va)·t = 0.5
            denom = vb - va
            denom_safe = torch.where(
                denom.abs() < 1e-12, torch.full_like(denom, 1e-12), denom
            )
            t = (0.5 - va) / denom_safe
            # If va is the below side, below portion is [0, t]; else [t, 1].
            below_a = t * L
            below_b = (1.0 - t) * L
            crossing_below = a_below * below_a + (1.0 - a_below) * below_b
            return both_below * L + crossing_mask * crossing_below

        def _axis_below_length(axis):
            n_minus, n_plus, h_minus, h_plus = _shortest_axis_neighbours(axis)
            valid = (n_minus >= 0) & (n_plus >= 0)
            n_m_safe = n_minus.clamp_min(0)
            n_p_safe = n_plus.clamp_min(0)
            v_minus = phi[n_m_safe]
            v_plus = phi[n_p_safe]
            below = _below_segment(v_minus, phi, h_minus) + _below_segment(
                phi, v_plus, h_plus
            )
            # Missing-neighbour nodes don't contribute a thinness signal.
            return torch.where(valid, below, torch.full_like(below, float("inf")))

        # Restrict to INTERIOR dips: nodes deep inside the body whose
        # ``hidden_phi`` has been pushed below 0.5 by the optimizer
        # gradient.  Exterior nodes have ``hidden_phi ≈ 0`` by
        # construction (= ``_sdf_to_phi(positive SDF)``); they're not
        # "dips" we want to filter, and including them would make the
        # mask cover tens of thousands of nodes (the entire bbox-padded
        # outer-octree exterior).  Body-boundary nodes (``|sdf| <
        # band·ε``) also shouldn't be filtered — they belong to the
        # phase-field transition band tracked by step (a).
        band_k = float(self.params.get("phi_reproject_band_k", 3.0))
        sdf_now = wp.to_torch(self.e.dof_values).detach().to(self._design_torch_dtype())
        interior_solid = sdf_now < -band_k * eps  # deep solid mask

        bx = _axis_below_length(0)
        by = _axis_below_length(1)
        bz = _axis_below_length(2)
        min_below = torch.minimum(torch.minimum(bx, by), bz)

        is_interior_dip = (phi < 0.5) & interior_solid
        is_thin = is_interior_dip & (min_below < threshold)
        n_thin = int(is_thin.sum())
        if n_thin == 0:
            return phi
        n_dip = int(is_interior_dip.sum())
        self.log(
            f"  PF thin filter: {n_thin}/{n_dip} interior dipping nodes forced to 0.51 "
            f"(below-φ-0.5 axis-min < 2ε={threshold:.4f})"
        )
        out = phi.clone()
        out[is_thin] = 0.51
        return out

    def _carry_over_hidden_phi_parent_interp(self, stash):
        """Re-populate ``self._hidden_phi`` on the NEW outer octree from a
        stash of the OLD octree (nodes, elements, hidden_phi).

        • Nodes matched to an old octree node by bg-lattice index copy the
          old ``_hidden_phi`` verbatim.
        • Unmatched new nodes (refined positions) get trilinear interp
          from the 8 corner hidden_phi values of the OLD element that
          contains them, evaluated at the new node's local coord.
        """
        from .remeshing import octree_node_bg_indices

        device = str(self.device)
        bg_n = int(self._sussman_grid_shape[0])

        old_nodes = stash["nodes"].to(device=device, dtype=torch.float64)
        old_elements = stash["elements"].to(device=device, dtype=torch.long)
        field_torch_dtype = self._design_torch_dtype()
        field_wp_dtype = self._design_wp_dtype()
        old_phi = stash["hidden_phi"].to(device=device, dtype=field_torch_dtype)
        new_nodes = self._prev_octree_nodes.to(device=device, dtype=torch.float64)

        axis_scale = self._remesh_axis_scale
        geo_min = self._remesh_geo_min

        # 1. Match new nodes to old nodes by flat bg-lattice key.
        old_idx = octree_node_bg_indices(
            old_nodes, bg_n, axis_scale, geo_min, device=device
        )
        new_idx = octree_node_bg_indices(
            new_nodes, bg_n, axis_scale, geo_min, device=device
        )
        old_key = old_idx[:, 0] * bg_n * bg_n + old_idx[:, 1] * bg_n + old_idx[:, 2]
        new_key = new_idx[:, 0] * bg_n * bg_n + new_idx[:, 1] * bg_n + new_idx[:, 2]

        lookup = torch.full((bg_n**3,), -1, dtype=torch.long, device=device)
        lookup[old_key] = torch.arange(
            old_key.shape[0], device=device, dtype=torch.long
        )
        matched_old = lookup[new_key]  # (N_new,), -1 if unmatched
        matched_mask = matched_old >= 0

        # Gather matched values; unmatched slots will be overwritten in step 3.
        new_phi = old_phi[matched_old.clamp_min(0)].clone()
        unmatched_mask = ~matched_mask
        if not bool(unmatched_mask.any()):
            self._hidden_phi = new_phi.clamp(0.0, 1.0).contiguous()
            return

        # 2. Build a bg-cell → old-element owner lookup in a single
        # batched scatter (no Python loop over element sizes).  Each old
        # element is padded to ``max_n`` along each axis; oversized
        # entries are filtered by a validity mask.  ``max_n`` is small
        # (≤ 2^(octree_depth - init_depth) + 1, e.g. 9 for depth-7/4).
        axis_scale_t = torch.as_tensor(axis_scale, device=device, dtype=torch.float64)
        geo_min_t = torch.as_tensor(geo_min, device=device, dtype=torch.float64)
        elem_node_pos = old_nodes[old_elements]  # (E, 8, 3)
        spacing = 2.0 / (bg_n - 1)
        lo_norm = elem_node_pos.amin(dim=1) / axis_scale_t + geo_min_t
        hi_norm = elem_node_pos.amax(dim=1) / axis_scale_t + geo_min_t
        i_lo = ((lo_norm + 1.0) / spacing).round().long().clamp(0, bg_n - 1)  # (E, 3)
        i_hi = ((hi_norm + 1.0) / spacing).round().long().clamp(0, bg_n - 1)
        n_per = (i_hi - i_lo + 1).clamp_min(1)  # (E, 3)
        E = i_lo.shape[0]
        max_n = int(n_per.max().item())

        off = torch.arange(max_n, device=device)
        offx = off.view(1, -1, 1, 1)
        offy = off.view(1, 1, -1, 1)
        offz = off.view(1, 1, 1, -1)
        valid3 = (
            (offx < n_per[:, 0:1].view(E, 1, 1, 1))
            & (offy < n_per[:, 1:2].view(E, 1, 1, 1))
            & (offz < n_per[:, 2:3].view(E, 1, 1, 1))
        )  # (E, max_n, max_n, max_n)
        gx_all = (i_lo[:, 0].view(E, 1, 1, 1) + offx).expand(E, max_n, max_n, max_n)
        gy_all = (i_lo[:, 1].view(E, 1, 1, 1) + offy).expand(E, max_n, max_n, max_n)
        gz_all = (i_lo[:, 2].view(E, 1, 1, 1) + offz).expand(E, max_n, max_n, max_n)
        e_expand = (
            torch.arange(E, device=device)
            .view(E, 1, 1, 1)
            .expand(E, max_n, max_n, max_n)
        )
        owner = torch.full((bg_n, bg_n, bg_n), -1, dtype=torch.long, device=device)
        owner[gx_all[valid3], gy_all[valid3], gz_all[valid3]] = e_expand[valid3]

        # 3. Lookup parent element per unmatched new node and trilinear
        # interp its 8 corner phi values at the node's local coord.
        u_sel = unmatched_mask.nonzero(as_tuple=True)[0]
        u_idx = new_idx[u_sel]
        e_idx = owner[u_idx[:, 0], u_idx[:, 1], u_idx[:, 2]]
        valid_u = e_idx >= 0
        if not bool(valid_u.any()):
            self._hidden_phi = new_phi.clamp(0.0, 1.0).contiguous()
            return
        u_sel_v = u_sel[valid_u]
        e_nodes = old_elements[e_idx[valid_u]]  # (U', 8) in VTK hex ordering
        e_corner_pos = old_nodes[e_nodes]  # (U', 8, 3)
        e_lo = e_corner_pos.amin(dim=1)
        e_hi = e_corner_pos.amax(dim=1)
        local = ((new_nodes[u_sel_v] - e_lo) / (e_hi - e_lo).clamp_min(1e-12)).clamp(
            0.0, 1.0
        )
        # Trilinear weights for VTK hex ordering (x,y,z flags → node idx).
        fx = local[:, 0:1]
        fy = local[:, 1:2]
        fz = local[:, 2:3]
        wx = torch.cat([1.0 - fx, fx], dim=1)  # (U', 2)  idx: [0]=1-fx, [1]=fx
        wy = torch.cat([1.0 - fy, fy], dim=1)
        wz = torch.cat([1.0 - fz, fz], dim=1)
        # Node i (VTK): (xf, yf, zf) ∈ {0,1}^3. Mapping is:
        #   0:(0,0,0) 1:(1,0,0) 2:(1,1,0) 3:(0,1,0)
        #   4:(0,0,1) 5:(1,0,1) 6:(1,1,1) 7:(0,1,1)
        xf = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0], device=device, dtype=torch.long)
        yf = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], device=device, dtype=torch.long)
        zf = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], device=device, dtype=torch.long)
        w = wx[:, xf] * wy[:, yf] * wz[:, zf]  # (U', 8)
        # Coordinates stay f64 through the topology lookup and local-coordinate
        # calculation; the interpolated phase-field values follow design precision.
        new_phi[u_sel_v] = (w.to(field_torch_dtype) * old_phi[e_nodes]).sum(dim=1)

        self._hidden_phi = new_phi.clamp(0.0, 1.0).contiguous()
        if self._has_affine_constraints:
            phi_wp = wp.from_torch(self._hidden_phi, dtype=field_wp_dtype)
            self._prolongate_vector(phi_wp)
            self._hidden_phi = wp.to_torch(phi_wp).clone()

    def _localized_phi_to_bg_overwrite(self, base_grid, affected_elem_mask, device):
        """For each fully-interior outer hex element with at least one
        ``_hidden_phi`` value below 0.5, trilinear-interpolate the
        element's 8 nodal ``_hidden_phi`` values onto the bg cells
        inside its bbox, convert φ → SDF, and OVERWRITE the corresponding
        cells of ``base_grid``.

        Vectorised across all affected elements via
        ``torch.nn.functional.grid_sample`` (one batched 3-D trilinear
        call): only the bg cells covered by an affected element are
        touched, and there is no Python loop over elements.  Cheap when
        few elements are dipping (typical case — only the interior
        regions actively carving a void).
        """
        import torch.nn.functional as F

        n = int(self._sussman_grid_shape[0])
        elements_l = self._prev_octree_elements.long().to(device=device)
        affected = torch.where(affected_elem_mask.to(device=device))[0]
        K = int(affected.shape[0])
        if K == 0:
            return base_grid

        nodes_d = self._prev_octree_nodes.to(dtype=torch.float64, device=device)
        hidden_d = self._hidden_phi.to(dtype=self._design_torch_dtype(), device=device)
        affected_elements = elements_l[affected]  # (K, 8)
        elem_node_pos = nodes_d[affected_elements]  # (K, 8, 3)
        elem_node_phi = hidden_d[affected_elements]  # (K, 8) — VTK hex order

        elem_lo_p = elem_node_pos.amin(dim=1)  # (K, 3)
        elem_hi_p = elem_node_pos.amax(dim=1)

        axis_scale = torch.as_tensor(
            self._remesh_axis_scale, dtype=torch.float64, device=device
        )
        geo_min = torch.as_tensor(
            self._remesh_geo_min, dtype=torch.float64, device=device
        )
        spacing = 2.0 / (n - 1)
        elem_lo_norm = elem_lo_p / axis_scale + geo_min
        elem_hi_norm = elem_hi_p / axis_scale + geo_min
        i_lo = ((elem_lo_norm + 1.0) / spacing).floor().long().clamp(0, n - 1)  # (K, 3)
        i_hi = ((elem_hi_norm + 1.0) / spacing).ceil().long().clamp(0, n - 1)
        n_per = (i_hi - i_lo + 1).clamp_min(1)  # (K, 3)
        max_n = int(n_per.max().item())

        # Build a (K, 1, 2, 2, 2) source volume in (z, y, x) order matching
        # grid_sample's (D, H, W) = (z, y, x) convention.  VTK hex node
        # ordering: (x,y,z) flags → node index. Map ``(d=z, h=y, w=x)``:
        #   (0,0,0)→0  (0,0,1)→1  (0,1,0)→3  (0,1,1)→2
        #   (1,0,0)→4  (1,0,1)→5  (1,1,0)→7  (1,1,1)→6
        src = elem_node_phi.new_empty((K, 1, 2, 2, 2))
        src[:, 0, 0, 0, 0] = elem_node_phi[:, 0]
        src[:, 0, 0, 0, 1] = elem_node_phi[:, 1]
        src[:, 0, 0, 1, 1] = elem_node_phi[:, 2]
        src[:, 0, 0, 1, 0] = elem_node_phi[:, 3]
        src[:, 0, 1, 0, 0] = elem_node_phi[:, 4]
        src[:, 0, 1, 0, 1] = elem_node_phi[:, 5]
        src[:, 0, 1, 1, 1] = elem_node_phi[:, 6]
        src[:, 0, 1, 1, 0] = elem_node_phi[:, 7]

        # Build sample positions: (K, max_n, max_n, max_n, 3) where each
        # entry is (x, y, z) ∈ [-1, 1]^3 (grid_sample convention with
        # align_corners=True).  We pad each element's grid to max_n along
        # each axis; entries beyond ``n_per`` are clamped to the last
        # valid coord (their interpolated value is correct but they get
        # masked out below in the scatter).
        sample_dtype = hidden_d.dtype
        ix = torch.arange(max_n, device=device, dtype=sample_dtype)
        # Per-element local coord = ix / (n_per[k, ax] - 1) ∈ [0, 1].
        n_per_f = n_per.to(sample_dtype)  # (K, 3)
        denom = (n_per_f - 1.0).clamp_min(1.0)
        # Local fractional coord per (K, max_n) per axis; clamp invalid to 1.
        loc_x = (ix.view(1, -1) / denom[:, 0:1]).clamp(0.0, 1.0)  # (K, max_n)
        loc_y = (ix.view(1, -1) / denom[:, 1:2]).clamp(0.0, 1.0)
        loc_z = (ix.view(1, -1) / denom[:, 2:3]).clamp(0.0, 1.0)
        # Map [0, 1] → [-1, 1]
        coord_x = 2.0 * loc_x - 1.0
        coord_y = 2.0 * loc_y - 1.0
        coord_z = 2.0 * loc_z - 1.0
        # grid: (K, D, H, W, 3) with last-dim order (x, y, z).
        # Use broadcasting for outer product.
        gx = coord_x[:, None, None, :].expand(K, max_n, max_n, max_n)
        gy = coord_y[:, None, :, None].expand(K, max_n, max_n, max_n)
        gz = coord_z[:, :, None, None].expand(K, max_n, max_n, max_n)
        grid = torch.stack([gx, gy, gz], dim=-1)  # (K, max_n, max_n, max_n, 3)

        sampled = F.grid_sample(
            src, grid, mode="bilinear", padding_mode="border", align_corners=True
        )  # (K, 1, max_n, max_n, max_n)
        phi_block = sampled.squeeze(1)  # (K, max_n, max_n, max_n)
        sdf_block = _phi_to_sdf(phi_block.clamp(0.0, 1.0), float(self._epsilon))

        # Scatter-write into base_grid at each element's bg-cell range.
        # Build per-cell bg indices and a validity mask.  Total cells =
        # K * max_n^3; for K=10, max_n=9 (depth-4 in depth-7 grid) =
        # 7290 cells — cheap.
        kk = torch.arange(K, device=device).view(K, 1, 1, 1)
        offz = torch.arange(max_n, device=device).view(1, max_n, 1, 1)
        offy = torch.arange(max_n, device=device).view(1, 1, max_n, 1)
        offx = torch.arange(max_n, device=device).view(1, 1, 1, max_n)
        valid = (offz < n_per[kk, 2]) & (offy < n_per[kk, 1]) & (offx < n_per[kk, 0])
        gz_idx = i_lo[kk, 2] + offz
        gy_idx = i_lo[kk, 1] + offy
        gx_idx = i_lo[kk, 0] + offx

        new_grid = base_grid.clone()
        sdf_flat = sdf_block.to(new_grid.dtype)[valid]
        gx_flat = gx_idx.expand(K, max_n, max_n, max_n)[valid]
        gy_flat = gy_idx.expand(K, max_n, max_n, max_n)[valid]
        gz_flat = gz_idx.expand(K, max_n, max_n, max_n)[valid]
        new_grid[gx_flat, gy_flat, gz_flat] = sdf_flat

        return new_grid

    def _pf_reset_after_void_formation(self):
        """After the localized trilinear interp + MC reinit have carved a
        new void surface into the SDF, reset:
          • ``_hidden_phi`` at interior nodes to ``tanh(new SDF/ε)`` so
            the next round of dip accumulation starts from a clean
            baseline (no leftover sub-0.5 values that have already been
            promoted to actual void surfaces in the SDF).
          • ``_sussman_grid_sdf`` at deep-interior bg cells to the global
            most-negative SDF value (uniform "deep solid" baseline) so MC
            on the next iter doesn't see leftover dips as spurious zero
            crossings outside the new voids.
        Boundary band of ``_hidden_phi`` is unchanged (it tracks SDF and
        already does the right thing).  Hanging-node consistency is
        re-enforced via ``_prolongate_vector``.
        """
        eps = float(self._epsilon)
        field_torch_dtype = self._design_torch_dtype()
        field_wp_dtype = self._design_wp_dtype()
        sdf_now = wp.to_torch(self.e.dof_values).detach().to(field_torch_dtype)
        band_k = float(self.params.get("phi_reproject_band_k", 3.0))
        # Reset hidden_phi at EVERY node to ``_sdf_to_phi(sdf_now)`` —
        # the clean SDF-consistent baseline after MC has carved the new
        # void surface.  Covers:
        #   • Deep solid (sdf << 0): reset to ≈ 1 — wipes any leftover
        #     gradient-induced sub-0.5 dips that didn't form voids.
        #   • Deep void (sdf >> 0): reset to ≈ 0 — voids in their proper
        #     "0-material" state.
        #   • Boundary band (|sdf| < band·ε): includes both the body
        #     surface AND the walls of thin voids whose interior sits in
        #     band.  Resetting these to ``_sdf_to_phi(sdf)`` is the same
        #     value step (a) of the NEXT iter's ``_pf_update_hidden_phi``
        #     would assign — bringing it forward avoids leaving sub-0.5
        #     values inside thin voids visible in the iter-N save() and
        #     carried into the next iter's remesh interpolation.
        new_phi = _sdf_to_phi(sdf_now, eps).clamp(0.0, 1.0).contiguous()
        self._hidden_phi = new_phi
        if self._has_affine_constraints:
            phi_wp = wp.from_torch(self._hidden_phi, dtype=field_wp_dtype)
            self._prolongate_vector(phi_wp)
            self._hidden_phi = wp.to_torch(phi_wp).clone()

        # Bg-grid reset: set deep-SOLID bg cells to a uniform ``global_max_neg``
        # so MC next iter doesn't see leftover dip signatures as spurious
        # zero crossings outside any newly formed voids.  Deep-void cells
        # (positive sdf) are left alone so the void surface is preserved.
        # Dip signatures at octree-node bg cells are re-asserted next iter
        # via scatter + localized overwrite from the preserved hidden_phi.
        bg_sdf = self._sussman_grid_sdf
        bg_deep_solid = bg_sdf < -band_k * eps
        if bool(bg_deep_solid.any()):
            global_max_neg = float(bg_sdf.amin())
            bg_sdf_new = bg_sdf.clone()
            bg_sdf_new[bg_deep_solid] = global_max_neg
            self._sussman_grid_sdf = bg_sdf_new

    def _get_decoder_grid_sdf(self):
        """Return a fresh bg-grid SDF for the ``"decoder_marchingcubes"`` remeshing mode.

        Called by ``_post_optimizer_step_remesh_impl`` instead of scattering
        ``e.dof_values`` from old octree node positions.

        Returns
        -------
        Tensor
            Shape ``(n, n, n)`` float32, **physical SDF units**, matching
            ``self._sussman_grid_sdf.shape``.

        Notes
        -----
        Subclasses with access to a generative SDF decoder (e.g.
        ``LatentShapeOpt``) override this to re-query the decoder at positions
        near the current surface via ``hierarchical_extraction``.  The full
        background grid is then accurate for the new surface position, unlike
        the scatter step which only updates ~4 % of the grid (the old octree
        node positions) and leaves the rest stale.

        The base class raises ``NotImplementedError`` so a missing override is
        caught immediately rather than silently using a stale grid.
        """
        raise NotImplementedError(
            "remeshing_method='decoder_marchingcubes' requires a subclass that overrides "
            "_get_decoder_grid_sdf()"
        )

    def _post_optimizer_step_remesh_impl(self, iteration):
        import warp as wp
        from .remeshing import (
            clip_octree_to_bbox,
            make_trilinear_geo_func,
            remesh_to_octree,
            scale_octree_nodes_to_physical,
            scatter_octree_to_cube_grid,
            _surface_vertices_to_must_query_positions,
        )

        if not self._reinit_sdf:
            return

        # Clear per-iter MC BVH cache so each iter builds a fresh one.
        for _attr in (
            "_mc_wp_mesh",
            "_mc_wp_negate",
            "_mc_wp_keep_alive",
            "_mc_verts_norm",
            "_mc_faces",
            "_mc_bbox_diag_phys",
        ):
            self.__dict__.pop(_attr, None)
        if (iteration + 1) % self._reinit_every != 0:
            return

        device = str(self.device)

        # Phase-field mode: update the persistent ``_hidden_phi`` field
        # from the intermediate ∂J/∂e_phi gradient.  An element is
        # "affected" (i.e. carving a new void) iff:
        #   (a) ALL 8 of its nodes are deep interior (SDF < -band·ε),
        #       so the element is far inside the body — boundary-band and
        #       exterior elements are excluded and never counted; and
        #   (b) AT LEAST ONE of its 8 nodes has ``_hidden_phi < 0.5``
        #       — i.e. the gradient has carved a sub-0.5 dip at that
        #       node, so a void surface should appear inside the element.
        # This filters out the trivial "exterior nodes always have
        # φ ≈ 0" false positives that otherwise flood the affected set
        # with ALL body-boundary elements.
        pf_mode = self._use_phase_field_design()
        affected_elem_mask = None
        any_dipped = False
        # ``decoder_marchingcubes`` is not compatible with PF-mode dip
        # processing — it gets its bg grid directly from a decoder
        # re-query (``_get_decoder_grid_sdf``) with no scatter step,
        # so none of the PF-path machinery (hidden_phi update, thin
        # filter, localized overwrite, dip-driven must_query) applies.
        decoder_mode = self._remeshing_method == "decoder_marchingcubes"
        if pf_mode and not decoder_mode:
            # Keep the historical zero-argument hook call: illustration and
            # downstream recorder subclasses override this method to observe
            # the update and then call ``super()``.  Passing ``iteration`` here
            # would break those subclasses even though the numerical operation
            # itself is unchanged.
            self._pf_update_iteration = iteration
            try:
                self._pf_update_hidden_phi()
            finally:
                del self._pf_update_iteration
            sdf_now_t = (
                wp.to_torch(self.e.dof_values).detach().to(self._design_torch_dtype())
            )
            band_k = float(self.params.get("phi_reproject_band_k", 3.0))
            interior_node = sdf_now_t < -band_k * float(self._epsilon)
            elements_l = self._prev_octree_elements.long().to(self._hidden_phi.device)
            elem_all_interior = interior_node[elements_l].all(dim=1)

            # Apply thin-feature filter, but with rollback: if filtering
            # would eliminate ALL viable voids (no element survives the
            # ``elem_all_interior & elem_has_dipped`` test post-filter),
            # restore the pre-filter ``hidden_phi`` so the dips persist
            # and keep accumulating in subsequent iters.  This avoids the
            # case where the filter converts every gradient-induced dip
            # to 0.51 and the optimizer makes no progress toward a
            # resolvable void cluster.  When the filter retains some
            # dips, those drive void formation as usual.
            elem_dipped_pre = (self._hidden_phi[elements_l] < 0.5).any(dim=1)
            affected_pre = elem_all_interior & elem_dipped_pre
            if bool(affected_pre.any()):
                phi_pre = self._hidden_phi.clone()
                phi_post = self._pf_filter_thin_features(self._hidden_phi)
                elem_dipped_post = (phi_post[elements_l] < 0.5).any(dim=1)
                affected_post = elem_all_interior & elem_dipped_post
                if bool(affected_post.any()):
                    # Filter retained some viable dips (or no thin
                    # features at all) → commit and form voids around
                    # the survivors.
                    self._hidden_phi = phi_post
                    affected_elem_mask = affected_post
                    any_dipped = True
                else:
                    # Filter would eliminate every viable void this iter.
                    # Roll back so the dips persist and keep accumulating
                    # — eventually the dipped region will be wide enough
                    # to survive the filter and form a proper void.  All
                    # downstream dip-related processing (localized
                    # overwrite, must-query refinement, post-void reset)
                    # is gated on ``any_dipped`` and becomes a no-op.
                    self._hidden_phi = phi_pre
                    self.log(
                        "  PF: all dips sub-resolution this iter; "
                        "deferring void formation (dips preserved)"
                    )
                    any_dipped = False
                    affected_elem_mask = None

        # 1. Get an up-to-date background-grid SDF near the current surface.
        if decoder_mode:
            # Subclass re-queries its decoder over the full grid (e.g. via
            # hierarchical_extraction in LatentShapeOpt), bypassing the
            # stale scatter from old octree node positions.  PF-mode dip
            # processing is not run in this branch.
            new_grid_phys = self._get_decoder_grid_sdf()
            new_grid_phys = self._post_scatter_bg_grid_hook(new_grid_phys)
        else:
            # Per-iter bg-grid SDF construction — sparse scatter (cheap,
            # O(V)) at octree-aligned cells PLUS, in PF mode with dipping
            # nodes, a localized trilinear-interp overwrite on bg cells
            # inside elements that touch a dipping node.  This makes the
            # bg grid coherent around any nascent void surface (so MC can
            # extract it) without paying the O(N_bg) dense-interp cost
            # everywhere.
            #
            # Gating on ``any_dipped``:
            #   • PF mode with viable dips (any_dipped=True): use
            #     ``_phi_to_sdf(hidden_phi)`` so the scatter injects dip
            #     signal into the bg grid for MC to carve voids.
            #   • PF mode with deferred dips (any_dipped=False, filter
            #     rolled back) OR SDF mode: use the raw SDF directly.
            #     Otherwise the dipped ``hidden_phi`` would still inject
            #     sub-0.5 phi → positive-SDF pulses at dipping octree
            #     nodes, MC would carve tiny partial voids, and the new
            #     octree would partially refine around them (the
            #     "diamond" artifacts).  Using the raw SDF makes this
            #     path a true no-op for void formation while
            #     ``hidden_phi`` silently keeps accumulating its sub-0.5
            #     values for the next iter.
            if pf_mode and any_dipped:
                scatter_src = _phi_to_sdf(
                    self._hidden_phi, float(self._epsilon)
                ).contiguous()
            else:
                scatter_src = wp.to_torch(self.e.dof_values)

            new_grid_phys = scatter_octree_to_cube_grid(
                self._prev_octree_nodes,
                scatter_src,
                self._sussman_grid_sdf,
                self._remesh_axis_scale,
                self._remesh_geo_min,
                device=device,
            )

            if pf_mode and any_dipped:
                new_grid_phys = self._localized_phi_to_bg_overwrite(
                    new_grid_phys, affected_elem_mask, device
                )

            new_grid_phys = self._post_scatter_bg_grid_hook(new_grid_phys)

        # 2. Freeze BC region BEFORE reinit so reinit uses correct BC values
        #    as its boundary condition and doesn't propagate drifted BC values
        #    into the interior.
        flat = new_grid_phys.reshape(-1)
        flat[self._bg_frozen_mask] = self._bg_frozen_values
        new_grid_phys = flat.reshape(self._sussman_grid_shape)

        # 4. Reinit on the bg grid (physical units throughout).
        def _vol_proxy(g):
            """Phase-field volume proxy: mean φ(SDF) = mean 0.5*(1-tanh(SDF/ε)).
            Sensitive to sub-cell zero level set shifts unlike a simple sign count."""
            eps = float(self.params.get("phase_field_epsilon", 0.02))
            return (0.5 * (1.0 - torch.tanh(g.float() / eps))).mean().item()

        vol_before = _vol_proxy(new_grid_phys)
        if self._reinit_sdf_method == "sussman":
            dx, dy, dz = self._sussman_grid_spacing
            grad_before = ShapeOpt._eikonal_grad_mag(
                new_grid_phys.reshape(self._sussman_grid_shape),
                dx,
                dy,
                dz,
            )
            cleaned = reinitialize_sdf_sussman(
                new_grid_phys.to(torch.float64),
                dx,
                dy,
                dz,
                n_iters=self._reinit_n_iters,
            ).to(torch.float32)
            grad_after = ShapeOpt._eikonal_grad_mag(
                cleaned.reshape(self._sussman_grid_shape),
                dx,
                dy,
                dz,
            )
            # Re-apply frozen mask after Sussman (Sussman may drift BC values).
            flat = cleaned.reshape(-1)
            flat[self._bg_frozen_mask] = self._bg_frozen_values
            cleaned = flat.reshape(self._sussman_grid_shape)
            vol_after = _vol_proxy(cleaned)
            self.log(
                f"  SDF reinit (sussman): |∇φ| mean {grad_before:.4f}→{grad_after:.4f}, "
                f"|∇φ|-1 L∞ {abs(grad_before-1):.4f}→{abs(grad_after-1):.4f}, "
                f"vol_proxy {vol_before:.4f}→{vol_after:.4f} (Δ={vol_after-vol_before:+.4f})"
            )
        else:  # "marchingcubes"
            # pre-freeze already applied above; MC reinit sees correct BC.
            cleaned = self._mc_reinit_bg_grid(new_grid_phys)

        self._sussman_grid_sdf = cleaned

        # 5. Build a new outer octree from the cleaned bg grid.
        # Baseline shortcut: when init_octree_depth == octree_depth the octree
        # structure never changes.  Octree nodes lie at exact integer indices on
        # the bg lattice, so we can read back updated SDF values with a direct
        # lookup — no remesh_to_octree needed.
        if self._remesh_init_octree_depth == self._remesh_octree_depth:
            # Uniform hex mesh: nodes are on the bg-grid lattice at the stored
            # index ranges — direct slice, no index computation needed.
            ix_lo, ix_hi = self._hex_bg_ix
            iy_lo, iy_hi = self._hex_bg_iy
            iz_lo, iz_hi = self._hex_bg_iz
            _e_wp_dtype = self.e.dof_values.dtype
            _e_torch_dtype = (
                torch.float64 if _e_wp_dtype == wp.float64 else torch.float32
            )
            new_sdf = (
                cleaned[ix_lo : ix_hi + 1, iy_lo : iy_hi + 1, iz_lo : iz_hi + 1]
                .reshape(-1)
                .to(dtype=_e_torch_dtype)
                .contiguous()
            )
            wp.copy(
                src=wp.from_torch(new_sdf, dtype=_e_wp_dtype),
                dest=self.e.dof_values,
            )
            if pf_mode and any_dipped:
                self._pf_reset_after_void_formation()
            return
        # Always build a trilinear func on cleaned — used both to assign SDF
        # onto the new octree nodes and to write cleaned SDF back to e.dof_values
        # on the current outer nodes (so save() visualizes the reinitialized SDF).
        trilinear_func = make_trilinear_geo_func(cleaned)

        if self._remeshing_method == "interpolate":
            geo_func = trilinear_func
        else:  # "marchingcubes" or "decoder_marchingcubes"
            # Reuse BVH if MC reinit already built one this iter; else build fresh.
            if hasattr(self, "_mc_wp_mesh"):
                geo_func = self._mc_geo_func()
            else:
                geo_func = self._build_mc_surface_and_bvh(cleaned)

        # PF-mode: pass dip-node normalized positions as ``must_query_positions``
        # so ``hierarchical_extraction`` directly samples the BVH at those
        # sub-init-lattice points (instead of relying on init-depth band/sign
        # detection, which misses tiny voids whose only signal sits between
        # init-depth lattice nodes).
        #
        # PERSISTENT accumulation: once a node has been a deep-interior
        # dip (or a deferred/filtered dip) in any past iter, we keep its
        # normalized-cube position in ``_mq_positions_persistent`` for
        # the rest of the optimisation.  Each iter's newly-found interior
        # dips are unioned in (deduped to octree_depth-lattice precision)
        # — the accumulation is monotonic.  This ensures future remeshes
        # keep forcing refinement at those positions regardless of
        # whether the dip survived the thin-feature filter this iter
        # (deferred dips still register) and regardless of whether a
        # void formed there yet.
        mq_norm = None
        if pf_mode and self._hidden_phi is not None:
            band_k_d = float(self.params.get("phi_reproject_band_k", 3.0))
            sdf_now_d = wp.to_torch(self.e.dof_values).detach().to(torch.float64)
            interior_dip = (self._hidden_phi < 0.5) & (
                sdf_now_d < -band_k_d * float(self._epsilon)
            )
            axis_scale_t = torch.as_tensor(
                self._remesh_axis_scale, dtype=torch.float64, device=device
            )
            geo_min_t = torch.as_tensor(
                self._remesh_geo_min, dtype=torch.float64, device=device
            )
            # Accumulate this iter's interior-dip normalized positions.
            if bool(interior_dip.any()):
                dip_phys = self._prev_octree_nodes[interior_dip].to(
                    device=device, dtype=torch.float64
                )
                new_norm = dip_phys / axis_scale_t + geo_min_t
                prev = getattr(self, "_mq_positions_persistent", None)
                if prev is None or prev.numel() == 0:
                    combined = new_norm
                else:
                    combined = torch.cat([prev.to(new_norm.device), new_norm], dim=0)
                # Dedupe by snapping to the finest lattice (``octree_depth``)
                # and taking unique integer indices, then mapping back.
                max_d = int(self._remesh_octree_depth)
                spacing_fin = 2.0 / (2**max_d)
                idx_q = ((combined - (-1.0)) / spacing_fin).round().long()
                idx_unique = torch.unique(idx_q, dim=0)
                self._mq_positions_persistent = (
                    -1.0 + idx_unique.to(torch.float64) * spacing_fin
                )
            if (
                getattr(self, "_mq_positions_persistent", None) is not None
                and self._mq_positions_persistent.shape[0] > 0
            ):
                mq_norm = self._mq_positions_persistent.to(device=device)

        # If this optimization was bootstrapped from an explicit surface,
        # keep the same guarantee for every later explicit surface already
        # produced by MC reinit/remeshing.  _mc_verts_norm is the effective
        # surface after component filtering, loop closing, and BC graft/zip.
        # Emit only unique final-parent centres; hierarchical_extraction carries
        # them through every coarser transition.  Then union with phase-field
        # dip positions.
        if (
            getattr(self, "_refine_explicit_surfaces", False)
            and getattr(self, "_mc_verts_norm", None) is not None
        ):
            surface_mq, surface_counts = _surface_vertices_to_must_query_positions(
                self._mc_verts_norm.to(device=device, dtype=torch.float64),
                init_octree_depth=self._remesh_init_octree_depth,
                octree_depth=self._remesh_octree_depth,
            )
            if surface_mq.shape[0] > 0:
                mq_norm = (
                    surface_mq
                    if mq_norm is None
                    else torch.unique(torch.cat([surface_mq, mq_norm], dim=0), dim=0)
                )
                counts_text = ", ".join(f"d{d}:{n}" for d, n in surface_counts)
                self.log(
                    f"  Surface refinement: {self._mc_verts_norm.shape[0]} verts, "
                    f"{surface_mq.shape[0]} parent groups ({counts_text}), "
                    f"{mq_norm.shape[0]} total must-query positions"
                )

        octree_norm = remesh_to_octree(
            geo_func,
            init_octree_depth=self._remesh_init_octree_depth,
            octree_depth=self._remesh_octree_depth,
            domain_length=2.0,
            device=device,
            must_query_positions=mq_norm,
            sdf_truncation=self._sdf_truncation,
        )
        # MC geo_func returns exact BVH distances; replace with bg-grid values
        # so the octree SDF stays consistent with the reinitialized field.
        # (The MC surface is used only to determine octree structure / refinement.)
        if self._remeshing_method in ("marchingcubes", "decoder_marchingcubes"):
            nodes_f16 = octree_norm["nodes"].to(torch.float16).unsqueeze(0)
            octree_norm["sdf"] = trilinear_func(nodes_f16).squeeze(0).squeeze(-1)
        octree_phys = scale_octree_nodes_to_physical(
            octree_norm,
            self._remesh_axis_scale_t,
            self._remesh_geo_min_t,
        )
        # Clip the outer octree to the caller-supplied physical bbox.
        if self._remesh_clip_bbox is not None:
            octree_phys = clip_octree_to_bbox(
                octree_phys,
                self._remesh_clip_bbox[0],
                self._remesh_clip_bbox[1],
            )

        # Write the MC-reinit'd SDF back to e.dof_values on the CURRENT outer
        # octree nodes so that save() output is consistent with cleaned.
        # (The uniform-hex path above already does this via a direct slice.)
        # The pending octree is applied at the start of the NEXT step(), so
        # e.dof_values will be overwritten again then — this update is only
        # for save() consistency.
        _e_wp_dtype = self.e.dof_values.dtype
        _e_torch_dtype = torch.float64 if _e_wp_dtype == wp.float64 else torch.float32
        cur_nodes_norm = self._physical_to_normalized(
            self._prev_octree_nodes.to(dtype=torch.float64)
        )
        cur_sdf = (
            trilinear_func(cur_nodes_norm.to(torch.float16).unsqueeze(0))
            .squeeze(0)
            .squeeze(-1)
        )
        wp.copy(
            src=wp.from_torch(
                cur_sdf.to(_e_torch_dtype).contiguous(), dtype=_e_wp_dtype
            ),
            dest=self.e.dof_values,
        )

        # PF mode (adaptive remesh path): apply the same post-void reset
        # the uniform-hex path does, so the next iter starts from a
        # clean interior baseline.
        if pf_mode and any_dipped:
            self._pf_reset_after_void_formation()

        # 6. Defer the outer reinit_with_sdf to the start of the NEXT step().
        # PF mode: stash a sparse bg-grid representation of ``_hidden_phi``
        # for cross-remesh persistence.  The base layer is
        # ``_sdf_to_phi(_sussman_grid_sdf)`` (the clean post-reinit φ),
        # then we sparse-scatter the actual ``_hidden_phi`` at outer
        # octree node positions on top.  When ``step()`` interpolates
        # this stash onto the new outer mesh, refined nodes near old
        # dipping nodes pick up the dips, while non-dipping cells fall
        # back to the clean baseline.  Cheap (O(V) scatter); sufficient
        # because after a ``_pf_reset_after_void_formation`` the hidden
        # φ is essentially the clean baseline anyway.
        if (
            self._use_phase_field_design()
            and getattr(self, "_hidden_phi", None) is not None
        ):
            # Stash OLD octree + hidden_phi for parent-element interp at
            # the start of the next ``step()`` (after the pending remesh
            # is applied).  Matched new nodes (same bg-lattice index as
            # an old node) copy verbatim; unmatched new nodes get
            # trilinear interp from the OLD element that contains them.
            self._pending_old_hidden_phi_stash = {
                "nodes": self._prev_octree_nodes.clone(),
                "elements": self._prev_octree_elements.clone(),
                "hidden_phi": self._hidden_phi.clone(),
            }

        self._pending_remesh_octree = octree_phys
        self._pending_remesh_n_old = int(self.e_space.node_count())

    def _init_bg_frozen_mask(self):
        """Capture the iter-0 frozen BC region as a bg-grid mask + values.

        Called lazily on the first ``_post_optimizer_step_remesh`` call (i.e.
        after iter 0 has run and ``_reinit_fixed_bool`` has been populated by
        ``_rebuild_outer_free_mask``).  Each octree BC node maps to an exact
        integer index on the bg lattice (``hierarchical_extraction`` builds
        corners on the ``(2^d+1)^3`` cube), so we scatter the BC flags
        directly — no point-in-element search needed.
        """
        from .remeshing import octree_node_bg_indices

        device = str(self.device)
        bg_n = self._sussman_grid_shape[0]

        reinit_fixed = self._reinit_fixed_bool
        if reinit_fixed is None or not reinit_fixed.any():
            self._bg_frozen_mask = torch.zeros(bg_n**3, dtype=torch.bool, device=device)
            self._bg_frozen_values = torch.empty(0, dtype=torch.float32, device=device)
            return

        idx = octree_node_bg_indices(
            self._prev_octree_nodes,
            bg_n,
            self._remesh_axis_scale,
            self._remesh_geo_min,
            device=device,
        )
        bg_fixed = torch.zeros((bg_n, bg_n, bg_n), dtype=torch.bool, device=device)
        bg_fixed[idx[:, 0], idx[:, 1], idx[:, 2]] = reinit_fixed.to(device=device)
        self._bg_frozen_mask = bg_fixed.reshape(-1)
        self._bg_frozen_values = self._sussman_grid_sdf.reshape(-1)[
            self._bg_frozen_mask
        ].clone()
        n_frozen = int(self._bg_frozen_mask.sum().item())
        self.log(
            f"  Remesh BC fix init: {n_frozen}/{bg_n**3} bg cells frozen "
            f"(from {int(reinit_fixed.sum().item())} octree BC nodes)"
        )

    def _objective_function(self, tape):
        result = self._inner._objective_function(tape)
        self._last_se_raw = self._inner._last_se_raw
        self._last_vol_raw = self._inner._last_vol_raw
        return result

    def _post_tape_backward(self):
        # Inner's _post_tape_backward now handles prolongation for both SDF and e_phi.
        self._inner._post_tape_backward()

        # Expand inner SDF gradient (trimmed nodes) to outer (full mesh).
        # Prolongation already applied by inner._post_tape_backward().
        outer_dtype = self._design_torch_dtype()
        inner_grad = wp.to_torch(self._inner.e.dof_values.grad).to(outer_dtype)
        full_grad = torch.zeros_like(wp.to_torch(self.e.dof_values), dtype=outer_dtype)
        full_grad[self._trim_active_indices.long()] = inner_grad

        # PF mode: the SDF gradient drives the boundary band only; interior
        # evolution is delegated to the hidden φ field (e_phi) via the
        # shadow gradient.  Zero the interior portion of ∂J/∂SDF so GD on
        # SDF cannot compete with hidden-φ updates (which can create
        # topologically new interior boundaries via MC + winding number).
        if self._use_phase_field_design():
            band_k = float(self.params.get("phi_reproject_band_k", 3.0))
            sdf_outer = (
                wp.to_torch(self.e.dof_values).detach().to(self._design_torch_dtype())
            )
            interior_mask = sdf_outer.abs() >= band_k * float(self._epsilon)
            full_grad[interior_mask] = 0.0

        wp.copy(
            src=wp.from_torch(
                full_grad.to(dtype=self._design_torch_dtype()).contiguous(),
                dtype=self._design_wp_dtype(),
            ),
            dest=self.e.dof_values.grad,
        )
        self._last_e_grad = wp.to_torch(self.e.dof_values.grad).clone()

        # PF mode: map inner's φ-gradient onto outer node positions.
        # Prolongation already applied by inner._post_tape_backward(), so just expand.
        if (
            self._use_phase_field_design()
            and self.e_phi is not None
            and hasattr(self._inner, "_last_e_phi_grad")
        ):
            field_torch_dtype = self._design_torch_dtype()
            inner_phi_grad = self._inner._last_e_phi_grad.to(field_torch_dtype)
            full_phi_grad = torch.zeros_like(
                wp.to_torch(self.e_phi.dof_values), dtype=field_torch_dtype
            )
            full_phi_grad[self._trim_active_indices.long()] = inner_phi_grad
            self._last_e_phi_grad = full_phi_grad.clone()

    def save(self, **kwargs):
        """Save inner (full solution on trimmed mesh) + outer (SDF + surface)."""
        self._inner.save(**kwargs)
        time = kwargs.get("time", 0)
        # Use the SDF snapshotted at this iteration's step() (the shape solved
        # this iter); _post_optimizer_step has since copied the reinitialized
        # next-iter SDF into e.dof_values.  Fall back to live before first step().
        sdf_np = getattr(self, "_save_sdf", None)
        if sdf_np is None:
            sdf_np = self.e.dof_values.numpy().copy()
        # Outer: SDF + SDF gradient on full mesh
        fields = {"sdf": sdf_np}
        if hasattr(self, "_last_e_grad"):
            fields["sdf_grad"] = self._last_e_grad.cpu().numpy()
            del self._last_e_grad
        # PF mode: dump the persistent hidden φ field (drives interior void
        # formation) and its source ∂J/∂e_phi gradient so users can inspect
        # interior carving progress in ParaView.  Use the hidden φ snapshotted at
        # step() (the field used this iter); _post_optimizer_step has since
        # GD-updated _hidden_phi for the next iter.  phi_grad (_last_e_phi_grad,
        # from _post_tape_backward) is already this iter's gradient -- the driver
        # of that update -- so it matches the snapshotted φ.
        if self._use_phase_field_design():
            save_phi = getattr(self, "_save_hidden_phi", None)
            if save_phi is None:
                save_phi = getattr(self, "_hidden_phi", None)
            if save_phi is not None:
                fields["hidden_phi"] = save_phi.detach().cpu().numpy()
            if hasattr(self, "_last_e_phi_grad"):
                fields["phi_grad"] = self._last_e_phi_grad.cpu().numpy()
                del self._last_e_phi_grad
        self.vtu_writer.save_timestep(time, fields)
        # Surface: VTU + GLB with interpolated displacement
        if hasattr(self, "_surface_writer"):
            inner_u = self._inner.u.dof_values.numpy().copy()
            n_outer = self.e_space.node_count()
            full_u = np.zeros((n_outer, inner_u.shape[1]), dtype=np.float64)
            active = self._trim_active_indices.cpu().numpy()
            full_u[active] = inner_u
            # Surface snapshotted at this iteration's step() = the shape solved
            # this iter; the live _mc_verts_norm has since been rebuilt by
            # _post_optimizer_step for the NEXT shape.  Fall back to the live
            # surface only before the first step() snapshot exists.
            save_mc_verts = getattr(self, "_save_mc_verts_norm", None)
            if save_mc_verts is not None:
                save_mc_faces = self._save_mc_faces
            else:
                save_mc_verts = getattr(self, "_mc_verts_norm", None)
                save_mc_faces = getattr(self, "_mc_faces", None)
            if save_mc_verts is not None:
                # MC path: defer displacement interpolation + tensor→numpy
                # conversions to the writer's async worker so "Queued surface
                # save" logs fire immediately, matching the other Queued logs.
                from .remeshing import octree_hex_interpolate
                from .vtu_writer import (
                    compute_node_gradients,
                    compute_strain_tensor,
                    compute_stress_tensor,
                )

                device = self.device
                nodes_t = self._prev_octree_nodes.to(device=device, dtype=torch.float64)
                elements_t = self._prev_octree_elements.long()
                dim_snap = self.dim
                mu_snap = float(self.mu)
                kappa_snap = float(self.kappa)
                if getattr(self._inner, "_has_spatial_material", False):
                    # A material field varies from node to node, and self.mu is
                    # only the median standing in for it, so the surface stress
                    # takes each inner node's own E and nu -- as the volume
                    # VTU's does -- rather than one material for the whole part.
                    material = self._inner._material_values_torch.detach().to(
                        device=device, dtype=torch.float64
                    ).clone()
                    mu_snap = material[:, 0] / (2.0 * (1.0 + material[:, 1]))
                    kappa_snap = material[:, 0] / (3.0 * (1.0 - 2.0 * material[:, 1]))
                # Gradients come off the inner trimmed mesh, the one the
                # solution actually lives on and the one the volume VTU uses.
                # full_u is inner_u scattered into the outer node set with zeros
                # wherever the mesh was trimmed away, so differentiating it would
                # read the step down to zero as strain right where the surface is.
                # Cloned, not referenced: the worker must not read a mesh cache
                # that remeshing may replace under it.  Kept on the device so the
                # batched least-squares stays on the GPU, which matters once the
                # octree is deep enough for the node count to run to six figures.
                inner_cache = self._inner._mesh_cache
                inner_points = (
                    inner_cache["points"].to(device=device, dtype=torch.float64).clone()
                )
                inner_cells = inner_cache["cells"].to(device=device).long().clone()
                inner_u_np = inner_u
                active_idx = active
                full_u_np = full_u
                mc_verts_norm = save_mc_verts
                mc_faces = save_mc_faces
                normalized_to_physical = self._normalized_to_physical

                # Snapshot resolved SurfaceSelectors for async closure
                from .bc_surface_selectors import SurfaceSelector as _SS

                _surf_bc_selectors = [
                    (bc_type, idx, k)
                    for bc_type, bc_dict in (
                        ("dirichlet", self._original_dirichlet_bcs_dict),
                        ("neumann", self._original_neumann_bcs_dict),
                    )
                    for idx, k in enumerate(bc_dict)
                    if isinstance(k, _SS) and k.is_resolved
                ]

                def _compute_surface_mesh():
                    full_u_t = torch.as_tensor(
                        full_u_np, dtype=torch.float64, device=device
                    )
                    verts_phys_t = normalized_to_physical(
                        mc_verts_norm.to(dtype=torch.float64, device=device)
                    )
                    # Strain and stress come off one gradient reconstruction,
                    # and all three fields ride on one interpolation.  Locating
                    # each surface vertex inside the octree is what this costs;
                    # the per-field arithmetic afterwards is trivial beside it,
                    # so the search is paid once for a wide field rather than
                    # once per field.
                    gradients = compute_node_gradients(
                        None,
                        None,
                        None,
                        dim_snap,
                        mesh_cache={"points": inner_points, "cells": inner_cells},
                        disp_tensor=torch.as_tensor(
                            inner_u_np, dtype=torch.float64
                        ),
                    )
                    nodal = [full_u_t]
                    for values in (
                        compute_strain_tensor(
                            None, None, None, dim_snap, gradients=gradients
                        ),
                        compute_stress_tensor(
                            None, None, None, dim_snap, mu_snap, kappa_snap,
                            gradients=gradients,
                        ),
                    ):
                        # Scattered onto the outer nodes the same way full_u is,
                        # so every field is sampled by one pass over the octree.
                        scattered = torch.zeros(
                            (nodes_t.shape[0], values.shape[1]),
                            dtype=torch.float64,
                            device=device,
                        )
                        scattered[active_idx] = torch.as_tensor(
                            values, dtype=torch.float64, device=device
                        )
                        nodal.append(scattered)
                    widths = [field.shape[1] for field in nodal]
                    sampled = octree_hex_interpolate(
                        nodes_t, elements_t, torch.cat(nodal, dim=1), verts_phys_t
                    )
                    disp_verts, strain_verts, stress_verts = torch.split(
                        sampled, widths, dim=1
                    )
                    surf_point_data = {
                        "displacement": disp_verts.detach().cpu().numpy(),
                        "strain": strain_verts.detach().cpu().numpy(),
                        "stress": stress_verts.detach().cpu().numpy(),
                    }
                    # BC selector bbox regions on the surface mesh vertices
                    if _surf_bc_selectors:
                        vp = verts_phys_t.float().cpu()
                        x, y, z = vp[:, 0], vp[:, 1], vp[:, 2]
                        import numpy as _np

                        for bc_type, idx, sel in _surf_bc_selectors:
                            bbox_mask = sel(x, y, z).numpy().astype(_np.float32)
                            surf_point_data[f"bc_{bc_type}_{idx}_bbox"] = bbox_mask
                    return (
                        verts_phys_t.float().cpu().numpy(),
                        mc_faces.cpu().numpy(),
                        surf_point_data,
                    )

                self._surface_writer.save_from_mesh_lazy(time, _compute_surface_mesh)
            else:
                # Structured-grid path: MC from SDF + trilinear displacement interp.
                self._surface_writer.save_timestep(
                    time,
                    sdf_np,
                    grid_fields={"displacement": full_u},
                )

    def finalize(self):
        """Finalize all VTU writers."""
        # Inner: full solution PVD
        if getattr(self, "_inner", None) is not None:
            pvd_path = self._inner.vtu_writer.write_pvd()
            self._inner.vtu_writer.shutdown()
            self.log(
                f"Simulation finalized. Open {pvd_path} in ParaView to view results."
            )
        # Outer: SDF PVD
        sdf_pvd = self.vtu_writer.write_pvd("sdf.pvd")
        self.vtu_writer.shutdown()
        self.log(f"SDF field saved. Open {sdf_pvd} in ParaView to view SDF evolution.")
        # Surface PVD
        if hasattr(self, "_surface_writer"):
            surf_pvd = self._surface_writer.write_pvd()
            self._surface_writer.shutdown()
            self.log(f"Surface meshes saved. Open {surf_pvd} in ParaView.")
