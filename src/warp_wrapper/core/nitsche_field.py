"""
Diffused Nitsche method for neo-Hookean finite deformation on octree hex meshes.

Enforces boundary conditions on embedded geometry via phase-field-weighted
volumetric integration plus Nitsche penalty/consistency/symmetry terms,
eliminating the need for conforming meshes.
"""

import contextlib
import dataclasses
import math
import os
from typing import List, Optional
import warp as wp
import warp.fem as fem
from warp.fem import Coords
import torch
from .abstract_field import OctreeField
from .neoHookean import NeoHookean
from .utils import (
    identify_boundary_nodes,
    evaluate_bc_values,
)
from .nitsche_kernels import (
    precompute_qp_phi_grad,
    expand_element_beta_to_qp,
    cast_sdf_f64_to_f32,
    concat_sdf_arrays,
    accumulate_child_grad_to_parent,
)


@fem.integrand
def _qp_array_integrand(
    s: fem.Sample, domain: fem.Domain, f_arr: wp.array(dtype=wp.float32)
):
    """A per-QP array, integrated with the Nitsche quadrature's own weights.

    Indexed by ``s.qp_index`` exactly as the Nitsche integrands index their
    per-QP arrays, so a sum taken here is the one the residual assembles.
    """
    return f_arr[s.qp_index]


@dataclasses.dataclass
class NitscheChunk:
    """One group of boundary elements with similar QP counts.

    Boundary elements are bucketed by their leaf-cell count so that each chunk
    has a uniform ``qp_per_cell``, minimising padding waste in fem.integrate's
    dispatch buffers.
    """

    chunk_id: int
    # Indices into the boundary-local array (0..n_boundary-1)
    boundary_local_indices: torch.Tensor
    qp_per_cell: int
    n_elements: int

    # Warp arrays for ExplicitQuadrature — set by _build_correction_domain
    points_wp: Optional[wp.array] = None
    weights_wp: Optional[wp.array] = None

    # Subdomain / quadrature / test / trial — set by _build_correction_domain
    subdomain: Optional[fem.Subdomain] = None
    quadrature: Optional[fem.ExplicitQuadrature] = None
    test: Optional[fem.Field] = None
    trial: Optional[fem.Field] = None

    # Per-QP value arrays (flat, length = n_elements * qp_per_cell)
    # Set by _precompute_qp_fields and _build_nitsche_bc_masks
    phi_qp: Optional[wp.array] = None
    grad_phi_qp: Optional[wp.array] = None
    grad_phi_mag_qp: Optional[wp.array] = None
    beta_qp: Optional[wp.array] = None
    u_bar_qp: Optional[wp.array] = None
    t_bar_qp: Optional[wp.array] = None
    psi_D_qp: Optional[wp.array] = None
    psi_N_qp: Optional[wp.array] = None
    n_face_qp: Optional[wp.array] = None
    qp_active_wp: Optional[wp.array] = None

    # Torch tensors for QP mapping (set during adaptive quadrature build)
    qp_phi_source_elem: Optional[torch.Tensor] = None
    qp_phi_ref_coords: Optional[torch.Tensor] = None
    qp_active: Optional[torch.Tensor] = None
    qp_parent_elem: Optional[torch.Tensor] = None
    qp_parent_ref: Optional[torch.Tensor] = None
    qp_weights_flat: Optional[torch.Tensor] = None


# Standard 2×2×2 Gauss points in [0,1]^3 reference hex (warp node ordering)
_G_LO = 0.5 - 1.0 / (2.0 * math.sqrt(3.0))
_G_HI = 0.5 + 1.0 / (2.0 * math.sqrt(3.0))
_STD_GAUSS_PTS = [
    [_G_LO, _G_LO, _G_LO],
    [_G_HI, _G_LO, _G_LO],
    [_G_HI, _G_HI, _G_LO],
    [_G_LO, _G_HI, _G_LO],
    [_G_LO, _G_LO, _G_HI],
    [_G_HI, _G_LO, _G_HI],
    [_G_HI, _G_HI, _G_HI],
    [_G_LO, _G_HI, _G_HI],
]
_STD_GAUSS_W = 1.0 / 8.0  # each weight for unit cube

# Pre-computed 8×8 trilinear interpolation matrix: interp[qp, node] = N_node(qp)
# Used for interpolating nodal values (e.g. phi) to standard Gauss QPs.
_HEX_CORNERS = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]
_TRILINEAR_INTERP_8x8 = torch.zeros(8, 8, dtype=torch.float32)
for _qi in range(8):
    _xi, _eta, _zeta = _STD_GAUSS_PTS[_qi]
    for _ni in range(8):
        _x0, _y0, _z0 = _HEX_CORNERS[_ni]
        _Nx = _xi if _x0 == 1 else (1.0 - _xi)
        _Ny = _eta if _y0 == 1 else (1.0 - _eta)
        _Nz = _zeta if _z0 == 1 else (1.0 - _zeta)
        _TRILINEAR_INTERP_8x8[_qi, _ni] = _Nx * _Ny * _Nz


def bc_match_tolerance(epsilon: float, min_element_size: float) -> float:
    """Distance within which a BC region claims points off its surface.

    The one definition behind ``NitscheNeoHookean._bc_match_tolerance`` (see
    its docstring for why it is ``max(1.5 eps, 0.75 h)``), exposed so callers
    that resolve a ``SurfaceSelector`` before the solver exists use the same
    slack.  The workflow resolves saved-segment selectors itself, on the
    surface their labels were computed on; it used ``eps_expansion = h``
    there, which re-introduced the h-tall clamped strip this tolerance was
    tightened to remove.
    """
    return max(1.5 * float(epsilon), 0.75 * float(min_element_size))


class NitscheNeoHookean(OctreeField, NeoHookean):
    """Neo-Hookean solver using the diffused Nitsche method on octree hex meshes.

    Enforces Dirichlet/Neumann BCs on an embedded geometry boundary via
    phase-field integration instead of conforming mesh boundary conditions.

    Required params (in addition to OctreeField / NeoHookean params):
        octree_mesh: dict with keys 'nodes', 'elements', 'sdf', and optionally
                     'edge_hanging', 'face_hanging'
        quad_octree_mesh: dict with keys 'nodes', 'elements', and optionally
                          'edge_hanging', 'face_hanging' — finer mesh for quadrature
        quad_sdf: tensor — SDF values at quad_octree_mesh nodes
        phase_field_epsilon: float — phase field scale parameter
        update_beta_per_newton: bool (default False) — recompute beta each Newton iter
    """

    @staticmethod
    def _make_nitsche_stress(
        n: int,
        B_vals: tuple[float, ...] = (),
        scalar_dtype=wp.float32,
        phi_floor=1e-4,
        correction=False,
        symmetry_sign=1.0,
        sharp_volumetric=False,
        nitsche_phi_threshold=0.0,
        spatial_material=False,
        acceleration: tuple[float, ...] = (),
    ):
        """Build a Nitsche stress (residual) integrand.

        When ``correction=False``: full Nitsche integrand (volumetric weighted
        by phi_weight + consistency + symmetry + penalty + Neumann).

        When ``correction=True``: correction integrand for the two-integration
        split approach.  The volumetric term uses ``(phi_weight - 1)`` so that
        when added to the standard neo-Hookean integration the combined result
        equals ``phi_weight * P:grad_v + Nitsche_terms``.

        ``symmetry_sign``: +1.0 (standard symmetric Nitsche, default — K symmetric,
        coercive for β ≥ β_min), -1.0 (non-symmetric Nitsche — K non-symmetric,
        coercive for any β > 0), 0.0 (no symmetry term).

        ``sharp_volumetric``: if True, use step-function weight (1 if phi>0.5,
        phi_floor otherwise) for the volumetric term.  This eliminates O(eps)
        method error from the smooth phi transition in the volumetric integral.
        Nitsche BC terms still use smooth phi/grad_phi.

        ``nitsche_phi_threshold``: skip ALL Nitsche BC terms (consistency,
        symmetry, penalty, Neumann) at QPs where phi < threshold.  This
        prevents applying boundary forces at exterior QPs where stiffness is
        negligible, preserving coercivity.  Set to phi_floor for best results.
        """
        _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
        _one_third = scalar_dtype(1.0 / 3.0)
        _phi_floor = scalar_dtype(phi_floor)
        _one_minus_floor = scalar_dtype(1.0 - phi_floor)
        _two = scalar_dtype(2.0)
        _half = scalar_dtype(0.5)
        _one = scalar_dtype(1.0)
        _sharp_k = scalar_dtype(100.0)
        # correction=True: subtract 1 so combined with standard = phi_weight
        _vol_offset = scalar_dtype(1.0) if correction else scalar_dtype(0.0)
        _sym_sign = scalar_dtype(symmetry_sign)
        _sharp_vol = sharp_volumetric
        _phi_thr = scalar_dtype(nitsche_phi_threshold)
        _spatial_material = spatial_material
        vec3_t = wp.types.vector(length=3, dtype=scalar_dtype)
        body_components = [
            float(B_vals[i]) if i < len(B_vals) else 0.0 for i in range(n)
        ]
        body_const = wp.constant(vec3_t(*body_components))
        acceleration_components = [
            float(acceleration[i]) if i < len(acceleration) else 0.0
            for i in range(n)
        ]
        acceleration_const = wp.constant(vec3_t(*acceleration_components))

        @fem.integrand
        def nitsche_stress(
            s: fem.Sample,
            v: fem.Field,
            u: fem.Field,
            e: fem.Field,
            material: fem.Field,
            mu: scalar_dtype,
            kappa: scalar_dtype,
            phi_arr: wp.array(dtype=wp.float32),
            grad_phi_arr: wp.array(dtype=wp.vec3),
            grad_phi_mag_arr: wp.array(dtype=wp.float32),
            beta_arr: wp.array(dtype=wp.float32),
            u_bar_arr: wp.array(dtype=wp.vec3),
            t_bar_arr: wp.array(dtype=wp.vec3),
            psi_D_arr: wp.array(dtype=wp.vec3),
            psi_N_arr: wp.array(dtype=wp.float32),
            n_face_arr: wp.array(dtype=wp.vec3),
            qp_active_arr: wp.array(dtype=wp.int32),
            body_load_scale: scalar_dtype,
        ):
            qp = s.qp_index

            # Early-return for inactive (zero-weight) QPs — avoids expensive
            # matrix operations on padded QP slots.
            if qp_active_arr[qp] == 0:
                return scalar_dtype(0.0)

            # Pre-computed QP values (cast from float32 storage to compute dtype)
            phi = scalar_dtype(phi_arr[qp])
            gp_raw = grad_phi_arr[qp]
            gp = vec3_t(
                scalar_dtype(gp_raw[0]),
                scalar_dtype(gp_raw[1]),
                scalar_dtype(gp_raw[2]),
            )
            grad_phi_mag = scalar_dtype(grad_phi_mag_arr[qp])
            beta = scalar_dtype(beta_arr[qp])
            if _spatial_material:
                material_qp = fem.inner(material, s)
                E_qp = scalar_dtype(material_qp[0])
                nu_qp = scalar_dtype(material_qp[1])
                mu = E_qp / (scalar_dtype(2.0) * (scalar_dtype(1.0) + nu_qp))
                kappa = E_qp / (scalar_dtype(3.0) * (scalar_dtype(1.0) - scalar_dtype(2.0) * nu_qp))

            # Deformation gradient and PK1 stress
            grad_u = fem.grad(u, s)
            grad_v = fem.grad(v, s)
            F = wp.identity(n=n, dtype=scalar_dtype) + grad_u
            J = wp.determinant(F)
            F_inv_T = wp.transpose(wp.inverse(F))

            I1 = wp.ddot(F, F)
            iso = mu * wp.pow(J, _neg_two_thirds)
            P = iso * F + (kappa * wp.log(J) - iso * I1 * _one_third) * F_inv_T

            # --- Volumetric term ---
            # correction=False: phi_weight * P:grad_v  (standalone)
            # correction=True:  (phi_weight - 1) * P:grad_v  (added to standard)
            # sharp_volumetric: smoothed Heaviside H_s(phi) = 0.5*(1+tanh(k*(phi-0.5)))
            # replaces discontinuous branching while preserving the compute graph.
            if _sharp_vol:
                _hs = _half * (_one + wp.tanh(_sharp_k * (phi - _half)))
                phi_weight = _one_minus_floor * _hs * phi + _phi_floor - _vol_offset
            else:
                phi_weight = _one_minus_floor * phi + _phi_floor - _vol_offset
            result = wp.ddot(P, grad_v) * phi_weight
            local_body = body_const
            if _spatial_material:
                local_body = local_body + scalar_dtype(material_qp[2]) * acceleration_const
            result = result - body_load_scale * phi_weight * wp.dot(fem.inner(v, s), local_body)

            # --- Dirichlet Nitsche terms ---
            # Skip BC terms at exterior QPs (phi < threshold) where stiffness
            # is negligible — prevents unresisted penalty/consistency forces.
            psi_D_raw = psi_D_arr[qp]
            psi_D = vec3_t(
                scalar_dtype(psi_D_raw[0]),
                scalar_dtype(psi_D_raw[1]),
                scalar_dtype(psi_D_raw[2]),
            )
            if psi_D[0] > 0.5 or psi_D[1] > 0.5 or psi_D[2] > 0.5:
                if phi > _phi_thr:
                    # Field values at QP
                    u_at_qp = fem.inner(u, s)  # interpolated displacement (vec3_t)
                    v_at_qp = fem.inner(v, s)  # test function value (vec3_t)
                    ub_raw = u_bar_arr[qp]
                    u_bar_v = vec3_t(
                        scalar_dtype(ub_raw[0]),
                        scalar_dtype(ub_raw[1]),
                        scalar_dtype(ub_raw[2]),
                    )
                    u_minus_ubar = u_at_qp - u_bar_v
                    u_minus_ubar_D = wp.cw_mul(psi_D, u_minus_ubar)
                    v_at_qp_D = wp.cw_mul(psi_D, v_at_qp)

                    # Consistency: (P * grad_phi) . (psi_D * v)
                    P_gp = P * gp  # mat33 * vec3 -> vec3
                    consistency = wp.dot(P_gp, v_at_qp_D)

                    # Symmetry: (dP/dF[grad_v] * grad_phi) . (u - u_bar)
                    dF = grad_v
                    tr_Finv_dF = wp.ddot(F_inv_T, dF)
                    J_m23 = wp.pow(J, _neg_two_thirds)
                    lnJ = wp.log(J)
                    dJ23 = _neg_two_thirds * J_m23 * tr_Finv_dF
                    dI1 = _two * wp.ddot(F, dF)
                    dF_inv_T = -(F_inv_T * wp.transpose(dF) * F_inv_T)
                    third_I1 = I1 * _one_third
                    A = F - third_I1 * F_inv_T
                    dA = dF - (dI1 * _one_third) * F_inv_T - third_I1 * dF_inv_T
                    dP_v = mu * (dJ23 * A + J_m23 * dA) + kappa * (
                        tr_Finv_dF * F_inv_T + lnJ * dF_inv_T
                    )
                    dP_v_gp = dP_v * gp
                    symmetry = wp.dot(dP_v_gp, u_minus_ubar_D)

                    result = result + consistency + _sym_sign * symmetry

                    # Penalty acts only on requested displacement components.
                    penalty = (
                        beta * wp.dot(u_minus_ubar_D, v_at_qp_D) * grad_phi_mag
                    )
                    result = result + penalty

            # --- Neumann Nitsche term: -(t_bar . v) * (-grad_phi . n_face) ---
            # Use directional derivative -∇φ·n_face instead of |∇φ| to avoid
            # overcounting at edges/corners where multiple faces share QPs.
            psi_N = psi_N_arr[qp]
            if psi_N > 0.5:
                if phi > _phi_thr:
                    v_at_qp_n = fem.inner(v, s)
                    tb_raw = t_bar_arr[qp]
                    t_bar_v = vec3_t(
                        scalar_dtype(tb_raw[0]),
                        scalar_dtype(tb_raw[1]),
                        scalar_dtype(tb_raw[2]),
                    )
                    nf_raw = n_face_arr[qp]
                    nf = vec3_t(
                        scalar_dtype(nf_raw[0]),
                        scalar_dtype(nf_raw[1]),
                        scalar_dtype(nf_raw[2]),
                    )
                    # -∇φ·n_face > 0 on the correct face (φ decreases outward)
                    dphi_dn = wp.max(scalar_dtype(0.0), -wp.dot(gp, nf))
                    result = result - wp.dot(t_bar_v, v_at_qp_n) * dphi_dn

            return result

        return nitsche_stress

    @staticmethod
    def _make_nitsche_tangent(
        n: int,
        scalar_dtype=wp.float32,
        phi_floor=1e-4,
        correction=False,
        symmetry_sign=1.0,
        sharp_volumetric=False,
        nitsche_phi_threshold=0.0,
        spatial_material=False,
    ):
        """Build a Nitsche tangent (stiffness) integrand.

        When ``correction=True`` the volumetric dP term uses
        ``(phi_weight - 1)`` for the two-integration split approach.

        ``symmetry_sign``: +1.0 (standard symmetric Nitsche, default — K symmetric,
        coercive for β ≥ β_min), -1.0 (non-symmetric Nitsche — K non-symmetric,
        coercive for any β > 0), 0.0 (no symmetry term).

        ``nitsche_phi_threshold``: see ``_make_nitsche_stress``.

        The d²P/dF² Hessian block is controlled at runtime via the
        ``skip_d2P_flag`` integrand parameter (wp.int32, passed through
        values dict): 0 = compute d²P (exact Newton), 1 = skip (approximate,
        ~38% faster tangent with identical accuracy/convergence).
        """
        _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
        _one_third = scalar_dtype(1.0 / 3.0)
        _two = scalar_dtype(2.0)
        _half = scalar_dtype(0.5)
        _one = scalar_dtype(1.0)
        _sharp_k = scalar_dtype(100.0)
        _phi_floor = scalar_dtype(phi_floor)
        _one_minus_floor = scalar_dtype(1.0 - phi_floor)
        _vol_offset = scalar_dtype(1.0) if correction else scalar_dtype(0.0)
        _sym_sign = scalar_dtype(symmetry_sign)
        _sharp_vol = sharp_volumetric
        _phi_thr = scalar_dtype(nitsche_phi_threshold)
        _spatial_material = spatial_material
        vec3_t = wp.types.vector(length=3, dtype=scalar_dtype)

        @fem.integrand
        def nitsche_tangent(
            s: fem.Sample,
            v: fem.Field,
            u: fem.Field,
            du: fem.Field,
            e: fem.Field,
            material: fem.Field,
            mu: scalar_dtype,
            kappa: scalar_dtype,
            phi_arr: wp.array(dtype=wp.float32),
            grad_phi_arr: wp.array(dtype=wp.vec3),
            grad_phi_mag_arr: wp.array(dtype=wp.float32),
            beta_arr: wp.array(dtype=wp.float32),
            u_bar_arr: wp.array(dtype=wp.vec3),
            t_bar_arr: wp.array(dtype=wp.vec3),
            psi_D_arr: wp.array(dtype=wp.vec3),
            psi_N_arr: wp.array(dtype=wp.float32),
            n_face_arr: wp.array(dtype=wp.vec3),
            qp_active_arr: wp.array(dtype=wp.int32),
            skip_d2P_flag: wp.int32,
        ):
            qp = s.qp_index

            # Early-return for inactive (zero-weight) QPs
            if qp_active_arr[qp] == 0:
                return scalar_dtype(0.0)

            phi = scalar_dtype(phi_arr[qp])
            gp_raw = grad_phi_arr[qp]
            gp = vec3_t(
                scalar_dtype(gp_raw[0]),
                scalar_dtype(gp_raw[1]),
                scalar_dtype(gp_raw[2]),
            )
            grad_phi_mag = scalar_dtype(grad_phi_mag_arr[qp])
            beta = scalar_dtype(beta_arr[qp])
            if _spatial_material:
                material_qp = fem.inner(material, s)
                E_qp = scalar_dtype(material_qp[0])
                nu_qp = scalar_dtype(material_qp[1])
                mu = E_qp / (scalar_dtype(2.0) * (scalar_dtype(1.0) + nu_qp))
                kappa = E_qp / (scalar_dtype(3.0) * (scalar_dtype(1.0) - scalar_dtype(2.0) * nu_qp))

            # Deformation gradient
            grad_u = fem.grad(u, s)
            F = wp.identity(n=n, dtype=scalar_dtype) + grad_u
            J = wp.determinant(F)
            F_inv = wp.inverse(F)
            F_inv_T = wp.transpose(F_inv)

            # H1 = dF (trial increment), H2 = grad_v (test gradient)
            dF = fem.grad(du, s)
            grad_v = fem.grad(v, s)

            I1 = wp.ddot(F, F)
            J_m23 = wp.pow(J, _neg_two_thirds)
            lnJ = wp.log(J)
            third_I1 = I1 * _one_third
            A = F - third_I1 * F_inv_T

            # --- dP/dF[dF] (H1 = dF = grad_du) ---
            tr1 = wp.ddot(F_inv_T, dF)
            dJ23_1 = _neg_two_thirds * J_m23 * tr1
            dI1_1 = _two * wp.ddot(F, dF)
            G1 = -(F_inv_T * wp.transpose(dF) * F_inv_T)
            dA_1 = dF - (dI1_1 * _one_third) * F_inv_T - third_I1 * G1
            dP = mu * (dJ23_1 * A + J_m23 * dA_1) + kappa * (tr1 * F_inv_T + lnJ * G1)

            # --- Volumetric tangent ---
            # sharp_volumetric: smoothed Heaviside (same as stress integrand).
            if _sharp_vol:
                _hs = _half * (_one + wp.tanh(_sharp_k * (phi - _half)))
                phi_weight = _one_minus_floor * _hs * phi + _phi_floor - _vol_offset
            else:
                phi_weight = _one_minus_floor * phi + _phi_floor - _vol_offset
            result = wp.ddot(dP, grad_v) * phi_weight

            # --- Dirichlet Nitsche tangent ---
            # Skip at exterior QPs (phi < threshold) — see stress integrand.
            psi_D_raw = psi_D_arr[qp]
            psi_D = vec3_t(
                scalar_dtype(psi_D_raw[0]),
                scalar_dtype(psi_D_raw[1]),
                scalar_dtype(psi_D_raw[2]),
            )
            if psi_D[0] > 0.5 or psi_D[1] > 0.5 or psi_D[2] > 0.5:
                if phi > _phi_thr:
                    du_at_qp = fem.inner(du, s)
                    v_at_qp = fem.inner(v, s)
                    du_at_qp_D = wp.cw_mul(psi_D, du_at_qp)
                    v_at_qp_D = wp.cw_mul(psi_D, v_at_qp)

                    # Consistency projected onto requested test components.
                    consistency = wp.dot(dP * gp, v_at_qp_D)

                    # --- dP/dF[grad_v] (H2 = grad_v) ---
                    tr2 = wp.ddot(F_inv_T, grad_v)
                    dJ23_2 = _neg_two_thirds * J_m23 * tr2
                    dI1_2 = _two * wp.ddot(F, grad_v)
                    G2 = -(F_inv_T * wp.transpose(grad_v) * F_inv_T)
                    dA_2 = grad_v - (dI1_2 * _one_third) * F_inv_T - third_I1 * G2
                    dP_v = mu * (dJ23_2 * A + J_m23 * dA_2) + kappa * (
                        tr2 * F_inv_T + lnJ * G2
                    )

                    # Symmetry: (dP/dF[grad_v] * grad_phi) . du
                    symmetry = wp.dot(dP_v * gp, du_at_qp_D)

                    result = result + consistency + _sym_sign * symmetry

                    # d²P/dF²[grad_du, grad_v] correction (exact Newton for symmetry)
                    if skip_d2P_flag == 0:
                        u_at_qp = fem.inner(u, s)
                        ub_raw = u_bar_arr[qp]
                        u_bar_v = vec3_t(
                            scalar_dtype(ub_raw[0]),
                            scalar_dtype(ub_raw[1]),
                            scalar_dtype(ub_raw[2]),
                        )
                        u_minus_ubar = u_at_qp - u_bar_v
                        u_minus_ubar_D = wp.cw_mul(psi_D, u_minus_ubar)

                        B1 = F_inv_T * wp.transpose(dF)
                        B2 = F_inv_T * wp.transpose(grad_v)
                        dG1_dH2 = (B2 * B1 + B1 * B2) * F_inv_T

                        G2_H1 = wp.ddot(G2, dF)
                        H2_H1 = wp.ddot(grad_v, dF)

                        # Isochoric: mu * J^{-2/3} * [...]
                        d2P = (
                            mu
                            * J_m23
                            * (
                                _neg_two_thirds
                                * (
                                    _neg_two_thirds * tr1 * tr2 * A
                                    + G2_H1 * A
                                    + tr1 * dA_2
                                    + tr2 * dA_1
                                )
                                + _neg_two_thirds * H2_H1 * F_inv_T
                                - _one_third * dI1_1 * G2
                                - _one_third * dI1_2 * G1
                                - third_I1 * dG1_dH2
                            )
                        )
                        # Volumetric: kappa * [...]
                        d2P = d2P + kappa * (
                            G2_H1 * F_inv_T + tr1 * G2 + tr2 * G1 + lnJ * dG1_dH2
                        )
                        result = result + _sym_sign * wp.dot(
                            d2P * gp, u_minus_ubar_D
                        )

                    # Penalty acts only on requested trial/test components.
                    penalty = beta * wp.dot(du_at_qp_D, v_at_qp_D) * grad_phi_mag
                    result = result + penalty

            # No Neumann terms in tangent (load-independent)
            return result

        return nitsche_tangent

    @staticmethod
    def _make_hidden_phi_body_force_integrand(
        n: int,
        B_vals: tuple[float, ...],
        scalar_dtype=wp.float32,
        phi_floor=1e-4,
        spatial_material=False,
        acceleration: tuple[float, ...] = (),
    ):
        """Build differentiable interior ``w(e_phi) * b.v`` for hidden phi."""
        components = [float(B_vals[i]) if i < len(B_vals) else 0.0 for i in range(n)]
        vec_type = wp.types.vector(length=n, dtype=scalar_dtype)
        B_const = wp.constant(vec_type(*components))
        a_components = [
            float(acceleration[i]) if i < len(acceleration) else 0.0
            for i in range(n)
        ]
        a_const = wp.constant(vec_type(*a_components))
        _one_minus_floor = scalar_dtype(1.0 - phi_floor)
        _phi_floor = scalar_dtype(phi_floor)

        if spatial_material:
            @fem.integrand
            def hidden_phi_body_force(
                s: fem.Sample, v: fem.Field, e: fem.Field, material: fem.Field
            ):
                weight = _one_minus_floor * e(s) + _phi_floor
                rho = scalar_dtype(fem.inner(material, s)[2])
                return weight * wp.dot(fem.inner(v, s), B_const + rho * a_const)
        else:
            @fem.integrand
            def hidden_phi_body_force(s: fem.Sample, v: fem.Field, e: fem.Field):
                weight = _one_minus_floor * e(s) + _phi_floor
                return weight * wp.dot(fem.inner(v, s), B_const)

        return hidden_phi_body_force

    @staticmethod
    def _trim_exterior_elements(octree_mesh, epsilon, device, phi_threshold=1e-3):
        """Remove elements fully outside the geometry and compact node arrays.

        Modifies *octree_mesh* dict in-place: removes exterior elements,
        filters hanging-node constraints whose masters disappear, removes
        orphaned nodes, and remaps all node indices so the trimmed mesh is
        fully self-contained with no gaps.

        All operations are GPU-parallel (no Python loops over elements/nodes).

        Args:
            octree_mesh: dict with keys ``nodes``, ``elements``, ``sdf``/``node_sdf``,
                ``edge_hanging``, ``face_hanging``.
            epsilon: phase-field epsilon for computing phi from SDF.
            device: torch device string.
            phi_threshold: elements where **max** nodal phi < this are removed
                (all nodes have negligible phase-field value).
        """
        sdf_key = "sdf" if "sdf" in octree_mesh else "node_sdf"
        sdf_raw = octree_mesh[sdf_key]
        sdf = torch.as_tensor(sdf_raw, dtype=torch.float32, device=device)

        # Compute phi from SDF: phi = 0.5 * (1 - tanh(sdf / epsilon))
        phi = (0.5 * (1.0 - torch.tanh(sdf / epsilon))).clamp(0.0, 1.0)

        elements_t = torch.as_tensor(
            octree_mesh["elements"], dtype=torch.int32, device=device
        )
        elem_phi = phi[elements_t.long()]  # (n_elems, 8)
        max_phi = elem_phi.max(dim=1).values  # (n_elems,)
        keep_mask = max_phi >= phi_threshold  # (n_elems,) bool

        n_before = elements_t.shape[0]
        elements_t = elements_t[keep_mask]
        n_after = elements_t.shape[0]

        # --- Filter hanging-node constraints (before node compaction) ---
        n_nodes_old = sdf.shape[0]
        active = torch.zeros(n_nodes_old, dtype=torch.bool, device=device)
        active[elements_t.long().reshape(-1)] = True

        for key, n_masters in [("edge_hanging", 2), ("face_hanging", 4)]:
            h = octree_mesh.get(key, None)
            if h is None or (hasattr(h, "__len__") and len(h) == 0):
                continue
            h_t = torch.as_tensor(h, dtype=torch.int32, device=device)
            slave_ids = h_t[:, 0].long()
            master_ids = h_t[:, 1 : 1 + n_masters].long()
            # Drop constraints where the slave itself is not in any active element.
            # A phantom slave (fine element trimmed but constraint kept) reintroduces
            # a disconnected DOF into the inner mesh, creating a zero row/col in K_std.
            # The zero row causes dynamic BSR insertion at the tape Newton step
            # (SDF-dependent quadrature weights activate a previously-zero contribution),
            # staling scatter_mapping and making K_ext asymmetric.
            keep_rows = active[slave_ids] & active[master_ids].all(dim=1)
            octree_mesh[key] = h_t[keep_rows]

        # --- Collect all referenced nodes (elements + surviving constraints) ---
        active.zero_()
        active[elements_t.long().reshape(-1)] = True
        for key in ("edge_hanging", "face_hanging"):
            h = octree_mesh.get(key, None)
            if h is None or (hasattr(h, "__len__") and len(h) == 0):
                continue
            h_t = torch.as_tensor(h, dtype=torch.int32, device=device)
            active[h_t.long().reshape(-1)] = True

        # --- Build old→new index map and compact arrays ---
        # new_id[old_id] = new contiguous index, or -1 if removed
        new_id = torch.full((n_nodes_old,), -1, dtype=torch.int32, device=device)
        active_indices = active.nonzero(as_tuple=False).squeeze(1)
        new_id[active_indices] = torch.arange(
            active_indices.shape[0], dtype=torch.int32, device=device
        )

        # Remap elements
        octree_mesh["elements"] = new_id[elements_t.long()]

        # Remap hanging-node constraints
        for key in ("edge_hanging", "face_hanging"):
            h = octree_mesh.get(key, None)
            if h is None or (hasattr(h, "__len__") and len(h) == 0):
                continue
            h_t = torch.as_tensor(h, dtype=torch.int32, device=device)
            octree_mesh[key] = new_id[h_t.long()]

        # Compact node-indexed arrays
        nodes_t = torch.as_tensor(
            octree_mesh["nodes"], dtype=torch.float32, device=device
        )
        octree_mesh["nodes"] = nodes_t[active_indices]
        octree_mesh[sdf_key] = sdf[active_indices]
        # Store mapping: new_idx → old_idx (for ShapeOptTrimmed gradient expansion)
        octree_mesh["trim_active_indices"] = active_indices

        return n_before, n_after

    def _build_mesh_state(self):
        """Build mesh state with Nitsche-specific extensions.

        1. Save original BCs, replace with outer surface Dirichlet u=0
        2. Optionally trim exterior elements
        3. Call super()._build_mesh_state() for octree mesh setup
        4. Compute phase field from SDF
        5. Build adaptive quadrature from quad_octree_mesh
        6. Pre-compute QP fields (phi, grad_phi, u_bar, t_bar, psi_D, psi_N)
        7. Compute stabilization parameter beta
        """
        # The parent creates the new geometry/spaces and initializes its solver
        # cache before this method has built the matching Nitsche quadratures.
        # Mark design-dependent body assembly unavailable up front so dynamic
        # dispatch cannot pair the new test field with an old quadrature/probe.
        self._nitsche_body_state_ready = False

        # --- Save original BCs and replace with outer surface Dirichlet u=0 ---
        self._original_dirichlet_bcs_dict = self.params.get("dirichlet_bcs_dict", {})
        self._original_neumann_bcs_dict = self.params.get("neumann_bcs_dict", {})

        # --- Optionally trim exterior elements ---
        # sharp_volumetric requires trim_exterior (full stiffness, no phi weighting)
        if self.params.get("sharp_volumetric", False):
            self.params["trim_exterior"] = True
        if self.params.get("trim_exterior", False):
            epsilon = float(self.params.phase_field_epsilon)
            _phi_thresh = float(self.params.get("trim_phi_threshold", 1e-3))
            n_before, n_after = self._trim_exterior_elements(
                self.params.octree_mesh,
                epsilon,
                self.device,
                phi_threshold=_phi_thresh,
            )
            self.log(
                f"Trimmed exterior elements: {n_before} -> {n_after} "
                f"(removed {n_before - n_after})"
            )
            # After trimming, no hard Dirichlet BCs — all boundary conditions
            # are enforced purely through Nitsche penalty/consistency terms.
            self.params["dirichlet_bcs_dict"] = {}
        else:
            # Replace BCs: fixed u=0 on all 6 outer surfaces of the domain.
            # Derive bounds from actual mesh nodes (works for both octree and
            # user-supplied meshes that may not be centered at origin).
            _nodes_raw = self.params.octree_mesh.get("nodes")
            if _nodes_raw is not None:
                _nt = torch.as_tensor(_nodes_raw, dtype=torch.float32)
                _lo = _nt.min(dim=0).values  # (3,)
                _hi = _nt.max(dim=0).values  # (3,)
            else:
                _dh = float(self.params.octree_mesh.get("domain_length", 2.0)) / 2.0
                _lo = torch.tensor([-_dh, -_dh, -_dh])
                _hi = torch.tensor([_dh, _dh, _dh])
            _xlo, _ylo, _zlo = float(_lo[0]), float(_lo[1]), float(_lo[2])
            _xhi, _yhi, _zhi = float(_hi[0]), float(_hi[1]), float(_hi[2])
            self.params["dirichlet_bcs_dict"] = {
                lambda x, y, z: (
                    torch.isclose(x, torch.full_like(x, _xlo))
                    | torch.isclose(x, torch.full_like(x, _xhi))
                    | torch.isclose(y, torch.full_like(y, _ylo))
                    | torch.isclose(y, torch.full_like(y, _yhi))
                    | torch.isclose(z, torch.full_like(z, _zlo))
                    | torch.isclose(z, torch.full_like(z, _zhi))
                ): 0.0,
            }
        self.params["neumann_bcs_dict"] = {}

        _pf = float(self.params.get("phi_floor", 1e-4))
        self._phi_floor = _pf

        # --- Call parent to build octree mesh, function spaces, solver ---
        super()._build_mesh_state()

        # --- Phase field from SDF ---
        octree = self.params.octree_mesh
        sdf_raw = octree.get("sdf", octree.get("node_sdf"))
        assert sdf_raw is not None, "octree_mesh must have key 'sdf' or 'node_sdf'"
        sdf = torch.as_tensor(sdf_raw, dtype=torch.float32, device=self.device)
        self._sdf_nodal = sdf

        # Finest leaf edge, read off the node lattice of the mesh actually
        # handed over (as _get_or_build_mc_surface_physical does).  Not
        # domain_length / 2**depth: the octree dict need not carry the cube
        # side (ShapeOptTrimmed never sets it), and log2 of a clipped or
        # trimmed mesh's node count can fall a level short of the depth.
        span = self._domain_span
        npa_t = torch.tensor(
            list(self._prev_octree_npa), dtype=span.dtype, device=span.device
        )
        min_element_size = float((span / (npa_t - 1).clamp(min=1)).min())
        self._min_element_size = min_element_size

        epsilon = float(self.params.phase_field_epsilon)
        self._epsilon = epsilon

        phi_nodal = (0.5 * (1.0 - torch.tanh(sdf / epsilon))).clamp(0.0, 1.0)
        self._phi_nodal = phi_nodal

        # --- Resolve SurfaceSelector BC keys (before adaptive quadrature) ---
        self._resolve_surface_selector_bcs()

        # --- Clear stale BC base caches (sizes change across reinits) ---
        for attr in (
            "_u_bar_qp_base_torch",
            "_t_bar_qp_base_torch",
            "_chunk_bc_bases",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

        # --- Build adaptive quadrature ---
        self._build_adaptive_quadrature()
        # --- Build Nitsche integrands ---
        # correction=False → standalone (no vol_offset subtraction).
        _sd = self._integrand_scalar_dtype
        _pf = self._phi_floor
        # symmetry_sign: +1 (default, symmetric Nitsche), -1 (non-symmetric Nitsche), 0 (no symmetry term)
        _sym_mode = self.params.get("_symmetry_sign_mode", "default")
        _sym_sign = {"default": 1.0, "sym_neg": -1.0, "no_sym": 0.0}.get(_sym_mode, 1.0)
        _sharp_vol = bool(self.params.get("sharp_volumetric", False))
        _npt = float(self.params.get("nitsche_phi_threshold", _pf))
        _B = tuple(self.params.get("B", (0.0,) * self.dim))
        self._boundary_stress_integrand = self._make_nitsche_stress(
            self.dim,
            _B,
            scalar_dtype=_sd,
            phi_floor=_pf,
            correction=False,
            symmetry_sign=_sym_sign,
            sharp_volumetric=_sharp_vol,
            nitsche_phi_threshold=_npt,
            spatial_material=self._has_spatial_material,
            acceleration=tuple(self.params.get("material_acceleration", (0.0,) * self.dim)),
        )
        self._boundary_tangent_integrand = self._make_nitsche_tangent(
            self.dim,
            scalar_dtype=_sd,
            phi_floor=_pf,
            correction=False,
            symmetry_sign=_sym_sign,
            sharp_volumetric=_sharp_vol,
            nitsche_phi_threshold=_npt,
            spatial_material=self._has_spatial_material,
        )
        self._hidden_phi_body_force_integrand = (
            self._make_hidden_phi_body_force_integrand(
                self.dim,
                _B,
                scalar_dtype=_sd,
                phi_floor=_pf,
                spatial_material=self._has_spatial_material,
                acceleration=tuple(self.params.get("material_acceleration", (0.0,) * self.dim)),
            )
        )
        # --- Build standard quadrature ---
        self._build_standard_quadrature()
        # --- Pre-compute QP fields ---
        self._precompute_qp_fields()
        # --- Build BC mask fields (u_bar, t_bar, psi_D, psi_N) ---
        self._build_nitsche_bc_masks()
        # --- Assembly-only state: skip on wrappers that never call Nitsche
        # integrate (e.g. the outer ShapeOptTrimmed). beta is the ~0.6 s
        # eigenvalue computation; correction_dom + save_quad_vtu are cheap
        # but useless without beta.  Gated on a per-instance flag so the
        # default (inner solver) path is unchanged.
        _skip_asm = bool(getattr(self, "_skip_nitsche_assembly", False))

        # --- Compute stabilization parameter beta ---
        if not _skip_asm:
            self._compute_stabilization_beta(use_current_u=False)
        # --- Build correction Subdomain + ExplicitQuadrature for the split ---
        if not _skip_asm:
            self._build_correction_domain()
            # Needs the correction quadrature, so it follows it.  Factors are
            # computed on the first build only (see there).
            self._rescale_neumann_tractions()
        # --- Save quad mesh VTU ---
        if not _skip_asm:
            self._save_quad_mesh_vtu()

        # ShapeOpt.reinit_with_sdf() must first replace its function-space-
        # dependent e_phi field.  In all other solve-capable builds, the
        # current Nitsche state is now complete and the forward cache can be
        # assembled exclusively from current-mesh objects.
        if not _skip_asm and not getattr(
            self, "_defer_body_cache_until_design_field", False
        ):
            self._nitsche_body_state_ready = True
            self._refresh_body_force_cache()

    def _rebuild_nitsche_state(self):
        """Rebuild all SDF-dependent Nitsche state for optimization iterations.

        Assumes ``_sdf_nodal`` and ``_phi_nodal`` are already updated.
        Rebuilds: adaptive quadrature, standard quadrature, QP fields
        (phi/grad_phi), BC masks, stabilization beta, and correction domains.
        """
        # Invalidate the forward load before replacing any quadrature fields.
        self._nitsche_body_state_ready = False
        self._mesh_cache["body"] = None

        # Clear cached BC base values — they depend on the old QP array sizes
        # and will be re-cached on the next _scale_nitsche_bc_fields() call.
        for attr in (
            "_u_bar_qp_base_torch",
            "_t_bar_qp_base_torch",
            "_chunk_bc_bases",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

        self._build_adaptive_quadrature()
        self._build_standard_quadrature()
        self._precompute_qp_fields()
        self._build_nitsche_bc_masks()
        self._compute_stabilization_beta(use_current_u=False)
        self._build_correction_domain()
        self._rescale_neumann_tractions()
        self._nitsche_body_state_ready = True
        self._refresh_body_force_cache()

    # ------------------------------------------------------------------
    # Surface-selector BC resolution
    # ------------------------------------------------------------------

    def _get_or_build_mc_surface_physical(self):
        """Return (verts_phys (V,3), faces (F,3)) for SurfaceSelector resolution.

        Base-class implementation: scatter nodal SDF onto a regular grid
        inferred from ``_prev_octree_npa`` and ``_domain_lo``/``_domain_span``,
        then run marching cubes.

        ``ShapeOptTrimmed`` overrides this to reuse the already-cached
        ``_mc_verts_norm`` / ``_mc_faces`` at zero extra cost.
        """
        from .analytical_sdf import extract_surface, build_warp_mesh

        nodes = self._prev_octree_nodes          # (V, 3) float32
        sdf = self._sdf_nodal                    # (V,) float32
        device = str(sdf.device)
        npa = self._prev_octree_npa              # (npa_x, npa_y, npa_z)
        lo = self._domain_lo.to(dtype=torch.float32, device=sdf.device)
        span = self._domain_span.to(dtype=torch.float32, device=sdf.device)

        # Per-axis grid spacing for the inferred full regular grid
        npa_t = torch.tensor(list(npa), dtype=torch.float32, device=sdf.device)
        h = span / (npa_t - 1).clamp(min=1)

        # Scatter each node onto its grid cell; min SDF wins conflicts
        idx = ((nodes - lo.unsqueeze(0)) / h.unsqueeze(0).clamp(min=1e-10)).round().long()
        idx_max = (npa_t - 1).long()
        idx = torch.max(torch.zeros_like(idx), torch.min(idx, idx_max.unsqueeze(0)))
        flat_idx = idx[:, 0] * (npa[1] * npa[2]) + idx[:, 1] * npa[2] + idx[:, 2]

        grid_flat = torch.full(
            (npa[0] * npa[1] * npa[2],),
            float(span.max()),
            dtype=torch.float32,
            device=sdf.device,
        )
        grid_flat.scatter_reduce_(0, flat_idx, sdf, reduce="amin", include_self=True)
        grid_sdf = grid_flat.reshape(npa[0], npa[1], npa[2])
        from .bc_surface_segmentation import SignedDistanceGrid
        self._bc_sdf_grid_context = SignedDistanceGrid(
            values=grid_sdf, origin=lo, spacing=h,
        )

        # MC in a cubic domain of side = max span (centers output at [-L/2, L/2])
        domain_len = float(span.max())
        verts_norm, faces, probe_norm = extract_surface(
            grid_sdf, domain_length=domain_len, device=device
        )

        # Map [-L/2, L/2]^3 → physical coords with per-axis scaling
        center = (lo + span / 2.0).float()
        axis_scale = (span / domain_len).float()
        verts_phys = verts_norm * axis_scale.unsqueeze(0) + center.unsqueeze(0)
        probe_phys = probe_norm * axis_scale + center

        bbox_diag = float(
            torch.linalg.norm(verts_phys.amax(0) - verts_phys.amin(0))
        )
        wp_mesh, negate, keep_alive = build_warp_mesh(
            verts_phys, faces, probe_phys, bbox_diag, device=device
        )
        self._mc_wp_mesh = wp_mesh
        self._mc_wp_negate = negate
        self._mc_wp_keep_alive = keep_alive
        self._mc_faces = faces
        self._mc_bbox_diag_phys = bbox_diag
        return verts_phys, faces

    def _surface_selector_sdf_grid(self):
        """Return the retained physical SDF grid used to make the MC surface."""
        cached = getattr(self, "_bc_sdf_grid_context", None)
        if cached is not None:
            return cached
        if not hasattr(self, "_sussman_grid_sdf"):
            return None
        from .bc_surface_segmentation import SignedDistanceGrid
        shape = torch.tensor(
            list(self._sussman_grid_sdf.shape), dtype=torch.float64,
            device=self._sussman_grid_sdf.device,
        )
        if hasattr(self, "_normalized_to_physical"):
            corners = torch.tensor(
                [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                dtype=torch.float64, device=self._sussman_grid_sdf.device,
            )
            physical = self._normalized_to_physical(corners).float()
            origin = physical[0]
            spacing = (physical[1] - physical[0]) / (shape.float() - 1.0)
        else:
            return None
        self._bc_sdf_grid_context = SignedDistanceGrid(
            self._sussman_grid_sdf, origin, spacing
        )
        return self._bc_sdf_grid_context

    def _resolve_surface_selector_bcs(self):
        """Resolve any SurfaceSelector BC keys to BVH distance predicates.

        Fast-path no-op when all BC keys are regular lambdas or already
        resolved.  Must be called after ``_epsilon`` and ``_min_element_size``
        are set, and **before** ``_build_adaptive_quadrature`` (which calls
        ``_classify_bc_elements``, which iterates BC specs).
        """
        from .bc_surface_selectors import SurfaceSelector as _SS

        all_keys = (
            list(self._original_dirichlet_bcs_dict)
            + list(self._original_neumann_bcs_dict)
        )
        unresolved = [k for k in all_keys if isinstance(k, _SS) and not k.is_resolved]
        if not unresolved:
            return  # fast path — zero overhead for lambda-only dicts

        eps_tol = self._bc_match_tolerance()
        verts_phys, faces = self._get_or_build_mc_surface_physical()
        sdf_grid = self._surface_selector_sdf_grid()
        for sel in unresolved:
            sel.resolve(
                verts_phys, faces, eps_expansion=eps_tol, sdf_grid=sdf_grid,
            )
        self.log(
            f"  BC selectors resolved: {len(unresolved)} selector(s), "
            f"surface {verts_phys.shape[0]}V {faces.shape[0]}F"
        )
        for sel in unresolved:
            n_faces = int(sel._selected_face_mask.sum())
            if sel._patch_lo is None:
                self.log(f"  {sel.__class__.__name__}: 0 faces")
                continue
            lo = sel._patch_lo.cpu().float()
            hi = sel._patch_hi.cpu().float()
            self.log(f"  {sel.__class__.__name__}: {n_faces} faces, "
                     f"lo=({lo[0]:.4f}, {lo[1]:.4f}, {lo[2]:.4f})  "
                     f"hi=({hi[0]:.4f}, {hi[1]:.4f}, {hi[2]:.4f})")

    def _build_adaptive_quadrature(self):
        """Build ExplicitQuadrature with refined QPs in diffused boundary.

        When ``geometric_func`` is provided, uses adaptive octree refinement
        (via ``hierarchical_extraction``) to subdivide the domain near the
        surface.  This gives high-fidelity phi / grad_phi at quadrature points
        with far fewer sub-cells than uniform refinement.

        Falls back to the old uniform sub-cell approach when the octree state
        (``grid_sdf``, ``existing_anchors``) is not available in octree_mesh.
        """

        geometric_func = self.params.get("geometric_func", None)

        nodes_t = self._prev_octree_nodes  # coarse mesh nodes (float32 GPU)
        elements_t = self._prev_octree_elements  # coarse mesh elements (int32 GPU)
        n_coarse_elems = elements_t.shape[0]

        # Compute phi on coarse mesh nodes
        phi_nodal = self._phi_nodal

        # Identify boundary elements: elements with at least one node in the
        # phase-field transition zone (phi neither ~0 nor ~1).
        _phi_lo, _phi_hi = 1e-3, 1.0 - 1e-3
        elem_node_phi = phi_nodal[elements_t.long()]  # (n_elems, 8)
        is_boundary = ((elem_node_phi > _phi_lo) & (elem_node_phi < _phi_hi)).any(dim=1)
        self._boundary_elem_mask = is_boundary  # (n_coarse_elems,) bool
        boundary_elem_indices = torch.where(is_boundary)[0].to(dtype=torch.int32)
        self._boundary_elem_indices = boundary_elem_indices

        std_gauss_pts = torch.tensor(
            _STD_GAUSS_PTS, dtype=torch.float32, device=self.device
        )
        std_gauss_w = _STD_GAUSS_W

        if geometric_func is None:
            # No refinement: use standard quadrature for all elements
            qp_per_cell = 8
            points_t = (
                std_gauss_pts.unsqueeze(0).expand(n_coarse_elems, -1, -1).contiguous()
            )
            weights_t = torch.full(
                (n_coarse_elems, qp_per_cell),
                std_gauss_w,
                dtype=torch.float32,
                device=self.device,
            )

            self._qp_per_cell = qp_per_cell
            self._n_qp_elements = n_coarse_elems
            self._boundary_only_qps = False
            self._build_explicit_quadrature(
                points_t, weights_t, n_coarse_elems, qp_per_cell
            )
            self._qp_uses_child = False
            return

        # --- Determine refinement depth ---
        eps = self._epsilon
        boundary_idx_tmp = torch.where(is_boundary)[0]
        if boundary_idx_tmp.numel() > 0:
            bc = nodes_t[elements_t[boundary_idx_tmp].long()]
            h = (bc.max(dim=1).values - bc.min(dim=1).values).max().item()
        else:
            h = self._min_element_size

        if (
            "quad_refinement_depth" in self.params
            and self.params["quad_refinement_depth"] is not None
        ):
            ref_depth_bc = int(self.params["quad_refinement_depth"])
        else:
            ref_depth_bc = max(1, int(math.floor(math.log2(max(h / eps, 1.0)))) + 1)

        ref_depth_interior = int(
            self.params.get("quad_refinement_depth_interior", ref_depth_bc)
        )

        # --- Per-element refinement depth (BC elements get ref_depth_bc,
        #     other boundary elements get ref_depth_interior) ---
        n_boundary = int(is_boundary.sum().item())
        boundary_idx = torch.where(is_boundary)[0]
        per_elem_ref_depth = torch.full(
            (n_boundary,), ref_depth_interior, dtype=torch.long, device=self.device
        )

        if ref_depth_bc != ref_depth_interior and n_boundary > 0:
            # Classify boundary elements as BC-adjacent or interior
            is_bc_elem = self._classify_bc_elements(nodes_t, elements_t, boundary_idx)
            per_elem_ref_depth[is_bc_elem] = ref_depth_bc

        max_ref_depth = int(per_elem_ref_depth.max().item()) if n_boundary > 0 else 0

        if max_ref_depth == 0:
            # No subdivision: standard 8-pt Gauss on all elements
            qp_per_cell = 8
            points_t = (
                std_gauss_pts.unsqueeze(0).expand(n_coarse_elems, -1, -1).contiguous()
            )
            weights_t = torch.full(
                (n_coarse_elems, qp_per_cell),
                std_gauss_w,
                dtype=torch.float32,
                device=self.device,
            )
            self._qp_per_cell = qp_per_cell
            self._n_qp_elements = n_coarse_elems
            self._boundary_only_qps = False
            self._nitsche_chunks = []
            self._build_explicit_quadrature(
                points_t, weights_t, n_coarse_elems, qp_per_cell
            )
            self._qp_uses_child = False
            return

        # --- Adaptive refinement via recursive element subdivision ---
        self._build_adaptive_quadrature_recursive(
            nodes_t,
            elements_t,
            n_coarse_elems,
            is_boundary,
            std_gauss_pts,
            std_gauss_w,
            geometric_func,
            per_elem_ref_depth,
            max_ref_depth,
        )

    def _classify_bc_elements(self, nodes_t, elements_t, boundary_idx):
        """Classify boundary elements as BC-adjacent or interior.

        Returns a boolean tensor (n_boundary,) where True means the element
        has at least one node matching a Dirichlet or Neumann BC predicate.
        """
        n_boundary = boundary_idx.shape[0]
        elem_nodes = elements_t[boundary_idx].long()  # (n_boundary, 8)
        flat_node_ids = elem_nodes.reshape(-1)

        # Adjacent octree cells share most of their corners.  Evaluate every
        # distinct mesh node once, then scatter its exact Boolean result back
        # through the original cell-corner ordering.  Deduplicating integer
        # node IDs (rather than floating-point coordinates) preserves the
        # predicate and BVH inputs byte-for-byte.
        unique_node_ids, inverse = torch.unique(
            flat_node_ids, sorted=False, return_inverse=True
        )
        unique_positions = nodes_t[unique_node_ids]

        is_bc_node = torch.zeros(
            unique_positions.shape[0], dtype=torch.bool, device=self.device
        )

        eps_tol = self._bc_match_tolerance()
        _orig_isclose = torch.isclose

        def _widened_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            return _orig_isclose(
                a, b, rtol=rtol, atol=max(atol, eps_tol), equal_nan=equal_nan
            )

        torch.isclose = _widened_isclose
        try:
            for boundary_spec in self._original_dirichlet_bcs_dict:
                matched = identify_boundary_nodes(
                    boundary_spec, unique_positions, self.dim
                )
                is_bc_node[matched] = True
            for boundary_spec in self._original_neumann_bcs_dict:
                matched = identify_boundary_nodes(
                    boundary_spec, unique_positions, self.dim
                )
                is_bc_node[matched] = True
        finally:
            torch.isclose = _orig_isclose

        return is_bc_node[inverse].reshape(n_boundary, 8).any(dim=1)

    # ------------------------------------------------------------------
    # Adaptive quadrature via recursive element subdivision
    # ------------------------------------------------------------------
    @staticmethod
    def _trilinear_interp_batch(corner_phys, ref_coords):
        """Trilinear interpolation in [0,1]^3 hex reference space (warp ordering).

        Args:
            corner_phys: (N, 8, 3) physical positions of hex corners
            ref_coords: (N, 3) reference coordinates in [0,1]^3
        Returns:
            (N, 3) interpolated physical positions
        """
        xi = ref_coords[:, 0:1]
        eta = ref_coords[:, 1:2]
        zeta = ref_coords[:, 2:3]
        N = torch.stack(
            [
                (1 - xi) * (1 - eta) * (1 - zeta),  # (0,0,0)
                xi * (1 - eta) * (1 - zeta),  # (1,0,0)
                xi * eta * (1 - zeta),  # (1,1,0)
                (1 - xi) * eta * (1 - zeta),  # (0,1,0)
                (1 - xi) * (1 - eta) * zeta,  # (0,0,1)
                xi * (1 - eta) * zeta,  # (1,0,1)
                xi * eta * zeta,  # (1,1,1)
                (1 - xi) * eta * zeta,  # (0,1,1)
            ],
            dim=1,
        ).squeeze(
            -1
        )  # (N, 8)
        return (N.unsqueeze(-1) * corner_phys).sum(dim=1)

    def _build_adaptive_quadrature_recursive(
        self,
        nodes_t,
        elements_t,
        n_coarse_elems,
        is_boundary,
        std_gauss_pts,
        std_gauss_w,
        geometric_func,
        per_elem_ref_depth,
        max_ref_depth,
    ):
        """Adaptive quadrature via recursive element subdivision.

        Recursively subdivides boundary elements in [0,1]^3 reference space.
        At each depth, cells are refined if any corner has ``|sdf| < 0.5*eps``
        or if corner SDF signs are mixed (matching octree mesh generation
        criteria), AND the element's per-element refinement depth has not been
        reached.  SDF values evaluated during refinement are reused for the
        combined source mesh — no redundant ``geometric_func`` calls.

        After subdivision, boundary elements are grouped into chunks by their
        leaf-cell count to minimise padding waste in fem.integrate dispatch
        buffers.
        """
        n_boundary = int(is_boundary.sum().item())
        boundary_idx = torch.where(is_boundary)[0]  # (n_boundary,) long
        device = self.device
        eps = self._epsilon
        band = 0.5 * eps

        # --- Element-level refinement criteria ---
        elem_sdf = self._sdf_nodal[elements_t[boundary_idx].long()]  # (n_boundary, 8)
        has_band_node = (elem_sdf.abs() < band).any(dim=1)
        has_mixed_sign = (elem_sdf > 0).any(dim=1) & (elem_sdf < 0).any(dim=1)
        has_nonzero_depth = per_elem_ref_depth > 0
        needs_refinement = (has_band_node | has_mixed_sign) & has_nonzero_depth

        refined_bidx = torch.where(needs_refinement)[0]  # boundary-local
        n_refined = refined_bidx.shape[0]

        self.log(
            f"Recursive subdivision: max_ref_depth={max_ref_depth}, "
            f"{n_boundary} boundary elements "
            f"({n_refined} refined, {n_boundary - n_refined} standard Gauss)"
        )

        # Per-refined-element max depth (compact indexing)
        refined_max_depth = per_elem_ref_depth[refined_bidx]  # (n_refined,)

        # Hex corner offsets (warp ordering)
        hex_off = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.long,
            device=device,
        )

        max_grid = 2**max_ref_depth  # finest grid spacing = 1/max_grid
        gs = max_grid + 1  # nodes per dimension at finest level

        # --- SDF cache: dense tensor per refined element ---
        # sdf_cache[compact_idx, ix, iy, iz] stores SDF at ref coord
        # (ix/max_grid, iy/max_grid, iz/max_grid).  NaN = not yet evaluated.
        sdf_cache = torch.full(
            (max(n_refined, 1), gs, gs, gs),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )

        # Initialize with coarse mesh corner SDF (at grid positions {0, max_grid}^3)
        if n_refined > 0:
            for ci in range(8):
                gx, gy, gz = (hex_off[ci] * max_grid).tolist()
                sdf_cache[:n_refined, gx, gy, gz] = elem_sdf[refined_bidx, ci]

        # --- Recursive refinement loop ---
        current_lo = torch.zeros(n_refined, 3, dtype=torch.long, device=device)
        current_parent = torch.arange(n_refined, dtype=torch.long, device=device)

        final_lo_list = []
        final_span_list = []
        final_parent_list = []

        for depth in range(max_ref_depth):
            n_cells = current_lo.shape[0]
            if n_cells == 0:
                break
            span = max_grid >> depth  # cell edge in finest-grid units
            half_span = span >> 1

            # Corner positions in finest-grid integer coords
            corner_pos = (
                current_lo.unsqueeze(1) + hex_off.unsqueeze(0) * span
            )  # (n_cells, 8, 3)
            corner_sdf = sdf_cache[
                current_parent.unsqueeze(1).expand(-1, 8),
                corner_pos[:, :, 0],
                corner_pos[:, :, 1],
                corner_pos[:, :, 2],
            ]  # (n_cells, 8)

            # Refinement criteria: (band OR mixed signs) AND depth < element's max
            has_band_c = (corner_sdf.abs() < band).any(dim=1)
            has_mixed_c = (corner_sdf > 0).any(dim=1) & (corner_sdf < 0).any(dim=1)
            depth_ok = refined_max_depth[current_parent] > depth
            refine_mask = (has_band_c | has_mixed_c) & depth_ok

            keep_idx = torch.where(~refine_mask)[0]
            split_idx = torch.where(refine_mask)[0]

            # Save kept cells as final leaves
            if keep_idx.numel() > 0:
                final_lo_list.append(current_lo[keep_idx])
                final_span_list.append(
                    torch.full(
                        (keep_idx.shape[0],), span, dtype=torch.long, device=device
                    )
                )
                final_parent_list.append(current_parent[keep_idx])

            n_split = split_idx.shape[0]
            if n_split == 0:
                current_lo = current_lo[:0]  # empty — prevent double-count
                break

            # Generate 8 children per split cell
            split_lo = current_lo[split_idx]
            split_parent = current_parent[split_idx]
            child_offsets = hex_off * half_span  # (8, 3)
            children_lo = (split_lo.unsqueeze(1) + child_offsets.unsqueeze(0)).reshape(
                -1, 3
            )
            children_parent = split_parent.repeat_interleave(8)

            # Evaluate SDF at new child corners not yet in the cache
            child_corners = children_lo.unsqueeze(1) + hex_off.unsqueeze(0) * half_span
            flat_p = children_parent.unsqueeze(1).expand(-1, 8).reshape(-1)
            flat_x = child_corners[:, :, 0].reshape(-1)
            flat_y = child_corners[:, :, 1].reshape(-1)
            flat_z = child_corners[:, :, 2].reshape(-1)

            cached_vals = sdf_cache[flat_p, flat_x, flat_y, flat_z]
            needs_eval = torch.isnan(cached_vals)

            if needs_eval.any():
                eval_keys = torch.stack(
                    [
                        flat_p[needs_eval],
                        flat_x[needs_eval],
                        flat_y[needs_eval],
                        flat_z[needs_eval],
                    ],
                    dim=1,
                )
                unique_eval, _ = torch.unique(eval_keys, dim=0, return_inverse=True)

                # ref coords → physical coords via trilinear interpolation
                ref_pts = unique_eval[:, 1:4].float() / max_grid
                p_global = boundary_idx[refined_bidx[unique_eval[:, 0]]]
                p_nodes = nodes_t[elements_t[p_global].long()]
                phys_pts = self._trilinear_interp_batch(p_nodes, ref_pts)

                with torch.no_grad():
                    sdf_out = geometric_func(phys_pts.unsqueeze(0).to(torch.float16))
                    sdf_new = sdf_out.squeeze(0).squeeze(-1).to(torch.float32)

                sdf_cache[
                    unique_eval[:, 0],
                    unique_eval[:, 1],
                    unique_eval[:, 2],
                    unique_eval[:, 3],
                ] = sdf_new

            current_lo = children_lo
            current_parent = children_parent

        # Remaining cells (reached max depth) are final leaves
        if current_lo.shape[0] > 0:
            # Compute per-cell final span from parent element's max depth
            cell_max_depth = refined_max_depth[current_parent]
            cell_final_span = max_grid >> cell_max_depth.clamp(max=max_ref_depth)
            # Cells that already reached their element's max depth get span
            # equal to the grid size at that depth
            final_lo_list.append(current_lo)
            final_span_list.append(cell_final_span)
            final_parent_list.append(current_parent)

        # --- Assemble final leaf cells ---
        if final_lo_list:
            leaf_lo = torch.cat(final_lo_list, dim=0)
            leaf_span = torch.cat(final_span_list, dim=0)
            leaf_parent = torch.cat(final_parent_list, dim=0)
        else:
            leaf_lo = torch.zeros(0, 3, dtype=torch.long, device=device)
            leaf_span = torch.zeros(0, dtype=torch.long, device=device)
            leaf_parent = torch.zeros(0, dtype=torch.long, device=device)
        n_leaves = leaf_lo.shape[0]

        self.log(
            f"Recursive subdivision: {n_leaves} leaf cells in {n_refined} "
            f"refined elements (avg {n_leaves / max(n_refined, 1):.1f}/elem)"
        )

        # --- Build fine cell connectivity ---
        leaf_corners = leaf_lo.unsqueeze(1) + hex_off.unsqueeze(
            0
        ) * leaf_span.unsqueeze(1).unsqueeze(
            2
        )  # (n_leaves, 8, 3)

        # Flat key for deduplication: (parent_compact, ix, iy, iz)
        gs3 = gs * gs * gs
        gs2 = gs * gs
        flat_key = (
            leaf_parent.unsqueeze(1) * gs3
            + leaf_corners[:, :, 0] * gs2
            + leaf_corners[:, :, 1] * gs
            + leaf_corners[:, :, 2]
        )  # (n_leaves, 8)

        unique_keys, inverse = torch.unique(
            flat_key.reshape(-1), sorted=True, return_inverse=True
        )
        leaf_conn = inverse.reshape(-1, 8)  # (n_leaves, 8)

        # Decode unique keys → (parent_compact, ix, iy, iz)
        uk_parent = unique_keys // gs3
        uk_rem = unique_keys % gs3
        uk_ix = uk_rem // gs2
        uk_rem2 = uk_rem % gs2
        uk_iy = uk_rem2 // gs
        uk_iz = uk_rem2 % gs

        # SDF at unique fine nodes (already evaluated during refinement)
        fine_node_sdf = sdf_cache[uk_parent, uk_ix, uk_iy, uk_iz]

        # Physical positions via trilinear interpolation
        ref_coords = torch.stack([uk_ix, uk_iy, uk_iz], dim=1).float() / max_grid
        parent_global = boundary_idx[refined_bidx[uk_parent]]
        parent_node_phys = nodes_t[elements_t[parent_global].long()]
        fine_node_phys = self._trilinear_interp_batch(parent_node_phys, ref_coords)

        fine_node_phi = (0.5 * (1.0 - torch.tanh(fine_node_sdf / eps))).clamp(0.0, 1.0)

        del sdf_cache  # free memory

        # --- Build combined source mesh ---
        n_parent_nodes = nodes_t.shape[0]
        n_parent_elems = n_coarse_elems

        combined_nodes = torch.cat([nodes_t, fine_node_phys], dim=0)
        combined_sdf = torch.cat([self._sdf_nodal, fine_node_sdf], dim=0)

        # Persist child→parent interpolation data for differentiable path
        self._n_parent_nodes = n_parent_nodes
        self._child_parent_elem_wp = wp.from_torch(
            parent_global.to(torch.int32).contiguous(), dtype=wp.int32
        )
        self._child_ref_coords_wp = wp.from_torch(
            ref_coords.contiguous(), dtype=wp.vec3
        )
        self._parent_elements_wp = wp.from_torch(
            elements_t.to(torch.int32).contiguous(), dtype=wp.int32
        )
        # Store precomputed child SDF (from geometric_func) for tape replay
        self._child_sdf_precomputed = wp.from_torch(
            fine_node_sdf.contiguous(), dtype=wp.float32
        )
        fine_conn_global = leaf_conn + n_parent_nodes
        combined_elements = torch.cat(
            [
                elements_t.long(),
                fine_conn_global.long(),
            ],
            dim=0,
        ).to(torch.int32)

        # --- Reference coordinates for QP arrays ---
        leaf_lo_ref = leaf_lo.float() / max_grid  # (n_leaves, 3)
        leaf_span_ref = leaf_span.float() / max_grid  # (n_leaves,) scalar
        leaf_bidx = refined_bidx[leaf_parent]  # boundary-local index

        # --- Count leaf cells per boundary element ---
        counts = torch.zeros(n_boundary, dtype=torch.long, device=device)
        if n_leaves > 0:
            counts.scatter_add_(
                0,
                leaf_bidx,
                torch.ones(n_leaves, dtype=torch.long, device=device),
            )

        # Sort leaves by boundary-local parent index for sequential slot filling
        if n_leaves > 0:
            sort_order = torch.argsort(leaf_bidx)
            s_bidx = leaf_bidx[sort_order]
            s_leaf_lo_ref = leaf_lo_ref[sort_order]
            s_leaf_span_ref = leaf_span_ref[sort_order]

            group_starts = torch.cat(
                [
                    torch.tensor([0], device=device),
                    (s_bidx[1:] != s_bidx[:-1]).nonzero(as_tuple=True)[0] + 1,
                ]
            )
            offsets_in_parent = (
                torch.arange(n_leaves, device=device)
                - group_starts[
                    torch.searchsorted(
                        group_starts, torch.arange(n_leaves, device=device), right=True
                    )
                    - 1
                ]
            )
        else:
            sort_order = torch.zeros(0, dtype=torch.long, device=device)

        # --- Decide flat vs chunked integration path ---
        #
        # Flat: single subdomain over all boundary elements, uniform qp_per_cell
        #   padded to the maximum leaf count.  Simpler, one fem.integrate call.
        # Chunked: boundary elements grouped by leaf count into power-of-2
        #   buckets, each with its own subdomain.  Reduces dispatch buffer waste
        #   when max leaf count >> typical leaf count.
        #
        # Decision: estimate the flat dispatch buffer for fem.integrate's tangent
        # path (worst case).  If it fits within a GPU memory budget, use flat.
        # Buffer = n_boundary × qp_per_cell × test_dof(8) × trial_dof(8)
        #          × taylor_dof(3) × sizeof(float64)
        refined_mask_full = needs_refinement  # (n_boundary,) bool
        effective_counts = torch.ones(n_boundary, dtype=torch.long, device=device)
        effective_counts[refined_mask_full] = counts[refined_mask_full].clamp(min=1)

        # Bucket to next power of 2
        buckets = (
            2 ** torch.ceil(torch.log2(effective_counts.float().clamp(min=1)))
        ).long()
        buckets = buckets.clamp(min=1)
        unique_buckets = torch.unique(buckets, sorted=True)

        max_bucket = int(unique_buckets[-1].item()) if len(unique_buckets) > 0 else 1
        max_qpc = max_bucket * 8

        # Dispatch buffer estimate (float64 tangent accumulation)
        flat_buffer_bytes = n_boundary * max_qpc * 8 * 8 * 3 * 8
        # Budget: use the smaller of 25% of free VRAM and an absolute cap.
        # warp's fem.integrate allocates additional temporaries beyond this
        # estimate, so cap at 6 GB to leave headroom.
        free_vram, _ = torch.cuda.mem_get_info(device)
        budget = min(int(free_vram * 0.25), 6 * 1024**3)

        # Allow tests to force a specific path
        _force = self.params.get("_force_chunked", None)
        if _force is True:
            use_chunks = len(unique_buckets) > 1
        elif _force is False:
            use_chunks = False
        else:
            use_chunks = (len(unique_buckets) > 1) and (flat_buffer_bytes > budget)

        if use_chunks:
            self._build_qp_arrays_chunked(
                n_boundary,
                boundary_idx,
                unique_buckets,
                buckets,
                refined_mask_full,
                counts,
                n_leaves,
                leaf_bidx,
                sort_order,
                s_bidx if n_leaves > 0 else None,
                s_leaf_lo_ref if n_leaves > 0 else None,
                s_leaf_span_ref if n_leaves > 0 else None,
                offsets_in_parent if n_leaves > 0 else None,
                std_gauss_pts,
                std_gauss_w,
                n_parent_elems,
                combined_elements,
                combined_nodes,
                combined_sdf,
            )
        else:
            self._build_qp_arrays_flat(
                n_boundary,
                boundary_idx,
                max_qpc,
                refined_mask_full,
                counts,
                n_leaves,
                leaf_bidx,
                sort_order,
                s_bidx if n_leaves > 0 else None,
                s_leaf_lo_ref if n_leaves > 0 else None,
                s_leaf_span_ref if n_leaves > 0 else None,
                offsets_in_parent if n_leaves > 0 else None,
                std_gauss_pts,
                std_gauss_w,
                n_parent_elems,
                combined_elements,
                combined_nodes,
                combined_sdf,
            )

        # Store quad mesh data for VTU output
        if n_leaves > 0:
            self._quad_mesh_points = fine_node_phys
            self._quad_mesh_sdf = fine_node_sdf
            self._quad_mesh_phi = fine_node_phi
            self._quad_mesh_cells = leaf_conn

    # ------------------------------------------------------------------
    # Flat QP assembly (single subdomain, uniform qp_per_cell)
    # ------------------------------------------------------------------

    def _build_qp_arrays_flat(
        self,
        n_boundary,
        boundary_idx,
        qp_per_cell,
        refined_mask,
        counts,
        n_leaves,
        leaf_bidx,
        sort_order,
        s_bidx,
        s_leaf_lo_ref,
        s_leaf_span_ref,
        offsets_in_parent,
        std_gauss_pts,
        std_gauss_w,
        n_parent_elems,
        combined_elements,
        combined_nodes,
        combined_sdf,
    ):
        """Build flat boundary-only QP arrays for single-subdomain integration."""
        device = self.device
        total_qps = n_boundary * qp_per_cell

        points_t = torch.zeros(
            n_boundary, qp_per_cell, 3, dtype=torch.float32, device=device
        )
        weights_t = torch.zeros(
            n_boundary, qp_per_cell, dtype=torch.float32, device=device
        )
        phi_source = torch.zeros(total_qps, dtype=torch.int32, device=device)
        phi_ref = torch.zeros(total_qps, 3, dtype=torch.float32, device=device)
        active = torch.zeros(total_qps, dtype=torch.int32, device=device)
        parent_elem = torch.zeros(total_qps, dtype=torch.int32, device=device)
        parent_ref = torch.zeros(total_qps, 3, dtype=torch.float32, device=device)

        n_g = 8

        # --- Fill refined elements ---
        if n_leaves > 0:
            sel_s_bidx = s_bidx
            sel_lo_ref = s_leaf_lo_ref
            sel_span_ref = s_leaf_span_ref
            sel_offset = offsets_in_parent
            vol_frac = sel_span_ref**3

            xi_all = sel_lo_ref.unsqueeze(1) + std_gauss_pts.unsqueeze(
                0
            ) * sel_span_ref.unsqueeze(1).unsqueeze(2)

            cell_exp = torch.arange(n_leaves, device=device).repeat_interleave(n_g)
            g_exp = torch.arange(n_g, device=device).repeat(n_leaves)
            slot = sel_offset[cell_exp] * n_g + g_exp
            p_bidx = sel_s_bidx[cell_exp]  # boundary-local index

            points_t[p_bidx, slot] = xi_all[cell_exp, g_exp]
            weights_t[p_bidx, slot] = std_gauss_w * vol_frac[cell_exp]

            flat_idx = p_bidx * qp_per_cell + slot
            combined_leaf_idx = (n_parent_elems + sort_order[cell_exp]).to(torch.int32)
            phi_source[flat_idx] = combined_leaf_idx
            phi_ref[flat_idx] = std_gauss_pts[g_exp]
            active[flat_idx] = 1
            p_coarse = boundary_idx[sel_s_bidx[cell_exp]]
            parent_elem[flat_idx] = p_coarse.to(torch.int32)
            parent_ref[flat_idx] = xi_all[cell_exp, g_exp]

        # --- Fill unrefined boundary elements ---
        unrefined_bidx = torch.where(~refined_mask)[0]
        if unrefined_bidx.numel() > 0:
            n_u = unrefined_bidx.shape[0]
            points_t[unrefined_bidx, :n_g] = std_gauss_pts.unsqueeze(0).expand(
                n_u, -1, -1
            )
            weights_t[unrefined_bidx, :n_g] = std_gauss_w

            u_exp = unrefined_bidx.repeat_interleave(n_g)
            g_exp = torch.arange(n_g, device=device).repeat(n_u)
            flat_idx = u_exp * qp_per_cell + g_exp
            coarse_idx = (
                boundary_idx[unrefined_bidx].repeat_interleave(n_g).to(torch.int32)
            )
            phi_source[flat_idx] = coarse_idx
            phi_ref[flat_idx] = std_gauss_pts[g_exp]
            active[flat_idx] = 1
            parent_elem[flat_idx] = coarse_idx
            parent_ref[flat_idx] = std_gauss_pts[g_exp]

        # --- Store state ---
        self._nitsche_chunks = []
        self._boundary_only_qps = True
        self._qp_uses_child = True
        self._qp_per_cell = qp_per_cell
        self._n_qp_elements = n_boundary

        # Store QP mapping arrays as warp
        self._qp_parent_elem_wp = wp.from_torch(
            parent_elem.contiguous(), dtype=wp.int32
        )
        self._qp_parent_ref_wp = wp.from_torch(parent_ref.contiguous(), dtype=wp.vec3)
        self._qp_active_wp = wp.from_torch(active.contiguous(), dtype=wp.int32)

        # Store quadrature points/weights for correction domain
        q_torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        q_coord_dtype = wp.vec3d if self.use_mixed_precision else wp.vec3
        q_scalar_dtype = wp.float64 if self.use_mixed_precision else wp.float32
        pts = points_t.to(dtype=q_torch_dtype).contiguous()
        wts = weights_t.to(dtype=q_torch_dtype).contiguous()
        self._correction_points_wp = wp.from_torch(pts, dtype=q_coord_dtype)
        self._correction_weights_wp = wp.from_torch(wts, dtype=q_scalar_dtype)
        self._qp_weights_flat_wp = wp.from_torch(
            wts.reshape(-1).to(dtype=torch.float32).contiguous(), dtype=wp.float32
        )

        # Store phi source mapping for _precompute_qp_fields
        self._qp_phi_source_elem = phi_source
        self._qp_phi_ref_coords = phi_ref

        self.log(
            f"Flat QP: {n_boundary} boundary elements × {qp_per_cell} qp/cell "
            f"(dispatch buffer ≈ {n_boundary * qp_per_cell * 8 * 8 * 3 * 8 / 1e9:.2f} GB)"
        )

        # Store combined source mesh (shared with precompute)
        self._combined_source_elements = combined_elements
        self._combined_source_nodes = combined_nodes
        self._combined_source_sdf = combined_sdf

    # ------------------------------------------------------------------
    # Chunked QP assembly (multiple subdomains, per-bucket qp_per_cell)
    # ------------------------------------------------------------------

    def _build_qp_arrays_chunked(
        self,
        n_boundary,
        boundary_idx,
        unique_buckets,
        buckets,
        refined_mask,
        counts,
        n_leaves,
        leaf_bidx,
        sort_order,
        s_bidx,
        s_leaf_lo_ref,
        s_leaf_span_ref,
        offsets_in_parent,
        std_gauss_pts,
        std_gauss_w,
        n_parent_elems,
        combined_elements,
        combined_nodes,
        combined_sdf,
    ):
        """Build per-chunk QP arrays for multi-subdomain integration."""
        device = self.device
        n_g = 8

        chunks: list[NitscheChunk] = []
        for ci, bval in enumerate(unique_buckets):
            chunk_mask = buckets == bval
            chunk_bidx = torch.where(chunk_mask)[0]
            chunk_qpc = int(bval.item()) * 8
            n_chunk = chunk_bidx.shape[0]
            total_chunk_qps = n_chunk * chunk_qpc

            c_points = torch.zeros(
                n_chunk, chunk_qpc, 3, dtype=torch.float32, device=device
            )
            c_weights = torch.zeros(
                n_chunk, chunk_qpc, dtype=torch.float32, device=device
            )
            c_phi_source = torch.zeros(
                total_chunk_qps, dtype=torch.int32, device=device
            )
            c_phi_ref = torch.zeros(
                total_chunk_qps, 3, dtype=torch.float32, device=device
            )
            c_active = torch.zeros(total_chunk_qps, dtype=torch.int32, device=device)
            c_parent_elem = torch.zeros(
                total_chunk_qps, dtype=torch.int32, device=device
            )
            c_parent_ref = torch.zeros(
                total_chunk_qps, 3, dtype=torch.float32, device=device
            )

            # Fill refined elements in this chunk
            chunk_refined = chunk_mask & refined_mask
            if chunk_refined.any() and n_leaves > 0:
                bidx_to_chunk = torch.full(
                    (n_boundary,), -1, dtype=torch.long, device=device
                )
                bidx_to_chunk[chunk_bidx] = torch.arange(
                    n_chunk, dtype=torch.long, device=device
                )

                leaf_in_chunk_sorted = chunk_refined[leaf_bidx[sort_order]]
                leaf_idx_in_chunk = torch.where(leaf_in_chunk_sorted)[0]

                if leaf_idx_in_chunk.numel() > 0:
                    sel_s_bidx = s_bidx[leaf_idx_in_chunk]
                    sel_lo_ref = s_leaf_lo_ref[leaf_idx_in_chunk]
                    sel_span_ref = s_leaf_span_ref[leaf_idx_in_chunk]
                    sel_offset = offsets_in_parent[leaf_idx_in_chunk]
                    vol_frac = sel_span_ref**3

                    xi_all = sel_lo_ref.unsqueeze(1) + std_gauss_pts.unsqueeze(
                        0
                    ) * sel_span_ref.unsqueeze(1).unsqueeze(2)

                    cell_exp = torch.arange(
                        leaf_idx_in_chunk.shape[0], device=device
                    ).repeat_interleave(n_g)
                    g_exp = torch.arange(n_g, device=device).repeat(
                        leaf_idx_in_chunk.shape[0]
                    )
                    slot = sel_offset[cell_exp] * n_g + g_exp
                    p_chunk_idx = bidx_to_chunk[sel_s_bidx[cell_exp]]

                    c_points[p_chunk_idx, slot] = xi_all[cell_exp, g_exp]
                    c_weights[p_chunk_idx, slot] = std_gauss_w * vol_frac[cell_exp]

                    flat_idx = p_chunk_idx * chunk_qpc + slot
                    combined_leaf_idx = (
                        n_parent_elems + sort_order[leaf_idx_in_chunk[cell_exp]]
                    ).to(torch.int32)
                    c_phi_source[flat_idx] = combined_leaf_idx
                    c_phi_ref[flat_idx] = std_gauss_pts[g_exp]
                    c_active[flat_idx] = 1
                    p_coarse = boundary_idx[sel_s_bidx[cell_exp]]
                    c_parent_elem[flat_idx] = p_coarse.to(torch.int32)
                    c_parent_ref[flat_idx] = xi_all[cell_exp, g_exp]

            # Fill unrefined elements in this chunk
            chunk_unrefined = chunk_mask & (~refined_mask)
            chunk_unrefined_bidx = torch.where(chunk_unrefined)[0]
            if chunk_unrefined_bidx.numel() > 0:
                bidx_to_chunk = torch.full(
                    (n_boundary,), -1, dtype=torch.long, device=device
                )
                bidx_to_chunk[chunk_bidx] = torch.arange(
                    n_chunk, dtype=torch.long, device=device
                )

                n_u = chunk_unrefined_bidx.shape[0]
                chunk_local = bidx_to_chunk[chunk_unrefined_bidx]
                c_points[chunk_local, :n_g] = std_gauss_pts.unsqueeze(0).expand(
                    n_u, -1, -1
                )
                c_weights[chunk_local, :n_g] = std_gauss_w

                cl_exp = chunk_local.repeat_interleave(n_g)
                g_exp = torch.arange(n_g, device=device).repeat(n_u)
                flat_idx = cl_exp * chunk_qpc + g_exp
                coarse_idx = (
                    boundary_idx[chunk_unrefined_bidx]
                    .repeat_interleave(n_g)
                    .to(torch.int32)
                )
                c_phi_source[flat_idx] = coarse_idx
                c_phi_ref[flat_idx] = std_gauss_pts[g_exp]
                c_active[flat_idx] = 1
                c_parent_elem[flat_idx] = coarse_idx
                c_parent_ref[flat_idx] = std_gauss_pts[g_exp]

            chunk = NitscheChunk(
                chunk_id=ci,
                boundary_local_indices=chunk_bidx,
                qp_per_cell=chunk_qpc,
                n_elements=n_chunk,
                qp_phi_source_elem=c_phi_source,
                qp_phi_ref_coords=c_phi_ref,
                qp_active=c_active,
                qp_parent_elem=c_parent_elem,
                qp_parent_ref=c_parent_ref,
                qp_weights_flat=c_weights.reshape(-1),
            )

            q_torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
            q_coord_dtype = wp.vec3d if self.use_mixed_precision else wp.vec3
            q_scalar_dtype = wp.float64 if self.use_mixed_precision else wp.float32
            pts = c_points.to(dtype=q_torch_dtype).contiguous()
            wts = c_weights.to(dtype=q_torch_dtype).contiguous()
            chunk.points_wp = wp.from_torch(pts, dtype=q_coord_dtype)
            chunk.weights_wp = wp.from_torch(wts, dtype=q_scalar_dtype)
            chunk.qp_active_wp = wp.from_torch(c_active.contiguous(), dtype=wp.int32)

            chunks.append(chunk)

        self._nitsche_chunks = chunks
        self._boundary_only_qps = True
        self._qp_uses_child = True

        max_qpc = max(c.qp_per_cell for c in chunks) if chunks else 8
        self._qp_per_cell = max_qpc
        self._n_qp_elements = n_boundary

        # Estimate buffer savings
        flat_total = n_boundary * max_qpc
        chunked_total = sum(c.n_elements * c.qp_per_cell for c in chunks)
        self.log(
            f"Chunked QP: {len(chunks)} chunks, "
            + ", ".join(f"{c.n_elements}elem×{c.qp_per_cell}qp" for c in chunks)
            + f" (QP slots: {chunked_total} vs flat {flat_total}, "
            + f"{flat_total / max(chunked_total, 1):.1f}× saving)"
        )

        # Store combined source mesh (shared across chunks)
        self._combined_source_elements = combined_elements
        self._combined_source_nodes = combined_nodes
        self._combined_source_sdf = combined_sdf

    def _build_explicit_quadrature(self, points, weights, n_elems, qp_per_cell):
        """Store QP arrays for Nitsche correction quadrature.

        Args:
            points: (n_elems, qp_per_cell, 3) float32 GPU tensor.
                    n_elems = n_boundary when boundary_only_qps, else n_coarse.
            weights: (n_elems, qp_per_cell) float32 GPU tensor
        """

        q_torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        q_coord_dtype = wp.vec3d if self.use_mixed_precision else wp.vec3
        q_scalar_dtype = wp.float64 if self.use_mixed_precision else wp.float32
        pts = points.reshape(n_elems, qp_per_cell, 3).to(dtype=q_torch_dtype).contiguous()
        wts = weights.reshape(n_elems, qp_per_cell).to(dtype=q_torch_dtype).contiguous()

        self._correction_points_wp = wp.from_torch(pts, dtype=q_coord_dtype)
        self._correction_weights_wp = wp.from_torch(wts, dtype=q_scalar_dtype)

        # Store flat weights for beta computation
        self._qp_weights_flat_wp = wp.from_torch(
            wts.reshape(-1).to(dtype=torch.float32).contiguous(), dtype=wp.float32
        )

    def _build_standard_quadrature(self):
        """Build ExplicitQuadrature for the standard (non-Nitsche) integration path.

        Bakes ``phi_weight`` into the quadrature weights so the standard
        ``P:grad_v`` integrand effectively computes ``phi_weight * P:grad_v``:

        * Interior elements (phi ≈ 1): full weight → full stiffness
        * Exterior elements (phi ≈ 0): weight ≈ phi_floor * gauss_w → tiny
        * Boundary elements: epsilon weight → negligible (the full Nitsche
          integrand with adaptive quadrature provides the real contribution)

        This avoids any post-integration DOF scaling and keeps the BSR
        sparsity pattern intact (no zero-weight rows).
        """

        n_elems = self._prev_octree_elements.shape[0]
        std_gauss_pts = torch.tensor(
            _STD_GAUSS_PTS, dtype=torch.float32, device=self.device
        )
        std_gauss_w = _STD_GAUSS_W

        # --- Compute phi_weight at each standard QP via trilinear interpolation ---
        interp = _TRILINEAR_INTERP_8x8.to(device=self.device)

        elements = self._prev_octree_elements.long()  # (n_elems, 8)
        phi_elem = self._phi_nodal[elements]  # (n_elems, 8)
        phi_qp = phi_elem @ interp.T  # (n_elems, 8)
        _pf = self._phi_floor
        _sharp_vol = bool(self.params.get("sharp_volumetric", False))
        if _sharp_vol:
            phi_weight_qp = torch.where(
                phi_qp > 0.5, torch.ones_like(phi_qp), torch.full_like(phi_qp, _pf)
            )
        else:
            phi_weight_qp = (1.0 - _pf) * phi_qp + _pf  # (n_elems, 8)

        # Standard quadrature points (same for all elements)
        pts = std_gauss_pts.unsqueeze(0).expand(n_elems, -1, -1).contiguous()

        # Weights = gauss_weight * phi_weight
        wts = std_gauss_w * phi_weight_qp  # (n_elems, 8)
        # Zero boundary weights so Nitsche integrand provides their contribution
        _EPS_WEIGHT = 1e-10
        wts[self._boundary_elem_mask] = _EPS_WEIGHT
        wts = wts.to(dtype=torch.float32).contiguous()

        q_torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        q_coord_dtype = wp.vec3d if self.use_mixed_precision else wp.vec3
        q_scalar_dtype = wp.float64 if self.use_mixed_precision else wp.float32
        pts_wp = wp.from_torch(pts.to(dtype=q_torch_dtype), dtype=q_coord_dtype)
        wts_wp = wp.from_torch(wts.to(dtype=q_torch_dtype), dtype=q_scalar_dtype)
        domain = self._mesh_cache["domain"]
        self._standard_quadrature = fem.ExplicitQuadrature(domain, pts_wp, wts_wp)

    def _build_correction_domain(self):
        """Build per-chunk Subdomains + ExplicitQuadratures for Nitsche integration.

        When chunks are available (recursive subdivision path), each chunk gets
        its own Subdomain, ExplicitQuadrature, and test/trial fields.

        When no chunks exist (no-refinement or ref_depth=0 path), falls back to
        a single Subdomain over all boundary elements.
        """

        n_coarse = self._prev_octree_elements.shape[0]
        n_boundary = int(self._boundary_elem_mask.sum().item())
        domain = self._mesh_cache["domain"]
        boundary_idx = torch.where(self._boundary_elem_mask)[0]

        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            self.log(
                f"Correction domain: {n_boundary}/{n_coarse} boundary elements "
                f"in {len(self._nitsche_chunks)} chunks"
            )
            for chunk in self._nitsche_chunks:
                # Global element indices for this chunk
                global_idx = boundary_idx[chunk.boundary_local_indices].to(torch.int32)
                indices_wp = wp.from_torch(global_idx.contiguous(), dtype=wp.int32)
                chunk.subdomain = fem.Subdomain(domain, element_indices=indices_wp)
                chunk.quadrature = fem.ExplicitQuadrature(
                    chunk.subdomain,
                    chunk.points_wp,
                    chunk.weights_wp,
                )
                chunk.test = fem.make_test(space=self.space, domain=chunk.subdomain)
                chunk.trial = fem.make_trial(space=self.space, domain=chunk.subdomain)
        elif self._boundary_only_qps:
            # Flat boundary-only subdomain (recursive subdivision, no chunking)
            self.log(f"Correction domain: {n_boundary}/{n_coarse} boundary elements")
            indices_wp = wp.from_torch(
                boundary_idx.to(torch.int32).contiguous(), dtype=wp.int32
            )
            self._correction_subdomain = fem.Subdomain(
                domain, element_indices=indices_wp
            )
            self._correction_quadrature = fem.ExplicitQuadrature(
                self._correction_subdomain,
                self._correction_points_wp,
                self._correction_weights_wp,
            )
            self._correction_test = fem.make_test(
                space=self.space, domain=self._correction_subdomain
            )
            self._correction_trial = fem.make_trial(
                space=self.space, domain=self._correction_subdomain
            )
        else:
            # Full-array subdomain (no refinement, geometric_func=None)
            self.log(f"Correction domain: {n_boundary}/{n_coarse} boundary elements")
            mask_wp = wp.from_torch(
                self._boundary_elem_mask.to(torch.int32).contiguous(), dtype=wp.int32
            )
            self._correction_subdomain = fem.Subdomain(domain, element_mask=mask_wp)
            self._correction_quadrature = fem.ExplicitQuadrature(
                self._correction_subdomain,
                self._correction_points_wp,
                self._correction_weights_wp,
            )
            self._correction_test = fem.make_test(
                space=self.space, domain=self._correction_subdomain
            )
            self._correction_trial = fem.make_trial(
                space=self.space, domain=self._correction_subdomain
            )

    def _save_quad_mesh_vtu(self):
        """Write the sub-cell quadrature mesh as a standalone VTU with SDF and phi.

        Runs in a background thread so file I/O doesn't block solver construction.
        """
        if not hasattr(self, "_quad_mesh_points"):
            return
        import numpy as np
        import threading

        # Snapshot tensors to CPU numpy before handing off to thread
        points_np = self._quad_mesh_points.cpu().numpy().copy()
        cells_np = self._quad_mesh_cells.cpu().numpy().astype(np.int64).copy()
        sdf_np = self._quad_mesh_sdf.cpu().numpy().copy()
        phi_np = self._quad_mesh_phi.cpu().numpy().copy()
        n_cells = cells_np.shape[0]
        out_dir = os.path.join(self.path, "quad_mesh")
        async_io = getattr(self.params, "async_io", False)
        log_fn = self.log

        def _write():
            from .vtu_writer import VTUWriter

            mesh_info = {
                "dim": 3,
                "points": points_np,
                "cells": cells_np,
                "cell_types": np.full(n_cells, 12, dtype=np.uint8),
            }
            writer = VTUWriter(
                out_dir, mesh_info, fix_orientation=False, async_io=async_io, log=log_fn
            )
            writer.save_timestep(time=0.0, fields={"SDF": sdf_np, "phi": phi_np})
            writer.write_pvd()
            writer.shutdown()
            log_fn(
                f"Quad mesh VTU saved: {out_dir} "
                f"({points_np.shape[0]} nodes, {n_cells} cells)"
            )

        self._quad_vtu_thread = threading.Thread(target=_write, daemon=True)
        self._quad_vtu_thread.start()

    def _precompute_qp_fields(self):
        """Pre-compute phi, grad_phi, |grad_phi| at all adaptive quadrature points."""

        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            self._precompute_qp_fields_chunked()
            return

        # --- Non-chunked path ---
        total_qps = self._n_qp_elements * self._qp_per_cell
        phi_qp = wp.zeros(total_qps, dtype=wp.float32, device=self.device)
        grad_phi_qp = wp.zeros(total_qps, dtype=wp.vec3, device=self.device)
        grad_phi_mag_qp = wp.zeros(total_qps, dtype=wp.float32, device=self.device)

        # Use combined source mesh if available (recursive subdivision path),
        # else use parent coarse mesh (geometric_func=None or ref_depth=0).
        if hasattr(self, "_combined_source_elements"):
            source_elements_wp = wp.from_torch(
                self._combined_source_elements.contiguous(), dtype=wp.int32
            )
            source_nodes_wp = wp.from_torch(
                self._combined_source_nodes.float().contiguous(), dtype=wp.vec3
            )
            source_sdf_wp = wp.from_torch(
                self._combined_source_sdf.contiguous(), dtype=wp.float32
            )
            source_elem_t = self._qp_phi_source_elem
            ref_coords_t = self._qp_phi_ref_coords

            # Persist warp arrays for re-launch during optimization
            self._source_elements_wp = source_elements_wp
            self._source_node_coords_wp = source_nodes_wp
            self._qp_phi_source_elem_wp = wp.from_torch(
                source_elem_t.contiguous(), dtype=wp.int32
            )
            self._qp_phi_ref_coords_wp = wp.from_torch(
                ref_coords_t.contiguous(), dtype=wp.vec3
            )

            wp.launch(
                precompute_qp_phi_grad,
                dim=total_qps,
                inputs=[
                    self._qp_phi_source_elem_wp,
                    self._qp_phi_ref_coords_wp,
                    self._qp_active_wp,
                    self._source_elements_wp,
                    self._source_node_coords_wp,
                    source_sdf_wp,
                    self._epsilon,
                    phi_qp,
                    grad_phi_qp,
                    grad_phi_mag_qp,
                ],
                device=self.device,
            )

            # Release torch tensors (warp arrays are the persistent copies)
            del self._combined_source_elements
            del self._combined_source_nodes
            del self._combined_source_sdf
            del self._qp_phi_source_elem
            del self._qp_phi_ref_coords
        else:
            # No refinement: standard Gauss on coarse mesh
            parent_elements_wp = wp.from_torch(
                self._prev_octree_elements.contiguous(), dtype=wp.int32
            )
            parent_node_coords_wp = wp.from_torch(
                self._prev_octree_nodes.contiguous(), dtype=wp.vec3
            )
            parent_node_sdf_wp = wp.from_torch(
                self._sdf_nodal.contiguous(), dtype=wp.float32
            )

            # Persist source mesh arrays for re-launch during optimization
            self._source_elements_wp = parent_elements_wp
            self._source_node_coords_wp = parent_node_coords_wp

            n_coarse_elems = self._prev_octree_elements.shape[0]
            qp_per_cell = self._qp_per_cell
            total = n_coarse_elems * qp_per_cell

            std_pts = torch.tensor(
                _STD_GAUSS_PTS, dtype=torch.float32, device=self.device
            )

            source_elem = torch.arange(
                n_coarse_elems, device=self.device, dtype=torch.int32
            ).repeat_interleave(qp_per_cell)
            ref_coords = std_pts.repeat(n_coarse_elems, 1)
            active = torch.ones(total, dtype=torch.int32, device=self.device)

            self._qp_phi_source_elem_wp = wp.from_torch(
                source_elem.contiguous(), dtype=wp.int32
            )
            self._qp_phi_ref_coords_wp = wp.from_torch(
                ref_coords.contiguous(), dtype=wp.vec3
            )
            self._qp_active_wp = wp.from_torch(active.contiguous(), dtype=wp.int32)

            wp.launch(
                precompute_qp_phi_grad,
                dim=total,
                inputs=[
                    self._qp_phi_source_elem_wp,
                    self._qp_phi_ref_coords_wp,
                    self._qp_active_wp,
                    self._source_elements_wp,
                    self._source_node_coords_wp,
                    parent_node_sdf_wp,
                    self._epsilon,
                    phi_qp,
                    grad_phi_qp,
                    grad_phi_mag_qp,
                ],
                device=self.device,
            )

            self._qp_parent_elem_wp = self._qp_phi_source_elem_wp
            self._qp_parent_ref_wp = self._qp_phi_ref_coords_wp

        self._phi_qp = phi_qp
        self._grad_phi_qp = grad_phi_qp
        self._grad_phi_mag_qp = grad_phi_mag_qp

    def _precompute_qp_fields_chunked(self):
        """Pre-compute phi, grad_phi, |grad_phi| per chunk."""

        combined_elements_wp = wp.from_torch(
            self._combined_source_elements.contiguous(), dtype=wp.int32
        )
        combined_nodes_wp = wp.from_torch(
            self._combined_source_nodes.float().contiguous(), dtype=wp.vec3
        )
        combined_sdf_wp = wp.from_torch(
            self._combined_source_sdf.contiguous(), dtype=wp.float32
        )

        # Persist source mesh arrays for re-launch during optimization
        self._source_elements_wp = combined_elements_wp
        self._source_node_coords_wp = combined_nodes_wp

        for chunk in self._nitsche_chunks:
            total = chunk.n_elements * chunk.qp_per_cell
            phi_qp = wp.zeros(total, dtype=wp.float32, device=self.device)
            grad_phi_qp = wp.zeros(total, dtype=wp.vec3, device=self.device)
            grad_phi_mag_qp = wp.zeros(total, dtype=wp.float32, device=self.device)

            # Persist per-chunk mapping arrays for re-launch
            chunk.qp_phi_source_elem_wp = wp.from_torch(
                chunk.qp_phi_source_elem.contiguous(), dtype=wp.int32
            )
            chunk.qp_phi_ref_coords_wp = wp.from_torch(
                chunk.qp_phi_ref_coords.contiguous(), dtype=wp.vec3
            )

            wp.launch(
                precompute_qp_phi_grad,
                dim=total,
                inputs=[
                    chunk.qp_phi_source_elem_wp,
                    chunk.qp_phi_ref_coords_wp,
                    chunk.qp_active_wp,
                    combined_elements_wp,
                    combined_nodes_wp,
                    combined_sdf_wp,
                    self._epsilon,
                    phi_qp,
                    grad_phi_qp,
                    grad_phi_mag_qp,
                ],
                device=self.device,
            )
            chunk.phi_qp = phi_qp
            chunk.grad_phi_qp = grad_phi_qp
            chunk.grad_phi_mag_qp = grad_phi_mag_qp

        # Release torch tensors (warp arrays are the persistent copies)
        del self._combined_source_elements
        del self._combined_source_nodes
        del self._combined_source_sdf

    def _bc_match_tolerance(self):
        """Distance within which a BC predicate claims a point off its surface.

        The predicates are evaluated at QPs and nodes of the diffuse band, not
        on the surface, so they need slack.  But everything inside the slack
        is clamped, including the free faces next to a Dirichlet face: at the
        old ``max(2 eps, 0.75 h)`` (= h at eps = h/2) that was a strip ~h tall,
        which shortens the body by O(h), and it also clamped the Dirichlet
        band's 0.79 h QP layer on grid-aligned faces.  Twist cube, depth 6,
        float32: rel. L2 aligned 1.31e-2 -> 5.4e-3, rotated 1.80e-2 -> 1.10e-2.
        Used for the QP/node BC masks, the BC-element classification and
        the SurfaceSelector distance threshold alike.
        """
        return bc_match_tolerance(self._epsilon, self._min_element_size)

    def _build_nitsche_bc_masks(self):
        """Build u_bar, t_bar, psi_D, psi_N from original BC definitions.

        For the diffused Nitsche method, surface BCs become volume BCs across
        the diffuse band around the SDF=0 surface, so torch.isclose is widened
        to ``_bc_match_tolerance()`` for the predicates to reach its QPs.
        """

        node_positions = self._node_positions_torch()
        n_nodes = self.space.node_count()
        with self._widened_bc_isclose():
            self._build_nitsche_bc_masks_inner(node_positions, n_nodes)

    @contextlib.contextmanager
    def _widened_bc_isclose(self):
        """Widen torch.isclose to ``_bc_match_tolerance()`` while BC predicates run."""
        eps_tol = self._bc_match_tolerance()
        _orig_isclose = torch.isclose

        def _widened_isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            return _orig_isclose(
                a, b, rtol=rtol, atol=max(atol, eps_tol), equal_nan=equal_nan
            )

        torch.isclose = _widened_isclose
        try:
            yield
        finally:
            torch.isclose = _orig_isclose

    def _build_nitsche_bc_masks_inner(self, node_positions, n_nodes):
        """Core BC mask construction (called with widened isclose tolerance).

        Evaluates u_bar, t_bar, psi_D, psi_N directly at QP physical positions
        rather than interpolating from nodal values.  This avoids the tapering
        artefact where u_bar linearly decays through boundary elements because
        only surface-adjacent nodes match the BC predicate.
        """
        # --- Nodal fields (for VTU output only) ---
        _torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        u_bar_n = torch.zeros(n_nodes, 3, dtype=_torch_dtype, device=self.device)
        psi_D_components_n = torch.zeros(
            n_nodes, 3, dtype=_torch_dtype, device=self.device
        )
        t_bar_n = torch.zeros(n_nodes, 3, dtype=_torch_dtype, device=self.device)
        psi_N_n = torch.zeros(n_nodes, dtype=_torch_dtype, device=self.device)
        u_bar_count = torch.zeros(
            n_nodes, 3, dtype=_torch_dtype, device=self.device
        )
        t_bar_count = torch.zeros(
            n_nodes, 3, dtype=_torch_dtype, device=self.device
        )

        for boundary_spec, value_spec in self._original_dirichlet_bcs_dict.items():
            nodes = identify_boundary_nodes(boundary_spec, node_positions, self.dim)
            assert nodes.numel() > 0, (
                f"Nitsche Dirichlet BC matched 0 nodes. Check that the boundary "
                f"predicate targets positions within {self._bc_match_tolerance():.4f} "
                f"of actual mesh nodes."
            )
            values, comp_list = evaluate_bc_values(
                value_spec, nodes, node_positions, self.dim
            )
            if comp_list is None:
                u_bar_n[nodes, : self.dim] += values.to(dtype=_torch_dtype)
                u_bar_count[nodes, : self.dim] += 1.0
                psi_D_components_n[nodes, : self.dim] = 1.0
            else:
                comps = torch.as_tensor(comp_list, dtype=torch.long, device=self.device)
                node_grid = nodes[:, None].expand(-1, comps.numel())
                comp_grid = comps[None, :].expand(nodes.numel(), -1)
                u_bar_n[node_grid, comp_grid] += values[:, comps].to(
                    dtype=_torch_dtype
                )
                u_bar_count[node_grid, comp_grid] += 1.0
                psi_D_components_n[node_grid, comp_grid] = 1.0

        u_bar_n = torch.where(
            u_bar_count > 0.0,
            u_bar_n / u_bar_count.clamp_min(1.0),
            u_bar_n,
        )

        for boundary_spec, traction_spec in self._original_neumann_bcs_dict.items():
            nodes = identify_boundary_nodes(boundary_spec, node_positions, self.dim)
            assert nodes.numel() > 0, (
                f"Nitsche Neumann BC matched 0 nodes. Check that the boundary "
                f"predicate targets positions within {self._bc_match_tolerance():.4f} "
                f"of actual mesh nodes."
            )
            values, comp_list = evaluate_bc_values(
                traction_spec, nodes, node_positions, self.dim
            )
            if comp_list is None:
                t_bar_n[nodes, : self.dim] += values.to(dtype=_torch_dtype)
                t_bar_count[nodes, : self.dim] += 1.0
            else:
                comps = torch.as_tensor(comp_list, dtype=torch.long, device=self.device)
                node_grid = nodes[:, None].expand(-1, comps.numel())
                comp_grid = comps[None, :].expand(nodes.numel(), -1)
                t_bar_n[node_grid, comp_grid] += values[:, comps].to(
                    dtype=_torch_dtype
                )
                t_bar_count[node_grid, comp_grid] += 1.0
            psi_N_n[nodes] = 1.0

        t_bar_n = torch.where(
            t_bar_count > 0.0,
            t_bar_n / t_bar_count.clamp_min(1.0),
            t_bar_n,
        )

        self._u_bar_nodal = u_bar_n.to(dtype=torch.float32)
        self._t_bar_nodal = t_bar_n.to(dtype=torch.float32)
        self._psi_D_nodal = psi_D_components_n.any(dim=1).to(dtype=torch.float32)
        self._psi_N_nodal = psi_N_n.to(dtype=torch.float32)

        # --- Evaluate BCs directly at QP physical positions ---
        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            self._build_bc_masks_chunked()
        else:
            self._build_bc_masks_flat()

    def _qp_physical_positions(self, qp_parent_elem, qp_parent_ref, qp_active):
        """Compute physical positions of QPs via trilinear interpolation."""
        qp_elem = qp_parent_elem.long()
        elem_nodes = self._prev_octree_elements[qp_elem]
        node_coords = self._prev_octree_nodes[elem_nodes.long()]

        xi = qp_parent_ref[:, 0:1]
        eta = qp_parent_ref[:, 1:2]
        zeta = qp_parent_ref[:, 2:3]
        cx = torch.tensor(
            [0, 1, 1, 0, 0, 1, 1, 0], dtype=torch.float32, device=self.device
        )
        cy = torch.tensor(
            [0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.float32, device=self.device
        )
        cz = torch.tensor(
            [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32, device=self.device
        )
        fx = cx * xi + (1 - cx) * (1 - xi)
        fy = cy * eta + (1 - cy) * (1 - eta)
        fz = cz * zeta + (1 - cz) * (1 - zeta)
        N = fx * fy * fz
        N = N * qp_active.unsqueeze(-1).float()
        return (N.unsqueeze(-1) * node_coords).sum(dim=1)

    def _evaluate_bcs_at_positions(
        self, qp_phys, qp_active, grad_phi_torch, qp_elem=None, qp_phi=None
    ):
        """Evaluate BC predicates/values at given QP positions.

        Under ``sharp_volumetric``, ``qp_elem``/``qp_phi`` (the element and phi
        of each QP) let a Dirichlet region that reaches into an element lying
        wholly outside the surface claim all of that element's QPs.  Such an
        element has only phi_floor bulk stiffness, so a region edge through it
        leaves some of its nodes pinned by beta and the rest nearly free: a
        ~1e7 stiffness contrast inside one element.  That stalled AMG once
        ``_bc_match_tolerance`` stopped claiming the 0.79 h QP layer
        (TestSharpVsDiffuseGradient: Jacobi-scaled eigenvalue 3e-6, on the
        layer just outside the side faces at the clamped end).  The diffuse
        weight keeps those elements stiff enough, and there the claim costs
        accuracy (twist cube, depth 6: aligned 4.1e-3 -> 6.1e-3, rotated
        Newton failure), so it is sharp-only.

        Returns (u_bar, t_bar, psi_D, psi_N, n_face) as float32 tensors.
        """
        total = qp_phys.shape[0]
        dev = self.device

        exterior_qp = None
        if (
            self.params.get("sharp_volumetric", False)
            and qp_elem is not None
            and qp_phi is not None
            and total > 0
        ):
            elem = qp_elem.long()
            n_elem = int(elem.max().item()) + 1
            active = qp_active > 0
            has_inside = torch.zeros(n_elem, dtype=torch.bool, device=dev)
            has_inside[elem[active & (qp_phi >= 0.5)]] = True
            exterior_qp = active & ~has_inside[elem]

        _torch_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        u_bar = torch.zeros(total, 3, dtype=_torch_dtype, device=dev)
        u_bar_count = torch.zeros(total, 3, dtype=_torch_dtype, device=dev)
        psi_D = torch.zeros(total, 3, dtype=_torch_dtype, device=dev)
        t_bar = torch.zeros(total, 3, dtype=_torch_dtype, device=dev)
        t_bar_count = torch.zeros(total, 3, dtype=_torch_dtype, device=dev)
        psi_N = torch.zeros(total, dtype=_torch_dtype, device=dev)
        n_face = torch.zeros(total, 3, dtype=torch.float32, device=dev)

        for boundary_spec, value_spec in self._original_dirichlet_bcs_dict.items():
            qps = identify_boundary_nodes(boundary_spec, qp_phys, self.dim)
            if qps.numel() == 0:
                continue
            if exterior_qp is not None:
                claimed = torch.zeros(total, dtype=torch.bool, device=dev)
                claimed[qps] = True
                claimed_elem = torch.zeros(n_elem, dtype=torch.bool, device=dev)
                # Only active QPs may claim their element.  Padding slots sit at
                # the origin with parent element 0, so a region that contains the
                # origin would otherwise claim element 0 whole.
                claimed_elem[elem[qps[active[qps]]]] = True
                claimed |= exterior_qp & claimed_elem[elem]
                qps = torch.where(claimed)[0]
            values, comp_list = evaluate_bc_values(value_spec, qps, qp_phys, self.dim)
            if comp_list is None:
                u_bar[qps, : self.dim] += values.to(dtype=_torch_dtype)
                u_bar_count[qps, : self.dim] += 1.0
                psi_D[qps, : self.dim] = 1.0
            else:
                comps = torch.as_tensor(comp_list, dtype=torch.long, device=dev)
                qp_grid = qps[:, None].expand(-1, comps.numel())
                comp_grid = comps[None, :].expand(qps.numel(), -1)
                u_bar[qp_grid, comp_grid] += values[:, comps].to(
                    dtype=_torch_dtype
                )
                u_bar_count[qp_grid, comp_grid] += 1.0
                psi_D[qp_grid, comp_grid] = 1.0

        u_bar = torch.where(
            u_bar_count > 0.0,
            u_bar / u_bar_count.clamp_min(1.0),
            u_bar,
        )

        # Per-region record for _rescale_neumann_tractions, which needs the
        # quadrature built after this (see there).
        rescale = bool(self.params.get("nitsche_neumann_rescale", True))
        neumann_regions = []
        for boundary_spec, traction_spec in self._original_neumann_bcs_dict.items():
            qps = identify_boundary_nodes(boundary_spec, qp_phys, self.dim)
            if qps.numel() == 0:
                continue
            values, comp_list = evaluate_bc_values(
                traction_spec, qps, qp_phys, self.dim
            )
            contribution = torch.zeros(qps.numel(), 3, dtype=_torch_dtype, device=dev)
            if comp_list is None:
                contribution[:, : self.dim] = values.to(dtype=_torch_dtype)
                t_bar[qps, : self.dim] += contribution[:, : self.dim]
                t_bar_count[qps, : self.dim] += 1.0
            else:
                comps = torch.as_tensor(comp_list, dtype=torch.long, device=dev)
                qp_grid = qps[:, None].expand(-1, comps.numel())
                comp_grid = comps[None, :].expand(qps.numel(), -1)
                contribution[:, comps] = values[:, comps].to(dtype=_torch_dtype)
                t_bar[qp_grid, comp_grid] += contribution[:, comps]
                t_bar_count[qp_grid, comp_grid] += 1.0
            psi_N[qps] = 1.0
            if rescale:
                neumann_regions.append({
                    "qps": qps,
                    "contribution": contribution,
                    "spec": boundary_spec,
                    "name": getattr(boundary_spec, "__name__", type(boundary_spec).__name__),
                })

            gp_matched = grad_phi_torch[qps]
            gp_mag = gp_matched.norm(dim=1, keepdim=True).clamp(min=1e-10)
            avg_normal = (-gp_matched / gp_mag).mean(dim=0)
            avg_normal = avg_normal / avg_normal.norm().clamp(min=1e-10)
            n_face[qps] = avg_normal.unsqueeze(0)

        self._neumann_regions = neumann_regions
        self._neumann_t_count = t_bar_count

        t_bar = torch.where(
            t_bar_count > 0.0,
            t_bar / t_bar_count.clamp_min(1.0),
            t_bar,
        )

        inactive = qp_active == 0
        u_bar[inactive] = 0.0
        t_bar[inactive] = 0.0
        psi_D[inactive] = 0.0
        psi_N[inactive] = 0.0
        n_face[inactive] = 0.0

        return (
            u_bar.to(torch.float32),
            t_bar.to(torch.float32),
            psi_D.to(torch.float32),
            psi_N.to(torch.float32),
            n_face,
        )

    def _build_bc_masks_chunked(self):
        """Evaluate BCs on concatenated chunk QPs, then scatter back per-chunk."""
        # Concatenate all chunk QP mapping data
        all_parent_elem = torch.cat([c.qp_parent_elem for c in self._nitsche_chunks])
        all_parent_ref = torch.cat([c.qp_parent_ref for c in self._nitsche_chunks])
        all_active = torch.cat(
            [wp.to_torch(c.qp_active_wp) for c in self._nitsche_chunks]
        )
        all_grad_phi = torch.cat(
            [wp.to_torch(c.grad_phi_qp) for c in self._nitsche_chunks]
        )

        qp_phys = self._qp_physical_positions(
            all_parent_elem, all_parent_ref, all_active
        )

        all_phi = torch.cat([wp.to_torch(c.phi_qp) for c in self._nitsche_chunks])
        u_bar, t_bar, psi_D, psi_N, n_face = self._evaluate_bcs_at_positions(
            qp_phys, all_active, all_grad_phi, qp_elem=all_parent_elem, qp_phi=all_phi
        )

        # Scatter back into per-chunk warp arrays
        offset = 0
        for chunk in self._nitsche_chunks:
            n = chunk.n_elements * chunk.qp_per_cell
            chunk.u_bar_qp = wp.from_torch(
                u_bar[offset : offset + n].contiguous(), dtype=wp.vec3
            )
            chunk.t_bar_qp = wp.from_torch(
                t_bar[offset : offset + n].contiguous(), dtype=wp.vec3
            )
            chunk.psi_D_qp = wp.from_torch(
                psi_D[offset : offset + n].contiguous(), dtype=wp.vec3
            )
            chunk.psi_N_qp = wp.from_torch(
                psi_N[offset : offset + n].contiguous(), dtype=wp.float32
            )
            chunk.n_face_qp = wp.from_torch(
                n_face[offset : offset + n].contiguous(), dtype=wp.vec3
            )
            offset += n

    def _build_bc_masks_flat(self):
        """Evaluate BCs on flat (non-chunked) QP arrays."""
        Q = self._qp_per_cell
        total_qps = self._n_qp_elements * Q

        qp_elem = wp.to_torch(self._qp_parent_elem_wp)
        qp_ref = wp.to_torch(self._qp_parent_ref_wp)
        qp_act = wp.to_torch(self._qp_active_wp)

        qp_phys = self._qp_physical_positions(qp_elem, qp_ref, qp_act)
        grad_phi_torch = wp.to_torch(self._grad_phi_qp)

        u_bar, t_bar, psi_D, psi_N, n_face = self._evaluate_bcs_at_positions(
            qp_phys,
            qp_act,
            grad_phi_torch,
            qp_elem=qp_elem,
            qp_phi=wp.to_torch(self._phi_qp),
        )

        self._u_bar_qp = wp.from_torch(u_bar.contiguous(), dtype=wp.vec3)
        self._t_bar_qp = wp.from_torch(t_bar.contiguous(), dtype=wp.vec3)
        self._psi_D_qp = wp.from_torch(psi_D.contiguous(), dtype=wp.vec3)
        self._psi_N_qp = wp.from_torch(psi_N.contiguous(), dtype=wp.float32)
        self._n_face_qp = wp.from_torch(n_face.contiguous(), dtype=wp.vec3)

    def _neumann_region_area(self, boundary_spec):
        """Surface area a Neumann region's traction is meant to load.

        A SurfaceSelector is measured on ``bc_reference_surface`` when the
        solver has one (ShapeOptTrimmed passes its explicit source mesh), via
        ``loaded_area_on``.  That is the surface the region was approved on and
        a force was divided by.  The solver's own MC surface is only an
        approximation of it: a rotated bar's top face, selected there by a 10
        deg ``aligned_normal``, lost its bevelled rim and came out at 0.898 of
        the true area, and the rescaled load with it.  Without a reference
        surface, the selector's resolved patch (``loaded_area``) is used.

        A plain predicate claims the faces of the solver's own MC surface whose
        centroids it accepts, under the same widened ``isclose`` it gets at the
        QPs (the caller holds ``_widened_bc_isclose``), weighted the same way.
        Faces at an edge that the widening also claims sit at right angles to
        the region and carry ~0 weight.
        """
        from .bc_surface_selectors import (
            SurfaceSelector, _compute_face_attributes, _loaded_area,
        )

        if isinstance(boundary_spec, SurfaceSelector) and boundary_spec.is_resolved:
            reference = self.params.get("bc_reference_surface", None)
            if reference is None:
                return boundary_spec.loaded_area()
            verts = torch.as_tensor(
                reference["vertices"], dtype=torch.float32, device=self.device
            )
            faces = torch.as_tensor(
                reference["faces"], dtype=torch.int64, device=self.device
            )
            return boundary_spec.loaded_area_on(verts, faces)
        verts, faces = self._get_or_build_mc_surface_physical()
        centroids, normals, areas = _compute_face_attributes(verts, faces)
        claimed = identify_boundary_nodes(boundary_spec, centroids, self.dim)
        if claimed.numel() == 0:
            return 0.0
        return _loaded_area(areas[claimed].double(), normals[claimed].double())

    def _integrate_qp_array(self, f):
        """Integrate a per-QP float array over the Nitsche quadrature (all chunks)."""
        chunks = getattr(self, "_nitsche_chunks", None)
        pieces = []
        if chunks:
            offset = 0
            for chunk in chunks:
                n = chunk.n_elements * chunk.qp_per_cell
                pieces.append((f[offset : offset + n], chunk.quadrature))
                offset += n
        else:
            pieces.append((f, self._correction_quadrature))
        total = 0.0
        for values, quadrature in pieces:
            total += float(fem.integrate(
                _qp_array_integrand,
                quadrature=quadrature,
                values={"f_arr": wp.from_torch(
                    values.to(torch.float32).contiguous(), dtype=wp.float32
                )},
                output_dtype=wp.float64,
                accumulate_dtype=wp.float64,
            ))
        return total

    def _rescale_neumann_tractions(self):
        """Scale each Neumann region so its load integrates to traction x area.

        The Neumann term loads ``t . v * max(0, -grad(phi) . n_face)`` at the QPs
        a region's predicate claims.  Over an unbounded band that weight
        integrates to the region's loaded area, so the total force is
        ``t * area``, which is what a traction chosen as force / loaded_area
        assumes.  The discrete sum is not that.  The predicate cuts the band at
        ``_bc_match_tolerance``, which keeps tanh(tol/eps) of the sech^2
        kernel (96.4 % at 2 eps, 90.5 % at 1.5 eps).  And the workflow's
        ShapeOptTrimmed integrates with plain 2x2x2 Gauss points on cells 2 eps
        wide, so the sum also depends on where the surface falls between them.
        Measured by summing R(u=0): a rotated bar under uniaxial traction
        received 95.8 % of its load at 2 eps and 87.7 % at 1.5 eps, and a flat
        table seat 99.5 %.  The missing load showed up one for one as a
        too-stiff displacement.

        Each region r gets ``t_r *= A_r / W_r``, where ``A_r`` is its loaded
        area (``_neumann_region_area``) and ``W_r`` is the same QP weight
        integrated with the same quadrature and gating the residual uses.  Its
        total force is then exactly ``t_r * A_r``, whatever the band width or
        mesh alignment; the distribution within the band is unchanged.  Where
        regions overlap, t_bar averages them as before, and each region's share
        is weighted accordingly.

        The factors are computed once, on the solver's first assembled build,
        and kept for its lifetime.  Design updates (``_rebuild_nitsche_state``)
        and the remeshes of an optimization (``reinit_with_sdf``) reuse them,
        so the load does not jump between iterations and the adjoint, which
        treats t_bar as data, stays consistent.  Recomputing on every rebuild
        broke the tape/FD check of TestShapeOptTrimmed (ratio 0.005): a
        +-1e-4 SDF perturbation at a node on the surface changes the MC
        triangulation, so a predicate region's area, and the objective, jumped
        by a finite amount.  A region that claims no QPs on that first build
        keeps factor 1.  ``nitsche_neumann_rescale=False`` restores the
        unscaled traction.
        """
        regions = getattr(self, "_neumann_regions", None)
        if not regions:
            return
        chunks = getattr(self, "_nitsche_chunks", None)
        if chunks:
            def cat(name):
                return torch.cat([wp.to_torch(getattr(c, name)) for c in chunks])
            phi, grad_phi, n_face = cat("phi_qp"), cat("grad_phi_qp"), cat("n_face_qp")
            psi_N, active = cat("psi_N_qp"), cat("qp_active_wp")
        else:
            phi = wp.to_torch(self._phi_qp)
            grad_phi = wp.to_torch(self._grad_phi_qp)
            n_face = wp.to_torch(self._n_face_qp)
            psi_N = wp.to_torch(self._psi_N_qp)
            active = wp.to_torch(self._qp_active_wp)

        # The residual's own gating (see _make_nitsche_stress): active QP,
        # psi_N > 0.5, phi above nitsche_phi_threshold.
        threshold = float(self.params.get("nitsche_phi_threshold", self._phi_floor))
        loaded = (active > 0) & (psi_N > 0.5) & (phi > threshold)
        weight = (-(grad_phi.double() * n_face.double()).sum(dim=1)).clamp_min(0.0)
        weight = torch.where(loaded, weight, torch.zeros_like(weight))
        per_qp_count = self._neumann_t_count.amax(dim=1).clamp_min(1.0)

        factors = getattr(self, "_neumann_rescale_factors", None)
        if factors is None:
            factors = {}
            for region in regions:
                qps = region["qps"]
                share = torch.zeros_like(weight)
                share[qps] = weight[qps] / per_qp_count[qps]
                integrated = self._integrate_qp_array(share)
                with self._widened_bc_isclose():
                    area = float(self._neumann_region_area(region["spec"]))
                if area > 0.0 and integrated > 0.0:
                    factor = area / integrated
                else:
                    factor = 1.0
                    self.log(
                        f"  Neumann rescale skipped for {region['name']}: loaded "
                        f"area {area:.6g}, integrated weight {integrated:.6g}"
                    )
                self.log(
                    f"  Neumann rescale {region['name']}: loaded area {area:.6g}, "
                    f"integrated weight {integrated:.6g}, factor {factor:.4f}"
                )
                factors[region["spec"]] = factor
            self._neumann_rescale_factors = factors

        t_bar = torch.zeros_like(self._neumann_t_count)
        for region in regions:
            # A region that claimed no QPs on the mesh's own build has no factor.
            factor = factors.get(region["spec"], 1.0)
            t_bar[region["qps"]] += factor * region["contribution"]
        t_bar = torch.where(
            self._neumann_t_count > 0.0,
            t_bar / self._neumann_t_count.clamp_min(1.0),
            t_bar,
        )
        t_bar[active == 0] = 0.0
        t_bar = t_bar.to(torch.float32)

        if chunks:
            offset = 0
            for chunk in chunks:
                n = chunk.n_elements * chunk.qp_per_cell
                chunk.t_bar_qp = wp.from_torch(
                    t_bar[offset : offset + n].contiguous(), dtype=wp.vec3
                )
                offset += n
        else:
            self._t_bar_qp = wp.from_torch(t_bar.contiguous(), dtype=wp.vec3)

    def _compute_stabilization_beta(self, use_current_u=False):
        """Compute per-element stabilization parameter beta.

        Two modes:
        * ``heuristic_beta=True`` (fast, ~0s): uniform β = C_pen·(λ+2μ)/h · safety
        * ``heuristic_beta=False`` (default): eigenvalue-based per-element β

        The eigenvalue path assembles LHS (T^T T) and RHS (B^T C B) matrices
        using vectorized PyTorch and solves a batched generalized eigenvalue problem.
        It follows ``use_mixed_precision``; numerically problematic f32 batches
        are retried in f64 before falling back to the heuristic value.
        """

        n_coarse = self._prev_octree_elements.shape[0]
        total_qps = self._n_qp_elements * self._qp_per_cell
        n_boundary = self._boundary_elem_indices.shape[0]

        mu = float(self.mu)
        kappa = float(self.kappa)
        lambda_eff = kappa - 2.0 * mu / 3.0
        h = self._min_element_size
        C_pen = 20.0
        beta_fallback = float(C_pen * (lambda_eff + 2.0 * mu) / h)

        beta_element = torch.zeros(n_coarse, dtype=torch.float32, device=self.device)

        if n_boundary == 0:
            self._beta_element = beta_element
            if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
                for chunk in self._nitsche_chunks:
                    chunk.beta_qp = wp.zeros(
                        chunk.n_elements * chunk.qp_per_cell,
                        dtype=wp.float32,
                        device=self.device,
                    )
            else:
                self._beta_qp = wp.zeros(
                    self._n_qp_elements * self._qp_per_cell,
                    dtype=wp.float32,
                    device=self.device,
                )
            self.log("Beta: 0 boundary elements")
            return

        # --- Fast heuristic beta: uniform value for all boundary elements ---
        if self.params.get("heuristic_beta", False):
            _sf = float(self.params.get("beta_safety_factor", 5.0))
            beta_val = float(beta_fallback * _sf)
            beta_element[self._boundary_elem_indices.long()] = beta_val
            self._beta_element = beta_element
            self.log(
                f"Beta (heuristic): {beta_val:.2f} for {n_boundary} boundary elements"
            )

            if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
                for chunk in self._nitsche_chunks:
                    n_qp = chunk.n_elements * chunk.qp_per_cell
                    chunk.beta_qp = wp.from_torch(
                        torch.full(
                            (n_qp,), beta_val, dtype=torch.float32, device=self.device
                        ).contiguous(),
                        dtype=wp.float32,
                    )
            elif self._boundary_only_qps:
                Q = self._qp_per_cell
                beta_qp_t = torch.full(
                    (n_boundary * Q,), beta_val, dtype=torch.float32, device=self.device
                )
                self._beta_qp = wp.from_torch(beta_qp_t.contiguous(), dtype=wp.float32)
            else:
                total_qps = self._n_qp_elements * self._qp_per_cell
                beta_qp = wp.zeros(total_qps, dtype=wp.float32, device=self.device)
                beta_element_wp = wp.from_torch(
                    beta_element.contiguous(), dtype=wp.float32
                )
                wp.launch(
                    expand_element_beta_to_qp,
                    dim=n_coarse,
                    inputs=[beta_element_wp, self._qp_per_cell, beta_qp],
                    device=self.device,
                )
                self._beta_qp = beta_qp
            return

        B = n_boundary
        device = self.device
        beta_dtype = torch.float64 if self.use_mixed_precision else torch.float32
        bdry_idx = self._boundary_elem_indices.long()  # (B,)
        node_coords = self._prev_octree_nodes.to(beta_dtype)  # (n_nodes, 3)

        # VTK hex corner coords (constant)
        cx = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0], dtype=beta_dtype, device=device)
        cy = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=beta_dtype, device=device)
        cz = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=beta_dtype, device=device)
        sx = 2 * cx - 1
        sy = 2 * cy - 1
        sz = 2 * cz - 1
        eye24 = torch.eye(24, dtype=beta_dtype, device=device).unsqueeze(0)
        reg_floor = (lambda_eff + 2.0 * mu) * h * 1e-6

        beta_boundary = torch.full(
            (B,), beta_fallback, dtype=torch.float32, device=device
        )
        n_failed = 0

        # --- Gather QP data depending on chunked vs flat mode ---
        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            # Chunked mode: process each NitscheChunk separately (uniform Q per chunk)
            for nchunk in self._nitsche_chunks:
                Q = nchunk.qp_per_cell
                B_c = nchunk.n_elements
                # boundary-local indices for this chunk's elements
                c_bidx = nchunk.boundary_local_indices  # (B_c,) into [0..n_boundary)

                _all_ref = nchunk.qp_parent_ref
                _all_active = wp.to_torch(nchunk.qp_active_wp)
                _all_weights = nchunk.qp_weights_flat
                _all_phi = wp.to_torch(nchunk.phi_qp)
                _all_gphi = wp.to_torch(nchunk.grad_phi_qp)
                _all_gpm = wp.to_torch(nchunk.grad_phi_mag_qp)

                itemsize = torch.tensor([], dtype=beta_dtype).element_size()
                _mem_chunk = max(1, min(B_c, int(2e9 / (Q * 200 * itemsize))))
                for c_start in range(0, B_c, _mem_chunk):
                    c_end = min(c_start + _mem_chunk, B_c)
                    Bc = c_end - c_start
                    qp_off = torch.arange(c_start, c_end, device=device) * Q
                    qp_idx = qp_off.unsqueeze(1) + torch.arange(
                        Q, device=device
                    ).unsqueeze(0)
                    qf = qp_idx.reshape(-1)

                    beta_out_idx = c_bidx[c_start:c_end]  # map back to beta_boundary
                    bi = bdry_idx[beta_out_idx]

                    n_failed += self._eigenvalue_beta_batch(
                        Bc,
                        Q,
                        qf,
                        bi,
                        beta_boundary,
                        beta_out_idx,
                        _all_ref,
                        _all_active,
                        _all_weights,
                        _all_phi,
                        _all_gphi,
                        _all_gpm,
                        node_coords,
                        cx,
                        cy,
                        cz,
                        sx,
                        sy,
                        sz,
                        eye24,
                        reg_floor,
                        mu,
                        lambda_eff,
                        beta_fallback,
                        beta_dtype,
                        device,
                    )
        else:
            # Flat mode (non-chunked)
            Q = self._qp_per_cell
            _all_ref = wp.to_torch(self._qp_parent_ref_wp)
            _all_active = wp.to_torch(self._qp_active_wp)
            _all_weights = wp.to_torch(self._qp_weights_flat_wp)
            _all_phi = wp.to_torch(self._phi_qp)
            _all_gphi = wp.to_torch(self._grad_phi_qp)
            _all_gpm = wp.to_torch(self._grad_phi_mag_qp)

            itemsize = torch.tensor([], dtype=beta_dtype).element_size()
            _mem_chunk = max(1, min(B, int(2e9 / (Q * 200 * itemsize))))
            for c_start in range(0, B, _mem_chunk):
                c_end = min(c_start + _mem_chunk, B)
                Bc = c_end - c_start

                if self._boundary_only_qps:
                    qp_off = torch.arange(c_start, c_end, device=device) * Q
                else:
                    bi_global = bdry_idx[c_start:c_end]
                    qp_off = bi_global * Q
                qp_idx = qp_off.unsqueeze(1) + torch.arange(Q, device=device).unsqueeze(
                    0
                )
                qf = qp_idx.reshape(-1)

                beta_out_idx = torch.arange(
                    c_start, c_end, device=device, dtype=torch.long
                )
                bi = bdry_idx[c_start:c_end]

                n_failed += self._eigenvalue_beta_batch(
                    Bc,
                    Q,
                    qf,
                    bi,
                    beta_boundary,
                    beta_out_idx,
                    _all_ref,
                    _all_active,
                    _all_weights,
                    _all_phi,
                    _all_gphi,
                    _all_gpm,
                    node_coords,
                    cx,
                    cy,
                    cz,
                    sx,
                    sy,
                    sz,
                    eye24,
                    reg_floor,
                    mu,
                    lambda_eff,
                    beta_fallback,
                    beta_dtype,
                    device,
                )

        if n_failed > 0:
            self.log(
                f"  {n_failed}/{B} elements used heuristic "
                f"beta_fallback={beta_fallback:.1f}"
            )

        # Safety factor (must compensate d²P/dF² correction in tangent)
        _sf = float(self.params.get("beta_safety_factor", 5.0))
        beta_boundary = beta_boundary * _sf
        material_scale = None
        if self._has_spatial_material:
            element_nodes = self._prev_octree_elements[bdry_idx].long()
            material = self._material_values_torch[element_nodes]
            local_mu = material[..., 0] / (2.0 * (1.0 + material[..., 1]))
            local_kappa = material[..., 0] / (3.0 * (1.0 - 2.0 * material[..., 1]))
            local_stiffness = (local_kappa + 4.0 * local_mu / 3.0).mean(dim=1)
            reference_stiffness = max(lambda_eff + 2.0 * mu, 1.0e-30)
            material_scale = (local_stiffness / reference_stiffness).clamp_min(1.0e-6)
            beta_boundary = beta_boundary * material_scale

        # Cap extreme beta values that arise from pathologically small trimmed
        # boundary elements.  Without a cap, the condition number of the global
        # system is dominated by beta_max / E, causing CG to stall.
        # Floor: prevent zero beta at interface elements (ensures Nitsche
        # penalty is always active where grad_phi is nonzero).
        _beta_min = float(C_pen * (lambda_eff + 2.0 * mu) / h * 0.01)
        beta_minimum = (_beta_min * material_scale if material_scale is not None else
                        torch.full_like(beta_boundary, _beta_min))
        n_floored = int((beta_boundary < beta_minimum).sum().item())
        if n_floored > 0:
            self.log(f"  Floored {n_floored}/{B} beta values to {_beta_min:.1f}")
        beta_boundary = torch.maximum(beta_boundary, beta_minimum)

        _beta_max = self.params.get("beta_max", None)
        if _beta_max is not None:
            _beta_max = float(_beta_max)
            n_capped = int((beta_boundary > _beta_max).sum().item())
            if n_capped > 0:
                self.log(f"  Capped {n_capped}/{B} beta values to {_beta_max:.1f}")
            beta_boundary = beta_boundary.clamp(max=_beta_max)

        beta_element[self._boundary_elem_indices.long()] = beta_boundary
        self._beta_element = beta_element

        self.log(
            f"Beta: [{beta_boundary.min().item():.2f}, {beta_boundary.max().item():.2f}] "
            f"over {B} boundary elements"
        )

        # Expand beta to per-QP (constant within element)
        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            for nchunk in self._nitsche_chunks:
                c_bidx = nchunk.boundary_local_indices
                chunk_beta = beta_boundary[c_bidx]
                Q_c = nchunk.qp_per_cell
                beta_qp_t = chunk_beta.repeat_interleave(Q_c)
                nchunk.beta_qp = wp.from_torch(beta_qp_t.contiguous(), dtype=wp.float32)
        elif self._boundary_only_qps:
            Q = self._qp_per_cell
            beta_qp_t = beta_boundary.repeat_interleave(Q)
            self._beta_qp = wp.from_torch(beta_qp_t.contiguous(), dtype=wp.float32)
        else:
            Q = self._qp_per_cell
            beta_qp = wp.zeros(total_qps, dtype=wp.float32, device=device)
            beta_element_wp = wp.from_torch(beta_element.contiguous(), dtype=wp.float32)
            wp.launch(
                expand_element_beta_to_qp,
                dim=n_coarse,
                inputs=[beta_element_wp, Q, beta_qp],
                device=device,
            )
            self._beta_qp = beta_qp

    def _eigenvalue_beta_batch(
        self,
        Bc,
        Q,
        qf,
        bi,
        beta_boundary,
        beta_out_idx,
        _all_ref,
        _all_active,
        _all_weights,
        _all_phi,
        _all_gphi,
        _all_gpm,
        node_coords,
        cx,
        cy,
        cz,
        sx,
        sy,
        sz,
        eye24,
        reg_floor,
        mu,
        lambda_eff,
        beta_fallback,
        compute_dtype,
        device,
    ):
        """Compute eigenvalue-based beta for a batch of boundary elements.

        Returns the number of elements that fell back to heuristic beta.
        """
        ref_coords = _all_ref[qf].to(compute_dtype).reshape(Bc, Q, 3)
        active = _all_active[qf].reshape(Bc, Q)
        weights = _all_weights[qf].to(compute_dtype).reshape(Bc, Q)
        phi = _all_phi[qf].to(compute_dtype).reshape(Bc, Q)
        grad_phi = _all_gphi[qf].to(compute_dtype).reshape(Bc, Q, 3)
        gpm = _all_gpm[qf].to(compute_dtype).reshape(Bc, Q)

        elem_nodes = self._prev_octree_elements[bi]  # (Bc, 8)
        phys = node_coords[elem_nodes.long()].to(compute_dtype)  # (Bc, 8, 3)

        xi = ref_coords[:, :, 0:1]
        eta = ref_coords[:, :, 1:2]
        zeta = ref_coords[:, :, 2:3]

        fx = cx * xi + (1 - cx) * (1 - xi)
        fy = cy * eta + (1 - cy) * (1 - eta)
        fz = cz * zeta + (1 - cz) * (1 - zeta)

        dN_dxi = sx * fy * fz
        dN_deta = fx * sy * fz
        dN_dzeta = fx * fy * sz
        dN_ref = torch.stack([dN_dxi, dN_deta, dN_dzeta], dim=-1)

        phys_exp = phys.unsqueeze(1).expand(-1, Q, -1, -1)
        J = torch.einsum("bqni,bqnj->bqij", phys_exp, dN_ref)
        detJ = torch.linalg.det(J)
        J_inv_T = torch.linalg.inv(J).transpose(-1, -2)
        dN_phys = torch.einsum("bqij,bqnj->bqni", J_inv_T, dN_ref)

        safe_gpm = torch.clamp(gpm, min=1e-12)
        n_hat = grad_phi / safe_gpm.unsqueeze(-1)
        active_mask = (active > 0).to(compute_dtype)
        _npt = float(
            self.params.get("nitsche_phi_threshold", self.params.get("phi_floor", 1e-4))
        )
        phi_mask = (phi > _npt).to(compute_dtype)

        w_lhs = weights * detJ * gpm * active_mask * phi_mask
        _sharp_vol = bool(self.params.get("sharp_volumetric", False))
        if _sharp_vol:
            phi_vol = torch.where(
                phi > 0.5, torch.ones_like(phi), torch.full_like(phi, self._phi_floor)
            )
        else:
            phi_vol = phi
        w_rhs = weights * detJ * phi_vol * active_mask

        a = torch.einsum("bqnd,bqd->bqn", dN_phys, n_hat)  # (Bc, Q, 8)

        # === LHS (A_e): squared traction ∫ T^k·T^l |∇φ| dV ===
        # T[b,q,n,d,i] = μ·a_n·δ_{di} + μ·n̂_d·g_{n,i} + λ_eff·g_{n,d}·n̂_i
        I3 = torch.eye(3, dtype=compute_dtype, device=device).view(1, 1, 1, 3, 3)
        T = (
            mu * a.unsqueeze(-1).unsqueeze(-1) * I3
            + mu * n_hat.unsqueeze(2).unsqueeze(4) * dN_phys.unsqueeze(3)
            + lambda_eff * dN_phys.unsqueeze(4) * n_hat.unsqueeze(2).unsqueeze(3)
        )  # (Bc, Q, 8, 3, 3) — axes: (b, q, n, d, i)

        sqrt_wl = torch.sqrt(torch.clamp(w_lhs, min=0.0))  # (Bc, Q)
        T_w = T.reshape(Bc, Q, 24, 3) * sqrt_wl[:, :, None, None]
        T_w_mat = T_w.reshape(Bc * Q, 24, 3)
        lhs = (
            torch.bmm(T_w_mat, T_w_mat.transpose(-1, -2))
            .reshape(Bc, Q, 24, 24)
            .sum(1)
        )  # (Bc, 24, 24)

        # === RHS (B_e): stiffness ∫ ε(δuᵏ):C:ε(δuˡ) φ dV ===
        sqrt_wr = torch.sqrt(torch.clamp(w_rhs, min=0.0))  # (Bc, Q)
        dNwr = dN_phys * sqrt_wr[:, :, None, None]  # (Bc, Q, 8, 3)

        # Term 1 (isotropic shear): μ δ_{dk,dl} ∇N_{nk}·∇N_{nl}
        dNwr_perm = dNwr.permute(0, 1, 3, 2).reshape(Bc, Q * 3, 8)
        S_sum = torch.bmm(dNwr_perm.permute(0, 2, 1), dNwr_perm)  # (Bc, 8, 8)
        rhs = torch.zeros(Bc, 24, 24, dtype=compute_dtype, device=device)
        for d in range(3):
            rhs[:, d::3, d::3] = mu * S_sum

        # Term 2 (cross term): μ (∇N_{nk})_{dl} (∇N_{nl})_{dk}
        # cross[b, dl, dk, nk, nl] = Σ_q dNwr[b,q,nk,dl] dNwr[b,q,nl,dk]
        cross = torch.einsum("bqnf,bqme->bfenm", dNwr, dNwr)  # (Bc, 3, 3, 8, 8)
        CT = cross.permute(0, 3, 2, 4, 1).reshape(Bc, 24, 24)  # → (b, nk, dk, nl, dl)
        rhs += mu * CT

        # Term 3 (volumetric): λ_eff (∇N_{nk})_{dk} (∇N_{nl})_{dl}
        vwr = dNwr.reshape(Bc, Q, 24)
        VV = torch.bmm(vwr.permute(0, 2, 1), vwr)
        rhs += lambda_eff * VV

        # Regularize & symmetrize
        rhs_trace = rhs.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True).unsqueeze(-1)
        reg_scale = torch.clamp(rhs_trace.abs() / 24.0, min=reg_floor)
        rhs = rhs + reg_scale * eye24
        rhs = 0.5 * (rhs + rhs.mT)
        lhs = 0.5 * (lhs + lhs.mT)

        def retry_f64(local_indices):
            """Retry only numerically problematic f32 elements in f64."""
            if local_indices.numel() == 0:
                return 0
            qf_rows = qf.reshape(Bc, Q)[local_indices].reshape(-1)
            return self._eigenvalue_beta_batch(
                int(local_indices.numel()),
                Q,
                qf_rows,
                bi[local_indices],
                beta_boundary,
                beta_out_idx[local_indices],
                _all_ref,
                _all_active,
                _all_weights,
                _all_phi,
                _all_gphi,
                _all_gpm,
                node_coords,
                cx,
                cy,
                cz,
                sx,
                sy,
                sz,
                eye24,
                reg_floor,
                mu,
                lambda_eff,
                beta_fallback,
                torch.float64,
                device,
            )

        n_failed = 0
        try:
            L = torch.linalg.cholesky(rhs)
            C = torch.linalg.solve_triangular(L, lhs, upper=False)
            C = torch.linalg.solve_triangular(L, C.mT, upper=False).mT
            eigvals = torch.linalg.eigvalsh(C)
            beta_values = eigvals[:, -1]
            valid = torch.isfinite(beta_values) & (beta_values >= 0.0)
            if bool(valid.all()):
                beta_boundary[beta_out_idx] = beta_values.float()
            elif compute_dtype == torch.float32:
                beta_boundary[beta_out_idx[valid]] = beta_values[valid].float()
                n_failed += retry_f64(torch.where(~valid)[0])
            else:
                beta_boundary[beta_out_idx[valid]] = beta_values[valid].float()
                n_bad = int((~valid).sum().item())
                beta_boundary[beta_out_idx[~valid]] = beta_fallback
                n_failed += n_bad
        except Exception:
            if compute_dtype == torch.float32:
                return retry_f64(torch.arange(Bc, device=device))
            for i in range(Bc):
                try:
                    L_i = torch.linalg.cholesky(rhs[i : i + 1])
                    C_i = torch.linalg.solve_triangular(
                        L_i, lhs[i : i + 1], upper=False
                    )
                    C_i = torch.linalg.solve_triangular(L_i, C_i.mT, upper=False).mT
                    ev = torch.linalg.eigvalsh(C_i)
                    beta_boundary[beta_out_idx[i]] = ev[0, -1].clamp(min=0.0).float()
                except Exception:
                    beta_boundary[beta_out_idx[i]] = beta_fallback
                    n_failed += 1
        return n_failed

    # ---- Integration hooks (three-zone architecture) ----
    #
    # Zone 1 (Interior): Standard neo-Hookean, full stiffness, default Gauss.
    # Zone 2 (Exterior): Standard neo-Hookean, scaled by phi_floor.
    # Zone 3 (Boundary): Full Nitsche integrand, adaptive quadrature.
    #
    # _default_quadrature() returns _standard_quadrature (zero weights on
    # boundary elements) so the standard path covers zones 1+2 only.
    # _add_to_residual/_add_to_tangent then:
    #   a) scale exterior DOF entries by phi_floor (vectorized torch),
    #   b) add full Nitsche on boundary subdomain.

    def _body_force_depends_on_design(self):
        """The diffuse body load contributes to the residual design VJP."""
        return True

    def _body_force_cache_ready(self):
        """Body assembly requires quadrature/probe state from this mesh."""
        return bool(getattr(self, "_nitsche_body_state_ready", False))

    def _assemble_body_forces(self, domain, test, tape=None):
        """Assemble the two-zone phase-weighted body load on GPU.

        Standard quadrature already contains the SDF-derived phase weight. In
        hidden-phi mode its unit-valued ``e_phi`` additionally remains on tape,
        exactly like the interior stress residual. The boundary body term is
        part of the Nitsche stress integrand, avoiding a second subdomain pass.
        """
        if not hasattr(self, "_standard_quadrature"):
            return super()._assemble_body_forces(domain, test, tape=tape)

        result = wp.zeros(
            self.space.node_count(),
            dtype=self.dtype,
            requires_grad=tape is not None,
            device=self.device,
        )
        _sd = self._integrand_scalar_dtype
        use_hidden_phi = (
            hasattr(self, "_use_phase_field_design")
            and self._use_phase_field_design()
            and getattr(self, "e_phi", None) is not None
        )

        std_kw = dict(
            output=result,
            add=True,
            fields={"v": test},
            quadrature=self._standard_quadrature,
            accumulate_dtype=_sd,
        )
        if self._has_spatial_material:
            std_kw["fields"]["material"] = self.material_field
        std_integrand = self._body_force_integrand
        if use_hidden_phi:
            std_integrand = self._hidden_phi_body_force_integrand
            std_kw["fields"]["e"] = self.e_phi

        def _integrate(integrand, **kwargs):
            if tape is None:
                fem.integrate(integrand, **kwargs)
            else:
                with tape:
                    fem.integrate(integrand, **kwargs)

        _integrate(std_integrand, **std_kw)
        return result

    def _default_quadrature(self):
        """Return ExplicitQuadrature with zero weights on boundary elements.

        The standard neo-Hookean integration uses this quadrature so that
        boundary elements contribute nothing.  Their contribution comes
        entirely from the full Nitsche integrand in ``_add_to_residual`` /
        ``_add_to_tangent``.
        """
        if (
            hasattr(self, "_standard_quadrature")
            and self._standard_quadrature is not None
        ):
            return self._standard_quadrature
        return super()._default_quadrature()

    def _nitsche_values(self, chunk: Optional[NitscheChunk] = None):
        """Values dict for the full Nitsche boundary integrand.

        When *chunk* is given, returns that chunk's per-QP arrays.
        Otherwise returns the flat (non-chunked) arrays.
        """
        _sd = self._integrand_scalar_dtype
        if chunk is not None:
            return {
                "mu": _sd(self.mu),
                "kappa": _sd(self.kappa),
                "phi_arr": chunk.phi_qp,
                "grad_phi_arr": chunk.grad_phi_qp,
                "grad_phi_mag_arr": chunk.grad_phi_mag_qp,
                "beta_arr": chunk.beta_qp,
                "u_bar_arr": chunk.u_bar_qp,
                "t_bar_arr": chunk.t_bar_qp,
                "psi_D_arr": chunk.psi_D_qp,
                "psi_N_arr": chunk.psi_N_qp,
                "n_face_arr": chunk.n_face_qp,
                "qp_active_arr": chunk.qp_active_wp,
                "body_load_scale": _sd(self.current_load_scale),
            }
        return {
            "mu": _sd(self.mu),
            "kappa": _sd(self.kappa),
            "phi_arr": self._phi_qp,
            "grad_phi_arr": self._grad_phi_qp,
            "grad_phi_mag_arr": self._grad_phi_mag_qp,
            "beta_arr": self._beta_qp,
            "u_bar_arr": self._u_bar_qp,
            "t_bar_arr": self._t_bar_qp,
            "psi_D_arr": self._psi_D_qp,
            "psi_N_arr": self._psi_N_qp,
            "n_face_arr": self._n_face_qp,
            "qp_active_arr": self._qp_active_wp,
            "body_load_scale": _sd(self.current_load_scale),
        }

    def _add_to_residual(self, R, tape=None):
        """Add full Nitsche on boundary subdomain (phi_weight baked into standard QP weights).

        When *tape* is not None, wraps each fem.integrate call inside
        ``with tape:`` so the computation graph is recorded for backward.
        """
        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            for chunk in self._nitsche_chunks:
                kwargs = dict(
                    output=R,
                    add=True,
                    fields={
                        "u": self.u,
                        "v": chunk.test,
                        "e": self._current_preprocessed_e,
                        "material": getattr(self, "material_field", self.e),
                    },
                    values=self._nitsche_values(chunk),
                    quadrature=chunk.quadrature,
                    accumulate_dtype=self._integrand_scalar_dtype,
                )
                if tape is not None:
                    with tape:
                        fem.integrate(self._boundary_stress_integrand, **kwargs)
                else:
                    fem.integrate(self._boundary_stress_integrand, **kwargs)
        else:
            kwargs = dict(
                output=R,
                add=True,
                fields={
                    "u": self.u,
                    "v": self._correction_test,
                    "e": self._current_preprocessed_e,
                    "material": getattr(self, "material_field", self.e),
                },
                values=self._nitsche_values(),
                quadrature=self._correction_quadrature,
                accumulate_dtype=self._integrand_scalar_dtype,
            )
            if tape is not None:
                with tape:
                    fem.integrate(self._boundary_stress_integrand, **kwargs)
            else:
                fem.integrate(self._boundary_stress_integrand, **kwargs)

    def _add_to_tangent(self, K):
        """Add full Nitsche on boundary subdomain (phi_weight baked into standard QP weights)."""
        skip_d2P_flag = wp.int32(1 if self.params.get("skip_d2P", True) else 0)
        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            for chunk in self._nitsche_chunks:
                tangent_values = self._nitsche_values(chunk)
                tangent_values.pop("body_load_scale")
                tangent_values["skip_d2P_flag"] = skip_d2P_flag
                fem.integrate(
                    self._boundary_tangent_integrand,
                    output=K,
                    add=True,
                    fields={
                        "u": self.u,
                        "du": chunk.trial,
                        "v": chunk.test,
                        "e": self._current_preprocessed_e,
                        "material": getattr(self, "material_field", self.e),
                    },
                    values=tangent_values,
                    quadrature=chunk.quadrature,
                    accumulate_dtype=self._tangent_accumulate_dtype(),
                )
        else:
            tangent_values = self._nitsche_values()
            tangent_values.pop("body_load_scale")
            tangent_values["skip_d2P_flag"] = skip_d2P_flag
            fem.integrate(
                self._boundary_tangent_integrand,
                output=K,
                add=True,
                fields={
                    "u": self.u,
                    "du": self._correction_trial,
                    "v": self._correction_test,
                    "e": self._current_preprocessed_e,
                    "material": getattr(self, "material_field", self.e),
                },
                values=tangent_values,
                quadrature=self._correction_quadrature,
                accumulate_dtype=self._tangent_accumulate_dtype(),
            )

    def _tangent_accumulate_dtype(self):
        """Use float64 tangent accumulation for Nitsche to preserve d²P precision."""
        if self.params.get("tangent_f64", False):
            return wp.float64
        return wp.float32

    def _post_tangent_assembly(self, K):
        """Keep float64 tangent as-is — no conversion needed.

        The condensation and solver paths now handle float64 blocks natively.
        AMGX already operates in dDDI (float64) mode, so passing float64 K
        directly avoids the lossy float64→float32→float64 roundtrip.
        """
        return K

    def _pre_tape_residual(self, tape):
        """Re-run phi/grad_phi computation on tape so backward flows to self.e.

        For the non-refined path, f64 design values are cast to the f32 QP
        kernel input on tape; native f32 design values are used directly.

        For the refined path: casts parent SDF f64→f32 on tape when needed, uses
        precomputed child SDF (from geometric_func, more accurate than
        interpolating from parent), concatenates, and runs precompute kernel
        on tape. After tape.backward(), child SDF gradients are accumulated
        to parent nodes via _post_tape_backward().
        """
        sdf = self.e.dof_values
        native_f32 = sdf.dtype == wp.float32

        has_refinement = hasattr(self, "_n_parent_nodes")

        if has_refinement:
            # Refined path: use precomputed child SDF from geometric_func
            n_parent = self._n_parent_nodes
            n_child = self._child_sdf_precomputed.shape[0]
            n_combined = n_parent + n_child

            parent_sdf_f32 = sdf if native_f32 else wp.zeros(
                n_parent, dtype=wp.float32, requires_grad=True, device=self.device
            )
            combined_sdf = wp.zeros(
                n_combined,
                dtype=wp.float32,
                requires_grad=True,
                device=self.device,
            )
            self._child_sdf_precomputed.requires_grad = True
            with tape:
                if not native_f32:
                    wp.launch(
                        cast_sdf_f64_to_f32,
                        dim=n_parent,
                        inputs=[sdf],
                        outputs=[parent_sdf_f32],
                        device=self.device,
                    )
                wp.launch(
                    concat_sdf_arrays,
                    dim=n_combined,
                    inputs=[
                        parent_sdf_f32,
                        self._child_sdf_precomputed,
                        n_parent,
                    ],
                    outputs=[combined_sdf],
                    device=self.device,
                )
            # Store for post-backward gradient accumulation
            self._child_sdf_tape = self._child_sdf_precomputed
            sdf_for_kernel = combined_sdf
        else:
            # Non-refined path: simple cast
            if native_f32:
                sdf_for_kernel = sdf
            else:
                n_nodes = sdf.shape[0]
                sdf_f32 = wp.zeros(
                    n_nodes, dtype=wp.float32, requires_grad=True, device=self.device
                )
                with tape:
                    wp.launch(
                        cast_sdf_f64_to_f32,
                        dim=n_nodes,
                        inputs=[sdf],
                        outputs=[sdf_f32],
                        device=self.device,
                    )
                sdf_for_kernel = sdf_f32

        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            for chunk in self._nitsche_chunks:
                total = chunk.n_elements * chunk.qp_per_cell
                chunk.phi_qp = wp.zeros(
                    total, dtype=wp.float32, requires_grad=True, device=self.device
                )
                chunk.grad_phi_qp = wp.zeros(
                    total, dtype=wp.vec3, requires_grad=True, device=self.device
                )
                chunk.grad_phi_mag_qp = wp.zeros(
                    total, dtype=wp.float32, requires_grad=True, device=self.device
                )
                with tape:
                    wp.launch(
                        precompute_qp_phi_grad,
                        dim=total,
                        inputs=[
                            chunk.qp_phi_source_elem_wp,
                            chunk.qp_phi_ref_coords_wp,
                            chunk.qp_active_wp,
                            self._source_elements_wp,
                            self._source_node_coords_wp,
                            sdf_for_kernel,
                            self._epsilon,
                        ],
                        outputs=[
                            chunk.phi_qp,
                            chunk.grad_phi_qp,
                            chunk.grad_phi_mag_qp,
                        ],
                        device=self.device,
                    )
        else:
            total = self._n_qp_elements * self._qp_per_cell
            self._phi_qp = wp.zeros(
                total, dtype=wp.float32, requires_grad=True, device=self.device
            )
            self._grad_phi_qp = wp.zeros(
                total, dtype=wp.vec3, requires_grad=True, device=self.device
            )
            self._grad_phi_mag_qp = wp.zeros(
                total, dtype=wp.float32, requires_grad=True, device=self.device
            )
            with tape:
                wp.launch(
                    precompute_qp_phi_grad,
                    dim=total,
                    inputs=[
                        self._qp_phi_source_elem_wp,
                        self._qp_phi_ref_coords_wp,
                        self._qp_active_wp,
                        self._source_elements_wp,
                        self._source_node_coords_wp,
                        sdf_for_kernel,
                        self._epsilon,
                    ],
                    outputs=[
                        self._phi_qp,
                        self._grad_phi_qp,
                        self._grad_phi_mag_qp,
                    ],
                    device=self.device,
                )

    def _post_tape_backward(self):
        """Accumulate child SDF gradients to parent nodes after tape.backward().

        In the refined path, child SDF values come from geometric_func (not
        interpolated from parent nodes), so tape.backward() produces gradients
        on the child SDF array but they don't flow to self.e.dof_values.
        This method scatters those gradients to parent nodes using trilinear
        interpolation weights: grad_parent[k] += Σ_j N_k(ξ_j) · grad_child[j].
        """
        if not hasattr(self, "_child_sdf_tape") or self._child_sdf_tape is None:
            return
        child_grad = self._child_sdf_tape.grad
        if child_grad is None:
            self._child_sdf_tape = None
            return
        wp.launch(
            accumulate_child_grad_to_parent,
            dim=self._child_sdf_tape.shape[0],
            inputs=[
                child_grad,
                self._child_parent_elem_wp,
                self._child_ref_coords_wp,
                self._parent_elements_wp,
                self.e.dof_values.grad,
            ],
            device=self.device,
        )
        self._child_sdf_tape = None

    def _pre_newton_iteration(self, iteration):
        # Scale Nitsche u_bar / t_bar by current_load_scale (matches how the
        # base class scales standard Dirichlet values and Neumann loads).
        if iteration == 0:
            self._scale_nitsche_bc_fields()
        if self.params.get("update_beta_per_newton", False):
            self._compute_stabilization_beta(use_current_u=True)

    def _scale_nitsche_bc_fields(self):
        """Scale u_bar and t_bar QP arrays by current_load_scale for load stepping."""

        scale = self.current_load_scale

        if hasattr(self, "_nitsche_chunks") and self._nitsche_chunks:
            # Chunked mode: scale each chunk's u_bar/t_bar independently
            if not hasattr(self, "_chunk_bc_bases"):
                self._chunk_bc_bases = []
                for chunk in self._nitsche_chunks:
                    self._chunk_bc_bases.append(
                        (
                            wp.to_torch(chunk.u_bar_qp).clone(),
                            wp.to_torch(chunk.t_bar_qp).clone(),
                        )
                    )
            for chunk, (u_base, t_base) in zip(
                self._nitsche_chunks, self._chunk_bc_bases
            ):
                wp.to_torch(chunk.u_bar_qp).copy_(u_base * scale)
                wp.to_torch(chunk.t_bar_qp).copy_(t_base * scale)
        else:
            # Flat mode
            if not hasattr(self, "_u_bar_qp_base_torch"):
                self._u_bar_qp_base_torch = wp.to_torch(self._u_bar_qp).clone()
                self._t_bar_qp_base_torch = wp.to_torch(self._t_bar_qp).clone()
            wp.to_torch(self._u_bar_qp).copy_(self._u_bar_qp_base_torch * scale)
            wp.to_torch(self._t_bar_qp).copy_(self._t_bar_qp_base_torch * scale)

    # ---- VTU output hooks ----

    def _snapshot_extra(self) -> dict:
        extra = super()._snapshot_extra()
        extra["phi"] = self._phi_nodal.cpu().numpy()
        extra["sdf"] = self._sdf_nodal.cpu().numpy()
        extra["beta"] = self._beta_element.cpu().numpy()
        extra["psi_D"] = self._psi_D_nodal.cpu().numpy()
        extra["psi_N"] = self._psi_N_nodal.cpu().numpy()
        # BC selector bbox masks (one float32 field per resolved SurfaceSelector key).
        # Computed on the main thread alongside the cheap GPU→CPU copies above so
        # the background writer can use them without touching 'self'.
        from .bc_surface_selectors import SurfaceSelector as _SS
        all_bcs = (
            [("dirichlet", k, i)
             for i, k in enumerate(self._original_dirichlet_bcs_dict)]
            + [("neumann", k, i)
               for i, k in enumerate(self._original_neumann_bcs_dict)]
        )
        has_sel = any(isinstance(k, _SS) and k.is_resolved for _, k, _ in all_bcs)
        if has_sel:
            pts = self._mesh_cache.get("points")
            if pts is not None:
                # Evaluate on whatever device the mesh already lives on, and
                # bring back only the mask; ``compute_unsigned_dist`` moves the
                # culled candidates to ``_bvh_device`` itself.
                pts_f = pts.float()
                x, y, z = pts_f[:, 0], pts_f[:, 1], pts_f[:, 2]
                import numpy as _np
                for bc_type, k, idx in all_bcs:
                    if isinstance(k, _SS) and k.is_resolved:
                        bbox_mask = k(x, y, z).cpu().numpy().astype(_np.float32)
                        extra[f"_bc_{bc_type}_{idx}_bbox"] = bbox_mask
        return extra

    def _get_point_data(self, point_data: dict, extra: dict) -> dict:
        point_data = super()._get_point_data(point_data, extra)
        point_data["phi"] = extra["phi"]
        point_data["sdf"] = extra["sdf"]
        point_data["psi_D"] = extra["psi_D"]
        point_data["psi_N"] = extra["psi_N"]
        # BC selector regions: bbox mask AND diffuse Nitsche boundary (psi > 0)
        import numpy as _np
        for key, val in extra.items():
            if not key.startswith("_bc_"):
                continue
            # key format: "_bc_{type}_{idx}_bbox"
            parts = key.split("_")  # ['', 'bc', type, idx, 'bbox']
            bc_type, idx = parts[2], parts[3]
            psi = extra["psi_D"] if bc_type == "dirichlet" else extra["psi_N"]
            region = (val * (psi > 0).astype(_np.float32))
            point_data[f"bc_{bc_type}_{idx}_bbox"] = val
            point_data[f"bc_{bc_type}_{idx}_region"] = region
        return point_data

    def _get_cell_data(self, cell_data: dict, extra: dict) -> dict:
        cell_data = super()._get_cell_data(cell_data, extra)
        cell_data["beta"] = extra["beta"]
        return cell_data
