import warp as wp
import warp.fem as fem
import numpy as np
import torch
from .abstract_field import AbstractField, OctreeField
from .vtu_writer import compute_node_gradients


class NeoHookean(AbstractField):
    @staticmethod
    def _make_stress_integrand(n: int, scalar_dtype=wp.float32,
                               spatial_material: bool = False):
        _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
        _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
        _one_third = scalar_dtype(1.0 / 3.0)

        if spatial_material:
            @fem.integrand
            def spatial_stress(
                s: fem.Sample, v: fem.Field, u: fem.Field, e: fem.Field,
                material: fem.Field, mu: scalar_dtype, kappa: scalar_dtype,
            ):
                props = fem.inner(material, s)
                E_qp = scalar_dtype(props[0])
                nu_qp = scalar_dtype(props[1])
                mu_qp = E_qp / (scalar_dtype(2.0) * (scalar_dtype(1.0) + nu_qp))
                kappa_qp = E_qp / (scalar_dtype(3.0) * (scalar_dtype(1.0) - scalar_dtype(2.0) * nu_qp))
                F = wp.identity(n=n, dtype=scalar_dtype) + fem.grad(u, s)
                J = wp.determinant(F)
                F_inv_T = wp.transpose(wp.inverse(F))
                I1 = wp.ddot(F, F) + _dim_offset
                iso = mu_qp * wp.pow(J, _neg_two_thirds)
                P = iso * F + (kappa_qp * wp.log(J) - iso * I1 * _one_third) * F_inv_T
                return wp.ddot(P, fem.grad(v, s))
            return spatial_stress

        @fem.integrand
        def stress(
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

            return wp.ddot(P, fem.grad(v, s))

        return stress

    @staticmethod
    def _make_tangent_integrand(n: int, scalar_dtype=wp.float32,
                                spatial_material: bool = False):
        _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
        _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
        _two_thirds = scalar_dtype(2.0 / 3.0)
        _one_third = scalar_dtype(1.0 / 3.0)
        _two = scalar_dtype(2.0)

        if spatial_material:
            @fem.integrand
            def spatial_tangent(
                s: fem.Sample, v: fem.Field, u: fem.Field, du: fem.Field,
                e: fem.Field, material: fem.Field,
                mu: scalar_dtype, kappa: scalar_dtype,
            ):
                props = fem.inner(material, s)
                E_qp = scalar_dtype(props[0])
                nu_qp = scalar_dtype(props[1])
                mu_qp = E_qp / (scalar_dtype(2.0) * (scalar_dtype(1.0) + nu_qp))
                kappa_qp = E_qp / (scalar_dtype(3.0) * (scalar_dtype(1.0) - scalar_dtype(2.0) * nu_qp))
                F = wp.identity(n=n, dtype=scalar_dtype) + fem.grad(u, s)
                J = wp.determinant(F)
                F_inv = wp.inverse(F)
                F_inv_T = wp.transpose(F_inv)
                dF = fem.grad(du, s)
                grad_v = fem.grad(v, s)
                I1 = wp.ddot(F, F) + _dim_offset
                J_m23 = wp.pow(J, _neg_two_thirds)
                lnJ = wp.log(J)
                tr_Finv_dF = wp.ddot(F_inv_T, dF)
                dJ23 = _neg_two_thirds * J_m23 * tr_Finv_dF
                dI1 = _two * wp.ddot(F, dF)
                dF_inv_T = -(F_inv_T * wp.transpose(dF) * F_inv_T)
                third_I1 = I1 * _one_third
                A = F - third_I1 * F_inv_T
                dA = dF - (dI1 * _one_third) * F_inv_T - third_I1 * dF_inv_T
                dP = mu_qp * (dJ23 * A + J_m23 * dA) + kappa_qp * (
                    tr_Finv_dF * F_inv_T + lnJ * dF_inv_T
                )
                return wp.ddot(dP, grad_v)
            return spatial_tangent

        @fem.integrand
        def tangent(
            s: fem.Sample,
            v: fem.Field,
            u: fem.Field,
            du: fem.Field,
            e: fem.Field,
            mu: scalar_dtype,
            kappa: scalar_dtype,
        ):
            F = wp.identity(n=n, dtype=scalar_dtype) + fem.grad(u, s)
            J = wp.determinant(F)
            F_inv = wp.inverse(F)
            F_inv_T = wp.transpose(F_inv)
            dF = fem.grad(du, s)
            grad_v = fem.grad(v, s)

            I1 = wp.ddot(F, F) + _dim_offset
            J_m23 = wp.pow(J, _neg_two_thirds)
            lnJ = wp.log(J)

            tr_Finv_dF = wp.ddot(F_inv_T, dF)
            dJ23 = _neg_two_thirds * J_m23 * tr_Finv_dF
            dI1 = _two * wp.ddot(F, dF)

            dF_inv_T = -(F_inv_T * wp.transpose(dF) * F_inv_T)

            third_I1 = I1 * _one_third
            A = F - third_I1 * F_inv_T
            dA = dF - (dI1 * _one_third) * F_inv_T - third_I1 * dF_inv_T

            dP = mu * (dJ23 * A + J_m23 * dA) + kappa * (
                tr_Finv_dF * F_inv_T + lnJ * dF_inv_T
            )

            return wp.ddot(dP, grad_v)

        return tangent

    def _strain_energy_density(self, gradients=None, material_values=None):
        """Return nodal strain-energy density for the current configuration.
        GPU-accelerated via batched torch linalg.

        Args:
            gradients: torch.Tensor (N, dim, dim) on GPU, or None to compute.

        Returns:
            torch.Tensor (N,) on GPU with strain energy density per node.
        """
        if gradients is None:
            gradients = compute_node_gradients(self.u, self.geo, self.space, self.dim)
        dim = self.dim
        device = gradients.device
        N = gradients.shape[0]
        I = (
            torch.eye(dim, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(N, -1, -1)
        )

        F = I + gradients  # (N, dim, dim)
        J = torch.linalg.det(F)  # (N,)
        C = torch.bmm(F.mT, F)  # (N, dim, dim)
        I1 = C.diagonal(dim1=-2, dim2=-1).sum(-1)  # (N,)
        if dim == 2:
            I1 = I1 + 1.0
        J_m23 = J.pow(-2.0 / 3.0)

        if material_values is None:
            mu, kappa = self.mu, self.kappa
        else:
            material = torch.as_tensor(
                material_values, dtype=gradients.dtype, device=device
            )
            mu = material[:, 0] / (2.0 * (1.0 + material[:, 1]))
            kappa = material[:, 0] / (3.0 * (1.0 - 2.0 * material[:, 1]))
        energy = 0.5 * mu * (J_m23 * I1 - 3.0) + 0.5 * kappa * J.log().pow(2)
        return energy


class OctreeNeoHookean(OctreeField, NeoHookean):
    """NeoHookean solver for octree hex meshes with hanging nodes."""

    pass
