"""Topology optimization using density-based approach with Helmholtz PDE filter.

Provides TopoOpt class which inherits NeoHookean and implements:
- Helmholtz PDE density filter (assembled once per mesh, solved per step)
- Tanh SIMP projection for regularised Heaviside
- Compliance-weighted objective function (rho^6 * strain energy)
- Volume fraction constraint
"""

import warp as wp
import warp.fem as fem

from .neoHookean import NeoHookean
from .utils import (
    solve_linear_system,
    _simp_projection_kernel,
    _simp_projection_kernel_f64,
    _axpy_f32,
    _axpy_f64,
    _mask_inplace,
)


# ---------------------------------------------------------------------------
# Integrands
# ---------------------------------------------------------------------------


def _make_helmholtz_bilinear(scalar_dtype=wp.float32):
    """Create Helmholtz filter bilinear integrand with the given scalar type."""

    @fem.integrand
    def helmholtz_bilinear(
        s: fem.Sample,
        u: fem.Field,
        v: fem.Field,
        rf_sq: scalar_dtype,
    ):
        """Helmholtz filter bilinear form: u*v + rf^2 * grad(u) . grad(v)."""
        return u(s) * v(s) + rf_sq * wp.dot(fem.grad(u, s), fem.grad(v, s))

    return helmholtz_bilinear


@fem.integrand
def _helmholtz_rhs(s: fem.Sample, e: fem.Field, v: fem.Field):
    """Helmholtz filter RHS: e * v (mass-type)."""
    return e(s) * v(s)


def _make_compliance_integrand(n: int, scalar_dtype=wp.float32):
    """Create a compliance integrand that returns rho^6 * W(F) for dimension n."""
    _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
    _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
    _half = scalar_dtype(0.5)
    _three = scalar_dtype(3.0)

    @fem.integrand
    def compliance(
        s: fem.Sample,
        u: fem.Field,
        e_proc: fem.Field,
        mu: scalar_dtype,
        kappa: scalar_dtype,
    ):
        F = wp.identity(n=n, dtype=scalar_dtype) + fem.grad(u, s)
        J = wp.determinant(F)
        I1 = wp.ddot(F, F) + _dim_offset
        J_m23 = wp.pow(J, _neg_two_thirds)
        lnJ = wp.log(J)
        W = _half * mu * (J_m23 * I1 - _three) + _half * kappa * lnJ * lnJ
        rho = e_proc(s)
        return rho * rho * rho * rho * rho * rho * W

    return compliance


@fem.integrand
def _density_integrand(s: fem.Sample, e_proc: fem.Field):
    """Volume integral of density: integral rho dx."""
    return e_proc(s)


# ---------------------------------------------------------------------------
# TopoOpt class
# ---------------------------------------------------------------------------


class TopoOpt(NeoHookean):
    """Density-based topology optimization with Neo-Hookean material.

    Preprocessing chain: raw e -> Helmholtz PDE filter -> tanh SIMP projection.
    Objective: integral of preprocessed_e^6 * strain energy density.
    Constraint: integral(preprocessed_e)/volume - target_volume_fraction <= 0.
    """

    # ------------------------------------------------------------------
    # SIMP-penalized integrands (rho^6 scaling)
    # ------------------------------------------------------------------

    @staticmethod
    def _make_stress_integrand(n: int, scalar_dtype=wp.float32):
        _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
        _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
        _one_third = scalar_dtype(1.0 / 3.0)

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

            rho = e(s)
            return rho * rho * rho * rho * rho * rho * wp.ddot(P, fem.grad(v, s))

        return stress

    @staticmethod
    def _make_tangent_integrand(n: int, scalar_dtype=wp.float32):
        _dim_offset = scalar_dtype(0.0 if n == 3 else 1.0)
        _neg_two_thirds = scalar_dtype(-2.0 / 3.0)
        _two_thirds = scalar_dtype(2.0 / 3.0)
        _one_third = scalar_dtype(1.0 / 3.0)
        _two = scalar_dtype(2.0)

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

            rho = e(s)
            return rho * rho * rho * rho * rho * rho * wp.ddot(dP, grad_v)

        return tangent

    @property
    def density_filter_radius(self):
        return float(self.params.get("density_filter_radius", 0.01))

    @property
    def target_volume_fraction(self):
        return float(self.params.get("target_volume_fraction", 0.5))

    @property
    def _domain_volume(self):
        """Total volume of the computational domain."""
        if self.dim == 2:
            return float(self.params.width) * float(self.params.height)
        return (
            float(self.params.width)
            * float(self.params.height)
            * float(self.params.depth)
        )

    def _init_solver_cache(self):
        """Extend parent solver cache with Helmholtz filter infrastructure."""
        super()._init_solver_cache()
        mc = self._mesh_cache

        # Determine scalar dtype from the native geometry precision.
        _use_f64 = self.use_mixed_precision and self.dim in (2, 3)
        _scalar_dtype = wp.float64 if _use_f64 else wp.float32

        # Helmholtz filter: test/trial on scalar e_space
        mc["helm_test"] = fem.make_test(space=self.e_space, domain=mc["domain"])
        mc["helm_trial"] = fem.make_trial(space=self.e_space, domain=mc["domain"])

        # Build Helmholtz bilinear integrand with correct scalar type
        _helmholtz_bilinear_fn = _make_helmholtz_bilinear(_scalar_dtype)
        mc["helmholtz_bilinear"] = _helmholtz_bilinear_fn

        # Assemble Helmholtz stiffness (constant for a given mesh)
        rf = self.density_filter_radius
        _helm_kw = {}
        if _use_f64:
            _helm_kw["accumulate_dtype"] = wp.float64
        mc["helm_K"] = fem.integrate(
            _helmholtz_bilinear_fn,
            fields={"u": mc["helm_trial"], "v": mc["helm_test"]},
            values={"rf_sq": _scalar_dtype(rf * rf)},
            **_helm_kw,
        )
        mc["helm_solve_cache"] = {}
        mc["_scalar_dtype"] = _scalar_dtype

        # Prebuild compliance integrand (dimension-specific)
        self._compliance_integrand = _make_compliance_integrand(
            self.dim, scalar_dtype=_scalar_dtype
        )

    def _bounds(self):
        return (0.001, 1.0)

    # ------------------------------------------------------------------
    # Preprocessing: Helmholtz filter + tanh SIMP projection
    # ------------------------------------------------------------------

    def _preprocess_optimized(self, tape=None):
        """Apply Helmholtz PDE filter + tanh SIMP projection to self.e."""
        mc = self._mesh_cache
        n_e = self.e_space.node_count()
        _use_f64 = self.e.dof_values.dtype is wp.float64
        _accum_dtype = wp.float64 if _use_f64 else wp.float32

        # --- Step 1: Helmholtz filter ---
        # Assemble RHS: integral(e * v dx)
        _rhs_kw = {}
        if _use_f64:
            _rhs_kw["accumulate_dtype"] = _accum_dtype
        rhs = fem.integrate(
            _helmholtz_rhs,
            fields={"e": self.e, "v": mc["helm_test"]},
            output_dtype=_accum_dtype,
            **_rhs_kw,
        )

        self.log(
            f"Applying Helmholtz filter with radius {self.density_filter_radius}..."
        )
        # Solve: helm_K * filtered_dof = rhs
        filtered_dof = solve_linear_system(
            mc["helm_K"],
            rhs,
            method="cg",
            device=self.device,
            dim=1,
            cg_backend=str(self.params.get("cg_backend_assembled", "amgx")).lower(),
            _cache=mc["helm_solve_cache"],
            use_mixed_precision=_use_f64,
        )

        # Write into filtered field
        if not hasattr(self, "_filtered_density_field"):
            self._filtered_density_field = self.e_space.make_field()
        wp.utils.array_cast(
            in_array=filtered_dof,
            out_array=self._filtered_density_field.dof_values,
        )
        filtered_density = self._filtered_density_field

        # Record adjoint for filter (if tape)
        if tape is not None:
            filtered_density.dof_values.requires_grad = True
            _helm_K = mc["helm_K"]
            _helm_cache = mc["helm_solve_cache"]
            _device = self.device
            _e_dof = self.e.dof_values
            _helm_test = mc["helm_test"]
            _n_e = n_e
            _cg_backend = str(self.params.get("cg_backend_assembled", "amgx")).lower()
            _filtered = filtered_density

            def _filter_adjoint():
                # d_loss/d_filtered: same dtype as density
                adj_rhs = _filtered.dof_values.grad

                self.log("Applying Helmholtz filter adjoint...")

                # Solve helm_K * lam = adj_rhs (symmetric)
                lam = solve_linear_system(
                    _helm_K,
                    adj_rhs,
                    method="cg",
                    device=_device,
                    dim=1,
                    cg_backend=_cg_backend,
                    _cache=_helm_cache,
                    use_mixed_precision=_use_f64,
                )

                # d_loss/d_e = M * lam where M is the mass part of helm_K.
                # Assemble: integral(lam_field * v dx)
                # NOTE: use a separate field — do NOT reuse _filtered here.
                # Reusing _filtered would overwrite _current_preprocessed_e.dof_values
                # with lam, corrupting the VTU Preprocessed_optimized output.
                if not hasattr(self, "_lam_density_field"):
                    self._lam_density_field = self.e_space.make_field()
                lam_field = self._lam_density_field
                wp.utils.array_cast(in_array=lam, out_array=lam_field.dof_values)

                _lam_kw = {}
                if _use_f64:
                    _lam_kw["accumulate_dtype"] = _accum_dtype
                e_grad_contribution = fem.integrate(
                    _helmholtz_rhs,
                    fields={"e": lam_field, "v": _helm_test},
                    output_dtype=_accum_dtype,
                    **_lam_kw,
                )

                if _use_f64:
                    # Density grad is float64 — add directly
                    wp.launch(
                        _axpy_f64,
                        dim=_n_e,
                        inputs=[e_grad_contribution, _e_dof.grad, wp.float64(1.0)],
                        device=_device,
                    )
                else:
                    # Cast f64 -> f32 and add to e.grad
                    e_grad_f32 = wp.zeros(_n_e, dtype=wp.float32, device=_device)
                    wp.utils.array_cast(
                        in_array=e_grad_contribution, out_array=e_grad_f32
                    )
                    wp.launch(
                        _axpy_f32,
                        dim=_n_e,
                        inputs=[e_grad_f32, _e_dof.grad, wp.float32(1.0)],
                        device=_device,
                    )

            tape.record_func(
                _filter_adjoint,
                arrays=[filtered_density.dof_values, self.e.dof_values],
            )

        # --- Step 2: tanh SIMP projection ---
        if not hasattr(self, "_projected_density_field"):
            self._projected_density_field = self.e_space.make_field()
        projected = self._projected_density_field
        if tape is not None:
            projected.dof_values.requires_grad = True

        if _use_f64:
            beta = wp.float64(float(self.params.get("simp_beta", 3.0)))
            eta = wp.float64(float(self.params.get("simp_eta", 0.5)))
            _proj_kernel = _simp_projection_kernel_f64
        else:
            beta = wp.float32(float(self.params.get("simp_beta", 3.0)))
            eta = wp.float32(float(self.params.get("simp_eta", 0.5)))
            _proj_kernel = _simp_projection_kernel

        if tape is not None:
            with tape:
                wp.launch(
                    _proj_kernel,
                    dim=n_e,
                    inputs=[
                        filtered_density.dof_values,
                        projected.dof_values,
                        beta,
                        eta,
                    ],
                    device=self.device,
                )
        else:
            wp.launch(
                _proj_kernel,
                dim=n_e,
                inputs=[
                    filtered_density.dof_values,
                    projected.dof_values,
                    beta,
                    eta,
                ],
                device=self.device,
            )

        # Store for VTU output
        self._filtered_density = filtered_density
        return projected

    # ------------------------------------------------------------------
    # Objective: integral of rho^6 * W (compliance-weighted)
    # ------------------------------------------------------------------

    def _objective_function(self, tape):
        """Compute integral of preprocessed_e^6 * strain_energy_density.

        Must be called inside ``with tape:`` context.
        Returns wp.array shape (1,) with requires_grad=True.
        """
        mc = self._mesh_cache
        preprocessed_e = getattr(self, "_current_preprocessed_e", self.e)

        obj = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=self.device)
        _scalar_dtype = mc.get("_scalar_dtype", wp.float32)
        _use_f64 = _scalar_dtype is wp.float64
        _int_kw = dict(
            fields={"u": self.u, "e_proc": preprocessed_e},
            values={"mu": _scalar_dtype(self.mu), "kappa": _scalar_dtype(self.kappa)},
            domain=mc["domain"],
            output=obj,
        )
        if _use_f64:
            _int_kw["accumulate_dtype"] = wp.float64
        fem.integrate(self._compliance_integrand, **_int_kw)
        return self._normalized_scalar(obj, "compliance")

    # ------------------------------------------------------------------
    # Constraint: volume fraction
    # ------------------------------------------------------------------

    def _constraints(self):
        """Return [volume_constraint] callable list."""

        def volume_constraint(tape):
            mc = self._mesh_cache
            preprocessed_e = getattr(self, "_current_preprocessed_e", self.e)

            _use_f64 = preprocessed_e.dof_values.dtype is wp.float64
            vol = wp.zeros(1, dtype=wp.float64, requires_grad=True, device=self.device)
            _vol_kw = dict(
                fields={"e_proc": preprocessed_e},
                output=vol,
                quadrature=fem.RegularQuadrature(domain=mc["domain"], order=2),
            )
            if _use_f64:
                _vol_kw["accumulate_dtype"] = wp.float64
            fem.integrate(_density_integrand, **_vol_kw)

            # constraint = vol / domain_volume - target_vf
            domain_vol = self._domain_volume
            target_vf = self.target_volume_fraction

            result = wp.zeros(
                1, dtype=wp.float64, requires_grad=True, device=self.device
            )
            wp.launch(
                _volume_constraint_kernel,
                dim=1,
                inputs=[vol, result, wp.float64(domain_vol), wp.float64(target_vf)],
                device=self.device,
            )
            return result

        return [volume_constraint]

    # ------------------------------------------------------------------
    # VTU output
    # ------------------------------------------------------------------

    def _snapshot_extra(self) -> dict:
        """Snapshot filtered density on the main thread."""
        extra = super()._snapshot_extra()
        if hasattr(self, "_filtered_density"):
            extra["Filtered_density"] = self._filtered_density.dof_values.numpy().copy()
        return extra

    def _get_point_data(self, point_data: dict, extra: dict) -> dict:
        """Add Filtered_density to point output."""
        if "Filtered_density" in extra:
            point_data["Filtered_density"] = extra["Filtered_density"]
        return point_data


# ---------------------------------------------------------------------------
# Helper kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _volume_constraint_kernel(
    vol: wp.array(dtype=wp.float64),
    result: wp.array(dtype=wp.float64),
    domain_vol: wp.float64,
    target_vf: wp.float64,
):
    """result[0] = vol[0] / domain_vol - target_vf."""
    result[0] = vol[0] / domain_vol - target_vf
