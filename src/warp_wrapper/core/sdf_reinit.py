"""SDF reinitialization to enforce the eikonal condition |∇φ| = 1.

Uses the Sussman–Smereka–Osher (1994) PDE-based method with Godunov
upwind scheme on a uniform structured grid. Runs on GPU via PyTorch.
"""

import torch


def reinitialize_sdf_sussman(
    sdf_3d,
    dx,
    dy,
    dz,
    n_iters=None,
    cfl=0.5,
    sign_phi0=None,
):
    """Reinitialize SDF via the Sussman–Smereka–Osher reinitialization PDE.

    Solves  ∂φ/∂τ + S(φ₀)(|∇φ| − 1) = 0  using the Godunov upwind scheme
    on a uniform structured grid.

    Args:
        sdf_3d: (Nx, Ny, Nz) torch tensor (any device).
        dx, dy, dz: grid spacing in each direction.
        n_iters: pseudo-time steps (default: ``max(Nx, Ny, Nz)``).
        cfl: CFL number for pseudo-time step (default 0.5).
        sign_phi0: optional reference SDF for the sign function S(φ₀).
            When calling in batches, pass the **original** (pre-reinit) SDF
            so that S stays fixed across batches.  If *None*, uses *sdf_3d*.

    Returns:
        Reinitialized SDF tensor of the same shape, dtype, and device.
    """
    phi = sdf_3d.to(dtype=torch.float64).clone()
    phi0 = phi.clone() if sign_phi0 is None else sign_phi0.to(dtype=torch.float64)

    h_min = min(dx, dy, dz)
    dt = cfl * h_min

    if n_iters is None:
        n_iters = max(phi.shape)

    # Smoothed sign function  S(φ₀) = φ₀ / √(φ₀² + h²)
    eps_s = h_min
    S = phi0 / torch.sqrt(phi0 ** 2 + eps_s ** 2)

    Sp = torch.clamp(S, min=0.0)
    Sm = torch.clamp(S, max=0.0)

    idx = 1.0 / dx
    idy = 1.0 / dy
    idz = 1.0 / dz

    z = torch.zeros_like  # shorthand

    for _ in range(n_iters):
        # ---- Finite differences (zero-gradient Neumann at boundaries) ----
        diff_x = (phi[1:, :, :] - phi[:-1, :, :]) * idx
        Dxm = torch.cat([z(phi[:1, :, :]), diff_x], dim=0)
        Dxp = torch.cat([diff_x, z(phi[:1, :, :])], dim=0)

        diff_y = (phi[:, 1:, :] - phi[:, :-1, :]) * idy
        Dym = torch.cat([z(phi[:, :1, :]), diff_y], dim=1)
        Dyp = torch.cat([diff_y, z(phi[:, :1, :])], dim=1)

        diff_z = (phi[:, :, 1:] - phi[:, :, :-1]) * idz
        Dzm = torch.cat([z(phi[:, :, :1]), diff_z], dim=2)
        Dzp = torch.cat([diff_z, z(phi[:, :, :1])], dim=2)

        # ---- Godunov upwind fluxes ----
        # G⁺ (for S > 0)
        ax_p = torch.maximum(Dxm.clamp(min=0.0), (-Dxp).clamp(min=0.0)) ** 2
        ay_p = torch.maximum(Dym.clamp(min=0.0), (-Dyp).clamp(min=0.0)) ** 2
        az_p = torch.maximum(Dzm.clamp(min=0.0), (-Dzp).clamp(min=0.0)) ** 2
        Gp = torch.sqrt(ax_p + ay_p + az_p) - 1.0

        # G⁻ (for S < 0)
        ax_m = torch.maximum((-Dxm).clamp(min=0.0), Dxp.clamp(min=0.0)) ** 2
        ay_m = torch.maximum((-Dym).clamp(min=0.0), Dyp.clamp(min=0.0)) ** 2
        az_m = torch.maximum((-Dzm).clamp(min=0.0), Dzp.clamp(min=0.0)) ** 2
        Gm = torch.sqrt(ax_m + ay_m + az_m) - 1.0

        # ---- Update ----
        phi -= dt * (Sp * Gp + Sm * Gm)

    return phi
