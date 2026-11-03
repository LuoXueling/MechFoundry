"""Diagnose how to best move SDF in the descent direction via latent updates.

After ``LatentShapeOpt`` upcasts the VAE to fp32, the production path applies
a unit-rescaled latent gradient (``Δlatent = -lr · sqrt(N) · g/||g||``) with
SGD+momentum.  This script answers two questions:

  1. Is the unit-rescaled gradient direction near-optimal, or can we find a
     latent step that gives a better cos(fem_grad, Δsdf)?
  2. How much of the residual cos(fem_grad, Δsdf) > -1 is decoder
     nonlinearity vs. column-space limits of J = ∂sdf/∂latent?

Approach:
  - Run a single LatentShapeOpt step to capture (fem_pts, fem_grad) and the
    raw latent gradient g = J^T · fem_grad (one VJP).
  - Sweep candidate latent step directions:
        plain SGD       Δl = -lr · g
        sign step       Δl = -lr · sign(g)             (Adam asymptote)
        unit direction  Δl = -lr · g / ||g||
        L2 rescaled     Δl = -lr · sqrt(N) · g/||g||   (production)
        Krylov GN       Δl ∈ argmin ||J·Δl + fem_grad||²
                        (over span{g, J^T J g, (J^T J)² g})
  - For each: take the step in fp32, recompute decoder SDF at the same
    fem_pts, and report cos(fem_grad, Δsdf_actual) plus a JVP-based linear
    prediction cos(fem_grad, J·Δl).
  - One fp16 cast probe to confirm the upcast was necessary.

Run from repo root:
    conda run -n mechfoundry-warp117 python src/warp_wrapper/tests/diagnose_grad_alignment.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "dependency", "TripoSG"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src", "warp_wrapper"))

import copy

import torch
import trimesh
import warp as wp

from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper
from core.latent_shape_opt import LatentShapeOpt


def _decoder_chain(latents, vae, query_pts):
    """Forward chain that mirrors ``compute_latent_grad`` exactly.

    Caller controls dtype: ``latents`` and ``query_pts`` should already be
    in the vae weight dtype (fp32 after ``LatentShapeOpt`` upcasts).
    """
    z = vae.post_quant(latents)
    hidden = z
    for block in vae.decoder.blocks[:-1]:
        hidden = block(hidden)
    kv = hidden
    queries = vae.embedder(query_pts)
    q = vae.decoder.proj_query(queries)
    last = vae.decoder.blocks[-1](q, encoder_hidden_states=kv)
    logits = vae.decoder.proj_out(vae.decoder.norm_out(last))
    sdf = (torch.sigmoid(-logits) * 2 - 1) / 100
    return sdf.squeeze(0).squeeze(-1).float()


def cos(a, b):
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item()


def main():
    device = "cuda"
    dtype = torch.float16
    weights = os.path.join(
        _REPO_ROOT, "src", "dependency", "TripoSG", "pretrained_weights", "TripoSG"
    )
    pipe = TripoSGPipelineWrapper.from_pretrained(weights).to(device, dtype)

    # Build a 4:1:1 box → encode to latents (matches the latent-shape example).
    # Uniform area-weighted surface sampling (subdivided box vertices cluster at
    # corners/edges and fragment the decode); seeded for reproducibility.
    mesh = trimesh.creation.box(extents=[1.75, 0.4375, 0.4375])
    points, face_idx = trimesh.sample.sample_surface(mesh, 100000, seed=42)
    surface = (
        torch.cat(
            [
                torch.FloatTensor(points.copy()),
                torch.FloatTensor(mesh.face_normals[face_idx].copy()),
            ],
            dim=-1,
        )
        .unsqueeze(0)
        .to(device, dtype=dtype)
    )
    with torch.no_grad():
        gen = torch.Generator(device="cpu").manual_seed(42)
        latents = pipe.vae.encode(surface, seed=42).latent_dist.sample(generator=gen)

    overrides = {
        "project_name": "diag_grad_alignment",
        "output_root": os.path.join(_HERE, "test_output"),
        "max_optimize_iters": 1,
        "output_per_optimize_step": 0,
        "use_mixed_precision": True,
        "async_io": True,
    }
    problem = LatentShapeOpt(
        "configs.config_shape_opt_cantilever",
        override_params=overrides,
        pipe=pipe,
        latents=latents,
        octree_depth=6,
        init_octree_depth=3,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        physical_extents=(4.0, "x"),
        latent_lr=1e-3,
        remesh_every=1,
        decoder_chunk_size=10000,
        epsilon_to_size_ratio=1.0,
    )

    # ------------------------------------------------------------------
    # 1. Run a single FEM step to populate _last_fem_pts/_last_fem_grad
    # ------------------------------------------------------------------
    problem.log("=== diag iter 0: forward + backward ===")
    tape = wp.Tape()
    problem.e.dof_values.requires_grad = True
    problem.step(tape=tape)
    with tape:
        obj = problem._objective_function(tape)
    tape.zero()
    problem.e.dof_values.grad.zero_()
    if problem._inner.e.dof_values.grad is not None:
        problem._inner.e.dof_values.grad.zero_()
    tape.backward(loss=obj)
    problem._post_tape_backward()

    # _compute_latent_grad sets _last_fem_pts/_last_fem_grad as a side effect.
    # Its return is the L2-rescaled (production) gradient — for diagnostics we
    # want the *raw* J^T·fem_grad too, so we re-call pipe.compute_latent_grad
    # below with the captured queries.
    _ = problem._compute_latent_grad(problem._last_e_grad)

    fem_pts_fp16 = problem._last_fem_pts            # (1, M, 3) fp16
    fem_grad = problem._last_fem_grad               # (M,) fp32

    # Production VAE is now fp32 (LatentShapeOpt upcasts it).
    vae = pipe.vae
    assert next(vae.parameters()).dtype == torch.float32, (
        "expected fp32 vae after LatentShapeOpt upcast"
    )

    # fp32 query points for the linear-probe forward chain.
    fem_pts_fp32 = fem_pts_fp16.float()

    latents_save = problem._latents.detach().clone()  # fp32
    M = fem_grad.numel()
    print(
        f"\nlatents shape={tuple(latents_save.shape)} dtype={latents_save.dtype}  "
        f"M_query={M}"
    )

    # ------------------------------------------------------------------
    # 2. Raw J^T · fem_grad via the production VJP path
    # ------------------------------------------------------------------
    g_raw = pipe.compute_latent_grad(
        problem._latents, fem_pts_fp16, fem_grad, chunk_size=10000
    ).detach()
    sqrtN = float(g_raw.numel()) ** 0.5
    print(
        f"raw latent_grad: norm={g_raw.norm().item():.4e}  "
        f"max|.|={g_raw.abs().max().item():.4e}  "
        f"nonzero={int((g_raw.abs() > 0).sum())}/{g_raw.numel()}"
    )

    # ------------------------------------------------------------------
    # 3. Helpers: forward eval (no_grad) and finite-difference JVP
    # ------------------------------------------------------------------
    def forward(latents_fp32):
        with torch.no_grad():
            return _decoder_chain(latents_fp32, vae, fem_pts_fp32)

    def jvp_fd(direction, eps=1e-3):
        """Finite-difference J·direction in the linearization at latents_save.

        Normalizes ``direction`` so the perturbation has fixed L2 norm,
        then unscales — keeps FD step in a regime where decoder
        nonlinearity is negligible (cos with true JVP ≈ 1).
        """
        n = direction.norm()
        if float(n) < 1e-30:
            return torch.zeros_like(fem_grad)
        u = direction / n
        sdf_p = forward(latents_save + eps * u)
        sdf_m = forward(latents_save - eps * u)
        return (sdf_p - sdf_m) / (2.0 * eps) * float(n)

    sdf_baseline = forward(latents_save)

    def step_and_measure(delta_latent):
        """Apply Δlatent in fp32, return (Δsdf_actual, Δsdf_predicted_linear)."""
        sdf_post = forward(latents_save + delta_latent)
        delta_actual = sdf_post - sdf_baseline
        delta_pred = jvp_fd(delta_latent)
        return delta_actual, delta_pred

    def report_row(label, lr, delta_latent):
        d_act, d_pred = step_and_measure(delta_latent)
        c_act = cos(fem_grad, d_act)
        c_pred = cos(fem_grad, d_pred)
        c_lin = cos(d_pred, d_act)
        print(
            f"  {label:<22s} lr={lr:8.0e}  cos_act={c_act:+.4f}  "
            f"cos_pred={c_pred:+.4f}  cos(pred,act)={c_lin:+.4f}  "
            f"|Δsdf|={d_act.norm().item():.3e}"
        )

    # ------------------------------------------------------------------
    # 4. Direction sweep at multiple step sizes
    # ------------------------------------------------------------------
    print("\n=== Plain SGD: Δlatent = -lr · g ===")
    for lr in [1e-3, 1e-4, 1e-5, 1e-6]:
        report_row("plain_sgd", lr, -lr * g_raw)

    print("\n=== Sign step: Δlatent = -lr · sign(g)  (Adam asymptote) ===")
    for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
        report_row("sign", lr, -lr * g_raw.sign())

    print("\n=== Unit direction: Δlatent = -lr · g/||g||  (||Δl||₂ = lr) ===")
    for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        report_row("unit_dir", lr, -lr * g_raw / (g_raw.norm() + 1e-30))

    print("\n=== L2 rescaled (production): Δlatent = -lr · sqrt(N) · g/||g|| ===")
    for lr in [1e-2, 1e-3, 1e-4, 1e-5]:
        d = -lr * sqrtN * g_raw / (g_raw.norm() + 1e-30)
        report_row("prod_rescaled", lr, d)

    # ------------------------------------------------------------------
    # 5. Krylov-subspace Gauss-Newton: best Δlatent in span{g, J^T J g, ...}
    # ------------------------------------------------------------------
    # Build a 3-vector basis  B = [v0, v1, v2]  in latent space and the
    # matching JVP basis  A = [J·v0, J·v1, J·v2]  in sdf space.  Solve
    #     min_α ||A^T α + fem_grad||₂²    (descent ⇒ J·Δl ≈ -fem_grad)
    # via 3×3 normal equations on (A A^T).  This is equivalent to two
    # CG iterations on the normal equations, restricted to the Krylov
    # subspace generated by g = J^T·fem_grad.
    print("\n=== Krylov Gauss-Newton optimal direction (3-vector basis) ===")

    def vjp(grad_out):
        return pipe.compute_latent_grad(
            problem._latents, fem_pts_fp16, grad_out.to(torch.float32),
            chunk_size=10000,
        ).detach()

    v0 = g_raw
    A0 = jvp_fd(v0)              # J·v0
    v1 = vjp(A0)                 # J^T J v0
    A1 = jvp_fd(v1)              # J·v1
    v2 = vjp(A1)                 # (J^T J)² v0
    A2 = jvp_fd(v2)              # J·v2

    # Stack basis as (3, M) for the sdf side and a flat (3, D) for latent side.
    A = torch.stack([A0, A1, A2], dim=0)              # (3, M)
    B_shape = v0.shape
    B = torch.stack(
        [v0.reshape(-1), v1.reshape(-1), v2.reshape(-1)], dim=0
    )                                                 # (3, D)

    AAT = A @ A.T                                     # (3, 3)
    rhs = -(A @ fem_grad)                             # (3,)
    # Tikhonov-regularize the 3×3 to handle degenerate Krylov directions
    reg = 1e-8 * AAT.diag().mean() * torch.eye(3, device=AAT.device, dtype=AAT.dtype)
    alpha = torch.linalg.solve(AAT + reg, rhs)        # (3,)
    print(f"  Krylov coeffs α = {alpha.tolist()}")

    delta_l_opt_flat = (alpha[:, None] * B).sum(dim=0)        # (D,)
    delta_l_opt = delta_l_opt_flat.reshape(B_shape)
    delta_sdf_pred_lin = (alpha[:, None] * A).sum(dim=0)      # (M,)

    print(
        f"  predicted ||J·Δl + fem_grad||/||fem_grad|| = "
        f"{((delta_sdf_pred_lin + fem_grad).norm() / fem_grad.norm()).item():.4f}  "
        f"(0 = exactly aligned)"
    )

    # Sweep step lengths along the Krylov direction (rescaled to several
    # latent norms to compare against the production rule).
    krylov_norm = float(delta_l_opt.norm())
    print(f"  natural ||Δl_opt||₂ = {krylov_norm:.4e}")
    if krylov_norm > 1e-30:
        unit = delta_l_opt / krylov_norm
        for target_norm in [krylov_norm, sqrtN * 1e-3, sqrtN * 1e-4, sqrtN * 1e-5]:
            scaled = unit * target_norm
            d_act, d_pred = step_and_measure(scaled)
            c_act = cos(fem_grad, d_act)
            c_pred = cos(fem_grad, d_pred)
            print(
                f"  krylov_step  ||Δl||={target_norm:.3e}  "
                f"cos_act={c_act:+.4f}  cos_pred={c_pred:+.4f}  "
                f"|Δsdf|={d_act.norm().item():.3e}"
            )

    # ------------------------------------------------------------------
    # 6. fp16 cast probe — confirm the LatentShapeOpt upcast still matters
    # ------------------------------------------------------------------
    print("\n=== fp16 vae forward (legacy) vs fp32 (production) ===")
    vae_fp16 = copy.deepcopy(vae).half()
    for p in vae_fp16.parameters():
        p.requires_grad_(False)

    def forward_fp16(latents_fp32):
        with torch.no_grad():
            return _decoder_chain(
                latents_fp32.half(), vae_fp16, fem_pts_fp16
            )

    sdf_baseline_fp16 = forward_fp16(latents_save)
    for lr in [1e-3, 1e-4, 1e-5]:
        d = -lr * sqrtN * g_raw / (g_raw.norm() + 1e-30)
        sdf_post_fp16 = forward_fp16(latents_save + d)
        d_act_fp16 = sdf_post_fp16 - sdf_baseline_fp16
        c_fp16 = cos(fem_grad, d_act_fp16)
        sdf_post_fp32 = forward(latents_save + d)
        d_act_fp32 = sdf_post_fp32 - sdf_baseline
        c_fp32 = cos(fem_grad, d_act_fp32)
        # How much of the fp16 step survived the cast?
        new_lat_fp16 = (latents_save + d).half()
        n_changed = int((new_lat_fp16 != latents_save.half()).sum())
        print(
            f"  prod_rescaled lr={lr:.0e}  cos_fp16={c_fp16:+.4f}  "
            f"cos_fp32={c_fp32:+.4f}  fp16-coords-moved={n_changed}/{latents_save.numel()}"
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(
        "\nKey takeaways:\n"
        "  - cos_pred is the linearized upper bound (J·Δl); cos_act adds\n"
        "    decoder nonlinearity. Their gap shrinks at small lr.\n"
        "  - Plain SGD, unit direction, and L2 rescale share the SAME direction\n"
        "    (g/||g||), so they share the same cos at matching ||Δl||.\n"
        "  - Sign(g) injects coordinates orthogonal to J^T·fem_grad and\n"
        "    typically gives WORSE cos than the gradient direction.\n"
        "  - Krylov GN can lift cos_pred above the single-VJP bound by\n"
        "    spending extra JVPs/VJPs to project onto the column space of J.\n"
        "  - fp16 cast collapses small lr steps; the production fp32 path\n"
        "    preserves cos_act ≈ cos_fp32 down to lr ≈ 1e-5.\n"
    )

    problem._latents.data.copy_(latents_save)


if __name__ == "__main__":
    main()
