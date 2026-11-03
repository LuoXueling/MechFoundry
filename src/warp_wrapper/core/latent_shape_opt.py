"""Latent-space shape optimization via a generative SDF decoder.

Optimizes a latent tensor by backpropagating FEM loss gradients through a
decoder pipeline.  Inherits from ``ShapeOptTrimmed`` to reuse the full-mesh /
trimmed-mesh inner solver architecture, VTU output, and all FEM/Nitsche
machinery.

The decoder pipeline (``pipe``) must implement:
    ``pipe.make_geometric_func(latents)``
        → callable ``(1, N, 3) float32 → (1, N, 1) float``
    ``pipe.compute_latent_grad(latents, query_pts, grad_outputs, chunk_size)``
        → latent gradient tensor (VJP, ``J^T @ grad_outputs``)
    ``pipe.compute_latent_jvp(latents, query_pts, direction, chunk_size)``
        → sdf-space tensor (JVP, ``J @ direction``) — used by the pull-back
        conjugate-gradient correction in ``_pullback_cg_correction``
    ``pipe.make_vjp_cache(latents, query_pts, chunk_size)``
        → callable ``w -> J^T @ w`` reusing one cached forward pass for
        repeated VJPs (e.g. across CG iterations)
    ``pipe.vae.parameters()``
        → iterable of decoder parameters (to freeze)
    ``pipe.sdf_truncation_threshold``
        → float, the decoder's normalized-space SDF clamp bound (e.g. 0.01);
        the decoder maps logits → truncated SDF in ``[-thr, +thr]``
    ``pipe.sdf_linear_keep_frac``
        → float in (0, 1]; query nodes whose ``|decoded SDF| > frac * threshold``
        sit in the saturated sigmoid tail (untrustworthy Jacobian) and are
        dropped from the latent VJP in ``_compute_latent_grad``

This keeps model-specific logic (TripoSG, etc.) out of the optimizer.
"""

import math
import warnings

import numpy as np
import torch
import warp as wp

from .shape_opt import ShapeOptTrimmed

try:
    from src.triposg_wrapper.utils import (
        filter_largest_interior_component as _filter_sdf,
    )
except ImportError:
    from triposg_wrapper.utils import filter_largest_interior_component as _filter_sdf

try:
    from src.triposg_wrapper.hierarchical_extraction import (
        hierarchical_extraction as _hierarchical_extraction,
    )
except ImportError:
    from triposg_wrapper.hierarchical_extraction import (
        hierarchical_extraction as _hierarchical_extraction,
    )


@wp.kernel
def _closest_point_dist_kernel(
    mesh: wp.uint64,
    pts: wp.array(dtype=wp.vec3),
    max_dist: float,
    out: wp.array(dtype=float),
):
    """Per-point distance to the closest point on ``mesh``'s triangles."""
    i = wp.tid()
    q = wp.mesh_query_point_no_sign(mesh, pts[i], max_dist)
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        out[i] = wp.length(pts[i] - cp)
    else:
        out[i] = max_dist


def _wp_mesh(verts, faces):
    """Build a warp BVH mesh from torch ``(V,3)`` verts and ``(F,3)`` faces."""
    return wp.Mesh(
        points=wp.from_torch(
            verts.contiguous().to(torch.float32), dtype=wp.vec3
        ),
        indices=wp.from_torch(
            faces.reshape(-1).contiguous().to(torch.int32), dtype=wp.int32
        ),
    )


def _directed_surface_dists(mesh, pts, max_dist):
    """Distance from each of ``pts`` ``(N,3)`` torch to ``mesh``'s surface.

    O(log F) per query via the warp mesh BVH, parallel over the N points.
    """
    wp_pts = wp.from_torch(pts.contiguous().to(torch.float32), dtype=wp.vec3)
    out = wp.zeros(pts.shape[0], dtype=float, device=wp_pts.device)
    wp.launch(
        _closest_point_dist_kernel,
        dim=pts.shape[0],
        inputs=[mesh.id, wp_pts, float(max_dist), out],
        device=wp_pts.device,
    )
    return wp.to_torch(out)


def _max_query_dist(va, vb):
    """BVH search cutoff: the bbox diagonal of the two vertex sets (the closest
    surface point is always within it)."""
    lo = torch.minimum(va.amin(0), vb.amin(0))
    hi = torch.maximum(va.amax(0), vb.amax(0))
    return float(torch.linalg.norm(hi - lo)) + 1e-6


def _surface_dists_both_ways(va, fa, vb, fb):
    """Point-to-surface distances each way: ``va``→mesh(b) and ``vb``→mesh(a).

    Uses warp mesh BVH queries (O(N log F)) instead of an all-pairs cdist, and
    measures true point-to-triangle distance rather than vertex-to-vertex.
    Returns the two per-point distance tensors ``(d_ab, d_ba)``.
    """
    md = _max_query_dist(va, vb)
    mesh_a, mesh_b = _wp_mesh(va, fa), _wp_mesh(vb, fb)
    d_ab = _directed_surface_dists(mesh_b, va, md)
    d_ba = _directed_surface_dists(mesh_a, vb, md)
    return d_ab, d_ba


def _surface_hausdorff(va, fa, vb, fb):
    """Symmetric (two-sided) Hausdorff distance between two triangle surfaces."""
    d_ab, d_ba = _surface_dists_both_ways(va, fa, vb, fb)
    return float(torch.maximum(d_ab.max(), d_ba.max()))


def _surface_chamfer(va, fa, vb, fb):
    """Symmetric Chamfer distance between two triangle surfaces."""
    d_ab, d_ba = _surface_dists_both_ways(va, fa, vb, fb)
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


class LatentShapeOpt(ShapeOptTrimmed):
    """Shape optimization in a generative model's latent space.

    The design variable is a latent tensor rather than per-node SDF values.
    Each iteration:

    1. ``step()`` applies any pending deferred remesh (from previous iter)
    2. Forward FEM solve on trimmed mesh (inherited ``step``)
    3. Warp tape backward → ``dL/d_sdf`` (inherited)
    4. ``pipe.compute_latent_grad`` bridge → ``dL/d_latents``
    5. ``latent_optimizer`` update on latents (SGD / AdamW / RMSProp)
    6. ``_post_optimizer_step_remesh_impl``: ``latent2octree`` (hierarchical
       extraction + island filter) → scatter to bg grid → island filter →
       MC reinit → new octree → deferred remesh

    If a latent update distorts the surface (its Hausdorff distance from the
    previous surface exceeds ``_REVERT_HAUSDORFF_EPS_FACTOR * phase_field_epsilon``)
    or the BC-zip stitch raises, the update is reverted: the previous surface mesh
    is re-encoded into fresh latents (``pipe.mesh2latent``) and optimization
    continues from there.  ``_REVERT_MAX_STREAK`` consecutive reverts abort.
    """

    # Revert a latent update when the new surface's Hausdorff distance from the
    # previous surface exceeds this multiple of the (physical) phase_field_epsilon.
    _REVERT_HAUSDORFF_EPS_FACTOR = 2.0
    # Abort after this many consecutive reverts (stuck oscillating).
    _REVERT_MAX_STREAK = 5
    # During a revert, re-encode the previous surface until the re-encoded
    # surface's Chamfer distance from it drops below this multiple of eps (eps/8).
    _REVERT_CHAMFER_EPS_FACTOR = 1.0 / 8.0
    # Cap on re-encode attempts per revert.  Each attempt uses a fresh
    # deterministic surface sample; after a failing attempt the sample-point
    # budget grows by _REVERT_REENCODE_GROWTH (more points -> less smoothing ->
    # more faithful re-encode).  If exhausted, keep the best (lowest-Chamfer) one.
    _REVERT_MAX_REENCODE = 10
    # Point-budget growth factor applied after each failing re-encode attempt.
    _REVERT_REENCODE_GROWTH = 1.5
    # Base seed for the per-call deterministic query-node subsample RNG.
    _GRAD_SUBSAMPLE_SEED = 20260721

    def __init__(
        self,
        config_module,
        override_params=None,
        *,
        pipe=None,
        latents=None,
        octree_depth=8,
        init_octree_depth=4,
        bounds=(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0),
        physical_extents=(4.0, "x"),
        latent_lr=1e-4,
        latent_weight_decay=0.0,
        latent_optimizer="sgd",
        remesh_every=1,
        grad_threshold=1e-10,
        n_grad_points=1000,
        decoder_chunk_size=10000,
        sdf_penalty_weight=0.0,
        epsilon_to_size_ratio=0.5,
        precond=None,
    ):
        if pipe is None or latents is None:
            raise ValueError("pipe and latents are required")
        if override_params is None:
            override_params = {}

        # Latent optimization has not been qualified for the native f32
        # ShapeOpt path.  Keep its FEM/adjoint path in f64 irrespective of the
        # base ShapeOpt configuration or caller overrides.
        warnings.warn(
            "LatentShapeOpt enforces double precision: any requested "
            "use_mixed_precision=False or tangent_f64=False is ignored.",
            UserWarning,
            stacklevel=2,
        )
        override_params["use_mixed_precision"] = True
        override_params["tangent_f64"] = True

        # Always export surface VTU/GLB (MC path provides the mesh).
        override_params.setdefault("output_surface_vtu", True)
        override_params.setdefault("output_surface_glb", True)
        override_params.setdefault("output_surface_glb_deformed", True)

        self._pipe = pipe
        self._octree_depth = octree_depth
        self._init_octree_depth = init_octree_depth
        self._bounds_tuple = tuple(bounds)
        self._domain_length = float(bounds[3] - bounds[0])
        self._remesh_every = remesh_every
        self._grad_threshold = grad_threshold
        self._n_grad_points = n_grad_points
        self._decoder_chunk_size = decoder_chunk_size
        self._sdf_penalty_weight = float(sdf_penalty_weight)
        # Stash for epsilon auto-compute in _init_remeshing_state override.
        self._epsilon_to_size_ratio = epsilon_to_size_ratio
        self._config_module_latent = config_module
        # Pull-back-metric gradient correction config, a single dict (all keys
        # optional).  The raw single-VJP latent gradient is corrected by solving
        # the full pull-back-metric system (J^T J + lambda I) x = g_raw via
        # matrix-free CG (see _pullback_cg_correction), where J = d(sdf)/d(latent)
        # is the decoder Jacobian.  This inverts the decoder's local anisotropy
        # (whose ~166x tip disparity drives the x~3.6 ring) instead of just
        # renormalizing the gradient norm.  Controls + efficiency toggles:
        #   enabled      : (default True) apply the pull-back CG correction.  When
        #                  False the raw VJP gradient J^T fem_grad is used directly,
        #                  L2-rescaled to sqrt(N) (per-coord RMS step ~lr); the CG
        #                  keys below are then unused.
        #   damping      : Tikhonov ratio x Rayleigh-quotient curvature scale.
        #   cg_iters, cg_rtol : CG max iterations / relative residual tolerance.
        #                  At rtol=1e-1 CG converges in ~10-20 iters (not iter-
        #                  bound at 50), so per-matvec cost -- not iteration count
        #                  -- dominates.
        #   cg_warm_start: (default on) start CG from the previous optimizer iter's
        #                  solution; the latent-space natural-gradient direction is
        #                  smooth in optimizer time, so this ~halves CG iterations
        #                  (benchmark: 20.6 -> 10.6 avg iters).
        #   cg_matvec    : "jvp" (default) is the forward-JVP-then-VJP matvec.
        #                  "reverse" computes J^T J x by double-backward (graph
        #                  built once, no per-matvec primal recompute) -- ~40%
        #                  cheaper per CG iter (0.30 vs 0.50 s) and the fastest
        #                  config overall (with warm-start, ~1.5x vs jvp).  It is
        #                  *correct but VRAM-hungry*: 2nd-order grad through
        #                  attention needs the MATH SDPA kernel, whose
        #                  double-backward graph for the 2048-token self-attention
        #                  is ~40GB -- fits a 48GB card at n_grad_points<=1000 but
        #                  OOMs on smaller GPUs, hence opt-in.
        # Fall back to the centralized defaults in default_config.py when a
        # precond key is not supplied by the caller (the precond dict overrides).
        from .default_config import params as _default_params

        precond = precond or {}
        self._precond_enabled = bool(
            precond.get("enabled", _default_params["latent_precond_enabled"])
        )
        self._precond_damping = float(
            precond.get("damping", _default_params["latent_precond_damping"])
        )
        self._precond_cg_iters = int(
            precond.get("cg_iters", _default_params["latent_precond_cg_iters"])
        )
        self._precond_cg_rtol = float(
            precond.get("cg_rtol", _default_params["latent_precond_cg_rtol"])
        )
        self._precond_cg_warm_start = bool(
            precond.get(
                "cg_warm_start", _default_params["latent_precond_cg_warm_start"]
            )
        )
        self._precond_cg_matvec = precond.get(
            "cg_matvec", _default_params["latent_precond_cg_matvec"]
        )
        if self._precond_cg_matvec not in ("jvp", "reverse"):
            raise ValueError(
                "precond cg_matvec must be 'jvp' or 'reverse', got "
                f"{self._precond_cg_matvec!r}"
            )

        device = override_params.get("device", "cuda")

        # Upcast VAE to fp32; at fp16, latent updates quantize to zero.
        # mean_delta per latent element is ~1e-6, fp16 epsilon is ~1e-3 — all
        # sub-fp16 changes would round to zero, making delta_sdf ≈ 0 and
        # killing the gradient effectiveness signal.
        pipe.vae = pipe.vae.float()
        pipe.vae.eval()
        # Freeze decoder — only latents will have grad.
        for p in pipe.vae.parameters():
            p.requires_grad_(False)

        # Float32 for SGD numerical stability; decoder auto-casts to its own dtype.
        self._latents = (
            latents.clone()
            .detach()
            .to(device=device, dtype=torch.float32)
            .requires_grad_(True)
        )

        # When an implicit geo_func is provided (Path B), remeshing=True is
        # always required: there is no pre-built mesh, so the initial octree
        # must be constructed from the geo_func via hierarchical extraction.
        # _init_remeshing_state auto-computes axis_scale from the MC surface
        # extracted from that grid (physical_extents sub-branch).
        override_params.setdefault("remeshing", True)
        override_params["remeshing_geo_func"] = {
            "geo_func": self._pipe.make_geometric_func(self._latents),
            "sdf_truncation": self._pipe.sdf_truncation_threshold,
            "physical_extents": physical_extents,
            # Remove disconnected interior blobs from the raw decoder SDF grid
            # before MC surface extraction (mirrors the old latent2octree path).
            "filter_sdf_func": lambda sdf: _filter_sdf(
                sdf, domain_length=2.0, log=self.log
            ),
        }
        override_params.setdefault("remeshing_method", "decoder_marchingcubes")
        override_params.setdefault("reinit_sdf_method", "marchingcubes")
        override_params.setdefault("sdf_bc_fix_method", "zip")
        override_params["remesh_octree_depth"] = octree_depth
        override_params["remesh_init_octree_depth"] = init_octree_depth

        super().__init__(config_module, override_params=override_params)

        # Alias infrastructure SDF scale used by _compute_latent_grad / _post_optimizer_step.
        self._sdf_scale = self._remesh_sdf_scale

        # Initial outer node positions for decoder queries and gradient computation.
        pts = self.vtu_writer.mesh_info["points"]
        self._outer_node_positions = (
            pts.to(dtype=torch.float64, device=self.device)
            if isinstance(pts, torch.Tensor)
            else torch.tensor(np.asarray(pts), dtype=torch.float64, device=self.device)
        )

        self._latent_optimizer_type = latent_optimizer
        # ``rmsprop`` ignores ``latent_weight_decay``: decaying latents pulls them
        # toward the origin of the decoder's prior, which is a shape prior rather
        # than a regularizer on free parameters.
        if latent_optimizer == "sgd":
            self._latent_optimizer = torch.optim.SGD(
                [self._latents],
                lr=latent_lr,
                momentum=0.0,
                weight_decay=latent_weight_decay,
            )
        elif latent_optimizer == "adam":
            self._latent_optimizer = torch.optim.AdamW(
                [self._latents],
                lr=latent_lr,
                weight_decay=latent_weight_decay,
            )
        elif latent_optimizer == "rmsprop":
            self._latent_optimizer = torch.optim.RMSprop(
                [self._latents],
                lr=latent_lr,
                weight_decay=0.0,
            )
        else:
            raise ValueError(
                f"unknown latent_optimizer {latent_optimizer!r}; expected one of "
                "'sgd', 'adam', 'rmsprop'"
            )

        # Count of consecutive reverted (distorted / stitch-failed) updates.
        self._revert_streak = 0
        # Monotonic index for the deterministic query-node subsample RNG.
        self._grad_call_idx = 0

    # ------------------------------------------------------------------
    # Override _init_remeshing_state to compute epsilon after axis_scale known
    # ------------------------------------------------------------------

    def _init_remeshing_state(self, override_params):
        """Extend Path B to auto-compute phase_field_epsilon after axis_scale."""
        super()._init_remeshing_state(override_params)
        # At this point self._remesh_axis_scale is available.
        if override_params.get("phase_field_epsilon") is None:
            import importlib
            from .default_config import params as default_params

            module_path = self._config_module_latent.replace("/", ".").removesuffix(
                ".py"
            )
            cfg = default_params.copy()
            cfg.update(importlib.import_module(module_path).params)
            cfg.update(override_params)
            if cfg.get("phase_field_epsilon") is None:
                h_min = (2.0 / (2**self._remesh_octree_depth)) * min(
                    self._remesh_axis_scale
                )
                if cfg.get("quad_refinement_depth") is not None:
                    h_min /= 2 ** cfg["quad_refinement_depth"]
                eps = self._epsilon_to_size_ratio * h_min
                override_params["phase_field_epsilon"] = eps
                print(f"  Auto phase_field_epsilon = {eps:.6e} (h_min={h_min:.6e})")

    # ------------------------------------------------------------------
    # Skip beta on the outer solver (never runs FEM integrands)
    # ------------------------------------------------------------------

    def _compute_stabilization_beta(self, **kwargs):
        pass

    # ------------------------------------------------------------------
    # Latent gradient computation (warp → torch bridge)
    # ------------------------------------------------------------------

    def _get_boundary_element_nodes(self):
        """Return unique outer node indices of boundary elements."""
        inner = self._inner
        boundary_elems = inner._prev_octree_elements[
            torch.where(inner._boundary_elem_mask)[0]
        ]
        inner_nodes = torch.unique(boundary_elems.reshape(-1))
        return self._trim_active_indices.long()[inner_nodes]

    def _pullback_cg_correction(self, g_raw, vjp_fn, query_pts):
        """Solve ``(J^T J + lambda*I) x = g_raw`` via matrix-free CG.

        ``J^T J`` is the pull-back metric the Euclidean metric on sdf-space
        induces on latent space through the decoder map -- this corrects the
        raw single-VJP gradient ``g_raw`` for the decoder's local anisotropy
        instead of just renormalizing its norm.

        Reuses ``warp``'s matrix-free CG (same convergence criterion already
        proven for the FEM Newton solves in ``utils.py``) -- the matvec body
        crosses into torch via the zero-copy ``wp.to_torch``/``wp.from_torch``
        bridge since the actual operator is the TripoSG decoder's ``J^T J``,
        not a warp kernel.

        Damping ``lambda`` is ``precond_damping`` times a Rayleigh quotient
        ``g_raw^T (J^T J) g_raw / g_raw^T g_raw`` (one extra matvec), so it adapts
        to the operator scale instead of a fixed absolute constant.

        Efficiency toggles (resolved in __init__):
          - ``cg_matvec`` (default ``"jvp"``): ``"jvp"`` is the forward-JVP-then-VJP
            matvec (``compute_latent_jvp`` + ``vjp_fn``).  ``"reverse"`` builds
            ``x -> J^T J x`` by double-backward once (``pipe.make_gn_matvec``, no
            per-matvec primal recompute) -- ~40% cheaper per CG iter, but its
            MATH-kernel 2nd-order graph is ~40GB (fits a 48GB card at
            ``n_grad_points<=1000``, OOMs smaller GPUs) so it stays opt-in.
          - ``cg_warm_start`` (default on): initialise CG from the previous iter's
            solution (``self._pullback_x_prev``); the natural-gradient direction is
            smooth in optimizer time.  Cheap, mildly helpful, never hurts.

        Benchmark (cantilever, 20 iters, RTX A6000; sgd lr 2e-3, vol-penalty 100;
        rtol 1e-1, damping 1e-3).  ``per-CG-it`` is the reliable metric --
        ``wall``/iter-count are confounded by the stochastic optimization
        trajectories (random query subsample + truncated CG):

            pullback variant            per-CG-it   wall(20it)   mean cos
            baseline (jvp, no precond)     0.50        332 s       -0.61
            jvp + warm-start (default)     0.50        319 s       -0.77
            reverse (matvec only)          0.30        263 s       -0.66
            reverse + warm-start           0.34        211 s       -0.80

        Reverse-mode is the robust win (~40% cheaper per CG iter, ~1.3x wall),
        opt-in on VRAM; warm-start is the safe default.

        Args:
            g_raw: latent-space tensor, same shape as ``self._latents``.
            vjp_fn: callable ``w -> J^T @ w`` from ``pipe.make_vjp_cache``
                (used by the ``"jvp"`` matvec and the Rayleigh damping).
            query_pts: (1, M, 3) normalized-space query points (``fem_pts``).
        Returns:
            x: latent-space tensor, same shape as ``g_raw``.
        """
        from warp.optim.linear import cg as wp_cg, LinearOperator

        shape = g_raw.shape
        n = g_raw.numel()
        chunk = self._decoder_chunk_size

        # Reverse-mode J^T J matvec (built once) or the forward-JVP + VJP path.
        if self._precond_cg_matvec == "reverse":
            gn = self._pipe.make_gn_matvec(self._latents, query_pts, chunk_size=chunk)

            def apply_JTJ(x_t):
                return gn(x_t)
        else:

            def apply_JTJ(x_t):
                Jx = self._pipe.compute_latent_jvp(
                    self._latents, query_pts, x_t, chunk_size=chunk
                )
                return vjp_fn(Jx)

        # Damping from a Rayleigh quotient (one extra matvec on g_raw), so it
        # adapts to the operator scale instead of a fixed absolute constant.
        Gg = apply_JTJ(g_raw)
        rayleigh = (g_raw * Gg).sum() / (g_raw * g_raw).sum().clamp_min(1e-30)
        damping = float(self._precond_damping * rayleigh.clamp_min(1e-30))

        def matvec(x, y, z, alpha, beta):
            x_t = wp.to_torch(x).reshape(shape)
            Gx = apply_JTJ(x_t) + damping * x_t
            y_t = wp.to_torch(y).reshape(shape)
            z_t = wp.to_torch(z).reshape(shape)
            z_t.copy_(alpha * Gx + beta * y_t)

        b_flat = wp.from_torch(g_raw.reshape(-1).contiguous(), dtype=wp.float32)
        # Warm-start from the previous iteration's solution when shapes match.
        x_prev = getattr(self, "_pullback_x_prev", None)
        if self._precond_cg_warm_start and x_prev is not None and x_prev.shape == shape:
            x_flat = wp.from_torch(
                x_prev.reshape(-1).contiguous().to(torch.float32), dtype=wp.float32
            )
        else:
            x_flat = wp.zeros(n, dtype=wp.float32, device=self.device)
        A_op = LinearOperator(
            shape=(n, n),
            dtype=wp.float32,
            device=wp.get_device(self.device),
            matvec=matvec,
        )
        atol = self._precond_cg_rtol * float(g_raw.norm())
        niter, res_norm, _ = wp_cg(
            A=A_op,
            b=b_flat,
            x=x_flat,
            atol=atol,
            maxiter=self._precond_cg_iters,
            check_every=1,
            use_cuda_graph=False,
        )
        x = wp.to_torch(x_flat).reshape(shape).clone()
        if self._precond_cg_warm_start:
            self._pullback_x_prev = x.detach()
        self.log(
            f"  Pull-back CG: iters={niter} res={res_norm:.3e} atol={atol:.3e} "
            f"damping={damping:.4e} matvec={self._precond_cg_matvec} "
            f"warm={self._precond_cg_warm_start}"
        )
        return x

    def _compute_latent_grad(self, sdf_grad):
        """Compute dL/d_latents from dL/d_sdf at boundary nodes via VJP.

        Selects non-frozen boundary element nodes with nonzero FEM gradient,
        optionally subsamples to n_grad_points, then computes the VJP through
        the decoder to get the raw latent gradient ``g_raw = J^T fem_grad``.  When
        ``precond["enabled"]`` (default) this is corrected for the decoder's local
        anisotropy by solving the pull-back metric ``(J^T J + lambda I) x = g_raw``
        via matrix-free CG (see ``_pullback_cg_correction``); otherwise the raw VJP
        gradient is used directly and L2-rescaled to sqrt(N).
        """
        # Query only non-frozen boundary element nodes with nonzero gradient
        boundary_nodes = self._get_boundary_element_nodes()
        query_mask = torch.zeros(len(sdf_grad), dtype=torch.bool, device=self.device)
        query_mask[boundary_nodes] = True
        query_mask &= ~self._e_fixed_bool
        query_mask &= sdf_grad.abs() >= self._grad_threshold
        query_indices = torch.where(query_mask)[0]

        if len(query_indices) == 0:
            raise RuntimeError(
                "No query nodes for latent gradient — all boundary nodes are "
                "either frozen or have near-zero SDF gradient"
            )

        # Restrict the query set to the sigmoid's linear band,
        # |decoded sdf| <= keep_frac * truncation.  A boundary node where the
        # decoder reads a saturated SDF (|sdf| ~ truncation) sits on the decoder's
        # plateau, far from its sdf=0 surface; there sigmoid'(logit) -> 0, so the
        # node's decoder Jacobian both vanishes in magnitude and points in a
        # noise-dominated direction.  Such a near-dead, noise-directioned row would
        # pollute the pull-back metric J^T J that the CG correction solves against.
        # Filtering them here -- before subsampling, so the n_grad_points budget is
        # spent only on trustworthy near-surface nodes -- is the principled fix.
        # Decoder evaluated at the current latents, no_grad.
        q_norm_pos = self._physical_to_normalized(
            self._outer_node_positions[query_indices]
        ).to(torch.float32)
        geo_func = self._pipe.make_geometric_func(self._latents)
        with torch.no_grad():
            decoded_q = geo_func(q_norm_pos.unsqueeze(0)).squeeze(0).squeeze(-1).float()
        keep_bound = (
            self._pipe.sdf_linear_keep_frac * self._pipe.sdf_truncation_threshold
        )
        keep_bool = decoded_q.abs() <= keep_bound
        query_indices = query_indices[keep_bool]

        if len(query_indices) == 0:
            raise RuntimeError(
                "No query nodes for latent gradient — all candidate boundary "
                "nodes fall in the saturated sigmoid tail (|decoded SDF| > "
                f"{self._pipe.sdf_linear_keep_frac} * "
                f"{self._pipe.sdf_truncation_threshold})"
            )

        # Report how many nodes satisfy the linear-band criterion and their mean
        # |FEM gradient|, before subsampling / the pull-back CG solve.
        kept_abs_grad = sdf_grad[query_indices].abs()
        self.log(
            f"Latent-grad query nodes: {len(query_indices)} satisfy "
            f"|decoded SDF| <= {keep_bound:.3g} "
            f"(mean |grad|={kept_abs_grad.mean().item():.4e})"
        )

        # Subsample to a fixed number of points for efficiency.  Seed the RNG
        # deterministically per gradient call (fixed base + call index) so the
        # whole optimization -- every step, revert trigger, and recovery -- is
        # reproducible run-to-run.
        if self._n_grad_points > 0 and len(query_indices) > self._n_grad_points:
            gen = torch.Generator(device=self.device).manual_seed(
                self._GRAD_SUBSAMPLE_SEED + self._grad_call_idx
            )
            perm = torch.randperm(
                len(query_indices), device=self.device, generator=gen
            )
            query_indices = query_indices[perm[: self._n_grad_points]]
        self._grad_call_idx += 1

        normalized_pos = self._physical_to_normalized(
            self._outer_node_positions[query_indices]
        )
        fem_pts = normalized_pos.to(dtype=torch.float32).unsqueeze(0)
        fem_grad = (sdf_grad[query_indices] * self._sdf_scale).to(torch.float32)

        vjp_fn = self._pipe.make_vjp_cache(
            self._latents, fem_pts, chunk_size=self._decoder_chunk_size
        )
        latent_grad = vjp_fn(fem_grad)  # g_raw = J^T fem_grad

        # Pull-back-metric correction: (J^T J + lambda I)^-1 g_raw via matrix-free
        # CG (reusing vjp_fn's cached forward for every matvec).  The solve already
        # returns a curvature-consistent magnitude, so no terminal L2 rescale.
        # When disabled, the raw VJP gradient is used and rescaled below.
        if self._precond_enabled:
            latent_grad = self._pullback_cg_correction(latent_grad, vjp_fn, fem_pts)

        # Stash for grad-effectiveness diagnostic in _post_optimizer_step.
        self._last_fem_grad_reinit = fem_grad
        self._last_query_indices = query_indices

        # BC-freeze penalty.
        fem_grad_mean = fem_grad.abs().mean()
        penalty_pts, penalty_grad = self._bc_freeze_penalty(fem_grad_mean)
        if penalty_pts is not None:
            penalty_latent_grad = self._pipe.compute_latent_grad(
                self._latents,
                penalty_pts,
                penalty_grad,
                chunk_size=self._decoder_chunk_size,
            )
            latent_grad = latent_grad + penalty_latent_grad

        torch.cuda.empty_cache()

        # With the pull-back correction disabled, the raw VJP gradient has no
        # curvature-consistent scale, so L2-rescale to sqrt(N): ``-lr * latent_grad``
        # then has L2 norm ``sqrt(N) * lr`` and (for an approximately uniform grad)
        # per-coord RMS step ``lr``.  The CG solve already provides this scale, so
        # it is skipped when the correction is enabled.
        if not self._precond_enabled:
            gnorm = latent_grad.norm()
            if gnorm > 0:
                latent_grad = latent_grad * (float(latent_grad.numel()) ** 0.5 / gnorm)

        return latent_grad

    def _bc_freeze_penalty(self, fem_grad_mean):
        """Compute penalty grad_outputs that push decoder SDF at frozen BC
        positions back toward their initial values.

        Returns (query_pts, grad_outputs) or (None, None) if no penalty.
        """
        if self._sdf_penalty_weight <= 0:
            return None, None

        # Lazily capture initial decoder SDF at frozen BC positions
        if not hasattr(self, "_frozen_bc_pts_norm"):
            reinit_fixed = getattr(self, "_reinit_fixed_bool", None)
            if reinit_fixed is None or not reinit_fixed.any():
                return None, None
            frozen_idx = torch.where(reinit_fixed)[0]
            pos_norm = self._physical_to_normalized(
                self._outer_node_positions[frozen_idx]
            )
            self._frozen_bc_pts_norm = pos_norm
            geo_func = self._pipe.make_geometric_func(self._latents)
            with torch.no_grad():
                self._frozen_bc_target = (
                    geo_func(pos_norm.to(torch.float32).unsqueeze(0))
                    .squeeze(0)
                    .squeeze(-1)
                    .float()
                )
            self.log(
                f"  BC penalty: captured {len(frozen_idx)} frozen positions, "
                f"target SDF range [{self._frozen_bc_target.min():.4e}, "
                f"{self._frozen_bc_target.max():.4e}]"
            )

        pts_norm = self._frozen_bc_pts_norm
        geo_func = self._pipe.make_geometric_func(self._latents)
        with torch.no_grad():
            current_sdf = (
                geo_func(pts_norm.to(torch.float32).unsqueeze(0))
                .squeeze(0)
                .squeeze(-1)
                .float()
            )

        diff = current_sdf - self._frozen_bc_target
        penalty_grad = (self._sdf_penalty_weight * diff).to(torch.float32)

        query_pts = pts_norm.to(dtype=torch.float32).unsqueeze(0)
        return query_pts, penalty_grad

    # ------------------------------------------------------------------
    # Per-iteration hooks
    # ------------------------------------------------------------------

    def _eval_decoder_at_outer_nodes(self):
        """Evaluate decoder at all outer octree nodes in chunks.

        Returns (N,) float32 tensor of normalized-space SDF values.
        """
        nodes_norm = self._physical_to_normalized(self._outer_node_positions)
        N = nodes_norm.shape[0]
        geo_func = self._pipe.make_geometric_func(self._latents)
        parts = []
        with torch.no_grad():
            for i in range(0, N, self._decoder_chunk_size):
                pts = (
                    nodes_norm[i : i + self._decoder_chunk_size]
                    .to(torch.float32)
                    .unsqueeze(0)
                )
                parts.append(geo_func(pts).squeeze(0).squeeze(-1).float())
        return torch.cat(parts)  # (N,) normalized SDF

    def _post_optimizer_step(self, iteration):
        """Update e.dof_values from decoded latents, then remesh.

        Sets ``e.dof_values = φ_decoder(latents, x) · sdf_scale`` directly.
        The parent then runs: scatter → island filter → BC freeze → MC reinit →
        rebuild octree.  The reinitialized SDF is what the inner simulation sees.
        """
        with torch.no_grad():
            sdf_new_norm = self._eval_decoder_at_outer_nodes()

            sdf_phys = (sdf_new_norm * self._sdf_scale).to(torch.float64).contiguous()
            wp.copy(
                src=wp.from_torch(sdf_phys, dtype=wp.float64), dest=self.e.dof_values
            )

            # Gradient effectiveness: does the latent step move the SDF in the
            # descent direction?
            qi = getattr(self, "_last_query_indices", None)
            fem_grad = getattr(self, "_last_fem_grad_reinit", None)
            outer_sdf_before = getattr(self, "_outer_sdf_before", None)
            if (
                qi is not None
                and len(qi) > 0
                and fem_grad is not None
                and outer_sdf_before is not None
            ):
                delta_sdf_raw_qi = sdf_new_norm[qi] - outer_sdf_before[qi]
                fg_norm = fem_grad.norm()
                ds_norm = delta_sdf_raw_qi.norm()
                cos_align = (
                    (fem_grad * delta_sdf_raw_qi).sum() / (fg_norm * ds_norm + 1e-30)
                ).item()
                n_query = len(qi)
            else:
                cos_align = float("nan")
                n_query = 0
                delta_sdf_raw_qi = None
            self.log(
                f"  Grad effectiveness: cos(fem_grad, Δsdf)={cos_align:.4f} "
                f"(want -1, n_query={n_query}) "
                + (
                    f"|Δsdf[qi]|.mean={delta_sdf_raw_qi.abs().mean().item():.4e} "
                    f"|fem_grad|.mean={fem_grad.abs().mean().item():.4e}"
                    if delta_sdf_raw_qi is not None and fem_grad is not None
                    else ""
                )
            )

        super()._post_optimizer_step(iteration)

    def _post_scatter_bg_grid_hook(self, grid):
        """Filter disconnected interior islands after scatter, before BC freeze."""
        return _filter_sdf(grid, domain_length=2.0, log=self.log)

    def _get_decoder_grid_sdf(self):
        """Re-run hierarchical extraction aligned to the current latent.

        Returns a ``(2^D+1, 2^D+1, 2^D+1)`` float32 physical-SDF grid.
        Adaptive octree refinement focuses decoder queries near the **current**
        zero level set, so the entire background grid is accurate for the new
        surface position — unlike scatter from old octree nodes which leaves
        ~96 % of grid cells stale.

        Called by ``ShapeOptTrimmed._post_optimizer_step_remesh_impl`` when
        ``remeshing_method="decoder_marchingcubes"`` (the default for
        ``LatentShapeOpt``).
        """
        geo_func = self._pipe.make_geometric_func(self._latents)
        fresh_raw, _ = _hierarchical_extraction(
            geometric_func=geo_func,
            bounds=self._bounds_tuple,  # normalized (-1,-1,-1,1,1,1)
            init_octree_depth=self._init_octree_depth,
            octree_depth=self._octree_depth,
            device=str(self.device),
            sdf_truncation=self._pipe.sdf_truncation_threshold,
        )
        return fresh_raw.float() * self._sdf_scale  # (n,n,n) float32 physical SDF

    def reinit_with_sdf(self, octree_mesh):
        """Override to keep _outer_node_positions in sync after remesh."""
        super().reinit_with_sdf(octree_mesh)
        self._outer_node_positions = octree_mesh["nodes"].to(
            dtype=torch.float64,
            device=self.device,
        )

    def _revert_to_previous_mesh(self, it, prev_vn, prev_f, reason):
        """Re-encode the previous surface mesh into fresh latents and rebuild.

        Invoked when a latent update distorts the surface (Hausdorff exceeds
        ``_REVERT_HAUSDORFF_EPS_FACTOR * phase_field_epsilon``) or the BC-zip
        stitch raised.  ``self._latents`` is overwritten **in place** (it is the
        optimizer's leaf param — reassigning would orphan the optimizer), the
        optimizer state and pullback CG warm-start are reset (the reverted shape
        starts fresh so stale momentum can't immediately re-drive the bad step),
        and the remesh state (``e.dof_values`` / MC surface / pending octree) is
        rebuilt from the reverted latents.

        Each re-encode samples the surface at a distinct but fixed seed (the
        attempt index), so attempts differ from one another yet the whole loop is
        deterministic run-to-run.  It starts at half the previous mesh's vertex
        count and, after each failing attempt, grows the point budget by
        ``_REVERT_REENCODE_GROWTH`` (less smoothing -> more faithful), retrying
        until the re-encoded surface's Chamfer distance from the previous surface
        drops below ``_REVERT_CHAMFER_EPS_FACTOR * eps`` (eps/8), up to
        ``_REVERT_MAX_REENCODE`` attempts, keeping the best.  Aborts after
        ``_REVERT_MAX_STREAK`` consecutive reverts.
        """
        import trimesh

        if prev_vn is None:
            # No previous surface to fall back on (only possible if the very
            # first remesh fails; the bootstrap should have populated it).
            self.save(time=it)
            self.finalize()
            raise RuntimeError(
                f"Cannot revert at iter {it} ({reason}): no previous surface mesh"
            )

        # Re-encode the previous surface in the decoder's NORMALIZED frame (the
        # same frame main encodes in) — NOT the physical mesh used for Hausdorff.
        # Start by downsampling to half the mesh's vertex count: area-weighted
        # resampling below the mesh's native resolution smooths out the ill
        # details that triggered the revert.  After each failing attempt the point
        # budget grows by _REVERT_REENCODE_GROWTH (less smoothing -> more faithful),
        # so the schedule walks from smooth toward faithful until Chamfer < eps/8.
        prev_mesh = trimesh.Trimesh(
            vertices=prev_vn.detach().cpu().numpy(),
            faces=prev_f.detach().cpu().numpy(),
            process=False,
        )
        n_pts = max(int(prev_vn.shape[0]) // 2, 1)
        prev_phys = self._normalized_to_physical(prev_vn.double()).float()
        eps = float(self.params.get("phase_field_epsilon", 0.02))
        ch_target = self._REVERT_CHAMFER_EPS_FACTOR * eps  # eps/8

        # Retry until the re-encoded surface's Chamfer distance from the previous
        # surface drops below ``ch_target`` (eps/8), keeping the best.  Each
        # attempt uses a *distinct but fixed* surface-sampling seed (the attempt
        # index), so attempts differ while the whole loop is deterministic
        # run-to-run.
        best_lat = None
        best_chamfer = float("inf")
        success = False

        def _reencode_once(sample_seed, n_points):
            """Sample+encode the previous mesh, load latents, rebuild remesh state.

            ``sample_seed`` fixes the surface sampling (determinism); ``n_points``
            is the surface-sample budget.  Returns ``(curr_phys, curr_faces,
            None)`` on success, or ``(None, None, exc)`` if the rebuild (BC-zip
            stitch etc.) raised for this sample.
            """
            new_lat = self._pipe.mesh2latent(
                prev_mesh, n_points=n_points, sample_seed=sample_seed
            )
            with torch.no_grad():
                self._latents.copy_(
                    new_lat.reshape(self._latents.shape).to(self._latents)
                )
            # Reset optimizer momentum/moments and the pullback CG warm-start.
            self._latent_optimizer.state.clear()
            self._pullback_x_prev = None
            self._outer_sdf_before = self._eval_decoder_at_outer_nodes()
            try:
                self._post_optimizer_step(it)
            except Exception as exc:
                return None, None, exc
            curr_phys = self._normalized_to_physical(
                self._mc_verts_norm.double()
            ).float()
            return curr_phys, self._mc_faces, None

        for attempt in range(1, self._REVERT_MAX_REENCODE + 1):
            curr_phys, curr_faces, exc = _reencode_once(
                sample_seed=attempt, n_points=n_pts
            )
            # Log every attempt (including failed rebuilds) with its index.
            if curr_phys is None:
                self.log(
                    f"  Re-encode attempt {attempt}/{self._REVERT_MAX_REENCODE} "
                    f"(n_pts={n_pts}): rebuild failed: {exc!r}"
                )
                n_pts = int(n_pts * self._REVERT_REENCODE_GROWTH)  # grow on failure
                continue
            chamfer = _surface_chamfer(prev_phys, prev_f, curr_phys, curr_faces)
            self.log(
                f"  Re-encode attempt {attempt}/{self._REVERT_MAX_REENCODE} "
                f"(n_pts={n_pts}): chamfer={chamfer:.6e} "
                f"(target < {ch_target:.6e}, eps={eps:.6e})"
            )
            if chamfer < best_chamfer:
                best_chamfer = chamfer
                best_lat = self._latents.detach().clone()
            if chamfer < ch_target:
                success = True
                break
            n_pts = int(n_pts * self._REVERT_REENCODE_GROWTH)  # grow on miss

        if best_lat is None:
            # No re-encode could even be rebuilt — the last-good shape is
            # unrecoverable; abort with saved state.
            self.log(f"  Revert failed at iter {it}: no re-encode could be rebuilt")
            self.save(time=it)
            self.finalize()
            raise RuntimeError(
                f"Revert at iter {it} ({reason}) failed: no re-encode rebuilt in "
                f"{self._REVERT_MAX_REENCODE} attempts"
            )

        if not success:
            # Cap reached without hitting the target: restore the best (lowest
            # Chamfer) encode and rebuild once so all state matches it.
            self.log(
                f"  Re-encode did not reach target in {self._REVERT_MAX_REENCODE} "
                f"attempts; keeping best chamfer={best_chamfer:.6e}"
            )
            with torch.no_grad():
                self._latents.copy_(best_lat.reshape(self._latents.shape).to(self._latents))
            self._latent_optimizer.state.clear()
            self._pullback_x_prev = None
            self._outer_sdf_before = self._eval_decoder_at_outer_nodes()
            try:
                self._post_optimizer_step(it)
            except Exception as exc:
                self.log(f"  Revert rebuild (best) failed at iter {it}: {exc!r}")
                self.save(time=it)
                self.finalize()
                raise

        self._revert_streak += 1
        self.log(
            f"  Reverted latent update ({reason}); re-encoded previous surface "
            f"(consecutive reverts: {self._revert_streak}/{self._REVERT_MAX_STREAK})"
        )
        if self._revert_streak >= self._REVERT_MAX_STREAK:
            self.log(
                f"  Aborting: {self._revert_streak} consecutive reverts at iter {it}"
            )
            self.save(time=it)
            self.finalize()
            raise RuntimeError(
                f"{self._revert_streak} consecutive distorted/failed latent "
                f"updates at iter {it}; aborting"
            )

    # ------------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------------

    def optimize(self):
        """Latent-space optimization loop."""
        max_iters = int(self.params.get("max_optimize_iters", 1000))
        save_freq = int(self.params.get("output_per_optimize_step", 1))
        rel_tol = float(self.params.get("optimize_relative_tol", 1e-4))
        obj_history = []
        best_obj = float("inf")
        best_obj_iter = 0

        it = 0
        while it < max_iters:
            self.log(f"===== Latent Optimization Iteration {it} =====")

            tape = wp.Tape()
            self.e.dof_values.requires_grad = True
            try:
                self.step(tape=tape)
            except Exception as e:
                self.save(time=it)
                self.finalize()
                import time as _time

                _time.sleep(1)
                raise e

            with tape:
                obj = self._objective_function(tape)
            tape.zero()
            self.e.dof_values.grad.zero_()
            if self._inner.e.dof_values.grad is not None:
                self._inner.e.dof_values.grad.zero_()
            tape.backward(loss=obj)
            self._post_tape_backward()
            obj_val = float(obj.numpy()[0])

            latent_grad = self._compute_latent_grad(self._last_e_grad)
            grad_norm = latent_grad.norm().item()
            n_nonzero = int((latent_grad.abs() > 0).sum())
            assert not math.isnan(
                grad_norm
            ), f"NaN in latent gradient at iteration {it}"
            assert n_nonzero > 0, (
                f"All-zero latent gradient at iteration {it} — "
                f"fp16 backward overflow (all NaN sanitized to 0)"
            )
            self.log(
                f"  Latent grad: norm={grad_norm:.6e}, "
                f"max={latent_grad.abs().max().item():.6e}, "
                f"nonzero={n_nonzero}/{latent_grad.numel()}"
            )

            # Stash decoder SDF BEFORE the optimizer step for the
            # gradient effectiveness diagnostic in _post_optimizer_step.
            self._outer_sdf_before = self._eval_decoder_at_outer_nodes()

            latents_before = self._latents.detach().clone()
            self._latent_optimizer.zero_grad()
            self._latents.grad = (
                latent_grad  # pass rescaled grad (norm=sqrt(N)) directly
            )
            self._latent_optimizer.step()

            delta = (self._latents.detach() - latents_before).abs()
            self.log(
                f"  Latent update: max_delta={delta.max().item():.6e}, "
                f"mean_delta={delta.mean().item():.6e}"
            )

            # Stash the previous (last-good) surface mesh before the remesh
            # overwrites it, so a distorted update or a BC-stitch failure can be
            # reverted by re-encoding it.
            prev_vn = getattr(self, "_mc_verts_norm", None)
            prev_f = getattr(self, "_mc_faces", None)
            if prev_vn is not None:
                prev_vn = prev_vn.clone()
                prev_f = prev_f.clone()

            # Set e.dof_values from decoder, then scatter → MC reinit → rebuild
            # octree (this is where the BC-zip stitch runs and may raise).
            revert = False
            reason = None
            try:
                self._post_optimizer_step(it)
            except Exception as exc:
                self.log(f"  Remesh/BC-zip stitch failed: {exc!r}")
                revert = True
                reason = "stitch-fail"

            # Distortion check: Hausdorff(prev, new) vs
            # _REVERT_HAUSDORFF_EPS_FACTOR * phase_field_epsilon (physical),
            # via warp BVH point-to-surface queries.
            if not revert and prev_vn is not None:
                eps = float(self.params.get("phase_field_epsilon", 0.02))
                prev_phys = self._normalized_to_physical(prev_vn.double()).float()
                curr_phys = self._normalized_to_physical(
                    self._mc_verts_norm.double()
                ).float()
                hausdorff = _surface_hausdorff(
                    prev_phys, prev_f, curr_phys, self._mc_faces
                )
                thresh = self._REVERT_HAUSDORFF_EPS_FACTOR * eps
                self.log(
                    f"  Surface Hausdorff (prev→new)={hausdorff:.6e} "
                    f"(threshold {self._REVERT_HAUSDORFF_EPS_FACTOR:g}*eps={thresh:.6e}, "
                    f"eps={eps:.6e})"
                )
                if hausdorff > thresh:
                    revert = True
                    reason = "distortion"

            if revert:
                self._revert_to_previous_mesh(it, prev_vn, prev_f, reason)
            else:
                self._revert_streak = 0

            if save_freq > 0 and it % save_freq == 0:
                self.log("Start saving outputs")
                self.save(time=it)

            obj_history.append(obj_val)
            if obj_val < best_obj:
                best_obj = obj_val
                best_obj_iter = it

            if (
                self._check_convergence(it, obj_history, best_obj_iter, rel_tol)
                == "converged"
            ):
                break
            it += 1
        else:
            self.log(f"  Did not converge in {max_iters} iterations")

        self.finalize()
        self._optimize_history = obj_history
