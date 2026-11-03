from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import PIL
import PIL.Image
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from ..dependency.TripoSG.triposg.pipelines.pipeline_triposg import (
    TripoSGPipeline,
    retrieve_timesteps,
)
from .hierarchical_extraction import hierarchical_extraction
from .utils import filter_largest_interior_component, sdf2octree
from diffusers.image_processor import PipelineImageInput


class TripoSGPipelineWrapper(TripoSGPipeline):
    _optional_components = ["tokenizer", "text_encoder"]

    # TripoSG decodes logits to a truncated SDF  sdf = (2*sigmoid(logit) - 1) * T,
    # T = sdf_truncation_threshold, so the normalized SDF is bounded to [-T, T].
    # The per-node decoder Jacobian carries the sigmoid-slope factor
    #   d sdf / d z  =  2 T * sigmoid'(logit) * d logit/d z,
    #   sigmoid'(logit) = sigmoid(1-sigmoid) = (1 - (sdf/T)^2) / 4,
    # which is O(1) only near the zero-crossing (logit~0, sdf~0) and decays to 0
    # on the saturated plateaus |sdf|->T.  A node on the plateau is therefore
    # (a) off the decoder's surface (its sdf=0 isocontour) -- differentiating it
    # asks how the SDF moves far from the surface, not how the surface moves --
    # and (b) has a Jacobian whose magnitude vanishes and whose *direction* is
    # swamped by the (roughly constant) fp decode noise floor.  Such nodes are
    # filtered out before the latent VJP (see _compute_latent_grad): they would
    # otherwise inject noise-directioned, near-zero-||J|| rows into the pull-back
    # metric J^T J that the CG correction solves against.
    sdf_truncation_threshold = 0.01

    # Keep only query nodes whose |normalized SDF| <= sdf_linear_keep_frac * T,
    # i.e. the near-linear band of the sigmoid.  frac controls how much of the
    # truncation band is trusted: sigmoid'(logit)/peak = 1 - frac^2, so at
    # frac=0.5 kept nodes have decoder sensitivity >= 0.75 of peak (|logit| <~ 1.1),
    # and at frac=0.75 >= 0.44 of peak (|logit| <~ 2.0).  0.75 widens the pool
    # while still excluding the saturated plateau tail where ||J|| -> 0.
    sdf_linear_keep_frac = 0.75

    def __init__(
        self,
        vae,
        transformer,
        scheduler,
        image_encoder_dinov2,
        feature_extractor_dinov2,
        tokenizer=None,
        text_encoder=None,
    ):
        """Build either the image or scribble/text TripoSG pipeline.

        ``tokenizer`` and ``text_encoder`` are present in the
        ``VAST-AI/TripoSG-scribble`` checkpoint and absent from the original
        image-only checkpoint.  Declaring them as optional Diffusers components
        lets the same wrapper class load both checkpoint layouts.
        """
        super().__init__(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder_dinov2=image_encoder_dinov2,
            feature_extractor_dinov2=feature_extractor_dinov2,
        )
        self.register_modules(tokenizer=tokenizer, text_encoder=text_encoder)

    @property
    def supports_text_and_scribble(self) -> bool:
        return self.tokenizer is not None and self.text_encoder is not None

    def _require_text_and_scribble_checkpoint(self):
        if not self.supports_text_and_scribble:
            raise ValueError(
                "text2latent/scribble2latent require the "
                "VAST-AI/TripoSG-scribble checkpoint (the image-only TripoSG "
                "checkpoint has no tokenizer or text encoder)."
            )

    @staticmethod
    def _image_batch_size(image: PipelineImageInput) -> int:
        if isinstance(image, PIL.Image.Image):
            return 1
        if isinstance(image, list):
            if not image:
                raise ValueError("image/scribble lists must not be empty")
            return len(image)
        if isinstance(image, torch.Tensor):
            return image.shape[0]
        raise ValueError("Invalid input type for image")

    def encode_text(self, prompt, device, num_shapes_per_prompt, batch_size=None):
        """Encode one prompt or a batch-aligned list of prompts."""
        self._require_text_and_scribble_checkpoint()
        if isinstance(prompt, str):
            prompts = [prompt] * (batch_size or 1)
        elif isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            prompts = prompt
            if not prompts:
                raise ValueError("prompt lists must not be empty")
            if batch_size is not None and len(prompts) != batch_size:
                raise ValueError(
                    f"Received {len(prompts)} prompts for an image batch of "
                    f"size {batch_size}."
                )
        else:
            raise TypeError("prompt must be a string or a list of strings")

        input_ids = self.tokenizer(
            prompts,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(device)
        text_embeds = self.text_encoder(input_ids).last_hidden_state
        text_embeds = text_embeds.repeat_interleave(num_shapes_per_prompt, dim=0)
        return text_embeds, torch.zeros_like(text_embeds)

    @torch.no_grad()
    def _denoise_latents(
        self,
        *,
        batch_size,
        encoder_hidden_states,
        encoder_hidden_states_2=None,
        num_inference_steps,
        num_tokens,
        timesteps,
        guidance_scale,
        num_shapes_per_prompt,
        generator,
        latents,
        attention_kwargs,
        callback_on_step_end,
        callback_on_step_end_tensor_inputs,
    ):
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        device = self._execution_device

        if self.do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [torch.zeros_like(encoder_hidden_states), encoder_hidden_states], dim=0
            )
            if encoder_hidden_states_2 is not None:
                encoder_hidden_states_2 = torch.cat(
                    [torch.zeros_like(encoder_hidden_states_2), encoder_hidden_states_2],
                    dim=0,
                )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)
        latents = self.prepare_latents(
            batch_size * num_shapes_per_prompt,
            num_tokens,
            self.transformer.config.in_channels,
            encoder_hidden_states.dtype,
            device,
            generator,
            latents,
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                transformer_kwargs = {
                    "encoder_hidden_states": encoder_hidden_states,
                    "attention_kwargs": attention_kwargs,
                    "return_dict": False,
                }
                if encoder_hidden_states_2 is not None:
                    transformer_kwargs["encoder_hidden_states_2"] = encoder_hidden_states_2
                noise_pred = self.transformer(
                    latent_model_input,
                    t.expand(latent_model_input.shape[0]).to(dtype=latents.dtype),
                    **transformer_kwargs,
                )[0]
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_cond - noise_pred_uncond
                    )

                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {
                        name: {"latents": latents}[name]
                        for name in callback_on_step_end_tensor_inputs
                    }
                    callback_outputs = callback_on_step_end(
                        self, i, t, callback_kwargs
                    )
                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps
                    and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return latents

    # Copied from TripoSGPipeline.__call__ and removed mesh extraction code
    @torch.no_grad()
    def img2latent(
        self,
        image: PipelineImageInput,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 7.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        if self.supports_text_and_scribble:
            raise ValueError(
                "img2latent is for the image-only TripoSG checkpoint; use "
                "scribble2latent with the TripoSG-scribble checkpoint."
            )
        batch_size = self._image_batch_size(image)
        device = self._execution_device
        image_embeds, _ = self.encode_image(
            image, device, num_shapes_per_prompt
        )
        return self._denoise_latents(
            batch_size=batch_size,
            encoder_hidden_states=image_embeds,
            num_inference_steps=num_inference_steps,
            num_tokens=num_tokens,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_shapes_per_prompt=num_shapes_per_prompt,
            generator=generator,
            latents=latents,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

    @torch.no_grad()
    def scribble2latent(
        self,
        scribble: PipelineImageInput,
        prompt: Union[str, List[str]] = "",
        num_inference_steps: int = 16,
        num_tokens: int = 512,
        timesteps: List[int] = None,
        guidance_scale: float = 0.0,
        num_shapes_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_confidence: Optional[float] = None,
        scribble_confidence: float = 0.4,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    ):
        """Generate latents from a scribble, optionally guided by text."""
        self._require_text_and_scribble_checkpoint()
        batch_size = self._image_batch_size(scribble)
        device = self._execution_device
        text_embeds, _ = self.encode_text(
            prompt, device, num_shapes_per_prompt, batch_size=batch_size
        )
        image_embeds, _ = self.encode_image(
            scribble, device, num_shapes_per_prompt
        )
        has_prompt = bool(prompt) if isinstance(prompt, str) else any(prompt)
        if prompt_confidence is None:
            prompt_confidence = 1.0 if has_prompt else 0.0
        attention_kwargs = dict(attention_kwargs or {})
        attention_kwargs.update(
            cross_attention_scale=prompt_confidence,
            cross_attention_2_scale=scribble_confidence,
        )
        return self._denoise_latents(
            batch_size=batch_size,
            encoder_hidden_states=text_embeds,
            encoder_hidden_states_2=image_embeds,
            num_inference_steps=num_inference_steps,
            num_tokens=num_tokens,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_shapes_per_prompt=num_shapes_per_prompt,
            generator=generator,
            latents=latents,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )

    @torch.no_grad()
    def text2latent(
        self,
        prompt: Union[str, List[str]],
        **kwargs,
    ):
        """Generate latents from text with the scribble branch disabled."""
        self._require_text_and_scribble_checkpoint()
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        if batch_size < 1:
            raise ValueError("prompt lists must not be empty")
        blank = [PIL.Image.new("RGB", (224, 224), "white") for _ in range(batch_size)]
        scribble = blank[0] if batch_size == 1 else blank
        kwargs["scribble_confidence"] = 0.0
        if "prompt_confidence" not in kwargs:
            kwargs["prompt_confidence"] = 1.0
        return self.scribble2latent(scribble, prompt=prompt, **kwargs)

    def make_geometric_func(self, latents):
        """Create an SDF query function from (detached) latents.

        Returns a callable ``(N,3) → (N,) float`` that evaluates the
        TripoSG decoder, scaling the sigmoid output by
        ``sdf_truncation_threshold`` (the /100 truncation convention).  The
        latent/query dtype follows the VAE weight dtype — if a caller
        upcasts ``self.vae`` to fp32 (e.g. ``LatentShapeOpt`` does this
        to avoid fp16 quantization of latent updates), the geometric
        func runs in fp32 too.
        """
        if latents is None or not isinstance(latents, torch.Tensor):
            raise TypeError("latents must be a torch.Tensor")
        if latents.ndim != 3 or latents.shape[0] != 1:
            raise ValueError(
                "Geometry decoding requires latents for exactly one shape with "
                "shape (1, num_tokens, channels); slice batched latents as "
                "latents[i:i+1]."
            )
        target_dtype = next(self.vae.parameters()).dtype
        latents = latents.detach().to(
            device=self._execution_device, dtype=target_dtype
        )
        vae = self.vae
        return (
            lambda x: (
                torch.sigmoid(
                    vae.decode(latents, sampled_points=x.to(target_dtype)).sample
                )
                * 2
                - 1
            )
            * self.sdf_truncation_threshold
        )

    @torch.no_grad()
    def latent2sdf(
        self,
        latents: Optional[torch.FloatTensor] = None,
        bounds: Union[Tuple[float], List[float], float] = (
            -1.005,
            -1.005,
            -1.005,
            1.005,
            1.005,
            1.005,
        ),
        init_octree_depth: int = 8,
        octree_depth: int = 9,
    ):
        geometric_func = self.make_geometric_func(latents)

        sdf, anchors = hierarchical_extraction(
            geometric_func=geometric_func,
            bounds=bounds,
            init_octree_depth=init_octree_depth,
            octree_depth=octree_depth,
            device=self._execution_device,
            sdf_truncation=self.sdf_truncation_threshold,
        )

        return sdf, anchors

    @torch.no_grad()
    def latent2octree(
        self,
        latents,
        bounds,
        init_octree_depth,
        octree_depth,
        domain_length,
    ):
        """Decode latents → filtered SDF grid → octree mesh dict.

        Returns a dict with ``nodes``, ``elements``, ``sdf``, ``grid_sdf``,
        ``existing_anchors``, and optional hanging node constraints.
        """
        device = self._execution_device
        geometric_func = self.make_geometric_func(latents)

        sdf_grid, anchors = hierarchical_extraction(
            geometric_func=geometric_func,
            bounds=bounds,
            init_octree_depth=init_octree_depth,
            octree_depth=octree_depth,
            device=device,
            sdf_truncation=self.sdf_truncation_threshold,
        )
        sdf_grid = filter_largest_interior_component(
            sdf_grid,
            domain_length=domain_length,
        )
        nodes, nodal_sdf, elements, edge_h, face_h = sdf2octree(
            cell_tensor=anchors,
            sdf_tensor=sdf_grid,
            domain_length=domain_length,
        )

        octree_mesh = {
            "nodes": nodes,
            "elements": elements,
            "sdf": nodal_sdf,
            "grid_sdf": sdf_grid,
            "existing_anchors": anchors,
        }
        if edge_h is not None and len(edge_h) > 0:
            octree_mesh["edge_hanging"] = edge_h
        if face_h is not None and len(face_h) > 0:
            octree_mesh["face_hanging"] = face_h
        return octree_mesh

    def mesh2latent(
        self,
        mesh,
        n_points=100000,
        seed=42,
        sample_seed=None,
        device=None,
        dtype=None,
    ):
        """Encode a surface mesh into latents (mean of the latent distribution).

        Replicates the encode used in the latent shape-optimization example:
        area-weighted
        uniform surface sampling → ``(1, N, 6)`` ``[points, face_normals]`` tensor
        → ``vae.encode(surface, seed=seed).latent_dist.mean``.  ``device``/``dtype``
        default to the VAE's own (so this returns fp16 before the VAE is upcast and
        fp32 after — matching each caller's needs).

        Args:
            mesh: a watertight ``trimesh.Trimesh`` in the decoder's normalized frame.
            n_points: surface samples fed to the encoder.
            seed: encoder sampling seed (``vae.encode`` ``seed`` kwarg).
            sample_seed: optional seed for the surface point sampling (None = random).
        """
        import trimesh  # noqa: F401 (dependency; imported lazily like other paths)

        points, face_idx = trimesh.sample.sample_surface(
            mesh, n_points, seed=sample_seed
        )
        normals = mesh.face_normals[face_idx]
        p = next(self.vae.parameters())
        device = device or p.device
        dtype = dtype or p.dtype
        surface = (
            torch.cat(
                [
                    torch.FloatTensor(np.asarray(points)),
                    torch.FloatTensor(np.asarray(normals)),
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .to(device, dtype)
        )
        with torch.no_grad():
            return self.vae.encode(surface, seed=seed).latent_dist.mean

    def _decode_chain(self, latents, query_pts, chunk_size=10000):
        """Shared forward chain for VJP/JVP -- mirrors ``vae.decode`` exactly
        but keeps ``latents`` attached to the graph (no ``.detach()``), so it
        can be differentiated in reverse mode (``make_vjp_cache`` /
        ``compute_latent_grad``) or forward mode (``compute_latent_jvp``)
        without the two paths ever drifting apart.

        Follows the vae weight dtype.  fp16 here causes severe quantization
        of latent updates (a 1e-3 update on a unit-magnitude latent gets
        rounded to 0 because fp16 has only ~1e-3 relative precision), which
        destroys gradient signal in latent shape optimization.  Callers
        (e.g. ``LatentShapeOpt``) upcast the vae to fp32 to avoid this; for
        legacy fp16 callers the behavior is unchanged.

        Args:
            latents: (1, T, C) latent tensor, dtype following caller.
            query_pts: (1, M, 3) query positions in normalized space.
            chunk_size: Max queries per decoder forward chunk.
        Returns:
            sdf_pred: (M,) float32.
        """
        vae = self.vae
        target_dtype = next(vae.parameters()).dtype
        z = vae.post_quant(latents.to(target_dtype))

        hidden_states = z
        for block in vae.decoder.blocks[:-1]:
            hidden_states = block(hidden_states)
        kv_cache = hidden_states

        all_sdf = []
        M = query_pts.shape[1]
        for i in range(0, M, chunk_size):
            chunk = query_pts[:, i : i + chunk_size, :].to(z.dtype)
            queries = vae.embedder(chunk)
            q = vae.decoder.proj_query(queries)
            l = vae.decoder.blocks[-1](q, encoder_hidden_states=kv_cache)
            logits = vae.decoder.proj_out(vae.decoder.norm_out(l))
            logits = logits * -1
            sdf_chunk = (torch.sigmoid(logits) * 2 - 1) * self.sdf_truncation_threshold
            all_sdf.append(sdf_chunk)

        return torch.cat(all_sdf, dim=1).squeeze(0).squeeze(-1).float()

    def make_vjp_cache(self, latents, query_pts, chunk_size=10000):
        """Build the decode graph once; return a callable ``w -> J^T @ w``.

        Reuses the forward pass (the expensive ``blocks[:-1]`` self-attention
        over all latent tokens) across repeated VJPs at different ``w`` --
        e.g. across the conjugate-gradient iterations in the pull-back-metric
        correction -- via ``retain_graph=True``.

        Args:
            latents: (1, T, C) float32 latent tensor.
            query_pts: (1, M, 3) float32 query positions in normalized space.
            chunk_size: Max queries per decoder forward chunk.
        Returns:
            vjp(grad_outputs) -> latent_grad, same shape as ``latents``.
        """
        latents = latents.detach().requires_grad_(True)
        sdf_pred = self._decode_chain(latents, query_pts, chunk_size)

        def vjp(grad_outputs):
            return torch.autograd.grad(
                sdf_pred, latents, grad_outputs=grad_outputs, retain_graph=True
            )[0].float()

        return vjp

    def make_gn_matvec(self, latents, query_pts, chunk_size=10000):
        """Build the decode graph once; return a callable ``x -> (J^T J) @ x``.

        The Gauss-Newton / pull-back-metric operator ``J^T J`` (J the decoder
        Jacobian) applied matrix-free in **pure reverse mode**, for the
        conjugate-gradient solve in ``LatentShapeOpt._pullback_cg_correction``.
        Unlike the forward-JVP-then-VJP matvec (``compute_latent_jvp`` + ``vjp``),
        the expensive primal forward (``sdf``) and the first-order graph
        ``J^T w`` are built **once** here (``create_graph=True``); each matvec is
        then two backward passes on the cached graph -- no per-matvec primal
        recompute:

            J x   = grad( <J^T w, x>, w )      # w is a dummy output-space probe;
                                               # J^T w is linear in w, so d/dw = J
            J^T J x = grad( sdf, latents, J x )

        Runs under the **MATH** SDPA backend: double-backward (the second-order
        grad through attention) is required here, and only the MATH kernel
        implements it -- the mem-efficient/flash kernels raise
        ``derivative for aten::_scaled_dot_product_efficient_attention_backward
        is not implemented``.  So this trades the JVP path's per-matvec primal
        recompute for a once-built graph, but stays on the same (slow, memory-
        heavy) MATH kernel; whether it is a net win is settled by benchmark.
        Decoder graph is double-backward safe (no in-place ops).

        Args:
            latents: (1, T, C) float32 latent tensor.
            query_pts: (1, M, 3) float32 query positions in normalized space.
            chunk_size: Max queries per decoder forward chunk.
        Returns:
            gn_matvec(x) -> J^T J @ x, same shape as ``latents`` (float32).
        """
        latents = latents.detach().requires_grad_(True)
        with sdpa_kernel(SDPBackend.MATH):
            sdf_pred = self._decode_chain(latents, query_pts, chunk_size)  # (M,)
            w = torch.zeros_like(sdf_pred, requires_grad=True)
            # J^T w as a graph function of w (linear in w); built once, reused.
            (jt_w,) = torch.autograd.grad(
                sdf_pred, latents, grad_outputs=w, create_graph=True
            )

        def gn_matvec(x):
            with sdpa_kernel(SDPBackend.MATH):
                # d/dw <J^T w, x> = J x  (output space, (M,))
                (jx,) = torch.autograd.grad(jt_w, w, grad_outputs=x, retain_graph=True)
                # J^T (J x)  (latent space)
                (jtjx,) = torch.autograd.grad(
                    sdf_pred, latents, grad_outputs=jx, retain_graph=True
                )
            return jtjx.float()

        return gn_matvec

    def compute_latent_grad(
        self,
        latents,
        query_pts,
        grad_outputs,
        chunk_size=10000,
    ):
        """Compute dL/d_latents via VJP through the VAE decoder.

        Single-shot convenience wrapper around ``make_vjp_cache`` -- builds
        the decode graph once and applies it to ``grad_outputs`` (the FEM
        adjoint dL/d_sdf scaled to normalized space).

        Args:
            latents: (1, T, C) float32 latent tensor.
            query_pts: (1, M, 3) float32 query positions in normalized space.
            grad_outputs: (M,) float32 FEM adjoint · sdf_scale at query nodes.
            chunk_size: Max queries per decoder forward chunk.
        Returns:
            latent_grad: same shape as latents, float32.
        """
        return self.make_vjp_cache(latents, query_pts, chunk_size)(grad_outputs)

    def compute_latent_jvp(self, latents, query_pts, direction, chunk_size=10000):
        """Forward-mode JVP: ``J @ direction``, output in sdf-space (M,).

        Forces the math SDPA backend -- the decoder's cross-attention
        (``TripoSGAttnProcessor2_0``) dispatches to the memory-efficient
        kernel by default, which has no forward-AD rule and raises
        ``NotImplementedError`` under ``torch.func.jvp`` otherwise.

        Args:
            latents: (1, T, C) float32 latent tensor.
            query_pts: (1, M, 3) float32 query positions in normalized space.
            direction: (1, T, C) float32 tangent direction in latent space.
            chunk_size: Max queries per decoder forward chunk.
        Returns:
            jvp_out: (M,) float32.
        """
        latents = latents.detach()

        def f(lat):
            return self._decode_chain(lat, query_pts, chunk_size)

        with sdpa_kernel(SDPBackend.MATH):
            _, jvp_out = torch.func.jvp(f, (latents,), (direction,))
        return jvp_out.float()
