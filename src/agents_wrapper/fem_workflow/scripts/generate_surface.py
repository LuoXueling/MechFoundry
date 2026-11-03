#!/usr/bin/env python
"""Predefined TripoSG surface generation for the four supported input modes."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parents[2]
TRIPOSG_ROOT = REPO_ROOT / "src/dependency/TripoSG"
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(TRIPOSG_ROOT))

from src.triposg_wrapper.triposg_pipeline_wrapper import TripoSGPipelineWrapper
from src.triposg_wrapper.utils import filter_largest_interior_component, sdf2surfacemesh

BOUNDS = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)

# How long a resident generator waits for a job before giving the GPU back.
# It holds the model the whole time -- around 4.4 GB of device memory measured
# on a loaded scribble variant -- and the only other thing that ever stops it is
# a forward solve asking for the card, so without this a session that generated
# a shape and went no further left the weights resident for as long as the
# machine stayed up.  The same half hour the tool worker waits, for the same
# reason: long enough to cover the pauses inside one sitting with the geometry
# card, short enough that a conversation which has moved on is not still paying.
IDLE_TIMEOUT = 1800.0

# How long the worker waits for a caller that has connected to say what it
# wants.  The monitor sends its request immediately, so this only ever expires
# on a caller that died mid-connect -- which would otherwise wedge the worker
# exactly as an unbounded accept does, with its idle clock never reached.
REQUEST_TIMEOUT = 30.0

# The code a resident worker runs with, so it can tell when that is no longer
# what is on disk.  It holds what it imported for as long as it lives, and a
# request carrying a setting its copy does not know -- the erosion radius, say
# -- would otherwise be served as if the setting were absent.
SOURCES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parents[3] / "triposg_wrapper" / "utils.py",
    Path(__file__).resolve().parents[3] / "triposg_wrapper" / "triposg_pipeline_wrapper.py",
)


def source_digest() -> str:
    digest = hashlib.sha256()
    try:
        for path in SOURCES:
            digest.update(path.read_bytes())
    except OSError:
        return ""
    return digest.hexdigest()


LOADED_DIGEST = source_digest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("text", "image", "scribble", "text_scribble"))
    parser.add_argument("--prompt")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--init-octree-depth", type=int, default=4)
    parser.add_argument("--octree-depth", type=int, default=7)
    parser.add_argument("--faces", type=int, default=-1)
    # No default: the image branch falls back to the pipeline's own 7.0, and
    # the other two modes refuse the flag outright rather than take a number
    # they cannot honour.
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--prompt-confidence", type=float, default=1.0)
    parser.add_argument("--scribble-confidence", type=float, default=0.4)
    # Voxels the largest-component filter erodes before choosing the one body to
    # keep.  Erosion parts pieces joined only by something thinner than twice
    # this, so any positive value drops thin members (slats, spindles, rails)
    # and whatever hangs from them; 0 keeps every part joined to the body and
    # still drops floating fragments.
    parser.add_argument("--erosion-radius", type=int, default=0)
    parser.add_argument("--weights-dir", type=Path)
    parser.add_argument("--serve", type=Path, help="Run a persistent local worker on this Unix socket")
    parser.add_argument("--preload", choices=("image", "scribble"), default="scribble")
    parser.add_argument("--idle-timeout", type=float, default=IDLE_TIMEOUT,
                        help="Stop serving after this long without a job")
    return parser.parse_args()


class SurfaceGenerator:
    """Keep one TripoSG variant resident and reuse it across generation jobs."""

    def __init__(self, device="cuda"):
        self.device = device
        self.pipe = None
        self.variant = None
        self.rmbg = None

    def load(self, variant: str, weights_dir: Path | None = None):
        if self.variant == variant and self.pipe is not None:
            return
        self.pipe = None
        self.rmbg = None
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        repo_id = "VAST-AI/TripoSG" if variant == "image" else "VAST-AI/TripoSG-scribble"
        default_name = "TripoSG" if variant == "image" else "TripoSG-scribble"
        weights = weights_dir or TRIPOSG_ROOT / "pretrained_weights" / default_name
        snapshot_download(repo_id=repo_id, local_dir=weights)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.pipe = TripoSGPipelineWrapper.from_pretrained(weights, dtype=dtype).to(
            self.device, dtype
        )
        self.variant = variant
        if variant == "image":
            from src.dependency.TripoSG.scripts.briarmbg import BriaRMBG
            rmbg_path = TRIPOSG_ROOT / "pretrained_weights/RMBG-1.4"
            snapshot_download(repo_id="briaai/RMBG-1.4", local_dir=rmbg_path)
            self.rmbg = BriaRMBG.from_pretrained(rmbg_path).to(self.device).eval()


def decimate_watertight(surface, faces: int):
    """``(surface reduced to about faces faces, None)`` by quadric edge collapse with its topology
    kept (MeshLab), or ``(surface, why)`` when it was kept whole: the result was not one watertight
    body, or decimation failed (pymeshlab missing included).  A cap must never turn a usable surface
    into a rejected one, and the result says when it was not applied."""
    import trimesh

    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(np.asarray(surface.vertices, dtype=np.float64),
                                   np.asarray(surface.faces, dtype=np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(faces), preservetopology=True,
                                                    preservenormal=True, qualitythr=0.3)
        mesh = ms.current_mesh()
        reduced = trimesh.Trimesh(mesh.vertex_matrix(), mesh.face_matrix(), process=True)
    except Exception as exc:
        return surface, f"skipped, failed: {type(exc).__name__}"
    if not reduced.is_watertight or len(reduced.split(only_watertight=False)) != 1:
        return surface, "skipped, not watertight"
    return reduced, None


def generate(args, resident: SurfaceGenerator | None = None):
    if args.mode is None or args.output is None:
        raise ValueError("--mode and --output are required for a generation job")
    if not 0.0 <= args.prompt_confidence <= 1.0:
        raise ValueError("--prompt-confidence must be between 0 and 1")
    if not 0.0 <= args.scribble_confidence <= 1.0:
        raise ValueError("--scribble-confidence must be between 0 and 1")
    if args.faces != -1 and not 1000 <= args.faces <= 300000:
        raise ValueError("--faces must be -1 (unlimited) or between 1000 and 300000")
    if not 0 <= args.seed <= 100:
        raise ValueError("--seed must be between 0 and 100")
    # A request from a caller older than the setting carries none.
    erosion_radius = getattr(args, "erosion_radius", 0)
    if not 0 <= erosion_radius <= 8:
        raise ValueError("--erosion-radius must be between 0 and 8 voxels")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("TripoSG surface generation requires CUDA")
    if args.mode in ("text", "text_scribble") and not args.prompt:
        raise ValueError(f"{args.mode} requires --prompt")
    if args.mode in ("image", "scribble", "text_scribble") and not args.input:
        raise ValueError(f"{args.mode} requires --input")
    if args.guidance_scale is not None and args.mode != "image":
        # Every mode but "image" runs on TripoSG-scribble, which upstream calls
        # CFG-distilled and samples with guidance_scale=0
        # (scripts/inference_triposg_scribble.py:37).  In the shared denoiser
        # (TripoSGPipelineWrapper._denoise_latents) a scale of 1 or less does
        # nothing, and a scale above 1 runs classifier-free guidance against
        # zeroed conditioning -- every denoising step twice, on a checkpoint
        # not trained for it.  Neither is what the caller asked for, so the
        # flag is refused rather than dropped.  Text and scribble steer
        # through prompt_confidence and scribble_confidence instead.
        raise ValueError(
            f"--guidance-scale does not apply to --mode {args.mode}: it runs on "
            "the CFG-distilled scribble checkpoint, which has no classifier-free "
            "guidance to scale; use --prompt-confidence and --scribble-confidence"
        )

    image_mode = args.mode == "image"
    resident = resident or SurfaceGenerator(args.device)
    resident.load("image" if image_mode else "scribble", args.weights_dir)
    pipe = resident.pipe
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    if args.mode == "text":
        latents = pipe.text2latent(
            args.prompt, generator=generator, num_inference_steps=args.steps,
            prompt_confidence=args.prompt_confidence,
        )
    elif args.mode in ("scribble", "text_scribble"):
        image = Image.open(args.input).convert("RGB")
        kwargs = {"generator": generator, "num_inference_steps": args.steps}
        if args.mode == "text_scribble":
            kwargs["prompt"] = args.prompt
        kwargs["scribble_confidence"] = args.scribble_confidence
        kwargs["prompt_confidence"] = args.prompt_confidence
        latents = pipe.scribble2latent(image, **kwargs)
    else:
        from src.dependency.TripoSG.scripts.image_process import prepare_image
        image = prepare_image(str(args.input), bg_color=np.ones(3), rmbg_net=resident.rmbg)
        guidance = ({} if args.guidance_scale is None
                    else {"guidance_scale": args.guidance_scale})
        latents = pipe.img2latent(image=image, generator=generator,
                                  num_inference_steps=args.steps, **guidance)

    sdf, _ = pipe.latent2sdf(latents, bounds=BOUNDS,
                             init_octree_depth=args.init_octree_depth,
                             octree_depth=args.octree_depth)
    sdf = filter_largest_interior_component(
        sdf, erosion_radius=erosion_radius, domain_length=2.0)
    # The face cap is applied to the one watertight body kept below (decimate_watertight), not to the
    # raw marching-cubes surface: decimating that first could leave the kept body open.
    surface = sdf2surfacemesh(sdf=sdf, n_faces=-1,
                              octree_depth=args.octree_depth, bounds=BOUNDS)
    if surface.is_empty:
        raise RuntimeError("TripoSG produced an empty surface")
    components = surface.split(only_watertight=False)
    discarded_components = 0
    if len(components) > 1:
        surface = max(components, key=lambda component: component.area)
        surface.remove_unreferenced_vertices()
        discarded_components = len(components) - 1
    if not surface.is_watertight:
        raise RuntimeError(
            "TripoSG produced no usable watertight surface component "
            f"(seed {args.seed}); try changing the random seed"
        )
    decimation = None
    if args.faces != -1 and len(surface.faces) > args.faces:
        surface, decimation = decimate_watertight(surface, args.faces)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    surface.export(args.output)
    result = {"artifacts": [str(args.output.resolve())],
              "vertices": len(surface.vertices), "faces": len(surface.faces),
              "discarded_components": discarded_components}
    if decimation:
        result["decimation"] = decimation
    return result


def serve(args):
    socket_path = args.serve.resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    resident = SurfaceGenerator(args.device)
    resident.load(args.preload, args.weights_dir)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(os.fspath(socket_path))
        server.listen(4)
        # The listening socket is what the idle clock runs on: it is reset by
        # every accepted job, and the accepted connection is handed back
        # blocking, so a generation that outlasts the timeout is not disturbed
        # by it.  Only the waiting is bounded.
        server.settimeout(args.idle_timeout)
        running = True
        while running:
            try:
                connection, _ = server.accept()
            except (socket.timeout, TimeoutError):
                break
            with connection:
                try:
                    connection.settimeout(REQUEST_TIMEOUT)
                    request = json.loads(connection.makefile("r", encoding="utf-8").readline())
                    # The request is in; the work it asks for takes as long as
                    # it takes, and answering it must not time out either.
                    connection.settimeout(None)
                    if request.get("action") == "shutdown":
                        response = {"ok": True, "result": "shutting_down"}
                        running = False
                    elif source_digest() != LOADED_DIGEST:
                        # Say so and stop: the caller runs this job itself, and
                        # the next preload starts a worker with the current code.
                        response = {"ok": False, "stale": True,
                                    "error": "generator code changed on disk since this worker loaded it"}
                        running = False
                    else:
                        job = argparse.Namespace(**request)
                        job.input = Path(job.input) if job.input else None
                        job.output = Path(job.output)
                        job.weights_dir = Path(job.weights_dir) if job.weights_dir else None
                        response = {"ok": True, "result": generate(job, resident)}
                except Exception as exc:
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                # A caller that has gone away -- the one this answer is for when
                # the request above timed out -- must cost this worker nothing
                # beyond the answer nobody reads.
                with contextlib.suppress(OSError):
                    connection.sendall(
                        (json.dumps(response, separators=(",", ":")) + "\n").encode()
                    )
    finally:
        server.close()
        socket_path.unlink(missing_ok=True)


def main():
    args = parse_args()
    if args.serve:
        serve(args)
    else:
        print(json.dumps(generate(args), separators=(",", ":")))
    # CUDA/PyTorch background state can make interpreter teardown linger after
    # all requested artifacts are safely written.  Flush protocol output and let
    # the OS reclaim the process (and GPU allocations) immediately on success.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
