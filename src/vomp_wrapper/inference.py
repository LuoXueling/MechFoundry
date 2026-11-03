#!/usr/bin/env python
"""Cold or resident VoMP inference entry point."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))

from src.vomp_wrapper.artifact import GRID_SIZE, SCHEMA_VERSION as ARTIFACT_SCHEMA, write_artifact
from src.vomp_wrapper.availability import PINNED_MODEL_REVISION, PINNED_SOURCE_REVISION, settings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# Which renderer draws the views VoMP reads.
#
# "raster" puts Blender's own scene through nvdiffrast: 4.12s against 71s for
# the same 150 views, silhouette IoU 0.9999, per-pixel correlation 0.954, and
# a whole prediction in 31.7s against 118s.
#
# It does not yet render what Blender renders closely enough to give the same
# answer.  Run end to end against the path-traced baseline on the same mesh,
# the materials move: Young's modulus by 25% at the median and 4.2x in the
# mean, density by 9.7%, with correlations of 0.62 and 0.39.  The model reads
# more of the shading than a 0.954 pixel correlation suggests; what is missing
# is the bounce Cycles is allowed and the true shape of the area lights.
# Closing that is what would make the two interchangeable.  Until then this is
# the fast answer rather than the same answer, and "blender" -- selectable by
# VOMP_RENDERER or --renderer -- is what to check a prediction against.
RENDERER = os.environ.get("VOMP_RENDERER", "raster")


# glTF is Y-up; VoMP was trained on assets authored Z-up (the USD packs its
# dataset was built from), and it has learned that pose: a Hunyuan3D
# reconstruction of a GVM test chair, as generated, correlated 0.16 with
# VoMP's prediction on the real chair (leather 13.6 GPa); the same file turned
# Z-up, 0.95 (leather 0.34 GPa against 0.37).  So a glTF surface goes to VoMP
# turned Z-up, (x, y, z) -> (x, -z, y), and its voxels are turned back after.
Y_UP_FORMATS = (".glb", ".gltf")


def to_z_up(points):
    import numpy as np

    points = np.asarray(points)
    return np.stack([points[:, 0], -points[:, 2], points[:, 1]], axis=1)


def from_z_up(points):
    import numpy as np

    points = np.asarray(points)
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], axis=1)


def present_z_up(surface: Path, directory: Path) -> Path:
    """The surface as VoMP should see it: a Z-up copy of a glTF, else the file itself.

    Node transforms are baked in, because VoMP's voxeliser reads each
    geometry's own vertices and ignores the scene graph around them.
    """
    import numpy as np
    import trimesh

    if Path(surface).suffix.lower() not in Y_UP_FORMATS:
        return Path(surface)
    turn = np.eye(4)
    turn[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
    parts = trimesh.load(surface, force="scene", process=False).dump()
    for part in parts:
        part.apply_transform(turn)
    directory.mkdir(parents=True, exist_ok=True)
    turned = directory / "surface_z_up.glb"
    trimesh.Scene(parts).export(turned)
    return turned


def restore_frame(results: dict, surface: Path, presented: Path | None = None) -> dict:
    """VoMP's voxel coordinates at the points they were sampled, in the file's frame and scale.

    ``presented`` is the file VoMP actually ran on, when it was a Z-up copy of
    ``surface``; its voxels are turned back into ``surface``'s frame.
    """
    import numpy as np
    # The 64³ grid spans the bounding cube: centre = bbox centre, scale = the
    # largest bbox extent.  That frame is upstream VoMP's, not ours --
    # vomp/inference/utils.py::get_mesh_transform_params at the checkout
    # pinned in src/dependency/VoMP (ac6826f), which is not vendored.
    from vomp.inference.utils import get_mesh_transform_params

    coords = np.asarray(results["voxel_coords_world"], dtype=np.float64)
    # VoMP lifts features at its voxeliser's lattice points but reports each
    # voxel as (index + 1/2) / 64 - 1/2, half a cell above where it was
    # sampled.  Taken off here, so the grid receives the lattice points.
    coords = coords - 0.5 / GRID_SIZE
    presented = Path(presented or surface)
    center, scale = get_mesh_transform_params(str(presented))
    coords = coords * scale + np.asarray(center)
    if presented != Path(surface):
        # The turn maps the lattice onto itself, so nothing is resampled.
        coords = from_z_up(coords)
        center = from_z_up(np.asarray(center, dtype=np.float64)[None])[0]
    return {**results, "voxel_coords_world": coords,
            "transform_center": center, "transform_scale": scale}


class VompRuntime:
    def __init__(self, root: Path, weights: Path, blender: str | None = None,
                 use_trt: bool = False, renderer: str | None = None):
        self.root = Path(root).resolve()
        self.weights = Path(weights).resolve()
        self.blender = blender
        self.use_trt = bool(use_trt)
        self.renderer = (renderer or RENDERER).lower()
        if self.renderer not in {"raster", "blender"}:
            raise ValueError(
                f"unknown renderer {self.renderer!r}; expected 'raster' or 'blender'"
            )
        self.patches: list[str] = []
        self.model = None

    def preload(self):
        if self.model is not None:
            return
        sys.path.insert(0, os.fspath(self.root))
        from vomp.inference import Vomp
        self.model = Vomp.from_checkpoint(
            config_path=str(self.weights / "inference.json"),
            geometry_checkpoint_dir=str(self.weights / "geometry_transformer.pt"),
            matvae_checkpoint_dir=str(self.weights / "matvae.safetensors"),
            normalization_params_path=str(self.weights / "normalization_params.json"),
            use_trt=self.use_trt,
        )
        self.model.eval()
        # VoMP otherwise loads its 1.1 GB DINO encoder on the first inference,
        # making the button pay model startup even though the resident worker
        # was requested while the user selected this mode.  Warm the shared
        # lazy instance before the server advertises readiness.
        from vomp.inference.utils import LazyLoadDino
        encoder = LazyLoadDino(
            model_name="dinov2_vitl14_reg", device=self.model.device,
            use_trt=self.use_trt,
        ).get_model()
        from src.vomp_wrapper.patches import install
        # The encoder is a shared lazy instance, so patching the one warmed
        # here is patching the one the feature stage will ask for.  Left alone
        # under TensorRT, which has its own precision.
        self.patches = install(
            self.model, rasterize=self.renderer == "raster",
            half_encoder=not self.use_trt, dino=encoder,
        )

    def infer(self, *, surface: Path, artifact: Path, cache: Path,
              unit_system: str, num_views: int, dino_batch_size: int,
              seed: int, blender: str | None = None) -> list[str]:
        self.preload()
        surface = surface.resolve()
        cache.mkdir(parents=True, exist_ok=True)
        key = {
            "surface_sha256": _sha256(surface),
            "source_revision": PINNED_SOURCE_REVISION,
            "model_revision": PINNED_MODEL_REVISION,
            "num_views": int(num_views), "dino_batch_size": int(dino_batch_size),
            "seed": int(seed), "unit_system": unit_system,
            # Two renderers draw different pixels, so they cache separately --
            # an artifact built from path-traced views is not the one these
            # settings now ask for.
            "renderer": self.renderer,
            # What the rasteriser draws: a mesh's own base colour, every
            # material of a multi-material mesh, and both sides of a face.
            **({"raster_shading": "base-colour-per-material-two-sided-large-textures"} if self.renderer == "raster" else {}),
            "artifact_schema": ARTIFACT_SCHEMA,
            # Voxels projected as VoMP was trained (``patches.TURNED_FORMATS``);
            # artifacts from the interlude that turned them are not reused.
            "voxel_frame": "file",
            # glTF surfaces are shown to VoMP Z-up, the pose it was trained on.
            "pose": "z-up",
        }
        manifest = artifact / "manifest.json"
        key_path = artifact / "request.json"
        try:
            if manifest.is_file() and json.loads(key_path.read_text(encoding="utf-8")) == key:
                # Artifacts created before surface previews were introduced are
                # still valid inference caches.  Backfill only the cheap review
                # assets instead of forcing Blender and DINO to run again.
                from src.vomp_wrapper.visualize import surface_visualizations_current
                if not surface_visualizations_current(artifact):
                    from src.vomp_wrapper.artifact import load_artifact
                    from src.vomp_wrapper.visualize import write_surface_visualizations
                    write_surface_visualizations(load_artifact(artifact), surface)
                return [str(artifact.resolve()), str(manifest.resolve())]
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        # Unit conversion happens only while publishing the artifact.  Keep it
        # out of the expensive render/inference cache identity so changing the
        # workflow unit system can reuse the same views and raw prediction.
        inference_key = {name: value for name, value in key.items()
                         if name not in ("unit_system", "artifact_schema")}
        job = cache / hashlib.sha256(
            json.dumps(inference_key, sort_keys=True).encode()
        ).hexdigest()[:20]
        job.mkdir(parents=True, exist_ok=True)
        blender_path = blender or self.blender
        if not blender_path and self.renderer != "raster":
            raise ValueError("mesh inference requires VOMP_BLENDER or --blender")
        presented = present_z_up(surface, job)
        results = self.model.get_mesh_materials(
            str(presented), output_dir=str(job), num_views=int(num_views),
            blender_path=blender_path, query_points="voxel_centers",
            dino_batch_size=int(dino_batch_size), seed=int(seed),
            # Match the released model's training/inference regime. Sparse
            # interactive caps produced spatially meaningless nearest-filled
            # fields on multi-part assets.
            # The scale is restored below, with VoMP's half-cell offset taken
            # off (``restore_frame``).
            max_voxels=32768, return_original_scale=False,
            # Only Blender has jobs to spread: four processes took its 150
            # frames from 159s to 71s, eight took 74s, and CPU-only eight took
            # 78s, so the gain is spent by four and the ceiling is not the GPU.
            # Four rather than the core count because this runs beside a solver
            # that wants the rest of the machine.  The rasteriser renders every
            # view on one device in 4.12s and dispatches through the same
            # single-process path, so it asks for one.
            num_render_jobs=1 if self.renderer == "raster" else 4,
        )
        results = restore_frame(results, surface, presented)
        temporary = artifact.with_name(f".{artifact.name}.{os.getpid()}.tmp")
        backup = artifact.with_name(f".{artifact.name}.{os.getpid()}.old")
        for path in (temporary, backup):
            if path.exists():
                shutil.rmtree(path)
        temporary.mkdir(parents=True)
        material = write_artifact(
            temporary, results, unit_system=unit_system, provenance=key
        )
        from src.vomp_wrapper.visualize import write_surface_visualizations
        write_surface_visualizations(material, surface)
        (temporary / "request.json").write_text(
            json.dumps(key, indent=2, sort_keys=True), encoding="utf-8"
        )
        if artifact.exists():
            os.replace(artifact, backup)
        try:
            os.replace(temporary, artifact)
        except BaseException:
            if backup.exists() and not artifact.exists():
                os.replace(backup, artifact)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return [str(artifact.resolve()), str((artifact / "manifest.json").resolve())]


def parser() -> argparse.ArgumentParser:
    _, root, weights, blender = settings()
    result = argparse.ArgumentParser()
    result.add_argument("--surface", type=Path, required=True)
    result.add_argument("--artifact", type=Path, required=True)
    result.add_argument("--cache", type=Path, required=True)
    result.add_argument("--unit-system", required=True)
    # Match the released model's documented configuration. Fewer views are a
    # diagnostic preview only and do not preserve useful spatial assignment.
    result.add_argument("--num-views", type=int, default=150)
    result.add_argument("--dino-batch-size", type=int, default=16)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--vomp-root", type=Path, default=root)
    result.add_argument("--weights", type=Path, default=weights)
    result.add_argument("--blender", default=blender)
    result.add_argument("--use-trt", action="store_true")
    result.add_argument("--renderer", choices=("raster", "blender"), default=RENDERER)
    return result


def run(args, runtime: VompRuntime | None = None):
    runtime = runtime or VompRuntime(args.vomp_root, args.weights, args.blender,
                                     args.use_trt, getattr(args, "renderer", None))
    return runtime.infer(
        surface=args.surface, artifact=args.artifact, cache=args.cache,
        unit_system=args.unit_system, num_views=args.num_views,
        dino_batch_size=args.dino_batch_size, seed=args.seed, blender=args.blender,
    )


def main():
    args = parser().parse_args()
    print(json.dumps({"artifacts": run(args)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
