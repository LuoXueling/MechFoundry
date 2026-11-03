#!/usr/bin/env python
"""PartField inference and segmentation, runnable cold or by the resident server."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))

from src.partfield_wrapper.availability import settings


class _FeaturesSaved(Exception):
    """Stop upstream prediction after its useful feature array is durable."""


def _without_diagnostic_ply(predict_step):
    """Skip PartField's unconditional PCA and diagnostic feature PLY export.

    Upstream saves the full face-feature array immediately before computing a
    three-component PCA solely to colour ``feat_pca_*.ply``.  The workflow uses
    the feature array, not that visualization.  Intercepting the successful
    save here keeps the external checkout unmodified and lets Lightning finish
    the prediction step normally.
    """
    def feature_only(*args, **kwargs):
        original_save = np.save

        def save_then_stop(file, array, *save_args, **save_kwargs):
            original_save(file, array, *save_args, **save_kwargs)
            name = Path(os.fspath(file)).name
            if name.startswith("part_feat_") and name.endswith(".npy"):
                raise _FeaturesSaved

        np.save = save_then_stop
        try:
            return predict_step(*args, **kwargs)
        except _FeaturesSaved:
            return None
        finally:
            np.save = original_save

    return feature_only


@contextlib.contextmanager
def _without_input_ply():
    """Suppress the dataloader's unused canonical-input PLY copy."""
    import trimesh

    original_export = trimesh.Trimesh.export

    def export(mesh, file_obj=None, *args, **kwargs):
        try:
            target = Path(os.fspath(file_obj))
        except TypeError:
            target = None
        if (target is not None and target.suffix.lower() == ".ply"
                and target.name.startswith("input_")):
            return None
        return original_export(mesh, file_obj, *args, **kwargs)

    trimesh.Trimesh.export = export
    try:
        yield
    finally:
        trimesh.Trimesh.export = original_export


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
            text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class PartFieldRuntime:
    def __init__(self, root: Path, checkpoint: Path, points_per_face: int = 500):
        self.root = Path(root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.points_per_face = int(points_per_face)
        self.model = None
        self.trainer = None
        self._cached_surface_key = None
        self._cached_surface_sha256 = None
        self._cached_mesh = None
        self._source_revision = None

    def _surface_key(self, surface: Path):
        stat = surface.stat()
        return (str(surface.resolve()), stat.st_size, stat.st_mtime_ns)

    def _surface_sha256(self, surface: Path) -> str:
        key = self._surface_key(surface)
        if key != self._cached_surface_key:
            self._cached_surface_key = key
            self._cached_surface_sha256 = _sha256(surface)
            self._cached_mesh = None
        return self._cached_surface_sha256

    def _surface_mesh(self, surface: Path):
        import trimesh

        self._surface_sha256(surface)
        if self._cached_mesh is None:
            self._cached_mesh = trimesh.load(surface, force="mesh", process=False)
        return self._cached_mesh

    def _config(self, data_path: Path):
        sys.path.insert(0, os.fspath(self.root))
        from partfield.config import setup
        namespace = argparse.Namespace(
            config_file=str(self.root / "configs" / "final" / "demo.yaml"),
            opts=[
                "continue_ckpt", str(self.checkpoint),
                "result_name", "features",
                "dataset.data_path", str(data_path),
                "dataset.val_num_workers", "0",
                "dataset.val_batch_size", "1",
                "n_point_per_face", str(self.points_per_face),
                "n_sample_each", "10000",
                "preprocess_mesh", "False",
            ],
        )
        return setup(namespace, freeze=False)

    def _ensure_model(self, cfg):
        if self.model is not None:
            self.model.cfg = cfg
            return
        import torch
        from lightning.pytorch import Trainer
        from partfield.model_trainer_pvcnn_only_demo import Model
        # Lightning's generic loader eagerly materializes a second 1.2 GB copy
        # of this trusted, locally installed checkpoint.  Mapping its tensor
        # storage and loading the same state dict cuts both cold-start I/O and
        # peak host memory; PartField's Model has no checkpoint-load hook.
        checkpoint = torch.load(
            self.checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
        self.model = Model(cfg)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        del checkpoint
        self.model.predict_step = _without_diagnostic_ply(self.model.predict_step)
        self.model.eval()
        # A preload is only useful once the weights are actually resident.  If
        # Lightning performs this placement on the first prediction, the socket
        # advertises a warm worker while that call still copies a large model.
        self.model.cuda()
        self.trainer = Trainer(
            devices=1, accelerator="gpu", precision="16-mixed", strategy="auto",
            logger=False, enable_checkpointing=False, enable_progress_bar=False,
        )
        torch.cuda.empty_cache()

    def _loader(self, mesh):
        """PartField's inference batch without an OBJ write/read round trip."""
        import torch
        import trimesh

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        scale = 1.8 / (upper - lower).max()
        normalized = (vertices - 0.5 * (lower + upper)) * scale
        normalized_mesh = trimesh.Trimesh(
            vertices=normalized, faces=faces, process=False
        )
        points, _ = trimesh.sample.sample_surface(normalized_mesh, 100_000)
        item = {
            "uid": "canonical",
            "pc": torch.as_tensor(points, dtype=torch.float32),
            "vertices": torch.as_tensor(normalized),
            "faces": torch.as_tensor(faces),
        }
        return torch.utils.data.DataLoader(
            [item], batch_size=1, shuffle=False, pin_memory=True
        )

    def preload(self):
        """Load weights before publishing a warm-worker socket."""
        self._ensure_model(self._config(self.root))

    def _infer(self, job: Path, mesh) -> Path:
        cfg = self._config(self.root)
        self._ensure_model(cfg)
        self.model.cfg = cfg
        loader = self._loader(mesh)
        previous = Path.cwd()
        try:
            os.chdir(job)
            self.trainer.predict(
                self.model, dataloaders=loader, ckpt_path=None
            )
        finally:
            os.chdir(previous)
        feature_dir = job / "exp_results" / "features"
        candidates = sorted(feature_dir.glob("part_feat_canonical_0*.npy"))
        if not candidates:
            raise RuntimeError("PartField did not produce face features")
        return candidates[-1]

    def features(self, surface: Path, cache_dir: Path):
        import trimesh

        cache_dir.mkdir(parents=True, exist_ok=True)
        features_path = cache_dir / "features.npy"
        manifest_path = cache_dir / "manifest.json"
        key = {
            "surface_sha256": self._surface_sha256(surface),
            "source_revision": self._revision(),
            "checkpoint": str(self.checkpoint),
            "checkpoint_size": self.checkpoint.stat().st_size,
            "checkpoint_mtime_ns": self.checkpoint.stat().st_mtime_ns,
            "points_per_face": self.points_per_face,
        }
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == key:
                return features_path, self._surface_mesh(surface)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

        mesh = self._surface_mesh(surface)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"not a nonempty triangle surface: {surface}")
        job = cache_dir / "job"
        if job.exists():
            shutil.rmtree(job)
        job.mkdir(parents=True)
        generated_features = self._infer(job, mesh)
        features = np.load(generated_features, mmap_mode="r")
        if features.shape[0] != len(mesh.faces):
            raise ValueError("PartField did not return one feature vector per face")
        del features
        os.replace(generated_features, features_path)
        (cache_dir / "microclusters.npz").unlink(missing_ok=True)
        manifest_path.write_text(json.dumps(key, indent=2, sort_keys=True), encoding="utf-8")
        shutil.rmtree(job, ignore_errors=True)
        return features_path, mesh

    def _revision(self) -> str:
        if self._source_revision is None:
            self._source_revision = _revision(self.root)
        return self._source_revision

    def segment(self, *, surface: Path, output_npz: Path, output_json: Path,
                output_glb: Path, clusters: int, feature_cache: Path,
                cluster_interpreter: Path | None, cluster_environment: str | None):
        features_path, mesh = self.features(surface, feature_cache)
        cluster_script = Path(__file__).with_name("gpu_cluster.py")
        from src.partfield_wrapper.gpu_cluster import (
            gpu_agglomerative_available, run as run_clustering,
        )
        if gpu_agglomerative_available():
            return run_clustering(argparse.Namespace(
                surface=surface, features=features_path,
                output_npz=output_npz, output_json=output_json,
                output_glb=output_glb, clusters=clusters,
                microcluster_cache=feature_cache / "microclusters.npz",
                backend="gpu_agglomerative",
            ), mesh=mesh)

        arguments = [
            str(cluster_script), "--surface", str(surface),
            "--features", str(features_path),
            "--output-npz", str(output_npz),
            "--output-json", str(output_json),
            "--output-glb", str(output_glb),
            "--clusters", str(clusters),
            "--backend", "gpu_kmeans",
        ]
        if cluster_interpreter is not None:
            command = [str(cluster_interpreter), *arguments]
        elif cluster_environment:
            command = ["conda", "run", "-n", cluster_environment, "python", *arguments]
        else:
            command = [sys.executable, *arguments]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic was emitted"
            raise RuntimeError(f"PartField GPU clustering failed: {detail}")
        if result.stdout:
            print(result.stdout, end="")
        return [str(path.resolve()) for path in (output_npz, output_json, output_glb)]


def parser():
    default_env, default_root, default_checkpoint = settings()
    result = argparse.ArgumentParser()
    result.add_argument("--surface", type=Path, required=True)
    result.add_argument("--output-npz", type=Path, required=True)
    result.add_argument("--output-json", type=Path, required=True)
    result.add_argument("--output-glb", type=Path, required=True)
    result.add_argument("--feature-cache", type=Path, required=True)
    result.add_argument("--clusters", type=int, default=5)
    result.add_argument("--partfield-root", type=Path, default=default_root)
    result.add_argument("--checkpoint", type=Path, default=default_checkpoint)
    result.add_argument("--points-per-face", type=int, default=500)
    result.add_argument("--cluster-interpreter", type=Path)
    result.add_argument("--cluster-environment")
    return result


def run(args, runtime: PartFieldRuntime | None = None):
    runtime = runtime or PartFieldRuntime(
        args.partfield_root, args.checkpoint, args.points_per_face
    )
    return runtime.segment(
        surface=args.surface.resolve(), output_npz=args.output_npz.resolve(),
        output_json=args.output_json.resolve(), output_glb=args.output_glb.resolve(),
        clusters=args.clusters, feature_cache=args.feature_cache.resolve(),
        cluster_interpreter=args.cluster_interpreter,
        cluster_environment=args.cluster_environment,
    )


def main():
    args = parser().parse_args()
    print(json.dumps({"artifacts": run(args)}, separators=(",", ":")))


if __name__ == "__main__":
    main()
