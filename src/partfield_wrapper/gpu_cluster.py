#!/usr/bin/env python
"""GPU clustering adapters for cached PartField face features."""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(REPO_ROOT))
sys.path.insert(0, os.fspath(REPO_ROOT / "src" / "warp_wrapper"))

DEFAULT_MICROCLUSTERS = 512
MICROCLUSTER_CACHE_VERSION = 3


def gpu_agglomerative_available() -> bool:
    """Return whether the optional RAPIDS backend is installed in this Python."""
    return (importlib.util.find_spec("cuml") is not None
            and importlib.util.find_spec("cupy") is not None)


def _load_packaged_nvrtc() -> None:
    """Expose Torch's pip-packaged NVRTC to CuPy 13 without shell activation."""
    import torch

    site_packages = Path(torch.__file__).resolve().parent.parent
    directory = site_packages / "nvidia" / "cuda_nvrtc" / "lib"
    builtins = sorted(directory.glob("libnvrtc-builtins.so.*"))
    runtime = directory / "libnvrtc.so.12"
    if builtins and runtime.is_file():
        ctypes.CDLL(os.fspath(builtins[-1]), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(os.fspath(runtime), mode=ctypes.RTLD_GLOBAL)


def _validated_features(path: Path, face_count: int) -> np.ndarray:
    # The channel count is the PartField checkpoint's (448 for the upstream
    # model at 373025d, per model_trainer_pvcnn_only_demo.py); nothing here
    # assumes it, and the microcluster cache records it.
    # Shape validation and cached hierarchy cuts must not page a typical 269 MB
    # feature matrix into RAM.  The cold microcluster path makes its one writable
    # host copy immediately before the H2D transfer.
    features = np.load(path, mmap_mode="r")
    if features.ndim != 2 or features.shape[0] != face_count:
        raise ValueError(
            f"PartField features have shape {features.shape}, expected "
            f"({face_count}, channels)"
        )
    return features


def _microcluster_cache(path: Path | None, features: np.ndarray, count: int):
    if path is None or not path.is_file():
        return None
    try:
        with np.load(path) as saved:
            labels = saved["labels"]
            cuts = saved["cuts"]
            channels = int(saved["feature_channels"])
            faces = int(saved["face_count"])
            max_clusters = int(saved["max_clusters"])
            version = int(saved["version"])
        if (labels.shape == (features.shape[0],)
                and cuts.shape == (max_clusters - 1, count)
                and channels == features.shape[1]
                and faces == features.shape[0]
                and version == MICROCLUSTER_CACHE_VERSION):
            return (labels.astype(np.int32, copy=False),
                    cuts.astype(np.int16, copy=False), max_clusters)
    except (OSError, ValueError, KeyError):
        pass
    return None


def _cuml_microclusters(features: np.ndarray, count: int):
    """Quantize face embeddings on GPU before hierarchical merging."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PartField feature clustering requires CUDA")
    _load_packaged_nvrtc()
    import cupy as cp
    from cuml.cluster import KMeans

    # The mmap is read-only, but this tensor is only the source of an H2D copy.
    # Sharing it avoids a full-size intermediate host allocation and memcpy.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The given NumPy array is not writable"
        )
        host_features = torch.from_numpy(features)
    values = host_features.to(device="cuda", dtype=torch.float32)
    norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    if bool((~torch.isfinite(values)).any()) or bool((norms <= 1.0e-12).any()):
        raise ValueError("PartField returned non-finite or zero-length face features")
    values = values / norms
    cuda_values = cp.from_dlpack(values)
    model = KMeans(
        n_clusters=count, max_iter=20, n_init=1,
        random_state=1340818006, output_type="cupy",
    )
    labels = torch.utils.dlpack.from_dlpack(model.fit_predict(cuda_values))
    # Ward's merge cost is exact only for leaf centres that are the means of
    # their members; cuML's centres are one Lloyd step stale after max_iter.
    labels = labels.long()
    sums = torch.zeros((count, values.shape[1]), dtype=values.dtype,
                       device=values.device).index_add_(0, labels, values)
    sizes = torch.bincount(labels, minlength=count).to(values.dtype)
    centers = sums / sizes.clamp_min(1.0)[:, None]
    return labels, centers


def _weighted_gpu_ward_cuts(centers, micro_labels, max_clusters: int):
    """Build reusable 2..max_clusters weighted-Ward cuts on the CUDA device."""
    import torch

    device_centers = torch.as_tensor(
        centers, dtype=torch.float32, device="cuda"
    ).clone()
    leaf_labels = torch.as_tensor(
        micro_labels, dtype=torch.long, device="cuda"
    )
    count = len(device_centers)
    weights = torch.bincount(leaf_labels, minlength=count).to(torch.float32)
    # A leaf no face belongs to never enters the hierarchy: it would merge at
    # zero cost and leave a 0/0 centroid.  Its cut label stays -1.
    occupied = weights > 0
    active = occupied.clone()
    membership = torch.arange(count, dtype=torch.long, device="cuda")
    infinity = torch.tensor(float("inf"), device="cuda")

    column = weights[:, None]
    costs = (column * column.T / (column + column.T).clamp_min(1.0)
             * torch.cdist(device_centers, device_centers).square())
    costs.fill_diagonal_(float("inf"))
    costs[~occupied, :] = infinity
    costs[:, ~occupied] = infinity
    cuts = torch.full(
        (max_clusters - 1, count), -1, dtype=torch.int16, device="cuda"
    )

    def cut():
        roots = torch.unique(membership[occupied])
        lookup = torch.full((count,), -1, dtype=torch.long, device="cuda")
        lookup[roots] = torch.arange(len(roots), device="cuda")
        labels = lookup[membership].to(torch.int16)
        labels[~occupied] = -1
        return labels

    active_count = int(occupied.sum())
    if 2 <= active_count <= max_clusters:
        cuts[active_count - 2] = cut()
    for _ in range(active_count - 2):
        pair = int(torch.argmin(costs))
        left, right = divmod(pair, count)
        total = weights[left] + weights[right]
        device_centers[left] = (
            weights[left] * device_centers[left]
            + weights[right] * device_centers[right]
        ) / total
        weights[left] = total
        active[right] = False
        membership = torch.where(membership == right, left, membership)

        delta2 = (device_centers - device_centers[left]).square().sum(1)
        row = (weights[left] * weights
               / (weights[left] + weights).clamp_min(1.0) * delta2)
        row[~active] = infinity
        row[left] = infinity
        costs[left, :] = row
        costs[:, left] = row
        costs[right, :] = infinity
        costs[:, right] = infinity
        active_count -= 1
        if active_count <= max_clusters:
            cuts[active_count - 2] = cut()
    return cuts


def _weighted_gpu_ward(centers: np.ndarray, micro_labels: np.ndarray,
                       clusters: int) -> np.ndarray:
    """Compatibility helper returning one weighted-Ward cut."""
    cuts = _weighted_gpu_ward_cuts(centers, micro_labels, int(clusters))
    leaf_labels = np.asarray(micro_labels, dtype=np.int64)
    return cuts[int(clusters) - 2].cpu().numpy()[leaf_labels]


def gpu_agglomerative(features: np.ndarray, clusters: int, *,
                      cache_path: Path | None = None,
                      microclusters: int = DEFAULT_MICROCLUSTERS):
    """Cluster PartField features with a fast, GPU-native hierarchy.

    cuML GPU K-means first quantizes the faces into microclusters.  A weighted
    Ward hierarchy then merges those groups on the same CUDA device.  With the
    member means as leaf centres, each merge is the one upstream PartField's
    face-level Ward would make between those groups; the approximation is the
    quantization itself, since faces sharing a microcluster are never split.
    """
    clusters = int(clusters)
    microclusters = min(int(microclusters), int(features.shape[0]))
    if not 2 <= clusters <= microclusters:
        raise ValueError("cluster count must be between 2 and the microcluster count")

    max_clusters = min(microclusters, max(20, clusters))
    cached = _microcluster_cache(cache_path, features, microclusters)
    if cached is None:
        micro_labels, centers = _cuml_microclusters(features, microclusters)
        cuts = _weighted_gpu_ward_cuts(centers, micro_labels, max_clusters)
        if isinstance(cuts, np.ndarray):
            host_cuts = cuts.astype(np.int16, copy=False)
            host_micro_labels = np.asarray(
                micro_labels, dtype=np.int32
            )
            labels = host_cuts[clusters - 2].astype(
                np.int64, copy=False
            )[host_micro_labels]
        else:
            labels = cuts[clusters - 2].long()[micro_labels.long()].cpu().numpy()
            host_micro_labels = micro_labels.cpu().numpy().astype(
                np.int32, copy=False
            )
            host_cuts = cuts.cpu().numpy()
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path, labels=host_micro_labels, cuts=host_cuts,
                version=np.int64(MICROCLUSTER_CACHE_VERSION),
                max_clusters=np.int64(max_clusters),
                face_count=np.int64(features.shape[0]),
                feature_channels=np.int64(features.shape[1]),
            )
    else:
        micro_labels, cuts, cached_max_clusters = cached
        if clusters > cached_max_clusters:
            return gpu_agglomerative(
                features, clusters, cache_path=None, microclusters=microclusters
            )
        labels = cuts[clusters - 2].astype(np.int64, copy=False)[micro_labels]
    if np.unique(labels).size != clusters:
        raise RuntimeError("GPU Ward clustering returned an incomplete cut")
    return labels, {
        "clustering": "gpu_weighted_ward",
        "linkage": "ward",
        "preclustering": "cuml_gpu_kmeans",
        "microclusters": microclusters,
        "feature_channels": int(features.shape[1]),
    }


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--surface", type=Path, required=True)
    result.add_argument("--features", type=Path, required=True)
    result.add_argument("--output-npz", type=Path, required=True)
    result.add_argument("--output-json", type=Path, required=True)
    result.add_argument("--output-glb", type=Path, required=True)
    result.add_argument("--clusters", type=int, required=True)
    result.add_argument("--microcluster-cache", type=Path)
    result.add_argument(
        "--backend", choices=("auto", "gpu_agglomerative", "gpu_kmeans"),
        default="auto",
    )
    return result


def run(args, *, mesh=None):
    from src.agents_wrapper.fem_workflow.segmentation_artifacts import write_segmentation

    if mesh is None:
        import trimesh
        mesh = trimesh.load(args.surface, force="mesh", process=False)
    features = _validated_features(args.features, len(mesh.faces))
    backend = getattr(args, "backend", "auto")
    if backend != "gpu_kmeans" and gpu_agglomerative_available():
        labels, method_settings = gpu_agglomerative(
            features, int(args.clusters),
            cache_path=getattr(args, "microcluster_cache", None),
        )
    elif backend == "gpu_agglomerative":
        raise RuntimeError(
            "GPU agglomerative clustering requires cuml-cu12 and cupy-cuda12x"
        )
    else:
        import torch
        from core.bc_surface_segmentation import gpu_kmeans

        if not torch.cuda.is_available():
            raise RuntimeError("PartField feature clustering requires CUDA")
        values = torch.tensor(
            np.asarray(features), dtype=torch.float32, device="cuda"
        )
        norms = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        if bool((norms <= 1.0e-12).any()):
            raise ValueError("PartField returned a zero-length face feature")
        labels, _, _ = gpu_kmeans(values / norms, int(args.clusters))
        labels = labels.cpu().numpy()
        method_settings = {
            "clustering": "gpu_kmeans",
            "feature_channels": int(features.shape[1]),
        }
    return write_segmentation(
        mesh, labels, method="partfield", num_clusters=args.clusters,
        output_npz=args.output_npz, output_json=args.output_json,
        output_glb=args.output_glb, method_settings=method_settings,
    )


def main():
    artifacts = run(parser().parse_args())
    print(json.dumps({"artifacts": artifacts}, separators=(",", ":")))


if __name__ == "__main__":
    main()
