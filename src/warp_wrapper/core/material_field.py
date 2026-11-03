"""Fast regular-grid material transfer for adaptive octree FEM."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class SampledMaterial:
    values: torch.Tensor
    flat_indices: torch.Tensor
    bounds: torch.Tensor
    manifest: dict

    @property
    def youngs_modulus(self):
        return self.values[:, 0]

    @property
    def poisson_ratio(self):
        return self.values[:, 1]

    @property
    def density(self):
        return self.values[:, 2]


def sample_material_artifact(
    artifact: str | Path,
    points: torch.Tensor,
    *,
    non_blocking: bool = True,
) -> SampledMaterial:
    """Gather all material channels with one O(N) CUDA index operation.

    The artifact is intentionally a contiguous channels-last ``.npy`` rather
    than compressed NPZ: it can be memory-mapped and copied to the device once,
    while all node lookup and channel gathering stays on the GPU.
    """
    root = Path(artifact).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    # 2 moved the grid half a cell so VoMP's voxels sit at cell centres; the
    # bounds carry that, so both read the same way here.
    if manifest.get("schema_version") not in (1, 2):
        raise ValueError(f"unsupported material artifact schema: {manifest.get('schema_version')}")
    host = np.load(root / "material_grid.npy", mmap_mode="r", allow_pickle=False)
    if host.shape != (64, 64, 64, 3) or host.dtype != np.float32:
        raise ValueError(f"invalid material grid: {host.shape} {host.dtype}")
    device = points.device
    # A writable contiguous staging allocation avoids PyTorch's warning for a
    # read-only mmap and is the sole host copy in the solver process.
    staging = torch.from_numpy(np.array(host, copy=True, order="C"))
    if points.is_cuda and torch.cuda.is_available():
        staging = staging.pin_memory()
    grid = staging.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
    bounds = torch.as_tensor(manifest["bounds"], dtype=points.dtype, device=device)
    span = (bounds[1] - bounds[0]).clamp_min(torch.finfo(points.dtype).eps)
    indices = torch.floor((points - bounds[0]) / span * 64.0).long().clamp_(0, 63)
    flat_indices = (
        indices[:, 0] * (64 * 64) + indices[:, 1] * 64 + indices[:, 2]
    )
    values = grid.view(-1, 3).index_select(0, flat_indices).contiguous()
    # Reduce all validation predicates before the single device-to-host sync.
    invalid = torch.stack((
        (~torch.isfinite(values)).any(),
        ((values[:, 0] <= 0) | (values[:, 2] <= 0)).any(),
        ((values[:, 1] <= -1) | (values[:, 1] >= 0.5)).any(),
    )).cpu().tolist()
    if invalid[0]:
        raise ValueError("material artifact contains non-finite values")
    if invalid[1]:
        raise ValueError("material modulus and density must be positive")
    if invalid[2]:
        raise ValueError("material Poisson ratio must lie strictly between -1 and 0.5")
    return SampledMaterial(values, flat_indices, bounds, manifest)
