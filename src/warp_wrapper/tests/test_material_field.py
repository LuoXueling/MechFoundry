import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.material_field import sample_material_artifact


def _artifact(tmp_path):
    root = tmp_path / "material"
    root.mkdir()
    x, y, z = np.meshgrid(
        np.arange(64, dtype=np.float32), np.arange(64, dtype=np.float32),
        np.arange(64, dtype=np.float32), indexing="ij",
    )
    grid = np.stack((x + 1.0, 0.2 + y / 640.0, z + 1000.0), axis=-1)
    np.save(root / "material_grid.npy", grid.astype(np.float32), allow_pickle=False)
    np.save(root / "valid_mask.npy", np.ones((64, 64, 64), dtype=bool), allow_pickle=False)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "bounds": [[-1, -1, -1], [1, 1, 1]],
    }))
    return root


def test_batched_nearest_lookup_has_exact_boundary_semantics(tmp_path):
    artifact = _artifact(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = torch.tensor([
        [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0],
    ], dtype=torch.float32, device=device)
    sampled = sample_material_artifact(artifact, points)
    assert sampled.flat_indices.cpu().tolist() == [0, 64**3 - 1, 32 * 64**2 + 32 * 64 + 32]
    np.testing.assert_allclose(sampled.values.cpu().numpy(), [
        [1.0, 0.2, 1000.0], [64.0, 0.2 + 63 / 640.0, 1063.0],
        [33.0, 0.2 + 32 / 640.0, 1032.0],
    ], rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA performance gate")
def test_one_million_node_lookup_is_linear_and_fast(tmp_path):
    artifact = _artifact(tmp_path)
    points = torch.rand((1_000_000, 3), device="cuda") * 2.0 - 1.0
    sample_material_artifact(artifact, points)
    torch.cuda.synchronize()
    start = time.perf_counter()
    sampled = sample_material_artifact(artifact, points)
    torch.cuda.synchronize()
    assert time.perf_counter() - start < 0.250
    assert sampled.values.shape == (1_000_000, 3)
