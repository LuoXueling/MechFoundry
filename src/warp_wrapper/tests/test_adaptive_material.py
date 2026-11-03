"""End-to-end coverage for VoMP-style fields in diffused Nitsche."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tests.test_nitsche import _make_cube_octree_mesh, _make_sdf_for_inner_cube


def _write_material(root: Path) -> Path:
    root.mkdir()
    x = np.linspace(1.0e4, 2.0e4, 64, dtype=np.float32)[:, None, None]
    grid = np.empty((64, 64, 64, 3), dtype=np.float32)
    grid[..., 0] = x
    grid[..., 1] = 0.30
    grid[..., 2] = 1000.0 + 100.0 * (x / 1.0e4 - 1.0)
    np.save(root / "material_grid.npy", grid, allow_pickle=False)
    np.save(root / "valid_mask.npy", np.ones(grid.shape[:3], dtype=bool), allow_pickle=False)
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "bounds": [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
        "shape": [64, 64, 64, 3],
        "channels": ["youngs_modulus", "poisson_ratio", "density"],
    }), encoding="utf-8")
    return root


@pytest.mark.parametrize("mixed_precision", [False, True])
def test_adaptive_material_compiles_and_converges_with_density_load(
        tmp_path, mixed_precision):
    """Compile every spatial stress/tangent/body kernel in a real solve."""
    nodes, elements = _make_cube_octree_mesh(n=4)
    sdf = _make_sdf_for_inner_cube(nodes, truncate=0.5)
    artifact = _write_material(tmp_path / "material")

    from core.nitsche_field import NitscheNeoHookean

    solver = NitscheNeoHookean(
        "configs.config_nitsche_example",
        override_params={
            "output_root": str(tmp_path / "solve"),
            "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
            "phase_field_epsilon": 0.2,
            "material_artifact": str(artifact),
            "material_acceleration": (0.0, -1.0e-3, 0.0),
            "B": (0.0, 0.0, 0.0),
            "dirichlet_bcs_dict": {
                lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
            },
            "neumann_bcs_dict": {},
            "use_mixed_precision": mixed_precision,
            # The tiny synthetic load reaches the normal float32 assembly
            # floor after one Newton update; this test gates compilation and
            # field plumbing rather than nonlinear tolerance behavior.
            "relative_tol": 5.0e-3,
            "absolute_tol": 1.0e-3,
            "max_iter": 30,
            "linear_solver": "cg",
            "cg_backend_assembled": "amgx",
            "ew_enabled": False,
            "line_search_enabled": False,
            "load_steps": 1,
        },
    )
    sampled = solver._material_values_torch.cpu().numpy()
    assert sampled[:, 0].min() < sampled[:, 0].max()
    assert sampled[:, 2].min() < sampled[:, 2].max()
    solver.forward()
    assert np.isfinite(solver.final_residual)
    assert solver.final_residual < 1.0e-3
    solver.save(time=0)
    solver.finalize()
    volume_files = list((tmp_path / "solve").rglob("*.vtu"))
    output_text = next(
        path.read_text(encoding="utf-8") for path in volume_files
        if 'Name="youngs_modulus"' in path.read_text(encoding="utf-8")
    )
    for field in ("youngs_modulus", "poisson_ratio", "density", "mu", "kappa"):
        assert f'Name="{field}"' in output_text
