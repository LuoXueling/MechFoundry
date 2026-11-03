"""Subprocess helper for comparing JAX-AMG and pyamgx on Diffused Nitsche."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.nitsche_field import NitscheNeoHookean
from tests.test_nitsche import _make_cube_octree_mesh, _make_sdf_for_inner_cube


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jaxamg", "pyamgx"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    nodes, elements = _make_cube_octree_mesh(n=8)
    sdf = _make_sdf_for_inner_cube(nodes, inner_lo=-0.5, inner_hi=0.5)
    run_dir = args.output.parent / f"run-{args.backend}"
    solver = NitscheNeoHookean(
        "configs.config_nitsche_example",
        override_params={
            "output_root": str(run_dir),
            "octree_mesh": {"nodes": nodes, "elements": elements, "sdf": sdf},
            "phase_field_epsilon": 0.02,
            "dirichlet_bcs_dict": {
                lambda x, y, z: torch.isclose(x, torch.full_like(x, -0.5)): 0.0,
            },
            "neumann_bcs_dict": {
                lambda x, y, z: torch.isclose(x, torch.full_like(x, 0.5)): (
                    0.01,
                    0.0,
                    0.0,
                ),
            },
            "use_mixed_precision": False,
            "relative_tol": 1.0e-5,
            "absolute_tol": 1.0e-6,
            "max_iter": 30,
            "linear_solver": "cg",
            "cg_backend_assembled": args.backend,
            "ew_enabled": False,
            "line_search_enabled": False,
            "load_steps": 1,
        },
    )
    solver.forward()
    displacement = solver.u.dof_values.numpy()
    np.savez(
        args.output,
        displacement=displacement,
        residual=np.asarray(solver.final_residual),
        dtype=np.asarray(str(displacement.dtype)),
    )


if __name__ == "__main__":
    main()
