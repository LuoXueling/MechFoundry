"""Solve the reference bunny with surface-selected Nitsche boundary data."""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, os.fspath(HERE))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from core.nitsche_field import NitscheNeoHookean  # noqa: E402
from core.remeshing import octree_hex_interpolate  # noqa: E402
from core.vtu_writer import VTUSurfaceWriter  # noqa: E402
from problem_setup import build_problem  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traction-z", type=float, default=-0.020)
    parser.add_argument("--output", type=Path, default=HERE / "bunny_surface.vtu")
    args = parser.parse_args()

    setup = build_problem()
    bunny = setup["bunny"]
    octree = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in setup["octree"].items()
    }
    octree["domain_length"] = setup["domain_length"]

    scratch = HERE / f"_{args.output.stem}"
    shutil.rmtree(scratch, ignore_errors=True)
    params = {
        "project_name": "bunny_nitsche",
        "output_root": os.fspath(scratch),
        "octree_mesh": octree,
        "phase_field_epsilon": setup["epsilon"],
        "relative_tol": 1e-3,
        "use_mixed_precision": False,
        "tangent_f64": False,
        "cg_backend_assembled": "jaxamg",
        "async_io": True,
        "trim_exterior": True,
        # Omitting geometric_func selects the standard 2x2x2 Gauss rule.
        "dirichlet_bcs_dict": {setup["base_selector"]: 0.0},
        "neumann_bcs_dict": {
            setup["ear_selector"]: (0.0, 0.0, args.traction_z)
        },
        "output_per_step": 0,
    }
    solver = NitscheNeoHookean("config", override_params=params)
    start = time.perf_counter()
    solver.forward()
    solve_seconds = time.perf_counter() - start

    solved_mesh = solver.params.octree_mesh
    nodes = torch.as_tensor(
        solved_mesh["nodes"], dtype=torch.float64, device="cuda"
    )
    elements = torch.as_tensor(
        solved_mesh["elements"], dtype=torch.long, device="cuda"
    )
    displacement = torch.as_tensor(
        solver.get_displacement(), dtype=torch.float64, device="cuda"
    )
    surface_nodes = torch.as_tensor(
        np.asarray(bunny.vertices), dtype=torch.float64, device="cuda"
    )
    surface_displacement = octree_hex_interpolate(
        nodes, elements, displacement, surface_nodes
    ).cpu().numpy()

    writer = VTUSurfaceWriter(
        os.fspath(HERE),
        grid_shape=None,
        write_vtu=True,
        write_glb=False,
        write_glb_deformed=False,
        async_io=False,
    )
    writer.save_from_mesh(
        0,
        np.asarray(bunny.vertices),
        np.asarray(bunny.faces),
        {
            "displacement": surface_displacement,
            "displacement_magnitude": np.linalg.norm(
                surface_displacement, axis=1
            ),
        },
    )
    generated = HERE / "vtu_surface" / "surface_t0000.vtu"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(generated, args.output)
    shutil.rmtree(HERE / "vtu_surface", ignore_errors=True)
    shutil.rmtree(scratch, ignore_errors=True)

    maximum = float(np.linalg.norm(surface_displacement, axis=1).max())
    print(
        f"wrote {args.output}: umax={maximum:.4e}, "
        f"Newton={solver.newton_iterations}, solve={solve_seconds:.2f} s"
    )


if __name__ == "__main__":
    main()
