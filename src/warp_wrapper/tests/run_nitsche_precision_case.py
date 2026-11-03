"""Run the convergence-study Nitsche case in an isolated precision process."""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WRAPPER = ROOT / "src" / "warp_wrapper"
ILLUSTRATION = ROOT / "illustrations" / "nitscheconv"
for path in (str(ROOT), str(WRAPPER), str(ILLUSTRATION)):
    if path not in sys.path:
        sys.path.insert(0, path)

import nitsche_comparison_helper as csn
import nc_prep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("f32", "f64"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mixed = args.precision == "f64"
    # Match the convergence illustration: diffuse volumetric formulation,
    # unrefined quadrature, and epsilon=h/2.  Depth 3 is its coarsest published
    # point and keeps this suitable for the regular unit-test suite.
    csn.SOLVER_CONFIG.update(
        # The convergence illustration tightens this to 1e-10 for its f64
        # timing study.  All-f32 reaches a nonlinear residual floor around
        # 2e-8, so use the production convergence criterion here and judge
        # solution quality separately with the volume error below.
        relative_tol=1.0e-5,
        sharp_volumetric=False,
        async_io=True,
    )
    solver, elapsed, _ = csn.run_nitsche(
        output_dir=str(args.output.parent / args.precision),
        octree_depth=3,
        quad_refinement_depth=0,
        use_mixed_precision=mixed,
        tangent_f64=mixed,
        cg_backend_assembled="jaxamg",
    )

    ref = np.load(ILLUSTRATION / "nc_ref_64.npz")["u"]
    nc_prep.REF_N = 64
    reference = nc_prep.make_interp(ref)(nc_prep.sample_lattice())
    nodes, elements, _phi, displacement = nc_prep.octree_arrays(solver)
    sampled = nc_prep.hex_eval(
        nodes, elements, displacement, nc_prep.sample_lattice()
    )
    error = nc_prep.rel_l2_volume(sampled, reference)
    values = solver.u.dof_values.numpy()
    np.savez(
        args.output,
        displacement=values,
        dtype=np.asarray(str(values.dtype)),
        error=error,
        elapsed=elapsed,
        residual=float(solver.relative_residual_history[-1]),
    )


if __name__ == "__main__":
    main()
