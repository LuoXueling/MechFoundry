#!/usr/bin/env python
"""Re-run a session's case with another assembled linear-solver backend.

The workflow runs ``cg_backend_assembled="amgx"`` by default (settings
``linear_solver: "jaxamg"``), which uses JAX-AMG and raises when a linear solve
ends above its tolerance; ``linear_solver: "pyamgx"`` runs pyamgx in the graph.  On a genuinely
ill-conditioned system -- members only a few leaves thick -- AMG reaches only
||b - Kx||/||b|| of about 1e-2 to 1e-3 in its iterations, and the solve fails
at Newton iteration 0.  pyamgx runs the same AMG but keeps its last iterate
whatever AMGX reports, and Warp's native CG stops on its own test; either way
Newton's residual check is what validates the result (inexact Newton); see
references/known_issues.md #1 for the measured agreement.  Accept a run only if
Newton converged, and cross-check the two backends.

This runs the workflow's own ``run_forward`` on the session's
``case/case.json`` with the backend swapped, writing a normal run folder into
the session's ``solve/``.  Nothing else about the case changes.

Run under the compute environment (``mechfoundry-warp117``); ``drive.py
solve-backend`` does that, on an idle GPU, and is the way to call it.

    backend_solve.py CASE_JSON {pyamgx,native,jaxamg,amgx} [--jaxamg-max-iters N]

``--jaxamg-max-iters`` raises JAX-AMG's Krylov cap (1000) for a solve that "was
still converging and ran out of iterations": the same backend, more iterations.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[4]
for path in (REPO_ROOT, REPO_ROOT / "src" / "warp_wrapper",
             REPO_ROOT / "src" / "agents_wrapper" / "fem_workflow" / "scripts"):
    sys.path.insert(0, str(path))

BACKENDS = ("pyamgx", "native", "jaxamg", "amgx")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("backend", choices=BACKENDS)
    parser.add_argument("--jaxamg-max-iters", type=int, default=None)
    args = parser.parse_args(argv)

    import core.shape_opt as shape_opt

    base = shape_opt.ShapeOptTrimmed
    backend = args.backend

    extra = {"jaxamg_max_iters": args.jaxamg_max_iters} if args.jaxamg_max_iters else {}

    class WithBackend(base):
        def __init__(self, config, override_params=None, **kwargs):
            override_params = dict(override_params or {}, cg_backend_assembled=backend, **extra)
            super().__init__(config, override_params=override_params, **kwargs)

    # run_forward imports ShapeOptTrimmed from the module when it runs.
    shape_opt.ShapeOptTrimmed = WithBackend
    import workflow_tool

    workflow_tool.run_forward(argparse.Namespace(case=args.case.resolve()))
    print(f"backend_solve: finished with cg_backend_assembled={backend}", flush=True)


if __name__ == "__main__":
    main()
