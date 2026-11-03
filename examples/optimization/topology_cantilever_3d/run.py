"""Run the self-contained 3-D cantilever topology-optimization example."""

import argparse
import os
import sys
from pathlib import Path

LEAF_DIR = Path(__file__).resolve().parent
REPO_ROOT = LEAF_DIR.parents[2]
sys.path.insert(0, os.fspath(LEAF_DIR))
sys.path.insert(0, os.fspath(REPO_ROOT / "src/warp_wrapper"))

from core.topo_opt import TopoOpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-iters", type=int, default=1000)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--nz", type=int, default=None)
    parser.add_argument("--output", type=Path, default=LEAF_DIR / "output")
    args = parser.parse_args()
    overrides = {"max_optimize_iters": args.n_iters, "output_root": str(args.output)}
    for key in ("nx", "ny", "nz"):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value
    TopoOpt("config", override_params=overrides).optimize()


if __name__ == "__main__":
    main()
