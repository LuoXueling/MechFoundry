import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.neoHookean import NeoHookean


def main():
    solver = NeoHookean("tests.examples.three_d_affine_mix2.config")
    solver.forward()
    solver.save(time=0)
    solver.finalize()


if __name__ == "__main__":
    main()
