"""Compose the standalone shape-optimization loop figure (conda `mechfoundry-warp117`).

    python so_figure.py                     # -> shape_opt_loop.{png,pdf}

The merged figure's design (../_shared/loop_layout.py, shared with som_figure.py) with
NO hidden phase field: panel a's row 2 is dJ/du -> (adjoint + auto-diff, the dJ/dphi
dot on the lane) -> dJ/dd -> Step on d, the step centred under Geometry + BC; panel b
is the run's design evolution.  Every render is the shape run's (panel_so_m_*,
iteration 40; dJ/du from so_dJdu.py).  Legend text: legend_notes.md.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))
import loop_chain as L  # noqa: E402
import loop_layout as LL  # noqa: E402


def _p(name):
    return os.path.join(HERE, name)


def figure():
    cap = int(LL.stats_file(_p("so_stats.txt"))["capture_iter"])
    cube = [_p(f"panel_so_m_{t}.png") for t in ("a", "phi", "b", "d", "e", "gu")]
    R = dict(zip(("d", "phi", "u", "g", "step", "gu"), LL.crop_common(cube)))
    R["geom"] = LL.crop_common([_p("panel_so_m_g.png")])[0]
    its = [0, 20, cap, 60, 80, 100]
    evo = [LL.lift(a) for a in L.common_crop([_p(f"panel_so_evo_{i}.png") for i in its],
                                             pad=14)]
    R["init"] = evo[0]
    return LL.LoopFigure(R, [("Shape optimization", evo, its, cap,
                              LL.timing(_p("so_fr_stats.txt")))])


if __name__ == "__main__":
    figure().build("shape_opt_loop")
