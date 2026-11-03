"""Compose the standalone hidden-phase-field loop figure (conda `mechfoundry-warp117`).

    python hp_figure.py                     # -> hidden_phi_loop.{png,pdf}

The merged figure's design (../_shared/loop_layout.py) with the hidden phase field
phibar SOLID -- it is part of this method: panel a's row 2 is dJ/du -> (adjoint +
auto-diff) -> the dJ/dphi dot, from which BOTH dJ/dd (auto-diff of the tanh map) and
dJ/dphibar (dJ/dphi at phi = 1 on the interior nodes) come -> one step station (Step on
phibar | Step on d) centred under Geometry + BC.  Renders are the hidden-phi run's
(render_hp.py's l_* slots, iteration 43; Geometry + BC = the restored surface with its
BC regions, hp_full_*), except dJ/du: the run stored no displacement vector, so the
station shows the shape run's dJ/du (../shapeopt/panel_so_m_gu.png) -- stated in
legend_notes.md.  Panel b: this run's design evolution.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "..", "shapeopt")
sys.path.insert(0, os.path.join(HERE, "..", "_shared"))
import loop_chain as L  # noqa: E402
import loop_layout as LL  # noqa: E402


def _p(name):
    return os.path.join(HERE, name)


def figure():
    st = LL.stats_file(_p("hp_stats.txt"))
    cap = int(st["iter_capture"])
    keys = ("d", "phi", "u", "g", "gphi", "step", "stepphi")
    slots = ("a_d", "phi", "b", "g_d", "g_phi", "d_d", "d_hid")
    R = dict(zip(keys, LL.crop_common([_p(f"hp_panel_l_{s}.png") for s in slots])))
    R["geom"] = LL.crop_common([_p("hp_panel_l_geom.png")])[0]
    R["gu"] = LL.crop_common([os.path.join(SO, "panel_so_m_gu.png")])[0]
    its = [20, int(st["iter_pf_on"]), cap, 60, 80, 90]
    evo = [LL.lift(a) for a in L.common_crop(
        [_p("hp_panel_l_evo_cap.png" if i == cap else f"hp_panel_l_evo_{i}.png")
         for i in its], pad=14)]
    R["init"] = LL.lift(L.common_crop([_p("hp_panel_l_evo_0.png")], pad=14)[0])
    return LL.LoopFigure(R, [("Concurrent topology and shape optimization", evo, its, cap,
                              LL.timing(_p("hp_fr_stats.txt")))],
                         phibar="solid", phi_symbol=r"$\varphi$, " + LL.PHIBAR)


if __name__ == "__main__":
    figure().build("hidden_phi_loop")
