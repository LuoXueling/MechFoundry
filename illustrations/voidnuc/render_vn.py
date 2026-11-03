"""Materialize void-nucleation panels from the canonical hidden-phi run.

The void-nucleation figure is a focused view of the same captured iteration as
``hidden_phi_loop``. It must therefore not launch a second optimization (the
old implementation did, and silently compared two different GPU executions).
``build.sh hiddenphi`` renders the canonical panels; this adapter gives the
focused figure stable local names and records exactly which hidden-phi artifact
each panel uses.

Run with the normal Python environment, not pvbatch::

    python render_vn.py

No numerical field is recomputed and no image is resampled here.
"""

from pathlib import Path
import shutil


HERE = Path(__file__).resolve().parent
HP = HERE.parent / "hiddenphi"


def _stats(path):
    out = {}
    for line in path.read_text().splitlines():
        key, *values = line.split()
        if key:
            out[key] = values
    return out


stats = _stats(HP / "hp_stats.txt")
evo_iters = [int(x) for x in stats["evo_iters"]]
final_iter = int(stats["iter_final"][0])

# The six field panels are exact views of hp_pre.vtk. In particular, (c) uses
# the post-step fields hidden_c/sdf_c, while (a) uses hidden/sdf from the start
# of the same nucleating iteration.
sources = {
    "vn_panel_a_phi.png": "hp_panel_a_hid.png",
    "vn_panel_a_d.png": "hp_panel_a_d.png",
    "vn_panel_b_phi.png": "hp_panel_c_phi.png",
    "vn_panel_b_d.png": "hp_panel_c_d.png",
    "vn_panel_c_phi.png": "hp_panel_d_hid.png",
    "vn_panel_c_d.png": "hp_panel_d_d.png",
    "vn_panel_d.png": "hp_panel_e.png",
    "vn_panel_e.png": f"hp_panel_evo_{final_iter}.png",
}
sources.update({f"vn_panel_e{it}.png": f"hp_panel_evo_{it}.png"
                for it in evo_iters})

missing = [str(HP / src) for src in sources.values() if not (HP / src).is_file()]
if missing:
    raise FileNotFoundError(
        "hidden-phi panels are missing; run `./build.sh hiddenphi` first:\n  "
        + "\n  ".join(missing)
    )

for dst, src in sources.items():
    shutil.copyfile(HP / src, HERE / dst)
    print(f"wrote {dst} <- hiddenphi/{src}")

print(f"VOID PANELS DONE ({len(sources)} aliases from one hidden-phi capture)")
