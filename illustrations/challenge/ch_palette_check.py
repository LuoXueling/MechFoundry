"""Gate (CPU): colour-vision check on the challenge figure's identity colours.

Reuses ``overview/ov_palette_check.py``'s port of the dataviz validator
(CIEDE2000 under normal vision and Machado-2009 protan / deutan / tritan).
The figure adds ONE identity colour to method_overview's palette: FAIL, the red
of the agent row's failure marks.  It must stay >= 6 CVD from every colour it
can sit beside (the Neumann orange, the Dirichlet teal, the agent violet, the
backward magenta, the ink and the human grey).  It is never the only carrier:
each failure also has its "!" mark or its text, so a near pair is reported, the
hard floor enforced.
"""

import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overview"))
import ov_palette_check as P  # noqa: E402
import ov_style as S  # noqa: E402

FAIL = "#c0392b"
COLOURS = {"fail": FAIL, "Neumann": S.PALETTE["Neumann (ears)"],
           "Dirichlet": S.PALETTE["Dirichlet (base)"], "agent": S.PALETTE["agent"],
           "backward": S.PALETTE["backward"], "ink": S.PALETTE["ink"], "human": "#4a515b"}

bad = []
for (n1, h1), (n2, h2) in itertools.combinations(COLOURS.items(), 2):
    if "fail" not in (n1, n2):
        continue                   # the other pairs are method_overview's, gated there
    lab1, lab2 = P.rgb2lab(P.hex2rgb(h1)), P.rgb2lab(P.hex2rgb(h2))
    normal = P.de2000(lab1, lab2)
    cvd = min(P.de2000(P.rgb2lab(P.simulate(P.hex2rgb(h1), k)),
                       P.rgb2lab(P.simulate(P.hex2rgb(h2), k))) for k in P.M)
    print(f"{n1:>10} / {n2:<10} normal {normal:5.1f}  CVD min {cvd:5.1f}")
    if cvd < 6.0:
        bad.append((n1, n2, cvd))
if bad:
    sys.exit(f"CVD below 6: {bad}")
print("palette ok")
