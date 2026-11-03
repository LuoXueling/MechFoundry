"""Gate (conda ``mechfoundry-warp117``, CPU): colour check on the segment tints.

Run by ``build.sh pfbunny`` before anything is rendered.  It reads
``pf_palette.txt`` -- every PartField part's tint and which parts touch, as
``pf_prep.py`` handed them to the renderer -- so the gate cannot drift from the
figure.

The tints are the overview's pastel set (the user's choice for Figure 1, which
this figure now matches), placed by ``ov_palette_check.segment_tints``.  So the
rule is the overview's, not all-pairs: two parts that TOUCH must differ by
normal-vision CIEDE2000 >= 15 (hard); parts that do not touch may look alike, a
reader never has to tell them apart at a boundary.  Reported, not failed:

  * touching pairs under CVD simulation below 6.  No assignment of these eight
    tints lifts every touching pair over 6 on this segmentation (the best of
    all 8! reaches 2.8), so a failure here would be a veto on the palette
    itself, which is the user's call;
  * a tint against the page below 15: the bunny's shading draws its outline.

All-pairs CIEDE2000 machinery is imported from ``../overview/ov_palette_check.py``
rather than copied, so both gates stay on one implementation.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "overview"))

import ov_palette_check as validator  # noqa: E402

tint, area, touching = {}, {}, []
for line in open(os.path.join(HERE, "pf_palette.txt")):
    words = line.split()
    if words[0] == "touch":
        touching.append((words[1], words[2]))
    else:
        tint[words[0]], area[words[0]] = words[1], float(words[2])

failures = []
print("TOUCHING PARTS (normal hard >= 15; CVD reported)")
for a, b in touching:
    dn, dc = validator.pair(a, tint[a], b, tint[b])
    flag = "  <-- FAIL" if dn < validator.NORMAL_FLOOR else (
        "  <- cvd" if dc < validator.CVD_FLOOR else "")
    if dn < validator.NORMAL_FLOOR:
        failures.append((a, b, dn))
    print(f"  {a} ({area[a]:.1%}) / {b} ({area[b]:.1%})   normal {dn:5.1f}   cvd {dc:5.1f}{flag}")
print("PARTS vs the page (reported)")
for a in tint:
    dn, dc = validator.pair(a, tint[a], "page", "#ffffff")
    if dn < validator.NORMAL_FLOOR:
        print(f"  {a} {tint[a]} / page   normal {dn:5.1f}   cvd {dc:5.1f}  <- shading carries it")
if failures:
    for a, b, dn in failures:
        print(f"FAIL {a} / {b}: normal {dn:.1f}")
    sys.exit(1)
print("palette OK")
