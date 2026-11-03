"""Palette gate for the convergence plot's three series (any env, no deps but numpy).

A Python port of the dataviz skill's ``scripts/validate_palette.js`` — same
thresholds, same Machado-Oliveira-Fernandes (2009) severity-1.0 matrices, same
OKLab dE x100 — because there is no ``node`` on this box and the rule is to
COMPUTE the CVD separation, never to eyeball it.  Run it whenever a series
colour changes:

    conda run -n mechfoundry-warp117 python nc_palette_check.py

Checks, on ALL pairs (three lines that cross each other are never merely
"adjacent"): OKLCH lightness band, chroma floor, CVD separation under
protan/deutan (tritan reported), the normal-vision floor, and WCAG contrast
against the white page.
"""

import sys

import numpy as np

# role "baseline" = the reference scheme's curve, drawn as a dark neutral on
# purpose: it is what the other two are measured against, not a third
# competitor.  The lightness band and the chroma floor are the rules that keep
# equal-salience categorical SLOTS equal, so they do not apply to it (they
# would forbid any neutral at any lightness); every separation gate still does.
PALETTE = [("Body-fitted FEM", "#3b4250", "baseline"),
           ("Nitsche, aligned", "#4c72b0", "series"),
           ("Nitsche, rotated", "#c8622a", "series")]
SURFACE = "#ffffff"        # the figure saves on opaque white

BAND = (0.43, 0.77)        # OKLCH L, light mode
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 8.0, 6.0
NORMAL_FLOOR = 15.0
CONTRAST_MIN = 3.0

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def lin(hexs):
    c = np.array([int(hexs.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(rgb_lin):
    m = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(m @ rgb_lin)
    n = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                  [1.9779984951, -2.4285922050, 0.4505937099],
                  [0.0259040371, 0.7827717662, -0.8086757660]])
    return n @ lms


def sim(rgb_lin, kind):
    return np.clip(np.array(MACHADO[kind]) @ rgb_lin, 0.0, 1.0)


def dE(h1, h2, kind=None):
    a = oklab(sim(lin(h1), kind) if kind else lin(h1))
    b = oklab(sim(lin(h2), kind) if kind else lin(h2))
    return 100 * float(np.linalg.norm(a - b))


def rel_lum(h):
    return float(np.array([0.2126, 0.7152, 0.0722]) @ lin(h))


def contrast(a, b):
    hi, lo = sorted((rel_lum(a), rel_lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


fail = []
print(f"{'slot':<20s} {'hex':<9s} {'L':>6s} {'C':>6s} {'contrast':>9s}")
for name, hx, role in PALETTE:
    L, a, b = oklab(lin(hx))
    C = float(np.hypot(a, b))
    ct = contrast(hx, SURFACE)
    print(f"{name:<20s} {hx:<9s} {L:6.3f} {C:6.3f} {ct:9.2f}")
    if role == "series":
        if not BAND[0] <= L <= BAND[1]:
            fail.append(f"{name}: L {L:.3f} outside {BAND}")
        if C < CHROMA_FLOOR:
            fail.append(f"{name}: C {C:.3f} < chroma floor {CHROMA_FLOOR}")
    else:
        print(f"    (by design) {name} is the neutral baseline: L/chroma gates "
              f"do not apply, separation gates still do")
    if ct < CONTRAST_MIN:
        # The reference validator reports this as a WARN, not a FAIL: it is
        # legal only with mandatory secondary encoding (here: marker shapes and
        # the direct legend).  Thin plot lines have little ink, so take it as a
        # reason to darken the hue rather than to lean on the exemption.
        print(f"    (warn) {name}: contrast {ct:.2f} < {CONTRAST_MIN} — needs "
              f"the marker shapes and legend to carry identity")

print(f"\n{'pair':<40s} {'normal':>7s} {'protan':>7s} {'deutan':>7s} {'tritan':>7s}")
for i in range(len(PALETTE)):
    for j in range(i + 1, len(PALETTE)):
        (n1, h1, _), (n2, h2, _) = PALETTE[i], PALETTE[j]
        d = {k: dE(h1, h2, k) for k in ("protan", "deutan", "tritan")}
        dn = dE(h1, h2)
        print(f"{n1 + ' / ' + n2:<40s} {dn:7.1f} {d['protan']:7.1f} "
              f"{d['deutan']:7.1f} {d['tritan']:7.1f}")
        worst = min(d["protan"], d["deutan"])
        if worst < CVD_FLOOR:
            fail.append(f"{n1}/{n2}: CVD dE {worst:.1f} < floor {CVD_FLOOR}")
        elif worst < CVD_TARGET:
            print(f"    (warn) {n1}/{n2}: CVD dE {worst:.1f} below target "
                  f"{CVD_TARGET} — needs the marker shapes to carry identity")
        if dn < NORMAL_FLOOR:
            fail.append(f"{n1}/{n2}: normal-vision dE {dn:.1f} < {NORMAL_FLOOR}")

print()
for f in fail:
    print("FAIL:", f)
print("PASS" if not fail else "FAILED")
sys.exit(1 if fail else 0)
