"""Gate (CPU): colour-vision check on the overview's own colours.

Run FIRST by ``build.sh overview``; a non-zero exit stops the build.  Port of
``dataviz/scripts/validate_palette.js`` — all-pairs CIEDE2000 under normal
vision and under simulated protanopia / deuteranopia / tritanopia (Machado,
Oliveira & Fernandes 2009, severity 1.0).  Thresholds: normal >= 15, CVD >= 6
hard / 8 target.

Three tiers, because not every pair of colours has to be told apart:

  IDENTITY   every colour that names something (``ov_style.PALETTE``): ink,
             text, frame, page, the Neumann / Dirichlet sets, the agent violet,
             the backward magenta.  All pairs, hard.
  SEGMENTS   the pastel segment tints against the two BC colours only, hard:
             the tints sit one station before the BC regions, and a tint read
             as "orange" there would name the wrong set.  Between themselves the
             tints are allowed to be close — the ink seams carry segment
             identity, not the fills.
  FIELD ENDS the far ends of the d, phi and |dJ/dd| ramps against the identity
             colours, reported as warnings: a field is read by its own ramp.

History: the first overview's two loop legs (#b5651d / #2f7d3a, inherited from
hidden_phi_loop) failed at CVD 4.9; the |dJ/dd| ramp stops at #c51b7d because
the next PiYG step, #8e0152, sits CVD 4.6 from the d ramp's brick.
"""

import itertools
import sys

import numpy as np

M = {  # Machado 2009, severity 1.0
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

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import ov_style as S  # noqa: E402

PALETTE = dict(S.PALETTE)
SEGMENTS = {f"segment tint {i}": c for i, c in enumerate(S.SEGMENT_TINTS)}
BC = {k: v for k, v in PALETTE.items() if k.startswith(("Neumann", "Dirichlet"))}


def _hex(rgb):
    return "#" + "".join(f"{int(round(255 * c)):02x}" for c in rgb)


FIELD_ENDS = {
    "d inside end": _hex(S.SDF_RGB[1:4]), "d outside end": _hex(S.SDF_RGB[-3:]),
    "phi end": _hex(S.PHI_RGB[-3:]), "|dJ/dd| end": _hex(S.GRAD_RGB[-3:]),
    "E soft end": _hex(S.E_SOFT), "E stiff end": _hex(S.E_STIFF),   # the "steel" map
}
NORMAL_FLOOR, CVD_FLOOR, CVD_TARGET = 15.0, 6.0, 8.0


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def srgb2lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    return np.where(c <= 0.0031308, c * 12.92,
                    1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def rgb2lab(rgb):
    lin = srgb2lin(rgb)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = m @ lin / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def simulate(rgb, kind):
    return np.clip(lin2srgb(np.array(M[kind]) @ srgb2lin(rgb)), 0, 1)


def de2000(l1, l2):
    L1, a1, b1 = l1
    L2, a2, b2 = l2
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cb = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cb ** 7 / (Cb ** 7 + 25 ** 7))) if Cb > 0 else 0.5
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    else:
        dhp = h2p - h1p - 360 * np.sign(h2p - h1p)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    if C1p * C2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7)) if Cbp > 0 else 0
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                   + Rt * (dCp / Sc) * (dHp / Sh))


def pair(n1, h1, n2, h2):
    r1, r2 = hex2rgb(h1), hex2rgb(h2)
    dn = de2000(rgb2lab(r1), rgb2lab(r2))
    dc = min(de2000(rgb2lab(simulate(r1, k)), rgb2lab(simulate(r2, k))) for k in M)
    return dn, dc


def segment_tints(labels, face_adjacency, tints):
    """One tint per PartField part, touching parts the most different pair.

    Every assignment of ``tints`` to the parts is tried (8! = 40 320 for the
    overview's k = 8) and the one whose closest TOUCHING pair is farthest apart
    under normal vision wins; parts that do not touch may share a look-alike.
    Shared by ``ov_surfaces.py`` and ``pfbunny/pf_prep.py``, so the overview's
    segmentation panel and ``partfield_segmentation`` colour the same run alike.
    Returns ({label: tint index}, the touching pairs, the winning min dE).
    """
    ids = [int(v) for v in np.unique(labels)]
    assert len(ids) <= len(tints), f"{len(ids)} parts, {len(tints)} tints"
    la, lb = labels[face_adjacency[:, 0]], labels[face_adjacency[:, 1]]
    cut = la != lb
    touch = {tuple(sorted(p)) for p in zip(la[cut].tolist(), lb[cut].tolist())}
    dE = {(i, j): pair("a", tints[i], "b", tints[j])[0]
          for i in range(len(tints)) for j in range(len(tints)) if i != j}
    best = max(itertools.permutations(range(len(tints)), len(ids)),
               key=lambda perm: min(dE[(perm[ids.index(a)], perm[ids.index(b)])]
                                    for a, b in touch))
    colour = {v: best[i] for i, v in enumerate(ids)}
    return colour, sorted(touch), min(dE[(colour[a], colour[b])] for a, b in touch)


def check_e_map():
    """The E map's sRGB stops must lie on its stated CIELAB line (ov_style.E_LAB),
    in order, lightness falling from soft to stiff (hard)."""
    a, b = np.array(S.E_LAB[0]), np.array(S.E_LAB[1])
    n = len(S.E_RGB)
    worst = max(de2000(rgb2lab(np.array(c)), a + (b - a) * k / (n - 1))
                for k, c in enumerate(S.E_RGB))
    L = [rgb2lab(np.array(c))[0] for c in S.E_RGB]
    assert worst < 0.5, f"E map stops stray {worst:.2f} DeltaE00 from E_LAB"
    assert all(y < x for x, y in zip(L, L[1:])), "E map lightness not monotone"
    print(f"E MAP (hard): stops on the Lab line within {worst:.2f} DeltaE00, "
          f"L* {L[0]:.1f} -> {L[-1]:.1f} monotone")


def main():
    failures, warnings = [], []
    check_e_map()
    print("IDENTITY (hard)")
    for (n1, h1), (n2, h2) in itertools.combinations(PALETTE.items(), 2):
        dn, dc = pair(n1, h1, n2, h2)
        bad = dc < CVD_FLOOR or dn < NORMAL_FLOOR
        (failures if bad else []).append((n1, n2, dn, dc))
        flag = "  <-- FAIL" if bad else ("  <- watch" if dc < CVD_TARGET else "")
        print(f"  {n1:>18} / {n2:<18} normal {dn:5.1f}   cvd {dc:5.1f}{flag}")
    print("SEGMENT TINTS vs the BC colours (hard)")
    for n1, h1 in SEGMENTS.items():
        for n2, h2 in BC.items():
            dn, dc = pair(n1, h1, n2, h2)
            bad = dc < CVD_FLOOR or dn < NORMAL_FLOOR
            (failures if bad else []).append((n1, n2, dn, dc))
            print(f"  {n1:>18} / {n2:<18} normal {dn:5.1f}   cvd {dc:5.1f}"
                  + ("  <-- FAIL" if bad else ""))
    print("FIELD ENDS vs identity (warn)")
    for n1, h1 in FIELD_ENDS.items():
        for n2, h2 in PALETTE.items():
            dn, dc = pair(n1, h1, n2, h2)
            if dc < CVD_FLOOR:
                warnings.append((n1, n2, dn, dc))
                print(f"  {n1:>18} / {n2:<18} normal {dn:5.1f}   cvd {dc:5.1f}  <- warn")
    for n1, n2, dn, dc in warnings:
        print(f"WARN {n1} / {n2}: cvd {dc:.1f}")
    if failures:
        for n1, n2, dn, dc in failures:
            print(f"FAIL {n1} / {n2}: normal {dn:.1f}, cvd {dc:.1f}")
        sys.exit(1)
    print("palette OK")


if __name__ == "__main__":
    main()
