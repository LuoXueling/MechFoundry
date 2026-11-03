#!/usr/bin/env python3
"""Compare final-iter SE / Vol / Total / Time across runs vs a reference.

Usage:
    python tests/compare_runs.py <ref_dir> <variant_dir> [<variant_dir> ...]

Each argument should be an output root (e.g. ``../../output/baseline_uniform``)
that contains a single timestamped sub-directory with a ``log.txt`` file.
Alternatively, pass the path directly to a ``log.txt`` file.
"""

import os
import re
import sys
from datetime import datetime


def parse_log(path):
    """Return (SE, Vol, Total, elapsed_seconds) from the last iter line."""
    if not path.endswith("log.txt"):
        # Find the most-recently-modified sub-directory log.
        candidates = [
            os.path.join(path, d, "log.txt")
            for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
            and os.path.isfile(os.path.join(path, d, "log.txt"))
        ]
        if not candidates:
            raise FileNotFoundError(f"No log.txt found under {path}")
        path = max(candidates, key=os.path.getmtime)

    with open(path) as f:
        lines = f.readlines()

    se = vol = total = None
    for line in reversed(lines):
        m = re.search(r"SE=([\d.]+).*?Vol=([\d.]+).*?Total=([\S]+)", line)
        if m:
            se, vol, total = float(m.group(1)), float(m.group(2)), float(m.group(3))
            break

    ts_re = re.compile(r"\[(\d{2}-\d{2}-\d{4},\d{2}:\d{2}:\d{2})\.\d+\]")
    times = [
        datetime.strptime(m.group(1), "%m-%d-%Y,%H:%M:%S")
        for line in lines
        for m in [ts_re.match(line)]
        if m
    ]
    elapsed = (times[-1] - times[0]).total_seconds() if len(times) >= 2 else None
    return se, vol, total, elapsed


def pct(base, val):
    return f"{abs(val - base) / max(abs(base), 1e-12) * 100:.2f}%"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    ref_path = sys.argv[1]
    ref_se, ref_vol, ref_total, ref_t = parse_log(ref_path)
    ref_label = os.path.basename(ref_path.rstrip("/"))

    hdr = f"{'Run':<30} {'SE':>8} {'Vol':>8} {'Total':>12} {'Time':>8}  ΔSE%   ΔVol%  ΔTotal%"
    print(hdr)
    print("-" * len(hdr))
    tstr = f"{int(ref_t)}s" if ref_t else "?"
    print(f"{ref_label + ' (ref)':<30} {ref_se:>8.4f} {ref_vol:>8.4f} {ref_total:>12.6f} {tstr:>8}")

    for path in sys.argv[2:]:
        label = os.path.basename(path.rstrip("/"))
        try:
            se, vol, total, t = parse_log(path)
            tstr = f"{int(t)}s" if t else "?"
            print(
                f"{label:<30} {se:>8.4f} {vol:>8.4f} {total:>12.6f} {tstr:>8}"
                f"  {pct(ref_se, se):<7} {pct(ref_vol, vol):<7} {pct(ref_total, total)}"
            )
        except Exception as e:
            print(f"{label:<30}  ERROR: {e}")


if __name__ == "__main__":
    main()
