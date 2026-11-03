"""Check an evaluator's eval_<k>.json against its packet and the rubric.

    python3 validate.py FILE INDEX_JSON

Reports: entry count, missing / duplicate / unknown (label, category, object) keys, missing
fields, invalid ratings or criterion scores, and ratings that do not follow mechanically from
the criteria (High: all met; Moderate: none failed and one or two partial; Low: any failed or
three or more partial).  Exit status 0 only if the file is complete and valid; a rating that
disagrees with its criteria is reported but not fatal (the Fable round was not held to it)."""
import json, sys
from collections import Counter

CRIT = {"geometry": ("G1", "G2", "G3", "G4"), "bc": ("B1", "B2", "B3", "B4"),
        "material": ("M1", "M2", "M3")}
RATINGS = {"High", "Moderate", "Low", "n/a"}
SCORES = {"met", "partial", "failed", "n/a"}
FIELDS = ("label", "category", "object", "geometry", "bc", "material", "criteria",
          "geometry_note", "bc_note", "material_note")


def derive(scores):
    if all(s == "n/a" for s in scores):
        return "n/a"
    if "failed" in scores:
        return "Low"
    p = scores.count("partial")
    return "High" if p == 0 else "Moderate" if p <= 2 else "Low"


def check(path, index_path):
    idx = json.load(open(index_path))
    want = {(e["label"], e["category"], e["object"]) for e in idx}
    try:
        ev = json.load(open(path))
    except (OSError, ValueError) as err:
        return False, [f"unreadable: {err}"]
    keys = [(e.get("label"), e.get("category"), e.get("object")) for e in ev]
    msgs = [f"{len(ev)} entries"]
    dup = [k for k, n in Counter(keys).items() if n > 1]
    missing, unknown = want - set(keys), set(keys) - want
    if dup: msgs.append(f"duplicates: {dup}")
    if missing: msgs.append(f"missing {len(missing)}: {sorted(missing)[:5]}...")
    if unknown: msgs.append(f"unknown keys: {sorted(unknown)[:5]}")
    bad, mismatch = [], []
    for e in ev:
        k = (e.get("label"), e.get("category"), e.get("object"))
        miss = [f for f in FIELDS if f not in e]
        if miss:
            bad.append((k, "missing " + ",".join(miss)))
            continue
        for cat, ids in CRIT.items():
            if e[cat] not in RATINGS:
                bad.append((k, f"{cat}={e[cat]!r}"))
            sc = [e["criteria"].get(i) for i in ids]
            if e[cat] == "n/a" and all(s in (None, "n/a") for s in sc):
                continue                  # a setting never reached: criteria may be left out
            if any(s not in SCORES for s in sc):
                bad.append((k, f"{cat} criteria {dict(zip(ids, sc))}"))
            elif derive(sc) != e[cat]:
                mismatch.append((k, cat, e[cat], derive(sc)))
            if not str(e[f"{cat}_note"]).strip() and e[cat] not in ("High", "n/a"):
                bad.append((k, f"empty {cat}_note"))
    if bad: msgs.append(f"invalid {len(bad)}: {bad[:5]}")
    msgs.append(f"rating != derived from criteria: {len(mismatch)} {mismatch[:5]}")
    ok = len(ev) == len(want) and not (dup or missing or unknown or bad)
    return ok, msgs


if __name__ == "__main__":
    ok, msgs = check(sys.argv[1], sys.argv[2])
    print(("VALID" if ok else "INVALID") + ": " + "; ".join(msgs))
    sys.exit(0 if ok else 1)
