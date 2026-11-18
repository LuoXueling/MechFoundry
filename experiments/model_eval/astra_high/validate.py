"""Check an evaluator's eval_<k>.json against its packet and the rubric.

    python3 validate.py FILE INDEX_JSON
    python3 validate.py --round open_models --family fable [--live]

The second form checks every evaluator of a family: eval_<k>.json from the family's committed
ratings folder (with --live: the output file the evaluator wrote, where its prompt sent it)
against the round's packet_<k>/index.json.  The rubric is the same for every round, so CRIT and
the rating rule below are not per round; the entry count follows from the packet.

Reports: entry count, missing / duplicate / unknown (label, category, object) keys, missing
fields, invalid ratings or criterion scores, and ratings that do not follow mechanically from
the criteria (High: all met; Moderate: none failed and one or two partial; Low: any failed or
three or more partial).  Exit status 0 only if the file is complete and valid; a rating that
disagrees with its criteria is reported but not fatal (the Fable round was not held to it)."""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from round_config import Round                                   # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("index", nargs="?")
    ap.add_argument("--round")
    ap.add_argument("--family")
    ap.add_argument("--live", action="store_true",
                    help="the files the evaluators wrote (output/), not the committed copies")
    a = ap.parse_args()
    if a.file and a.index:
        pairs = [(a.file, a.index)]
    elif a.round:
        rnd = Round(a.round)
        fam = a.family or rnd.default_family()
        pairs = [(rnd.out_file(fam, k) if a.live else rnd.ratings_dir(fam) / f"eval_{k}.json",
                  rnd.packets / f"packet_{k}" / "index.json") for k in range(1, rnd.evaluators + 1)]
    else:
        ap.error("give FILE INDEX_JSON, or --round [--family]")
    all_ok = True
    for f, idx in pairs:
        ok, msgs = check(f, idx)
        all_ok &= ok
        print((f"{f}: " if len(pairs) > 1 else "") + ("VALID" if ok else "INVALID") + ": " + "; ".join(msgs))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
