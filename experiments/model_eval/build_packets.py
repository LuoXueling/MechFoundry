"""Build one blinded packet per evaluator for the three-model evaluation (Haiku, Sonnet, Opus
on the 50 objects of experiments/agentic_batch/objects_50.txt).

    python3 build_packets.py [--evaluators 3]

Each evaluator k gets its own random model -> label (A/B/C) mapping, kept in
mappings/mapping_<k>.json beside this script, and a packet under PACKETS (output/, git-ignored:
copies of figures and links into run folders) packet_<k>/ with:
  figures/<category>_<label>.png   copies of illustrations/agentic_results_<category><suffix>.png
  runs/<label>/<category>/<row>_reply.md, <row>_session -> run/workflow/<sid>   (symlinks)
  index.json                       one entry per simulation
The run of each instruction is the latest one across the model's batches (as the figures use)."""
import argparse, csv, json, random, re, shutil, sys
from pathlib import Path

REPO = Path("/home/xl2254/implicit_optimize")
HERE = Path(__file__).resolve().parent
LIST = REPO / "experiments/agentic_batch/objects_50.txt"
PACKETS = REPO / "output/agentic_object_record/model_eval"   # the packets the evaluators read
MAPPINGS = HERE / "mappings"
MODELS = {"sonnet": ("output/agentic_batch50", ""),
          "haiku": ("output/agentic_batch50_haiku", "_haiku"),
          "opus": ("output/agentic_batch50_opus", "_opus")}


def categories():
    cats, cur = [], None
    for line in LIST.read_text().splitlines():
        s = line.strip()
        if s.startswith("# ") and len(s) < 40 and not s.startswith("# Demonstration"):
            cur = (re.sub(r"[^a-z]+", "_", s[2:].lower()).strip("_"), [])
            cats.append(cur)
        elif s and not s.startswith("#") and cur:
            cur[1].append(s)
    assert len(cats) == 10 and all(len(o) == 5 for _, o in cats), cats
    return cats


def latest_runs(root, base=REPO):
    """instruction -> (run dir, status), the latest across the root's batches.  base is the
    folder the batch roots (and summary.csv's repo-relative paths) hang from: the repo, or a
    folder holding an extract of an archived batch."""
    out = {}
    for b in sorted((base / root / "_batches").iterdir(), key=lambda p: p.name):
        if (b / "summary.csv").exists():
            for row in csv.DictReader(open(b / "summary.csv")):
                p = base / row["path"]
                if p.exists():
                    out[row["instruction"]] = (p, row.get("status", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluators", type=int, default=3)
    a = ap.parse_args()
    cats = categories()
    runs = {m: latest_runs(root) for m, (root, _) in MODELS.items()}
    for m in MODELS:
        missing = [o for _, objs in cats for o in objs if o not in runs[m]]
        if missing:
            sys.exit(f"{m}: no run for {missing}")
        for slug, _ in cats:
            fig = REPO / f"illustrations/agentic_results_{slug}{MODELS[m][1]}.png"
            if not fig.exists():
                sys.exit(f"missing figure {fig}")
    MAPPINGS.mkdir(exist_ok=True)
    for k in range(1, a.evaluators + 1):
        labels = list("ABC")
        random.Random(1000 + k).shuffle(labels)
        mapping = dict(zip(MODELS, labels))
        (MAPPINGS / f"mapping_{k}.json").write_text(json.dumps(mapping, indent=1))
        pk = PACKETS / f"packet_{k}"
        if pk.exists():
            shutil.rmtree(pk)
        (pk / "figures").mkdir(parents=True)
        index = []
        for m, lab in sorted(mapping.items(), key=lambda kv: kv[1]):
            for slug, objs in cats:
                fig = pk / f"figures/{slug}_{lab}.png"
                shutil.copy(REPO / f"illustrations/agentic_results_{slug}{MODELS[m][1]}.png", fig)
                d = pk / "runs" / lab / slug
                d.mkdir(parents=True)
                for i, o in enumerate(objs, 1):
                    run, status = runs[m][o]
                    reply = d / f"{i}_reply.md"
                    src = run / "agent_reply.md"
                    if src.exists():
                        reply.symlink_to(src)
                    else:
                        reply.write_text("(the agent left no final reply)\n")
                    sid = None
                    try:
                        sid = json.load(open(run / "metrics.json")).get("session_id")
                    except (OSError, ValueError):
                        pass
                    sess = run / "workflow" / sid if sid else None
                    link = None
                    if sess and sess.exists():
                        link = d / f"{i}_session"
                        link.symlink_to(sess, target_is_directory=True)
                    index.append({"label": lab, "category": slug, "object": o,
                                  "figure": str(fig), "row": i, "reply": str(reply),
                                  "session": str(link) if link else None, "status": status})
        (pk / "index.json").write_text(json.dumps(index, indent=1))
        print(f"packet_{k}: {len(index)} simulations, mapping in mappings/mapping_{k}.json")


if __name__ == "__main__":
    main()
