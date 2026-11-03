"""Make self-contained copies of the Fable round's packets for the Astra evaluators.

    python3 copy_packets.py [--restore DIR] [--dest DIR]

The packets of the Fable round (output/agentic_object_record/model_eval/packet_<k>, built by
../build_packets.py) hold the figures and *symlinks* into the run folders.  By the Astra round
the Haiku and Sonnet run folders had been archived (output/archives/agentic_batch50{,_haiku}
.tar.zst), so those links dangle.  This script rebuilds each packet outside the repo with the same
labels, figures, replies and session files, resolving every link either in the repo (Opus, still
live) or in DIR (the subset of the Haiku/Sonnet archives extracted there: agent_reply.md and the
workflow session folder of the 150 runs the packets use).

Files are hard links (same filesystem, no extra space) and real files, not symlinks, so the
packet no longer reveals the run folder's name (which contains the batch, hence the model).
Bulky solver/history data no evaluator is pointed at is left out, for all three models alike:
versions/, solve/api_fem_forward-*/ (VTU), solve/*.glb and *.disp.bin, the PartField feature
volume and checkpoints.sqlite.  index.json is rewritten to the new paths, otherwise unchanged."""
import argparse, json, os, re, shutil
from pathlib import Path

REPO = Path("/home/xl2254/implicit_optimize")
SRC = REPO / "output/agentic_object_record/model_eval"
SKIP = re.compile(r"(^|/)(versions|partfield_features|api_fem_forward-[^/]*)(/|$)"
                  r"|(^|/)checkpoints\.sqlite$|\.disp\.bin$|^solve/[^/]*\.glb$")


def resolve(link, restore):
    """The live path of a packet symlink: in the repo, or under the restore dir."""
    t = os.readlink(link)
    if os.path.exists(t):
        return Path(t)
    alt = Path(restore) / t.split(str(REPO) + "/", 1)[1]
    if alt.exists():
        return alt
    raise FileNotFoundError(f"{link} -> {t}: neither live nor restored")


def link_tree(src, dst):
    n = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        rel = "" if rel == "." else rel
        dirs[:] = [d for d in dirs if not SKIP.search(f"{rel}/{d}".lstrip("/"))]
        for f in files:
            r = f"{rel}/{f}".lstrip("/")
            if SKIP.search(r):
                continue
            (dst / rel).mkdir(parents=True, exist_ok=True)
            os.link(Path(root) / f, dst / r)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", default="/home/xl2254/astra_eval_work/restore")
    ap.add_argument("--dest", default="/home/xl2254/astra_eval_work")
    a = ap.parse_args()
    for k in (1, 2, 3):
        src, dst = SRC / f"packet_{k}", Path(a.dest) / f"packet_{k}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src / "figures", dst / "figures")
        index, nfiles = [], 0
        for e in json.load(open(src / "index.json")):
            e = dict(e)
            d = dst / "runs" / e["label"] / e["category"]
            d.mkdir(parents=True, exist_ok=True)
            reply = d / f"{e['row']}_reply.md"
            rsrc = Path(e["reply"])
            if rsrc.is_symlink():
                os.link(resolve(rsrc, a.restore), reply)
            else:                                    # the "(no final reply)" placeholder
                shutil.copy(rsrc, reply)
            e["reply"] = str(reply)
            e["figure"] = str(dst / "figures" / Path(e["figure"]).name)
            if e["session"]:
                s = d / f"{e['row']}_session"
                nfiles += link_tree(resolve(Path(e["session"]), a.restore), s)
                e["session"] = str(s)
            index.append(e)
        (dst / "index.json").write_text(json.dumps(index, indent=1))
        print(f"{dst}: {len(index)} simulations, {nfiles} session files")


if __name__ == "__main__":
    main()
