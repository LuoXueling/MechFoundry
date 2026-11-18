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
volume and checkpoints.sqlite.  index.json is rewritten to the new paths, otherwise unchanged.

This is the paper round's one-off conversion of symlink packets.  A new round builds its packets
self-contained from the start (build_packets.py with "packet_mode": "copy", same exclusions, same
link_tree), and both evaluator families read those."""
import argparse, json, os, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from build_packets import REPO, SKIP, link_tree, place          # noqa: E402,F401  (SKIP: the exclusions)
from round_config import Round, main_checkout                    # noqa: E402

PAPER = Round("paper")
SRC = PAPER.packets
# The packet links were made when the repo lived at the paper round's prompt_repo
# (/home/xl2254/implicit_optimize); it has since moved, so a link target is looked up under
# any of the repo paths (the old one, this checkout, its main checkout), then under the restore dir.
REPO_PATHS = sorted({str(PAPER.prompt_repo), str(REPO), str(main_checkout())}, key=len, reverse=True)


def resolve(link, restore):
    """The live path of a packet symlink: in the repo, or under the restore dir."""
    t = os.readlink(link)
    if os.path.exists(t):
        return Path(t)
    for prefix in REPO_PATHS:
        if t.startswith(prefix + "/"):
            rel = t[len(prefix) + 1:]
            for alt in (REPO / rel, main_checkout() / rel, Path(restore) / rel):
                if alt.exists():
                    return alt
    raise FileNotFoundError(f"{link} -> {t}: neither live nor restored")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restore", default="/home/xl2254/astra_eval_work/restore")
    ap.add_argument("--dest", default="/home/xl2254/astra_eval_work")
    ap.add_argument("--src", default=str(SRC), help="the folder of the Fable packets")
    a = ap.parse_args()
    for k in (1, 2, 3):
        src, dst = Path(a.src) / f"packet_{k}", Path(a.dest) / f"packet_{k}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src / "figures", dst / "figures")
        index, nfiles = [], 0
        for e in json.load(open(src / "index.json")):
            e = dict(e)
            d = dst / "runs" / e["label"] / e["category"]
            d.mkdir(parents=True, exist_ok=True)
            reply = d / f"{e['row']}_reply.md"
            # index.json names the packet by the repo path it was built at; read the packet
            # where it is now (SRC), whichever path the index carries.
            here = src / "runs" / e["label"] / e["category"]
            rsrc = here / f"{e['row']}_reply.md"
            if rsrc.is_symlink():
                place(resolve(rsrc, a.restore), reply)
            else:                                    # the "(no final reply)" placeholder
                shutil.copy(rsrc, reply)
            e["reply"] = str(reply)
            e["figure"] = str(dst / "figures" / Path(e["figure"]).name)
            if e["session"]:
                s = d / f"{e['row']}_session"
                nfiles += link_tree(resolve(here / f"{e['row']}_session", a.restore), s)
                e["session"] = str(s)
            index.append(e)
        (dst / "index.json").write_text(json.dumps(index, indent=1))
        print(f"{dst}: {len(index)} simulations, {nfiles} session files")


if __name__ == "__main__":
    main()
