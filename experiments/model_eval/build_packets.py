"""Build one blinded packet per evaluator for a grading round (by default the paper round:
Haiku, Sonnet, Opus on the 50 objects of experiments/agentic_batch/objects_50.txt).

    python3 build_packets.py [--round paper] [--evaluators 3]
    python3 build_packets.py --round open_models          (then make_prompts.py, see README)

The round (rounds/<name>.json, see round_config.py) names the models, their batch roots and
figure suffixes, the label seed and where mappings and packets go.  Each evaluator k gets its own
random model -> label (A, B, ...) mapping, kept in <mappings>/mapping_<k>.json, and a packet
<packets>/packet_<k>/ (under output/, git-ignored) with:
  figures/<category>_<label>.png   copies of illustrations/agentic_results_<category><suffix>.png
  runs/<label>/<category>/<row>_reply.md, <row>_session/   the agent's reply and its session
  index.json                       one entry per simulation
The run of each instruction is the latest one across the model's batches (as the figures use).

Two packet modes:
  symlink  the paper round's Fable packets: the reply and the session are symlinks into the run
           folders.  Kept so that the paper round rebuilds exactly as it was built.
  copy     every new round: real files (hard links where the batch root is on the same
           filesystem, else copies), so the packet is self-contained, no path in it names a batch
           folder (which names the model), and both evaluator families (Fable in Claude Code,
           Astra in Codex) read the very same packet through a read-only bind.  The bulky solver
           and history data no evaluator is pointed at is left out, as astra_high/copy_packets.py
           did for the Astra round: versions/, solve/api_fem_forward-*/ (VTU), solve/*.glb and
           *.disp.bin, the PartField feature volume and checkpoints.sqlite.
           A copy-mode build ends with leak_check.py over every packet and fails if any text in
           it names an operator model or harness (see leak_check.py for what counts).

A copy-mode round can also carry "packet_edits" (round_config.py): blinding redactions of single
packet files, applied after the files are placed and before the blinding check.  Each edited file
is written as a new file (temp file + os.replace), so its hard link into the batch folder is broken
and the run's own file is never touched.  An edit whose "old" text does not occur exactly once
stops the build, so a redaction cannot silently stop applying when a run is replaced."""
import argparse, csv, json, os, re, shutil, sys, tempfile
from pathlib import Path

from round_config import DEFAULT_ROUND, HERE, REPO, Round   # noqa: F401  (HERE, REPO re-exported)

# Session files no evaluator is pointed at, left out of copied packets (relative to the session).
SKIP = re.compile(r"(^|/)(versions|partfield_features|api_fem_forward-[^/]*)(/|$)"
                  r"|(^|/)checkpoints\.sqlite$|\.disp\.bin$|^solve/[^/]*\.glb$")

# The paper round, as the scripts that predate round configs named it.  agreement.py and older
# notebooks import these; new code goes through Round.
_PAPER = Round(DEFAULT_ROUND)
MAPPINGS = _PAPER.mappings
PACKETS = _PAPER.packets
LIST = _PAPER.objects_list
MODELS = {m: (_PAPER.root(m), _PAPER.model[m].get("suffix", "")) for m in _PAPER.models}


def categories():
    return _PAPER.categories()


def latest_runs(root, base=REPO):
    """instruction -> (run dir, status), the latest across the root's batches.  base is the
    folder the batch roots (and summary.csv's repo-relative paths) hang from: the repo, or a
    folder holding an extract of an archived batch."""
    out = {}
    for b in sorted((base / root / "_batches").iterdir(), key=lambda p: p.name):
        if (b / "summary.csv").exists():
            for row in csv.DictReader(open(b / "summary.csv")):
                p = base / row["path"]
                if not p.exists():
                    # A run folder sits directly in its batch root, so a root given as an
                    # absolute path (a scratch extract, a test round) finds its runs too.  For a
                    # repo-relative root this is the same path again, so nothing changes there.
                    p = base / root / Path(row["path"]).name
                if p.exists():
                    out[row["instruction"]] = (p, row.get("status", ""))
    return out


def place(src, dst):
    """A real file at dst with src's content: a hard link when both sit on one filesystem (no
    extra space; the packet is bound read-only, so the batch file cannot be changed through it),
    else a copy."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def link_tree(src, dst):
    """Copy (hard-link) a session folder into the packet without the SKIP files.  Symlinks inside
    the session are resolved to their content, so the packet holds no link at all."""
    n = 0
    for root, dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        rel = "" if rel == "." else rel
        dirs[:] = [d for d in dirs if not SKIP.search(f"{rel}/{d}".lstrip("/"))]
        for f in files:
            r = f"{rel}/{f}".lstrip("/")
            if SKIP.search(r):
                continue
            s = Path(root) / f
            if s.is_symlink() and not s.exists():
                continue                                  # a dangling link carries nothing
            (dst / rel).mkdir(parents=True, exist_ok=True)
            place(s.resolve(), dst / r)
            n += 1
    return n


def session_dir(run):
    """The workflow session folder of a run (metrics.json names it), or None."""
    sid = None
    try:
        sid = json.load(open(run / "metrics.json")).get("session_id")
    except (OSError, ValueError):
        pass
    sess = run / "workflow" / sid if sid else None
    return sess if sess and sess.exists() else None


def resolve_edits(rnd, runs, cats):
    """The round's packet_edits checked against the runs before anything is built: [(edit, slug,
    row, rel)], rel being the file relative to runs/<label>/<slug>/.  An edit names its model and
    its object (the instruction), not a packet path, because the label and so the path differ for
    every evaluator.  "file" is "reply" or "session/<path in the session folder>".  Each edit's
    "old" must occur exactly once in the run's own file, and that file must be one a packet
    holds (not a SKIP file)."""
    edits = rnd.cfg.get("packet_edits", [])
    if edits and rnd.packet_mode != "copy":
        sys.exit(f"round {rnd.name}: packet_edits need packet_mode copy (a symlink packet's files "
                 f"are the run's own, an edit would change the batch)")
    where = {o: (slug, i) for slug, objs in cats for i, o in enumerate(objs, 1)}
    out = []
    for n, e in enumerate(edits, 1):
        tag = f"packet_edits[{n}] ({e.get('model')}, {e.get('object')!r}, {e.get('file')})"
        missing = [key for key in ("model", "object", "file", "old", "new", "why") if key not in e]
        if missing:
            sys.exit(f"{tag}: no {', '.join(missing)}")
        if e["model"] not in rnd.model:
            sys.exit(f"{tag}: no model {e['model']!r} in round {rnd.name}")
        if e["object"] not in where:
            sys.exit(f"{tag}: {e['object']!r} is not one of the round's objects")
        slug, row = where[e["object"]]
        run = runs[e["model"]][e["object"]][0]
        f = e["file"]
        if f == "reply":
            src, rel = run / "agent_reply.md", f"{row}_reply.md"
        elif f.startswith("session/") and ".." not in Path(f).parts and len(f) > len("session/"):
            inner = f[len("session/"):]
            sess = session_dir(run)
            if sess is None or SKIP.search(inner):
                sys.exit(f"{tag}: no such file in the packet (no session, or a left-out file)")
            src, rel = sess / inner, f"{row}_session/{inner}"
        else:
            sys.exit(f'{tag}: "file" is "reply" or "session/<path in the session folder>"')
        count_once(src, e["old"], tag)
        out.append((e, slug, row, rel))
    return out


def count_once(path, old, tag, consequence="so no packet is built"):
    try:
        n = path.read_bytes().count(old.encode())
    except OSError as err:
        sys.exit(f"{tag}: cannot read {path}: {err}")
    if n != 1:
        sys.exit(f"{tag}: the old text occurs {n} times in {path}, not once; the run changed or "
                 f"the edit is wrong, {consequence}")


def apply_edit(path, old, new, tag):
    """Replace old by new in the packet file at path, writing a NEW file: the placed file is a
    hard link to the batch's own file, and writing through it would change the run."""
    count_once(path, old, tag, "so the build stops here (an earlier edit of this file took it?)")
    data = path.read_bytes().replace(old.encode(), new.encode(), 1)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, path.stat().st_mode & 0o777)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=DEFAULT_ROUND, help="rounds/<name>.json, or a path to one")
    ap.add_argument("--evaluators", type=int, default=None, help="default: the round's")
    ap.add_argument("--no-leak-check", action="store_true",
                    help="skip the blinding check of a copy-mode build (never for a real round)")
    a = ap.parse_args()
    rnd = Round(a.round)
    n_eval = a.evaluators or rnd.evaluators
    cats = rnd.categories()
    runs = {m: latest_runs(rnd.root(m)) for m in rnd.models}
    for m in rnd.models:
        missing = [o for _, objs in cats for o in objs if o not in runs[m]]
        if missing:
            sys.exit(f"{m}: no run for {missing}")
        for slug, _ in cats:
            fig = rnd.figure(slug, m)
            if not fig.exists():
                sys.exit(f"missing figure {fig}")
    edits = resolve_edits(rnd, runs, cats)
    copy = rnd.packet_mode == "copy"
    where = os.path.relpath(rnd.mappings, HERE) if HERE in rnd.mappings.parents else rnd.mappings
    rnd.mappings.mkdir(parents=True, exist_ok=True)
    for k in range(1, n_eval + 1):
        mapping = rnd.mapping(k)
        (rnd.mappings / f"mapping_{k}.json").write_text(json.dumps(mapping, indent=1))
        pk = rnd.packets / f"packet_{k}"
        if pk.exists():
            shutil.rmtree(pk)
        (pk / "figures").mkdir(parents=True)
        index, nfiles = [], 0
        for m, lab in sorted(mapping.items(), key=lambda kv: kv[1]):
            for slug, objs in cats:
                fig = pk / f"figures/{slug}_{lab}.png"
                shutil.copy(rnd.figure(slug, m), fig)
                d = pk / "runs" / lab / slug
                d.mkdir(parents=True)
                for i, o in enumerate(objs, 1):
                    run, status = runs[m][o]
                    reply = d / f"{i}_reply.md"
                    src = run / "agent_reply.md"
                    if src.exists():
                        if copy:
                            place(src.resolve(), reply)
                        else:
                            reply.symlink_to(src)
                    else:
                        reply.write_text("(the agent left no final reply)\n")
                    sess = session_dir(run)
                    link = None
                    if sess:
                        link = d / f"{i}_session"
                        if copy:
                            nfiles += link_tree(sess, link)
                        else:
                            link.symlink_to(sess, target_is_directory=True)
                    index.append({"label": lab, "category": slug, "object": o,
                                  "figure": str(fig), "row": i, "reply": str(reply),
                                  "session": str(link) if link else None, "status": status})
        (pk / "index.json").write_text(json.dumps(index, indent=1))
        for e, slug, row, rel in edits:
            target = pk / "runs" / mapping[e["model"]] / slug / rel
            apply_edit(target, e["old"], e["new"], f"packet_{k}, {e['model']} {e['object']!r}")
        print(f"packet_{k}: {len(index)} simulations"
              + (f", {nfiles} session files" if copy else "")
              + (f", {len(edits)} packet edit(s)" if edits else "")
              + f", mapping in {where}/mapping_{k}.json")
    if copy and not a.no_leak_check:
        from leak_check import check_round
        if not check_round(rnd, n_eval):
            sys.exit("build_packets.py: the blinding check failed (see above); fix the source or "
                     "add a reviewed exception to the round's leak_allow, then rebuild")


if __name__ == "__main__":
    main()
