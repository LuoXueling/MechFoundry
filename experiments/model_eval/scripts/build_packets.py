"""Build one blinded packet per evaluator for the six-model grading.

    python3 scripts/build_packets.py          (then make_prompts.py, see ../README.md)

The config (../grading.json, see config.py) names the models, their batch roots and figure
suffixes, the label seed and where the packets go.  Each evaluator k gets its own random model ->
label (A-F) mapping, kept in results/mappings/mapping_<k>.json, and a packet
<packets>/packet_<k>/ (under output/, git-ignored) with:
  figures/<category>_<label>.png   copies of the staged category figures
  runs/<label>/<category>/<row>_reply.md, <row>_session/   the agent's reply and its session
  index.json                       one entry per simulation
The run of each instruction is the latest one across the model's batches (as the figures use).

The packet holds real files (hard links where the batch root is on the same filesystem, else
copies), so it is self-contained, no path in it names a batch folder (which names the model), and
both evaluator families (Fable in Claude Code, Astra in Codex) read the very same packet through a
read-only bind.  The bulky solver and history data no evaluator is pointed at is left out, as the
paper's Astra packets did: versions/, solve/api_fem_forward-*/ (VTU), solve/*.glb and *.disp.bin,
the PartField feature volume and checkpoints.sqlite.

The config's "packet_edits" are blinding redactions of single packet files, applied after the
files are placed and before the blinding check.  Each edited file is written as a new file (temp
file + os.replace), so its hard link into the batch folder is broken and the run's own file is
never touched.  An edit whose "old" text does not occur exactly once stops the build, so a
redaction cannot silently stop applying when a run is replaced.  The build ends with
leak_check.py over every packet and fails if any text in it names an operator model or harness
(see leak_check.py for what counts)."""
import argparse, csv, json, os, re, shutil, sys, tempfile
from pathlib import Path

from config import EVAL, REPO, Grading

# Session files no evaluator is pointed at, left out of copied packets (relative to the session).
SKIP = re.compile(r"(^|/)(versions|partfield_features|api_fem_forward-[^/]*)(/|$)"
                  r"|(^|/)checkpoints\.sqlite$|\.disp\.bin$|^solve/[^/]*\.glb$")


def batch_root(root):
    """The batch root in the repo, or a stop naming the archive to extract: the Claude batches
    were archived to output/archives/ after the paper's grading."""
    path = REPO / root
    if not (path / "_batches").exists():
        name = Path(root).name
        sys.exit(f"{path} has no _batches/: extract its archive first,\n"
                 f"  zstd -dc output/archives/{name}.tar.zst | tar -x -C output")
    return path


def latest_runs(root):
    """instruction -> (run dir, status), the latest across the root's batches."""
    out = {}
    path = batch_root(root)
    for b in sorted((path / "_batches").iterdir(), key=lambda p: p.name):
        if (b / "summary.csv").exists():
            for row in csv.DictReader(open(b / "summary.csv")):
                p = REPO / row["path"]
                if not p.exists():
                    # A run folder sits directly in its batch root, so a root given as an
                    # absolute path (a scratch extract) finds its runs too.  For a repo-relative
                    # root this is the same path again, so nothing changes there.
                    p = path / Path(row["path"]).name
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


def resolve_edits(g, runs, cats):
    """The config's packet_edits checked against the runs before anything is built: [(edit, slug,
    row, rel)], rel being the file relative to runs/<label>/<slug>/.  An edit names its model and
    its object (the instruction), not a packet path, because the label and so the path differ for
    every evaluator.  "file" is "reply" or "session/<path in the session folder>".  Each edit's
    "old" must occur exactly once in the run's own file, and that file must be one a packet
    holds (not a SKIP file)."""
    edits = g.cfg.get("packet_edits", [])
    where = {o: (slug, i) for slug, objs in cats for i, o in enumerate(objs, 1)}
    out = []
    for n, e in enumerate(edits, 1):
        tag = f"packet_edits[{n}] ({e.get('model')}, {e.get('object')!r}, {e.get('file')})"
        missing = [key for key in ("model", "object", "file", "old", "new", "why") if key not in e]
        if missing:
            sys.exit(f"{tag}: no {', '.join(missing)}")
        if e["model"] not in g.model:
            sys.exit(f"{tag}: no model {e['model']!r} in the config")
        if e["object"] not in where:
            sys.exit(f"{tag}: {e['object']!r} is not one of the objects")
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
    ap.add_argument("--no-leak-check", action="store_true",
                    help="skip the blinding check (only for a check of this script, never for "
                         "the packets the evaluators read)")
    a = ap.parse_args()
    g = Grading()
    cats = g.categories()
    runs = {m: latest_runs(g.root(m)) for m in g.models}
    for m in g.models:
        missing = [o for _, objs in cats for o in objs if o not in runs[m]]
        if missing:
            sys.exit(f"{m}: no run for {missing}")
        for slug, _ in cats:
            fig = g.figure(slug, m)
            if not fig.exists():
                sys.exit(f"missing figure {fig}")
    edits = resolve_edits(g, runs, cats)
    where = os.path.relpath(g.mappings, EVAL) if EVAL in g.mappings.parents else g.mappings
    g.mappings.mkdir(parents=True, exist_ok=True)
    for k in range(1, g.evaluators + 1):
        mapping = g.mapping(k)
        (g.mappings / f"mapping_{k}.json").write_text(json.dumps(mapping, indent=1))
        pk = g.packet_dir(k)
        if pk.exists():
            shutil.rmtree(pk)
        (pk / "figures").mkdir(parents=True)
        index, nfiles = [], 0
        for m, lab in sorted(mapping.items(), key=lambda kv: kv[1]):
            for slug, objs in cats:
                fig = pk / f"figures/{slug}_{lab}.png"
                shutil.copy(g.figure(slug, m), fig)
                d = pk / "runs" / lab / slug
                d.mkdir(parents=True)
                for i, o in enumerate(objs, 1):
                    run, status = runs[m][o]
                    reply = d / f"{i}_reply.md"
                    src = run / "agent_reply.md"
                    if src.exists():
                        place(src.resolve(), reply)
                    else:
                        reply.write_text("(the agent left no final reply)\n")
                    sess = session_dir(run)
                    link = None
                    if sess:
                        link = d / f"{i}_session"
                        nfiles += link_tree(sess, link)
                    index.append({"label": lab, "category": slug, "object": o,
                                  "figure": str(fig), "row": i, "reply": str(reply),
                                  "session": str(link) if link else None, "status": status})
        (pk / "index.json").write_text(json.dumps(index, indent=1))
        for e, slug, row, rel in edits:
            target = pk / "runs" / mapping[e["model"]] / slug / rel
            apply_edit(target, e["old"], e["new"], f"packet_{k}, {e['model']} {e['object']!r}")
        print(f"packet_{k}: {len(index)} simulations, {nfiles} session files"
              + (f", {len(edits)} packet edit(s)" if edits else "")
              + f", mapping in {where}/mapping_{k}.json")
    if not a.no_leak_check:
        from leak_check import check_grading
        if not check_grading(g):
            sys.exit("build_packets.py: the blinding check failed (see above); fix the source or "
                     "add a packet_edit or a reviewed leak_allow to grading.json, then rebuild")


if __name__ == "__main__":
    main()
