"""Blinding check: does any text in a packet name an operator model, a harness, an evaluator or a
batch folder?

    python3 scripts/leak_check.py            (every packet of the grading)
    python3 scripts/leak_check.py DIR ...    (these folders, with the grading's terms)

build_packets.py runs it at the end of every build and fails the build on any hit.
Exit status 0 only when no packet has a hit.

What is scanned: every file of the packet (each inode once: copies hard-link the same reply and
session files into all packets), its path relative to the packet, the whole text of every file that
is text (no NUL byte in its first 8 KiB), and the tEXt/iTXt/zTXt chunks of PNGs (matplotlib writes
one; a title or a run name would land there).  Other binaries (GLB, NPZ, ...) are not readable as
text by an evaluator and are skipped.  A symlink in a packet is itself a failure: its
target path names the batch folder.

What identifies the operator (fatal), matched case-insensitively and not inside a longer
alphanumeric word (so "astragalus" or base64 noise does not fire):
  - the Claude operator models and their maker: claude (also claude_session, Claude Code),
    anthropic, haiku, sonnet, opus;
  - the open-weight operators: kimi, moonshot(ai), K3.x; deepseek-ai (the Hugging Face org),
    DeepSeek-V<n>[.<m>] and v4.1-flash, flash-1017 (the operator's model id and its date);
  - the harnesses and the evaluators: codex, fable, astra;
  - batch folders: agentic_batch* and each graded model's batch-root name;
  - the config's own per-model "leak_terms" (grading.json).
What does NOT identify the operator, deliberately: a bare "deepseek" and the workflow's own model
ids (deepseek-flash, deepseek-chat, deepseek-reasoner).  The FEM workflow's own LLM calls go to
DeepSeek for every operator, so every session, whichever agent drove it, carries
"provider": "deepseek" and "model": "deepseek-flash" in session.json, operator/last_event.json,
driver_log.jsonl and api_log.jsonl.  The DeepSeek operator's Codex config can even name the same
API model id ("deepseek-flash"), but that config (codex_home/, codex_model.json, metrics.json) is
never in a packet.  So a "deepseek" in a packet says which LLM the workflow used, not which agent
drove it.  Because an agent could still write it about itself, the check also prints, as a
non-fatal warning, how often each label's agent-written text (the reply and operator/*.md) says
"deepseek": a lopsided count is worth reading before the evaluators run.
Also a warning only: Codex tool names (apply_patch, exec_command, write_stdin, view_image).  They
name the harness, not the model; the three open-weight operators share it, but beside the Claude
runs it is a tell, so a warning means reading those lines (and a packet_edit if they are one).

A reviewed false positive can be let through with the config's "leak_allow": a list of
{"pattern": REGEX, "why": TEXT}; a hit is allowed when REGEX matches the line it is on."""
import argparse, os, re, struct, sys, zlib
from collections import Counter
from pathlib import Path

from config import Grading

FATAL = [
    (r"claude", "Claude (operator model family / Claude Code)"),
    (r"anthropic", "Anthropic"),
    (r"haiku", "operator model (Haiku)"), (r"sonnet", "operator model (Sonnet)"),
    (r"opus", "operator model (Opus)"),
    (r"kimi\w*", "operator model (Kimi)"), (r"moonshot\w*", "operator maker (Moonshot AI)"),
    (r"k3\.\d\w*", "operator model version (Kimi K3.x)"),
    (r"deepseek-ai", "operator Hugging Face org (deepseek-ai)"),
    (r"deepseek[-_ ]?v\d+(?:\.\d+)*", "operator model version (DeepSeek-V*)"),
    (r"v4\.1[-_ ]?flash", "operator model id (V4.1-Flash)"),
    (r"flash[-_ ]?1017", "operator model id date (-1017)"),
    (r"codex\w*", "harness (Codex)"),
    (r"fable", "evaluator (Fable)"), (r"astra", "evaluator (Astra)"),
    (r"agentic_batch\w*", "batch folder name"),
]
WARN = [(r"apply_patch|exec_command|write_stdin|view_image", "Codex tool name")]
EDGE = r"(?<![A-Za-z0-9])(?:{})(?![A-Za-z0-9])"


def compile_terms(g):
    terms = list(FATAL)
    for m in g.models:
        known = {t for t, _ in terms}
        terms += [(t, f"{m} leak term") for t in g.model[m].get("leak_terms", []) if t not in known]
        terms.append((re.escape(Path(g.root(m)).name), f"{m} batch root"))
    return [(re.compile(EDGE.format(t), re.I), why) for t, why in terms]


def png_text(data):
    """The text chunks of a PNG (matplotlib's Software tag, any Title/Description)."""
    out, i = [], 8
    while i + 8 <= len(data):
        n, typ = struct.unpack(">I", data[i:i + 4])[0], data[i + 4:i + 8]
        body = data[i + 8:i + 8 + n]
        if typ == b"tEXt":
            out.append(body.replace(b"\0", b": ").decode("latin-1"))
        elif typ == b"zTXt":
            key, _, rest = body.partition(b"\0")
            try:
                out.append(key.decode("latin-1") + ": " + zlib.decompress(rest[1:]).decode("latin-1"))
            except zlib.error:
                pass
        elif typ == b"iTXt":
            out.append(body.replace(b"\0", b" ").decode("utf-8", "replace"))
        if typ == b"IEND":
            break
        i += 12 + n
    return "\n".join(out)


def file_text(path):
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return png_text(data)
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", "replace")


def scan(packet, terms, allow=(), label_of=None):
    """(hits, warnings, deepseek counts per label) for one packet folder."""
    hits, warns, ds = [], Counter(), Counter()
    warn_re = [(re.compile(EDGE.format(t), re.I), why) for t, why in WARN]
    seen = {}
    for root, dirs, files in os.walk(packet):
        for name in dirs + files:
            p = Path(root) / name
            rel = str(p.relative_to(packet))
            if p.is_symlink():
                hits.append((rel, 0, "symlink", f"-> {os.readlink(p)}"))
            for rx, why in terms:
                if rx.search(rel):
                    hits.append((rel, 0, why, "(in the path)"))
        for name in files:
            p = Path(root) / name
            if p.is_symlink():
                continue
            rel = str(p.relative_to(packet))
            st = p.stat()
            key = (st.st_dev, st.st_ino)
            if key not in seen:
                text = file_text(p)
                found = []
                if text:
                    for ln, line in enumerate(text.splitlines(), 1):
                        for rx, why in terms:
                            for mt in rx.finditer(line):
                                if not any(re.search(a, line) for a in allow):
                                    s = max(0, mt.start() - 50)
                                    found.append((ln, why, line[s:mt.end() + 50].strip()))
                        for rx, why in warn_re:
                            if rx.search(line):
                                warns[why] += 1
                seen[key] = (found, text)
            found, text = seen[key]
            hits += [(rel, ln, why, ctx) for ln, why, ctx in found]
            if text and label_of and (name.endswith("_reply.md")
                                      or ("/operator/" in rel and name.endswith(".md"))):
                lab = label_of(rel)
                if lab:
                    ds[lab] += len(re.findall(r"deepseek", text, re.I))
    return hits, warns, ds


def report(packet, hits, warns, ds, limit=40):
    print(f"{packet}: {'CLEAN' if not hits else f'{len(hits)} HIT(S)'}")
    by = Counter(why for _, _, why, _ in hits)
    for why, n in by.most_common():
        print(f"  {n:5d}  {why}")
    for rel, ln, why, ctx in hits[:limit]:
        print(f"    {rel}:{ln}  [{why}]  {ctx[:160]}")
    if len(hits) > limit:
        print(f"    ... {len(hits) - limit} more")
    for why, n in warns.items():
        print(f"  warning: {n} line(s) with a {why} (harness only; not fatal)")
    if ds:
        print("  'deepseek' in agent-written text (reply, operator/*.md) per label: "
              + ", ".join(f"{lab} {n}" for lab, n in sorted(ds.items()))
              + "  (the workflow's own LLM; read them if one label stands out)")


def check_paths(paths, g):
    terms = compile_terms(g)
    allow = [a["pattern"] for a in g.cfg.get("leak_allow", [])]
    ok = True
    for packet in map(Path, paths):
        def label_of(rel):
            parts = rel.split("/")
            return parts[1] if len(parts) > 2 and parts[0] == "runs" else None
        hits, warns, ds = scan(packet, terms, allow, label_of)
        report(packet, hits, warns, ds)
        ok &= not hits
    return ok


def check_grading(g):
    print("blinding check:")
    return check_paths([g.packet_dir(k) for k in range(1, g.evaluators + 1)], g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="packet folders (default: every packet of the grading)")
    a = ap.parse_args()
    g = Grading()
    ok = check_paths(a.paths, g) if a.paths else check_grading(g)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
