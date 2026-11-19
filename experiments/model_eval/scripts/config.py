"""The grading: which batches are graded, by whom, and where every file of it lives.

    python3 scripts/config.py show
    python3 scripts/config.py runner-env --family FAMILY --k K [--smoke]   (used by the runners)

One blinded grading of the six operator sets (the coding agents) on the 50 objects of
experiments/agentic_batch/objects_50.txt, described by ../grading.json:

  models        the graded batches, in order: key, display name, batch root, figure suffix,
                and extra leak-check terms that name that operator (see leak_check.py).  The
                order fixes the label draw and the table's column order, so it is part of the
                grading's identity: never reorder the models once the evaluators have run.
  evaluators    the number of evaluators per family (3; combine.py's vote needs an odd number)
  label_seed    evaluator k shuffles the labels with random.Random(label_seed + k)
  figures       the staged category figures, {slug} the category and {suffix} the model's
  packets       the folder of packet_<k>/ (the evaluators' reading material, under output/)
  results       the folder of what is kept in git: mappings/, <family>/ (prompts, ratings,
                tables), agreement.md and drift.md
  packet_edits  blinding redactions build_packets.py applies to every packet after placing the
                files, a list of {"model": KEY, "object": INSTRUCTION, "file": "reply" |
                "session/<path in the session folder>", "old": TEXT, "new": TEXT, "why": TEXT}.
                An edit names the model and the object, not a packet path, because the label (and
                so the path) differs for every evaluator.  "old" must occur exactly once, else the
                build stops; the edited file is written anew, so the batch's own file is never
                changed.  Kept in the config rather than done by hand so that a rebuild cannot
                lose them.
  leak_allow    optional reviewed false positives of the blinding check (leak_check.py)
  families      the evaluator families (Fable via Claude Code, Astra via Codex), each with its
                model and effort, and for the runners the output file the prompt names, the log
                folder and the private CLI home ({k} = evaluator, {packets} = the packets folder,
                so that nothing of a run lands outside the grading's own output folder)
  cost_note     the sentence under combine.py's table that says what the cost column is

Relative paths are relative to the repository root.  GRADING_CONFIG=PATH loads another config
instead; it exists for the checks of the scripts themselves (a scratch packets or results folder),
never for the real grading."""
import argparse, json, os, random, re, shlex, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent              # scripts/
EVAL = HERE.parent                                   # experiments/model_eval
# The repo root is where these files sit, not a hard-coded path: the repo moved from
# /home/xl2254/implicit_optimize to /mnt/hdd/xl2254/implicit_optimize after the paper's grading,
# and a worktree is a checkout of its own.
REPO = HERE.parents[2]
TEMPLATE = EVAL / "evaluator_prompt.md"
NUMBER_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def repo_path(p):
    p = Path(p)
    return p if p.is_absolute() else REPO / p


class Grading:
    def __init__(self, path=None):
        self.file = Path(path or os.environ.get("GRADING_CONFIG") or EVAL / "grading.json").resolve()
        self.cfg = json.loads(self.file.read_text())
        self.models = [m["key"] for m in self.cfg["models"]]
        self.model = {m["key"]: m for m in self.cfg["models"]}
        self.evaluators = int(self.cfg["evaluators"])
        self.label_seed = int(self.cfg["label_seed"])
        self.objects_list = repo_path(self.cfg["objects"])
        self.packets = repo_path(self.cfg["packets"])
        self.results = repo_path(self.cfg["results"])
        self.mappings = self.results / "mappings"
        self.families = self.cfg["families"]
        self.cost_note = self.cfg.get("cost_note", "")

    # ---- models, labels, figures --------------------------------------------------------------
    def labels(self):
        return [chr(ord("A") + i) for i in range(len(self.models))]

    def mapping(self, k):
        """model key -> label for evaluator k: the labels shuffled by random.Random(label_seed + k),
        zipped in model order."""
        labels = self.labels()
        random.Random(self.label_seed + k).shuffle(labels)
        return dict(zip(self.models, labels))

    def load_mapping(self, k):
        return json.load(open(self.mappings / f"mapping_{k}.json"))

    def root(self, m):
        return self.model[m]["root"]

    def display(self, m):
        return self.model[m].get("display") or m.capitalize()

    def figure(self, slug, m):
        return repo_path(self.cfg["figures"].format(slug=slug, suffix=self.model[m].get("suffix", "")))

    def categories(self):
        return categories(self.objects_list)

    def n_objects(self):
        return sum(len(o) for _, o in self.categories())

    # ---- families -----------------------------------------------------------------------------
    def family(self, f):
        if f not in self.families:
            raise SystemExit(f"no evaluator family {f!r} (there are {', '.join(self.families)})")
        return self.families[f]

    def family_dir(self, f):
        """The family's prompts, ratings and tables (kept in git)."""
        self.family(f)
        return self.results / f

    def prompt_file(self, f, k):
        return self.family_dir(f) / f"prompt_{k}.md"

    def packet_dir(self, k):
        """Evaluator k's packet; both families read the same one."""
        return self.packets / f"packet_{k}"

    def family_path(self, f, key, k=None):
        """The family's output file ("out"), log folder or CLI home, under output/."""
        return repo_path(self.family(f)[key].format(k=k, packets=self.cfg["packets"]))


def categories(objects_list):
    """[(slug, [object, ...]), ...] from the objects list: the short "# Heading" lines open a
    category, the lines after it are its objects."""
    cats, cur = [], None
    for line in Path(objects_list).read_text().splitlines():
        s = line.strip()
        if s.startswith("# ") and len(s) < 40 and not s.startswith("# Demonstration"):
            cur = (re.sub(r"[^a-z]+", "_", s[2:].lower()).strip("_"), [])
            cats.append(cur)
        elif s and not s.startswith("#") and cur:
            cur[1].append(s)
    assert len(cats) == 10 and all(len(o) == 5 for _, o in cats), cats
    return cats


def main_checkout():
    """The main checkout of this repo (a worktree sits inside it under .claude/worktrees): both
    it and REPO must be hidden from an evaluator."""
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",
                                 "--git-common-dir"], capture_output=True, text=True, check=True)
        return Path(common.stdout.strip()).parent
    except (OSError, subprocess.CalledProcessError):
        return REPO


def runner_env(g, fam, k, smoke=False):
    """Shell assignments for the runners: every path of evaluator k of this family, so the
    shell scripts carry no paths of their own.  Refuses to start an evaluator whose committed
    prompt is missing or stale (it must be exactly what make_prompts.py renders now: the prompt
    is the record of what the evaluator was sent).  With smoke, the prompt is not needed and the
    output and logs go to <logs>/smoke/."""
    f = g.family(fam)
    out = g.family_path(fam, "out", k)
    logs = g.family_path(fam, "logs", k)
    prompt = g.prompt_file(fam, k)
    home = g.family_path(fam, "home", k)
    if smoke:                                   # nothing of a smoke test in the real dirs
        logs = logs / "smoke"
        out = logs / f"evaluator_{k}_out" / out.name
        home = logs / f"home_{k}"
    else:
        from make_prompts import render
        if not prompt.exists() or prompt.read_text() != render(g, fam, k):
            raise SystemExit(f"{prompt} is missing or not what make_prompts.py renders; "
                             f"run scripts/make_prompts.py first")
    hide = [str(REPO), str(main_checkout())]
    env = {"FAMILY": fam, "K": str(k), "MODEL": f["model"], "EFFORT": f["effort"],
           "PACKET": str(g.packet_dir(k)), "OUT": str(out.parent), "OUT_FILE": str(out),
           "LOGS": str(logs), "CLI_HOME": str(home),
           "PROMPT": str(prompt), "HIDE": " ".join(sorted(set(hide)))}
    return "\n".join(f"{key}={shlex.quote(v)}" for key, v in env.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("show", "runner-env"))
    ap.add_argument("--family")
    ap.add_argument("--k", type=int)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    g = Grading()
    if a.cmd == "runner-env":
        if not (a.family and a.k):
            sys.exit("runner-env needs --family and --k")
        print(runner_env(g, a.family, a.k, a.smoke))
        return
    print(f"{g.file}: {len(g.models)} models, {g.evaluators} evaluators per family, "
          f"packets {g.packets}, results {g.results}")
    for m in g.models:
        print(f"  {m}: {g.display(m)}, {g.root(m)}, figures {g.figure('<slug>', m).name}")
    for fam in g.families:
        print(f"  family {fam}: {g.family(fam)['model']} effort {g.family(fam)['effort']}; "
              f"prompts and ratings {g.family_dir(fam)}")


if __name__ == "__main__":
    main()
