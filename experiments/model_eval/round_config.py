"""A grading round: which batches are graded, by whom, and where every file of the round lives.

    python3 round_config.py show [--round NAME]
    python3 round_config.py runner-env --round NAME --family FAMILY --k K   (used by the runners)

A round is one blinded grading of a set of operator batches (the coding agents) on the 50
objects of experiments/agentic_batch/objects_50.txt.  It is described by rounds/<name>.json:

  models        the graded batches, in order: key, display name, batch root, figure suffix,
                and extra leak-check terms that name that operator (see leak_check.py).  The
                order fixes the label draw and the table's column order, so it is part of the
                round's identity: never reorder the models of a round that has been graded.
  evaluators    the number of evaluators per family (3; combine.py's vote needs an odd number)
  label_seed    evaluator k shuffles the labels with random.Random(label_seed + k)
  mappings      the folder of mapping_<k>.json (model key -> label)
  packets       the folder of packet_<k>/ (the evaluators' reading material)
  packet_mode   "symlink" (the paper round's Fable packets: links into the run folders) or
                "copy" (self-contained real files, which every new round uses)
  packet_edits  optional, copy mode only: blinding redactions build_packets.py applies to every
                packet after placing the files, a list of {"model": KEY, "object": INSTRUCTION,
                "file": "reply" | "session/<path in the session folder>", "old": TEXT,
                "new": TEXT, "why": TEXT}.  An edit names the model and the object, not a packet
                path, because the label (and so the path) differs for every evaluator.  "old" must
                occur exactly once, else the build stops; the edited file is written anew, so the
                batch's own file is never changed.  Kept in the config rather than done by hand so
                that a rebuild cannot lose them.
  prompt_repo   the repository path as the prompts name it (default: this checkout).  The
                paper round was graded when the repo lived at /home/xl2254/implicit_optimize, so
                its committed prompts name that path; keeping it here lets make_prompts.py
                regenerate them byte for byte after the move to /mnt/hdd.
  families      the evaluator families (Fable via Claude Code, Astra via Codex), each with its
                model and effort, the folder of its prompts and of its committed ratings, the
                packet and output file as the prompt names them ({k} = evaluator), and for the
                runners: the log folder, the private CLI home ({k}) and extra folders to hide.
  agreement     the two families agreement.py compares, and where it writes agreement.md
  cost_note     the sentence under combine.py's table that says what the cost column is

Relative paths are relative to the repository root, absolute ones are taken as they are."""
import argparse, json, random, re, shlex, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The repo root is where this file sits, not a hard-coded path: the repo moved from
# /home/xl2254/implicit_optimize to /mnt/hdd/xl2254/implicit_optimize after the paper round, and
# a worktree is a checkout of its own.
REPO = HERE.parents[1]
ROUNDS = HERE / "rounds"
DEFAULT_ROUND = "paper"
NUMBER_WORDS = {2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}


def repo_path(p, base=None):
    p = Path(p)
    return p if p.is_absolute() else (base or REPO) / p


class Round:
    def __init__(self, name=DEFAULT_ROUND):
        path = Path(name)
        if path.suffix != ".json":
            path = ROUNDS / f"{name}.json"
        self.file = path.resolve()
        self.cfg = json.loads(path.read_text())
        self.name = self.cfg["name"]
        self.models = [m["key"] for m in self.cfg["models"]]
        self.model = {m["key"]: m for m in self.cfg["models"]}
        self.evaluators = int(self.cfg.get("evaluators", 3))
        self.label_seed = int(self.cfg.get("label_seed", 1000))
        self.objects_list = repo_path(self.cfg.get("objects", "experiments/agentic_batch/objects_50.txt"))
        self.figure_template = self.cfg.get("figures", "illustrations/agentic_results_{slug}{suffix}.png")
        self.mappings = repo_path(self.cfg["mappings"])
        self.packets = repo_path(self.cfg["packets"])
        self.packet_mode = self.cfg.get("packet_mode", "copy")
        self.prompt_repo = Path(self.cfg.get("prompt_repo") or REPO)
        self.families = self.cfg.get("families", {})
        self.cost_note = self.cfg.get("cost_note", "")

    # ---- models, labels, figures --------------------------------------------------------------
    def labels(self):
        return [chr(ord("A") + i) for i in range(len(self.models))]

    def mapping(self, k):
        """model key -> label for evaluator k.  For three models this is exactly the paper
        round's draw: list("ABC") shuffled by random.Random(1000 + k), zipped in model order."""
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
        return repo_path(self.figure_template.format(slug=slug, suffix=self.model[m].get("suffix", "")))

    def categories(self):
        """[(slug, [object, ...]), ...] from the objects list: the short "# Heading" lines open a
        category, the lines after it are its objects."""
        cats, cur = [], None
        for line in self.objects_list.read_text().splitlines():
            s = line.strip()
            if s.startswith("# ") and len(s) < 40 and not s.startswith("# Demonstration"):
                cur = (re.sub(r"[^a-z]+", "_", s[2:].lower()).strip("_"), [])
                cats.append(cur)
            elif s and not s.startswith("#") and cur:
                cur[1].append(s)
        assert len(cats) == 10 and all(len(o) == 5 for _, o in cats), cats
        return cats

    def n_objects(self):
        return sum(len(o) for _, o in self.categories())

    # ---- families -----------------------------------------------------------------------------
    def family(self, f):
        if f not in self.families:
            raise SystemExit(f"round {self.name}: no evaluator family {f!r} "
                             f"(it has {', '.join(self.families) or 'none'})")
        return self.families[f]

    def default_family(self):
        return next(iter(self.families))

    def prompt_file(self, f, k):
        return repo_path(self.family(f)["prompts"]) / f"prompt_{k}.md"

    def ratings_dir(self, f):
        return repo_path(self.family(f)["ratings"])

    def _fmt(self, template, k):
        return template.format(k=k, packets=self.cfg["packets"])

    def packet_dir(self, f, k, for_prompt=False):
        """Evaluator k's packet: as the prompt names it (prompt_repo) or as it is on disk."""
        return repo_path(self._fmt(self.family(f)["packet"], k),
                         self.prompt_repo if for_prompt else REPO)

    def out_file(self, f, k, for_prompt=False):
        return repo_path(self._fmt(self.family(f)["out"], k), self.prompt_repo if for_prompt else REPO)

    def family_path(self, f, key, k=None):
        v = self.family(f).get(key)
        return None if v is None else repo_path(self._fmt(v, k))


def main_checkout():
    """The main checkout of this repo (a worktree sits inside it under .claude/worktrees): both
    it and REPO must be hidden from an evaluator."""
    try:
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--path-format=absolute",
                                 "--git-common-dir"], capture_output=True, text=True, check=True)
        return Path(common.stdout.strip()).parent
    except (OSError, subprocess.CalledProcessError):
        return REPO


def runner_env(rnd, fam, k, smoke=False):
    """Shell assignments for the runners: every path of evaluator k of this family, so the
    shell scripts carry no paths of their own.  Refuses a round whose committed prompt for this
    evaluator is missing or stale (it must be exactly what make_prompts.py renders now: the
    prompt is the record of what the evaluator was sent).  With smoke, the prompt is not needed
    and the output and logs go to <logs>/smoke/."""
    f = rnd.family(fam)
    if rnd.prompt_repo != REPO:
        raise SystemExit(
            f"round {rnd.name}: its prompts name the repo as {rnd.prompt_repo}, but it is at "
            f"{REPO}; the evaluator would be sent to paths that do not exist.  Re-running this "
            f"round needs a new round config (with prompt_repo left out) and new prompts.")
    for key in ("logs", "home"):
        if not f.get(key):
            raise SystemExit(f"round {rnd.name}, family {fam}: no {key!r} in the config")
    out = rnd.out_file(fam, k)
    logs = rnd.family_path(fam, "logs", k)
    prompt = rnd.prompt_file(fam, k)
    home = rnd.family_path(fam, "home", k)
    if smoke:                                   # nothing of a smoke test in the real dirs
        logs = logs / "smoke"
        out = logs / f"evaluator_{k}_out" / out.name
        home = logs / f"home_{k}"
    else:
        from make_prompts import render
        if not prompt.exists() or prompt.read_text() != render(rnd, fam, k):
            raise SystemExit(f"{prompt} is missing or not what make_prompts.py renders for "
                             f"round {rnd.name}; run make_prompts.py --round {rnd.name} first")
    hide = [str(REPO), str(main_checkout())] + [str(repo_path(h)) for h in f.get("hide", [])]
    env = {"ROUND": rnd.name, "FAMILY": fam, "K": str(k),
           "MODEL": f["model"], "EFFORT": f["effort"],
           "PACKET": str(rnd.packet_dir(fam, k)), "OUT": str(out.parent), "OUT_FILE": str(out),
           "LOGS": str(logs), "CLI_HOME": str(home),
           "PROMPT": str(prompt), "HIDE": " ".join(sorted(set(hide)))}
    return "\n".join(f"{key}={shlex.quote(v)}" for key, v in env.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("show", "runner-env"))
    ap.add_argument("--round", default=DEFAULT_ROUND)
    ap.add_argument("--family")
    ap.add_argument("--k", type=int)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    rnd = Round(a.round)
    if a.cmd == "runner-env":
        if not (a.family and a.k):
            sys.exit("runner-env needs --family and --k")
        print(runner_env(rnd, a.family, a.k, a.smoke))
        return
    print(f"round {rnd.name} ({rnd.file}): {len(rnd.models)} models, {rnd.evaluators} evaluators "
          f"per family, packets {rnd.packets} ({rnd.packet_mode})")
    for m in rnd.models:
        print(f"  {m}: {rnd.display(m)}, {rnd.root(m)}, figures {rnd.figure('<slug>', m).name}")
    for fam in rnd.families:
        print(f"  family {fam}: {rnd.family(fam)['model']} effort {rnd.family(fam)['effort']}; "
              f"prompts {rnd.prompt_file(fam, 1).parent}, ratings {rnd.ratings_dir(fam)}")


if __name__ == "__main__":
    main()
