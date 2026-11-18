"""Write (or check) the evaluator prompts of a round from evaluator_prompt.md.

    python3 make_prompts.py [--round paper] [--family fable|astra] [--check]

evaluator_prompt.md is the one template every evaluator of every round gets; only its
placeholders change:
  {N_AGENTS}       the number of graded agents as a word ("Three", "Two")
  {LABELS}         their labels: "**A**, **B**, **C**" (the paper round's wording) for three or
                   more, "**A** and **B**" for two
  {N_OBJECTS}      the instructions each agent ran (50)
  {N_SIMULATIONS}  agents x instructions (150, 100)
  {PACKET}, {OUT}  evaluator k's packet and output file, as the round names them for the prompt
                   (the paper round keeps its old repo path, see round_config.py)
The rubric and every instruction stay as they are; str.replace is used rather than str.format
because the template's JSON example is full of braces.

Without --check it writes <family prompts>/prompt_<k>.md for every family of the round (the
prompts are committed, kept exactly as sent).  With --check it writes nothing and exits non-zero
unless every committed prompt equals the generated text byte for byte: for the paper round that is
the proof that the template change left the six prompts the evaluators got untouched."""
import argparse, sys

from round_config import DEFAULT_ROUND, HERE, NUMBER_WORDS, Round

TEMPLATE = HERE / "evaluator_prompt.md"


def labels_text(labels):
    bold = [f"**{x}**" for x in labels]
    return " and ".join(bold) if len(bold) == 2 else ", ".join(bold)


def render(rnd, family, k):
    n_obj = rnd.n_objects()
    text = TEMPLATE.read_text()
    for key, val in (("{N_AGENTS}", NUMBER_WORDS[len(rnd.models)]),
                     ("{LABELS}", labels_text(rnd.labels())),
                     ("{N_OBJECTS}", str(n_obj)),
                     ("{N_SIMULATIONS}", str(n_obj * len(rnd.models))),
                     ("{PACKET}", str(rnd.packet_dir(family, k, for_prompt=True))),
                     ("{OUT}", str(rnd.out_file(family, k, for_prompt=True)))):
        assert key in text, f"{key} not in {TEMPLATE.name}"
        text = text.replace(key, val)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=DEFAULT_ROUND)
    ap.add_argument("--family", default=None, help="default: every family of the round")
    ap.add_argument("--check", action="store_true", help="compare with the committed prompts")
    a = ap.parse_args()
    rnd = Round(a.round)
    ok = True
    for fam in ([a.family] if a.family else list(rnd.families)):
        for k in range(1, rnd.evaluators + 1):
            path, text = rnd.prompt_file(fam, k), render(rnd, fam, k)
            if a.check:
                same = path.exists() and path.read_bytes() == text.encode()
                ok &= same
                print(f"{'identical' if same else 'DIFFERS  '}  {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
                print(f"wrote {path}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
