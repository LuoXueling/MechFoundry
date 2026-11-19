"""Write (or check) the evaluator prompts from evaluator_prompt.md.

    python3 scripts/make_prompts.py [--family fable|astra] [--check]

evaluator_prompt.md is the one template every evaluator gets (the paper's evaluators got it too,
see records/); only its placeholders change:
  {N_AGENTS}       the number of graded agents as a word ("Six")
  {LABELS}         their labels: "**A**, **B**, ..." (the paper's wording)
  {N_OBJECTS}      the instructions each agent ran (50)
  {N_SIMULATIONS}  agents x instructions (300)
  {PACKET}, {OUT}  evaluator k's packet and output file
The rubric and every instruction stay as they are; str.replace is used rather than str.format
because the template's JSON example is full of braces.

Without --check it writes results/<family>/prompt_<k>.md for every family (the prompts are
committed, kept exactly as sent).  With --check it writes nothing and exits non-zero unless every
committed prompt equals the generated text byte for byte; the runners refuse a prompt that fails
this, so what is committed is what the evaluator got."""
import argparse, sys

from config import NUMBER_WORDS, TEMPLATE, Grading


def render(g, family, k):
    n_obj = g.n_objects()
    text = TEMPLATE.read_text()
    for key, val in (("{N_AGENTS}", NUMBER_WORDS[len(g.models)]),
                     ("{LABELS}", ", ".join(f"**{x}**" for x in g.labels())),
                     ("{N_OBJECTS}", str(n_obj)),
                     ("{N_SIMULATIONS}", str(n_obj * len(g.models))),
                     ("{PACKET}", str(g.packet_dir(k))),
                     ("{OUT}", str(g.family_path(family, "out", k)))):
        assert key in text, f"{key} not in {TEMPLATE.name}"
        text = text.replace(key, val)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default=None, help="default: both families")
    ap.add_argument("--check", action="store_true", help="compare with the committed prompts")
    a = ap.parse_args()
    g = Grading()
    ok = True
    for fam in ([a.family] if a.family else list(g.families)):
        for k in range(1, g.evaluators + 1):
            path, text = g.prompt_file(fam, k), render(g, fam, k)
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
