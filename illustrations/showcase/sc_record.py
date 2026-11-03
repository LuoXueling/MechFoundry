"""Keep each sweep row's authoring prompt in experiments/agentic_batch (CPU, no GPU).

    python3 sc_record.py REPO_ROOT          # from illustrations/showcase, after the build

The material and loading rows each start with ONE Claude call that writes all of the row's
cases (ab_sweep.sh).  Its prompt is kept deep in the batch folder
(output/showcase/<sweep>/_batches/<batch>/authoring/prompt.txt), which sc_stats.json names
(rows[].calls.authoring.prompt_source).  This copies it, as sent, to
experiments/agentic_batch/showcase_authoring_prompt_<row>.txt (material, bc) beside the sweep
scripts and the run-prompt templates, so the figure's prompts are tracked with the code.  A row
without an authoring call (shape) is skipped.
"""

import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    repo = sys.argv[1]
    # the checkout this script lives in (REPO_ROOT only locates the git-ignored sweep files)
    repo_out = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments", "agentic_batch")
    stats = json.load(open(os.path.join(HERE, "sc_stats.json")))
    for row in stats.get("rows", []):
        auth = (row.get("calls") or {}).get("authoring")
        if not auth:
            continue
        src = os.path.join(repo, auth["prompt_source"])            # <sweep>/_batches/<b>/authoring/prompt.txt
        name = {"material": "material", "loading": "bc"}.get(row["key"], row["key"])
        dst = os.path.join(repo_out, f"showcase_authoring_prompt_{name}.txt")
        shutil.copy(src, dst)
        print(f"recorded {row['key']}: {os.path.relpath(dst, repo)}")


if __name__ == "__main__":
    main()
