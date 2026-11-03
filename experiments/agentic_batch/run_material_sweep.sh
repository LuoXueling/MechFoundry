#!/usr/bin/env bash
# Re-run a finished simulation N times with N different per-part material assignments: the same
# body, segmentation, supports and loads, a new heterogeneous material each run.  One headless
# Claude Code session per run, recorded in run_batch.sh's layout.
#
#   experiments/agentic_batch/run_material_sweep.sh [options] -n N SOURCE
#
# SOURCE: a solved session folder (e.g. output/agentic_batch/<run>/workflow/<session_id>), or a
# run folder of run_batch.sh (its workflow/<session_id> is taken).  It is only read.
#
# Options (beside the common ones of ab_common.sh: -j, --model, --timeout, --out, --template,
# --effort, --no-path-note, --memory, --cache-ttl, --gpus, --dry-run):
#   -n, --runs N          how many material variants (default 2)
#   --variants FILE       a JSON list of variants to use instead of authoring them
#                         ([{"label", "description", "parts": [{"segment", "segment_label",
#                         "material", "youngs_modulus", "poisson_ratio", "density"}]}])
#   --author-model M      the model that writes the variants (default: --model)
#   -h, --help
#
# The variants: one Claude call writes all N at once, every one heterogeneous (at least two
# materials, the stiffest E 5-500 x the softest, so the workflow takes them as per-part), assigned
# to the segments freely, without regard to what the real object is made of (rubber legs on a steel
# body, a glass shell over foam), and different from each other.  Each run's agent is told its
# variant is the user's given demonstration case: it writes the brief's `Materials by part` and a
# `"mode": "parts"` plan.material listing every segment, and the runner rewinds to the material
# card of a clone of the source (drive.py clone --fresh-solve) and solves.  ab_sweep.sh has the
# flow.  metrics.json records `variant` (label, description, the parts) and `case` (the material
# the solve read: mode, each part's E/nu/rho from the part field, the E contrast).
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ab_common.sh
. "$HERE/ab_common.sh"
SWEEP_KIND=material
# shellcheck source=ab_sweep.sh
. "$HERE/ab_sweep.sh"
