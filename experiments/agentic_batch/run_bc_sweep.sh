#!/usr/bin/env bash
# Re-run a finished simulation N times with N different supports and loads: the same body,
# segmentation and material, a new load case each run.  One headless Claude Code session per run,
# recorded in run_batch.sh's layout.
#
#   experiments/agentic_batch/run_bc_sweep.sh [options] -n N SOURCE
#
# SOURCE: a solved session folder (e.g. output/agentic_batch/<run>/workflow/<session_id>), or a
# run folder of run_batch.sh (its workflow/<session_id> is taken).  It is only read.
#
# Options (beside the common ones of ab_common.sh: -j, --model, --timeout, --out, --template,
# --effort, --no-path-note, --memory, --cache-ttl, --gpus, --dry-run):
#   -n, --runs N          how many load cases (default 2)
#   --variants FILE       a JSON list of load cases to use instead of authoring them
#                         ([{"label", "description", "supports": [...], "loads": [...]}])
#   --author-model M      the model that writes the load cases (default: --model)
#   -h, --help
#
# The load cases: one Claude call writes all N at once, chosen freely rather than as the object's
# likeliest or governing use, but physically possible (supports and loads on real surfaces of
# named segments, supports that stop every rigid-body motion, sizes plausible for the object's
# mass and size), and different from each other in support region, load region, direction and kind
# (a patch, a distributed load, a twist, a body load).  Each run's agent is told its case is the
# user's given demonstration case, so the skill's preference for the governing case does not apply
# (its physical checks do): it rewrites the brief's Use, Contacts and Load sections and the plan's
# regions, and the runner rewinds to the regions card of a clone of the source (drive.py clone
# --fresh-solve) and solves.  ab_sweep.sh has the flow.  metrics.json records `variant` (label,
# description, supports, loads) and `case` (the held and loaded regions and body loads solved).
set -u -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=ab_common.sh
. "$HERE/ab_common.sh"
SWEEP_KIND=bc
# shellcheck source=ab_sweep.sh
. "$HERE/ab_sweep.sh"
