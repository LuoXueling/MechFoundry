#!/usr/bin/env bash
# Stub test of run_realizations.sh, run_material_sweep.sh and run_bc_sweep.sh: no API calls, no
# GPU.  Everything goes to a temp dir (runs, a fake source session, fake transcripts); `claude` is
# tests/stub_claude.py and `drive.py clone` is tests/stub_drive.py.  Prints the checker's verdict.
#   bash experiments/agentic_batch/tests/test_variants_stub.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="$(mktemp -d /tmp/agentic_variants_stub.XXXXXX)"
export AB_WORKFLOW_ROOT="$TMP/fem_workflow_operator" AB_PROJECTS_DIR="$TMP/projects"
export STUB_LOG="$TMP/stub.log" CLAUDE_BIN="$HERE/stub_claude.py" AB_DRIVE_BIN="$HERE/stub_drive.py"
export CLAUDE_CODE_SESSION_ID=parent-session CLAUDECODE=1
mkdir -p "$AB_WORKFLOW_ROOT" "$AB_PROJECTS_DIR"

# A finished source session, as run_batch.sh copies one: three segments, a per-part case.
SRC="$TMP/source_run/workflow/aaaaaaaaaaaa"
mkdir -p "$SRC/operator" "$SRC/brief_snapshots" "$SRC/boundary_conditions" "$SRC/case" "$SRC/materials/parts/field" "$SRC/geometry"
: > "$SRC/checkpoints.sqlite"
echo '{"session_id": "aaaaaaaaaaaa"}' > "$TMP/source_run/metrics.json"
printf 'png' > "$SRC/geometry/views.png"
printf 'png' > "$SRC/operator/segments.png"
cat > "$SRC/session.json" <<'EOF'
{"session_id": "aaaaaaaaaaaa", "object": "source", "unit_system": "SI (m-kg-s)",
 "segment_description": "0: seat; 1: legs; 2: backrest"}
EOF
echo '{"instruction": "a stub chair, loaded by a sitting adult"}' > "$SRC/brief_snapshots/instruction.json"
printf '# Case brief\n\n## Use\nA person sits.\n\n## Size and material\nReal mass: 5 kg\n' > "$SRC/operator/brief.md"
cat > "$SRC/boundary_conditions/segments.json" <<'EOF'
{"bounds": [[0, 0, 0], [0.5, 0.9, 0.5]], "segments": [
 {"id": 0, "area_fraction": 0.4, "centroid": [0.25, 0.45, 0.25], "bounds": [[0, 0.4, 0], [0.5, 0.5, 0.5]], "facing_fraction_45deg": {"+y": 0.5, "-y": 0.5}},
 {"id": 1, "area_fraction": 0.4, "centroid": [0.25, 0.2, 0.25], "bounds": [[0, 0, 0], [0.5, 0.4, 0.5]], "facing_fraction_45deg": {"+x": 0.5}},
 {"id": 2, "area_fraction": 0.2, "centroid": [0.25, 0.7, 0.0], "bounds": [[0, 0.5, 0], [0.5, 0.9, 0.05]], "facing_fraction_45deg": {"+z": 0.6}}]}
EOF
cat > "$SRC/case/case.json" <<'EOF'
{"material": {"mode": "parts", "artifact": "/elsewhere/materials/parts/field"},
 "boundary_conditions": {"dirichlet": [{"name": "feet", "components": [0, 1, 2]}],
                         "neumann": [{"name": "seat_top", "traction": [0, -100, 0]}]},
 "loads": {"acceleration": [0, -9.80665, 0], "body_force": [0, 0, 0]}}
EOF
cat > "$SRC/materials/parts/field/manifest.json" <<'EOF'
{"provenance": {"parts": [{"id": 0, "youngs_modulus": 1e10}, {"id": 1, "youngs_modulus": 1e8}, {"id": 2, "youngs_modulus": 1e10}]},
 "ranges": {"youngs_modulus": [1e8, 1e10]}}
EOF
cat > "$TMP/bc_variants.json" <<'EOF'
[{"label": "clamped legs, side push", "description": "legs clamped; 50 N along +x on the backrest",
  "supports": [{"segments": [1], "faces": "-y", "held": "all"}],
  "loads": [{"kind": "patch", "segments": [2], "faces": "+z", "direction": [1, 0, 0], "size": 50, "unit": "N", "why": "a push"}]},
 {"label": "seat twist", "description": "legs clamped; 10 N m about y on the seat",
  "supports": [{"segments": [1], "faces": "-y", "held": "all"}],
  "loads": [{"kind": "torsion", "segments": [0], "faces": "+y", "direction": [0, 1, 0], "size": 10, "unit": "N m", "why": "a twist"}]}]
EOF
find "$SRC" -type f | sort | xargs md5sum > "$TMP/source_before.md5"

S="$HERE/.."
bash "$S/run_realizations.sh" -j 2 -n 3 --seed-base 40 --timeout 30 --out "$TMP/out_real" "a stub bench sleep1"
echo "run_realizations.sh exit: $?"
STUB_AUTHOR_BAD=1 bash "$S/run_material_sweep.sh" -j 2 -n 2 --timeout 30 --out "$TMP/out_mat" "$TMP/source_run"
echo "run_material_sweep.sh exit: $?"
bash "$S/run_bc_sweep.sh" -j 1 -n 2 --timeout 30 --variants "$TMP/bc_variants.json" --out "$TMP/out_bc" "$SRC"
echo "run_bc_sweep.sh exit: $?"
python3 "$HERE/check_variants_stub.py" "$TMP"
echo "temp dir: $TMP"
