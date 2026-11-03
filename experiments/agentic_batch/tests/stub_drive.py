#!/usr/bin/env python3
"""A stand-in for `drive.py clone` (via AB_DRIVE_BIN) in the sweep stub test: copies the source
session folder to $AB_WORKFLOW_ROOT/<new id> and prints the command's JSON, as the real one does
(the real clone, with its checkpoints, is tested in operator_skill/tests/test_clone_and_seed.py).
Logs each call to $STUB_LOG.drive."""
import json
import os
import shutil
import sys
import uuid

args = sys.argv[1:]
assert args[0] == "clone", args
assert os.environ.get("FEM_OPERATOR_OUTPUT_ROOT", "").startswith("output/fem_workflow_operator/_slots/"), \
    "stub_drive: not run from a slot"
source = args[1]
assert "--fresh-solve" in args, "stub_drive: the sweep must clone with --fresh-solve"
assert "--instruction" in args and " | " in args[args.index("--instruction") + 1], \
    "stub_drive: the sweep must give the copy its variant as the instruction"
note = args[args.index("--note") + 1] if "--note" in args else None
new = uuid.uuid4().hex[:12]
dest = os.path.join(os.environ["AB_WORKFLOW_ROOT"], new)
shutil.copytree(source, dest)
os.makedirs(os.path.join(dest, "operator"), exist_ok=True)
json.dump({"source": source, "note": note}, open(os.path.join(dest, "operator", "cloned_from.json"), "w"))
before = json.load(open(os.path.join(source, "session.json")))
json.dump({**before, "session_id": new, "object": note}, open(os.path.join(dest, "session.json"), "w"))
with open(os.environ.get("STUB_LOG", "/dev/null") + ".drive", "a") as fh:
    fh.write(" ".join(args) + "\n")
print(f"session_id: {new}", file=sys.stderr)
print(json.dumps({"ok": True, "session_id": new, "cloned_from": os.path.basename(source),
                  "kind": "shape_opt_settings"}, indent=1))
