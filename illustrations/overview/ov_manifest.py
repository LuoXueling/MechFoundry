"""Read an overview asset only if it is the file ``ov_assets.py`` recorded.

Shared by ``ov_bunny_loop.py`` and ``ov_surfaces.py``: the generative assets are
not reproducible, so a build must refuse one that changed under it rather than
quietly draw something nobody recorded the provenance of.
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
MANIFEST = json.load(open(os.path.join(ASSETS, "MANIFEST.json")))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def asset(rel):
    """An asset path, refused unless it is the file the manifest recorded."""
    path = os.path.join(ASSETS, rel)
    entry = MANIFEST.get(rel)
    assert entry is not None, f"{rel} is not in assets/MANIFEST.json (run ov_assets.py)"
    assert _sha(path) == entry["sha256"], f"{rel} changed since ov_assets.py recorded it"
    return path


def has(rel):
    return rel in MANIFEST and os.path.exists(os.path.join(ASSETS, rel))
