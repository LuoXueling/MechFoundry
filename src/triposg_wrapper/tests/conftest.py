import os
import sys
from pathlib import Path

# TripoSG imports itself as the top-level package ``triposg``.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, os.fspath(REPO_ROOT / "src" / "dependency" / "TripoSG"))
