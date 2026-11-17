"""Central paths for Toonover."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("TOONOVER_HOME", Path(__file__).resolve().parent.parent))
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"
WORK = ROOT / "work"
OUTPUT = ROOT / "output"

for _d in (MODELS, WORK, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)
