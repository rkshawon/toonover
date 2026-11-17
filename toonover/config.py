"""Central settings and paths for Toonover."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(os.environ.get("TOONOVER_HOME", Path(__file__).resolve().parent.parent))
ASSETS = ROOT / "assets"
MODELS = ROOT / "models"
WORK = ROOT / "work"
OUTPUT = ROOT / "output"

for _d in (MODELS, WORK, OUTPUT):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class RenderSettings:
    width: int = 720            # output video width (9:16 Shorts by default)
    height: int = 1280
    supersample: int = 2        # draw at 2x then downscale = smooth anti-aliased lines
    max_fps: float = 30.0
    max_seconds: float = 90.0   # memes are short; keeps jobs cheap
    top_bar: int = 170          # black title bar (like popular cartoon Shorts)
    bottom_bar: int = 120       # black channel-name bar
    channel_name: str = "Toonover"
    title: str = ""
    background: str = "stage"   # stage | living_room | street
    cast: list[str] = field(default_factory=list)  # character ids, left -> right; empty = auto
    camera_moves: bool = True   # gentle zoom on close-ups / active speaker
    max_characters: int = 4
    crf: int = 20
