"""Try several redraw settings on one picture and save a labelled comparison sheet.

Pick the tile you like and copy its numbers into assets/redraw_presets.json.
"""
from __future__ import annotations

import copy
from pathlib import Path

import cv2
import numpy as np

from .comfy import ComfyRedrawer

# (label, overrides)
VARIANTS = [
    ("A current", {}),
    ("B closer face", {"control_strength": 0.95, "tile_strength": 0.6}),
    ("C more cartoon", {"control_strength": 0.6, "tile_strength": 0.35, "cfg": 8.0}),
    ("D bolder style", {"cfg": 9.0, "lora_boost": 0.5}),
    ("E softer guides", {"control_end": 0.7, "tile_end": 0.4}),
    ("F seed 2", {"seed": 777}),
]


def tune(image: Path, style: str, describe: str = "", out: Path | None = None) -> Path:
    frame = cv2.imread(str(image))
    if frame is None:
        raise FileNotFoundError(image)
    base = ComfyRedrawer(style, describe=describe)
    tiles = []
    for label, over in VARIANTS:
        rd = copy.copy(base)
        rd.p = copy.deepcopy(base.p)
        over = dict(over)
        boost = over.pop("lora_boost", 0)
        rd.p.update(over)
        if boost:
            rd.p["_loras_found"] = [(n, s + boost) for n, s in rd.p.get("_loras_found", [])]
        print(f"  {label} ...", flush=True)
        img = cv2.resize(rd(frame), (384, int(384 * frame.shape[0] / frame.shape[1])))
        txt = label + "  " + ", ".join(f"{k}={v}" for k, v in over.items()) + (f" lora+{boost}" if boost else "")
        bar = np.zeros((28, img.shape[1], 3), np.uint8)
        cv2.putText(bar, txt[:48], (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, img]))
    src = cv2.resize(frame, (tiles[0].shape[1], tiles[0].shape[0] - 28))
    bar = np.zeros((28, src.shape[1], 3), np.uint8)
    cv2.putText(bar, "original", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    tiles.insert(0, np.vstack([bar, src]))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    out = out or image.with_name(f"{image.stem}_{style}_tune.png")
    cv2.imwrite(str(out), np.vstack(rows))
    return out
