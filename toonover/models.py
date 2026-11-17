"""Download the small MediaPipe tracking models (run once, ~15 MB total).

These are lightweight landmark trackers (not generative AI): they only find
where faces / bodies are, so the cartoon can copy the motion.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

from .config import MODELS

MODEL_URLS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def _valid(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 100_000


def ensure_models(log=print) -> dict[str, Path | None]:
    """Return paths of available models, downloading missing ones.

    Face model is required. Pose model is optional (arms fall back to
    automatic gestures without it).
    """
    out: dict[str, Path | None] = {}
    for name, url in MODEL_URLS.items():
        path = MODELS / name
        if not _valid(path):
            try:
                log(f"Downloading {name} ...")
                tmp = path.with_suffix(".part")
                urllib.request.urlretrieve(url, tmp)
                tmp.replace(path)
            except Exception as e:  # network blocked etc.
                log(f"  could not download {name}: {e}")
        out[name] = path if _valid(path) else None
    import mediapipe as mp
    if out["face_landmarker.task"] is None and not hasattr(mp, "solutions"):
        raise RuntimeError(
            "Face model missing. Download it manually from\n  "
            + MODEL_URLS["face_landmarker.task"]
            + f"\nand save it into {MODELS}"
        )
    return out
