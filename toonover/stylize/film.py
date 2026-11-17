"""'vintage' finishing pass: 1940s theatrical-cartoon film look (no AI).

Warm faded colour grade, gentle vignette, soft film grain and a tiny gate weave.
Grain changes every frame, like real film.
"""
from __future__ import annotations

import cv2
import numpy as np

_rng = np.random.default_rng(7)


def _lut():
    x = np.arange(256, dtype=np.float32) / 255.0
    # lifted blacks, softened highlights (old film print)
    y = 0.06 + 0.9 * (x ** 0.95)
    return np.clip(y * 255, 0, 255).astype(np.uint8)


_LUT = _lut()
_VIGNETTE: dict = {}


def vintage(img: np.ndarray, frame_index: int = 0, grain: float = 4.0, weave: bool = True) -> np.ndarray:
    h, w = img.shape[:2]
    out = cv2.LUT(img, _LUT)
    # warm, slightly desaturated three-strip look
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.88
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out[..., 2] *= 1.04   # R
    out[..., 0] *= 0.93   # B
    key = (h, w)
    if key not in _VIGNETTE:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
        _VIGNETTE[key] = np.clip(1.0 - 0.18 * np.clip(r - 0.55, 0, None) ** 1.5, 0.75, 1)[..., None]
    out *= _VIGNETTE[key]
    if grain > 0:
        g = _rng.normal(0, grain, (h // 2, w // 2)).astype(np.float32)
        out += cv2.resize(g, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    if weave:
        dx, dy = _rng.normal(0, 0.35), _rng.normal(0, 0.35)
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out
