"""Tiny vector drawing kit on top of OpenCV: flat fills + thick ink outlines."""
from __future__ import annotations

import math

import cv2
import numpy as np

INK = (25, 22, 22)       # BGR outline colour
SHIFT = 3                # sub-pixel precision for OpenCV drawing
_F = 1 << SHIFT


def hex2bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def shade(c, f: float):
    return tuple(int(max(0, min(255, v * f))) for v in c)


def _fix(pts) -> np.ndarray:
    return np.round(np.asarray(pts, dtype=np.float64) * _F).astype(np.int32).reshape(-1, 1, 2)


def poly(img, pts, fill=None, outline=INK, lw: float = 2.0):
    p = _fix(pts)
    if fill is not None:
        cv2.fillPoly(img, [p], fill, cv2.LINE_AA, SHIFT)
    if outline is not None and lw > 0:
        cv2.polylines(img, [p], True, outline, max(1, int(round(lw))), cv2.LINE_AA, SHIFT)


def stroke(img, pts, color=INK, lw: float = 2.0, closed=False):
    cv2.polylines(img, [_fix(pts)], closed, color, max(1, int(round(lw))), cv2.LINE_AA, SHIFT)


def limb(img, pts, color, width: float, lw: float):
    """Thick rounded limb with an ink outline (draw outline then fill)."""
    w_out = int(round(width + 2 * lw))
    w_in = max(1, int(round(width)))
    p = _fix(pts)
    cv2.polylines(img, [p], False, INK, w_out, cv2.LINE_AA, SHIFT)
    for q in pts:
        circle(img, q, (width + 2 * lw) / 2, INK, None)
    cv2.polylines(img, [p], False, color, w_in, cv2.LINE_AA, SHIFT)
    for q in pts:
        circle(img, q, width / 2, color, None)


def circle(img, c, r, fill=None, outline=INK, lw: float = 2.0):
    poly(img, ellipse_pts(c, r, r), fill, outline, lw)


def ellipse_pts(c, rx, ry, angle=0.0, a0=0.0, a1=2 * math.pi, n=40):
    t = np.linspace(a0, a1, n, endpoint=(a1 - a0) < 2 * math.pi - 1e-6)
    x, y = rx * np.cos(t), ry * np.sin(t)
    ca, sa = math.cos(angle), math.sin(angle)
    return np.stack([c[0] + x * ca - y * sa, c[1] + x * sa + y * ca], axis=1)


class Frame2D:
    """Local coordinate frame: origin, rotation, scale (units -> pixels)."""

    def __init__(self, origin, angle: float, scale: float):
        self.o = np.asarray(origin, dtype=np.float64)
        self.a = angle
        self.s = scale
        ca, sa = math.cos(angle), math.sin(angle)
        self.R = np.array([[ca, -sa], [sa, ca]])

    def __call__(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        return pts @ self.R.T * self.s + self.o

    def pt(self, x, y):
        return self([[x, y]])[0]

    def child(self, local_origin, d_angle: float = 0.0):
        return Frame2D(self.pt(*local_origin), self.a + d_angle, self.s)
