"""Procedural flat-2D backgrounds. Drop a PNG in assets/backgrounds/<name>.png to override."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .config import ASSETS
from .draw import INK, circle, ellipse_pts, hex2bgr, poly, shade, stroke

BACKGROUNDS = ("stage", "living_room", "street")


def make_background(name: str, w: int, h: int, floor_y: float) -> tuple[np.ndarray, tuple]:
    """Return (image, shadow_colour)."""
    png = ASSETS / "backgrounds" / f"{name}.png"
    if png.exists():
        img = cv2.resize(cv2.imread(str(png)), (w, h), interpolation=cv2.INTER_AREA)
        fy = int(min(max(floor_y + 4, 0), h - 1))
        return img, shade(tuple(int(v) for v in img[fy, w // 2]), 0.7)
    fn = {"stage": _stage, "living_room": _living_room, "street": _street}.get(name, _stage)
    return fn(w, h, floor_y)


def _stage(w, h, fy):
    lw = max(2, w / 300)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = hex2bgr("#1B2140")
    # back wall panels
    for i in range(0, w, int(w / 7)):
        stroke(img, [(i, 0), (i, fy)], hex2bgr("#232A52"), lw * 2)
    # neon ring logo (original, not any real show's)
    cx, cy, r = w / 2, h * 0.17, w * 0.11
    for k, a in ((3.2, 0.25), (2.0, 0.5)):
        overlay = img.copy()
        circle(overlay, (cx, cy), r * 1.0, None, hex2bgr("#9FD8FF"), r * 0.18 * k)
        img = cv2.addWeighted(overlay, a, img, 1 - a, 0)
    circle(img, (cx, cy), r, hex2bgr("#10142A"), hex2bgr("#E8F6FF"), r * 0.12)
    cv2.putText(img, "T", (int(cx - r * 0.42), int(cy + r * 0.45)), cv2.FONT_HERSHEY_DUPLEX, r / 22, (245, 250, 255), int(r / 9), cv2.LINE_AA)
    # curtains
    for side in (-1, 1):
        x0 = 0 if side < 0 else w
        cw = w * 0.2
        folds = 5
        for f in range(folds):
            a = x0 + side * cw * f / folds
            b = x0 + side * cw * (f + 1) / folds
            colr = hex2bgr("#B3122A") if f % 2 == 0 else hex2bgr("#8E0D21")
            poly(img, [(a, 0), (b, 0), (b + side * 6, fy), (a, fy)], colr, None)
        stroke(img, [(x0 + side * cw, 0), (x0 + side * cw, fy)], INK, lw)
    # valance
    poly(img, [(0, 0), (w, 0), (w, h * 0.05), (0, h * 0.05)], hex2bgr("#8E0D21"), INK, lw)
    # floor
    poly(img, [(0, fy - h * 0.06), (w, fy - h * 0.06), (w, h), (0, h)], hex2bgr("#6B4426"), INK, lw)
    for i in range(-6, 14):
        stroke(img, [(w / 2 + i * w * 0.09, fy - h * 0.06), (w / 2 + i * w * 0.2, h)], hex2bgr("#5A381F"), lw)
    # spotlight
    overlay = img.copy()
    poly(overlay, [(w * 0.38, 0), (w * 0.62, 0), (w * 0.95, fy + h * 0.05), (w * 0.05, fy + h * 0.05)], (220, 235, 255), None)
    img = cv2.addWeighted(overlay, 0.12, img, 0.88, 0)
    return img, hex2bgr("#4A2E18")


def _living_room(w, h, fy):
    lw = max(2, w / 300)
    img = np.zeros((h, w, 3), np.uint8)
    wall = hex2bgr("#D8D68A")
    img[:] = wall
    # peeled plaster patches
    for (x, y, s) in ((0.15, 0.12, 0.08), (0.78, 0.08, 0.06), (0.85, 0.42, 0.05)):
        pts = [(w * x + math.cos(a) * w * s * (0.7 + 0.3 * math.sin(a * 3)),
                h * y + math.sin(a) * w * s * 0.6) for a in np.linspace(0, 6.28, 12, endpoint=False)]
        poly(img, pts, shade(wall, 0.82), INK, lw * 0.7)
    # door
    poly(img, [(w * 0.02, fy * 0.25), (w * 0.22, fy * 0.25), (w * 0.22, fy), (w * 0.02, fy)], hex2bgr("#8A5A3B"), INK, lw)
    poly(img, [(w * 0.05, fy * 0.3), (w * 0.19, fy * 0.3), (w * 0.19, fy * 0.6), (w * 0.05, fy * 0.6)], hex2bgr("#7A4E33"), INK, lw)
    circle(img, (w * 0.19, fy * 0.65), w * 0.01, hex2bgr("#E0B84A"), INK, lw * 0.6)
    # photo frames
    for (x, y, fw, fh) in ((0.42, 0.16, 0.11, 0.1), (0.57, 0.2, 0.08, 0.08), (0.48, 0.3, 0.09, 0.07)):
        poly(img, [(w * x, h * y), (w * (x + fw), h * y), (w * (x + fw), h * y + w * fh), (w * x, h * y + w * fh)],
             hex2bgr("#9B6B3E"), INK, lw)
        poly(img, [(w * (x + 0.012), h * y + w * 0.012), (w * (x + fw - 0.012), h * y + w * 0.012),
                   (w * (x + fw - 0.012), h * y + w * (fh - 0.012)), (w * (x + 0.012), h * y + w * (fh - 0.012))],
             hex2bgr("#BFE3D0"), None)
    # dustbin
    poly(img, [(w * 0.27, fy - h * 0.12), (w * 0.37, fy - h * 0.12), (w * 0.355, fy), (w * 0.285, fy)], hex2bgr("#5B8FA8"), INK, lw)
    # floor tiles
    poly(img, [(0, fy - h * 0.02), (w, fy - h * 0.02), (w, h), (0, h)], hex2bgr("#C9BBA0"), INK, lw)
    rows = 5
    for r in range(rows):
        y0 = fy - h * 0.02 + (h - fy) * (r / rows) ** 1.2
        y1 = fy - h * 0.02 + (h - fy) * ((r + 1) / rows) ** 1.2
        n = 8
        for cidx in range(-2, n + 2):
            if (cidx + r) % 2:
                continue
            xa = w / 2 + (cidx - n / 2) * w / n * (1 + r * 0.25)
            xb = w / 2 + (cidx + 1 - n / 2) * w / n * (1 + r * 0.25)
            xc = w / 2 + (cidx + 1 - n / 2) * w / n * (1 + (r + 1) * 0.25)
            xd = w / 2 + (cidx - n / 2) * w / n * (1 + (r + 1) * 0.25)
            poly(img, [(xa, y0), (xb, y0), (xc, y1), (xd, y1)], hex2bgr("#8E7F66"), None)
    stroke(img, [(0, fy - h * 0.02), (w, fy - h * 0.02)], INK, lw)
    return img, hex2bgr("#7A6C55")


def _street(w, h, fy):
    lw = max(2, w / 300)
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = hex2bgr("#8FD3F0")
    for (x, y) in ((0.2, 0.08), (0.7, 0.13)):
        for dx, r in ((0, 0.06), (0.06, 0.05), (-0.06, 0.045)):
            circle(img, (w * (x + dx), h * y), w * r, (255, 255, 255), None)
    cols = ["#E8A87C", "#C38D9E", "#85CDCA", "#E27D60", "#F6D55C", "#9FB4C7"]
    x = -w * 0.05
    i = 0
    while x < w:
        bw = w * (0.18 + 0.06 * (i % 3))
        top = h * (0.1 + 0.08 * ((i * 7) % 4))
        c = hex2bgr(cols[i % len(cols)])
        poly(img, [(x, top), (x + bw, top), (x + bw, fy), (x, fy)], c, INK, lw)
        for yy in np.arange(top + h * 0.04, fy - h * 0.1, h * 0.09):
            for xx in np.arange(x + bw * 0.15, x + bw - bw * 0.25, bw * 0.35):
                poly(img, [(xx, yy), (xx + bw * 0.2, yy), (xx + bw * 0.2, yy + h * 0.05), (xx, yy + h * 0.05)],
                     hex2bgr("#FFF6C8"), INK, lw * 0.6)
        x += bw
        i += 1
    poly(img, [(0, fy - h * 0.03), (w, fy - h * 0.03), (w, h), (0, h)], hex2bgr("#7D7D7D"), INK, lw)
    for xx in np.arange(0, w, w * 0.18):
        poly(img, [(xx, fy + (h - fy) * 0.5), (xx + w * 0.1, fy + (h - fy) * 0.5),
                   (xx + w * 0.1, fy + (h - fy) * 0.56), (xx, fy + (h - fy) * 0.56)], (240, 240, 240), None)
    return img, hex2bgr("#5E5E5E")
