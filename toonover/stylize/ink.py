"""Ink lines (XDoG) - classic edge stylization, no AI."""
from __future__ import annotations

import cv2
import numpy as np


def xdog(gray: np.ndarray, sigma: float = 0.9, k: float = 1.6, tau: float = 0.985,
         eps: float = -0.01, phi: float = 60.0) -> np.ndarray:
    """Extended Difference-of-Gaussians. Returns ink strength 0..1 (1 = black line)."""
    g = gray.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(g, (0, 0), sigma)
    g2 = cv2.GaussianBlur(g, (0, 0), sigma * k)
    d = g1 - tau * g2
    e = np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    return np.clip(1.0 - e, 0, 1)


def clean(ink: np.ndarray, threshold: float = 0.35, min_area: int = 12) -> np.ndarray:
    """Drop tiny specks so only real lines remain. Returns 0..1 float map."""
    mask = (ink > threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, np.uint8)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    m = keep[lab].astype(np.float32)
    return cv2.GaussianBlur(m * np.clip(ink * 1.5, 0, 1), (3, 3), 0)


def region_edges(img_bgr: np.ndarray, low: int = 60, high: int = 140, width: int = 1) -> np.ndarray:
    """Outlines between flat colour regions (cel borders)."""
    e = cv2.Canny(img_bgr, low, high)
    if width > 1:
        e = cv2.dilate(e, np.ones((width, width), np.uint8))
    return cv2.GaussianBlur(e.astype(np.float32) / 255.0, (3, 3), 0)


def apply_ink(img: np.ndarray, ink: np.ndarray, color=(22, 18, 18), strength: float = 1.0) -> np.ndarray:
    a = np.clip(ink * strength, 0, 1)[..., None]
    return (img.astype(np.float32) * (1 - a) + np.array(color, np.float32) * a).astype(np.uint8)
