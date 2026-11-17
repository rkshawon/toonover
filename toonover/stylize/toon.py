"""'toon' style - no AI. Rotoscope-like cel cartoon traced from the real footage.

smooth (edge-preserving) -> saturation boost -> per-shot brightness bands x hue palette
-> temporal hysteresis (no flicker) -> ink lines (XDoG + cel borders)

Because it traces the real frame, the person stays 100% recognizable.
"""
from __future__ import annotations

import cv2
import numpy as np

from .ink import apply_ink, clean, xdog
from .temporal import FlowTracker


class ToonStylizer:
    """Cel shading = a few flat brightness bands x a few flat hues, chosen per shot."""
    name = "toon"

    def __init__(self, bands: int = 6, hues: int = 10, saturation: float = 1.05, ink: float = 1.0,
                 hysteresis: float = 7.0, line_detail: float = 1.5, smooth: float = 1.0,
                 softness: float = 0.3, hue_flatness: float = 0.75):
        self.kL, self.kab = bands, hues
        self.sat = saturation
        self.ink_strength = ink
        self.hyst = hysteresis
        self.line_sigma = 1.2 / max(line_detail, 0.2)
        self.smooth_s = 50 * smooth
        self.soft = softness          # keep a little of the real shading inside each band
        self.flat = hue_flatness      # 1 = fully flat hues, 0 = original hues
        self.reset()

    def reset(self):
        self.L_centers = self.ab_centers = None
        self.lut_L = self.lut_ab = None
        self.prev_L = self.prev_ab = self.prev_ink = None

    # -- colour -------------------------------------------------------------
    def _smooth(self, frame):
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
        small = cv2.edgePreservingFilter(small, flags=cv2.RECURS_FILTER, sigma_s=self.smooth_s, sigma_r=0.4)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if self.sat != 1.0:
            hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] = np.clip(hsv[..., 1] * self.sat, 0, 255)
            out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out

    @staticmethod
    def _kmeans(x, k):
        x = x.astype(np.float32)
        idx = np.random.default_rng(0).choice(len(x), size=min(40000, len(x)), replace=False)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, _, c = cv2.kmeans(x[idx], k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        return c

    def _build_palette(self, lab):
        Lc = np.sort(self._kmeans(lab[..., 0].reshape(-1, 1), self.kL)[:, 0])
        self.L_centers = Lc
        self.lut_L = np.abs(np.arange(256, dtype=np.float32)[:, None] - Lc[None]).argmin(1).astype(np.uint8)
        abc = self._kmeans(lab[..., 1:].reshape(-1, 2), self.kab)
        self.ab_centers = abc
        g = np.arange(32, dtype=np.float32) * 8 + 4
        grid = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        self.lut_ab = ((grid[:, None] - abc[None]) ** 2).sum(-1).argmin(1).astype(np.uint8).reshape(32, 32)

    def _hyst(self, value, labels, prev, centers):
        """Keep last frame's label if it is still (almost) as good -> no flicker."""
        if prev is None:
            return labels
        d_cur = np.linalg.norm(value - centers[labels], axis=-1) if value.ndim == 3 else np.abs(value - centers[labels])
        d_prev = np.linalg.norm(value - centers[prev], axis=-1) if value.ndim == 3 else np.abs(value - centers[prev])
        return np.where(d_prev <= d_cur + self.hyst, prev, labels)

    # -- main ---------------------------------------------------------------
    def process(self, frame: np.ndarray, flow: FlowTracker) -> np.ndarray:
        if flow.is_cut:
            self.reset()
        sm = self._smooth(frame)
        lab = cv2.cvtColor(sm, cv2.COLOR_BGR2Lab)
        if self.L_centers is None:
            self._build_palette(lab)
        L = lab[..., 0]
        ab = lab[..., 1:]
        lL = self.lut_L[L]
        lab_ = self.lut_ab[ab[..., 0] >> 3, ab[..., 1] >> 3]
        pL = flow.warp(self.prev_L, nearest=True) if self.prev_L is not None else None
        pab = flow.warp(self.prev_ab, nearest=True) if self.prev_ab is not None else None
        lL = self._hyst(L.astype(np.float32), lL, pL, self.L_centers)
        lab_ = self._hyst(ab.astype(np.float32), lab_, pab, self.ab_centers)
        self.prev_L, self.prev_ab = lL, lab_
        Lf = self.L_centers[lL] + self.soft * (L.astype(np.float32) - self.L_centers[lL])
        abf = self.flat * self.ab_centers[lab_] + (1 - self.flat) * ab.astype(np.float32)
        flat = np.clip(np.dstack([Lf * 1.04 + 5, abf]), 0, 255).astype(np.uint8)
        color = cv2.medianBlur(cv2.cvtColor(flat, cv2.COLOR_Lab2BGR), 5)

        gray = cv2.cvtColor(cv2.bilateralFilter(frame, 7, 30, 5), cv2.COLOR_BGR2GRAY)
        ink = xdog(gray, sigma=self.line_sigma)
        prev_ink = flow.warp(self.prev_ink) if self.prev_ink is not None else None
        if prev_ink is not None:
            ink = 0.55 * ink + 0.45 * prev_ink
        self.prev_ink = ink
        return apply_ink(color, clean(ink, 0.25, 12), strength=self.ink_strength)
