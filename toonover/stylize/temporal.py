"""Frame-to-frame coherence: optical flow (classic DIS, no AI) + scene-cut detection.

Stylizing every frame independently makes colours and lines 'boil'. Instead we
warp the previous stylized result onto the current frame and only change pixels
that really changed. This is what makes the cartoon look smooth.
"""
from __future__ import annotations

import cv2
import numpy as np


class FlowTracker:
    def __init__(self, scale: float = 0.5, cut_threshold: float = 0.45):
        self.scale = scale
        self.cut_threshold = cut_threshold
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev_small: np.ndarray | None = None
        self.prev_hist = None
        self.map_x = self.map_y = None
        self.is_cut = True
        self.frame_index = -1

    def _hist(self, frame):
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    def update(self, frame: np.ndarray) -> bool:
        """Feed the next frame. Returns True if this frame starts a new shot."""
        self.frame_index += 1
        h, w = frame.shape[:2]
        hist = self._hist(frame)
        cut = self.prev_hist is None or cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > self.cut_threshold
        self.prev_hist = hist
        small = cv2.cvtColor(cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY)
        if cut or self.prev_small is None:
            self.map_x = self.map_y = None
        else:
            # backward flow: for each pixel of the current frame, where was it in the previous one
            flow = self.dis.calc(small, self.prev_small, None)
            flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / self.scale
            if getattr(self, "_grid_shape", None) != (h, w):
                self._gx, self._gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
                self._grid_shape = (h, w)
            self.map_x = self._gx + flow[..., 0]
            self.map_y = self._gy + flow[..., 1]
        self.prev_small = small
        self.is_cut = cut
        return cut

    def warp(self, prev: np.ndarray, nearest: bool = False) -> np.ndarray | None:
        """Warp an image aligned to the previous frame onto the current frame."""
        if self.map_x is None or prev is None:
            return None
        return cv2.remap(prev, self.map_x, self.map_y,
                         cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def stable_mask(self, cur: np.ndarray, prev_src: np.ndarray, tol: float = 18.0) -> np.ndarray | None:
        """0..1 mask: 1 where the warped previous source frame matches the current one (safe to reuse)."""
        w = self.warp(prev_src)
        if w is None:
            return None
        diff = cv2.absdiff(cv2.GaussianBlur(cur, (5, 5), 0), cv2.GaussianBlur(w, (5, 5), 0)).max(axis=2).astype(np.float32)
        m = np.clip(1.0 - diff / tol, 0, 1)
        return cv2.GaussianBlur(m, (7, 7), 0)
