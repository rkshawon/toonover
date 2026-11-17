"""Light AI styles: small image-to-image cartoon models (ONNX, CPU, ~8.7 MB each).

Bundled models are AnimeGANv2 generators from bryandlee/animegan2-pytorch (MIT),
converted to ONNX. Drop any compatible model (RGB in [-1,1] -> RGB in [-1,1]) into
models/styles/ and register it in STYLE_FILES.

Speed tricks: the model runs at a reduced size, and by default only on every 2nd
frame ('animated on twos', like real 2D cartoons); in-between frames are made by
warping with optical flow. Results are blended over time so they don't flicker.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .ink import apply_ink, clean, xdog
from .temporal import FlowTracker


class NeuralStylizer:
    def __init__(self, model_path: Path, size: int = 512, every: int = 2,
                 ink: float = 0.6, temporal: float = 0.5, threads: int | None = None):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads or max(1, (os.cpu_count() or 2))
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.name = Path(model_path).stem
        self.size = size
        self.every = max(1, every)
        self.ink = ink
        self.temporal = temporal
        self.reset()

    def reset(self):
        self.prev_out = None
        self.prev_src = None
        self.n = 0

    def _run(self, frame):
        h, w = frame.shape[:2]
        s = self.size / min(h, w)
        nh, nw = max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8)
        x = cv2.cvtColor(cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        x = (x.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.inp: x})[0][0].transpose(1, 2, 0)
        y = ((np.clip(y, -1, 1) + 1) * 127.5).astype(np.uint8)
        return cv2.resize(cv2.cvtColor(y, cv2.COLOR_RGB2BGR), (w, h), interpolation=cv2.INTER_CUBIC)

    def process(self, frame: np.ndarray, flow: FlowTracker) -> np.ndarray:
        if flow.is_cut:
            self.reset()
        warped = flow.warp(self.prev_out) if self.prev_out is not None else None
        if warped is None or self.n % self.every == 0:
            out = self._run(frame)
            if warped is not None and self.temporal > 0:
                m = flow.stable_mask(frame, self.prev_src)
                if m is not None:
                    a = (m * self.temporal)[..., None]
                    out = (out.astype(np.float32) * (1 - a) + warped.astype(np.float32) * a).astype(np.uint8)
        else:
            out = warped
        self.prev_out, self.prev_src = out, frame
        self.n += 1
        if self.ink > 0:
            gray = cv2.cvtColor(cv2.bilateralFilter(frame, 7, 30, 5), cv2.COLOR_BGR2GRAY)
            out = apply_ink(out, clean(xdog(gray, sigma=0.9), 0.3, 15), strength=self.ink)
        return out
