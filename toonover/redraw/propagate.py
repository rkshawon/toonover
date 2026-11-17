"""Keyframe redraw + motion propagation (EbSynth-style, no extra AI).

Only every Nth frame (and every shot start/end) is redrawn by the image model.
Frames in between are made by carrying the redrawn keyframes along the real
motion (optical flow) - forwards from the previous key and backwards from the
next key - then blending by distance and confidence. Result: a fully redrawn
cartoon that moves exactly like the original, for ~1/10 of the GPU work.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

_DIS = None


def _dis():
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _DIS


def _small_gray(f, scale):
    return cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


class _Grid:
    def __init__(self, h, w):
        self.gx, self.gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def _map(cur_small, other_small, grid: _Grid, scale: float):
    """Remap maps that pull pixels of `other` onto `cur`."""
    flow = _dis().calc(cur_small, other_small, None)
    h, w = grid.gx.shape
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    return grid.gx + flow[..., 0], grid.gy + flow[..., 1]


def _remap(img, m):
    return cv2.remap(img, m[0], m[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _match(cur, warped_src, tol=22.0):
    d = cv2.absdiff(cv2.GaussianBlur(cur, (5, 5), 0), cv2.GaussianBlur(warped_src, (5, 5), 0)).max(axis=2)
    return np.clip(1.0 - d.astype(np.float32) / tol, 0, 1)


def find_cuts(frames_iter, threshold: float = 0.45) -> tuple[list[int], int]:
    """Return (shot start indices, total frames)."""
    prev = None
    cuts = [0]
    n = 0
    for f in frames_iter:
        hsv = cv2.cvtColor(cv2.resize(f, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
        cv2.normalize(h, h)
        if prev is not None and cv2.compareHist(prev, h, cv2.HISTCMP_BHATTACHARYYA) > threshold and n - cuts[-1] > 2:
            cuts.append(n)
        prev = h
        n += 1
    return cuts, n


def plan_keyframes(cuts: list[int], total: int, every: int) -> list[tuple[int, int]]:
    """List of (start, end) inclusive segments; both ends are keyframes. Segments never cross a cut."""
    segs = []
    bounds = cuts + [total]
    for s, e in zip(bounds[:-1], bounds[1:]):
        last = e - 1
        if last <= s:
            segs.append((s, s))
            continue
        n = max(1, round((last - s) / every))
        keys = sorted(set(int(round(s + i * (last - s) / n)) for i in range(n + 1)))
        segs += list(zip(keys[:-1], keys[1:]))
    return segs


def propagate_video(read_frame: Callable[[int], np.ndarray], segments: list[tuple[int, int]],
                    redraw: Callable[[np.ndarray], np.ndarray], fallback: Callable[[np.ndarray], np.ndarray],
                    write: Callable[[np.ndarray], None], progress=None, flow_scale: float = 0.5):
    """read_frame(i) must support sequential access. Calls write() once per frame, in order."""
    key_cache: dict[int, np.ndarray] = {}
    grid = None
    total_keys = len({k for s in segments for k in s})
    done_keys = 0

    def key(i, frame):
        nonlocal done_keys
        if i not in key_cache:
            key_cache.clear() if len(key_cache) > 2 else None
            key_cache[i] = redraw(frame)
            done_keys += 1
            if progress:
                progress(done_keys / max(total_keys, 1))
        return key_cache[i]

    for si, (a, b) in enumerate(segments):
        frames = [read_frame(i) for i in range(a, b + 1)]
        if grid is None or grid.gx.shape != frames[0].shape[:2]:
            grid = _Grid(*frames[0].shape[:2])
        Sa = key(a, frames[0])
        last_in_shot = si + 1 == len(segments) or segments[si + 1][0] != b
        if a == b:
            write(Sa)
            continue
        Sb = key(b, frames[-1])
        L = len(frames)
        small = [_small_gray(f, flow_scale) for f in frames]

        # forward chain from key a
        Wf, cf = [Sa], [np.ones(frames[0].shape[:2], np.float32)]
        for j in range(1, L):
            m = _map(small[j], small[j - 1], grid, flow_scale)
            Wf.append(_remap(Wf[-1], m))
            cf.append(_remap(cf[-1], m) * _match(frames[j], _remap(frames[j - 1], m)))
        # backward chain from key b
        Wb, cb = [None] * L, [None] * L
        Wb[-1], cb[-1] = Sb, np.ones(frames[0].shape[:2], np.float32)
        for j in range(L - 2, -1, -1):
            m = _map(small[j], small[j + 1], grid, flow_scale)
            Wb[j] = _remap(Wb[j + 1], m)
            cb[j] = _remap(cb[j + 1], m) * _match(frames[j], _remap(frames[j + 1], m))

        end = L if last_in_shot else L - 1      # key b is written by the next segment
        for j in range(end):
            if j == 0:
                write(Sa)
                continue
            if j == L - 1:
                write(Sb)
                continue
            w = j / (L - 1)
            wa = cv2.GaussianBlur((1 - w) * cf[j], (9, 9), 0)[..., None]
            wb = cv2.GaussianBlur(w * cb[j], (9, 9), 0)[..., None]
            den = wa + wb
            out = (Wf[j].astype(np.float32) * wa + Wb[j].astype(np.float32) * wb) / np.maximum(den, 1e-4)
            conf = np.clip(den / 0.35, 0, 1)            # both chains lost track -> disoccluded area
            if conf.min() < 0.95:
                fb = fallback(frames[j]).astype(np.float32)
                out = out * conf + fb * (1 - conf)
            write(np.clip(out, 0, 255).astype(np.uint8))
