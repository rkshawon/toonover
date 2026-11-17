"""Turn raw tracking into a smooth 'performance' for each cartoon character.

- decides how many characters appear and who plays whom (left -> right)
- normalizes each person's face numbers against their own neutral face
- smooths everything (One Euro) so the cartoon moves fluidly
- detects who is talking, and plans gentle camera zooms
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .smooth import OneEuro, Smoother
from .track import Det, TrackResult


@dataclass
class Pose:
    present: float = 0.0       # 0..1 fade (1 = tracked person visible)
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    dx: float = 0.0            # body lean, -1..1
    dy: float = 0.0            # head bob, -1..1
    mouth_open: float = 0.0    # 0..1
    mouth_wide: float = 0.0    # -1..1
    smile: float = 0.0         # -1..1
    eye: float = 1.0           # 0 closed .. 1 open .. 1.3 wide
    brow: float = 0.0          # -1 frown .. 1.5 raised
    talk: float = 0.0          # 0..1 speaking activity
    arm_L: tuple[float, float] | None = None
    arm_R: tuple[float, float] | None = None


@dataclass
class Camera:
    zoom: float = 1.0
    focus: float = 0.5   # 0..1 horizontal focus in the scene


def _clamp(v, a, b):
    return a if v < a else b if v > b else v


def choose_slot_count(tr: TrackResult, max_chars: int, forced: int | None) -> int:
    if forced:
        return max(1, min(forced, 6))
    counts = Counter(len(f) for f in tr.frames)
    with_faces = sum(c for n, c in counts.items() if n > 0) or 1
    k = 1
    for n in range(1, max_chars + 1):
        if sum(c for m, c in counts.items() if m >= n) / with_faces >= 0.05:
            k = n
    return k


def assign_slots(tr: TrackResult, k: int) -> dict[int, int]:
    """Map track id -> slot index (0 = leftmost)."""
    # slot centres from frames where exactly k people are visible
    ranks = defaultdict(list)
    for dets in tr.frames:
        if len(dets) == k:
            for r, d in enumerate(sorted(dets, key=lambda d: d.cx)):
                ranks[r].append(d.cx)
    aspect = tr.width / max(tr.height, 1)
    if len(ranks) == k:
        centres = [float(np.median(ranks[r])) for r in range(k)]
    else:
        centres = [aspect * (r + 1) / (k + 1) for r in range(k)]

    votes: dict[int, Counter] = defaultdict(Counter)
    for dets in tr.frames:
        if len(dets) == k:
            for r, d in enumerate(sorted(dets, key=lambda d: d.cx)):
                votes[d.track][r] += 3
        for d in dets:
            nearest = min(range(k), key=lambda s: abs(centres[s] - d.cx))
            votes[d.track][nearest] += 1
    return {t: c.most_common(1)[0][0] for t, c in votes.items()}


def direct(tr: TrackResult, k: int, camera_moves: bool = True) -> tuple[list[list[Pose]], list[Camera]]:
    fps = tr.fps or 30.0
    dt = 1.0 / fps
    slot_of = assign_slots(tr, k)

    # per-track neutral baseline = median of that person's values
    per_track: dict[int, list[Det]] = defaultdict(list)
    for dets in tr.frames:
        for d in dets:
            per_track[d.track].append(d)
    base = {}
    for t, ds in per_track.items():
        arr = lambda a: float(np.median([getattr(d, a) for d in ds]))
        base[t] = {a: arr(a) for a in ("cx", "cy", "size", "mouth_wide", "smile", "brow", "roll")}
        base[t]["eye"] = float(np.percentile([(d.eye_l + d.eye_r) / 2 for d in ds], 70)) or 0.3

    smoothers = [Smoother() for _ in range(k)]
    present_f = [OneEuro(2.0, 0.0) for _ in range(k)]
    out: list[list[Pose]] = []
    mouth_hist = [[] for _ in range(k)]
    win = max(3, int(fps * 0.4))

    for fi, dets in enumerate(tr.frames):
        by_slot: dict[int, Det] = {}
        for d in dets:
            s = slot_of.get(d.track, 0)
            if s not in by_slot or d.size > by_slot[s].size:
                by_slot[s] = d
        row = []
        for s in range(k):
            sm = smoothers[s]
            d = by_slot.get(s)
            if d is not None:
                b = base[d.track]
                raw = dict(
                    yaw=d.yaw, pitch=d.pitch, roll=d.roll - b["roll"],
                    dx=_clamp((d.cx - b["cx"]) / max(b["size"], 1e-3) * 0.6, -1, 1),
                    dy=_clamp((d.cy - b["cy"]) / max(b["size"], 1e-3) * 0.6, -1, 1),
                    mouth_open=_clamp((d.mouth_open - 0.012) / 0.09, 0, 1),
                    mouth_wide=_clamp((d.mouth_wide - b["mouth_wide"]) / 0.08, -1, 1),
                    smile=_clamp((d.smile - b["smile"]) / 0.025, -1, 1),
                    eye=_clamp(((d.eye_l + d.eye_r) / 2) / b["eye"], 0, 1.35),
                    brow=_clamp((d.brow - b["brow"]) / 0.035, -1, 1.5),
                )
                pres = 1.0
            else:
                raw = dict(yaw=0, pitch=0, roll=0, dx=0, dy=0, mouth_open=0, mouth_wide=0,
                           smile=0, eye=1.0, brow=0)
                pres = 0.0
            p = Pose(present=_clamp(present_f[s](pres, dt), 0, 1))
            for key, v in raw.items():
                setattr(p, key, sm(f"{key}.{s}", v, dt))
            if d is not None and d.arms:
                for side in ("L", "R"):
                    if side in d.arms:
                        up, fo = d.arms[side]
                        # smooth angles via unit vectors to avoid wrap-around jumps
                        ux = sm(f"au{side}x.{s}", math.cos(up), dt); uy = sm(f"au{side}y.{s}", math.sin(up), dt)
                        fx = sm(f"af{side}x.{s}", math.cos(fo), dt); fy = sm(f"af{side}y.{s}", math.sin(fo), dt)
                        setattr(p, f"arm_{side}", (math.atan2(uy, ux), math.atan2(fy, fx)))
            mh = mouth_hist[s]
            mh.append(p.mouth_open if d is not None else 0.0)
            if len(mh) > win:
                mh.pop(0)
            p.talk = _clamp(float(np.std(mh)) * 6.0, 0, 1)
            row.append(p)
        out.append(row)

    # camera plan
    cams: list[Camera] = []
    zf, ff = OneEuro(0.6, 0.0), OneEuro(0.6, 0.0)
    for fi, (dets, row) in enumerate(zip(tr.frames, out)):
        zoom, focus = 1.0, 0.5
        if camera_moves and k > 1:
            if len(dets) == 1 and dets[0].size > 0.30:          # source is a close-up
                s = slot_of.get(dets[0].track, 0)
                zoom, focus = 1.35, (s + 0.5) / k
            else:
                talker = max(range(k), key=lambda s: row[s].talk)
                if row[talker].talk > 0.35:
                    zoom, focus = 1.1, (talker + 0.5) / k
        cams.append(Camera(zoom=zf(zoom, dt), focus=ff(focus, dt)))
    return out, cams
