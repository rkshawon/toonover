"""Pass 1: watch the real video and record how each person moves.

Uses MediaPipe Face Landmarker (+ optional Pose Landmarker). Output is plain
numbers per frame (head position/tilt, mouth open, smile, brows, blink, arms)
which the cartoon renderer later copies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np

# Face mesh landmark ids
FOREHEAD, CHIN, NOSE = 10, 152, 1
CHEEK_R, CHEEK_L = 234, 454          # image-left / image-right side of face
LIP_UP, LIP_DN, MOUTH_L, MOUTH_R = 13, 14, 61, 291
EYE_R = (159, 145, 33, 133)          # top, bottom, outer, inner (image-left eye)
EYE_L = (386, 374, 263, 362)         # image-right eye
BROW_R, BROW_L = 105, 334


@dataclass
class Det:
    cx: float; cy: float; size: float      # normalized to frame height
    yaw: float; pitch: float; roll: float
    mouth_open: float; mouth_wide: float; smile: float
    eye_l: float; eye_r: float; brow: float
    arms: dict | None = None                 # {"L": (upper, fore), "R": (...)} screen-space radians
    track: int = -1


@dataclass
class TrackResult:
    fps: float
    width: int
    height: int
    frames: list[list[Det]] = field(default_factory=list)
    cuts: list[int] = field(default_factory=list)


def _p(lms, i, w, h):
    l = lms[i]
    return np.array([l.x * w, l.y * h], dtype=np.float32)


def face_features(lms, w: int, h: int) -> Det:
    P = lambda i: _p(lms, i, w, h)
    top, chin, nose = P(FOREHEAD), P(CHIN), P(NOSE)
    cr, cl = P(CHEEK_R), P(CHEEK_L)
    fh = float(np.linalg.norm(chin - top)) + 1e-6
    fw = float(np.linalg.norm(cl - cr)) + 1e-6

    mid_x = (cr + cl) / 2
    mid_y = (top + chin) / 2
    yaw = float(np.clip((nose[0] - mid_x[0]) / fw * 2.2, -1, 1))
    pitch = float(np.clip((nose[1] - mid_y[1]) / fh * 3.0, -1, 1))
    er_o, el_o = P(EYE_R[2]), P(EYE_L[2])
    roll = float(math.atan2(el_o[1] - er_o[1], el_o[0] - er_o[0]))

    up, dn, ml, mr = P(LIP_UP), P(LIP_DN), P(MOUTH_L), P(MOUTH_R)
    mouth_open = float(np.linalg.norm(dn - up) / fh)
    mouth_wide = float(np.linalg.norm(mr - ml) / fw)
    centre_y = (up[1] + dn[1]) / 2
    # corners above mouth centre => smile (rotate-invariant enough for memes)
    smile = float(((centre_y - (ml[1] + mr[1]) / 2) / fh))

    def eye(ids):
        t, b, o, i = (P(k) for k in ids)
        return float(np.linalg.norm(b - t) / (np.linalg.norm(o - i) + 1e-6))

    brow = float(((P(EYE_R[0])[1] - P(BROW_R)[1]) + (P(EYE_L[0])[1] - P(BROW_L)[1])) / 2 / fh)

    return Det(cx=float(nose[0] / h), cy=float(nose[1] / h), size=fh / h,
               yaw=yaw, pitch=pitch, roll=roll,
               mouth_open=mouth_open, mouth_wide=mouth_wide, smile=smile,
               eye_l=eye(EYE_L), eye_r=eye(EYE_R), brow=brow)


def pose_arms(lms, w: int, h: int) -> tuple[np.ndarray, dict]:
    """Return (nose_xy_norm_by_h, arms) with screen-space angles."""
    P = lambda i: np.array([lms[i].x * w, lms[i].y * h])
    vis = lambda i: getattr(lms[i], "visibility", 1.0) or 0.0
    arms = {}
    sh = [(11, 13, 15), (12, 14, 16)]
    # screen-left arm = the shoulder with smaller x
    sh.sort(key=lambda t: lms[t[0]].x)
    for side, (s, e, wr) in zip(("L", "R"), sh):
        if min(vis(s), vis(e), vis(wr)) < 0.5:
            continue
        S, E, W = P(s), P(e), P(wr)
        up = math.atan2(E[1] - S[1], E[0] - S[0])
        fo = math.atan2(W[1] - E[1], W[0] - E[0])
        arms[side] = (up, fo)
    return P(0) / h, arms


class _Tracker:
    """Greedy nearest-neighbour ID assignment, reset on scene cuts."""

    def __init__(self):
        self.next_id = 0
        self.last: dict[int, tuple[float, float, int]] = {}  # id -> (cx, cy, frame)

    def reset(self):
        self.last.clear()

    def assign(self, dets: list[Det], frame: int):
        used = set()
        pairs = []
        for di, d in enumerate(dets):
            for tid, (x, y, f) in self.last.items():
                if frame - f > 15:
                    continue
                dist = math.hypot(d.cx - x, d.cy - y)
                if dist < max(0.12, d.size * 0.8):
                    pairs.append((dist, di, tid))
        pairs.sort()
        taken_d = set()
        for dist, di, tid in pairs:
            if di in taken_d or tid in used:
                continue
            dets[di].track = tid
            taken_d.add(di); used.add(tid)
        for d in dets:
            if d.track < 0:
                d.track = self.next_id
                self.next_id += 1
            self.last[d.track] = (d.cx, d.cy, frame)


def _is_cut(prev_hist, frame) -> tuple[bool, np.ndarray]:
    hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    if prev_hist is None:
        return False, hist
    score = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
    return score > 0.45, hist


def track_video(video_path, models: dict, max_people: int = 4, progress=None) -> TrackResult:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    detect_faces, detect_poses, close = _backend(models, max_people)

    res = TrackResult(fps=fps, width=w, height=h)
    tracker = _Tracker()
    prev_hist = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cut, prev_hist = _is_cut(prev_hist, frame)
        if cut:
            res.cuts.append(i)
            tracker.reset()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int(i * 1000 / fps)
        dets = [face_features(lms, w, h) for lms in detect_faces(rgb, ts)]
        dets = [d for d in dets if d.size > 0.06]  # ignore tiny background faces
        if dets:
            for lms in detect_poses(rgb, ts, len(dets)):
                nose, arms = pose_arms(lms, w, h)
                best = min(dets, key=lambda d: math.hypot(d.cx - nose[0], d.cy - nose[1]))
                if math.hypot(best.cx - nose[0], best.cy - nose[1]) < best.size:
                    best.arms = arms
        tracker.assign(dets, i)
        res.frames.append(dets)
        i += 1
        if progress and i % 10 == 0:
            progress(min(i / total, 1.0))
    cap.release()
    close()
    return res


def _backend(models: dict, max_people: int):
    """Pick MediaPipe Tasks (current) or legacy Solutions (older mediapipe, no downloads needed).

    Returns (detect_faces(rgb, ts) -> [landmarks], detect_poses(rgb, ts, n_faces) -> [landmarks], close()).
    """
    import os

    use_legacy = os.environ.get("TOONOVER_BACKEND") == "legacy" or not models.get("face_landmarker.task")
    if use_legacy:
        if not hasattr(mp, "solutions"):
            raise RuntimeError("Face model missing - run once with internet so it can be downloaded.")
        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=max_people,
                                             min_detection_confidence=0.5, min_tracking_confidence=0.5)
        ps = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

        def faces(rgb, ts):
            r = fm.process(rgb)
            return [f.landmark for f in (r.multi_face_landmarks or [])]

        def poses(rgb, ts, n):
            if n != 1:      # legacy pose tracks a single person only
                return []
            r = ps.process(rgb)
            return [r.pose_landmarks.landmark] if r.pose_landmarks else []

        return faces, poses, lambda: (fm.close(), ps.close())

    from mediapipe.tasks.python import BaseOptions, vision

    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(models["face_landmarker.task"])),
        running_mode=vision.RunningMode.VIDEO, num_faces=max_people,
        min_face_detection_confidence=0.5, min_tracking_confidence=0.5))
    pose = None
    if models.get("pose_landmarker_full.task"):
        try:
            pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(models["pose_landmarker_full.task"])),
                running_mode=vision.RunningMode.VIDEO, num_poses=max_people))
        except Exception:
            pose = None

    def faces(rgb, ts):
        return face.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts).face_landmarks

    def poses(rgb, ts, n):
        if pose is None:
            return []
        return pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts).pose_landmarks

    def close():
        face.close()
        if pose:
            pose.close()

    return faces, poses, close
