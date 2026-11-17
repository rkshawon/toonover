"""Real -> cartoon conversion: the same video, same people, same motion, drawn as a cartoon."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import OUTPUT, WORK
from .ingest import fetch, normalize, require_ffmpeg
from .render import VideoWriter, _text_block
from .stylize import make_stylizer
from .stylize.temporal import FlowTracker


def _layout(w: int, h: int, title: str, channel: str):
    """Black title / channel bars like popular cartoon Shorts. Returns (template, y_offset)."""
    if not title and not channel:
        return None, 0
    top = (int(h * 0.12) // 2) * 2 if title else 0
    bot = (int(h * 0.08) // 2) * 2 if channel else 0
    img = Image.new("RGB", (w, h + top + bot), (0, 0, 0))
    d = ImageDraw.Draw(img)
    if title:
        _text_block(d, title, (16, 0, w - 16, top), int(top * 0.38))
    if channel:
        _text_block(d, channel, (16, h + top, w - 16, h + top + bot), int(bot * 0.45))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), top


def convert_video(source: str, style: str = "toon", out_path: Path | None = None, title: str = "",
                  channel: str = "", max_seconds: float = 90, max_fps: float = 30,
                  ai_size: int = 512, ai_every: int = 2, auto_title: bool = False,
                  log=print, progress=None, job_id: str | None = None) -> Path:
    prog = progress or (lambda s, f: None)
    require_ffmpeg()
    job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = WORK / job_id
    job.mkdir(parents=True, exist_ok=True)

    prog("download", 0.0)
    src = fetch(source, job, log)
    if auto_title and not title:
        info = job / "source_info.json"
        if info.exists():
            title = (json.loads(info.read_text(encoding="utf-8")).get("title") or "")[:70]

    prog("prepare", 0.0)
    video, audio, fps = normalize(src, job, max_fps, max_seconds, log)

    log(f"Style: {style}")
    stylizer = make_stylizer(style, ai_size=ai_size, ai_every=ai_every)
    flow = FlowTracker()

    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    template, y0 = _layout(w, h, title, channel)
    ow, oh = (w, h) if template is None else (template.shape[1], template.shape[0])

    out_path = out_path or (OUTPUT / f"toonover_{job_id}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = VideoWriter(out_path, ow, oh, fps, audio)
    log(f"Cartoonizing {total} frames at {fps:g} fps")
    t0 = time.time()
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            flow.update(frame)
            toon = stylizer.process(frame, flow)
            if template is not None:
                canvas = template.copy()
                canvas[y0:y0 + h] = toon
                toon = canvas
            vw.write(toon)
            i += 1
            if i % 5 == 0:
                prog("cartoonize", i / total)
    finally:
        cap.release()
        vw.close()
    prog("done", 1.0)
    log(f"Done in {time.time() - t0:.0f}s ({(time.time() - t0) / max(i, 1) * 1000:.0f} ms/frame) -> {out_path}")
    return out_path


class _SeqReader:
    """Sequential frame reader that can return the same frame twice (segment boundaries)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.i = -1
        self.f = None

    def __call__(self, i: int):
        while self.i < i:
            ok, f = self.cap.read()
            if not ok:
                break
            self.f, self.i = f, self.i + 1
        return self.f


def convert_redraw(source: str, style: str, out_path: Path | None = None, title: str = "", channel: str = "",
                   max_seconds: float = 90, max_fps: float = 30, keyframe_every: int | None = None,
                   comfy_url: str | None = None, redrawer=None, auto_title: bool = False, describe: str = "",
                   log=print, progress=None, job_id: str | None = None) -> Path:
    """GPU redraw styles: keyframes redrawn by ComfyUI, in-betweens carried along the motion."""
    from .redraw.comfy import DEFAULT_URL, ComfyRedrawer
    from .redraw.propagate import find_cuts, plan_keyframes, propagate_video

    prog = progress or (lambda s, f: None)
    require_ffmpeg()
    job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = WORK / job_id
    job.mkdir(parents=True, exist_ok=True)
    if redrawer is None:
        log(f"Connecting to ComfyUI at {comfy_url or DEFAULT_URL}")
        redrawer = ComfyRedrawer(style, comfy_url or DEFAULT_URL, describe=describe)
    every = keyframe_every or (redrawer.p.get("keyframe_every", 10) if hasattr(redrawer, "p") else 10)

    prog("download", 0.0)
    src = fetch(source, job, log)
    if auto_title and not title:
        info = job / "source_info.json"
        if info.exists():
            title = (json.loads(info.read_text(encoding="utf-8")).get("title") or "")[:70]
    prog("prepare", 0.0)
    video, audio, fps = normalize(src, job, max_fps, max_seconds, log)

    def frames():
        cap = cv2.VideoCapture(str(video))
        while True:
            ok, f = cap.read()
            if not ok:
                break
            yield f
        cap.release()

    cuts, total = find_cuts(frames())
    segs = plan_keyframes(cuts, total, every)
    n_keys = len({k for s in segs for k in s})
    log(f"{total} frames, {len(cuts)} shots, {n_keys} keyframes to redraw (every {every} frames)")

    cap = cv2.VideoCapture(str(video))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    template, y0 = _layout(w, h, title, channel)
    ow, oh = (w, h) if template is None else (template.shape[1], template.shape[0])
    out_path = out_path or (OUTPUT / f"toonover_{job_id}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = VideoWriter(out_path, ow, oh, fps, audio)

    post = None
    if getattr(redrawer, "p", {}).get("post") == "vintage":
        from .stylize.film import vintage
        post = vintage
    count = {"i": 0}

    def write(img):
        if post is not None:
            img = post(img, count["i"])
            count["i"] += 1
        if template is not None:
            canvas = template.copy()
            canvas[y0:y0 + h] = img
            img = canvas
        vw.write(img)

    t0 = time.time()
    try:
        propagate_video(_SeqReader(video), segs, redrawer, None, write,
                        progress=lambda f: prog("redraw", f))
    finally:
        vw.close()
    prog("done", 1.0)
    log(f"Done in {time.time() - t0:.0f}s -> {out_path}")
    return out_path


def convert_image(path: Path, style: str, out_path: Path, ai_size: int = 512, describe: str = "") -> Path:
    frame = cv2.imread(str(path))
    if frame is None:
        raise FileNotFoundError(path)
    from .redraw.comfy import presets
    if style in presets():
        from .redraw.comfy import ComfyRedrawer
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rd = ComfyRedrawer(style, describe=describe)
        img = rd(frame)
        if rd.p.get("post") == "vintage":
            from .stylize.film import vintage
            img = vintage(img, 0, weave=False)
        cv2.imwrite(str(out_path), img)
        return out_path
    st = make_stylizer(style, ai_size=ai_size, ai_every=1)
    fl = FlowTracker()
    fl.update(frame)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), st.process(frame, fl))
    return out_path
