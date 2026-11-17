"""End-to-end: link/file -> tracked performance -> cartoon video."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from .character import load_cast
from .config import ASSETS, OUTPUT, WORK, RenderSettings
from .director import choose_slot_count, direct
from .ingest import fetch, normalize, require_ffmpeg
from .models import ensure_models
from .render import Renderer, VideoWriter
from .track import track_video


def make_cartoon(source: str, rs: RenderSettings, out_path: Path | None = None,
                 log=print, progress=None, job_id: str | None = None) -> Path:
    """progress(stage: str, fraction: float) is called as work happens."""
    prog = progress or (lambda s, f: None)
    require_ffmpeg()
    job_id = job_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    job = WORK / job_id
    job.mkdir(parents=True, exist_ok=True)

    prog("download", 0.0)
    src = fetch(source, job, log)
    if not rs.title:
        info = job / "source_info.json"
        if info.exists():
            rs.title = (json.loads(info.read_text(encoding="utf-8")).get("title") or "")[:70]

    prog("prepare", 0.0)
    video, audio, fps = normalize(src, job, rs.max_fps, rs.max_seconds, log)

    log("Loading tracking models")
    models = ensure_models(log)

    log("Tracking faces & bodies")
    tr = track_video(video, models, max_people=rs.max_characters,
                     progress=lambda f: prog("track", f))
    log(f"  {len(tr.frames)} frames, {len(tr.cuts)} scene cuts")

    cast = load_cast()
    order = json.loads((ASSETS / "cast.json").read_text(encoding="utf-8"))["default_order"]
    k = choose_slot_count(tr, rs.max_characters, len(rs.cast) or None)
    ids = (rs.cast or order)[:k]
    while len(ids) < k:
        ids.append(order[len(ids) % len(order)])
    specs = [cast[i] for i in ids]
    log(f"Cast: {', '.join(s.name for s in specs)}")

    perf, cams = direct(tr, k, rs.camera_moves)

    out_path = out_path or (OUTPUT / f"toonover_{job_id}.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = Renderer(rs, specs)
    vw = VideoWriter(out_path, rs.width, rs.height, tr.fps, audio, rs.crf)
    n = len(perf)
    log("Rendering cartoon")
    try:
        for i, (poses, cam) in enumerate(zip(perf, cams)):
            vw.write(r.frame(poses, cam, i / tr.fps))
            if i % 10 == 0:
                prog("render", i / max(n, 1))
    finally:
        vw.close()
    prog("done", 1.0)
    log(f"Saved {out_path}")
    return out_path
