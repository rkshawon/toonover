"""Get a video (link or local file) and normalize it with ffmpeg."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"'{tool}' not found on PATH. On Windows run:  winget install Gyan.FFmpeg  "
                "then open a new terminal."
            )


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), re.I))


def download(url: str, job_dir: Path, log=print) -> Path:
    """Download with yt-dlp (YouTube, Facebook, Instagram, TikTok ...).

    Note: only use videos you have the right to use - platform terms usually
    forbid downloading other people's content.
    """
    import yt_dlp

    out_tmpl = str(job_dir / "source.%(ext)s")
    opts = {
        "outtmpl": out_tmpl,
        "format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
    }
    log(f"Downloading {url}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))
    if not path.exists():  # merged file may have a different extension
        cands = sorted(job_dir.glob("source.*"))
        if not cands:
            raise RuntimeError("Download failed")
        path = cands[0]
    (job_dir / "source_info.json").write_text(
        json.dumps({"title": info.get("title"), "url": url, "uploader": info.get("uploader")}, indent=2),
        encoding="utf-8",
    )
    return path


def probe(path: Path) -> dict:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(r.stdout)


def normalize(src: Path, job_dir: Path, max_fps: float, max_seconds: float, log=print) -> tuple[Path, Path | None, float]:
    """Make a clean constant-fps working copy + a wav of the audio.

    Returns (video_path, audio_path_or_None, fps).
    """
    info = probe(src)
    v = next(s for s in info["streams"] if s["codec_type"] == "video")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1) if float(num) else 30.0
    fps = min(round(fps, 3), max_fps)

    video = job_dir / "work.mp4"
    log(f"Normalizing video ({fps:g} fps)")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(max_seconds),
         "-vf", f"fps={fps},scale='if(gt(iw,ih),-2,min(720,iw))':'if(gt(iw,ih),min(720,ih),-2)'",
         "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", str(video)],
        check=True,
    )
    audio = None
    if has_audio:
        audio = job_dir / "audio.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", str(max_seconds),
             "-vn", "-ac", "1", "-ar", "44100", str(audio)],
            check=True,
        )
    return video, audio, fps


def fetch(source: str, job_dir: Path, log=print) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    if is_url(source):
        return download(source, job_dir, log)
    p = Path(source).expanduser()
    if not p.exists():
        raise FileNotFoundError(source)
    return p
