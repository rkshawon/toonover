"""Title-bar text and the ffmpeg video writer."""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ASSETS

FONT_CANDIDATES = [
    *sorted(str(p) for p in (ASSETS / "fonts").glob("*.ttf")),
    "C:/Windows/Fonts/seguibl.ttf", "C:/Windows/Fonts/ariblk.ttf", "C:/Windows/Fonts/arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Black.ttf",
]


def _font(size: int):
    for f in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default(size=size)


def _text_block(draw: ImageDraw.ImageDraw, text: str, box: tuple[int, int, int, int], max_size: int):
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    for size in range(max_size, 14, -2):
        font = _font(size)
        for width in (40, 28, 22, 16):
            lines = textwrap.wrap(text, width=width) or [""]
            if len(lines) > 2:
                continue
            widths = [draw.textlength(l, font=font) for l in lines]
            lh = size * 1.15
            if max(widths) <= bw * 0.94 and lh * len(lines) <= bh * 0.9:
                y = y0 + (bh - lh * len(lines)) / 2
                for l, lw_ in zip(lines, widths):
                    draw.text((x0 + (bw - lw_) / 2, y), l, font=font, fill=(255, 255, 255),
                              stroke_width=max(1, size // 18), stroke_fill=(0, 0, 0))
                    y += lh
                return


class VideoWriter:
    """Pipe raw frames into ffmpeg (H.264) and mux the original audio."""

    def __init__(self, path: Path, w: int, h: int, fps: float, audio: Path | None, crf: int = 20):
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-"]
        if audio:
            cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "160k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(path)]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray):
        self.p.stdin.write(frame.tobytes())

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding")
