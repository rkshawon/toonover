"""Pass 2: draw the cartoon frame by frame and stream it into ffmpeg."""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .backgrounds import make_background
from .character import Character, Spec
from .config import ASSETS, RenderSettings
from .director import Camera, Pose

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


def make_template(rs: RenderSettings) -> np.ndarray:
    img = Image.new("RGB", (rs.width, rs.height), (0, 0, 0))
    d = ImageDraw.Draw(img)
    if rs.title:
        _text_block(d, rs.title, (20, 0, rs.width - 20, rs.top_bar), int(rs.top_bar * 0.38))
    if rs.channel_name:
        _text_block(d, rs.channel_name, (20, rs.height - rs.bottom_bar, rs.width - 20, rs.height),
                    int(rs.bottom_bar * 0.42))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


class Renderer:
    def __init__(self, rs: RenderSettings, specs: list[Spec]):
        self.rs = rs
        self.ss = rs.supersample
        self.sw = rs.width
        self.sh = rs.height - rs.top_bar - rs.bottom_bar
        W, H = self.sw * self.ss, self.sh * self.ss
        self.W, self.H = W, H
        self.floor_y = H * 0.93
        self.bg, self.shadow_col = make_background(rs.background, W, H, self.floor_y)
        self.template = make_template(rs)
        k = len(specs)
        self.chars = [Character(s, seed=i) for i, s in enumerate(specs)]
        # layout: spread characters across the stage
        margin = W * (0.25 if k <= 2 else 0.18 if k == 3 else 0.14)
        self.xs = [W / 2] if k == 1 else list(np.linspace(margin, W - margin, k))
        spacing = (W - 2 * margin) / max(k - 1, 1) if k > 1 else W
        U_h = H * 0.68 / 3.6
        U_w = min(spacing / 1.75, margin / 0.85)
        self.U = min(U_h, U_w) if k > 1 else U_h

    def frame(self, poses: list[Pose], cam: Camera, t: float) -> np.ndarray:
        img = self.bg.copy()
        for x, ch in zip(self.xs, self.chars):
            ch.shadow(img, x, self.floor_y, self.U * ch.s.height, self.shadow_col)
        for x, ch, p in zip(self.xs, self.chars, poses):
            ch.draw(img, x, self.floor_y, self.U, p, t)
        scene = cv2.resize(img, (self.sw, self.sh), interpolation=cv2.INTER_AREA)
        if cam.zoom > 1.01:
            z = cam.zoom
            fx = cam.focus * self.sw
            fy = self.sh * 0.5
            # keep the view inside the scene
            half_w, half_h = self.sw / (2 * z), self.sh / (2 * z)
            fx = min(max(fx, half_w), self.sw - half_w)
            fy = min(max(fy, half_h), self.sh - half_h)
            M = np.float32([[z, 0, self.sw / 2 - z * fx], [0, z, self.sh / 2 - z * fy]])
            scene = cv2.warpAffine(scene, M, (self.sw, self.sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out = self.template.copy()
        out[self.rs.top_bar:self.rs.top_bar + self.sh] = scene
        return out


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
