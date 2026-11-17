"""Real -> cartoon styles.

  cartoon   = anime+toon (recommended AI look)    cartoon-flat = celeba+toon
  toon      no AI. Cel-shaded rotoscope traced from the footage (fast, free, 100% recognizable)
  anime     light AI (AnimeGANv2 face_paint v2): soft hand-painted cartoon look
  paint     light AI (face_paint v1): stronger, more stylized faces
  paprika   light AI: bold colourful anime film look
  celeba    light AI: clean, smooth cartoon faces
Add '+toon' to any AI style (e.g. 'anime+toon') to flatten it into cel colours + ink.
"""
from __future__ import annotations

from ..config import MODELS

STYLE_FILES = {
    "anime": "face_paint_512_v2.onnx",
    "paint": "face_paint_512_v1.onnx",
    "paprika": "paprika.onnx",
    "celeba": "celeba_distill.onnx",
}
# friendly names
ALIASES = {
    "cartoon": "anime+toon",       # recommended AI look: painted cartoon + flat cel colours + ink
    "cartoon-flat": "celeba+toon",
}
STYLES = ["toon", *ALIASES, *STYLE_FILES, *(f"{k}+toon" for k in STYLE_FILES)]
STYLE_HELP = {
    "toon": "no AI - cel-shaded trace of the footage (fastest, free)",
    "cartoon": "light AI - painted cartoon + flat colours + ink (best look)",
    "cartoon-flat": "light AI - simpler, flatter cartoon faces",
    "anime": "light AI - soft hand-painted look",
    "paint": "light AI - stronger stylized faces",
    "paprika": "light AI - bold colourful anime-film look",
    "celeba": "light AI - smooth cartoon faces",
}
# GPU redraw styles (need ComfyUI running) - real cartoon redraw, see assets/redraw_presets.json
REDRAW_HELP = {
    "ghibli": "GPU redraw - ghibli-inspired hand-painted anime film look",
    "toon2d": "GPU redraw - bright 2D TV cartoon, bold outlines, flat vivid colours",
    "cartoon3d": "GPU redraw - 3D animated movie look",
}
ALL_HELP = {**REDRAW_HELP, **STYLE_HELP}


class _Chain:
    def __init__(self, *stages):
        self.stages = stages
        self.name = "+".join(s.name for s in stages)

    def process(self, frame, flow):
        for s in self.stages:
            frame = s.process(frame, flow)
        return frame


def make_stylizer(style: str = "toon", ai_size: int = 512, ai_every: int = 2, **toon_kw):
    from .toon import ToonStylizer

    style = ALIASES.get(style, style)
    parts = style.split("+")
    stages = []
    for p in parts:
        if p == "toon":
            if stages:   # finishing pass on top of an AI style: keep more of its colours
                toon_kw = {"softness": 0.45, "hue_flatness": 0.6, "line_detail": 1.0, "smooth": 0.6, **toon_kw}
            stages.append(ToonStylizer(**toon_kw))
        elif p in STYLE_FILES:
            from .neural import NeuralStylizer
            path = MODELS / "styles" / STYLE_FILES[p]
            if not path.exists():
                raise FileNotFoundError(f"Style model missing: {path}")
            stages.append(NeuralStylizer(path, size=ai_size, every=ai_every, ink=0.0 if "toon" in parts else 0.6))
        else:
            raise ValueError(f"Unknown style '{p}'. Choose from: {', '.join(STYLES)}")
    return stages[0] if len(stages) == 1 else _Chain(*stages)
