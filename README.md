# Toonover

Paste a meme link (or upload a clip) → get **the same video as a cartoon**: same people, same motion,
same audio, same frame rate. The people stay recognizable because the cartoon is traced from the real footage.

```
link / file ─► yt-dlp ─► ffmpeg (normalize, audio) ─► optical flow (shot cuts + frame-to-frame motion)
           ─► stylizer (toon / AI style) with temporal smoothing ─► ffmpeg (H.264 + original audio)
```

## Styles

### GPU redraw styles (real cartoon) - `ghibli`, `toon2d`, `cartoon3d`

These **redraw** the video as a real cartoon (new shapes, painted colours, clean outlines) instead of filtering it.
They use an image model (SDXL) on **your own GPU** through ComfyUI, so they're still $0 per video.

How it stays smooth and cheap: only every 10th frame (and every shot start/end) is redrawn. The frames between
are made by carrying those redrawn keyframes along the real motion (optical flow, forward and backward)
and blending them, the way EbSynth does. A 30 s Short = ~90 redraws instead of 900.

Setup for the **RX 9060 XT 16 GB** (Windows 11): see [docs/GPU_SETUP_AMD.md](docs/GPU_SETUP_AMD.md). Then:

```powershell
# 1. start ComfyUI (run_amd_gpu.bat) and leave it open
# 2. test one picture, then a video
toonover.bat photo face.jpg -s ghibli
toonover.bat make "https://www.youtube.com/shorts/XXXX" -s ghibli -t "My title"
```

Tune prompts, strength and keyframe spacing in `assets/redraw_presets.json`
(`denoise` higher = more cartoon / less like the original, `control_strength` higher = keeps pose & face closer,
`keyframe_every` lower = smoother but slower). `--keyframe-every 6` on the command line overrides it.

### Free CPU styles

| Style | AI? | Look | Speed (8-core CPU, 720p) |
|---|---|---|---|
| `toon` (default) | no | cel-shaded trace: flat colour bands + ink lines | ~0.1 s/frame |
| `cartoon` | light, CPU | painted cartoon + flat colours + ink (**best look**) | ~0.3 s/frame |
| `cartoon-flat` | light, CPU | simpler, flatter cartoon faces | ~0.3 s/frame |
| `anime`, `paint`, `celeba` | light, CPU | raw AI looks (add `+toon` for cel finish) | ~0.3 s/frame |

`toonover styles` lists them all. Everything runs on your own PC: no GPU, no API, no per-video cost.

AI styles run the model on every 2nd frame (`--ai-every 2`, the way 2D cartoons are animated "on twos")
and fill in the frames between with optical flow. Results are blended over time so they don't flicker.

## Setup (Windows, project-local - nothing installed globally)

All tools live on E:\ only:

| What | Where |
|---|---|
| Portable Python 3.12, ffmpeg, 7-Zip, venv, caches | `E:\ProgramData\Toonover\` |
| ComfyUI (AMD) + models | `E:\ProgramData\Toonover\ComfyUI\` |
| Downloaded installers / wheels / models | `E:\ToonoverDownloads\` |

1. Download the big files with your browser into `E:\ToonoverDownloads` (see docs/GPU_SETUP_AMD.md).
2. Double-click `scripts\setup_local.bat` (safe to re-run; it installs offline from the downloaded wheels,
   extracts ComfyUI and moves the models into place).
3. Use `toonover.bat ...` from the project folder, and `start_comfyui.bat` for the GPU styles.

## Use

```powershell
cd D:\Work\toonover
toonover.bat make "https://www.youtube.com/shorts/XXXX"                    # no AI
toonover.bat make "https://www.youtube.com/shorts/XXXX" -s cartoon         # AI look
toonover.bat make clip.mp4 -s cartoon -t "Samay was Rich" --channel Toonover   # with title/channel bars
toonover.bat photo face.jpg -s cartoon                                      # test a style on one picture
toonover.bat serve                                                        # web app: http://127.0.0.1:8000
```

`make` options: `-o out.mp4`, `-t "title"` / `--auto-title` (top bar), `--channel "text"` (bottom bar),
`--max-seconds 90`, `--ai-size 512` (smaller = faster), `--ai-every 2` (3 = faster).

Output goes to `output\`, temp files to `work\` (safe to delete).

## Tuning the no-AI `toon` style

In `toonover/stylize/toon.py`, `ToonStylizer(...)`:
`bands` (brightness steps), `hues` (colour count), `softness` (keep some real shading),
`hue_flatness`, `line_detail`, `ink`, `smooth`, `hysteresis` (higher = steadier colours).

## Project layout

| Path | What it does |
|---|---|
| `toonover/convert.py` | real → cartoon pipeline (`convert_video`, `convert_image`) |
| `toonover/stylize/toon.py` | no-AI cel-shading stylizer |
| `toonover/stylize/neural.py` | ONNX cartoon models on CPU, "on twos" + flow blending |
| `toonover/stylize/temporal.py` | DIS optical flow, shot-cut detection, warping |
| `toonover/stylize/ink.py` | XDoG ink lines |
| `toonover/ingest.py` | yt-dlp download, ffmpeg normalize + audio |
| `toonover/render.py` | ffmpeg writer, title text |
| `toonover/web/app.py` | FastAPI web app |
| `models/styles/*.onnx` | AI style models (AnimeGANv2 by bryandlee/animegan2-pytorch, MIT licence) |

## Notes

- AI styles slightly "beautify" faces. If someone must look more like themselves, use `toon` or `celeba+toon`.
- Use clips you have the right to use. Downloading from platforms may break their terms, and cartoon versions
  of real people should stay clearly parody/commentary, not used to mislead or harass.

## Roadmap

1. Face-aware ink (landmark-guided eyes/brows/lips) for cleaner faces in `toon`
2. Person/background separation (flat painted backgrounds, crisper people)
3. Optional GPU / API video-to-video path for a fully redrawn look
