# GPU setup: AMD Radeon RX 9060 XT (16 GB) on Windows 11

Toonover's redraw styles (`ghibli`, `toon2d`, `cartoon3d`) send keyframes to **ComfyUI**, which runs the image
model on your Radeon through AMD ROCm. ROCm for Windows supports the RX 9060 XT (ROCm 7.x, Windows 11 only).

## 1. Driver

Install the latest **AMD Software: Adrenalin Edition** (26.1.1 or newer). If ComfyUI later says the GPU isn't
found, install the "AMD ROCm on Windows" / AI bundle driver AMD links from its ROCm-for-Windows release notes.

## 2. Downloads (big files: use your browser, save into `E:\ToonoverDownloads`)

| File | Size | Link |
|---|---|---|
| `ComfyUI_windows_portable_amd.7z` | 1.7 GB | https://github.com/Comfy-Org/ComfyUI/releases/download/v0.45.0/ComfyUI_windows_portable_amd.7z |
| `sd_xl_base_1.0.safetensors` | 6.9 GB | https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors |
| `diffusion_pytorch_model_promax.safetensors` | 2.5 GB | https://huggingface.co/xinsir/controlnet-union-sdxl-1.0/resolve/main/diffusion_pytorch_model_promax.safetensors |
| *(optional)* style LoRA `.safetensors` | ~0.2 GB | any SDXL style LoRA; set its name in `assets/redraw_presets.json` `"lora"` |

Small tools (portable Python, ffmpeg, 7-Zip, Python wheels) are already in place.

## 3. Install

Double-click `scripts\setup_local.bat`. It extracts ComfyUI to `E:\ProgramData\Toonover\ComfyUI\`, renames the
ControlNet file, and moves every model into the right ComfyUI folder (any other `.safetensors` left in the
downloads folder is treated as a LoRA).

Start the GPU server with `start_comfyui.bat` (keeps ROCm/torch caches on E:\ too). Leave it open.
If you get out-of-memory errors, add `--reserve-vram 3` to the python line in ComfyUI's `run_amd_gpu.bat`.

Better results: swap the base checkpoint for an SDXL **anime / illustration** checkpoint (change `"checkpoint"`
in `assets/redraw_presets.json`). Check each model's licence if you'll earn money from the videos.

## 4. Test

```powershell
cd D:\Work\toonover
toonover.bat photo some_face.jpg -s ghibli        # one picture: ~5-10 s on the 9060 XT
toonover.bat make clip.mp4 -s ghibli --max-seconds 10
```

## Speed / tuning

- Rough time on a 9060 XT: ~4-8 s per keyframe at ~1 MP, 22 steps. A 30 s / 30 fps clip with
  `keyframe_every: 10` = ~90 keyframes ≈ 6-12 min.
- Faster: `"steps": 16`, or `--keyframe-every 15`. Smoother: `--keyframe-every 6`.
- More cartoon: raise `denoise` (0.65-0.75). More like the real person: raise `control_strength` (0.7-0.85)
  or lower `denoise` (0.5-0.55).
- Keep `seed` fixed so keyframes look consistent with each other.
