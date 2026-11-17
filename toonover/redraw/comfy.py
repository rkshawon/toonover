"""Talk to a local ComfyUI (runs the image model on your GPU - AMD ROCm or NVIDIA).

We send one keyframe at a time: SDXL img2img + ControlNet (edges keep the pose,
face and composition) + optional style LoRA, and get the redrawn picture back.
"""
from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np

from ..config import ASSETS

DEFAULT_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")


def load_preset(name: str) -> dict:
    data = json.loads((ASSETS / "redraw_presets.json").read_text(encoding="utf-8"))
    if name not in data["presets"]:
        raise KeyError(f"Unknown redraw style '{name}'. Available: {', '.join(data['presets'])}")
    return {**data["defaults"], **data["presets"][name]}


def presets() -> list[str]:
    return list(json.loads((ASSETS / "redraw_presets.json").read_text(encoding="utf-8"))["presets"])


def build_workflow(p: dict, image_name: str, seed: int | None = None) -> dict:
    """ComfyUI API-format graph using only built-in nodes."""
    wf: dict = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": p["checkpoint"]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "toonover"}},
    }
    model, clip = ["1", 0], ["1", 1]
    if p.get("lora"):
        wf["2"] = {"class_type": "LoraLoader", "inputs": {
            "model": model, "clip": clip, "lora_name": p["lora"],
            "strength_model": p.get("lora_strength", 0.8), "strength_clip": p.get("lora_strength", 0.8)}}
        model, clip = ["2", 0], ["2", 1]
    wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p["prompt"], "clip": clip}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p["negative"], "clip": clip}}
    pos, neg = ["3", 0], ["4", 0]
    if p.get("controlnet"):
        wf["8"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": p["controlnet"]}}
        cn = ["8", 0]
        if "union" in p["controlnet"].lower():
            wf["10"] = {"class_type": "SetUnionControlNetType",
                        "inputs": {"control_net": cn, "type": "canny/lineart/anime_lineart/mlsd"}}
            cn = ["10", 0]
        wf["9"] = {"class_type": "Canny", "inputs": {"image": ["5", 0], "low_threshold": 0.25, "high_threshold": 0.6}}
        wf["11"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": pos, "negative": neg, "control_net": cn, "image": ["9", 0],
            "strength": p["control_strength"], "start_percent": 0.0, "end_percent": p["control_end"],
            "vae": ["1", 2]}}
        pos, neg = ["11", 0], ["11", 1]
    wf["12"] = {"class_type": "KSampler", "inputs": {
        "model": model, "seed": int(p["seed"] if seed is None else seed), "steps": p["steps"], "cfg": p["cfg"],
        "sampler_name": p["sampler"], "scheduler": p["scheduler"],
        "positive": pos, "negative": neg, "latent_image": ["7", 0], "denoise": p["denoise"]}}
    return wf


def sdxl_size(w: int, h: int, area: int = 1024 * 1024) -> tuple[int, int]:
    s = (area / (w * h)) ** 0.5
    return max(512, int(round(w * s / 64)) * 64), max(512, int(round(h * s / 64)) * 64)


class ComfyClient:
    def __init__(self, url: str = DEFAULT_URL, timeout: float = 600):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    # -- http helpers ---------------------------------------------------------
    def _get(self, path: str) -> bytes:
        with urllib.request.urlopen(self.url + path, timeout=60) as r:
            return r.read()

    def _post_json(self, path: str, data: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(data).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected the workflow: {e.read().decode(errors='ignore')[:1500]}") from e

    def check(self) -> dict:
        try:
            return json.loads(self._get("/system_stats"))
        except Exception as e:
            raise RuntimeError(
                f"Can't reach ComfyUI at {self.url}. Start ComfyUI first (or set COMFY_URL, "
                f"e.g. http://127.0.0.1:8000 for ComfyUI Desktop). ({e})") from e

    def upload(self, img_bgr: np.ndarray, name: str) -> str:
        ok, buf = cv2.imencode(".png", img_bgr)
        boundary = uuid.uuid4().hex
        body = io.BytesIO()
        for k, v in (("overwrite", "true"), ("type", "input")):
            body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
                   f"Content-Type: image/png\r\n\r\n".encode())
        body.write(buf.tobytes())
        body.write(f"\r\n--{boundary}--\r\n".encode())
        req = urllib.request.Request(self.url + "/upload/image", data=body.getvalue(),
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read())
        return (res.get("subfolder") + "/" if res.get("subfolder") else "") + res["name"]

    def run(self, workflow: dict) -> np.ndarray:
        pid = self._post_json("/prompt", {"prompt": workflow, "client_id": self.client_id})["prompt_id"]
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            hist = json.loads(self._get(f"/history/{pid}"))
            if pid in hist:
                h = hist[pid]
                status = h.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI error: {json.dumps(status.get('messages', []))[:1500]}")
                for out in h.get("outputs", {}).values():
                    for im in out.get("images", []):
                        q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im.get("subfolder", ""),
                                                    "type": im.get("type", "output")})
                        data = np.frombuffer(self._get(f"/view?{q}"), np.uint8)
                        return cv2.imdecode(data, cv2.IMREAD_COLOR)
            time.sleep(0.4)
        raise TimeoutError("ComfyUI took too long")


class ComfyRedrawer:
    """Redraw single frames in a preset style."""

    def __init__(self, style: str, url: str = DEFAULT_URL):
        self.style = style
        self.p = load_preset(style)
        self.client = ComfyClient(url)
        self.client.check()
        self.n = 0

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        sw, sh = sdxl_size(w, h)
        img = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_CUBIC if sw > w else cv2.INTER_AREA)
        name = self.client.upload(img, f"toonover_key_{self.client.client_id[:8]}_{self.n:05d}.png")
        self.n += 1
        out = self.client.run(build_workflow(self.p, name))
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
