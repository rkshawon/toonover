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
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["1", 2]}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "toonover"}},
    }
    if p.get("_init"):
        # continue from the previous drawing (video keyframe chaining)
        wf["6"] = {"class_type": "LoadImage", "inputs": {"image": p["_init"]}}
        wf["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["6", 0], "vae": ["1", 2]}}
    elif p["denoise"] >= 0.999:
        # redraw 100% from scratch: start from empty canvas, only the outlines guide the drawing
        wf["7"] = {"class_type": "EmptyLatentImage", "inputs": {"width": p["_w"], "height": p["_h"], "batch_size": 1}}
    else:
        wf["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["5", 0], "vae": ["1", 2]}}
    model, clip = ["1", 0], ["1", 1]
    loras = list(p.get("_loras_found", []))
    if p.get("lora"):
        loras.insert(0, (p["lora"], p.get("lora_strength", 0.8)))
    for i, (name, strength) in enumerate(loras):   # chain any number of style LoRAs
        nid = f"2{i}"
        wf[nid] = {"class_type": "LoraLoader", "inputs": {
            "model": model, "clip": clip, "lora_name": name,
            "strength_model": strength, "strength_clip": strength}}
        model, clip = [nid, 0], [nid, 1]
    text = p["prompt"] + (", " + p["describe"] if p.get("describe") else "")
    wf["3"] = {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": clip}}
    wf["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": p["negative"], "clip": clip}}
    pos, neg = ["3", 0], ["4", 0]
    if p.get("controlnet"):
        wf["8"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": p["controlnet"]}}
        union = "union" in p["controlnet"].lower()

        def typed(node_id, kind):
            if not union:
                return ["8", 0]
            wf[node_id] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["8", 0], "type": kind}}
            return [node_id, 0]

        # 1) outlines (face, eyes, mouth, pose) - made by Toonover, uploaded as p["_lines"]
        wf["16"] = {"class_type": "LoadImage", "inputs": {"image": p.get("_lines", image_name)}}
        wf["11"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": pos, "negative": neg, "control_net": typed("10", "canny/lineart/anime_lineart/mlsd"),
            "image": ["16", 0], "strength": p["control_strength"], "start_percent": 0.0,
            "end_percent": p["control_end"], "vae": ["1", 2]}}
        pos, neg = ["11", 0], ["11", 1]
        # 2) soft colour/layout guide (very blurred photo): keeps hair/skin/clothes colours + composition
        if p.get("tile_strength", 0) > 0 and p.get("_blur") and union:
            wf["17"] = {"class_type": "LoadImage", "inputs": {"image": p["_blur"]}}
            wf["19"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
                "positive": pos, "negative": neg, "control_net": typed("18", "tile"),
                "image": ["17", 0], "strength": p["tile_strength"], "start_percent": 0.0,
                "end_percent": p.get("tile_end", 0.6), "vae": ["1", 2]}}
            pos, neg = ["19", 0], ["19", 1]
    wf["12"] = {"class_type": "KSampler", "inputs": {
        "model": model, "seed": int(p["seed"] if seed is None else seed), "steps": p["steps"], "cfg": p["cfg"],
        "sampler_name": p["sampler"], "scheduler": p["scheduler"],
        "positive": pos, "negative": neg, "latent_image": ["7", 0],
        "denoise": p.get("chain_denoise", 0.55) if p.get("_init") else p["denoise"]}}
    return wf


def sdxl_size(w: int, h: int, area: int = 1024 * 1024) -> tuple[int, int]:
    s = (area / (w * h)) ** 0.5
    return max(512, int(round(w * s / 64)) * 64), max(512, int(round(h * s / 64)) * 64)


def make_guides(frame: np.ndarray, sw: int, sh: int, blur_div: float = 20):
    """Outline map (white lines on black) + very blurred colour guide, both at SDXL size."""
    from ..stylize.ink import clean, xdog

    big = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_CUBIC)
    g = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    # silhouette / big shapes
    a = clean(xdog(cv2.bilateralFilter(g, 9, 40, 9), sigma=1.3), 0.25, 40)
    # facial features: local contrast boost first (eyes, brows, mouth, nose)
    gc = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
    b = clean(xdog(cv2.bilateralFilter(gc, 9, 30, 9), sigma=1.8, tau=0.975, phi=40), 0.15, 40)
    lines = (np.clip(np.maximum(a, b) * 1.6, 0, 1) * 255).astype(np.uint8)
    lines = cv2.dilate(lines, np.ones((2, 2), np.uint8))
    k = max(3, int(max(sw, sh) / blur_div) | 1)
    blur = cv2.GaussianBlur(big, (0, 0), k / 2)
    return cv2.cvtColor(lines, cv2.COLOR_GRAY2BGR), blur


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

    def checkpoints(self) -> list[str]:
        try:
            info = json.loads(self._get("/object_info/CheckpointLoaderSimple"))
            return list(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])
        except Exception:
            return []

    def loras(self) -> list[str]:
        try:
            info = json.loads(self._get("/object_info/LoraLoader"))
            return list(info["LoraLoader"]["input"]["required"]["lora_name"][0])
        except Exception:
            return []

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

    def __init__(self, style: str, url: str = DEFAULT_URL, describe: str = ""):
        self.style = style
        self.p = load_preset(style)
        if describe:
            self.p["describe"] = describe
        self.client = ComfyClient(url)
        self.client.check()
        self.n = 0
        pref = self.p.get("checkpoint_preferred")
        if pref:
            have = self.client.checkpoints()
            stem = pref.rsplit(".", 1)[0].lower()
            match = next((c for c in have if c == pref or stem in c.lower()), None)
            if match:
                self.p["checkpoint"] = match
            else:
                print(f"  (tip: '{pref}' not found in ComfyUI checkpoints - using {self.p['checkpoint']})")
        # optional style LoRAs: used automatically when their file is in ComfyUI/models/loras
        wanted = self.p.get("loras", [])
        if wanted:
            have = self.client.loras()
            found = []
            for w in wanted:
                m = next((h for h in have if w["match"].lower() in h.lower()), None)
                if m:
                    found.append((m, w.get("strength", 0.8)))
                    if w.get("trigger"):
                        self.p["prompt"] = w["trigger"] + ", " + self.p["prompt"]
            self.p["_loras_found"] = found
            if found:
                print("  style LoRAs: " + ", ".join(f"{n} @ {st}" for n, st in found))
            else:
                print("  (no style LoRA found - using prompt only)")

    def __call__(self, frame: np.ndarray, init: np.ndarray | None = None,
                 init_conf: np.ndarray | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        sw, sh = sdxl_size(w, h)
        img = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_CUBIC if sw > w else cv2.INTER_AREA)
        tag = f"{self.client.client_id[:8]}_{self.n:05d}"
        name = self.client.upload(img, f"toonover_key_{tag}.png")
        lines, blur = make_guides(frame, sw, sh, self.p.get("guide_blur", 20))
        self.p["_lines"] = self.client.upload(lines, f"toonover_lines_{tag}.png")
        self.p["_blur"] = self.client.upload(blur, f"toonover_blur_{tag}.png")
        self.p["_init"] = None
        if init is not None:
            if init_conf is not None:   # where tracking was lost, start from the soft colour guide instead
                c = cv2.GaussianBlur(init_conf, (0, 0), 5)[..., None]
                soft = cv2.resize(blur, (w, h))
                init = (init.astype(np.float32) * c + soft.astype(np.float32) * (1 - c)).astype(np.uint8)
            init_img = cv2.resize(init, (sw, sh), interpolation=cv2.INTER_CUBIC)
            self.p["_init"] = self.client.upload(init_img, f"toonover_init_{tag}.png")
        self.n += 1
        self.p["_w"], self.p["_h"] = sw, sh
        out = self.client.run(build_workflow(self.p, name))
        return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
