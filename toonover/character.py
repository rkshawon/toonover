"""Procedural flat-2D cartoon characters (the Toonover cast).

Every body part is drawn from simple shapes, so characters can be posed by
numbers coming from the director (head tilt, mouth, brows, arms ...).
Units: 1.0 = head height. Character origin = between the feet.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ASSETS
from .director import Pose
from .draw import INK, Frame2D, circle, ellipse_pts, hex2bgr, limb, poly, shade, stroke

MOUTH_DARK = hex2bgr("#5A1A1E")
TONGUE = hex2bgr("#E86A72")
WHITE = (250, 250, 250)

HEAD_RX, HEAD_RY = 0.42, 0.5


@dataclass
class Spec:
    id: str
    name: str
    skin: str
    hair: str
    hair_style: str
    outfit: str = "tshirt"          # tshirt | jacket | kurta | vest | saree
    top: str = "#3DBA4E"
    inner: str = "#EEE6D2"          # jacket inner shirt
    pants: str = "#2E3440"
    shoes: str = "#2A2A2A"
    scarf: str = "#1FA3C6"
    sash: str = "#C2185B"
    accessories: list[str] = field(default_factory=list)
    height: float = 1.0             # relative body height


def load_cast(path: Path | None = None) -> dict[str, Spec]:
    data = json.loads((path or ASSETS / "cast.json").read_text(encoding="utf-8"))
    return {c["id"]: Spec(**c) for c in data["characters"]}


def _clamp(v, a, b):
    return a if v < a else b if v > b else v


class Character:
    def __init__(self, spec: Spec, seed: int = 0):
        self.s = spec
        self.rng = random.Random(seed)
        self.phase = self.rng.uniform(0, 6.28)
        self.next_blink = self.rng.uniform(0.5, 3.0)
        self.arm = {"L": None, "R": None}
        c = hex2bgr
        self.col = dict(skin=c(spec.skin), hair=c(spec.hair), top=c(spec.top), inner=c(spec.inner),
                        pants=c(spec.pants), shoes=c(spec.shoes), scarf=c(spec.scarf), sash=c(spec.sash))
        self.sleeve = {"vest": "none", "tshirt": "short", "saree": "short"}.get(spec.outfit, "full")

    # ------------------------------------------------------------------ helpers
    def _blink(self, t: float) -> float:
        if t > self.next_blink + 0.14:
            self.next_blink = t + self.rng.uniform(1.8, 4.5)
        return 0.08 if self.next_blink <= t <= self.next_blink + 0.14 else 1.0

    def _arm_angles(self, side: str, p: Pose, t: float):
        idle_up = math.pi / 2 + (0.18 if side == "L" else -0.18)
        idle_fo = math.pi / 2 + (-0.12 if side == "L" else 0.12)
        g = p.talk * (0.9 if side == "R" else 0.35)
        wob = math.sin(t * 4.2 + self.phase + (0 if side == "R" else 1.7))
        # talking gesture: elbow out, forearm swings up and outward
        if side == "L":
            fo = idle_fo + g * (1.9 + 0.35 * wob)
            up = idle_up + g * 0.45
        else:
            fo = idle_fo - g * (1.9 + 0.35 * wob)
            up = idle_up - g * 0.45
        target = getattr(p, f"arm_{side}")
        if target is not None:
            up, fo = target
        prev = self.arm[side]
        if prev is None:
            self.arm[side] = (up, fo)
        else:  # smooth on the circle
            k = 0.3
            mix = lambda a, b: math.atan2((1 - k) * math.sin(a) + k * math.sin(b),
                                          (1 - k) * math.cos(a) + k * math.cos(b))
            self.arm[side] = (mix(prev[0], up), mix(prev[1], fo))
        return self.arm[side]

    # ------------------------------------------------------------------ drawing
    def shadow(self, img, x, floor_y, U, color):
        poly(img, ellipse_pts((x, floor_y), 0.75 * U, 0.12 * U), color, None)

    def draw(self, img, x: float, floor_y: float, U: float, p: Pose, t: float):
        U *= self.s.height
        lw = max(2.0, U * 0.035)
        col = self.col
        root = Frame2D((x, floor_y), 0.0, U)
        breath = math.sin(t * 2.3 + self.phase)
        lean = _clamp(p.dx, -1, 1) * 0.08 + 0.015 * math.sin(t * 0.9 + self.phase)

        self._legs(img, root, lw)
        hip = root.child((0, -1.25), lean)
        chest = hip.child((0, -1.3 * (1 + 0.008 * breath)))

        bob = _clamp(p.dy, -1, 1) * 0.05 + 0.01 * math.sin(t * 2.3 + self.phase + 1)
        roll = _clamp(p.roll, -0.4, 0.4) + lean * 0.5
        neck = chest.child((0, -0.06 + bob), roll)
        head = neck.child((0, -0.48))

        self._back_hair(img, head, lw)
        # neck
        poly(img, chest([(-0.1, -0.16), (0.1, -0.16), (0.11, 0.1), (-0.11, 0.1)]), col["skin"], INK, lw)
        self._torso(img, chest, lw)
        self._head(img, head, p, t, lw)
        self._arms(img, chest, U, p, t, lw)

    def _legs(self, img, root, lw):
        c = self.col
        if self.s.outfit == "saree":
            poly(img, root([(-0.4, -1.32), (0.4, -1.32), (0.52, -0.08), (-0.52, -0.08)]), c["top"], INK, lw)
            stroke(img, root([(0.05, -1.25), (0.15, -0.1)]), shade(c["top"], 0.75), lw)
            stroke(img, root([(-0.15, -1.2), (-0.12, -0.1)]), shade(c["top"], 0.75), lw)
            for sx in (-1, 1):
                poly(img, ellipse_pts(root.pt(sx * 0.22, -0.05), 0.17 * root.s, 0.07 * root.s), c["skin"], INK, lw)
            return
        for sx in (-1, 1):
            pts = [(sx * 0.04, -1.32), (sx * 0.38, -1.32), (sx * 0.34, -0.1), (sx * 0.08, -0.1)]
            poly(img, root(pts), c["pants"], INK, lw)
            poly(img, ellipse_pts(root.pt(sx * 0.25, -0.06), 0.2 * root.s, 0.085 * root.s), c["shoes"], INK, lw)

    def _torso(self, img, chest, lw):
        c, o = self.col, self.s.outfit
        bottom = 1.85 if o == "kurta" else 1.32
        flare = 0.52 if o == "kurta" else 0.44
        pts = [(-0.17, 0.0), (-0.47, 0.12), (-0.53, 0.34), (-0.45, 1.0), (-flare, bottom),
               (flare, bottom), (0.45, 1.0), (0.53, 0.34), (0.47, 0.12), (0.17, 0.0)]
        poly(img, chest(pts), c["top"], INK, lw)
        if o == "jacket":
            poly(img, chest([(-0.16, 0.0), (0.16, 0.0), (0.1, 1.3), (-0.1, 1.3)]), c["inner"], INK, lw)
            stroke(img, chest([(-0.16, 0.0), (-0.24, 0.35), (-0.1, 0.6)]), INK, lw)
            stroke(img, chest([(0.16, 0.0), (0.24, 0.35), (0.1, 0.6)]), INK, lw)
        elif o == "vest":
            poly(img, chest([(-0.17, -0.01), (0.17, -0.01), (0.0, 0.3)]), c["skin"], INK, lw)
        elif o == "kurta":
            stroke(img, chest([(0.0, 0.02), (0.0, 0.55)]), INK, lw)
            for y in (0.2, 0.35, 0.5):
                circle(img, chest.pt(0.05, y), 0.022 * chest.s, WHITE, INK, lw * 0.6)
        elif o == "saree":
            poly(img, chest([(-0.5, 0.12), (-0.22, 0.0), (0.47, 1.0), (0.45, 1.32), (0.2, 1.32)]), c["sash"], INK, lw)
        else:  # tshirt collar
            stroke(img, chest(ellipse_pts((0, -0.02), 0.17, 0.12, 0, 0.2, math.pi - 0.2, 12)), INK, lw)

    def _arms(self, img, chest, U, p, t, lw):
        c = self.col
        sleeve = self.sleeve
        for side, sx in (("L", -1), ("R", 1)):
            up, fo = self._arm_angles(side, p, t)
            S = chest.pt(sx * 0.44, 0.22)
            E = S + 0.6 * U * np.array([math.cos(up), math.sin(up)])
            W = E + 0.55 * U * np.array([math.cos(fo), math.sin(fo)])
            w = 0.17 * U
            upper = c["skin"] if sleeve == "none" else c["top"]
            fore = c["top"] if sleeve == "full" else c["skin"]
            if sleeve == "short":
                mid = S + 0.55 * (E - S)
                limb(img, [S, mid], c["top"], w * 1.1, lw)
                limb(img, [mid, E, W], c["skin"], w * 0.9, lw)
            else:
                limb(img, [S, E], upper, w, lw)
                limb(img, [E, W], fore, w * 0.95, lw)
            circle(img, W + 0.04 * U * np.array([math.cos(fo), math.sin(fo)]), 0.1 * U, c["skin"], INK, lw)

    def _back_hair(self, img, h, lw):
        c, st = self.col, self.s.hair_style
        if st == "bun":
            circle(img, h.pt(0, -0.6), 0.17 * h.s, c["hair"], INK, lw)
        elif st == "long":
            poly(img, h([(-0.5, -0.1), (-0.56, 0.75), (0.56, 0.75), (0.5, -0.1)]), c["hair"], INK, lw)
            poly(img, h(ellipse_pts((0, 0.0), 0.52, 0.6)), c["hair"], INK, lw)
        elif st == "scarf":
            arc = ellipse_pts((0, 0.0), HEAD_RX * 1.25, HEAD_RY * 1.18, 0, math.pi - 0.7, 2 * math.pi + 0.7, 30)
            pts = np.vstack([[[-0.66, 1.0]], arc, [[0.66, 1.0]]])
            poly(img, h(pts), c["scarf"], INK, lw)

    def _head_shape(self, yaw):
        t = np.linspace(0, 2 * math.pi, 48, endpoint=False)
        s = np.sin(t)
        x = HEAD_RX * np.cos(t) * (1 - 0.06 * abs(yaw))
        x = np.where(s > 0, x * (1 - 0.2 * s ** 2), x)
        return np.stack([x + yaw * 0.03, HEAD_RY * s], axis=1)

    def _head(self, img, h, p: Pose, t: float, lw: float):
        c, acc = self.col, self.s.accessories
        yaw = _clamp(p.yaw, -1, 1)
        fx = yaw * 0.13
        fy = _clamp(p.pitch, -1, 1) * 0.06
        # ears
        for sx in (-1, 1):
            poly(img, h(ellipse_pts((sx * 0.42 - fx * 0.4, 0.04), 0.075, 0.11)), c["skin"], INK, lw)
        shape = self._head_shape(yaw)
        poly(img, h(shape), c["skin"], None)
        if "stubble" in acc:
            low = shape[shape[:, 1] > 0.2]
            if len(low) > 3:
                poly(img, h(low), shade(c["skin"], 0.88), None)
        poly(img, h(shape), None, INK, lw)

        shock = _clamp((p.brow - 0.4) / 0.7, 0, 1) * _clamp((p.mouth_open - 0.25) / 0.4, 0, 1)
        # eyes
        eye = p.eye if p.present > 0.5 else 1.0
        eye = min(eye, self._blink(t))
        eye = max(eye, 0.05) * (1 + 0.25 * shock)
        for sx in (-1, 1):
            ec = (sx * 0.17 + fx, -0.02 + fy)
            ry = 0.15 * _clamp(eye, 0, 1.35)
            if ry < 0.04:
                stroke(img, h(ellipse_pts(ec, 0.11, 0.035, 0, 0.15, math.pi - 0.15, 10)), INK, lw * 1.2)
            else:
                poly(img, h(ellipse_pts(ec, 0.12, ry)), WHITE, INK, lw)
                pr = 0.045 * (1 - 0.3 * shock)
                pc = (ec[0] + yaw * 0.05, ec[1] + fy * 0.3)
                circle(img, h.pt(*pc), pr * h.s, INK, None)
        # brows
        b = _clamp(p.brow, -1, 1.5)
        for sx in (-1, 1):
            inner_y = -0.21 - b * 0.06 + (0.05 * -b if b < 0 else 0)
            outer_y = -0.23 - b * 0.05
            stroke(img, h([(sx * 0.07 + fx, inner_y + fy), (sx * 0.28 + fx, outer_y + fy)]), c["hair"], lw * 2.2)
        # nose
        stroke(img, h([(fx * 1.2 + 0.0, 0.02 + fy), (fx * 1.2 + 0.05, 0.13 + fy), (fx * 1.2 - 0.02, 0.14 + fy)]), INK, lw * 0.9)
        # mouth
        self._mouth(img, h, (fx * 1.1, 0.27 + fy), p, shock, lw)
        if "mustache" in acc:
            mx, my = fx * 1.1, 0.19 + fy
            poly(img, h([(mx - 0.17, my + 0.06), (mx - 0.1, my - 0.02), (mx, my - 0.01), (mx + 0.1, my - 0.02),
                         (mx + 0.17, my + 0.06), (mx + 0.08, my + 0.03), (mx, my + 0.04), (mx - 0.08, my + 0.03)]),
                 c["hair"], INK, lw * 0.8)
        if "glasses" in acc:
            for sx in (-1, 1):
                poly(img, h(ellipse_pts((sx * 0.17 + fx, -0.02 + fy), 0.155, 0.14)), None, INK, lw * 1.1)
            stroke(img, h([(-0.02 + fx, -0.04 + fy), (0.02 + fx, -0.04 + fy)]), INK, lw * 1.1)
        if "bindi" in acc:
            circle(img, h.pt(fx, -0.17 + fy), 0.028 * h.s, hex2bgr("#D3222A"), INK, lw * 0.5)
        self._front_hair(img, h, yaw, lw)

    def _mouth(self, img, h, centre, p: Pose, shock: float, lw):
        cx, cy = centre
        s = _clamp(p.smile, -1, 1)
        op = _clamp(p.mouth_open, 0, 1)
        W = 0.11 * (1 + 0.35 * _clamp(p.mouth_wide, -1, 1) + 0.3 * max(s, 0)) * (1 - 0.15 * shock)
        Hm = 0.22 * op * (1 + 0.3 * shock)
        u = np.linspace(-1, 1, 21)
        top = -s * 0.045 * u ** 2 - Hm * 0.12 * (1 - u ** 2)
        if Hm < 0.025:
            stroke(img, h(np.stack([cx + W * u, cy + top], 1)), INK, lw * 1.15)
            return
        bot = -s * 0.045 * u ** 2 + Hm * np.sqrt(np.clip(1 - u ** 2, 0, 1))
        pts = np.vstack([np.stack([cx + W * u, cy + top], 1), np.stack([cx + W * u[::-1], cy + bot[::-1]], 1)])
        poly(img, h(pts), MOUTH_DARK, None)
        if Hm > 0.06:
            tb = min(0.045, Hm * 0.28)
            inner = u * 0.9
            ttop = -s * 0.045 * inner ** 2 - Hm * 0.12 * (1 - inner ** 2)
            teeth = np.vstack([np.stack([cx + W * inner, cy + ttop], 1),
                               np.stack([cx + W * inner[::-1], cy + ttop[::-1] + tb], 1)])
            poly(img, h(teeth), WHITE, None)
            poly(img, h(ellipse_pts((cx, cy + Hm * 0.72), W * 0.5, Hm * 0.24)), TONGUE, None)
        poly(img, h(pts), None, INK, lw)

    def _front_hair(self, img, h, yaw, lw):
        c, st = self.col, self.s.hair_style
        hc = c["hair"]
        ox = yaw * 0.03

        def arc(scale, a0, a1, n=24, bump=None):
            t = np.linspace(a0, a1, n)
            r = np.full_like(t, scale) if bump is None else scale + bump(t)
            return np.stack([HEAD_RX * r * np.cos(t) + ox, HEAD_RY * r * np.sin(t)], 1)

        L, R = math.pi + 0.3, 2 * math.pi - 0.3
        if st == "spiky":
            n = 15
            t = np.linspace(L, R, n)
            r = np.where(np.arange(n) % 2 == 0, 1.06, 1.26)
            top = np.stack([HEAD_RX * r * np.cos(t) + ox, HEAD_RY * r * np.sin(t)], 1)
            fringe = np.array([(0.33, -0.17), (0.2, -0.27), (0.1, -0.16), (-0.04, -0.28), (-0.14, -0.17), (-0.3, -0.25)]) + [ox, 0]
            poly(img, h(np.vstack([top, fringe])), hc, INK, lw)
        elif st == "messy":
            top = arc(1.1, L - 0.1, R + 0.1, 30, lambda t: 0.07 * np.sin(9 * t))
            fringe = np.array([(0.36, -0.1), (0.25, -0.22), (0.15, -0.14), (0.02, -0.25), (-0.1, -0.13), (-0.22, -0.24), (-0.37, -0.1)]) + [ox, 0]
            poly(img, h(np.vstack([top, fringe])), hc, INK, lw)
        elif st == "side_part":
            top = arc(1.09, L, R)
            fringe = np.array([(0.38, -0.12), (0.2, -0.24), (-0.05, -0.3), (-0.25, -0.26), (-0.38, -0.14)]) + [ox, 0]
            poly(img, h(np.vstack([top, fringe])), hc, INK, lw)
            stroke(img, h(np.array([(0.15, -0.5), (0.05, -0.32)]) + [ox, 0]), shade(hc, 1.8), lw)
        elif st == "bald_sides":
            for sx in (-1, 1):
                poly(img, h(np.array([(sx * 0.44, -0.12), (sx * 0.36, -0.24), (sx * 0.31, -0.06), (sx * 0.34, 0.12), (sx * 0.43, 0.08)]) + [ox, 0]), hc, INK, lw)
            stroke(img, h(arc(0.97, math.pi * 1.4, math.pi * 1.6, 8)), shade(c["skin"], 0.7), lw)
        elif st in ("bun", "long", "scarf"):
            top = arc(1.06, L - 0.1, R + 0.1)
            fringe = np.array([(0.38, -0.08), (0.18, -0.26), (0.0, -0.33), (-0.18, -0.26), (-0.38, -0.08)]) + [ox, 0]
            poly(img, h(np.vstack([top, fringe])), hc, INK, lw)
            stroke(img, h(np.array([(0.0, -0.33), (0.02, -0.5)]) + [ox, 0]), shade(hc, 2.2), lw)
            if st == "scarf":
                outer = arc(1.2, math.pi + 0.05, 2 * math.pi - 0.05, 30)
                inner = arc(0.98, 2 * math.pi - 0.25, math.pi + 0.25, 30)
                poly(img, h(np.vstack([outer, inner])), c["scarf"], INK, lw)
