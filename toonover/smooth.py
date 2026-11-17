"""One Euro filter: removes jitter from tracking while staying responsive."""
from __future__ import annotations

import math


class OneEuro:
    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev: float | None = None
        self.dx_prev = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, dt: float) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class Smoother:
    """A bag of One Euro filters keyed by name, with per-key tuning."""

    PRESETS = {
        # fast features (mouth, blink) stay snappy; body motion is calmer
        "mouth_open": (3.0, 0.3), "mouth_wide": (2.0, 0.1), "smile": (1.5, 0.05),
        "eye": (4.0, 0.5), "brow": (1.5, 0.05),
        "yaw": (1.0, 0.05), "pitch": (1.0, 0.05), "roll": (1.0, 0.05),
        "dx": (0.8, 0.02), "dy": (0.8, 0.02),
    }

    def __init__(self):
        self.f: dict[str, OneEuro] = {}

    def __call__(self, key: str, value: float, dt: float) -> float:
        if key not in self.f:
            base = key.split(".")[0]
            mc, beta = self.PRESETS.get(base, (1.2, 0.02))
            self.f[key] = OneEuro(mc, beta)
        return self.f[key](value, dt)
