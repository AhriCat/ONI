from __future__ import annotations
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class Keyframe:
    t: float  # seconds, timeline-local
    v: float
    interp: str = "cubic"  # or "linear", "hold"

class Curve:
    def __init__(self, keys: List[Keyframe] | None = None):
        self.keys: List[Keyframe] = sorted(keys or [], key=lambda k: k.t)

    def sample(self, t: float) -> float:
        if not self.keys:
            return 0.0
        if t <= self.keys[0].t:
            return self.keys[0].v
        if t >= self.keys[-1].t:
            return self.keys[-1].v
        # find segment
        for i in range(len(self.keys) - 1):
            a, b = self.keys[i], self.keys[i + 1]
            if a.t <= t <= b.t:
                if a.interp == "hold":
                    return a.v
                u = (t - a.t) / max(1e-9, (b.t - a.t))
                if a.interp == "linear" or b.interp == "linear":
                    return a.v * (1 - u) + b.v * u
                # cubic hermite with zero tangents (smoothstep-like)
                u2 = u * u
                u3 = u2 * u
                s = 3 * u2 - 2 * u3
                return a.v * (1 - s) + b.v * s
        return self.keys[-1].v
