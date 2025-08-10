from __future__ import annotations
from typing import Dict, Any
from ..nodes import Node, register

@register
class ColorAdjust(Node):
    NAME = "color_adjust"
    def apply(self, params: Dict[str, Any]) -> str:
        # params: exposure (st), contrast (c), saturation (s), gamma (g)
        st = float(params.get("exposure", 0.0))
        c  = float(params.get("contrast", 1.0))
        s  = float(params.get("saturation", 1.0))
        g  = float(params.get("gamma", 1.0))
        # Use eq filter then saturation via colorchannelmixer approximation
        eq = f"eq=brightness={st}:contrast={c}:gamma={g}"
        sat = f",hue=s={s}"
        return eq + sat
