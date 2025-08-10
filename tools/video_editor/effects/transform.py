from __future__ import annotations
from typing import Dict, Any
from ..nodes import Node, register

@register
class Transform(Node):
    NAME = "transform"
    def apply(self, params: Dict[str, Any]) -> str:
        # params: scale_w, scale_h, rotate_deg, crop_x, crop_y, crop_w, crop_h
        w = params.get("scale_w")
        h = params.get("scale_h")
        rot = float(params.get("rotate_deg", 0.0))
        cx = params.get("crop_x")
        cy = params.get("crop_y")
        cw = params.get("crop_w")
        ch = params.get("crop_h")
        parts = []
        if w or h:
            sw = w if w else "iw"
            sh = h if h else "ih"
            parts.append(f"scale={sw}:{sh}")
        if abs(rot) > 1e-6:
            parts.append(f"rotate={rot*3.14159265/180.0}:ow=rotw({rot*3.14159265/180.0}):oh=roth({rot*3.14159265/180.0})")
        if all(v is not None for v in [cx, cy, cw, ch]):
            parts.append(f"crop={cw}:{ch}:{cx}:{cy}")
        return ",".join(parts)
