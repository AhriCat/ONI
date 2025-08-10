from __future__ import annotations
from typing import Dict, Any
from ..nodes import Node, register

@register
class DrawText(Node):
    NAME = "drawtext"
    def apply(self, params: Dict[str, Any]) -> str:
        # params: text, x, y, fontsize, fontcolor, fontfile
        text = (params.get("text") or "").replace(':', '\\:')
        x = params.get("x", "(w-tw)/2")
        y = params.get("y", "(h-th)-40")
        fs = int(params.get("fontsize", 48))
        color = params.get("fontcolor", "white")
        fontfile = params.get("fontfile")
        fontopt = f":fontfile={fontfile}" if fontfile else ""
        return f"drawtext=text='{text}':x={x}:y={y}:fontsize={fs}:fontcolor={color}{fontopt}"
