from __future__ import annotations
from typing import Dict, Any
try:
    import av
except Exception as e:  # pragma: no cover
    av = None

class PyAVIO:
    def __init__(self):
        if av is None:
            raise RuntimeError("PyAV not installed")

    def read_info(self, path: str) -> Dict[str, Any]:
        with av.open(path) as c:
            return {
                "duration": float(c.duration or 0) / av.time_base if c.duration else 0,
                "streams": [s.type for s in c.streams],
            }
