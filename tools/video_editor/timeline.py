from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from .keyframes import Curve, Keyframe

@dataclass
class Clip:
    id: str
    src: str
    start: float = 0.0       # source inpoint (sec)
    duration: float = 0.0    # clip length on timeline (sec)
    offset: float = 0.0      # timeline position (sec) relative to track start
    params: Dict[str, Any] = field(default_factory=dict)  # effect params / curves by name

    def param_curve(self, name: str) -> Curve:
        v = self.params.get(name)
        if isinstance(v, Curve):
            return v
        if isinstance(v, list):
            return Curve([Keyframe(**k) if isinstance(k, dict) else k for k in v])
        return Curve([])

@dataclass
class Transition:
    id: str
    a_clip_id: str
    b_clip_id: str
    duration: float  # seconds
    kind: str = "crossfade"  # or "dip_to_black", etc.

@dataclass
class Track:
    id: str
    kind: str = "video"  # or "audio"
    clips: List[Clip] = field(default_factory=list)
    transitions: List[Transition] = field(default_factory=list)

@dataclass
class Project:
    id: str
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    sample_rate: int = 48000
    tracks: List[Track] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def duration(self) -> float:
        end_times = []
        for tr in self.tracks:
            for c in tr.clips:
                end_times.append(c.offset + c.duration)
        return max(end_times) if end_times else 0.0
