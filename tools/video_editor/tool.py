from __future__ import annotations
from typing import Dict, Any
from pydantic import BaseModel, Field
from .timeline import Project, Track, Clip
from .engine import RenderEngine

class RenderSpec(BaseModel):
    id: str = Field(..., description="Project id")
    fps: float = 30.0
    width: int = 1920
    height: int = 1080
    tracks: list[dict] = Field(default_factory=list)
    output: str = "out.mp4"
    hwaccel: str | None = None
    vcodec: str = "libx264"
    crf: int = 18
    preset: str = "medium"

class VideoEditingTool:
    """Thin wrapper to plug into ONI's tool registry.

    Usage (pseudo):
        tool = VideoEditingTool()
        result_path = tool.run(render_spec_dict)
    """
    def __init__(self):
        self.engine = RenderEngine()

    def make_project(self, spec: RenderSpec) -> Project:
        prj = Project(id=spec.id, fps=spec.fps, width=spec.width, height=spec.height)
        for t in spec.tracks:
            track = Track(id=t["id"], kind=t.get("kind", "video"))
            for c in t.get("clips", []):
                clip = Clip(
                    id=c["id"], src=c["src"], start=float(c.get("start", 0.0)),
                    duration=float(c["duration"]), offset=float(c.get("offset", 0.0)),
                    params=c.get("params", {})
                )
                track.clips.append(clip)
            prj.tracks.append(track)
        return prj

    def run(self, spec_dict: Dict[str, Any]) -> str:
        spec = RenderSpec(**spec_dict)
        project = self.make_project(spec)
        return self.engine.render(project, spec.output, hwaccel=spec.hwaccel,
                                  vcodec=spec.vcodec, crf=spec.crf, preset=spec.preset)
