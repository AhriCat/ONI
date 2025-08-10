from __future__ import annotations
import os
from .io.ffmpeg import FFmpegIO

class PreviewRenderer:
    def __init__(self, io: FFmpegIO | None = None):
        self.io = io or FFmpegIO()

    def proxy(self, src: str, out_dir: str, width: int = 640, fps: float = 24.0) -> str:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(out_dir, f"{base}_proxy.mp4")
        self.io.transcode([src], None, out, vcodec="libx264", crf=28, preset="veryfast", fps=fps, s=f"{width}x-1")
        return out

    def thumbnail(self, src: str, out_dir: str, t: float = 1.0, width: int = 640) -> str:
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(src))[0]
        out = os.path.join(out_dir, f"{base}_{int(t*1000)}ms.jpg")
        # Use ffmpeg single frame extraction via filtergraph
        self.io.transcode([src], f"select='gte(t,{t})',scale={width}:-1,trim=start={t}:end={t+1/999}", out,
                          vcodec="mjpeg", acodec="copy")
        return out
