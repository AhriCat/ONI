from __future__ import annotations
import json, subprocess, shutil
from typing import List, Dict, Any, Optional

class FFmpegError(RuntimeError):
    pass

class FFmpegIO:
    def __init__(self, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"):
        self.ffmpeg = shutil.which(ffmpeg) or ffmpeg
        self.ffprobe = shutil.which(ffprobe) or ffprobe

    def probe(self, path: str) -> Dict[str, Any]:
        cmd = [self.ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
        out = subprocess.run(cmd, capture_output=True)
        if out.returncode != 0:
            raise FFmpegError(out.stderr.decode("utf-8", errors="ignore"))
        return json.loads(out.stdout.decode("utf-8", errors="ignore"))

    def transcode(self, inputs: List[str], filtergraph: Optional[str], output: str,
                  vcodec: str = "libx264", acodec: str = "aac", crf: int = 18, preset: str = "medium",
                  fps: Optional[float] = None, s: Optional[str] = None, audio_only: bool = False,
                  hwaccel: Optional[str] = None, pixel_format: str = "yuv420p"):
        cmd = [self.ffmpeg, "-y"]
        for i in inputs:
            cmd += ["-i", i]
        if hwaccel:
            cmd += ["-hwaccel", hwaccel]
        if filtergraph:
            cmd += ["-filter_complex", filtergraph]
        if audio_only:
            cmd += ["-vn", "-c:a", acodec]
        else:
            if fps:
                cmd += ["-r", str(fps)]
            if s:
                cmd += ["-s", s]
            cmd += ["-c:v", vcodec, "-pix_fmt", pixel_format, "-crf", str(crf), "-preset", preset, "-c:a", acodec]
        cmd += [output]
        out = subprocess.run(cmd, capture_output=True)
        if out.returncode != 0:
            raise FFmpegError(out.stderr.decode("utf-8", errors="ignore"))
        return output

    def concat(self, parts: List[str], output: str, reencode: bool = False) -> str:
        # use concat demuxer when codecs match; otherwise reencode
        if not reencode:
            # create list file
            lst = "\n".join([f"file '{p.replace('\\', '/')}'" for p in parts])
            list_path = output + ".lst"
            with open(list_path, "w", encoding="utf-8") as f:
                f.write(lst)
            cmd = [self.ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output]
        else:
            cmd = [self.ffmpeg, "-y"]
            for p in parts:
                cmd += ["-i", p]
            cmd += ["-filter_complex", f"concat=n={len(parts)}:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]", output]
        out = subprocess.run(cmd, capture_output=True)
        if out.returncode != 0:
            raise FFmpegError(out.stderr.decode("utf-8", errors="ignore"))
        return output
