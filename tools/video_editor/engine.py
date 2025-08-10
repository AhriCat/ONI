from __future__ import annotations
from typing import Dict, List, Any, Optional
from .timeline import Project, Track, Clip, Transition
from .nodes import get as get_node
from .io.ffmpeg import FFmpegIO

class RenderEngine:
    def __init__(self, io: FFmpegIO | None = None):
        self.io = io or FFmpegIO()

    def _build_filtergraph_for_clip(self, clip: Clip, track_kind: str) -> str:
        # Gather per-clip node chains from params: expects params["nodes"] = [{name, params}, ...]
        chain = []
        nodes = clip.params.get("nodes", [])
        for nd in nodes:
            cls = get_node(nd["name"])  # raises if missing
            filt = cls().apply(nd.get("params", {}))
            if filt:
                chain.append(filt)
        return ",".join(chain)

    def render(self, project: Project, output: str, hwaccel: Optional[str] = None,
               vcodec: str = "libx264", crf: int = 18, preset: str = "medium") -> str:
        # Strategy: for each video track, lay out clips with trim/setpts and overlay; audio track: amix/aconcat
        # For simplicity: 1 video track + 1 audio track minimal working example; extendable to N tracks.
        vtracks = [t for t in project.tracks if t.kind == "video"]
        atracks = [t for t in project.tracks if t.kind == "audio"]
        inputs: List[str] = []
        filters: List[str] = []
        v_out_label = None
        a_out_label = None
        input_idx = 0

        # VIDEO
        if vtracks:
            v = vtracks[0]
            prev_label = None
            overlay_count = 0
            for c in sorted(v.clips, key=lambda x: x.offset):
                inputs.append(c.src)
                # trim to duration and set timebase
                vf_chain = [f"trim=start={c.start}:duration={c.duration}", "setpts=PTS-STARTPTS"]
                clip_effects = self._build_filtergraph_for_clip(c, "video")
                if clip_effects:
                    vf_chain.append(clip_effects)
                vf = ",".join(vf_chain)
                label = f"v{input_idx}"
                filters.append(f"[{input_idx}:v]{vf}[{label}]")
                if prev_label is None:
                    prev_label = label
                else:
                    # place via overlay with time offset
                    # pad previous to full timeline
                    start_ts = c.offset
                    of = f"[{prev_label}][{label}]overlay=eof_action=pass:enable='gte(t,{start_ts})'[ov{overlay_count}]"
                    filters.append(of)
                    prev_label = f"ov{overlay_count}"
                    overlay_count += 1
                input_idx += 1
            v_out_label = prev_label

        # AUDIO (simple concat or amix if overlap)
        if atracks:
            a = atracks[0]
            prev_label = None
            mix_count = 0
            for c in sorted(a.clips, key=lambda x: x.offset):
                inputs.append(c.src)
                af_chain = [f"atrim=start={c.start}:duration={c.duration}", "asetpts=PTS-STARTPTS"]
                clip_effects = self._build_filtergraph_for_clip(c, "audio")
                if clip_effects:
                    af_chain.append(clip_effects)
                af = ",".join(af_chain)
                label = f"a{input_idx}"
                filters.append(f"[{input_idx}:a]{af}[{label}]")
                if prev_label is None:
                    prev_label = label
                else:
                    filters.append(f"[{prev_label}][{label}]amix=inputs=2:normalize=0[mix{mix_count}]")
                    prev_label = f"mix{mix_count}"
                    mix_count += 1
                input_idx += 1
            a_out_label = prev_label

        # Map outputs
        map_flags = []
        if v_out_label:
            map_flags += ["-map", f"[{v_out_label}]"]
        if a_out_label:
            map_flags += ["-map", f"[{a_out_label}]"]

        filtergraph = ";".join(filters) if filters else None
        return self.io.transcode(inputs, filtergraph, output, vcodec=vcodec, crf=crf, preset=preset,
                                 fps=project.fps, s=f"{project.width}x{project.height}", hwaccel=hwaccel)
