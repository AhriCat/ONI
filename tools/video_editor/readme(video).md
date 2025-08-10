# ONI Video Editing Tool

### 1) Install
- Ensure **FFmpeg** is installed and in PATH
- `pip install -r requirements.txt`

### 2) Minimal spec
Create `example.json`:
```json
{
  "id": "demo",
  "fps": 30,
  "width": 1280,
  "height": 720,
  "output": "render.mp4",
  "tracks": [
    {
      "id": "v1",
      "kind": "video",
      "clips": [
        { "id": "c1", "src": "clip1.mp4", "start": 0, "duration": 3.0, "offset": 0.0,
          "params": {"nodes": [
            {"name": "transform", "params": {"scale_w": 1280, "scale_h": 720}},
            {"name": "color_adjust", "params": {"saturation": 1.2, "contrast": 1.05}}
          ]}}
        ,
        { "id": "c2", "src": "clip2.mp4", "start": 1.0, "duration": 4.0, "offset": 2.0,
          "params": {"nodes": [
            {"name": "drawtext", "params": {"text": "ONI", "fontsize": 64}}
          ]}}
      ]
    },
    {
      "id": "a1",
      "kind": "audio",
      "clips": [
        { "id": "m1", "src": "music.mp3", "start": 0, "duration": 6.0, "offset": 0.0,
          "params": {"nodes": [
            {"name": "gain", "params": {"db": -3}},
            {"name": "normalize", "params": {}}
          ]}}
      ]
    }
  ]
}
