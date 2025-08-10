#!/usr/bin/env python3
import argparse, json
from ONI=Public.tools.videoeditor.tool import VideoEditingTool

def main():
    p = argparse.ArgumentParser("oni_video")
    p.add_argument("spec", help="Path to JSON render spec")
    args = p.parse_args()
    tool = VideoEditingTool()
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)
    out = tool.run(spec)
    print(out)

if __name__ == "__main__":
    main()
