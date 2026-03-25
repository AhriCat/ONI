# File: evolution/monitor.py
"""Monitor ONI-DGM evolution progress."""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def get_evolution_status(archive_dir: str) -> Dict:
    """Get the latest evolution status from the archive state log."""
    archive_path = Path(archive_dir)
    state_file = archive_path / "archive_state.jsonl"
    if state_file.exists():
        with open(state_file) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if lines:
            return json.loads(lines[-1])
    return {'status': 'not_started'}


def get_generation_history(archive_dir: str) -> List[Dict]:
    """Load full generation-by-generation history from dgm_metadata.jsonl."""
    path = Path(archive_dir) / "dgm_metadata.jsonl"
    if not path.exists():
        return []
    history = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return history


def print_progress(archive_dir: str):
    """Print a formatted evolution progress report."""
    status = get_evolution_status(archive_dir)
    history = get_generation_history(archive_dir)

    print("\n" + "=" * 60)
    print("ONI-DGM EVOLUTION STATUS")
    print("=" * 60)
    print(f"  Generation:  {status.get('generation', 'N/A')}")
    print(f"  Variants:    {status.get('num_variants', 'N/A')}")
    best_score = status.get('best_score', 'N/A')
    if isinstance(best_score, float):
        print(f"  Best Score:  {best_score:.4f}")
    else:
        print(f"  Best Score:  {best_score}")
    print(f"  Best ID:     {status.get('best_variant', 'N/A')}")
    print(f"  Timestamp:   {status.get('timestamp', 'N/A')}")

    if history:
        print(f"\n  Score progression (last 10 gens):")
        for entry in history[-10:]:
            gen = entry.get('generation', '?')
            score = entry.get('best_score', 0.0)
            bar = '█' * int(score * 20)
            print(f"    Gen {gen:3d}: {score:.4f} |{bar:<20}|")

    print("=" * 60)


if __name__ == "__main__":
    import sys
    archive_dir = sys.argv[1] if len(sys.argv) > 1 else "./oni_archive"
    print_progress(archive_dir)
