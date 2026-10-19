"""Independent entry point; run_comparison.py and the team's scripts remain unchanged."""

from pathlib import Path

from light_rcnn.comparison.cli import main

if __name__ == "__main__":
    main(root=Path(__file__).resolve().parents[1])
