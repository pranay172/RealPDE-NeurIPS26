#!/usr/bin/env python3
"""Train the 3000-update CNO backbone on all 81 valid real trajectories."""

from pathlib import Path

from train_cno import main


REPO = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    main(
        default_config=REPO / "configs" / "backbone.json",
        default_artifacts=REPO / "artifacts" / "backbone",
    )
