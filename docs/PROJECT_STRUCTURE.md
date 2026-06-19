# Project Structure Guide

This document defines the target structure for keeping the repository maintainable.

## Top-level responsibilities

- `src/`: application and ML code.
- `tests/`: automated tests for `src/` modules.
- `notebooks/`: exploratory analysis, training workflows, and integration notebooks.
- `data/`: datasets, source assets, and generated experiment outputs.
- `models/`: trained checkpoints and exported model weights.
- `reports/`: plots, figures, and final report artifacts.
- `docs/`: reference and project documentation.

## Practical rules

- Keep repository root minimal; avoid dropping raw assets directly at top-level.
- Put source assets under `data/raw/`.
- Keep canonical datasets under stable paths (for this project: `data/segmentation/`).
- Store one-off debug outputs in explicit debug folders and keep them ignored by git.
- If notebook output is useful long-term, promote it into `reports/` or a documented dataset path.

## Snapshot archiving

- Use `./scripts/archive_data_snapshots.sh` to archive transient notebook outputs.
- The script moves populated transient folders into `data/experiments/archive/<timestamp>/` and recreates empty working folders.
- Use `--dry-run` to preview planned moves.

## Current canonical sprite source

- `data/raw/sprites/Arcade - Pac-Man - Miscellaneous - General Sprites.png`
