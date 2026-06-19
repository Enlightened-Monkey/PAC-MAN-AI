# Data Directory Conventions

This folder contains both canonical datasets and experimental artifacts.

## Canonical locations

- `raw/sprites/`: source sprite sheets used by dataset generation.
- `segmentation/train/` and `segmentation/test/`: canonical train/test splits for segmentation.
- `labeled_maps/`: generated labeled-map dataset snapshots that are kept for reuse.

## Experimental / transient locations

- `experiments/`: one-off debug and intermediate exports (old `debug_*`, probes, backups, previews).
- `notebook_map_preview/`, `notebook_map_preview_validation/`: generated snapshots from notebook experiments.
- `real_screenshot_predictions/`: inference outputs for real screenshots.

## Hygiene rules

- Keep large generated outputs out of git unless they are needed for reproducibility.
- Prefer writing notebook-generated files under a timestamped subfolder.
- If a dataset becomes canonical, move it under `segmentation/` and update `dataset_manifest.json`.
