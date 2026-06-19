#!/usr/bin/env bash
set -euo pipefail

# Archive transient data outputs under data/experiments/archive/<timestamp>/
# Usage:
#   ./scripts/archive_data_snapshots.sh
#   ./scripts/archive_data_snapshots.sh --dry-run

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"
ARCHIVE_ROOT="$DATA_DIR/experiments/archive"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$ARCHIVE_ROOT/$STAMP"

CANDIDATES=(
  "real_screenshot_predictions"
  "notebook_map_preview"
  "notebook_map_preview_validation"
)

mkdir -p "$DEST"
MOVED_ANY=false

for rel in "${CANDIDATES[@]}"; do
  src="$DATA_DIR/$rel"
  if [[ ! -d "$src" ]]; then
    continue
  fi

  if [[ -z "$(find "$src" -mindepth 1 -maxdepth 1 2>/dev/null)" ]]; then
    continue
  fi

  MOVED_ANY=true
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "[dry-run] mv '$src' '$DEST/$rel'"
  else
    mv "$src" "$DEST/$rel"
    mkdir -p "$src"
    echo "Moved: $src -> $DEST/$rel"
    echo "Recreated empty directory: $src"
  fi
done

if [[ "$MOVED_ANY" == "false" ]]; then
  echo "Nothing to archive."
  exit 0
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Dry run completed."
else
  echo "Archive created at: $DEST"
fi
