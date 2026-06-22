#!/usr/bin/env python3
"""Render one example labeled map with bounding boxes and labels.

Generates a fresh game state that has a fruit active (after 70 pellets eaten),
renders it with the sprite generator (so actors are visible), and writes an
annotated preview PNG with colour-coded bounding boxes.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_DIR = PROJECT_ROOT / "data" / "labeled_maps" / "images"
OUTPUT_PATH = IMAGE_DIR / "example_map_bbox.png"


def _build_sample() -> tuple[Image.Image, dict]:
    """Build a game state with fruit active and render it."""
    from src.environment.game_logic import GameState, ACTION_RIGHT, ACTION_LEFT, DIRECTION_DELTAS
    from src.dataset.pacman_map_dataset import PacmanMapDatasetGenerator

    state = GameState(seed=7)
    # Run using a greedy "keep moving" policy to eat pellets
    action = ACTION_RIGHT
    for _ in range(600):
        # Try to keep moving in current direction; if blocked, try all directions
        moved = False
        for candidate in [action, ACTION_RIGHT, ACTION_LEFT, ACTION_RIGHT + 2, ACTION_LEFT + 2]:
            candidate = candidate % 4
            dr, dc = DIRECTION_DELTAS[candidate]
            target = (state.pacman_pos[0] + dr, state.pacman_pos[1] + dc)
            if state._walkable_for_pacman(target):
                action = candidate
                moved = True
                break
        if not moved:
            action = (action + 1) % 4
        state.step(action)
        if state.fruit_active:
            break

    # If fruit still didn't spawn, force it on for the render
    if not state.fruit_active:
        state.fruit_active = True
        state.fruit_timer = 95

    gen = PacmanMapDatasetGenerator()
    pil_img, annotation, _ = gen.render_state(state, forced_fruit_idx=0)
    return pil_img.convert("RGBA"), annotation


def draw_example(image: Image.Image, annotation: dict, output_path: Path) -> Path:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    palette = {
        "pacman":          (255, 220,   0, 255),
        "blinky":          (255,  64,  64, 255),
        "pinky":           (255, 160, 220, 255),
        "inky":            ( 64, 224, 255, 255),
        "clyde":           (255, 180,  80, 255),
        "frightened_ghost":(  0, 100, 255, 255),
        "ghost_eyes":      (255, 255, 255, 255),
    }
    fruit_color = (120, 255, 120, 255)

    for obj in annotation.get("objects", []):
        x, y, w, h = [int(v) for v in obj["bbox"]]
        cls = str(obj["label"])
        color = palette.get(cls, fruit_color)
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=2)
        text = cls
        text_box = draw.textbbox((0, 0), text, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        tx = max(0, min(canvas.width - text_w - 4, x + 2))
        ty = max(0, y - text_h - 4)
        draw.rectangle((tx - 2, ty - 1, tx + text_w + 2, ty + text_h + 1), fill=(0, 0, 0, 180))
        draw.text((tx, ty), text, fill=color, font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> None:
    image, annotation = _build_sample()
    objects = [o["label"] for o in annotation.get("objects", [])]
    output = draw_example(image, annotation, OUTPUT_PATH)
    print(f"Objects: {objects}")
    print(f"Wrote example map to {output}")


if __name__ == "__main__":
    main()
