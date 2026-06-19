from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.dataset import (
    DEFAULT_GENERAL_SPRITE_SHEET,
    PacmanMapDatasetGenerator,
    PacmanSpriteSheetExtractor,
)
from src.environment.game_logic import GameState


def test_sprite_extractor_exports_required_assets(tmp_path: Path):
    extractor = PacmanSpriteSheetExtractor(DEFAULT_GENERAL_SPRITE_SHEET)
    exported = extractor.export_catalog(tmp_path)

    assert (tmp_path / "backgrounds" / "maze" / "maze_empty.png").exists()
    assert (
        tmp_path / "characters" / "pacman" / "normal" / "right" / "frame_00.png"
    ).exists()
    assert (
        tmp_path / "characters" / "ghosts" / "blinky" / "normal" / "right" / "frame_00.png"
    ).exists()
    assert (
        tmp_path / "characters" / "ghosts" / "pinky" / "normal" / "right" / "frame_00.png"
    ).exists()
    assert (tmp_path / "items" / "fruits" / "cherry" / "frame_00.png").exists()
    assert (tmp_path / "backgrounds" / "index.json").exists()
    assert (tmp_path / "tiles" / "index.json").exists()
    assert (tmp_path / "characters" / "index.json").exists()
    assert (tmp_path / "items" / "index.json").exists()
    assert exported["manifest"].exists()

    blinky = np.array(
        generator_image(
            tmp_path
            / "characters"
            / "ghosts"
            / "blinky"
            / "normal"
            / "right"
            / "frame_00.png"
        )
    )
    pinky = np.array(
        generator_image(
            tmp_path
            / "characters"
            / "ghosts"
            / "pinky"
            / "normal"
            / "right"
            / "frame_00.png"
        )
    )
    assert blinky.shape[2] == 4
    assert (blinky[:, :, 3] == 0).any()
    blinky_rgb = blinky[blinky[:, :, 3] > 0][:, :3].mean(axis=0)
    pinky_rgb = pinky[pinky[:, :, 3] > 0][:, :3].mean(axis=0)
    assert blinky_rgb[1] < pinky_rgb[1]


def generator_image(path: Path):
    from PIL import Image

    return Image.open(path).convert("RGBA")


def test_asset_audit_generates_json(tmp_path: Path):
    extractor = PacmanSpriteSheetExtractor(DEFAULT_GENERAL_SPRITE_SHEET)
    report_path = extractor.audit_catalog(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "assets" in report
    assert "pacman" in report["assets"]
    assert report["assets"]["blinky"]["processed_alpha_occupancy"] > 0.0


def test_render_state_returns_frame_annotation_and_mask():
    generator = PacmanMapDatasetGenerator(DEFAULT_GENERAL_SPRITE_SHEET)
    state = GameState(seed=7)

    frame, annotation, mask = generator.render_state(state)

    assert frame.size == (224, 248)
    assert mask.size == (224, 248)
    assert annotation["objects"][0]["label"] == "pacman"
    assert any(obj["label"] == "blinky" for obj in annotation["objects"])
    assert "hud" in annotation
    assert annotation["hud"]["lives"] == state.lives
    assert annotation["hud"]["score"] == state.score
    mask_values = np.array(mask)
    assert mask_values.max() >= 7


def test_generate_dataset_writes_images_masks_and_labels(tmp_path: Path):
    generator = PacmanMapDatasetGenerator(DEFAULT_GENERAL_SPRITE_SHEET)
    created = generator.generate_dataset(tmp_path, sample_count=3, seed=3, max_random_steps=5)

    assert len(created) == 3
    assert len(list((tmp_path / "images").glob("*.png"))) == 3
    assert len(list((tmp_path / "masks").glob("*.png"))) == 3
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_count"] == 3