from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.environment.game_logic import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    COLS,
    GameState,
    ROWS,
    TILE_DOOR,
    TILE_EMPTY,
    TILE_HOUSE,
    TILE_PELLET,
    TILE_POWER,
    TILE_WALL,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_GENERAL_SPRITE_CANDIDATES: tuple[Path, ...] = (
    _REPO_ROOT / "data" / "raw" / "sprites" / "Arcade - Pac-Man - Miscellaneous - General Sprites.png",
    _REPO_ROOT / "Arcade - Pac-Man - Miscellaneous - General Sprites.png",
)
DEFAULT_GENERAL_SPRITE_SHEET = next(
    (path for path in _DEFAULT_GENERAL_SPRITE_CANDIDATES if path.exists()),
    _DEFAULT_GENERAL_SPRITE_CANDIDATES[0],
)

FRAME_WIDTH = COLS * 8
FRAME_HEIGHT = ROWS * 8
TILE_SIZE = 8
ACTOR_SIZE = 16
BLACK_KEY_THRESHOLD = 8

PACMAN_DIRECTIONS: tuple[str, ...] = ("right", "left", "up", "down")
PACMAN_ROTATIONS: dict[str, int] = {
    "right": 0,
    "left": 180,
    "up": 90,
    "down": 270,
}

TILE_LABELS: dict[int, str] = {
    TILE_EMPTY: "empty",
    TILE_WALL: "wall",
    TILE_PELLET: "pellet",
    TILE_POWER: "power_pellet",
    TILE_DOOR: "ghost_door",
    TILE_HOUSE: "ghost_house",
}

CLASS_TO_ID: dict[str, int] = {
    "empty": 0,
    "wall": 1,
    "pellet": 2,
    "power_pellet": 3,
    "ghost_door": 4,
    "ghost_house": 5,
    "pacman": 6,
    "blinky": 7,
    "pinky": 8,
    "inky": 9,
    "clyde": 10,
    "frightened_ghost": 11,
    "ghost_eyes": 12,
    "fruit": 13,
}

GHOST_NAME_TO_LABEL = {
    "Blinky": "blinky",
    "Pinky": "pinky",
    "Inky": "inky",
    "Clyde": "clyde",
}

GHOST_ANIM_DIRECTIONS: tuple[str, ...] = (
    "right",
    "right",
    "left",
    "left",
    "up",
    "up",
    "down",
    "down",
)
GHOST_ANIM_PHASES: tuple[int, ...] = (0, 1, 0, 1, 0, 1, 0, 1)


@dataclass(frozen=True)
class CropBox:
    left: int
    top: int
    right: int
    bottom: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


def _tile_box(row: int, col: int) -> CropBox:
    return CropBox(col * TILE_SIZE, row * TILE_SIZE, (col + 1) * TILE_SIZE, (row + 1) * TILE_SIZE)


def _collectible_color(sprite_sheet: Image.Image, box: CropBox) -> tuple[int, int, int, int]:
    crop = sprite_sheet.crop(box.as_tuple())
    return crop.getpixel((crop.width // 2, crop.height // 2))


def _make_collectible_sprite(color: tuple[int, int, int, int], radius: int) -> Image.Image:
    sprite = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(sprite)
    center = TILE_SIZE // 2
    left = center - radius
    top = center - radius
    right = center + radius - 1
    bottom = center + radius - 1
    draw.ellipse((left, top, right, bottom), fill=color)
    return sprite


GENERAL_SHEET_CROPS: dict[str, CropBox] = {
    "maze_with_pellets": CropBox(0, 0, 224, 248),
    "maze_empty": CropBox(228, 0, 452, 248),
    "pellet": _tile_box(1, 1),
    "power_pellet": _tile_box(3, 1),
    # First movement frame of right-facing Pac-Man (top strip).
    "pacman": CropBox(456, 0, 472, 16),
    # Cherry (fruit for level 1).
    "fruit": CropBox(488, 48, 504, 64),
    "blinky": CropBox(456, 64, 472, 80),
    "pinky": CropBox(456, 80, 472, 96),
    "inky": CropBox(456, 96, 472, 112),
    "clyde": CropBox(456, 112, 472, 128),
    "frightened_ghost": CropBox(584, 64, 600, 80),
    # Eye-only sprite (used by renderer as a representative eaten-ghost state).
    "ghost_eyes": CropBox(584, 80, 600, 96),
}

PACMAN_RIGHT_FRAMES: tuple[CropBox, ...] = (
    # Correct movement cycle from the top row: open -> half-open -> closed -> half-open.
    CropBox(456, 0, 472, 16),
    CropBox(472, 0, 488, 16),
    CropBox(488, 0, 504, 16),
    CropBox(472, 0, 488, 16),
)

GHOST_ROW_BY_NAME: dict[str, int] = {
    "blinky": 64,
    "pinky": 80,
    "inky": 96,
    "clyde": 112,
}

FRIGHTENED_BLUE_FRAMES: tuple[CropBox, ...] = (
    CropBox(584, 64, 600, 80),
    CropBox(600, 64, 616, 80),
)

FRIGHTENED_FLASH_FRAMES: tuple[CropBox, ...] = (
    CropBox(616, 64, 632, 80),
    CropBox(632, 64, 648, 80),
)

EYES_FRAMES: tuple[CropBox, ...] = (
    CropBox(584, 80, 600, 96),
    CropBox(600, 80, 616, 96),
    CropBox(616, 80, 632, 96),
    CropBox(632, 80, 648, 96),
)

FRUIT_VARIANTS: tuple[tuple[str, CropBox], ...] = (
    ("cherry", CropBox(488, 48, 504, 64)),
    ("strawberry", CropBox(504, 48, 520, 64)),
    ("orange", CropBox(520, 48, 536, 64)),
    ("apple", CropBox(536, 48, 552, 64)),
    ("melon", CropBox(552, 48, 568, 64)),
    ("galaxian", CropBox(568, 48, 584, 64)),
    ("bell", CropBox(584, 48, 600, 64)),
    ("key", CropBox(600, 48, 616, 64)),
)

BACKGROUND_KEYS = {"maze_with_pellets", "maze_empty"}


def _key_black_to_alpha(image: Image.Image, threshold: int = BLACK_KEY_THRESHOLD) -> Image.Image:
    """Make near-black pixels transparent to remove sprite-sheet background."""
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    rgb = arr[:, :, :3]
    near_black = np.all(rgb <= threshold, axis=2)
    arr[near_black, 3] = 0
    return Image.fromarray(arr, mode="RGBA")


def _trim_to_content(image: Image.Image) -> Image.Image:
    """Trim transparent margins; keep original image if fully transparent."""
    rgba = image.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return rgba
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return rgba.crop((x0, y0, x1, y1))


def _fit_to_canvas(image: Image.Image, canvas_size: tuple[int, int]) -> Image.Image:
    """Resize while preserving aspect ratio and center on transparent canvas."""
    target_w, target_h = canvas_size
    rgba = image.convert("RGBA")
    src_w, src_h = rgba.size
    if src_w == 0 or src_h == 0:
        return Image.new("RGBA", canvas_size, (0, 0, 0, 0))

    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = rgba.resize((new_w, new_h), Image.Resampling.NEAREST)

    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    off_x = (target_w - new_w) // 2
    off_y = (target_h - new_h) // 2
    canvas.alpha_composite(resized, (off_x, off_y))
    return canvas


def _foreground_bbox_non_black(image: Image.Image, threshold: int = BLACK_KEY_THRESHOLD) -> tuple[int, int, int, int] | None:
    rgb = np.array(image.convert("RGB"))
    fg = np.any(rgb > threshold, axis=2)
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


class PacmanSpriteSheetExtractor:
    def __init__(self, general_sheet_path: str | Path = DEFAULT_GENERAL_SPRITE_SHEET) -> None:
        self.general_sheet_path = Path(general_sheet_path)

    def extract_catalog(self) -> dict[str, Image.Image]:
        sprite_sheet = Image.open(self.general_sheet_path).convert("RGBA")
        catalog: dict[str, Image.Image] = {}
        for name, box in GENERAL_SHEET_CROPS.items():
            if name == "pellet":
                catalog[name] = _make_collectible_sprite(_collectible_color(sprite_sheet, box), radius=1)
                continue
            if name == "power_pellet":
                catalog[name] = _make_collectible_sprite(_collectible_color(sprite_sheet, box), radius=2)
                continue
            cropped = sprite_sheet.crop(box.as_tuple())
            if name in BACKGROUND_KEYS:
                catalog[name] = cropped
                continue
            processed = _trim_to_content(_key_black_to_alpha(cropped))
            catalog[name] = processed
        return catalog

    def extract_structured_catalog(self) -> dict[str, Image.Image]:
        sprite_sheet = Image.open(self.general_sheet_path).convert("RGBA")
        structured: dict[str, Image.Image] = {}

        for name in ("maze_with_pellets", "maze_empty"):
            structured[f"backgrounds/maze/{name}.png"] = sprite_sheet.crop(
                GENERAL_SHEET_CROPS[name].as_tuple()
            )

        for name in ("pellet", "power_pellet"):
            radius = 1 if name == "pellet" else 2
            structured[f"tiles/collectibles/{name}.png"] = _make_collectible_sprite(
                _collectible_color(sprite_sheet, GENERAL_SHEET_CROPS[name]),
                radius=radius,
            )

        pacman_right = [
            _trim_to_content(_key_black_to_alpha(sprite_sheet.crop(box.as_tuple())))
            for box in PACMAN_RIGHT_FRAMES
        ]
        for direction in PACMAN_DIRECTIONS:
            rotation = PACMAN_ROTATIONS[direction]
            for frame_idx, frame in enumerate(pacman_right):
                rotated = frame.rotate(rotation, expand=True)
                key = (
                    f"characters/pacman/normal/{direction}/"
                    f"frame_{frame_idx:02d}.png"
                )
                structured[key] = rotated

        for ghost_name, row in GHOST_ROW_BY_NAME.items():
            for idx in range(8):
                left = 456 + idx * 16
                box = CropBox(left, row, left + 16, row + 16)
                direction = GHOST_ANIM_DIRECTIONS[idx]
                phase = GHOST_ANIM_PHASES[idx]
                frame = _trim_to_content(_key_black_to_alpha(sprite_sheet.crop(box.as_tuple())))
                key = (
                    f"characters/ghosts/{ghost_name}/normal/{direction}/"
                    f"frame_{phase:02d}.png"
                )
                structured[key] = frame

        for idx, box in enumerate(FRIGHTENED_BLUE_FRAMES):
            frame = _trim_to_content(_key_black_to_alpha(sprite_sheet.crop(box.as_tuple())))
            structured[
                f"characters/ghosts/shared/frightened/blue/frame_{idx:02d}.png"
            ] = frame

        for idx, box in enumerate(FRIGHTENED_FLASH_FRAMES):
            frame = _trim_to_content(_key_black_to_alpha(sprite_sheet.crop(box.as_tuple())))
            structured[
                f"characters/ghosts/shared/frightened/flash/frame_{idx:02d}.png"
            ] = frame

        for idx, box in enumerate(EYES_FRAMES):
            frame = _trim_to_content(_key_black_to_alpha(sprite_sheet.crop(box.as_tuple())))
            structured[f"characters/ghosts/shared/eyes/frame_{idx:02d}.png"] = frame

        for fruit_name, box in FRUIT_VARIANTS:
            frame = _trim_to_content(_key_black_to_alpha(sprite_sheet.crop(box.as_tuple())))
            structured[f"items/fruits/{fruit_name}/frame_00.png"] = frame

        return structured

    def audit_catalog(self, output_dir: str | Path) -> Path:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        sprite_sheet = Image.open(self.general_sheet_path).convert("RGBA")
        processed = self.extract_catalog()

        report: dict[str, Any] = {
            "sheet": str(self.general_sheet_path),
            "threshold": BLACK_KEY_THRESHOLD,
            "assets": {},
        }

        for name, box in GENERAL_SHEET_CROPS.items():
            raw = sprite_sheet.crop(box.as_tuple())
            raw_bbox = _foreground_bbox_non_black(raw)
            proc = processed[name]
            proc_alpha = np.array(proc.convert("RGBA").getchannel("A"))
            ys, xs = np.where(proc_alpha > 0)
            if len(xs) == 0:
                alpha_bbox = None
                alpha_occ = 0.0
            else:
                alpha_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                alpha_occ = float((proc_alpha > 0).mean())

            report["assets"][name] = {
                "crop": list(box.as_tuple()),
                "raw_size": list(raw.size),
                "raw_non_black_bbox": list(raw_bbox) if raw_bbox is not None else None,
                "processed_size": list(proc.size),
                "processed_alpha_bbox": list(alpha_bbox) if alpha_bbox is not None else None,
                "processed_alpha_occupancy": round(alpha_occ, 4),
            }

        report_path = output_root / "asset_audit.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report_path

    def export_catalog(self, output_dir: str | Path) -> dict[str, Path]:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        structured_catalog = self.extract_structured_catalog()
        exported: dict[str, Path] = {}
        for relative_path, image in structured_catalog.items():
            path = output_root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            exported[relative_path] = path

        backgrounds_index = {
            "maze": {
                "with_pellets": "backgrounds/maze/maze_with_pellets.png",
                "empty": "backgrounds/maze/maze_empty.png",
            }
        }
        (output_root / "backgrounds" / "index.json").write_text(
            json.dumps(backgrounds_index, indent=2), encoding="utf-8"
        )

        tiles_index = {
            "collectibles": {
                "pellet": "tiles/collectibles/pellet.png",
                "power_pellet": "tiles/collectibles/power_pellet.png",
            }
        }
        (output_root / "tiles" / "index.json").write_text(
            json.dumps(tiles_index, indent=2), encoding="utf-8"
        )

        characters_index = {
            "pacman": {
                "state": "normal",
                "directions": {
                    direction: f"characters/pacman/normal/{direction}/"
                    for direction in PACMAN_DIRECTIONS
                },
            },
            "ghosts": {
                "normal": {
                    ghost_name: {
                        direction: (
                            f"characters/ghosts/{ghost_name}/normal/{direction}/"
                        )
                        for direction in ("right", "left", "up", "down")
                    }
                    for ghost_name in GHOST_ROW_BY_NAME
                },
                "shared": {
                    "frightened": {
                        "blue": "characters/ghosts/shared/frightened/blue/",
                        "flash": "characters/ghosts/shared/frightened/flash/",
                    },
                    "eyes": "characters/ghosts/shared/eyes/",
                },
            },
        }
        (output_root / "characters" / "index.json").write_text(
            json.dumps(characters_index, indent=2), encoding="utf-8"
        )

        items_index = {
            "fruits": {
                fruit_name: f"items/fruits/{fruit_name}/frame_00.png"
                for fruit_name, _ in FRUIT_VARIANTS
            }
        }
        (output_root / "items" / "index.json").write_text(
            json.dumps(items_index, indent=2), encoding="utf-8"
        )

        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "sheet": str(self.general_sheet_path),
                    "frame_size": [FRAME_WIDTH, FRAME_HEIGHT],
                    "tile_size": TILE_SIZE,
                    "layout": {
                        "backgrounds": [
                            "backgrounds/maze/maze_with_pellets.png",
                            "backgrounds/maze/maze_empty.png",
                        ],
                        "tiles": [
                            "tiles/collectibles/pellet.png",
                            "tiles/collectibles/power_pellet.png",
                        ],
                        "characters": {
                            "pacman": {
                                direction: (
                                    f"characters/pacman/normal/{direction}/"
                                )
                                for direction in PACMAN_DIRECTIONS
                            },
                            "ghosts": {
                                ghost_name: (
                                    f"characters/ghosts/{ghost_name}/normal/"
                                )
                                for ghost_name in GHOST_ROW_BY_NAME
                            },
                            "frightened": "characters/ghosts/shared/frightened/",
                            "eyes": "characters/ghosts/shared/eyes/",
                        },
                        "items": {
                            "fruits": "items/fruits/",
                        },
                        "indexes": {
                            "backgrounds": "backgrounds/index.json",
                            "tiles": "tiles/index.json",
                            "characters": "characters/index.json",
                            "items": "items/index.json",
                        },
                    },
                    "crops": {
                        name: list(box.as_tuple())
                        for name, box in GENERAL_SHEET_CROPS.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        exported["manifest"] = manifest_path
        return exported


class PacmanMapDatasetGenerator:
    def __init__(
        self,
        general_sheet_path: str | Path = DEFAULT_GENERAL_SPRITE_SHEET,
        tile_size: int = TILE_SIZE,
        actor_size: int = ACTOR_SIZE,
    ) -> None:
        self.tile_size = int(tile_size)
        self.actor_size = int(actor_size)
        self.extractor = PacmanSpriteSheetExtractor(general_sheet_path)
        self.catalog = self.extractor.extract_catalog()
        self.frame_size = (COLS * self.tile_size, ROWS * self.tile_size)
        self._maze_background = self.catalog["maze_empty"].resize(
            self.frame_size, Image.Resampling.NEAREST
        )
        self._pellet_tile = self.catalog["pellet"].resize(
            (self.tile_size, self.tile_size), Image.Resampling.NEAREST
        )
        self._power_tile = self.catalog["power_pellet"].resize(
            (self.tile_size, self.tile_size), Image.Resampling.NEAREST
        )

    def render_state(self, state: GameState) -> tuple[Image.Image, dict[str, Any], Image.Image]:
        frame = self._maze_background.copy()
        mask = Image.new("L", self.frame_size, color=CLASS_TO_ID["empty"])
        self._paint_tile_mask(mask, state.maze)

        for row in range(ROWS):
            for col in range(COLS):
                cell = int(state.maze[row, col])
                pixel_pos = (col * self.tile_size, row * self.tile_size)
                if cell == TILE_PELLET:
                    frame.alpha_composite(self._pellet_tile, pixel_pos)
                elif cell == TILE_POWER:
                    frame.alpha_composite(self._power_tile, pixel_pos)

        objects: list[dict[str, Any]] = []

        pacman_sprite = self._pacman_sprite_for_direction(state.pacman_dir)
        pacman_bbox = self._place_actor(frame, mask, pacman_sprite, state.pacman_pos, CLASS_TO_ID["pacman"])
        objects.append(
            {
                "label": "pacman",
                "tile_position": [state.pacman_pos[1], state.pacman_pos[0]],
                "bbox": list(pacman_bbox),
                "direction": self._direction_name(state.pacman_dir),
            }
        )

        for ghost in state.ghosts:
            label, sprite = self._ghost_visual(ghost, frightened=state.frightened_timer > 0)
            bbox = self._place_actor(frame, mask, sprite, ghost.pos, CLASS_TO_ID[label])
            objects.append(
                {
                    "label": label,
                    "tile_position": [ghost.pos[1], ghost.pos[0]],
                    "bbox": list(bbox),
                    "direction": self._direction_name(ghost.direction),
                    "in_house": bool(ghost.in_house),
                    "eaten": bool(ghost.eaten),
                }
            )

        if state.fruit_active:
            bbox = self._place_actor(
                frame,
                mask,
                self._prepare_actor_sprite(self.catalog["fruit"]),
                state.fruit_pos,
                CLASS_TO_ID["fruit"],
            )
            objects.append(
                {
                    "label": "fruit",
                    "tile_position": [state.fruit_pos[1], state.fruit_pos[0]],
                    "bbox": list(bbox),
                }
            )

        annotation = {
            "image_size": {"width": self.frame_size[0], "height": self.frame_size[1]},
            "tile_size": self.tile_size,
            "class_to_id": CLASS_TO_ID,
            "objects": objects,
            "grid_labels": [
                [TILE_LABELS[int(cell)] for cell in row]
                for row in state.maze.tolist()
            ],
            "state": {
                "score": state.score,
                "lives": state.lives,
                "step_count": state.step_count,
                "pellets_eaten": state.pellets_eaten,
                "mode": "frightened" if state.frightened_timer > 0 else state.mode,
                "fruit_active": state.fruit_active,
            },
            "hud": {
                "score": state.score,
                "lives": state.lives,
                "level": int(getattr(state, "level", 1)),
                "fruit_active": state.fruit_active,
            },
        }
        return frame.convert("RGB"), annotation, mask

    def generate_dataset(
        self,
        output_dir: str | Path,
        sample_count: int,
        *,
        seed: int = 0,
        max_random_steps: int = 200,
    ) -> list[Path]:
        output_root = Path(output_dir)
        images_dir = output_root / "images"
        masks_dir = output_root / "masks"
        labels_dir = output_root / "labels"
        assets_dir = output_root / "assets"
        for directory in (images_dir, masks_dir, labels_dir, assets_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.extractor.export_catalog(assets_dir)
        rng = np.random.default_rng(seed)
        created_annotations: list[Path] = []

        for index in range(sample_count):
            sample_seed = int(rng.integers(0, 2**31 - 1))
            state = GameState(seed=sample_seed)
            rollout_steps = int(rng.integers(0, max_random_steps + 1))
            for _ in range(rollout_steps):
                if state.is_terminal():
                    break
                state.step(int(rng.integers(0, 4)))

            frame, annotation, mask = self.render_state(state)
            stem = f"sample_{index:05d}"
            image_path = images_dir / f"{stem}.png"
            mask_path = masks_dir / f"{stem}.png"
            label_path = labels_dir / f"{stem}.json"
            frame.save(image_path)
            mask.save(mask_path)
            label_path.write_text(json.dumps(annotation, indent=2), encoding="utf-8")
            created_annotations.append(label_path)

        dataset_manifest = {
            "sample_count": sample_count,
            "frame_size": list(self.frame_size),
            "tile_size": self.tile_size,
            "class_to_id": CLASS_TO_ID,
            "general_sheet_path": str(self.extractor.general_sheet_path),
        }
        (output_root / "dataset_manifest.json").write_text(
            json.dumps(dataset_manifest, indent=2), encoding="utf-8"
        )
        return created_annotations

    def _paint_tile_mask(self, mask: Image.Image, maze: np.ndarray) -> None:
        mask_arr = np.full((self.frame_size[1], self.frame_size[0]), CLASS_TO_ID["empty"], dtype=np.uint8)
        for row in range(ROWS):
            y0 = row * self.tile_size
            y1 = y0 + self.tile_size
            for col in range(COLS):
                x0 = col * self.tile_size
                x1 = x0 + self.tile_size
                label_name = TILE_LABELS[int(maze[row, col])]
                mask_arr[y0:y1, x0:x1] = CLASS_TO_ID[label_name]
        painted = Image.fromarray(mask_arr, mode="L")
        mask.paste(painted)

    def _pacman_sprite_for_direction(self, direction: int) -> Image.Image:
        sprite = self._prepare_actor_sprite(self.catalog["pacman"])
        rotations = {
            ACTION_RIGHT: 0,
            ACTION_DOWN: 270,
            ACTION_LEFT: 180,
            ACTION_UP: 90,
        }
        return sprite.rotate(rotations[direction], expand=True)

    def _ghost_visual(self, ghost: Any, frightened: bool) -> tuple[str, Image.Image]:
        if ghost.eaten:
            return "ghost_eyes", self._prepare_actor_sprite(self.catalog["ghost_eyes"])
        if frightened and not ghost.in_house:
            return "frightened_ghost", self._prepare_actor_sprite(self.catalog["frightened_ghost"])
        label = GHOST_NAME_TO_LABEL[ghost.name]
        return label, self._prepare_actor_sprite(self.catalog[label])

    def _prepare_actor_sprite(self, image: Image.Image) -> Image.Image:
        return _fit_to_canvas(image, (self.actor_size, self.actor_size))

    def _place_actor(
        self,
        frame: Image.Image,
        mask: Image.Image,
        sprite: Image.Image,
        tile_pos: tuple[int, int],
        class_id: int,
    ) -> tuple[int, int, int, int]:
        row, col = tile_pos
        sprite = sprite.convert("RGBA")
        center_x = col * self.tile_size + (self.tile_size // 2)
        center_y = row * self.tile_size + (self.tile_size // 2)
        left = center_x - (sprite.width // 2)
        top = center_y - (sprite.height // 2)
        frame.alpha_composite(sprite, (left, top))

        alpha = np.array(sprite.getchannel("A"))
        if np.any(alpha > 0):
            mask_arr = np.array(mask)
            x0 = max(left, 0)
            y0 = max(top, 0)
            x1 = min(left + sprite.width, self.frame_size[0])
            y1 = min(top + sprite.height, self.frame_size[1])

            sprite_x0 = x0 - left
            sprite_y0 = y0 - top
            sprite_x1 = sprite_x0 + (x1 - x0)
            sprite_y1 = sprite_y0 + (y1 - y0)
            sprite_alpha = alpha[sprite_y0:sprite_y1, sprite_x0:sprite_x1] > 0
            mask_slice = mask_arr[y0:y1, x0:x1]
            mask_slice[sprite_alpha] = class_id
            mask.paste(Image.fromarray(mask_arr, mode="L"))

        return (left, top, sprite.width, sprite.height)

    @staticmethod
    def _direction_name(direction: int) -> str:
        return {
            ACTION_UP: "up",
            ACTION_DOWN: "down",
            ACTION_LEFT: "left",
            ACTION_RIGHT: "right",
        }[direction]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Pac-Man sprites and generate labelled frame datasets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_assets = subparsers.add_parser("extract-assets", help="Save cropped assets to disk.")
    extract_assets.add_argument("--sheet", type=Path, default=DEFAULT_GENERAL_SPRITE_SHEET)
    extract_assets.add_argument("--output-dir", type=Path, required=True)

    audit_assets = subparsers.add_parser("audit-assets", help="Run asset quality audit and write JSON report.")
    audit_assets.add_argument("--sheet", type=Path, default=DEFAULT_GENERAL_SPRITE_SHEET)
    audit_assets.add_argument("--output-dir", type=Path, required=True)

    generate = subparsers.add_parser("generate", help="Generate labelled Pac-Man frames.")
    generate.add_argument("--sheet", type=Path, default=DEFAULT_GENERAL_SPRITE_SHEET)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--samples", type=int, default=64)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--max-random-steps", type=int, default=200)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "extract-assets":
        extractor = PacmanSpriteSheetExtractor(args.sheet)
        extractor.export_catalog(args.output_dir)
        return

    if args.command == "audit-assets":
        extractor = PacmanSpriteSheetExtractor(args.sheet)
        extractor.audit_catalog(args.output_dir)
        return

    generator = PacmanMapDatasetGenerator(args.sheet)
    generator.generate_dataset(
        args.output_dir,
        sample_count=args.samples,
        seed=args.seed,
        max_random_steps=args.max_random_steps,
    )


if __name__ == "__main__":
    main()