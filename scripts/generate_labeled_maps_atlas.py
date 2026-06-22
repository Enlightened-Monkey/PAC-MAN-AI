#!/usr/bin/env python3
"""Build a single labeled atlas image for all labeled-map assets.

Each PNG under data/labeled_maps/assets is placed exactly once on the sheet.
The asset is resized to fit a fixed tile, a bbox is drawn around the placed
thumbnail, and the relative asset path is rendered below it.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "data" / "labeled_maps" / "assets"
OUTPUT_PATH = PROJECT_ROOT / "data" / "labeled_maps" / "images" / "asset_atlas_bbox.png"


def collect_png_assets(root: Path) -> list[Path]:
    pacman_basic = root / "characters" / "pacman" / "normal" / "right" / "frame_02.png"
    assets = []
    for path in root.rglob("*.png"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = relative.parts
        if len(parts) >= 5 and parts[0] == "characters" and parts[1] == "pacman" and parts[2] == "normal":
            # Use one neutral Pac-Man frame as the canonical visualization asset.
            if path != pacman_basic:
                continue
        assets.append(path)
    assets.sort(key=lambda path: str(path.relative_to(root)).lower())
    return assets


def fit_into_box(image: Image.Image, box_w: int, box_h: int) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail((box_w, box_h), Image.Resampling.LANCZOS)
    return fitted


def short_label(asset_path: Path) -> str:
    parts = asset_path.relative_to(ASSET_ROOT).parts
    if not parts:
        return asset_path.name

    if parts[0] == "backgrounds" and len(parts) >= 3:
        return f"bg\n{parts[-1].removesuffix('.png')}"

    if parts[0] == "tiles" and len(parts) >= 3:
        return f"tile\n{parts[-1].removesuffix('.png')}"

    if parts[0] == "items" and len(parts) >= 4 and parts[1] == "fruits":
        return f"fruit\n{parts[2]}"

    if parts[0] == "characters" and len(parts) >= 4:
        if parts[1] == "pacman":
            frame = parts[-1].removesuffix(".png")
            return f"pacman\nbasic/{frame}"
        if parts[1] == "ghosts" and parts[2] == "shared":
            if parts[3] == "eyes":
                frame = parts[-1].removesuffix(".png").replace("frame_", "f")
                return f"eyes\n{frame}"
            if parts[3] == "frightened":
                variant = parts[4]
                frame = parts[-1].removesuffix(".png").replace("frame_", "f")
                return f"frightened\n{variant}/{frame}"
        if parts[1] == "ghosts" and len(parts) >= 6:
            ghost = parts[2]
            direction = parts[4]
            frame = parts[-1].removesuffix(".png").replace("frame_", "f")
            return f"{ghost}\n{direction}/{frame}"

    return "/".join(parts[-3:]).removesuffix(".png")


def build_atlas(
    asset_paths: list[Path],
    output_path: Path,
    columns: int = 8,
    cell_w: int = 160,
    cell_h: int = 180,
    inner_margin: int = 12,
) -> Path:
    if not asset_paths:
        raise RuntimeError(f"No PNG assets found under {ASSET_ROOT}")

    rows = math.ceil(len(asset_paths) / columns)
    atlas_w = columns * cell_w
    atlas_h = rows * cell_h

    atlas = Image.new("RGBA", (atlas_w, atlas_h), (18, 18, 24, 255))
    draw = ImageDraw.Draw(atlas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    label_h = 44
    image_box_h = cell_h - label_h - inner_margin * 2
    image_box_w = cell_w - inner_margin * 2

    for index, asset_path in enumerate(asset_paths):
        col = index % columns
        row = index // columns
        x0 = col * cell_w
        y0 = row * cell_h
        x1 = x0 + cell_w - 1
        y1 = y0 + cell_h - 1

        draw.rounded_rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), radius=10, outline=(60, 60, 80, 255), width=2)

        with Image.open(asset_path) as src:
            image = src.convert("RGBA")
        fitted = fit_into_box(image, image_box_w, image_box_h)

        image_x = x0 + (cell_w - fitted.width) // 2
        image_y = y0 + inner_margin
        atlas.alpha_composite(fitted, (image_x, image_y))

        bbox_color = (0, 229, 255, 255)
        draw.rectangle(
            (image_x, image_y, image_x + fitted.width - 1, image_y + fitted.height - 1),
            outline=bbox_color,
            width=2,
        )

        label = short_label(asset_path)
        bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=2, align="center")
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        label_x = x0 + (cell_w - text_w) // 2
        label_y = y0 + cell_h - label_h + max(0, (label_h - text_h) // 2) - 2
        draw.multiline_text((label_x, label_y), label, fill=(235, 235, 235, 255), font=font, spacing=2, align="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output_path)
    return output_path


def main() -> None:
    assets = collect_png_assets(ASSET_ROOT)
    output = build_atlas(assets, OUTPUT_PATH)
    print(f"Wrote atlas to {output}")
    print(f"Included {len(assets)} unique PNG assets")


if __name__ == "__main__":
    main()