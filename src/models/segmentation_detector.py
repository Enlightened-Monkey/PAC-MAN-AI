from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from src.dataset.pacman_map_dataset import CLASS_TO_ID
from src.environment.game_logic import DEFAULT_MAZE_ARR, TILE_PELLET, TILE_POWER


ID_TO_CLASS = {idx: name for name, idx in CLASS_TO_ID.items()}

# ---------------------------------------------------------------------------
# Stała maska slotów pellet / power_pellet wyprowadzona z arcade'owej mapy
# ---------------------------------------------------------------------------
# Wymiary renderowanego playfieldu (224×248 px, 28 kolumn × 31 wierszy)
_RENDER_W = 224
_RENDER_H = 248
_MAZE_ROWS = DEFAULT_MAZE_ARR.shape[0]   # 31
_MAZE_COLS = DEFAULT_MAZE_ARR.shape[1]   # 28
_TILE_W = _RENDER_W / _MAZE_COLS         # 8.0 px
_TILE_H = _RENDER_H / _MAZE_ROWS         # ~8.0 px


def _slot_center(row: int, col: int) -> tuple[int, int]:
    """Pikselowy środek kafelka (row, col) w 224×248 frame."""
    cx = int((col + 0.5) * _TILE_W)
    cy = int((row + 0.5) * _TILE_H)
    return cx, cy


# Precompute raz przy imporcie — lista (row, col, cx, cy) dla każdego slotu
_PELLET_SLOTS: list[tuple[int, int, int, int]] = []
_POWER_SLOTS: list[tuple[int, int, int, int]] = []
for _r in range(_MAZE_ROWS):
    for _c in range(_MAZE_COLS):
        _tile = int(DEFAULT_MAZE_ARR[_r, _c])
        if _tile == TILE_PELLET:
            _cx, _cy = _slot_center(_r, _c)
            _PELLET_SLOTS.append((_r, _c, _cx, _cy))
        elif _tile == TILE_POWER:
            _cx, _cy = _slot_center(_r, _c)
            _POWER_SLOTS.append((_r, _c, _cx, _cy))


def build_pellet_slot_mask(
    img_shape: tuple[int, int] = (_RENDER_H, _RENDER_W),
    include_power: bool = True,
) -> np.ndarray:
    """Zwraca uint8 maskę (H×W) z 255 w miejscach możliwych pelletów.

    Przydatna do nakładania na frame i do weryfikacji mapowania siatki.
    *img_shape* pozwala przeskalować maskę do innego rozmiaru niż 248×224.
    """
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    h, w = img_shape[:2]
    scale_x = w / _RENDER_W
    scale_y = h / _RENDER_H
    for _r, _c, cx, cy in _PELLET_SLOTS:
        px, py = int(cx * scale_x), int(cy * scale_y)
        if 0 <= px < w and 0 <= py < h:
            mask[py, px] = 255
    if include_power:
        for _r, _c, cx, cy in _POWER_SLOTS:
            px, py = int(cx * scale_x), int(cy * scale_y)
            if 0 <= px < w and 0 <= py < h:
                mask[py, px] = 255
    return mask


def detect_pellets_grid(
    image_rgb: np.ndarray,
    brightness_threshold: int = 160,
    sample_radius: int = 2,
) -> list[dict[str, Any]]:
    """Wykrywa pellety bez sieci neuronowej — przez próbkowanie jasności.

    Dla każdego stałego slotu z mapy arcade sprawdza, czy w okolicach środka
    kafelka w podanym *image_rgb* (224×248) jest wystarczająco jasny piksel
    (pellets i power_pellets to jasne punkty na ciemnym tle).

    Zwraca listę instances w tym samym formacie co :func:`extract_instances`.
    """
    h, w = image_rgb.shape[:2]
    scale_x = w / _RENDER_W
    scale_y = h / _RENDER_H

    # Grayscale do pomiaru jasności
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    instances: list[dict[str, Any]] = []

    for label, slots, tile_class_id in (
        ("pellet", _PELLET_SLOTS, CLASS_TO_ID["pellet"]),
        ("power_pellet", _POWER_SLOTS, CLASS_TO_ID["power_pellet"]),
    ):
        radius = sample_radius if label == "pellet" else sample_radius + 1
        for row, col, cx, cy in slots:
            px = int(cx * scale_x)
            py = int(cy * scale_y)
            x0 = max(0, px - radius)
            y0 = max(0, py - radius)
            x1 = min(w, px + radius + 1)
            y1 = min(h, py + radius + 1)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            if float(patch.max()) < brightness_threshold:
                continue
            bbox_w = int(_TILE_W * scale_x)
            bbox_h = int(_TILE_H * scale_y)
            bx = max(0, px - bbox_w // 2)
            by = max(0, py - bbox_h // 2)
            instances.append({
                "label": label,
                "class_id": tile_class_id,
                "bbox": [bx, by, bbox_w, bbox_h],
                "centroid": [float(px), float(py)],
                "area": (2 * radius + 1) ** 2,
                "grid": [row, col],
            })

    return instances

DEFAULT_GROUP_LAYERS: dict[str, tuple[str, ...]] = {
    "pacman": ("pacman",),
    "ghosts": ("blinky", "pinky", "inky", "clyde", "frightened_ghost", "ghost_eyes"),
    "fruit": ("fruit",),
    "collectibles": ("pellet", "power_pellet"),
    "walls_and_house": ("wall", "ghost_door", "ghost_house"),
}


class SegmentationDataset(Dataset):
    def __init__(self, dataset_dir: str | Path) -> None:
        root = Path(dataset_dir)
        self.images_dir = root / "images"
        self.masks_dir = root / "masks"
        if not self.images_dir.exists() or not self.masks_dir.exists():
            raise FileNotFoundError(
                f"Expected dataset structure with images/ and masks/ under: {root}"
            )

        self.image_paths = sorted(self.images_dir.glob("*.png"))
        if not self.image_paths:
            raise FileNotFoundError(f"No PNG files found in {self.images_dir}")

        self.mask_paths: list[Path] = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / img_path.name
            if not mask_path.exists():
                raise FileNotFoundError(f"Missing mask for image: {img_path.name}")
            self.mask_paths.append(mask_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.asarray(Image.open(self.image_paths[index]).convert("RGB"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(self.mask_paths[index]).convert("L"), dtype=np.int64)
        image_t = torch.from_numpy(image).permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)
        return image_t, mask_t


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 14) -> None:
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)


@dataclass
class TrainConfig:
    dataset_dir: Path
    output_path: Path
    epochs: int = 20
    batch_size: int = 16
    lr: float = 1e-3
    val_split: float = 0.1
    seed: int = 42
    device: str = "cpu"


class SegmentationDetector:
    def __init__(self, num_classes: int = len(CLASS_TO_ID), device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.num_classes = int(num_classes)
        self.model = TinyUNet(num_classes=self.num_classes).to(self.device)

    def train(self, config: TrainConfig) -> dict[str, float]:
        dataset = SegmentationDataset(config.dataset_dir)
        total_size = len(dataset)
        val_size = int(round(total_size * config.val_split))
        val_size = min(max(val_size, 1), total_size - 1) if total_size > 1 else 0

        if val_size > 0:
            train_size = total_size - val_size
            generator = torch.Generator().manual_seed(config.seed)
            train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
        else:
            train_ds = dataset
            val_ds = None

        train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False) if val_ds else None

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        best_val = float("inf")
        final_train_loss = 0.0
        final_val_loss = 0.0

        for _epoch in range(config.epochs):
            self.model.train()
            train_losses: list[float] = []
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)
                loss = criterion(logits, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            final_train_loss = float(np.mean(train_losses)) if train_losses else 0.0

            if val_loader is not None:
                self.model.eval()
                val_losses: list[float] = []
                with torch.no_grad():
                    for images, masks in val_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        logits = self.model(images)
                        val_loss = criterion(logits, masks)
                        val_losses.append(float(val_loss.item()))
                final_val_loss = float(np.mean(val_losses)) if val_losses else 0.0
                if final_val_loss < best_val:
                    best_val = final_val_loss
                    self.save(config.output_path)
            else:
                self.save(config.output_path)

        if val_loader is None:
            best_val = final_train_loss

        return {
            "train_loss": final_train_loss,
            "val_loss": final_val_loss if val_loader is not None else final_train_loss,
            "best_val_loss": best_val,
        }

    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        image = image_rgb.astype(np.float32) / 255.0
        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(x)
            mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        return mask

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "num_classes": self.num_classes,
                "class_to_id": CLASS_TO_ID,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SegmentationDetector":
        checkpoint = torch.load(path, map_location=device)
        detector = cls(num_classes=int(checkpoint["num_classes"]), device=device)
        detector.model.load_state_dict(checkpoint["state_dict"])
        detector.model.eval()
        return detector


def extract_instances(mask: np.ndarray, id_to_class: dict[int, str], min_area: int = 10) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    for class_id in sorted(id_to_class):
        if class_id == 0:
            continue
        class_name = id_to_class[class_id]
        binary = (mask == class_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for idx in range(1, num_labels):
            x, y, w, h, area = stats[idx]
            if int(area) < int(min_area):
                continue
            cx, cy = centroids[idx]
            instances.append(
                {
                    "label": class_name,
                    "class_id": int(class_id),
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "centroid": [float(cx), float(cy)],
                    "area": int(area),
                }
            )
    return instances


def build_class_layers(mask: np.ndarray, class_to_id: dict[str, int]) -> dict[str, np.ndarray]:
    layers: dict[str, np.ndarray] = {}
    for class_name, class_id in class_to_id.items():
        layer = np.zeros_like(mask, dtype=np.uint8)
        layer[mask == int(class_id)] = 255
        layers[class_name] = layer
    return layers


def build_group_layers(mask: np.ndarray, class_to_id: dict[str, int]) -> dict[str, np.ndarray]:
    group_layers: dict[str, np.ndarray] = {}
    for group_name, class_names in DEFAULT_GROUP_LAYERS.items():
        layer = np.zeros_like(mask, dtype=np.uint8)
        for class_name in class_names:
            class_id = class_to_id[class_name]
            layer[mask == class_id] = 255
        group_layers[group_name] = layer
    return group_layers


def detect_lives_icons(image_rgb: np.ndarray) -> list[dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    y0 = int(h * 0.80)
    x1 = int(w * 0.35)
    roi = image_rgb[y0:h, 0:x1]

    yellow = (
        (roi[:, :, 0] > 200)
        & (roi[:, :, 1] > 200)
        & (roi[:, :, 2] < 120)
    )
    binary = yellow.astype(np.uint8)

    num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    instances: list[dict[str, Any]] = []
    for idx in range(1, num_labels):
        x, y, bw, bh, area = stats[idx]
        if area < 30:
            continue
        cx, cy = centroids[idx]
        instances.append(
            {
                "label": "life_icon",
                "class_id": -1,
                "bbox": [int(x), int(y + y0), int(bw), int(bh)],
                "centroid": [float(cx), float(cy + y0)],
                "area": int(area),
            }
        )
    return instances


def detect_playfield_bbox(image_rgb: np.ndarray) -> tuple[int, int, int, int]:
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    blue_mask = cv2.inRange(hsv, (95, 70, 60), (140, 255, 255))
    blue_mask = cv2.morphologyEx(
        blue_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, w, h)

    best = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(best)
    if bw * bh < int(0.1 * w * h):
        return (0, 0, w, h)

    pad = int(round(0.02 * min(w, h)))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(w, x + bw + pad)
    y1 = min(h, y + bh + pad)
    return (x0, y0, x1 - x0, y1 - y0)


def _digit_templates() -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for d in range(10):
        canvas = np.zeros((28, 20), dtype=np.uint8)
        cv2.putText(
            canvas,
            str(d),
            (2, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            255,
            2,
            lineType=cv2.LINE_AA,
        )
        _, binary = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            templates[str(d)] = np.zeros((20, 12), dtype=np.uint8)
            continue
        crop = binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        templates[str(d)] = cv2.resize(crop, (12, 20), interpolation=cv2.INTER_NEAREST)
    return templates


def _recognize_digit(binary_digit: np.ndarray, templates: dict[str, np.ndarray]) -> str:
    resized = cv2.resize(binary_digit, (12, 20), interpolation=cv2.INTER_NEAREST)
    best_digit = ""
    best_score = -1.0
    for digit, tmpl in templates.items():
        overlap = np.logical_and(resized > 0, tmpl > 0).sum()
        union = np.logical_or(resized > 0, tmpl > 0).sum()
        score = float(overlap / union) if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_digit = digit
    return best_digit


def parse_hud_numbers(image_rgb: np.ndarray) -> dict[str, Any]:
    h, w = image_rgb.shape[:2]
    hud_h = int(round(0.25 * h))
    roi = image_rgb[:hud_h, :]

    white = (
        (roi[:, :, 0] > 180)
        & (roi[:, :, 1] > 180)
        & (roi[:, :, 2] > 180)
    ).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), dtype=np.uint8))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    digit_boxes: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 8:
            continue
        if bh < 6 or bw < 2:
            continue
        if y < int(0.3 * hud_h):
            continue
        digit_boxes.append((int(x), int(y), int(bw), int(bh)))

    templates = _digit_templates()
    left_digits: list[tuple[int, str]] = []
    right_digits: list[tuple[int, str]] = []
    split_x = w // 2

    for x, y, bw, bh in digit_boxes:
        digit_img = white[y:y + bh, x:x + bw]
        digit = _recognize_digit(digit_img, templates)
        if not digit:
            continue
        if x < split_x:
            left_digits.append((x, digit))
        else:
            right_digits.append((x, digit))

    left_digits.sort(key=lambda t: t[0])
    right_digits.sort(key=lambda t: t[0])
    score_text = "".join(d for _, d in left_digits) or None
    high_score_text = "".join(d for _, d in right_digits) or None

    return {
        "score_text": score_text,
        "high_score_text": high_score_text,
        "hud_roi": [0, 0, w, hud_h],
        "digit_count": len(digit_boxes),
    }


def remap_instances_to_image(
    instances: list[dict[str, Any]],
    crop_bbox: tuple[int, int, int, int],
    pred_shape: tuple[int, int],
) -> list[dict[str, Any]]:
    x0, y0, crop_w, crop_h = crop_bbox
    pred_h, pred_w = pred_shape
    sx = float(crop_w) / float(pred_w)
    sy = float(crop_h) / float(pred_h)

    remapped: list[dict[str, Any]] = []
    for obj in instances:
        x, y, w, h = obj["bbox"]
        cx, cy = obj["centroid"]
        remapped_obj = dict(obj)
        remapped_obj["bbox"] = [
            int(round(x0 + x * sx)),
            int(round(y0 + y * sy)),
            int(round(w * sx)),
            int(round(h * sy)),
        ]
        remapped_obj["centroid"] = [
            float(x0 + cx * sx),
            float(y0 + cy * sy),
        ]
        remapped.append(remapped_obj)
    return remapped


def train_cli(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        dataset_dir=Path(args.dataset_dir),
        output_path=Path(args.output),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        val_split=float(args.val_split),
        seed=int(args.seed),
        device=args.device,
    )
    detector = SegmentationDetector(device=cfg.device)
    metrics = detector.train(cfg)
    print(json.dumps(metrics, indent=2))
    print(f"Saved checkpoint: {cfg.output_path}")


def infer_cli(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = SegmentationDetector.load(args.model, device=args.device)

    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    pred_mask = detector.predict_mask(image)

    class_layers = build_class_layers(pred_mask, CLASS_TO_ID)
    group_layers = build_group_layers(pred_mask, CLASS_TO_ID)
    instances = extract_instances(pred_mask, ID_TO_CLASS, min_area=int(args.min_area))

    if args.detect_hud:
        instances.extend(detect_lives_icons(image))

    mask_path = output_dir / "pred_mask.png"
    Image.fromarray(pred_mask, mode="L").save(mask_path)

    class_dir = output_dir / "layers" / "classes"
    class_dir.mkdir(parents=True, exist_ok=True)
    for class_name, layer in class_layers.items():
        Image.fromarray(layer, mode="L").save(class_dir / f"{class_name}.png")

    group_dir = output_dir / "layers" / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    for group_name, layer in group_layers.items():
        Image.fromarray(layer, mode="L").save(group_dir / f"{group_name}.png")

    result = {
        "image": str(image_path),
        "mask": str(mask_path),
        "instances": instances,
        "layer_groups": {k: str(group_dir / f"{k}.png") for k in group_layers},
        "layer_classes": {k: str(class_dir / f"{k}.png") for k in class_layers},
    }
    result_path = output_dir / "prediction.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved prediction: {result_path}")


def infer_arcade_cli(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = SegmentationDetector.load(args.model, device=args.device)
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)

    playfield_bbox = detect_playfield_bbox(image)
    x0, y0, bw, bh = playfield_bbox
    crop = image[y0:y0 + bh, x0:x0 + bw]
    resized = cv2.resize(crop, (224, 248), interpolation=cv2.INTER_AREA)

    pred_mask = detector.predict_mask(resized)
    class_layers = build_class_layers(pred_mask, CLASS_TO_ID)
    group_layers = build_group_layers(pred_mask, CLASS_TO_ID)
    instances_local = extract_instances(pred_mask, ID_TO_CLASS, min_area=int(args.min_area))
    instances = remap_instances_to_image(instances_local, playfield_bbox, pred_mask.shape)

    hud = {
        "lives_icons": detect_lives_icons(image),
    }
    if args.parse_hud:
        hud.update(parse_hud_numbers(image))

    mask_path = output_dir / "pred_mask_playfield.png"
    Image.fromarray(pred_mask, mode="L").save(mask_path)

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    restored = cv2.resize(pred_mask, (bw, bh), interpolation=cv2.INTER_NEAREST)
    full_mask[y0:y0 + bh, x0:x0 + bw] = restored
    full_mask_path = output_dir / "pred_mask_full.png"
    Image.fromarray(full_mask, mode="L").save(full_mask_path)

    class_dir = output_dir / "layers" / "classes"
    class_dir.mkdir(parents=True, exist_ok=True)
    for class_name, layer in class_layers.items():
        Image.fromarray(layer, mode="L").save(class_dir / f"{class_name}.png")

    group_dir = output_dir / "layers" / "groups"
    group_dir.mkdir(parents=True, exist_ok=True)
    for group_name, layer in group_layers.items():
        Image.fromarray(layer, mode="L").save(group_dir / f"{group_name}.png")

    result = {
        "image": str(image_path),
        "playfield_bbox": [int(v) for v in playfield_bbox],
        "mask_playfield": str(mask_path),
        "mask_full": str(full_mask_path),
        "instances": instances,
        "hud": hud,
        "layer_groups": {k: str(group_dir / f"{k}.png") for k in group_layers},
        "layer_classes": {k: str(class_dir / f"{k}.png") for k in class_layers},
    }
    result_path = output_dir / "prediction_arcade.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved arcade prediction: {result_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run Pac-Man screenshot segmentation with layered outputs."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train a segmentation model on generated images/masks.")
    train.add_argument("--dataset-dir", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--val-split", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")

    infer = sub.add_parser("infer", help="Predict labels and positions from one screenshot.")
    infer.add_argument("--model", required=True)
    infer.add_argument("--image", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--min-area", type=int, default=10)
    infer.add_argument("--detect-hud", action="store_true")
    infer.add_argument("--device", default="cpu")

    infer_arcade = sub.add_parser(
        "infer-arcade",
        help="Predict from an arcade screenshot: auto-crop playfield + layered masks + HUD parse.",
    )
    infer_arcade.add_argument("--model", required=True)
    infer_arcade.add_argument("--image", required=True)
    infer_arcade.add_argument("--output-dir", required=True)
    infer_arcade.add_argument("--min-area", type=int, default=10)
    infer_arcade.add_argument("--parse-hud", action="store_true")
    infer_arcade.add_argument("--device", default="cpu")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        train_cli(args)
        return
    if args.command == "infer-arcade":
        infer_arcade_cli(args)
        return
    infer_cli(args)


if __name__ == "__main__":
    main()
