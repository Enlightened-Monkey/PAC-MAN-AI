"""Arcade-style RGB renderer for Pac-Man GameState.

Primary path  : sprite-based renderer using ``PacmanMapDatasetGenerator``
                (real arcade sprites from the project sprite sheet).
Fallback path : simple disk/wedge renderer when the sprite sheet is not found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import cv2

from src.environment.game_logic import (
    GameState,
    ROWS,
    COLS,
    TILE_WALL,
    TILE_PELLET,
    TILE_POWER,
    TILE_DOOR,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
)

# ---------------------------------------------------------------------------
# Sprite-based renderer (primary)
# ---------------------------------------------------------------------------

_sprite_gen = None  # lazy singleton


def _get_sprite_gen():
    """Return a cached PacmanMapDatasetGenerator, or None if unavailable."""
    global _sprite_gen
    if _sprite_gen is not None:
        return _sprite_gen
    try:
        from src.dataset.pacman_map_dataset import PacmanMapDatasetGenerator, DEFAULT_GENERAL_SPRITE_SHEET
        if not Path(DEFAULT_GENERAL_SPRITE_SHEET).exists():
            return None
        _sprite_gen = PacmanMapDatasetGenerator()
        return _sprite_gen
    except Exception:
        return None


def render_state_rgb_sprites(
    state: GameState,
    scale: int = 2,
    skip_actors: bool = False,
) -> np.ndarray:
    """Render GameState using real arcade sprite assets.

    Returns an ``(H, W, 3)`` uint8 numpy array.  Pixels are scaled by
    ``scale`` (nearest-neighbour) so the window is large enough to read.
    Falls back to the disk renderer if the sprite sheet is unavailable.
    When ``skip_actors=True`` actors (Pac-Man, ghosts, fruit) are omitted so
    an animated overlay can be drawn on top without double-frame artefacts.
    """
    gen = _get_sprite_gen()
    if gen is None:
        return render_state_rgb(state, cell_size=8 * scale)

    from PIL import Image as _Image
    pil_img, _, _ = gen.render_state(state, skip_actors=skip_actors)
    if scale != 1:
        new_w = pil_img.width * scale
        new_h = pil_img.height * scale
        pil_img = pil_img.resize((new_w, new_h), _Image.Resampling.NEAREST)
    return np.array(pil_img, dtype=np.uint8)


def render_state_with_hud_sprites(
    state: GameState,
    info: dict | None = None,
    scale: int = 2,
    skip_actors: bool = False,
) -> np.ndarray:
    """Sprite-rendered game + arcade HUD (score, 1UP, HIGH SCORE, lives strip)."""
    info = info or {}
    board = render_state_rgb_sprites(state, scale=scale, skip_actors=skip_actors)

    top_h = 34
    bottom_h = 30
    full = np.zeros((board.shape[0] + top_h + bottom_h, board.shape[1], 3), dtype=np.uint8)
    full[top_h: top_h + board.shape[0]] = board

    W = full.shape[1]
    # --- top HUD ---
    cv2.putText(full, "1UP", (14, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(full, f"{state.score:,}", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    high = int(info.get("high_score", state.score))
    cv2.putText(full, "HIGH SCORE", (W//2 - 82, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(full, f"{high:,}", (W//2 - 35, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(full, f"LVL {state.level}", (W - 95, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    pl = float(info.get("pellet_levels", state.level - 1))
    cv2.putText(full, f"PEL {pl:.2f}", (W - 95, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,220,170), 1, cv2.LINE_AA)

    # --- bottom HUD: life icons ---
    icon_y = full.shape[0] - 16
    for idx in range(int(state.lives)):
        _draw_pacman(full, icon_y, 18 + idx * 20, 7, ACTION_RIGHT)

    # --- mode annotation ---
    mode = "FRIGHTENED" if state.frightened_timer > 0 else state.mode.upper()
    cv2.putText(full, mode, (W//2 - 55, full.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,220,255), 1, cv2.LINE_AA)
    return full

# Arcade-inspired palette (RGB)
_COLOR_BG = np.array([0, 0, 0], dtype=np.uint8)
_COLOR_WALL = np.array([33, 33, 222], dtype=np.uint8)
_COLOR_PELLET = np.array([255, 200, 170], dtype=np.uint8)
_COLOR_POWER = np.array([255, 180, 120], dtype=np.uint8)
_COLOR_PACMAN = np.array([255, 255, 0], dtype=np.uint8)
_COLOR_FRIGHTENED = np.array([33, 33, 255], dtype=np.uint8)
_COLOR_EYES = np.array([255, 255, 255], dtype=np.uint8)
_COLOR_DOOR = np.array([255, 180, 200], dtype=np.uint8)

_GHOST_RGB = {
    "Blinky": np.array([255, 0, 0], dtype=np.uint8),
    "Pinky": np.array([255, 182, 255], dtype=np.uint8),
    "Inky": np.array([0, 255, 255], dtype=np.uint8),
    "Clyde": np.array([255, 180, 80], dtype=np.uint8),
}

_GHOST_LABEL = {
    "Blinky": "B",
    "Pinky": "P",
    "Inky": "I",
    "Clyde": "C",
}


def _draw_disk(img: np.ndarray, cy: int, cx: int, radius: int, color: np.ndarray) -> None:
    h, w, _ = img.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    img[y0:y1, x0:x1][mask] = color


def _pacman_wedge_mask(radius: int, direction: int) -> np.ndarray:
    """Boolean mask for Pac-Man mouth opening toward movement direction."""
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    dist = x * x + y * y <= radius * radius
    if direction == ACTION_RIGHT:
        mouth = (x >= 0) & (np.abs(y) <= np.abs(x) * 0.55)
    elif direction == ACTION_LEFT:
        mouth = (x <= 0) & (np.abs(y) <= np.abs(x) * 0.55)
    elif direction == ACTION_UP:
        mouth = (y <= 0) & (np.abs(x) <= np.abs(y) * 0.55)
    else:  # DOWN
        mouth = (y >= 0) & (np.abs(x) <= np.abs(y) * 0.55)
    return dist & ~mouth


def _draw_pacman(img: np.ndarray, cy: int, cx: int, radius: int, direction: int) -> None:
    h, w, _ = img.shape
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    local_r = radius
    mask = _pacman_wedge_mask(local_r, direction)
    ly0 = y0 - (cy - local_r)
    lx0 = x0 - (cx - local_r)
    sub = mask[ly0 : ly0 + (y1 - y0), lx0 : lx0 + (x1 - x0)]
    img[y0:y1, x0:x1][sub] = _COLOR_PACMAN


def render_state_rgb(
    state: GameState,
    cell_size: int = 14,
    pad: int = 2,
) -> np.ndarray:
    """
    Render GameState to an RGB image (H, W, 3) uint8.

    Each maze cell is ``cell_size`` pixels; actors are drawn as coloured disks
    with arcade-accurate ghost colours and a directional Pac-Man wedge.
    """
    gh, gw = ROWS * cell_size + 2 * pad, COLS * cell_size + 2 * pad
    img = np.zeros((gh, gw, 3), dtype=np.uint8)

    maze = state.maze
    for r in range(ROWS):
        for c in range(COLS):
            y0 = pad + r * cell_size
            x0 = pad + c * cell_size
            tile = int(maze[r, c])
            if tile == TILE_WALL:
                img[y0 : y0 + cell_size, x0 : x0 + cell_size] = _COLOR_WALL
            elif tile == TILE_DOOR:
                mid = y0 + cell_size // 2
                img[mid - 1 : mid + 2, x0 : x0 + cell_size] = _COLOR_DOOR
            elif tile == TILE_PELLET:
                cy = y0 + cell_size // 2
                cx = x0 + cell_size // 2
                _draw_disk(img, cy, cx, max(1, cell_size // 8), _COLOR_PELLET)
            elif tile == TILE_POWER:
                cy = y0 + cell_size // 2
                cx = x0 + cell_size // 2
                _draw_disk(img, cy, cx, max(2, cell_size // 4), _COLOR_POWER)

    frightened = state.frightened_timer > 0
    for g in state.ghosts:
        if g.in_house and not g.eaten:
            continue
        gr, gc = g.pos
        cy = pad + gr * cell_size + cell_size // 2
        cx = pad + gc * cell_size + cell_size // 2
        if g.eaten:
            _draw_disk(img, cy, cx, cell_size // 3, _COLOR_EYES)
            _draw_disk(img, cy - 1, cx - 2, 2, np.array([0, 0, 255], dtype=np.uint8))
            _draw_disk(img, cy - 1, cx + 2, 2, np.array([0, 0, 255], dtype=np.uint8))
        elif frightened:
            _draw_disk(img, cy, cx, cell_size // 2 - 1, _COLOR_FRIGHTENED)
        else:
            color = _GHOST_RGB.get(g.name, np.array([200, 200, 200], dtype=np.uint8))
            _draw_disk(img, cy, cx, cell_size // 2 - 1, color)

    pr, pc = state.pacman_pos
    pcy = pad + pr * cell_size + cell_size // 2
    pcx = pad + pc * cell_size + cell_size // 2
    _draw_pacman(img, pcy, pcx, cell_size // 2 - 1, state.pacman_dir)

    if state.fruit_active:
        fr, fc = state.fruit_pos
        fcy = pad + fr * cell_size + cell_size // 2
        fcx = pad + fc * cell_size + cell_size // 2
        _draw_disk(img, fcy, fcx, cell_size // 3, np.array([255, 0, 0], dtype=np.uint8))

    return img


def render_hud_text(state: GameState, info: dict | None = None) -> str:
    """One-line HUD string for notebooks."""
    info = info or {}
    pellet_levels = info.get("pellet_levels", (state.level - 1) + info.get("pellet_completion", 0))
    return (
        f"Score: {state.score:,}  |  1UP  |  Level: {state.level}  |  Lives: {state.lives}  |  "
        f"Pellet levels: {float(pellet_levels):.2f}"
    )


def _draw_life_icon(img: np.ndarray, cy: int, cx: int, radius: int = 8) -> None:
    _draw_pacman(img, cy, cx, radius, ACTION_RIGHT)


def render_state_with_hud(
    state: GameState,
    info: dict | None = None,
    cell_size: int = 14,
    pad: int = 2,
) -> np.ndarray:
    """Disk-based board + HUD (used as fallback or in notebooks)."""
    info = info or {}
    board = render_state_rgb(state, cell_size=cell_size, pad=pad)
    top_h = 34
    bottom_h = 30
    out = np.zeros((board.shape[0] + top_h + bottom_h, board.shape[1], 3), dtype=np.uint8)
    out[:] = _COLOR_BG
    out[top_h : top_h + board.shape[0]] = board
    cv2.putText(out, "1UP", (14, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{state.score:,}", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2, cv2.LINE_AA)
    high_score = info.get("high_score", state.score)
    cv2.putText(out, "HIGH SCORE", (out.shape[1]//2 - 85, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{int(high_score):,}", (out.shape[1]//2 - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"LVL {state.level}", (out.shape[1] - 98, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(out, f"PEL {float(info.get('pellet_levels', state.level-1)):.2f}", (out.shape[1]-98, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,220,170), 1, cv2.LINE_AA)
    icon_y = out.shape[0] - 16
    for idx in range(int(state.lives)):
        _draw_life_icon(out, icon_y, 18 + idx * 20, radius=7)
    mode = "FRIGHTENED" if state.frightened_timer > 0 else state.mode.upper()
    cv2.putText(out, mode, (out.shape[1]//2 - 55, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,220,255), 1, cv2.LINE_AA)
    return out


def ghost_legend_lines() -> list[str]:
    return [
        "Blinky (red)  Pinky (pink)  Inky (cyan)  Clyde (orange)",
        "Blue = frightened   White dots = eaten ghost eyes",
    ]
