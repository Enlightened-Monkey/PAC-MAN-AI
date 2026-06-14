"""Arcade-style RGB renderer for Pac-Man GameState."""

from __future__ import annotations

import numpy as np

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
    return (
        f"Score: {state.score:,}  |  Level: {state.level}  |  Lives: {state.lives}  |  "
        f"Pellets: {info.get('pellet_completion', 0)*100:.0f}%"
    )


def ghost_legend_lines() -> list[str]:
    return [
        "Blinky (red)  Pinky (pink)  Inky (cyan)  Clyde (orange)",
        "Blue = frightened   White dots = eaten ghost eyes",
    ]
