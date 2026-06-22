from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from src.environment.game_logic import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    COLS,
    DIRECTION_DELTAS,
    GameState,
    ROWS,
    TILE_PELLET,
    TILE_POWER,
    TILE_WALL,
)
from src.models.segmentation_detector import ID_TO_CLASS, SegmentationDetector, extract_instances
from src.utils.pacman_renderer import (
    render_state_rgb,  # kept for fallback / other callers
    render_state_rgb_sprites,
    render_state_with_hud_sprites,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ACTION_TO_NAME = {
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
}
ACTION_TO_DELTA = DIRECTION_DELTAS

FRIGHTENED_TICKS_BY_LEVEL: dict[int, int] = {
    1: 60,
    2: 40,
    3: 30,
    4: 20,
    5: 20,
    6: 50,
    7: 20,
    8: 20,
    9: 10,
    10: 50,
    11: 20,
    12: 10,
    13: 10,
    14: 10,
    15: 10,
    16: 10,
    17: 20,
    18: 10,
}


@dataclass
class AgentMemory:
    frame_idx: int = 0
    level_estimate: int = 1
    frightened_timer_est: int = 0
    steps_since_pellet_est: int = 0
    last_collectible_count: int = -1
    last_power_count: int = -1
    last_power_positions: list[tuple[int, int]] = field(default_factory=list)
    last_pacman_pos: tuple[int, int] | None = None
    last_action: int = ACTION_LEFT
    last_event: str = "init"
    initial_collectible_est: int = -1
    # Hidden persistent grids — walls never change; pellets/power only disappear
    wall_grid: np.ndarray | None = field(default=None, repr=False)
    pellet_grid: np.ndarray | None = field(default=None, repr=False)
    power_grid: np.ndarray | None = field(default=None, repr=False)


@dataclass
class ObservationBundle:
    pacman: tuple[int, int] | None
    walls: np.ndarray
    pellets: np.ndarray
    power: np.ndarray
    ghosts: np.ndarray
    frightened: np.ndarray
    fruit: np.ndarray
    level: int
    pellet_levels: float
    pellet_completion: float


class LiveDualAgentLab:
    def __init__(
        self,
        seed: int = 0,
        vision_model: str = "models/segmentation_unet_long.pt",
        fps: float = 12.0,
        device: str = "cpu",
        output_path: str | None = None,
        duration: float = 0.0,
        display: bool = True,
    ) -> None:
        self.seed = int(seed)
        self.fps = float(fps)
        self.frame_dt = 1.0 / max(self.fps, 0.1)
        self.vision_model = self._resolve_model_path(vision_model)
        self.output_path = output_path
        self.duration = float(duration)  # seconds; 0 = unlimited
        self.display = display
        self.detector = SegmentationDetector.load(
            self.vision_model,
            device=device,
        )
        self.direct_state = GameState(seed=self.seed)
        self.vision_state = GameState(seed=self.seed)
        self.direct_memory = AgentMemory()
        self.vision_memory = AgentMemory()
        self.best_direct_score = 0
        self.best_vision_score = 0
        self.direct_episode = 1
        self.vision_episode = 1
        
        # Animation system
        self.frame_counter = 0
        self.sprite_cache: dict[str, np.ndarray] = {}  # PIL->numpy cache for overlays
        self._load_sprite_cache()

    def _load_sprite_cache(self) -> None:
        """Load animated sprite frames from assets, scaled to render scale=2.

        Pac-Man animation cycle (4 frames):
          0 – open mouth  (directional frame_00)
          1 – half-open   (directional frame_01)
          2 – closed mouth / full circle  (shared frame_02, direction-independent)
          3 – half-open again (= frame_01, mouth opening back)
        """
        RENDER_SCALE = 2  # must match scale used in render_state_with_hud_sprites
        asset_root = PROJECT_ROOT / "data" / "labeled_maps" / "assets"
        if not asset_root.exists():
            return

        def _load_scale(path: Path) -> np.ndarray | None:
            if not path.exists():
                return None
            pil_img = PILImage.open(path).convert("RGBA")
            if RENDER_SCALE != 1:
                pil_img = pil_img.resize(
                    (pil_img.width * RENDER_SCALE, pil_img.height * RENDER_SCALE),
                    PILImage.Resampling.NEAREST,
                )
            return np.array(pil_img)

        # Shared closed-mouth sprite (frame_02) — used as frame index 2 for all directions.
        pacman_frame_02 = _load_scale(
            asset_root / "characters" / "pacman" / "normal" / "frame_02.png"
        )

        # Pac-Man: 4-frame animation per direction
        for direction in ("right", "left", "up", "down"):
            frame_00 = _load_scale(
                asset_root / "characters" / "pacman" / "normal" / direction / "frame_00.png"
            )
            frame_01 = _load_scale(
                asset_root / "characters" / "pacman" / "normal" / direction / "frame_01.png"
            )
            if frame_00 is not None:
                self.sprite_cache[f"pacman_{direction}_0"] = frame_00
            if frame_01 is not None:
                self.sprite_cache[f"pacman_{direction}_1"] = frame_01
            # frame 2 = shared closed-mouth (direction-independent circle)
            if pacman_frame_02 is not None:
                self.sprite_cache[f"pacman_{direction}_2"] = pacman_frame_02
            # frame 3 = half-open going back (reuse frame_01)
            if frame_01 is not None:
                self.sprite_cache[f"pacman_{direction}_3"] = frame_01

        # Ghost frames (2 per direction per ghost)
        for ghost_name in ("blinky", "pinky", "inky", "clyde"):
            for direction in ("right", "left", "up", "down"):
                for frame in (0, 1):
                    arr = _load_scale(
                        asset_root / "characters" / "ghosts" / ghost_name / "normal" / direction / f"frame_{frame:02d}.png"
                    )
                    if arr is not None:
                        self.sprite_cache[f"ghost_{ghost_name}_{direction}_{frame}"] = arr

        # Frightened ghost frames (blue)
        for frame in range(2):
            arr = _load_scale(
                asset_root / "characters" / "ghosts" / "shared" / "frightened" / "blue" / f"frame_{frame:02d}.png"
            )
            if arr is not None:
                self.sprite_cache[f"ghost_frightened_blue_{frame}"] = arr

        # Eyes frames (4 directions)
        for frame in range(4):
            arr = _load_scale(
                asset_root / "characters" / "ghosts" / "shared" / "eyes" / f"frame_{frame:02d}.png"
            )
            if arr is not None:
                self.sprite_cache[f"ghost_eyes_{frame}"] = arr

    @staticmethod
    def _blend_sprite_rgba(background: np.ndarray, sprite: np.ndarray, x: int, y: int) -> None:
        """Blend an RGBA sprite onto background (RGB) at given pixel coords. Modifies background in-place."""
        if sprite.shape[2] != 4:
            return
        
        h, w = sprite.shape[:2]
        x0, x1 = max(0, x), min(background.shape[1], x + w)
        y0, y1 = max(0, y), min(background.shape[0], y + h)
        
        if x0 >= x1 or y0 >= y1:
            return
        
        # Crop sprite to fit within bounds
        sx0 = max(0, -x)
        sy0 = max(0, -y)
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)
        
        sprite_crop = sprite[sy0:sy1, sx0:sx1]
        alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0
        sprite_rgb = sprite_crop[:, :, :3].astype(np.float32)
        bg_crop = background[y0:y1, x0:x1].astype(np.float32)
        
        # Alpha blend
        blended = (sprite_rgb * alpha + bg_crop * (1 - alpha)).astype(np.uint8)
        background[y0:y1, x0:x1] = blended

    def _overlay_animated_actors(self, frame_array: np.ndarray, state: GameState, x_offset: int = 0) -> None:
        """Overlay animated Pac-Man and ghosts on the frame. Modifies frame_array in-place.

        Centering: actors are centred on their tile.  The rendered board uses
        ``scale=2``, so each tile is ``8*2=16`` pixels.  The tile centre is at
        ``col*16 + 8``.  A sprite that is ``sw`` pixels wide should be placed
        at ``x = col*16 + 8 - sw//2`` so its centre aligns with the tile centre.
        ``x_offset`` shifts all actors horizontally (used for the right panel).
        """
        TILE_PX = 16   # 8 * render_scale=2
        HALF_TILE = 8  # TILE_PX // 2
        HUD_TOP = 34   # top HUD height in pixels

        # Pac-Man animation: 4-frame cycle, 1 tick = 1 frame
        pacman_frame = self.frame_counter % 4
        action_to_dir = {0: "up", 1: "down", 2: "left", 3: "right"}
        pac_dir = action_to_dir.get(state.pacman_dir, "right")
        pac_sprite_key = f"pacman_{pac_dir}_{pacman_frame}"

        if pac_sprite_key in self.sprite_cache and state.pacman_pos is not None:
            sprite = self.sprite_cache[pac_sprite_key]
            tile_r, tile_c = state.pacman_pos
            pixel_x = tile_c * TILE_PX + HALF_TILE - sprite.shape[1] // 2 + x_offset
            pixel_y = tile_r * TILE_PX + HALF_TILE - sprite.shape[0] // 2 + HUD_TOP
            self._blend_sprite_rgba(frame_array, sprite, pixel_x, pixel_y)

        # Ghost animation: 2-frame cycle, 1 tick = 1 frame
        ghost_frame = self.frame_counter % 2
        for ghost in state.ghosts:
            tile_r, tile_c = ghost.pos
            if ghost.eaten:
                eye_frame = self.frame_counter % 4
                key = f"ghost_eyes_{eye_frame}"
                if key in self.sprite_cache:
                    sprite = self.sprite_cache[key]
                    pixel_x = tile_c * TILE_PX + HALF_TILE - sprite.shape[1] // 2 + x_offset
                    pixel_y = tile_r * TILE_PX + HALF_TILE - sprite.shape[0] // 2 + HUD_TOP
                    self._blend_sprite_rgba(frame_array, sprite, pixel_x, pixel_y)
            elif state.frightened_timer > 0:
                fright_frame = self.frame_counter % 2
                key = f"ghost_frightened_blue_{fright_frame}"
                if key in self.sprite_cache:
                    sprite = self.sprite_cache[key]
                    pixel_x = tile_c * TILE_PX + HALF_TILE - sprite.shape[1] // 2 + x_offset
                    pixel_y = tile_r * TILE_PX + HALF_TILE - sprite.shape[0] // 2 + HUD_TOP
                    self._blend_sprite_rgba(frame_array, sprite, pixel_x, pixel_y)
            else:
                ghost_name = ghost.name.lower()
                ghost_dir = action_to_dir.get(ghost.direction, "right")
                key = f"ghost_{ghost_name}_{ghost_dir}_{ghost_frame}"
                if key in self.sprite_cache:
                    sprite = self.sprite_cache[key]
                    pixel_x = tile_c * TILE_PX + HALF_TILE - sprite.shape[1] // 2 + x_offset
                    pixel_y = tile_r * TILE_PX + HALF_TILE - sprite.shape[0] // 2 + HUD_TOP
                    self._blend_sprite_rgba(frame_array, sprite, pixel_x, pixel_y)

    @staticmethod
    def _resolve_model_path(model_arg: str) -> Path:
        requested = Path(model_arg).expanduser()
        candidates: list[Path] = []
        if requested.is_absolute():
            candidates.append(requested)
        else:
            candidates.append((Path.cwd() / requested).resolve())
            candidates.append((PROJECT_ROOT / requested).resolve())
        candidates.extend(
            [
                (PROJECT_ROOT / "models" / requested.name).resolve(),
                (PROJECT_ROOT / "notebooks" / "models" / requested.name).resolve(),
                (PROJECT_ROOT / "models" / "segmentation_unet.pt").resolve(),
                (PROJECT_ROOT / "models" / "segmentation_unet_long.pt").resolve(),
                (PROJECT_ROOT / "models" / "segmentation_unet_strong.pt").resolve(),
                (PROJECT_ROOT / "notebooks" / "models" / "segmentation_unet_combined.pt").resolve(),
            ]
        )
        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists() and candidate.is_file():
                return candidate
        raise FileNotFoundError(
            "Could not find a segmentation checkpoint. Checked:\n"
            + "\n".join(f" - {path}" for path in seen)
        )

    @staticmethod
    def _wrap_tunnel(pos: tuple[int, int]) -> tuple[int, int]:
        r, c = pos
        if r == 14 and c < 0:
            return r, COLS - 1
        if r == 14 and c >= COLS:
            return r, 0
        return pos

    @staticmethod
    def _nearest_distance(mask: np.ndarray, x: int, y: int) -> float:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return 9999.0
        dx = xs.astype(np.float32) - float(x)
        dy = ys.astype(np.float32) - float(y)
        return float(np.sqrt(dx * dx + dy * dy).min())

    @staticmethod
    def _tile_centroid(mask: np.ndarray) -> tuple[int, int] | None:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        return int(xs.mean()), int(ys.mean())

    @staticmethod
    def _tileify(binary: np.ndarray) -> np.ndarray:
        out = np.zeros((ROWS, COLS), dtype=bool)
        for r in range(ROWS):
            for c in range(COLS):
                y0 = r * 8
                x0 = c * 8
                y1 = min(binary.shape[0], y0 + 8)
                x1 = min(binary.shape[1], x0 + 8)
                if np.any(binary[y0:y1, x0:x1]):
                    out[r, c] = True
        return out

    def _snapshot_from_state(self, state: GameState) -> ObservationBundle:
        maze = state.maze
        walls = maze == TILE_WALL
        pellets = maze == TILE_PELLET
        power = maze == TILE_POWER
        ghosts = np.zeros((ROWS, COLS), dtype=bool)
        frightened = np.zeros((ROWS, COLS), dtype=bool)
        fruit = np.zeros((ROWS, COLS), dtype=bool)
        for ghost in state.ghosts:
            if ghost.eaten:
                continue
            gr, gc = ghost.pos
            if 0 <= gr < ROWS and 0 <= gc < COLS:
                if state.frightened_timer > 0:
                    frightened[gr, gc] = True
                else:
                    ghosts[gr, gc] = True
        if state.fruit_active:
            fr, fc = state.fruit_pos
            if 0 <= fr < ROWS and 0 <= fc < COLS:
                fruit[fr, fc] = True
        total = max(state.total_pellets, 1)
        completion = state.pellets_eaten / total
        pellet_levels = (state.level - 1) + completion
        return ObservationBundle(
            pacman=state.pacman_pos,
            walls=walls,
            pellets=pellets,
            power=power,
            ghosts=ghosts,
            frightened=frightened,
            fruit=fruit,
            level=state.level,
            pellet_levels=pellet_levels,
            pellet_completion=completion,
        )

    def _snapshot_from_mask(
        self,
        mask: np.ndarray,
        memory: AgentMemory,
        *,
        state: "GameState | None" = None,
    ) -> ObservationBundle:
        # --- Actors: detected from segmentation mask (dynamic, must be seen) ---
        pacman = self._tile_centroid(mask == self._class_id("pacman"))
        ghosts = np.zeros((ROWS, COLS), dtype=bool)
        frightened = np.zeros((ROWS, COLS), dtype=bool)
        fruit = self._tileify(mask == self._class_id("fruit"))
        for label in ("blinky", "pinky", "inky", "clyde"):
            cid = self._class_id(label)
            if cid >= 0:
                ghosts |= self._tileify(mask == cid)
        frightened_cid = self._class_id("frightened_ghost")
        if frightened_cid >= 0:
            frightened = self._tileify(mask == frightened_cid)

        # --- Static / quasi-static grids ------------------------------------ #
        # Walls and pellets are read directly from state.maze when available.  #
        # This is reliable and avoids segmentation noise for static layout.    #
        # Segmentation-based fallback is used only when state is not provided. #
        # -------------------------------------------------------------------- #
        if state is not None:
            memory.wall_grid = (state.maze == TILE_WALL)
            memory.pellet_grid = (state.maze == TILE_PELLET)
            memory.power_grid = (state.maze == TILE_POWER)
        else:
            # Fallback: infer from segmentation mask (less accurate)
            detected_walls = self._tileify(mask == self._class_id("wall"))
            detected_pellets = self._tileify(mask == self._class_id("pellet"))
            detected_power = self._tileify(mask == self._class_id("power_pellet"))
            if memory.wall_grid is None:
                memory.wall_grid = detected_walls.copy()
            # walls fixed once — never overwrite from noisy re-detection
            if memory.pellet_grid is None:
                memory.pellet_grid = detected_pellets.copy()
            else:
                memory.pellet_grid &= detected_pellets
            if memory.power_grid is None:
                memory.power_grid = detected_power.copy()
            else:
                memory.power_grid &= detected_power

        walls = memory.wall_grid
        pellets = memory.pellet_grid
        power = memory.power_grid

        collectible_count = int(pellets.sum() + power.sum())
        if memory.initial_collectible_est < 0:
            memory.initial_collectible_est = max(collectible_count, 1)
        if memory.last_collectible_count >= 0:
            if collectible_count < memory.last_collectible_count:
                memory.steps_since_pellet_est = 0
                memory.last_event = "pellet_eaten"
            else:
                memory.steps_since_pellet_est += 1
        if (
            memory.last_collectible_count >= 0
            and collectible_count - memory.last_collectible_count > 40
        ):
            # Level reset detected — grids already up to date (read from state.maze
            # every frame when state is provided; segmentation fallback re-detects below)
            memory.level_estimate = min(memory.level_estimate + 1, 21)  # guard dup increment
            memory.initial_collectible_est = max(collectible_count, 1)
            memory.steps_since_pellet_est = 0
            memory.last_event = "level_reset_detected"
        power_count = int(power.sum())
        if (
            memory.last_power_count >= 0
            and power_count < memory.last_power_count
            and pacman is not None
            and memory.last_power_positions
        ):
            px, py = pacman
            near = False
            for pos in memory.last_power_positions:
                dx = float(pos[1] - px)
                dy = float(pos[0] - py)
                if dx * dx + dy * dy <= 14.0 * 14.0:
                    near = True
                    break
            if near:
                memory.frightened_timer_est = FRIGHTENED_TICKS_BY_LEVEL.get(memory.level_estimate, 0)
                memory.last_event = "power_pellet_eaten"
        if frightened.any():
            memory.frightened_timer_est = max(memory.frightened_timer_est, 2)
        elif memory.frightened_timer_est > 0:
            memory.frightened_timer_est -= 1
        memory.last_collectible_count = collectible_count
        memory.last_power_count = power_count
        memory.last_power_positions = [(int(r), int(c)) for r, c in np.argwhere(power)]
        memory.last_pacman_pos = pacman
        completion = 1.0 - (collectible_count / max(memory.initial_collectible_est, 1))
        completion = float(max(0.0, min(1.0, completion)))
        pellet_levels = (memory.level_estimate - 1) + completion
        return ObservationBundle(
            pacman=(int(pacman[1] // 8), int(pacman[0] // 8)) if pacman is not None else None,
            walls=walls,
            pellets=pellets,
            power=power,
            ghosts=ghosts,
            frightened=frightened,
            fruit=fruit,
            level=memory.level_estimate,
            pellet_levels=pellet_levels,
            pellet_completion=completion,
        )

    @staticmethod
    def _class_id(label: str) -> int:
        from src.dataset.pacman_map_dataset import CLASS_TO_ID

        return int(CLASS_TO_ID.get(label, -1))

    def _update_memory_from_state(self, memory: AgentMemory, snapshot: ObservationBundle) -> None:
        collect_count = int(snapshot.pellets.sum() + snapshot.power.sum())
        if memory.initial_collectible_est < 0:
            memory.initial_collectible_est = max(collect_count, 1)
        if memory.last_collectible_count >= 0:
            if collect_count < memory.last_collectible_count:
                memory.steps_since_pellet_est = 0
                memory.last_event = "pellet_eaten"
            else:
                memory.steps_since_pellet_est += 1
        memory.last_collectible_count = collect_count
        power_count = int(snapshot.power.sum())
        memory.last_power_count = power_count
        memory.last_power_positions = [(int(r), int(c)) for r, c in np.argwhere(snapshot.power)]
        memory.level_estimate = snapshot.level
        memory.frightened_timer_est = max(int(memory.frightened_timer_est) - 1, 0)
        if snapshot.frightened.any():
            memory.frightened_timer_est = max(memory.frightened_timer_est, 2)
        memory.last_pacman_pos = snapshot.pacman

    def _choose_action(self, obs: ObservationBundle, memory: AgentMemory) -> int:
        if obs.pacman is None:
            return memory.last_action

        pr, pc = obs.pacman
        candidates: list[tuple[float, int]] = []
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nr, nc = pr + dr, pc + dc
            nr, nc = self._wrap_tunnel((nr, nc))
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if obs.walls[nr, nc]:
                continue
            pellet_window = obs.pellets[max(0, nr - 1) : min(ROWS, nr + 2), max(0, nc - 1) : min(COLS, nc + 2)]
            power_window = obs.power[max(0, nr - 1) : min(ROWS, nr + 2), max(0, nc - 1) : min(COLS, nc + 2)]
            pellet_score = float(pellet_window.sum() + 1.5 * power_window.sum())
            ghost_dist = self._nearest_distance(obs.ghosts, nc, nr)
            frightened_dist = self._nearest_distance(obs.frightened, nc, nr)
            ghost_penalty = 18.0 / (ghost_dist + 1.0)
            frightened_bonus = 0.0
            if memory.frightened_timer_est > 0 or obs.frightened.any():
                frightened_bonus = 12.0 / (frightened_dist + 1.0)
                ghost_penalty *= 0.35
            explore_bonus = min(memory.steps_since_pellet_est / 80.0, 1.0)
            forward_bonus = 0.8 if memory.last_action == action else 0.0
            score = pellet_score * 0.4 - ghost_penalty + frightened_bonus + explore_bonus + forward_bonus
            candidates.append((score, action))

        if not candidates:
            return memory.last_action
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    @staticmethod
    def _label_panel(frame: np.ndarray, title: str, subtitle: str, accent: tuple[int, int, int]) -> np.ndarray:
        out = frame.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 28), (8, 8, 8), -1)
        cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), accent, 2)
        cv2.putText(out, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, accent, 2, cv2.LINE_AA)
        cv2.putText(out, subtitle, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        return out

    def _render_vision_panel(self, state: GameState, memory: AgentMemory, action: int, score: int, mask: np.ndarray | None = None) -> np.ndarray:
        # Render with sprite assets — matches training data of the segmentation model
        if mask is None:
            frame8 = render_state_rgb_sprites(state, scale=1)
            mask = self.detector.predict_mask(frame8)
        instances = extract_instances(mask, ID_TO_CLASS, min_area=10)
        # Render WITHOUT actors — animated overlay will draw them to avoid double-frame artefacts
        vis = render_state_with_hud_sprites(
            state,
            info={"pellet_levels": (memory.level_estimate - 1) + memory.steps_since_pellet_est / 100.0, "high_score": self.best_vision_score},
            scale=2,
            skip_actors=True,
        )

        # ------------------------------------------------------------------ #
        # Semi-transparent grid overlays — agent's hidden knowledge layers     #
        # Walls:   static blue tint  (never re-detected, set once)             #
        # Pellets: yellow dots  (tracked — only disappear)                     #
        # Power:   magenta dots (same)                                         #
        # ------------------------------------------------------------------ #
        TILE_PX = 16   # 8 native × render_scale=2
        HUD_TOP = 34
        frame_h, frame_w = vis.shape[:2]

        def _tile_expand(grid: np.ndarray) -> np.ndarray:
            """Expand (ROWS, COLS) bool grid to pixel space (no HUD padding)."""
            return np.repeat(np.repeat(grid, TILE_PX, axis=0), TILE_PX, axis=1)

        vis_f = vis.astype(np.float32)

        # --- Wall layer (blue tint, alpha ≈ 0.50) ---
        if memory.wall_grid is not None:
            px = _tile_expand(memory.wall_grid)   # (ROWS*16, COLS*16)
            py0, py1 = HUD_TOP, HUD_TOP + px.shape[0]
            px0, px1 = 0, px.shape[1]
            region = vis_f[py0:py1, px0:px1]
            wall_color = np.array([30, 90, 220], dtype=np.float32)
            alpha_w = 0.50
            region[px] = region[px] * (1.0 - alpha_w) + wall_color * alpha_w
            vis_f[py0:py1, px0:px1] = region

        # --- Pellet layer (yellow dots inside tiles, alpha ≈ 0.55) ---
        if memory.pellet_grid is not None and memory.pellet_grid.any():
            # 4×4 px dot centred in each pellet tile
            DOT = 4
            for r, c in np.argwhere(memory.pellet_grid):
                r, c = int(r), int(c)
                cy = r * TILE_PX + HUD_TOP + TILE_PX // 2
                cx = c * TILE_PX + TILE_PX // 2
                y0, y1 = max(0, cy - DOT // 2), min(frame_h, cy + DOT // 2)
                x0, x1 = max(0, cx - DOT // 2), min(frame_w, cx + DOT // 2)
                vis_f[y0:y1, x0:x1] = vis_f[y0:y1, x0:x1] * 0.45 + np.array([255, 220, 0], np.float32) * 0.55

        # --- Power pellet layer (magenta, larger dot, alpha ≈ 0.60) ---
        if memory.power_grid is not None and memory.power_grid.any():
            DOT_P = 8
            for r, c in np.argwhere(memory.power_grid):
                r, c = int(r), int(c)
                cy = r * TILE_PX + HUD_TOP + TILE_PX // 2
                cx = c * TILE_PX + TILE_PX // 2
                y0, y1 = max(0, cy - DOT_P // 2), min(frame_h, cy + DOT_P // 2)
                x0, x1 = max(0, cx - DOT_P // 2), min(frame_w, cx + DOT_P // 2)
                vis_f[y0:y1, x0:x1] = vis_f[y0:y1, x0:x1] * 0.40 + np.array([200, 80, 255], np.float32) * 0.60

        vis = np.clip(vis_f, 0, 255).astype(np.uint8)

        # ------------------------------------------------------------------ #
        # Bounding boxes — skip static classes (wall, pellet, power_pellet)   #
        # Those are now shown via the grid overlay above.                      #
        # ------------------------------------------------------------------ #
        _STATIC_LABELS = {"wall", "pellet", "power_pellet"}
        for obj in instances:
            label = str(obj.get("label", "obj"))
            if label in _STATIC_LABELS:
                continue
            x, y, bw, bh = [int(v) * 2 for v in obj["bbox"]]
            color = (0, 255, 255) if (label == "pacman" or label.startswith("fruit")) else (255, 180, 80)
            cv2.rectangle(vis, (x, y + 34), (x + bw, y + bh + 34), color, 1)
            cv2.putText(vis, label, (x, max(y + 29, 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1, cv2.LINE_AA)

        subtitle = f"obs=grid+mask->agent | action={ACTION_TO_NAME[action]} | score={score}"
        return self._label_panel(vis, "VISION PIPELINE", subtitle, (255, 180, 80))

    def _render_direct_panel(self, state: GameState, action: int, score: int) -> np.ndarray:
        frame = render_state_with_hud_sprites(
            state,
            info={"pellet_levels": (state.level - 1) + state.pellets_eaten / max(state.total_pellets, 1), "high_score": self.best_direct_score},
            scale=2,
            skip_actors=True,
        )
        subtitle = f"obs=GameState | action={ACTION_TO_NAME[action]} | score={score}"
        return self._label_panel(frame, "DIRECT STATE", subtitle, (120, 255, 120))

    def _reset_state(self, which: str) -> None:
        if which == "direct":
            self.direct_episode += 1
            self.direct_state.reset(seed=self.seed + self.direct_episode)
            self.direct_memory = AgentMemory(level_estimate=self.direct_state.level)
        else:
            self.vision_episode += 1
            self.vision_state.reset(seed=self.seed + self.vision_episode)
            self.vision_memory = AgentMemory(level_estimate=self.vision_state.level)

    @staticmethod
    def _resolve_output_path(output_arg: str) -> Path:
        out = Path(output_arg).expanduser()
        if not out.is_absolute():
            out = (PROJECT_ROOT / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def run(self) -> None:
        print(f"[dual] Using vision model: {self.vision_model}")
        print("[dual] Left panel  = direct GameState policy (clean sprites)")
        print("[dual] Right panel = rendered frame -> segmentation -> bounding boxes")
        if self.output_path:
            print(f"[dual] Recording to: {self.output_path}")
            if self.duration > 0:
                print(f"[dual] Duration: {self.duration:.1f}s ({int(self.duration * self.fps)} frames)")
        if self.display:
            print("[dual] Press q to quit")

        writer: cv2.VideoWriter | None = None
        out_path: Path | None = None
        if self.output_path:
            out_path = self._resolve_output_path(self.output_path)

        if self.display:
            cv2.namedWindow("PAC-MAN Dual Agent Lab", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("PAC-MAN Dual Agent Lab", 1340, 760)

        recording_start = time.time()
        frame_count = 0

        try:
            while True:
                loop_start = time.time()

                # --- Direct-state agent step ---
                direct_snapshot = self._snapshot_from_state(self.direct_state)
                direct_action = self._choose_action(direct_snapshot, self.direct_memory)
                direct_reward, direct_done = self.direct_state.step(direct_action)
                self._update_memory_from_state(self.direct_memory, self._snapshot_from_state(self.direct_state))
                self.direct_memory.last_action = direct_action
                self.best_direct_score = max(self.best_direct_score, self.direct_state.score)

                # --- Vision-pipeline agent step ---
                vision_frame = render_state_rgb_sprites(self.vision_state, scale=1)
                vision_mask = self.detector.predict_mask(vision_frame)
                vision_snapshot = self._snapshot_from_mask(vision_mask, self.vision_memory, state=self.vision_state)
                vision_action = self._choose_action(vision_snapshot, self.vision_memory)
                vision_reward, vision_done = self.vision_state.step(vision_action)
                self.vision_memory.last_action = vision_action
                self.best_vision_score = max(self.best_vision_score, self.vision_state.score)

                # --- Build combined frame ---
                left_panel = self._render_direct_panel(self.direct_state, direct_action, self.direct_state.score)
                right_panel = self._render_vision_panel(self.vision_state, self.vision_memory, vision_action, self.vision_state.score, vision_mask)

                gap = np.zeros((left_panel.shape[0], 20, 3), dtype=np.uint8)
                gap[:] = (18, 18, 18)
                combined = np.hstack([left_panel, gap, right_panel])

                cv2.putText(
                    combined,
                    "ARCHITECTURE: internal simulation | direct-state agent | vision-from-render agent",
                    (16, 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    combined,
                    f"DIRECT reward={direct_reward:.1f}  VISION reward={vision_reward:.1f}",
                    (16, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

                # --- Apply animation overlays: left panel (direct) and right panel (vision) ---
                self._overlay_animated_actors(combined, self.direct_state, x_offset=0)
                right_x_offset = left_panel.shape[1] + gap.shape[1]
                self._overlay_animated_actors(combined, self.vision_state, x_offset=right_x_offset)
                
                # --- Initialise VideoWriter on first frame (size now known) ---
                if out_path is not None and writer is None:
                    h, w = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
                    if not writer.isOpened():
                        print(f"[dual] WARNING: could not open VideoWriter for {out_path}")
                        writer = None
                    else:
                        print(f"[dual] VideoWriter opened: {w}x{h} @ {self.fps:.1f} fps")

                # --- Write frame to MP4 (convert RGB to BGR) ---
                if writer is not None:
                    writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

                # --- Live display (convert RGB to BGR for cv2) ---
                if self.display:
                    cv2.imshow("PAC-MAN Dual Agent Lab", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                frame_count += 1
                self.frame_counter += 1

                # --- Duration / episode resets ---
                if direct_done:
                    self._reset_state("direct")
                if vision_done:
                    self._reset_state("vision")

                if self.duration > 0 and (time.time() - recording_start) >= self.duration:
                    print(f"[dual] Duration reached ({self.duration:.1f}s, {frame_count} frames).")
                    break

                elapsed = time.time() - loop_start
                if elapsed < self.frame_dt:
                    time.sleep(self.frame_dt - elapsed)

        finally:
            if writer is not None:
                writer.release()
                print(f"[dual] Saved: {out_path}  ({frame_count} frames)")
            if self.display:
                cv2.destroyAllWindows()
            self.frame_counter = 0  # Reset for next run


def main() -> None:
    import datetime

    parser = argparse.ArgumentParser(description="Pac-Man internal dual-agent lab (direct state vs vision pipeline).")
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed for both agents")
    parser.add_argument(
        "--vision-model",
        type=str,
        default="models/segmentation_unet_long.pt",
        help="Segmentation checkpoint used by the vision-side agent",
    )
    parser.add_argument("--fps", type=float, default=12.0, help="Visualization loop FPS")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for the vision segmentation model (default: cpu to avoid competing with RL training on GPU)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path for the output MP4 file. "
            "Defaults to records/dual_agent_<timestamp>.mp4 when --duration is set. "
            "Pass an explicit path to always record."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Recording duration in seconds (0 = unlimited / run until q).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the live cv2 window (useful for headless recording).",
    )
    args = parser.parse_args()

    # Auto-generate output path when duration is given but --output was omitted
    output_path = args.output
    if output_path is None and args.duration > 0:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"records/dual_agent_{ts}.mp4"

    lab = LiveDualAgentLab(
        seed=args.seed,
        vision_model=args.vision_model,
        fps=args.fps,
        device=args.device,
        output_path=output_path,
        duration=args.duration,
        display=not args.no_display,
    )
    lab.run()


if __name__ == "__main__":
    main()
