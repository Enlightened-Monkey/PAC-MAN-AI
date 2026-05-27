"""
pacman_env.py – Gymnasium-compatible Pac-Man environment.

PacmanEnv wraps the internal GameState (game_logic.py) and exposes the
standard gymnasium.Env interface so that any off-the-shelf RL algorithm
(DQN, PPO, etc.) can interact with the game without knowing its internals.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.environment.game_logic import (
    GameState,
    ROWS,
    COLS,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    DIRECTION_DELTAS,
)

# Observation vector length = 4 (pacman + frightened/flashing flags) + 8 (ghost positions)
#                             + 4 (frightened flags) + 2 (lives, pellets)
#                             + ROWS*COLS (maze) + 3 (fruit: active, row, col)
_OBS_SIZE = 4 + 8 + 4 + 2 + ROWS * COLS + 3


class PacmanEnv(gym.Env):
    """
    Custom Pac-Man environment for reinforcement-learning experiments.

    Observation space
    -----------------
    A 1-D float32 vector of length ``_OBS_SIZE`` containing:
      - Pac-Man normalised row/col position
      - Frightened and flashing mode flags (2 values)
      - Normalised ghost row/col positions (4 ghosts)
      - Ghost frightened flags
      - Remaining lives and pellet ratio
      - Flattened normalised maze layout
      - Active fruit flag + normalised position (3 values)

    Action space
    ------------
    Discrete(4): 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT

    Reward
    ------
    +10 per pellet, +50 per power pellet, +200 per eaten ghost,
    -500 on losing a life.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        step_penalty: float = 0.0,
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self._max_steps = max_steps
        self._step_penalty = float(step_penalty)
        self._reward_scale = float(reward_scale)
        self._state = GameState(seed=seed)

        if self._max_steps is not None and self._max_steps <= 0:
            raise ValueError("max_steps must be > 0 or None")
        if self._reward_scale <= 0:
            raise ValueError("reward_scale must be > 0")

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(_OBS_SIZE,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # gymnasium.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self._state.reset(seed=effective_seed)
        obs = self._state.to_observation()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        reward, done = self._state.step(int(action))
        scaled_reward = (float(reward) * self._reward_scale) + self._step_penalty
        obs = self._state.to_observation()

        truncated = False
        if self._max_steps is not None and self._state.step_count >= self._max_steps:
            truncated = not done

        info = self._get_info()
        # terminated = natural end (all pellets eaten or no lives left)
        # truncated  = time limit reached (not implemented here)
        return obs, scaled_reward, done, truncated, info

    def render(self) -> str | None:
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_info(self) -> dict[str, Any]:
        return {
            "score": self._state.score,
            "lives": self._state.lives,
            "step": self._state.step_count,
            "max_steps": self._max_steps,
        }

    def _render_ansi(self) -> str:
        """Return a simple text representation of the current board."""
        CELL_CHARS = {0: " ", 1: "#", 2: ".", 3: "o", 4: "-", 5: " "}
        rows = []
        maze = self._state.maze
        ghost_positions = {g.pos: g.name[0] for g in self._state.ghosts}
        for r in range(ROWS):
            row_str = ""
            for c in range(COLS):
                pos = (r, c)
                if pos == self._state.pacman_pos:
                    row_str += "C"
                elif pos in ghost_positions:
                    row_str += ghost_positions[pos]
                else:
                    row_str += CELL_CHARS.get(int(maze[r, c]), "?")
            rows.append(row_str)
        return "\n".join(rows)


class PacmanPrototypeEnv(PacmanEnv):
    """
    Lightweight training-focused Pac-Man variant.

    Defaults favour faster RL iterations:
      - shorter episodes via ``max_steps``
      - small per-step penalty to encourage efficient trajectories
      - scaled rewards to reduce variance in early training
    """

    def __init__(
        self,
        render_mode: str | None = None,
        seed: int | None = None,
        max_steps: int = 300,
        step_penalty: float = -0.01,
        reward_scale: float = 0.1,
    ) -> None:
        super().__init__(
            render_mode=render_mode,
            seed=seed,
            max_steps=max_steps,
            step_penalty=step_penalty,
            reward_scale=reward_scale,
        )


# ---------------------------------------------------------------------------
# PacmanGridEnv — 2D channel observation + action masking + reward shaping
# ---------------------------------------------------------------------------

from collections import deque
from src.environment.game_logic import (
    TILE_WALL, TILE_PELLET, TILE_POWER, TILE_DOOR, TILE_HOUSE,
)

# Channel layout (C, H=ROWS, W=COLS)
_GRID_CHANNELS = 6  # walls, pellets, power, pacman, ghosts_normal, ghosts_frightened


class PacmanGridEnv(gym.Env):
    """
    Pac-Man environment optimised for CNN policies + MaskablePPO.

    Observation: float32 tensor (6, ROWS, COLS) in [0, 1]
        ch0 walls, ch1 pellets, ch2 power, ch3 pacman,
        ch4 ghosts (not frightened, not eaten), ch5 ghosts frightened
    Action space: Discrete(4)  (UP / DOWN / LEFT / RIGHT)
    Action mask: legal moves from the current Pac-Man tile (no walls).

    Reward shaping
    --------------
    raw reward (pellet=10, power=50, ghost=200, death=-500) is divided by
    `reward_scale_div` (default 100) and a small `step_penalty` is added
    every tick. Optional potential-based shaping (PBRS) gives a dense
    signal proportional to how much closer Pac-Man got to the nearest
    remaining pellet (BFS distance on the static maze graph).
    """

    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int = 2000,
        step_penalty: float = -0.01,
        reward_scale_div: float = 100.0,
        pbrs_coef: float = 0.05,
        pbrs_gamma: float = 0.99,
    ) -> None:
        super().__init__()
        self._seed = seed
        self._max_steps = int(max_steps)
        self._step_penalty = float(step_penalty)
        self._reward_div = float(reward_scale_div)
        self._pbrs_coef = float(pbrs_coef)
        self._pbrs_gamma = float(pbrs_gamma)

        self._state = GameState(seed=seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(_GRID_CHANNELS, ROWS, COLS),
            dtype=np.float32,
        )

        # Pre-compute walkability map (Pac-Man passable tiles).
        self._walk_mask = np.isin(
            self._state.maze, [0, TILE_PELLET, TILE_POWER]  # 0 = TILE_EMPTY
        )
        self._prev_potential: float = 0.0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self._seed
        self._state.reset(seed=effective_seed)
        self._prev_potential = self._potential()
        return self._obs(), self._info()

    def step(self, action: int):
        raw_reward, done = self._state.step(int(action))

        # Scale + per-step penalty
        reward = raw_reward / self._reward_div + self._step_penalty

        # Potential-based shaping (preserves optimal policy)
        if self._pbrs_coef > 0.0 and not done:
            phi = self._potential()
            reward += self._pbrs_coef * (self._pbrs_gamma * phi - self._prev_potential)
            self._prev_potential = phi

        truncated = (not done) and self._state.step_count >= self._max_steps
        return self._obs(), float(reward), bool(done), bool(truncated), self._info()

    # ------------------------------------------------------------------
    # MaskablePPO hook
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return bool mask of legal moves from Pac-Man's current tile."""
        r, c = self._state.pacman_pos
        mask = np.zeros(4, dtype=bool)
        for action, (dr, dc) in DIRECTION_DELTAS.items():
            nr, nc = r + dr, c + dc
            # Tunnel wrap on tunnel row
            if nr == 14 and (nc < 0 or nc >= COLS):
                mask[action] = True
                continue
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            t = self._state.maze[nr, nc]
            if t in (TILE_WALL, TILE_DOOR, TILE_HOUSE):
                continue
            mask[action] = True
        # Safety: if no move is legal (shouldn't happen) allow all so the
        # policy sampler never crashes.
        if not mask.any():
            mask[:] = True
        return mask

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs(self) -> np.ndarray:
        st = self._state
        m = st.maze
        obs = np.zeros((_GRID_CHANNELS, ROWS, COLS), dtype=np.float32)
        obs[0] = (m == TILE_WALL).astype(np.float32)
        obs[1] = (m == TILE_PELLET).astype(np.float32)
        obs[2] = (m == TILE_POWER).astype(np.float32)
        pr, pc = st.pacman_pos
        if 0 <= pr < ROWS and 0 <= pc < COLS:
            obs[3, pr, pc] = 1.0
        frightened = st.frightened_timer > 0
        for g in st.ghosts:
            if g.eaten:
                continue
            gr, gc = g.pos
            if 0 <= gr < ROWS and 0 <= gc < COLS:
                if frightened:
                    obs[5, gr, gc] = 1.0
                else:
                    obs[4, gr, gc] = 1.0
        return obs

    def _info(self) -> dict:
        return {
            "score": self._state.score,
            "lives": self._state.lives,
            "step": self._state.step_count,
        }

    def _potential(self) -> float:
        """Negative BFS distance to the nearest pellet, normalised to [-1, 0]."""
        start = self._state.pacman_pos
        maze = self._state.maze
        # If no pellets left, potential = 0.
        if not np.any(np.isin(maze, [TILE_PELLET, TILE_POWER])):
            return 0.0
        visited = np.zeros((ROWS, COLS), dtype=bool)
        q = deque()
        q.append((start, 0))
        visited[start] = True
        max_d = ROWS + COLS
        while q:
            (r, c), d = q.popleft()
            tile = maze[r, c]
            if tile in (TILE_PELLET, TILE_POWER):
                return -d / max_d
            for dr, dc in DIRECTION_DELTAS.values():
                nr, nc = r + dr, c + dc
                if nr == 14 and nc < 0:
                    nc = COLS - 1
                elif nr == 14 and nc >= COLS:
                    nc = 0
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    continue
                if visited[nr, nc]:
                    continue
                if not self._walk_mask[nr, nc]:
                    continue
                visited[nr, nc] = True
                q.append(((nr, nc), d + 1))
        return -1.0  # unreachable (shouldn't happen)

    def render(self):
        return None

    def close(self):
        pass
