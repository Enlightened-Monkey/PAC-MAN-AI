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
)

# Observation vector length = 6 (pacman + mode) + 8 (ghost positions)
#                             + 4 (frightened flags) + 2 (lives, pellets)
#                             + ROWS*COLS (maze)
_OBS_SIZE = 6 + 8 + 4 + 2 + ROWS * COLS


class PacmanEnv(gym.Env):
    """
    Custom Pac-Man environment for reinforcement-learning experiments.

    Observation space
    -----------------
    A 1-D float32 vector of length ``_OBS_SIZE`` (≈ 460 values) containing:
      - Pac-Man normalised row/col position
      - Ghost mode flag (scatter vs chase)
      - Normalised ghost row/col positions (4 ghosts)
      - Ghost frightened flags
      - Remaining lives and pellet ratio
      - Flattened normalised maze layout

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
        self, render_mode: str | None = None, seed: int | None = None
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self._state = GameState(seed=seed)

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
        obs = self._state.to_observation()
        info = self._get_info()
        # terminated = natural end (all pellets eaten or no lives left)
        # truncated  = time limit reached (not implemented here)
        return obs, float(reward), done, False, info

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
        }

    def _render_ansi(self) -> str:
        """Return a simple text representation of the current board."""
        CELL_CHARS = {0: " ", 1: "#", 2: ".", 3: "o"}
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
