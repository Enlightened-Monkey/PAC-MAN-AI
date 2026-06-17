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
        
        if self._state._all_pellets_eaten() and self._state.lives > 0:
            reward += 1000.0
            self._state._next_level()
            
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
            "level": self._state.level,
        }

    def _render_ansi(self) -> str:
        """Return a beautifully clean, ANSI-free Unicode text representation of the current board."""
        rows = []
        state = self._state
        maze = state.maze
        
        ghosts_at_pos = {}
        for g in state.ghosts:
            ghosts_at_pos[g.pos] = g

        for r in range(ROWS):
            row_str = ""
            for c in range(COLS):
                pos = (r, c)
                if pos == state.pacman_pos:
                    direction = state.pacman_dir
                    pac_char = "◀"
                    if direction == ACTION_UP:
                        pac_char = "▲"
                    elif direction == ACTION_DOWN:
                        pac_char = "▼"
                    elif direction == ACTION_LEFT:
                        pac_char = "◀"
                    elif direction == ACTION_RIGHT:
                        pac_char = "▶"
                    row_str += pac_char
                elif pos in ghosts_at_pos:
                    g = ghosts_at_pos[pos]
                    if g.eaten:
                        row_str += "E"  # White eaten ghost eyes
                    elif state.frightened_timer > 0:
                        row_str += "S"  # Frightened/scared ghost
                    else:
                        row_str += g.name[0]  # B, P, I, C
                else:
                    tile_val = int(maze[r, c])
                    if tile_val == 1:    # Wall
                        row_str += "█"
                    elif tile_val == 2:  # Pellet
                        row_str += "·"
                    elif tile_val == 3:  # Power Pellet
                        row_str += "●"
                    elif tile_val == 4:  # Door
                        row_str += "═"
                    else:
                        row_str += " "
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
_GRID_BASE_CHANNELS = 9  # walls, pellets, power, pacman, ghosts×2, fruit, lives_hud, level_plane
# Optional ch9: global pellet_completion plane (enabled via include_completion_plane=True)
_GRID_CHANNELS = _GRID_BASE_CHANNELS  # default export for tests without completion plane

# Milestones only in deep endgame — avoids "farm 90% and die" local optimum.
_DEFAULT_MILESTONE_THRESHOLDS: tuple[float, ...] = (0.92, 0.96, 0.99)
_DEFAULT_MILESTONE_BONUSES: tuple[float, ...] = (150.0, 300.0, 600.0)
_GHOST_NEAR_FOR_POWER = 5  # Manhattan tiles

# Fixed HUD positions for remaining lives (bottom-left, like the arcade cabinet)
_LIVES_HUD_POSITIONS: tuple[tuple[int, int], ...] = ((30, 0), (30, 1), (30, 2))


class PacmanGridEnv(gym.Env):
    """
    Pac-Man environment optimised for CNN policies (PPO / MaskablePPO).

    Observation: float32 tensor (8, ROWS, COLS) in [0, 1]
        ch0 walls, ch1 pellets, ch2 power, ch3 pacman,
        ch4 ghosts (not frightened, not eaten), ch5 ghosts frightened,
        ch6 fruit (when active), ch7 lives HUD markers
    Action space: Discrete(4)  (UP / DOWN / LEFT / RIGHT)
    Action mask: optional helper for debugging; fair training should not use masks.

    Reward shaping (score-first)
    ----------------------------
    reward = score_delta / reward_scale_div + step_penalty (+ death_penalty on death)

    The reward tracks the arcade score directly (pellet=10, power=50,
    ghost=200..1600, fruit) so maximising return == maximising score.
    Death costs a flat `death_penalty` (already scaled, default -3.0) instead of
    the raw -500 -> -10 that previously dominated every pellet signal 50:1.
    A Level Completion Bonus (`level_bonus` raw, default 2500 -> +50 scaled)
    is awarded when all pellets are cleared.
    One-time milestone bonuses at 85/90/95/98% pellet completion encourage endgame.
    Endgame death surcharge (-2 scaled) applies when dying at >=85% completion.
    Optional PBRS is disabled by default (`pbrs_coef=0.0`) for human-fair training.

    Episode randomness: the constructor seed is used only for the FIRST episode;
    subsequent resets draw fresh RNG so the agent cannot memorise a fixed
    ghost-RNG replay (frightened moves and fruit timers are stochastic).
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        seed: int | None = None,
        max_steps: int = 5000,
        step_penalty: float = -0.0005,
        reward_scale_div: float = 50.0,
        death_penalty: float = -3.0,
        level_bonus: float = 5000.0,
        endgame_death_surcharge: float = -3.0,
        endgame_death_threshold: float = 0.85,
        idle_penalty: float = -0.02,
        wasted_power_penalty: float = -1.5,
        near_miss_penalty: float = -5.0,
        near_miss_threshold: float = 0.90,
        milestone_thresholds: tuple[float, ...] | None = None,
        milestone_bonuses: tuple[float, ...] | None = None,
        enable_milestones: bool = True,
        pbrs_coef: float = 0.0,
        pbrs_gamma: float = 0.99,
        human_fair: bool = True,
        include_completion_plane: bool = False,
        include_frightened_plane: bool = False,
        easy_endgame: bool = False,
        elroy_pellets_threshold: int | None = None,
        render_mode: str | None = "rgb_array",
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        self._max_steps = int(max_steps)
        self._step_penalty = float(step_penalty)
        self._reward_div = float(reward_scale_div)
        self._death_penalty = float(death_penalty)
        self._level_bonus = float(level_bonus)
        self._endgame_death_surcharge = float(endgame_death_surcharge)
        self._endgame_death_threshold = float(endgame_death_threshold)
        self._idle_penalty = float(idle_penalty)
        self._wasted_power_penalty = float(wasted_power_penalty)
        self._near_miss_penalty = float(near_miss_penalty)
        self._near_miss_threshold = float(near_miss_threshold)
        if human_fair and enable_milestones:
            self._milestone_thresholds = tuple(
                milestone_thresholds or _DEFAULT_MILESTONE_THRESHOLDS
            )
            self._milestone_bonuses = tuple(milestone_bonuses or _DEFAULT_MILESTONE_BONUSES)
        else:
            self._milestone_thresholds = ()
            self._milestone_bonuses = ()
        if len(self._milestone_thresholds) != len(self._milestone_bonuses):
            raise ValueError("milestone_thresholds and milestone_bonuses must have equal length")
        if human_fair:
            pbrs_coef = 0.0
        self._pbrs_coef = float(pbrs_coef)
        self._pbrs_gamma = float(pbrs_gamma)
        self._human_fair = bool(human_fair)
        self._include_completion_plane = bool(include_completion_plane)
        self._include_frightened_plane = bool(include_frightened_plane)
        extra = int(self._include_completion_plane) + int(self._include_frightened_plane)
        self._n_channels = _GRID_BASE_CHANNELS + extra

        elroy = elroy_pellets_threshold
        if easy_endgame and elroy is None:
            elroy = 5

        self._state = GameState(seed=seed, elroy_pellets_threshold=elroy)
        self._episode_start_lives = 3
        self._episode_deaths = 0
        self._max_level_reached = 1
        self._level_clears = 0
        self._milestones_hit: set[float] = set()
        self._episode_milestone_reward = 0.0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self._n_channels, ROWS, COLS),
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
        if seed is not None:
            self._state.reset(seed=seed)
        elif self._seed is not None:
            # Constructor seed applies to the first episode only; afterwards
            # each episode gets fresh randomness (no fixed-replay memorisation).
            self._state.reset(seed=self._seed)
            self._seed = None
        else:
            self._state.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))
        self._prev_potential = self._potential()
        self._episode_start_lives = self._state.lives
        self._episode_deaths = 0
        self._max_level_reached = self._state.level
        self._level_clears = 0
        self._milestones_hit = set()
        self._episode_milestone_reward = 0.0
        return self._obs(), self._info()

    def _pellet_completion(self) -> float:
        total = max(self._state.total_pellets, 1)
        return self._state.pellets_eaten / total

    def _apply_milestone_rewards(self, reward: float) -> float:
        completion = self._pellet_completion()
        for threshold, bonus in zip(self._milestone_thresholds, self._milestone_bonuses):
            if threshold in self._milestones_hit:
                continue
            if completion >= threshold:
                scaled = bonus / self._reward_div
                reward += scaled
                self._milestones_hit.add(threshold)
                self._episode_milestone_reward += scaled
        return reward

    def _reset_level_milestones(self) -> None:
        self._milestones_hit = set()

    @staticmethod
    def _min_ghost_dist(state: GameState) -> int:
        pr, pc = state.pacman_pos
        best = 999
        for g in state.ghosts:
            if g.eaten or g.in_house:
                continue
            gr, gc = g.pos
            best = min(best, abs(pr - gr) + abs(pc - gc))
        return best

    def step(self, action: int):
        lives_before = self._state.lives
        prev_score = self._state.score
        prev_pos = self._state.pacman_pos
        completion_before_death = self._pellet_completion()
        _, done = self._state.step(int(action))
        deaths_now = lives_before - self._state.lives
        if deaths_now > 0:
            self._episode_deaths += deaths_now

        # Score-first reward: track the arcade score delta directly, so the
        # optimisation objective == score maximisation. Death is a flat,
        # modest penalty (NOT -500/50=-10 which drowned out pellet signal).
        score_delta = self._state.score - prev_score
        reward = score_delta / self._reward_div + self._step_penalty
        if self._state.pacman_pos == prev_pos:
            reward += self._idle_penalty
        if score_delta == 50 and self._min_ghost_dist(self._state) > _GHOST_NEAR_FOR_POWER:
            reward += self._wasted_power_penalty
        if deaths_now > 0:
            reward += self._death_penalty * deaths_now
            if completion_before_death >= self._endgame_death_threshold:
                reward += self._endgame_death_surcharge * deaths_now

        reward = self._apply_milestone_rewards(reward)

        # Level completion bonus when clearing all pellets
        if self._state._all_pellets_eaten() and self._state.lives > 0:
            reward += self._level_bonus / self._reward_div
            self._state._next_level()
            self._level_clears += 1
            self._max_level_reached = max(self._max_level_reached, self._state.level)
            self._prev_potential = self._potential()
            self._reset_level_milestones()

        # Potential-based shaping (optional; off by default for human-fair training)
        if self._pbrs_coef > 0.0 and not done:
            phi = self._potential()
            reward += self._pbrs_coef * (self._pbrs_gamma * phi - self._prev_potential)
            self._prev_potential = phi

        truncated = (not done) and self._state.step_count >= self._max_steps
        if done and self._level_clears == 0 and self._pellet_completion() >= self._near_miss_threshold:
            reward += self._near_miss_penalty
        self._max_level_reached = max(self._max_level_reached, self._state.level)
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
        obs = np.zeros((self._n_channels, ROWS, COLS), dtype=np.float32)
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
        if st.fruit_active:
            fr, fc = st.fruit_pos
            if 0 <= fr < ROWS and 0 <= fc < COLS:
                obs[6, fr, fc] = 1.0
        for i in range(min(st.lives, len(_LIVES_HUD_POSITIONS))):
            lr, lc = _LIVES_HUD_POSITIONS[i]
            obs[7, lr, lc] = 1.0
        # Level plane: helps policy detect level transitions (maze resets, harder ghosts).
        obs[8, :, :] = min(st.level / 5.0, 1.0)
        ch = _GRID_BASE_CHANNELS
        if self._include_completion_plane:
            obs[ch, :, :] = self._pellet_completion()
            ch += 1
        if self._include_frightened_plane:
            max_ft = max(self._state.get_frightened_duration(self._state.level), 1)
            obs[ch, :, :] = min(self._state.frightened_timer / max_ft, 1.0)
        return obs

    def _info(self) -> dict:
        st = self._state
        total = max(st.total_pellets, 1)
        pellet_completion = st.pellets_eaten / total
        return {
            "score": st.score,
            "lives": st.lives,
            "step": st.step_count,
            "level": st.level,
            "pellets_eaten": st.pellets_eaten,
            "total_pellets": st.total_pellets,
            "pellet_completion": pellet_completion,
            "level_progress": (st.level - 1) + pellet_completion,
            "episode_deaths": self._episode_deaths,
            "max_level_reached": self._max_level_reached,
            "level_clears": self._level_clears,
            "milestone_rewards": self._episode_milestone_reward,
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

    def render_rgb(self) -> np.ndarray:
        from src.utils.pacman_renderer import render_state_rgb
        return render_state_rgb(self._state)

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_rgb()
        if self.render_mode == "ansi":
            helper = PacmanEnv(render_mode="ansi", seed=self._seed)
            helper._state = self._state
            return helper._render_ansi()
        return None

    def close(self):
        pass
