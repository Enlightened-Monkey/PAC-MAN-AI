"""
test_environment.py – Unit tests for the Pac-Man game logic and environment.

Run with:
    pytest tests/test_environment.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environment.game_logic import (
    GameState,
    ACTION_UP,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ROWS,
    COLS,
    PELLET_SCORE,
    POWER_PELLET_SCORE,
    GHOST_BASE_SCORE,
    LIFE_PENALTY,
    DEFAULT_MAZE,
)
from src.environment.pacman_env import PacmanEnv, _OBS_SIZE


# ---------------------------------------------------------------------------
# GameState tests
# ---------------------------------------------------------------------------


class TestGameStateInit:
    def test_maze_shape(self):
        state = GameState()
        assert state.maze.shape == (ROWS, COLS)

    def test_initial_lives(self):
        state = GameState()
        assert state.lives == 3

    def test_initial_score(self):
        state = GameState()
        assert state.score == 0

    def test_four_ghosts(self):
        state = GameState()
        assert len(state.ghosts) == 4

    def test_total_pellets_positive(self):
        state = GameState()
        assert state.total_pellets > 0

    def test_not_terminal_at_start(self):
        state = GameState()
        assert not state.is_terminal()


class TestGameStateReset:
    def test_reset_restores_score(self):
        state = GameState(seed=0)
        # eat some pellets via steps
        for _ in range(10):
            state.step(ACTION_RIGHT)
        state.reset(seed=0)
        assert state.score == 0

    def test_reset_restores_lives(self):
        state = GameState(seed=0)
        state.lives = 1
        state.reset(seed=0)
        assert state.lives == 3


class TestPacmanMovement:
    def test_wall_blocked(self):
        """Pac-Man must not move into a wall cell."""
        state = GameState(seed=42)
        initial_pos = state.pacman_pos
        # Force Pac-Man against a wall by checking neighbours
        maze = state.maze
        r, c = initial_pos
        for action, (dr, dc) in [
            (ACTION_UP, (-1, 0)),
            (ACTION_DOWN, (1, 0)),
            (ACTION_LEFT, (0, -1)),
            (ACTION_RIGHT, (0, 1)),
        ]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr, nc] == 1:
                state.step(action)
                assert state.pacman_pos == initial_pos, (
                    f"Pac-Man should not move into wall at ({nr},{nc})"
                )
                break

    def test_valid_move_changes_position(self):
        """Pac-Man should move when the target cell is not a wall."""
        state = GameState(seed=0)
        r, c = state.pacman_pos
        maze = state.maze
        for action, (dr, dc) in [
            (ACTION_UP, (-1, 0)),
            (ACTION_DOWN, (1, 0)),
            (ACTION_LEFT, (0, -1)),
            (ACTION_RIGHT, (0, 1)),
        ]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr, nc] != 1:
                state.step(action)
                assert state.pacman_pos != (r, c)
                break


class TestScoring:
    def test_pellet_collection(self):
        """Walking onto a pellet cell increments the score by PELLET_SCORE."""
        state = GameState(seed=0)
        r, c = state.pacman_pos
        maze = state.maze
        # Find an adjacent pellet
        for action, (dr, dc) in [
            (ACTION_UP, (-1, 0)),
            (ACTION_DOWN, (1, 0)),
            (ACTION_LEFT, (0, -1)),
            (ACTION_RIGHT, (0, 1)),
        ]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr, nc] == 2:
                reward, _ = state.step(action)
                assert state.score >= PELLET_SCORE
                assert reward >= PELLET_SCORE
                return
        pytest.skip("No adjacent pellet found at starting position.")

    def test_power_pellet_collection(self):
        """Power pellet awards POWER_PELLET_SCORE and frightens ghosts."""
        state = GameState(seed=0)
        # Place a power pellet directly adjacent to Pac-Man
        r, c = state.pacman_pos
        maze = state.maze
        for action, (dr, dc) in [
            (ACTION_UP, (-1, 0)),
            (ACTION_DOWN, (1, 0)),
            (ACTION_LEFT, (0, -1)),
            (ACTION_RIGHT, (0, 1)),
        ]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr, nc] != 1:
                maze[nr, nc] = 3
                reward, _ = state.step(action)
                assert reward >= POWER_PELLET_SCORE
                assert all(g.is_frightened for g in state.ghosts)
                return
        pytest.skip("No reachable cell found adjacent to Pac-Man.")


class TestGhostFrightened:
    def test_ghost_frightened_after_power_pellet(self):
        state = GameState(seed=0)
        r, c = state.pacman_pos
        for action, (dr, dc) in [
            (ACTION_UP, (-1, 0)),
            (ACTION_DOWN, (1, 0)),
            (ACTION_LEFT, (0, -1)),
            (ACTION_RIGHT, (0, 1)),
        ]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and state.maze[nr, nc] != 1:
                state.maze[nr, nc] = 3
                state.step(action)
                assert all(g.is_frightened for g in state.ghosts)
                return
        pytest.skip("No reachable cell for power-pellet test.")

    def test_ghost_frightened_timer_decrements(self):
        state = GameState(seed=0)
        for ghost in state.ghosts:
            ghost.frighten()
        initial_timer = state.ghosts[0].frightened_timer
        state.step(ACTION_LEFT)
        assert state.ghosts[0].frightened_timer < initial_timer


class TestObservationVector:
    def test_observation_shape(self):
        state = GameState(seed=0)
        obs = state.to_observation()
        assert obs.shape == (_OBS_SIZE,)

    def test_observation_dtype(self):
        state = GameState(seed=0)
        obs = state.to_observation()
        assert obs.dtype == np.float32

    def test_observation_range(self):
        state = GameState(seed=0)
        obs = state.to_observation()
        assert obs.min() >= 0.0 and obs.max() <= 1.0


class TestTerminalConditions:
    def test_no_lives_is_terminal(self):
        state = GameState(seed=0)
        state.lives = 0
        assert state.is_terminal()

    def test_all_pellets_eaten_is_terminal(self):
        state = GameState(seed=0)
        state.maze[state.maze == 2] = 0
        state.maze[state.maze == 3] = 0
        assert state.is_terminal()


# ---------------------------------------------------------------------------
# PacmanEnv tests
# ---------------------------------------------------------------------------


class TestPacmanEnv:
    def test_observation_space_shape(self):
        env = PacmanEnv()
        assert env.observation_space.shape == (_OBS_SIZE,)

    def test_action_space_size(self):
        env = PacmanEnv()
        assert env.action_space.n == 4

    def test_reset_returns_valid_obs(self):
        env = PacmanEnv(seed=0)
        obs, info = env.reset()
        assert obs.shape == (_OBS_SIZE,)
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        env = PacmanEnv(seed=0)
        env.reset()
        result = env.step(ACTION_RIGHT)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (_OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_obs_within_bounds(self):
        env = PacmanEnv(seed=0)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_multiple_episodes(self):
        """Environment can be reset and played multiple times."""
        env = PacmanEnv(seed=42)
        for _ in range(3):
            obs, _ = env.reset()
            done = False
            steps = 0
            while not done and steps < 50:
                action = env.action_space.sample()
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                steps += 1
            assert obs.shape == (_OBS_SIZE,)

    def test_ansi_render(self):
        env = PacmanEnv(render_mode="ansi", seed=0)
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str)
        assert "C" in rendered  # Pac-Man character
