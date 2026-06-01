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
from src.environment.pacman_env import PacmanEnv, PacmanPrototypeEnv, _OBS_SIZE


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
                assert state.frightened_timer > 0
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
                assert state.frightened_timer > 0
                return
        pytest.skip("No reachable cell for power-pellet test.")

    def test_ghost_frightened_timer_decrements(self):
        state = GameState(seed=0)
        state.frightened_timer = 30
        initial_timer = state.frightened_timer
        state.step(ACTION_LEFT)
        assert state.frightened_timer < initial_timer


class TestCollisionSwap:
    def test_collision_on_adjacent_swap_normal(self):
        """Pac-Man and a ghost in adjacent cells moving towards each other must collide."""
        state = GameState(seed=0)
        state.mode = "chase"
        
        # Position Pac-Man and Blinky (index 0) adjacent to each other
        state.pacman_pos = (23, 13)
        state.pacman_dir = ACTION_LEFT
        
        blinky = state.ghosts[0]
        blinky.in_house = False
        blinky.eaten = False
        blinky.pos = (23, 12)
        blinky.direction = ACTION_RIGHT
        
        # Deactivate all other ghosts by putting them inside house
        for g in state.ghosts[1:]:
            g.in_house = True
            g.pos = (14, 13)

        # Prevent other ghosts from being released
        state._update_ghost_house_release = lambda: None

        # Mock Blinky's target tile to be far to the right to force him to move RIGHT
        original_target_tile = state._target_tile
        def mock_target_tile(ghost):
            if ghost.name == "Blinky":
                return (23, 20)
            return original_target_tile(ghost)
        state._target_tile = mock_target_tile

        initial_lives = state.lives
        # Pac-Man moves LEFT (to 23, 12) and Blinky moves RIGHT (to 23, 13). They swap.
        state.step(ACTION_LEFT)
        
        # This swap must trigger a collision, causing Pac-Man to lose a life
        assert state.lives == initial_lives - 1

    def test_collision_on_adjacent_swap_frightened(self):
        """Frightened ghost and Pac-Man swapping tiles must result in eating the ghost."""
        state = GameState(seed=0)
        state.pacman_pos = (23, 13)
        state.pacman_dir = ACTION_LEFT
        
        blinky = state.ghosts[0]
        blinky.in_house = False
        blinky.eaten = False
        blinky.pos = (23, 12)
        blinky.direction = ACTION_RIGHT
        
        # Deactivate all other ghosts by putting them inside house
        for g in state.ghosts[1:]:
            g.in_house = True
            g.pos = (14, 13)
            
        # Prevent other ghosts from being released
        state._update_ghost_house_release = lambda: None

        state.frightened_timer = 50
        initial_score = state.score
        
        # Mock Blinky's movement in Frightened mode to guarantee he moves RIGHT
        original_move_random = state._move_random
        def mock_move_random(ghost):
            if ghost.name == "Blinky":
                ghost.direction = ACTION_RIGHT
                ghost.pos = (23, 13)
            else:
                original_move_random(ghost)
        state._move_random = mock_move_random
        
        # Step Pac-Man LEFT to swap positions with Blinky
        state.step(ACTION_LEFT)
        
        # Since they swap, they must collide. Since the ghost is frightened, Blinky is eaten.
        assert blinky.eaten is True
        assert state.score > initial_score



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

    def test_all_pellets_eaten_is_not_terminal_in_multilevel(self):
        state = GameState(seed=0)
        state.maze[state.maze == 2] = 0
        state.maze[state.maze == 3] = 0
        # In multi-level Pac-Man, clearing the board does NOT end the episode;
        # instead, it transitions to the next level.
        assert not state.is_terminal()
        assert state._all_pellets_eaten()



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

    def test_truncates_at_max_steps(self):
        env = PacmanEnv(seed=0, max_steps=1)
        env.reset()
        _, _, terminated, truncated, _ = env.step(ACTION_RIGHT)
        assert truncated or terminated


class TestPacmanPrototypeEnv:
    def test_prototype_uses_same_observation_shape(self):
        env = PacmanPrototypeEnv(seed=0)
        obs, _ = env.reset()
        assert obs.shape == (_OBS_SIZE,)

    def test_prototype_has_step_limit_in_info(self):
        env = PacmanPrototypeEnv(seed=0, max_steps=123)
        _, info = env.reset()
        assert info["max_steps"] == 123

    def test_prototype_applies_step_penalty(self):
        baseline = PacmanEnv(seed=0, max_steps=5, step_penalty=0.0, reward_scale=1.0)
        prototype = PacmanPrototypeEnv(
            seed=0,
            max_steps=5,
            step_penalty=-0.5,
            reward_scale=1.0,
        )
        baseline.reset()
        prototype.reset()
        _, reward_base, _, _, _ = baseline.step(ACTION_RIGHT)
        _, reward_proto, _, _, _ = prototype.step(ACTION_RIGHT)
        assert reward_proto == pytest.approx(reward_base - 0.5)


class TestGameStateNextLevel:
    def test_next_level_resets_state_variables(self):
        state = GameState(seed=0)
        state.pellets_eaten = 50
        state.ticks_since_pellet = 10
        state.using_global_dot_counter = True
        state.global_dot_counter = 15
        
        state._next_level()
        
        assert state.level == 2
        assert state.pellets_eaten == 0
        assert state.ticks_since_pellet == 0
        assert state.using_global_dot_counter is False
        assert state.global_dot_counter == 0

