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
    OPPOSITE,
)
from src.environment.pacman_env import PacmanEnv, PacmanPrototypeEnv, PacmanGridEnv, _OBS_SIZE, _GRID_CHANNELS


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
    def test_no_collision_on_adjacent_swap_normal(self):
        """Arcade pass-through: swapping tiles with a ghost does not kill Pac-Man."""
        state = GameState(seed=0)
        state.mode = "chase"

        state.pacman_pos = (23, 13)
        state.pacman_dir = ACTION_LEFT

        blinky = state.ghosts[0]
        blinky.in_house = False
        blinky.eaten = False
        blinky.pos = (23, 12)
        blinky.direction = ACTION_RIGHT

        for g in state.ghosts[1:]:
            g.in_house = True
            g.pos = (14, 13)

        state._update_ghost_house_release = lambda: None

        original_target_tile = state._target_tile
        def mock_target_tile(ghost):
            if ghost.name == "Blinky":
                return (23, 20)
            return original_target_tile(ghost)
        state._target_tile = mock_target_tile

        initial_lives = state.lives
        state.step(ACTION_LEFT)
        assert state.lives == initial_lives

    def test_collision_on_same_cell_normal(self):
        """Pac-Man and a ghost on the same tile must collide."""
        state = GameState(seed=0)
        state.mode = "chase"
        state.pacman_pos = (23, 13)
        state.pacman_dir = ACTION_LEFT
        state.maze[23, 13] = 0

        blinky = state.ghosts[0]
        blinky.in_house = False
        blinky.eaten = False
        blinky.pos = (23, 13)
        blinky.direction = ACTION_RIGHT

        for g in state.ghosts[1:]:
            g.in_house = True
            g.pos = (14, 13)

        state._update_ghost_house_release = lambda: None
        state._move_ghost = lambda g: None
        # Block Pac-Man movement so positions stay overlapping
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = 23 + dr, 13 + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                state.maze[nr, nc] = 1

        initial_lives = state.lives
        state.step(ACTION_LEFT)
        assert state.lives == initial_lives - 1

    def test_no_collision_on_adjacent_swap_frightened(self):
        """Frightened ghost swap does not count as eating (same-cell only)."""
        state = GameState(seed=0)
        state.pacman_pos = (23, 13)
        state.pacman_dir = ACTION_LEFT
        state.maze[23, 12] = 0

        blinky = state.ghosts[0]
        blinky.in_house = False
        blinky.eaten = False
        blinky.pos = (23, 12)
        blinky.direction = ACTION_RIGHT

        for g in state.ghosts[1:]:
            g.in_house = True
            g.pos = (14, 13)

        state._update_ghost_house_release = lambda: None
        state.frightened_timer = 50
        initial_score = state.score

        original_move_ghost = state._move_ghost
        def mock_move_ghost(g):
            if g.name == "Blinky":
                g.pos = (23, 14)
            else:
                original_move_ghost(g)
        state._move_ghost = mock_move_ghost

        state.step(ACTION_LEFT)
        assert not blinky.eaten
        assert state.score == initial_score

    def test_eat_ghost_on_same_cell_frightened(self):
        """Frightened ghost on the same tile as Pac-Man is eaten."""
        state = GameState(seed=0)
        state.pacman_pos = (23, 13)
        state.pacman_dir = ACTION_LEFT
        state.maze[23, 13] = 0

        blinky = state.ghosts[0]
        blinky.in_house = False
        blinky.eaten = False
        blinky.pos = (23, 13)
        blinky.direction = ACTION_RIGHT

        for g in state.ghosts[1:]:
            g.in_house = True
            g.pos = (14, 13)

        state._update_ghost_house_release = lambda: None
        state._move_ghost = lambda g: None
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = 23 + dr, 13 + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                state.maze[nr, nc] = 1

        state.frightened_timer = 50
        initial_score = state.score

        state.step(ACTION_LEFT)
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


class TestGhostHouseRelease:
    def test_personal_dot_release(self):
        state = GameState(seed=0)
        # Initially, Pinky is in house but gets released very quickly (limit is 0)
        pinky = next(g for g in state.ghosts if g.name == "Pinky")
        inky = next(g for g in state.ghosts if g.name == "Inky")
        clyde = next(g for g in state.ghosts if g.name == "Clyde")

        # Pinky limit is 0, Inky is 30, Clyde is 60.
        # Run step_count release logic manually to verify
        # pinky is the first in queue
        assert state.house_queue == ["Pinky", "Inky", "Clyde"]
        
        # Pinky should leave immediately since personal dots (0) >= limit (0)
        state._update_ghost_house_release()
        assert not pinky.in_house
        assert state.house_queue == ["Inky", "Clyde"]

        # Inky requires 30 dots
        # Let's simulate eating 29 dots
        state.pellets_eaten = 29
        state._update_ghost_house_release()
        assert inky.in_house

        # 30th dot eaten
        state.pellets_eaten = 30
        state._update_ghost_house_release()
        assert not inky.in_house
        assert state.house_queue == ["Clyde"]

        # Clyde requires 60 dots
        state.pellets_eaten = 59
        state._update_ghost_house_release()
        assert clyde.in_house

        state.pellets_eaten = 60
        state._update_ghost_house_release()
        assert not clyde.in_house
        assert len(state.house_queue) == 0

    def test_stuck_timer_release(self):
        state = GameState(seed=0)
        inky = next(g for g in state.ghosts if g.name == "Inky")
        
        # Release Pinky first
        state._update_ghost_house_release()
        assert state.house_queue == ["Inky", "Clyde"]
        
        # Set ticks_since_pellet to 40
        state.ticks_since_pellet = 40
        state._update_ghost_house_release()
        # Inky should be released by stuck timer
        assert not inky.in_house
        assert state.house_queue == ["Clyde"]
        assert state.ticks_since_pellet == 0

    def test_global_dot_release(self):
        state = GameState(seed=0)
        pinky = next(g for g in state.ghosts if g.name == "Pinky")
        inky = next(g for g in state.ghosts if g.name == "Inky")
        clyde = next(g for g in state.ghosts if g.name == "Clyde")

        # Simulate Pac-Man death to activate global counter
        state._respawn(is_death=True)
        assert state.using_global_dot_counter is True
        assert state.global_dot_counter == 0
        assert state.house_queue == ["Pinky", "Inky", "Clyde"]

        # Pinky global limit is 7
        state.global_dot_counter = 6
        state._update_ghost_house_release()
        assert pinky.in_house

        state.global_dot_counter = 7
        state._update_ghost_house_release()
        assert not pinky.in_house
        assert state.house_queue == ["Inky", "Clyde"]

        # Inky global limit is 17
        state.global_dot_counter = 16
        state._update_ghost_house_release()
        assert inky.in_house

        state.global_dot_counter = 17
        state._update_ghost_house_release()
        assert not inky.in_house
        assert state.house_queue == ["Clyde"]

        # Clyde global limit is 32
        state.global_dot_counter = 31
        state._update_ghost_house_release()
        assert clyde.in_house

        state.global_dot_counter = 32
        state._update_ghost_house_release()
        assert not clyde.in_house
        assert len(state.house_queue) == 0
        assert state.using_global_dot_counter is False


class TestPacmanGridEnv:
    def test_observation_space_shape(self):
        env = PacmanGridEnv(seed=0)
        assert env.observation_space.shape == (_GRID_CHANNELS, ROWS, COLS)
        assert _GRID_CHANNELS == 9

    def test_reset_obs_within_bounds(self):
        env = PacmanGridEnv(seed=0)
        obs, info = env.reset()
        assert obs.shape == (_GRID_CHANNELS, ROWS, COLS)
        assert env.observation_space.contains(obs)
        assert "pellet_completion" in info

    def test_pbrs_disabled_by_default(self):
        env = PacmanGridEnv(seed=0, human_fair=True)
        assert env._pbrs_coef == 0.0

    def test_level_clear_advances_level(self):
        env = PacmanGridEnv(seed=0, pbrs_coef=0.0)
        env.reset()
        env._state.maze[env._state.maze == 2] = 0
        env._state.maze[env._state.maze == 3] = 0
        _, reward, terminated, truncated, info = env.step(ACTION_LEFT)
        assert not terminated
        assert info["level"] == 2
        assert reward > 0
        assert info["pellet_completion"] == pytest.approx(0.0)
        assert info["level_clears"] == 1

    def test_reward_tracks_score_delta(self):
        env = PacmanGridEnv(seed=0)
        env.reset()
        prev_score = env._state.score
        lives_before = env._state.lives
        _, reward, _, _, info = env.step(ACTION_LEFT)
        if env._state.lives == lives_before and info["level"] == 1:
            expected = (env._state.score - prev_score) / 50.0 + env._step_penalty
            assert reward == pytest.approx(expected)

    def test_death_penalty_not_minus_ten(self):
        # LIFE_PENALTY (-500 raw) must NOT leak into the shaped reward;
        # death costs the flat death_penalty instead.
        env = PacmanGridEnv(seed=0, death_penalty=-3.0)
        assert env._death_penalty == pytest.approx(-3.0)

    def test_constructor_seed_used_for_first_episode_only(self):
        env = PacmanGridEnv(seed=123)
        env.reset()
        assert env._seed is None  # next reset draws fresh RNG

    def test_pellet_completion_tracks_current_level(self):
        env = PacmanGridEnv(seed=0)
        env.reset()
        env._state.pellets_eaten = 50
        info = env._info()
        assert info["pellet_completion"] == pytest.approx(50 / env._state.total_pellets)

    def test_level_plane_in_observation(self):
        env = PacmanGridEnv(seed=0)
        env.reset()
        obs = env._obs()
        assert obs[8].min() == obs[8].max() == pytest.approx(1 / 5.0)
        env._state.level = 3
        obs = env._obs()
        assert obs[8].min() == obs[8].max() == pytest.approx(3 / 5.0)

    def test_lives_channel_reflects_remaining_lives(self):
        env = PacmanGridEnv(seed=0)
        env.reset()
        obs, _ = env.reset()
        assert obs[7].sum() == pytest.approx(3.0)
        env._state.lives = 1
        obs = env._obs()
        assert obs[7].sum() == pytest.approx(1.0)

    def test_fruit_channel_when_active(self):
        env = PacmanGridEnv(seed=0)
        env.reset()
        env._state.fruit_active = True
        obs = env._obs()
        assert obs[6].sum() == pytest.approx(1.0)

    def test_power_pellet_reverses_in_house_ghosts(self):
        state = GameState(seed=0)
        state._update_ghost_house_release = lambda: None
        state._move_ghost = lambda g: None
        inky = next(g for g in state.ghosts if g.name == "Inky")
        assert inky.in_house
        original_dir = inky.direction
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
                assert inky.direction == OPPOSITE[original_dir]
                return
        pytest.skip("No reachable cell for power-pellet reversal test.")

    def test_milestone_bonus_once_at_threshold(self):
        env = PacmanGridEnv(seed=0)
        env.reset()
        env._state.pellets_eaten = env._state.total_pellets - 1
        reward1 = env._apply_milestone_rewards(0.0)
        reward2 = env._apply_milestone_rewards(0.0)
        assert reward1 == pytest.approx((150 + 300 + 600) / 50)
        assert reward2 == pytest.approx(0.0)
        assert len(env._milestones_hit) == 3

    def test_milestone_reset_after_level_clear(self):
        env = PacmanGridEnv(seed=0, pbrs_coef=0.0)
        env.reset()
        env._milestones_hit.add(0.90)
        env._state.maze[env._state.maze == 2] = 0
        env._state.maze[env._state.maze == 3] = 0
        env.step(ACTION_LEFT)
        assert 0.90 not in env._milestones_hit

    def test_endgame_death_surcharge(self):
        import types

        env = PacmanGridEnv(seed=0, endgame_death_surcharge=-2.0)
        env.reset()
        env._state.pellets_eaten = int(0.86 * env._state.total_pellets)

        def _mock_death_step(state, action: int):
            state.lives -= 1
            return 0, state.lives <= 0

        env._state.step = types.MethodType(_mock_death_step, env._state)
        _, reward, _, _, _ = env.step(ACTION_LEFT)
        milestone = 0.0  # 86% completion — below 92% milestone threshold
        expected = (
            env._death_penalty
            + env._endgame_death_surcharge
            + env._step_penalty
            + env._idle_penalty
            + milestone
        )
        assert reward == pytest.approx(expected)

    def test_completion_plane_when_enabled(self):
        env = PacmanGridEnv(seed=0, include_completion_plane=True)
        env.reset()
        assert env.observation_space.shape[0] == 10
        env._state.pellets_eaten = 50
        obs = env._obs()
        expected = 50 / env._state.total_pellets
        assert obs[9].min() == obs[9].max() == pytest.approx(expected)

    def test_easy_endgame_sets_elroy_threshold(self):
        env = PacmanGridEnv(seed=0, easy_endgame=True)
        env.reset()
        assert env._state.elroy_pellets_threshold == 5


