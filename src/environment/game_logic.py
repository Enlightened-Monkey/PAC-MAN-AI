"""
game_logic.py – Core Pac-Man game rules and ghost pathfinding.

Implements board layout, Pac-Man movement, ghost chase/scatter AI
(inspired by the GBA/Z80 original algorithms), and scoring mechanics.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Board layout
# ---------------------------------------------------------------------------

# 0 = empty path, 1 = wall, 2 = pellet, 3 = power pellet
DEFAULT_MAZE: list[list[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 3, 1],
    [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 2, 1, 1, 1, 1],
    [0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0],
    [1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1],
    [1, 1, 1, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 2, 1, 1, 1, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 2, 1],
    [1, 3, 2, 1, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 1, 2, 3, 1],
    [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1],
    [1, 2, 2, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 1],
    [1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1],
    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

ROWS = len(DEFAULT_MAZE)
COLS = len(DEFAULT_MAZE[0])

# Action indices
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

DIRECTION_DELTAS = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

# Scoring
PELLET_SCORE = 10
POWER_PELLET_SCORE = 50
GHOST_BASE_SCORE = 200
LIFE_PENALTY = -500


class Ghost:
    """A single ghost with a scatter target and chase behaviour."""

    FRIGHTENED_DURATION = 30  # steps

    def __init__(
        self,
        name: str,
        start_pos: tuple[int, int],
        scatter_target: tuple[int, int],
    ) -> None:
        self.name = name
        self.start_pos = start_pos
        self.scatter_target = scatter_target
        self.pos: tuple[int, int] = start_pos
        self.frightened_timer: int = 0
        self.eaten: bool = False

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def is_frightened(self) -> bool:
        return self.frightened_timer > 0

    def frighten(self) -> None:
        self.frightened_timer = self.FRIGHTENED_DURATION
        self.eaten = False

    def tick_frighten(self) -> None:
        if self.frightened_timer > 0:
            self.frightened_timer -= 1

    def reset(self) -> None:
        self.pos = self.start_pos
        self.frightened_timer = 0
        self.eaten = False

    # ------------------------------------------------------------------
    # Movement
    # ------------------------------------------------------------------

    def _valid_neighbours(
        self, maze: np.ndarray, exclude: tuple[int, int] | None = None
    ) -> list[tuple[int, int]]:
        r, c = self.pos
        neighbours = []
        for dr, dc in DIRECTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < ROWS and 0 <= nc < COLS and maze[nr, nc] != 1:
                if (nr, nc) != exclude:
                    neighbours.append((nr, nc))
        # Fallback: if all valid moves are blocked (degenerate maze), stay put
        # or return the position we came from to avoid an infinite loop.
        return neighbours or ([(r, c)] if exclude is None else [exclude])

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def move_chase(
        self,
        maze: np.ndarray,
        pacman_pos: tuple[int, int],
        pacman_dir: int,
        prev_pos: tuple[int, int] | None,
    ) -> None:
        """Move toward Pac-Man (Blinky-style direct chase)."""
        target = pacman_pos
        neighbours = self._valid_neighbours(maze, exclude=prev_pos)
        best = min(neighbours, key=lambda n: self._distance(n, target))
        self.pos = best

    def move_scatter(
        self, maze: np.ndarray, prev_pos: tuple[int, int] | None
    ) -> None:
        """Move toward the fixed scatter corner."""
        neighbours = self._valid_neighbours(maze, exclude=prev_pos)
        best = min(
            neighbours, key=lambda n: self._distance(n, self.scatter_target)
        )
        self.pos = best

    def move_frightened(self, maze: np.ndarray, rng: np.random.Generator) -> None:
        """Move randomly when frightened."""
        neighbours = self._valid_neighbours(maze)
        self.pos = neighbours[rng.integers(len(neighbours))]


class GameState:
    """Full mutable game state."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.maze = np.array(DEFAULT_MAZE, dtype=np.int8)
        self.score: int = 0
        self.lives: int = 3
        self.step_count: int = 0
        self.scatter_mode: bool = True
        self.scatter_timer: int = 20
        self.pacman_pos: tuple[int, int] = (16, 9)
        self.pacman_dir: int = ACTION_LEFT
        self.ghosts: list[Ghost] = [
            Ghost("Blinky", (9, 9), (0, COLS - 1)),
            Ghost("Pinky", (9, 10), (0, 0)),
            Ghost("Inky", (10, 9), (ROWS - 1, COLS - 1)),
            Ghost("Clyde", (10, 10), (ROWS - 1, 0)),
        ]
        self._prev_ghost_pos: list[tuple[int, int] | None] = [None] * 4
        self.total_pellets: int = int(np.sum(self.maze == 2)) + int(
            np.sum(self.maze == 3)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        self.__init__(seed=seed)

    def is_terminal(self) -> bool:
        return self.lives <= 0 or self._all_pellets_eaten()

    def _all_pellets_eaten(self) -> bool:
        return not np.any(np.isin(self.maze, [2, 3]))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[int, bool]:
        """
        Apply *action* for Pac-Man and advance the world by one step.

        Returns
        -------
        reward : int
        done   : bool
        """
        self.step_count += 1
        reward = 0

        # --- Move Pac-Man ---
        dr, dc = DIRECTION_DELTAS[action]
        nr, nc = self.pacman_pos[0] + dr, self.pacman_pos[1] + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and self.maze[nr, nc] != 1:
            self.pacman_pos = (nr, nc)
            self.pacman_dir = action

        # --- Collect pellets ---
        cell = self.maze[self.pacman_pos]
        if cell == 2:
            reward += PELLET_SCORE
            self.score += PELLET_SCORE
            self.maze[self.pacman_pos] = 0
        elif cell == 3:
            reward += POWER_PELLET_SCORE
            self.score += POWER_PELLET_SCORE
            self.maze[self.pacman_pos] = 0
            for ghost in self.ghosts:
                ghost.frighten()

        # --- Ghost mode cycling ---
        self.scatter_timer -= 1
        if self.scatter_timer <= 0:
            self.scatter_mode = not self.scatter_mode
            self.scatter_timer = 20 if self.scatter_mode else 40

        # --- Move ghosts ---
        for i, ghost in enumerate(self.ghosts):
            if ghost.eaten:
                continue
            prev = self._prev_ghost_pos[i]
            self._prev_ghost_pos[i] = ghost.pos
            if ghost.is_frightened:
                ghost.move_frightened(self.maze, self.rng)
            elif self.scatter_mode:
                ghost.move_scatter(self.maze, prev)
            else:
                ghost.move_chase(self.maze, self.pacman_pos, self.pacman_dir, prev)
            ghost.tick_frighten()

        # --- Collision detection ---
        for ghost in self.ghosts:
            if ghost.pos == self.pacman_pos:
                if ghost.is_frightened:
                    ghost.eaten = True
                    reward += GHOST_BASE_SCORE
                    self.score += GHOST_BASE_SCORE
                else:
                    self.lives -= 1
                    reward += LIFE_PENALTY
                    self._respawn()
                    break

        done = self.is_terminal()
        return reward, done

    def _respawn(self) -> None:
        self.pacman_pos = (16, 9)
        self.pacman_dir = ACTION_LEFT
        for ghost in self.ghosts:
            ghost.reset()
        self._prev_ghost_pos = [None] * 4

    # ------------------------------------------------------------------
    # Observation vector
    # ------------------------------------------------------------------

    def to_observation(self) -> np.ndarray:
        """
        Flatten the game state into a 1-D float32 observation vector.

        Layout
        ------
        [0]       pacman_row / ROWS
        [1]       pacman_col / COLS
        [2..5]    scatter_mode (1.0 / 0.0) repeated 4 – placeholder for mode
        [6..13]   ghost_row[i]/ROWS, ghost_col[i]/COLS  (4 ghosts → 8 values)
        [14..17]  ghost_frightened[i]  (0/1)
        [18]      lives / 3
        [19]      remaining_pellets / total_pellets
        [20..]    flattened maze (values 0..3 normalised to 0..1)
        """
        obs: list[float] = [
            self.pacman_pos[0] / ROWS,
            self.pacman_pos[1] / COLS,
            float(self.scatter_mode),
            float(self.scatter_mode),
            float(self.scatter_mode),
            float(self.scatter_mode),
        ]
        for ghost in self.ghosts:
            obs.append(ghost.pos[0] / ROWS)
            obs.append(ghost.pos[1] / COLS)
        for ghost in self.ghosts:
            obs.append(float(ghost.is_frightened))
        obs.append(self.lives / 3.0)
        remaining = int(np.sum(np.isin(self.maze, [2, 3])))
        obs.append(remaining / max(self.total_pellets, 1))
        obs.extend((self.maze.flatten() / 3.0).tolist())
        return np.array(obs, dtype=np.float32)
