"""
game_logic.py — Faithful tile-discrete reimplementation of the original
Pac-Man arcade (Namco, 1980).

Modelled after "The Pac-Man Dossier" and the project research report. The
game runs on a 28×31 tile grid (one tile = 8×8 px in the arcade, here a
discrete cell), and the four ghosts use their original Target-Tile
pathfinding algorithms with Up > Left > Down > Right tie-breaking and the
deterministic Scatter / Chase / Frightened state machine.

What is reproduced
------------------
- Authentic 28×31 maze (corridors, ghost house, side-tunnel warp)
- Per-ghost Target Tile algorithms:
    Blinky : pacman tile (Cruise Elroy when few pellets remain)
    Pinky  : pacman tile + 4·dir (with the historical Up-direction
             overflow bug: Up means -4 row AND -4 col)
    Inky   : double the vector from Blinky to (pacman + 2·dir)
    Clyde  : pacman tile if Euclidean distance > 8 tiles else scatter corner
- Look-ahead pathfinding: at every tile the ghost picks the next-tile
  exit minimising (Δx² + Δy²) to the Target Tile, never reversing,
  ties broken by Up > Left > Down > Right
- Scatter / Chase wave schedule with global "Reversal" signal on every
  mode transition (except Frightened → previous mode)
- Frightened mode: ghosts move at half speed and pick a uniformly random
  legal exit (still no 180° turn) for the duration of the power pellet
- Red Zones: 4 tiles above the ghost house where ghosts cannot turn UP
  during Chase / Scatter (suspended during Frightened)
- Tunnel: side warp between the leftmost and rightmost columns of the
  tunnel row; ghosts move at half speed inside the tunnel
- Ghost house with personal dot-count release (Blinky out, Pinky 0,
  Inky 30, Clyde 60 at level 1) and a global stuck-timer release
- Ghost-eating combo scoring: 200, 400, 800, 1600 within one Frightened
  phase, eaten ghost goes home as eyes
- Standard pellet (10), power pellet (50), and life-loss penalty
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Action constants (Discrete(4))
# ---------------------------------------------------------------------------

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

DIRECTION_DELTAS: dict[int, tuple[int, int]] = {
    ACTION_UP: (-1, 0),
    ACTION_DOWN: (1, 0),
    ACTION_LEFT: (0, -1),
    ACTION_RIGHT: (0, 1),
}

# Tie-break order for ghost decisions: Up > Left > Down > Right (PDF §9)
TIE_BREAK_ORDER: tuple[int, ...] = (ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT)

OPPOSITE: dict[int, int] = {
    ACTION_UP: ACTION_DOWN,
    ACTION_DOWN: ACTION_UP,
    ACTION_LEFT: ACTION_RIGHT,
    ACTION_RIGHT: ACTION_LEFT,
}

# ---------------------------------------------------------------------------
# Tile codes
# ---------------------------------------------------------------------------

TILE_EMPTY = 0       # walkable, nothing to eat
TILE_WALL = 1
TILE_PELLET = 2
TILE_POWER = 3
TILE_DOOR = 4        # ghost-house door (only ghosts/eyes may cross)
TILE_HOUSE = 5       # ghost-house interior (no pellets, ghosts only)

# ---------------------------------------------------------------------------
# Authentic 28-column × 31-row Pac-Man maze
# ---------------------------------------------------------------------------

# Symbols:  # wall   . pellet   o power pellet   - ghost door
#           _ ghost-house interior   space = empty corridor / tunnel
_MAZE_STR: list[str] = [
    "############################",  # 0
    "#............##............#",  # 1
    "#.####.#####.##.#####.####.#",  # 2
    "#o####.#####.##.#####.####o#",  # 3
    "#.####.#####.##.#####.####.#",  # 4
    "#..........................#",  # 5
    "#.####.##.########.##.####.#",  # 6
    "#.####.##.########.##.####.#",  # 7
    "#......##....##....##......#",  # 8
    "######.##### ## #####.######",  # 9
    "######.##### ## #####.######",  # 10
    "######.##          ##.######",  # 11
    "######.## ###--### ##.######",  # 12
    "######.## #______# ##.######",  # 13
    "      .   #______#   .      ",  # 14  <- tunnel row
    "######.## #______# ##.######",  # 15
    "######.## ######## ##.######",  # 16
    "######.##          ##.######",  # 17
    "######.## ######## ##.######",  # 18
    "######.## ######## ##.######",  # 19
    "#............##............#",  # 20
    "#.####.#####.##.#####.####.#",  # 21
    "#.####.#####.##.#####.####.#",  # 22
    "#o..##.......  .......##..o#",  # 23
    "###.##.##.########.##.##.###",  # 24
    "###.##.##.########.##.##.###",  # 25
    "#......##....##....##......#",  # 26
    "#.##########.##.##########.#",  # 27
    "#.##########.##.##########.#",  # 28
    "#..........................#",  # 29
    "############################",  # 30
]

_CHAR_TO_TILE: dict[str, int] = {
    "#": TILE_WALL,
    ".": TILE_PELLET,
    "o": TILE_POWER,
    "-": TILE_DOOR,
    "_": TILE_HOUSE,
    " ": TILE_EMPTY,
}


def _build_default_maze() -> np.ndarray:
    rows = len(_MAZE_STR)
    cols = len(_MAZE_STR[0])
    maze = np.zeros((rows, cols), dtype=np.int8)
    for r, line in enumerate(_MAZE_STR):
        if len(line) != cols:
            raise ValueError(
                f"Maze row {r} has width {len(line)}, expected {cols}"
            )
        for c, ch in enumerate(line):
            if ch not in _CHAR_TO_TILE:
                raise ValueError(f"Unknown maze char {ch!r} at ({r},{c})")
            maze[r, c] = _CHAR_TO_TILE[ch]
    return maze


# Build once and expose for unit tests / inspection
DEFAULT_MAZE_ARR: np.ndarray = _build_default_maze()
ROWS, COLS = DEFAULT_MAZE_ARR.shape
DEFAULT_MAZE: list[list[int]] = DEFAULT_MAZE_ARR.tolist()

# Side-tunnel warp row & columns (for Pac-Man and ghosts wraparound).
TUNNEL_ROW: int = 14
TUNNEL_COLS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27)

# Pac-Man start position (just below the ghost house, between cols 13/14).
PACMAN_START: tuple[int, int] = (23, 13)
PACMAN_START_DIR: int = ACTION_LEFT

# Ghost spawn tiles (Blinky outside, others inside the house).
GHOST_HOUSE_EXIT: tuple[int, int] = (11, 13)  # tile right above the door

# Scatter corners (PDF: each ghost has a "favourite corner")
SCATTER_CORNERS: dict[str, tuple[int, int]] = {
    "Blinky": (0, COLS - 3),       # top-right
    "Pinky":  (0, 2),              # top-left
    "Inky":   (ROWS - 1, COLS - 1),  # bottom-right
    "Clyde":  (ROWS - 1, 0),       # bottom-left
}

# Red Zones — tiles above the ghost house where Up turns are forbidden
# during Chase / Scatter (PDF §"Jednokierunkowe Strefy Restrykcji").
RED_ZONE_TILES: frozenset[tuple[int, int]] = frozenset({
    (11, 12), (11, 15),
    (17, 12), (17, 15),
})

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

PELLET_SCORE = 10
POWER_PELLET_SCORE = 50
GHOST_BASE_SCORE = 200          # combo: 200, 400, 800, 1600
LIFE_PENALTY = -500

# Extra life awarded when the player first reaches this score (PDF: 10 000 pts)
EXTRA_LIFE_THRESHOLD = 10_000
# Frightened ghosts start flashing (warning) during the last N ticks.
# Arcade value for level 1 is roughly the final 2 seconds (20 ticks).
FRIGHTENED_FLASH_THRESHOLD = 20

# ---------------------------------------------------------------------------
# Bonus fruit (cherry at level 1)
# ---------------------------------------------------------------------------

# Spawn position: open corridor below the ghost house (T-junction at row 17)
FRUIT_SPAWN_POS: tuple[int, int] = (17, 13)
# Pellet-eaten thresholds that trigger each of the two fruit appearances
FRUIT_SPAWN_THRESHOLDS: tuple[int, int] = (70, 170)
# Random duration the fruit stays on screen: 9–10 s → 90–100 ticks (inclusive).
# The variability is deliberate — the original arcade made the timeout
# unpredictable so players cannot exploit fixed-pattern routing.
FRUIT_TIMEOUT_MIN: int = 90
FRUIT_TIMEOUT_MAX: int = 100
# Points for collecting the fruit — level-indexed (level ≥ 8 → Key stays).
# Level: (score, name)
FRUIT_TABLE: dict[int, tuple[int, str]] = {
    1: (100,  "Cherry"),
    2: (300,  "Strawberry"),
    3: (500,  "Orange"),
    4: (700,  "Apple"),
    5: (1000, "Melon"),
    6: (2000, "Galaxian"),
    7: (3000, "Bell"),
    8: (5000, "Key"),
}
# Kept for backwards compatibility (level-1 value)
FRUIT_SCORE: int = FRUIT_TABLE[1][0]

# ---------------------------------------------------------------------------
# Wave schedule (Scatter / Chase) — PDF "Harmonogram Fal" for level 1,
# converted from seconds to ticks (1 step ≈ 0.1 s, so seconds × 10).
# Last entry is Chase forever.
# ---------------------------------------------------------------------------

WAVE_SCHEDULE_LVL1: list[tuple[str, int]] = [
    ("scatter", 70),   # 7 s
    ("chase",   200),  # 20 s
    ("scatter", 70),
    ("chase",   200),
    ("scatter", 50),
    ("chase",   200),
    ("scatter", 50),
    ("chase",   10**9),  # effectively forever
]

# Frightened phase length and combo reset
FRIGHTENED_DURATION = 60   # ~6 s
GHOST_EATEN_RETURN_DELAY = 5  # ticks an eaten ghost stays "eyes" before respawn

# Cruise Elroy thresholds (level 1, from the PDF / Pac-Man Dossier)
ELROY1_PELLETS_REMAINING = 20
ELROY2_PELLETS_REMAINING = 10

# Personal dot-count release thresholds (level 1)
DOT_RELEASE_LIMITS: dict[str, int] = {
    "Blinky": 0,
    "Pinky":  0,
    "Inky":   30,
    "Clyde":  60,
}
GLOBAL_RELEASE_TIMEOUT = 40  # ticks without eating a pellet → force-release

# Post-death global dot-counter limits (PDF §"Global Dot Counter"):
# after Pac-Man loses a life, a single shared counter replaces personal counters.
# Pinky leaves at 7, Inky at 17, Clyde at 32 global dots.
GLOBAL_DOT_COUNTER_LIMITS: dict[str, int] = {
    "Pinky": 7,
    "Inky":  17,
    "Clyde": 32,
}

# ---------------------------------------------------------------------------
# Ghost
# ---------------------------------------------------------------------------


@dataclass
class Ghost:
    """A single ghost agent."""

    name: str
    spawn_pos: tuple[int, int]
    pos: tuple[int, int]
    direction: int = ACTION_LEFT
    in_house: bool = False
    eaten: bool = False               # currently going home as eyes
    eyes_timer: int = 0               # ticks left as eyes
    move_phase: int = 0               # used for tunnel / frightened slowdown

    # ----- helpers -----

    def reset(self) -> None:
        self.pos = self.spawn_pos
        self.direction = ACTION_LEFT
        self.eaten = False
        self.eyes_timer = 0
        self.move_phase = 0
        # in_house flag is reset by GameState (depends on the ghost)

    @property
    def scatter_corner(self) -> tuple[int, int]:
        return SCATTER_CORNERS[self.name]


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    """
    Mutable Pac-Man game state (one episode).
    """

    # populated by __post_init__
    rng: np.random.Generator = field(default=None, repr=False)  # type: ignore[assignment]
    maze: np.ndarray = field(default=None, repr=False)          # type: ignore[assignment]
    score: int = 0
    lives: int = 3
    step_count: int = 0

    # Pac-Man
    pacman_pos: tuple[int, int] = PACMAN_START
    pacman_dir: int = PACMAN_START_DIR

    # Ghosts (created in __post_init__)
    ghosts: list[Ghost] = field(default_factory=list)

    # Wave / mode state
    wave_index: int = 0
    wave_timer: int = 0
    mode: str = "scatter"        # "scatter" | "chase"
    frightened_timer: int = 0
    ghost_combo: int = 0         # 0..3 → score = 200·2^combo
    pending_reversal: bool = False

    # Ghost-house release
    pellets_eaten: int = 0
    ticks_since_pellet: int = 0

    # Cached
    total_pellets: int = 0

    # Backwards-compatible attribute used by to_observation()
    @property
    def scatter_mode(self) -> bool:  # type: ignore[override]
        return self.mode == "scatter"

    @property
    def frightened_flashing(self) -> bool:
        """True during the final FRIGHTENED_FLASH_THRESHOLD ticks of frightened phase.

        Signals the agent that the safety window is about to close — a cue
        analogous to the blinking ghosts the player sees on screen.
        """
        return 0 < self.frightened_timer <= FRIGHTENED_FLASH_THRESHOLD

    def __init__(self, seed: int | None = None, level: int = 1) -> None:
        self.rng = np.random.default_rng(seed)
        self.maze = DEFAULT_MAZE_ARR.copy()
        self.score = 0
        self.lives = 3
        self.step_count = 0
        self.level = max(1, level)  # current level — determines fruit type/score

        self.pacman_pos = PACMAN_START
        self.pacman_dir = PACMAN_START_DIR

        # Ghosts: Blinky starts above the house, the others inside.
        self.ghosts = [
            Ghost("Blinky", spawn_pos=GHOST_HOUSE_EXIT, pos=GHOST_HOUSE_EXIT,
                  direction=ACTION_LEFT, in_house=False),
            Ghost("Pinky",  spawn_pos=(14, 13), pos=(14, 13),
                  direction=ACTION_UP, in_house=True),
            Ghost("Inky",   spawn_pos=(14, 11), pos=(14, 11),
                  direction=ACTION_UP, in_house=True),
            Ghost("Clyde",  spawn_pos=(14, 16), pos=(14, 16),
                  direction=ACTION_UP, in_house=True),
        ]

        # Wave schedule
        self.wave_index = 0
        self.wave_timer = WAVE_SCHEDULE_LVL1[0][1]
        self.mode = WAVE_SCHEDULE_LVL1[0][0]
        self.frightened_timer = 0
        self.ghost_combo = 0
        self.pending_reversal = False

        # Release counters
        self.pellets_eaten = 0
        self.ticks_since_pellet = 0
        # Global dot counter — activated after Pac-Man dies (PDF §"Global Dot Counter")
        self.using_global_dot_counter: bool = False
        self.global_dot_counter: int = 0
        # 1-UP flag: grant at most one extra life per game at EXTRA_LIFE_THRESHOLD
        self.extra_life_awarded: bool = False

        # Bonus fruit state
        self.fruit_active: bool = False
        self.fruit_pos: tuple[int, int] = FRUIT_SPAWN_POS
        self.fruit_timer: int = 0
        self.fruit_spawned_count: int = 0  # 0, 1, or 2
        # Cache the level-specific fruit score to avoid repeated lookups
        self._fruit_score: int = FRUIT_TABLE.get(self.level, FRUIT_TABLE[8])[0]

        self.total_pellets = int(np.sum(self.maze == TILE_PELLET)) + int(
            np.sum(self.maze == TILE_POWER)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        self.__init__(seed=seed)

    def is_terminal(self) -> bool:
        return self.lives <= 0 or self._all_pellets_eaten()

    def _all_pellets_eaten(self) -> bool:
        return not bool(np.any(np.isin(self.maze, [TILE_PELLET, TILE_POWER])))

    # ------------------------------------------------------------------
    # Walkability
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_tunnel(pos: tuple[int, int]) -> tuple[int, int]:
        """Apply side-tunnel wraparound on the tunnel row."""
        r, c = pos
        if r == TUNNEL_ROW:
            if c < 0:
                return (r, COLS - 1)
            if c >= COLS:
                return (r, 0)
        return pos

    def _walkable_for_pacman(self, pos: tuple[int, int]) -> bool:
        r, c = pos
        if r == TUNNEL_ROW and (c < 0 or c >= COLS):
            return True  # off-screen tunnel cells are walkable (warp)
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return False
        t = self.maze[r, c]
        return t in (TILE_EMPTY, TILE_PELLET, TILE_POWER)

    def _walkable_for_ghost(
        self, ghost: Ghost, pos: tuple[int, int]
    ) -> bool:
        r, c = pos
        if r == TUNNEL_ROW and (c < 0 or c >= COLS):
            return True
        if not (0 <= r < ROWS and 0 <= c < COLS):
            return False
        t = self.maze[r, c]
        if t == TILE_WALL:
            return False
        if t == TILE_DOOR:
            # only crossable when going home (eaten) or leaving house
            return ghost.eaten or ghost.in_house
        if t == TILE_HOUSE:
            return ghost.in_house or ghost.eaten
        return True

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[int, bool]:
        """
        Apply one action for Pac-Man, then update wave timers, ghost-house
        releases, ghosts and collisions. Returns (reward, done).
        """
        self.step_count += 1
        reward = 0

        # 1) Move Pac-Man (with tunnel wraparound)
        dr, dc = DIRECTION_DELTAS[action]
        target = self._wrap_tunnel((self.pacman_pos[0] + dr, self.pacman_pos[1] + dc))
        if self._walkable_for_pacman(target):
            self.pacman_pos = target
            self.pacman_dir = action

        # 2) Pellet / power-pellet collection
        cell = int(self.maze[self.pacman_pos])
        if cell == TILE_PELLET:
            reward += PELLET_SCORE
            self.score += PELLET_SCORE
            self.maze[self.pacman_pos] = TILE_EMPTY
            self.pellets_eaten += 1
            self.ticks_since_pellet = 0
            if self.using_global_dot_counter:
                self.global_dot_counter += 1
        elif cell == TILE_POWER:
            reward += POWER_PELLET_SCORE
            self.score += POWER_PELLET_SCORE
            self.maze[self.pacman_pos] = TILE_EMPTY
            self.pellets_eaten += 1
            self.ticks_since_pellet = 0
            if self.using_global_dot_counter:
                self.global_dot_counter += 1
            # Frighten all non-eaten ghosts and request a Reversal
            self.frightened_timer = FRIGHTENED_DURATION
            self.ghost_combo = 0
            for g in self.ghosts:
                if not g.eaten and not g.in_house:
                    g.direction = OPPOSITE[g.direction]
        else:
            self.ticks_since_pellet += 1

        # 1-UP: grant one extra life when score first crosses 10 000
        if not self.extra_life_awarded and self.score >= EXTRA_LIFE_THRESHOLD:
            self.lives += 1
            self.extra_life_awarded = True

        # 2b) Bonus fruit — spawn at thresholds; collect on contact.
        #     The timer countdown (and its ghost-freeze) happens at step 7
        #     after we know whether a ghost was eaten this tick.
        if (
            not self.fruit_active
            and self.fruit_spawned_count < len(FRUIT_SPAWN_THRESHOLDS)
            and self.pellets_eaten >= FRUIT_SPAWN_THRESHOLDS[self.fruit_spawned_count]
        ):
            self.fruit_active = True
            # Random duration: 90–100 ticks, mirrors the arcade's variability
            self.fruit_timer = int(self.rng.integers(FRUIT_TIMEOUT_MIN,
                                                     FRUIT_TIMEOUT_MAX + 1))

        if self.fruit_active and self.pacman_pos == FRUIT_SPAWN_POS:
            # Pac-Man collects the fruit
            reward += self._fruit_score
            self.score += self._fruit_score
            self.fruit_active = False
            self.fruit_timer = 0
            self.fruit_spawned_count += 1

        # 3) Wave schedule (Scatter ↔ Chase) — only ticks when NOT frightened
        if self.frightened_timer == 0:
            self._tick_wave_schedule()
        else:
            self.frightened_timer -= 1
            if self.frightened_timer == 0:
                # End of frightened: combo resets, no reversal (PDF)
                self.ghost_combo = 0

        # 4) Ghost-house release (personal & global counters)
        self._update_ghost_house_release()

        # 5) Move every ghost
        for g in self.ghosts:
            self._move_ghost(g)
        # The Reversal signal (if any) only acts on this tick.
        self.pending_reversal = False

        # 6) Collision detection
        ghost_eaten_this_tick = False
        for g in self.ghosts:
            if g.eaten or g.in_house:
                continue
            if g.pos == self.pacman_pos:
                if self.frightened_timer > 0:
                    # Eat ghost — combo: 200, 400, 800, 1600
                    pts = GHOST_BASE_SCORE * (2 ** self.ghost_combo)
                    self.ghost_combo = min(self.ghost_combo + 1, 3)
                    reward += pts
                    self.score += pts
                    g.eaten = True
                    g.eyes_timer = GHOST_EATEN_RETURN_DELAY
                    ghost_eaten_this_tick = True
                else:
                    self.lives -= 1
                    reward += LIFE_PENALTY
                    self._respawn()
                    break

        # 7) Fruit timer countdown.
        #    When Pac-Man eats a frightened ghost the global arcade timer freezes
        #    for that fraction of a second, halting the fruit timeout too.
        if self.fruit_active and not ghost_eaten_this_tick:
            self.fruit_timer -= 1
            if self.fruit_timer <= 0:
                # Fruit expired without being collected
                self.fruit_active = False
                self.fruit_spawned_count += 1

        return reward, self.is_terminal()

    # ------------------------------------------------------------------
    # Wave schedule
    # ------------------------------------------------------------------

    def _tick_wave_schedule(self) -> None:
        self.wave_timer -= 1
        if self.wave_timer > 0:
            return
        # Advance to next wave
        if self.wave_index < len(WAVE_SCHEDULE_LVL1) - 1:
            self.wave_index += 1
            new_mode, new_len = WAVE_SCHEDULE_LVL1[self.wave_index]
            if new_mode != self.mode:
                self.pending_reversal = True
            self.mode = new_mode
            self.wave_timer = new_len
        else:
            # Last wave (Chase forever) — keep timer high
            self.wave_timer = 10**9

    # ------------------------------------------------------------------
    # Ghost house
    # ------------------------------------------------------------------

    def _update_ghost_house_release(self) -> None:
        if self.using_global_dot_counter:
            # Post-death mode: shared counter with tighter limits (Pinky=7, Inky=17, Clyde=32)
            for g in self.ghosts:
                if not g.in_house:
                    continue
                limit = GLOBAL_DOT_COUNTER_LIMITS.get(g.name, 0)
                if self.global_dot_counter >= limit:
                    self._release_ghost(g)
                    # Once all in-house ghosts are out, revert to personal counters
                    if not any(gh.in_house for gh in self.ghosts):
                        self.using_global_dot_counter = False
                    return
        else:
            # Normal mode: per-ghost personal dot counters (canonical priority order)
            for g in self.ghosts:
                if not g.in_house:
                    continue
                limit = DOT_RELEASE_LIMITS.get(g.name, 0)
                if self.pellets_eaten >= limit:
                    self._release_ghost(g)
                    # release at most one per tick to mimic the original
                    return
        # Global stuck-timer release (PDF: 4 s of no eaten pellet → force-out)
        if self.ticks_since_pellet >= GLOBAL_RELEASE_TIMEOUT:
            for g in self.ghosts:
                if g.in_house:
                    self._release_ghost(g)
                    self.ticks_since_pellet = 0
                    return

    def _release_ghost(self, g: Ghost) -> None:
        """Teleport a ghost from inside the house up to the exit tile."""
        g.in_house = False
        g.pos = GHOST_HOUSE_EXIT
        g.direction = ACTION_LEFT  # exit moving left, like in the arcade

    # ------------------------------------------------------------------
    # Ghost movement
    # ------------------------------------------------------------------

    def _move_ghost(self, g: Ghost) -> None:
        # Tunnel slowdown: skip every other tick when on a tunnel column
        in_tunnel = (g.pos[0] == TUNNEL_ROW and g.pos[1] in TUNNEL_COLS)
        # Eaten ghosts (eyes) are the fastest objects — never slowed, even in tunnel
        slow = (not g.eaten) and (in_tunnel or self.frightened_timer > 0)
        if slow:
            g.move_phase = (g.move_phase + 1) % 2
            if g.move_phase == 1:
                return

        # Eaten ghosts ("eyes") head straight back to the spawn tile
        if g.eaten:
            target = g.spawn_pos
            self._step_towards(g, target, allow_reverse=False, in_red_zone_rule=False)
            if g.pos == g.spawn_pos:
                # Re-enter the house (placeholder — instantly become normal again)
                g.eaten = False
                g.eyes_timer = 0
                g.in_house = False
                g.direction = ACTION_UP
            return

        # In-house ghosts simply bob up and down inside the box
        if g.in_house:
            r, c = g.pos
            if g.direction == ACTION_UP and self.maze[r - 1, c] == TILE_HOUSE:
                g.pos = (r - 1, c)
            elif g.direction == ACTION_DOWN and self.maze[r + 1, c] == TILE_HOUSE:
                g.pos = (r + 1, c)
            else:
                g.direction = OPPOSITE[g.direction]
            return

        # Apply pending Reversal (only once, on mode transition tick)
        if self.pending_reversal and self.frightened_timer == 0:
            g.direction = OPPOSITE[g.direction]
        # Clear flag after the loop iteration finishes (handled in step())

        # Choose mode-appropriate target tile
        if self.frightened_timer > 0:
            self._move_random(g)
            return
        target = self._target_tile(g)
        self._step_towards(g, target, allow_reverse=False, in_red_zone_rule=True)

    def _step_towards(
        self,
        g: Ghost,
        target: tuple[int, int],
        *,
        allow_reverse: bool,
        in_red_zone_rule: bool,
    ) -> None:
        r, c = g.pos
        prev_dir = g.direction
        candidates: list[tuple[int, int, tuple[int, int]]] = []
        # PDF §9: Up > Left > Down > Right tie-break
        for action in TIE_BREAK_ORDER:
            if not allow_reverse and action == OPPOSITE[prev_dir]:
                continue
            # Red Zone: no Up turns at the four marked tiles in Chase/Scatter
            if (
                in_red_zone_rule
                and action == ACTION_UP
                and (r, c) in RED_ZONE_TILES
            ):
                continue
            dr, dc = DIRECTION_DELTAS[action]
            new_pos = self._wrap_tunnel((r + dr, c + dc))
            if not self._walkable_for_ghost(g, new_pos):
                continue
            d2 = (new_pos[0] - target[0]) ** 2 + (new_pos[1] - target[1]) ** 2
            candidates.append((d2, TIE_BREAK_ORDER.index(action), new_pos))
            # remember which action this candidate represents
        if not candidates:
            # Dead-end: forced reverse
            g.direction = OPPOSITE[prev_dir]
            dr, dc = DIRECTION_DELTAS[g.direction]
            new_pos = self._wrap_tunnel((r + dr, c + dc))
            if self._walkable_for_ghost(g, new_pos):
                g.pos = new_pos
            return
        # Pick the candidate with smallest d², ties broken by the predefined
        # action order (lower index in TIE_BREAK_ORDER wins).
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, action_idx, new_pos = candidates[0]
        g.direction = TIE_BREAK_ORDER[action_idx]
        g.pos = new_pos

    def _move_random(self, g: Ghost) -> None:
        r, c = g.pos
        choices: list[tuple[int, tuple[int, int]]] = []
        for action in (ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT):
            if action == OPPOSITE[g.direction]:
                continue
            dr, dc = DIRECTION_DELTAS[action]
            new_pos = self._wrap_tunnel((r + dr, c + dc))
            if self._walkable_for_ghost(g, new_pos):
                choices.append((action, new_pos))
        if not choices:
            g.direction = OPPOSITE[g.direction]
            dr, dc = DIRECTION_DELTAS[g.direction]
            new_pos = self._wrap_tunnel((r + dr, c + dc))
            if self._walkable_for_ghost(g, new_pos):
                g.pos = new_pos
            return
        action, new_pos = choices[int(self.rng.integers(len(choices)))]
        g.direction = action
        g.pos = new_pos

    # ------------------------------------------------------------------
    # Per-ghost Target Tile algorithms (PDF "Architektura Behawioralna")
    # ------------------------------------------------------------------

    def _target_tile(self, g: Ghost) -> tuple[int, int]:
        # Scatter mode: each ghost heads for its favourite corner —
        # except Blinky in Cruise Elroy, who keeps chasing.
        remaining = self.total_pellets - self.pellets_eaten
        elroy = (g.name == "Blinky") and (remaining <= ELROY1_PELLETS_REMAINING)

        if self.mode == "scatter" and not elroy:
            return g.scatter_corner

        # Chase mode (or Elroy override)
        if g.name == "Blinky":
            return self.pacman_pos

        if g.name == "Pinky":
            # 4 tiles ahead of Pac-Man, with the historical Up-direction
            # overflow bug: Up means -4 row AND -4 col.
            dr, dc = DIRECTION_DELTAS[self.pacman_dir]
            r = self.pacman_pos[0] + 4 * dr
            c = self.pacman_pos[1] + 4 * dc
            if self.pacman_dir == ACTION_UP:
                c -= 4
            return (r, c)

        if g.name == "Inky":
            # Pivot tile = Pac-Man + 2 ahead (with the same Up bug)
            dr, dc = DIRECTION_DELTAS[self.pacman_dir]
            pr = self.pacman_pos[0] + 2 * dr
            pc = self.pacman_pos[1] + 2 * dc
            if self.pacman_dir == ACTION_UP:
                pc -= 2
            blinky = self.ghosts[0].pos
            # Vector Blinky→pivot, doubled
            return (pr + (pr - blinky[0]), pc + (pc - blinky[1]))

        if g.name == "Clyde":
            # If Euclidean distance > 8 tiles → chase like Blinky,
            # otherwise → run to scatter corner.
            dr = g.pos[0] - self.pacman_pos[0]
            dc = g.pos[1] - self.pacman_pos[1]
            if dr * dr + dc * dc > 8 * 8:
                return self.pacman_pos
            return g.scatter_corner

        return self.pacman_pos  # fallback (should never trigger)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _respawn(self) -> None:
        self.pacman_pos = PACMAN_START
        self.pacman_dir = PACMAN_START_DIR
        for g in self.ghosts:
            g.reset()
        # Re-place ghosts in their starting house slots
        self.ghosts[0].in_house = False
        self.ghosts[0].pos = GHOST_HOUSE_EXIT
        self.ghosts[1].in_house = True
        self.ghosts[1].pos = (14, 13)
        self.ghosts[2].in_house = True
        self.ghosts[2].pos = (14, 11)
        self.ghosts[3].in_house = True
        self.ghosts[3].pos = (14, 16)
        self.ghost_combo = 0
        self.frightened_timer = 0
        self.ticks_since_pellet = 0
        # Activate global dot counter after death (unless Clyde is already outside)
        clyde = next(g for g in self.ghosts if g.name == "Clyde")
        if clyde.in_house:
            self.using_global_dot_counter = True
            self.global_dot_counter = 0

    # ------------------------------------------------------------------
    # Observation vector
    # ------------------------------------------------------------------

    def to_observation(self) -> np.ndarray:
        """
        Flatten the game state into a 1-D float32 vector in [0, 1].

        Layout
        ------
        [0]       pacman_row / ROWS
        [1]       pacman_col / COLS
        [2..5]    mode flags (4 floats): scatter, chase, frightened, flashing
        [6..13]   ghost (row/ROWS, col/COLS) for each of the 4 ghosts
        [14..17]  ghost frightened-active flag (1.0 if frightened & not eaten)
        [18]      lives / 3
        [19]      remaining_pellets / total_pellets
        [20..]    flattened maze (each cell / max tile code)
        """
        obs: list[float] = [
            self.pacman_pos[0] / ROWS,
            self.pacman_pos[1] / COLS,
            float(self.mode == "scatter"),
            float(self.mode == "chase"),
            float(self.frightened_timer > 0),
            float(self.frightened_flashing),
        ]
        for g in self.ghosts:
            obs.append(g.pos[0] / ROWS)
            obs.append(g.pos[1] / COLS)
        for g in self.ghosts:
            obs.append(float(self.frightened_timer > 0 and not g.eaten))
        obs.append(self.lives / 3.0)
        remaining = int(np.sum(np.isin(self.maze, [TILE_PELLET, TILE_POWER])))
        obs.append(remaining / max(self.total_pellets, 1))
        obs.extend((self.maze.astype(np.float32) / 5.0).flatten().tolist())
        # Bonus fruit: active flag + normalised position
        obs.append(float(self.fruit_active))
        obs.append(FRUIT_SPAWN_POS[0] / ROWS if self.fruit_active else 0.0)
        obs.append(FRUIT_SPAWN_POS[1] / COLS if self.fruit_active else 0.0)
        return np.array(obs, dtype=np.float32)
