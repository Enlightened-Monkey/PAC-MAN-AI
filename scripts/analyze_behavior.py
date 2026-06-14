#!/usr/bin/env python3
"""Analyze trained agent behavior: idle, power-pellet waste, flee loops."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environment.game_logic import ROWS, COLS, TILE_PELLET, TILE_POWER
from src.environment.pacman_env import PacmanGridEnv
from src.utils.maskable_env import load_trainable_model, predict_action, wrap_with_action_masker
from src.utils.obs_sync import include_flags_from_checkpoint, validate_model_env

_GHOST_NEAR_DIST = 5  # Manhattan tiles


def _unwrap(venv) -> PacmanGridEnv:
    env = venv.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def _min_ghost_dist(st) -> int:
    pr, pc = st.pacman_pos
    best = 999
    for g in st.ghosts:
        if g.eaten or g.in_house:
            continue
        gr, gc = g.pos
        best = min(best, abs(pr - gr) + abs(pc - gc))
    return best


def _cell_has_pellet(maze, r, c) -> bool:
    if not (0 <= r < ROWS and 0 <= c < COLS):
        return False
    return int(maze[r, c]) in (TILE_PELLET, TILE_POWER)


def analyze_episode(
    model: PPO,
    seed: int,
    n_stack: int,
    max_steps: int,
    *,
    include_completion_plane: bool = False,
    include_frightened_plane: bool = False,
) -> dict:
    base = PacmanGridEnv(
        seed=seed, max_steps=max_steps, human_fair=True,
        include_completion_plane=include_completion_plane,
        include_frightened_plane=include_frightened_plane,
    )
    base = wrap_with_action_masker(base)
    venv = DummyVecEnv([lambda e=base: Monitor(e)])
    if n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack, channels_order="first")
    validate_model_env(model, venv, context=f"analyze seed={seed}")

    obs = venv.reset()
    done = False
    prev_pos = base._state.pacman_pos
    prev_score = 0
    idle_steps = 0
    power_waste = 0
    power_smart = 0
    steps_no_pellet_ahead = 0  # moved but cell had no pellet and no power
    position_visits: Counter[tuple[int, int]] = Counter()
    loop_steps = 0
    frightened_eats = 0
    steps = 0
    last_info: dict = {}

    while not done:
        action, _ = predict_action(model, obs, venv, deterministic=True)
        obs, rewards, dones, infos = venv.step(action)
        st = base._state
        last_info = infos[0]
        steps += 1

        pos = st.pacman_pos
        position_visits[pos] += 1
        if pos == prev_pos:
            idle_steps += 1

        score_delta = st.score - prev_score
        if score_delta == 50:  # power pellet
            if _min_ghost_dist(st) > _GHOST_NEAR_DIST:
                power_waste += 1
            else:
                power_smart += 1
        if score_delta >= 200:  # ghost eaten while frightened
            frightened_eats += 1

        if pos != prev_pos:
            r, c = pos
            if not _cell_has_pellet(st.maze, r, c):
                steps_no_pellet_ahead += 1

        if position_visits[pos] >= 4:
            loop_steps += 1

        prev_pos = pos
        prev_score = st.score
        done = bool(dones[0])

    venv.close()
    total = max(st.total_pellets, 1)
    return {
        "seed": seed,
        "steps": steps,
        "score": last_info.get("score", st.score),
        "completion": last_info.get("pellet_completion", st.pellets_eaten / total),
        "deaths": last_info.get("episode_deaths", 0),
        "level_clears": last_info.get("level_clears", 0),
        "idle_pct": 100.0 * idle_steps / max(steps, 1),
        "power_waste": power_waste,
        "power_smart": power_smart,
        "frightened_eats": frightened_eats,
        "loop_pct": 100.0 * loop_steps / max(steps, 1),
        "no_pellet_move_pct": 100.0 * steps_no_pellet_ahead / max(steps, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "ppo_pacman.zip"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=300)
    parser.add_argument("--n-stack", type=int, default=4)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Missing: {ckpt}")
        sys.exit(1)

    inc_c, inc_f = include_flags_from_checkpoint(str(ckpt), args.n_stack)

    def _probe():
        e = PacmanGridEnv(seed=0, max_steps=8000, human_fair=True,
            include_completion_plane=inc_c, include_frightened_plane=inc_f)
        return Monitor(wrap_with_action_masker(e))

    probe = DummyVecEnv([_probe])
    probe = VecFrameStack(probe, n_stack=args.n_stack, channels_order="first")
    model = load_trainable_model(
        str(ckpt.with_suffix("")), probe, use_maskable=True, device="auto",
        create_kwargs=dict(
            learning_rate=1e-4, n_steps=512, batch_size=512,
            ent_coef=0.02, device="auto", tensorboard_log=None,
        ),
    )
    probe.close()
    results = [
        analyze_episode(
            model, seed, args.n_stack, 8000,
            include_completion_plane=inc_c, include_frightened_plane=inc_f,
        )
        for seed in range(args.seed_start, args.seed_start + args.episodes)
    ]

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    clears = sum(1 for r in results if r["level_clears"] > 0)
    lines = [
        "# Behavior Analysis",
        "",
        f"Checkpoint: `{ckpt.name}` | Episodes: {len(results)}",
        "",
        "## Aggregate",
        f"- Mean pellet completion: {mean('completion')*100:.1f}%",
        f"- Mean score: {mean('score'):.0f}",
        f"- Level clears: {clears}/{len(results)}",
        f"- Mean deaths/ep: {mean('deaths'):.2f}",
        f"- Idle (no movement) steps: {mean('idle_pct'):.1f}%",
        f"- Looping (revisit same tile 4+ times): {mean('loop_pct'):.1f}%",
        f"- Moves onto non-pellet cells: {mean('no_pellet_move_pct'):.1f}%",
        f"- Power pellets eaten (ghost far): {mean('power_waste'):.2f}/ep",
        f"- Power pellets eaten (ghost near): {mean('power_smart'):.2f}/ep",
        f"- Ghosts eaten while frightened: {mean('frightened_eats'):.2f}/ep",
        "",
        "## Interpretation",
        "- High idle% = policy outputs blocked moves or stands still.",
        "- High loop% + high no_pellet_move% = panic flee in empty corridors.",
        "- power_waste >> power_smart = eats power pellets for +score, ignores ghost mechanic.",
        "- frightened_eats ~ 0 = never learned to chase blue ghosts.",
    ]

    out = ROOT / "reports" / "behavior_analysis.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
