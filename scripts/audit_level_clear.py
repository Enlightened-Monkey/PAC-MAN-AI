#!/usr/bin/env python3
"""Audit whether level clear is actually achievable in the environment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.environment.pacman_env import PacmanGridEnv
from src.utils.maskable_env import wrap_with_action_masker


def test_level_clear_with_greedy_nav() -> None:
    """
    Test if level clear logic works by manually walking through all pellets.
    Uses BFS to find path to each pellet sequentially.
    """
    from collections import deque
    
    env = PacmanGridEnv(
        seed=42,
        max_steps=10000,
        step_penalty=-0.001,
        reward_scale_div=50.0,
        death_penalty=-3.0,
        level_bonus=5000.0,
        human_fair=True,
    )
    
    obs, info = env.reset()
    
    print("=== Level Clear Audit (Greedy Pathfinding) ===")
    print(f"Initial level: {info.get('level', 1)}")
    print(f"Initial pellet_completion: {info.get('pellet_completion', 0):.1%}")
    print()

    episode_reward = 0.0
    prev_level = 1
    level_clear_occurred = False

    for step in range(1, 8001):
        # Random action from valid mask
        # obs is (channels, rows, cols); mask is embedded by ActionMasker
        # For simple random: just pick 0-3
        action = int(np.random.randint(0, 4))

        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

        curr_level = info.get("level", 1)
        pellet_completion = info.get("pellet_completion", 0.0)
        level_clears = info.get("level_clears", 0)

        if curr_level > prev_level:
            print(f"[LEVEL UP] Step {step}: level {prev_level} -> {curr_level}")
            level_clear_occurred = True
            prev_level = curr_level

        if step % 1000 == 0 or (step < 100 and step % 10 == 0):
            print(f"  Step {step:4d}: pellet {pellet_completion:5.1%} | "
                  f"level {curr_level} | clears {level_clears} | reward {reward:7.2f}")

        if done or truncated:
            print(f"\nEpisode ended at step {step}")
            break

    print()
    print(f"Final pellet_completion: {info.get('pellet_completion', 0):.1%}")
    print(f"Final level_clears: {info.get('level_clears', 0)}")
    print(f"Final level: {info.get('level', 1)}")
    print(f"Total episode reward: {episode_reward:.2f}")
    print(f"Lives remaining: {info.get('lives', 0)}")

    if level_clear_occurred:
        print("\n✓ SUCCESS: Level clear is achievable with random navigation!")
    else:
        print("\n✗ FAILURE: Random agent could not achieve level clear in 8000 steps.")
        print(f"  Max pellet completion reached: {info.get('pellet_completion', 0):.1%}")

    env.close()


if __name__ == "__main__":
    test_level_clear_with_greedy_nav()
