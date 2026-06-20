#!/usr/bin/env python3
"""Validate PPO checkpoint vs PacmanGridEnv observation shapes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environment.pacman_env import PacmanGridEnv
from src.utils.obs_sync import (
    obs_channel_config_from_checkpoint,
    peek_checkpoint,
    validate_model_env,
)

N_STACK = 4


def build_vec(include_c: bool, include_f: bool, include_d: bool, max_steps: int = 2500):
    def make():
        return Monitor(
            PacmanGridEnv(
                seed=0,
                max_steps=max_steps,
                human_fair=True,
                include_completion_plane=include_c,
                include_frightened_plane=include_f,
                include_derived_planes=include_d,
            )
        )

    v = DummyVecEnv([make])
    return VecFrameStack(v, n_stack=N_STACK, channels_order="first")


def main() -> None:
    ckpt = ROOT / "models" / "ppo_pacman.zip"
    if not ckpt.exists():
        print(f"No checkpoint: {ckpt}")
        sys.exit(1)

    inc_c, inc_f, inc_d = obs_channel_config_from_checkpoint(str(ckpt), N_STACK)
    print(f"Checkpoint expects: completion={inc_c}, frightened={inc_f}, derived={inc_d}")

    vec = build_vec(inc_c, inc_f, inc_d)
    model = peek_checkpoint(str(ckpt.with_suffix("")), device="cpu")
    try:
        validate_model_env(model, vec, context="validate_training_setup")
        print(f"OK: model={model.observation_space.shape} env={vec.observation_space.shape}")
    except ValueError as e:
        print(f"FAIL: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
