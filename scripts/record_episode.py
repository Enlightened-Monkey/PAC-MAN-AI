#!/usr/bin/env python3
"""Record a single Pac-Man episode to MP4 (falls back to GIF if ffmpeg missing)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environment.pacman_env import PacmanGridEnv
from src.utils.maskable_env import load_trainable_model, predict_action, wrap_with_action_masker
from src.utils.obs_sync import include_flags_from_checkpoint, validate_model_env
from src.utils.pacman_renderer import render_state_rgb


def _unwrap_grid_env(venv) -> PacmanGridEnv:
    env = venv.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def record_episode(
    model,
    *,
    seed: int,
    out_path: Path,
    n_stack: int = 4,
    max_steps: int = 8000,
    include_completion_plane: bool = False,
    include_frightened_plane: bool = False,
    use_action_masks: bool = True,
    stochastic: bool = False,
    fps: int = 10,
) -> Path:
    import imageio.v2 as imageio

    base = PacmanGridEnv(
        seed=seed,
        max_steps=max_steps,
        human_fair=True,
        render_mode="rgb_array",
        include_completion_plane=include_completion_plane,
        include_frightened_plane=include_frightened_plane,
    )
    if use_action_masks:
        base = wrap_with_action_masker(base)

    venv = DummyVecEnv([lambda e=base: Monitor(e)])
    if n_stack > 1:
        venv = VecFrameStack(venv, n_stack=n_stack, channels_order="first")
    validate_model_env(model, venv, context=f"record seed={seed}")

    frames: list[np.ndarray] = []
    obs = venv.reset()
    done = False
    while not done:
        grid = _unwrap_grid_env(venv)
        frames.append(render_state_rgb(grid._state, cell_size=16))
        action, _ = predict_action(model, obs, venv, deterministic=not stochastic)
        obs, _, dones, _ = venv.step(action)
        done = bool(dones[0])

    grid = _unwrap_grid_env(venv)
    frames.append(render_state_rgb(grid._state, cell_size=16))
    venv.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(str(out_path), frames, fps=fps)
    except Exception:
        gif_path = out_path.with_suffix(".gif")
        imageio.mimsave(str(gif_path), frames, fps=fps)
        out_path = gif_path
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Pac-Man episode video")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "ppo_pacman.zip"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-maskable", action="store_true")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Missing checkpoint: {ckpt}")
        sys.exit(1)

    use_masks = not args.no_maskable
    inc_c, inc_f = include_flags_from_checkpoint(str(ckpt), args.n_stack)

    def _make():
        e = PacmanGridEnv(
            seed=0, max_steps=args.max_steps, human_fair=True,
            include_completion_plane=inc_c, include_frightened_plane=inc_f,
        )
        if use_masks:
            e = wrap_with_action_masker(e)
        return Monitor(e)

    probe = DummyVecEnv([_make])
    probe = VecFrameStack(probe, n_stack=args.n_stack, channels_order="first")
    model = load_trainable_model(
        str(ckpt.with_suffix("")),
        probe,
        use_maskable=use_masks,
        device="auto",
        create_kwargs=dict(
            learning_rate=1e-4,
            n_steps=512,
            batch_size=512,
            ent_coef=0.02,
            device="auto",
            tensorboard_log=None,
        ),
    )
    probe.close()

    out = Path(args.output) if args.output else ROOT / "reports" / f"episode_seed{args.seed}.mp4"
    path = record_episode(
        model,
        seed=args.seed,
        out_path=out,
        n_stack=args.n_stack,
        max_steps=args.max_steps,
        include_completion_plane=inc_c,
        include_frightened_plane=inc_f,
        use_action_masks=use_masks,
        stochastic=args.stochastic,
    )
    print(f"Saved -> {path} ({path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
