#!/usr/bin/env python3
"""Headless fair PPO / MaskablePPO training for Pac-Man."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecFrameStack,
)

from src.environment.pacman_env import PacmanGridEnv
from src.utils.lr_schedule import constant_schedule, linear_schedule
from src.utils.maskable_env import (
    create_maskable_ppo,
    load_trainable_model,
    wrap_with_action_masker,
)
from src.utils.mlflow_logger import MLflowLogger
from src.utils.ppo_cnn import policy_kwargs_for_size
from src.utils.obs_sync import (
    obs_channel_config_from_checkpoint,
    peek_checkpoint,
    validate_model_env,
)
from src.utils.training_callbacks import ConsoleCallback

_EPISODE_INFO_KEYS = (
    "score",
    "level",
    "pellet_completion",
    "pellet_levels",
    "episode_deaths",
    "max_level_reached",
    "level_clears",
    "milestone_rewards",
)


def make_env(
    seed: int,
    max_steps: int,
    *,
    arcade_frame_repeat: int = 1,
    include_completion_plane: bool = False,
    include_frightened_plane: bool = False,
    include_derived_planes: bool = False,
    easy_endgame: bool = False,
    use_action_masks: bool = True,
) -> callable:
    def _f():
        env = PacmanGridEnv(
            seed=seed,
            max_steps=max_steps,
            arcade_frame_repeat=arcade_frame_repeat,
            step_penalty=-0.0005,
            reward_scale_div=50.0,
            death_penalty=-3.0,
            level_bonus=5000.0,
            endgame_death_surcharge=-3.0,
            idle_penalty=-0.02,
            wasted_power_penalty=-1.5,
            near_miss_penalty=NEAR_MISS_PENALTY,
            late_endgame_fail_penalty=LATE_ENDGAME_FAIL_PENALTY,
            milestone_thresholds=MILESTONE_THRESHOLDS,
            milestone_bonuses=MILESTONE_BONUSES,
            enable_milestones=True,
            pbrs_coef=0.0,
            human_fair=True,
            include_completion_plane=include_completion_plane,
            include_frightened_plane=include_frightened_plane,
            include_derived_planes=include_derived_planes,
            easy_endgame=easy_endgame,
        )
        if use_action_masks:
            env = wrap_with_action_masker(env)
        return Monitor(env, info_keywords=_EPISODE_INFO_KEYS)

    return _f


N_STACK = 4
MILESTONE_THRESHOLDS = (0.75, 0.85, 0.92, 0.97)
MILESTONE_BONUSES = (250.0, 500.0, 1000.0, 2000.0)
NEAR_MISS_PENALTY = -12.0
LATE_ENDGAME_FAIL_PENALTY = -4.0


def build_vec_env(
    n_envs: int,
    max_steps: int,
    n_stack: int = N_STACK,
    *,
    arcade_frame_repeat: int = 1,
    include_completion_plane: bool = False,
    include_frightened_plane: bool = False,
    include_derived_planes: bool = False,
    easy_endgame: bool = False,
    use_action_masks: bool = True,
):
    env_fns = [
        make_env(
            seed=i,
            max_steps=max_steps,
            arcade_frame_repeat=arcade_frame_repeat,
            include_completion_plane=include_completion_plane,
            include_frightened_plane=include_frightened_plane,
            include_derived_planes=include_derived_planes,
            easy_endgame=easy_endgame,
            use_action_masks=use_action_masks,
        )
        for i in range(n_envs)
    ]
    if os.name == "nt" or n_envs == 1:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns)
    if n_stack > 1:
        vec = VecFrameStack(vec, n_stack=n_stack, channels_order="first")
    return vec


def _resolve_lr_schedule(mode: str, learning_rate: float):
    if mode == "linear":
        return linear_schedule(learning_rate)
    return constant_schedule(learning_rate)


def train(args: argparse.Namespace) -> None:
    checkpoint_stem_path = Path(args.checkpoint_stem).expanduser()
    if checkpoint_stem_path.suffix == ".zip":
        checkpoint_stem_path = checkpoint_stem_path.with_suffix("")
    if not checkpoint_stem_path.is_absolute():
        checkpoint_stem_path = ROOT / checkpoint_stem_path

    models_dir = checkpoint_stem_path.parent
    models_dir.mkdir(exist_ok=True)
    checkpoint_path = str(checkpoint_stem_path)
    checkpoint_zip = checkpoint_path + ".zip"
    warm_start_path = str(Path(args.warm_start_from).expanduser()) if args.warm_start_from else None
    if warm_start_path and warm_start_path.endswith(".zip"):
        warm_start_path = warm_start_path[: -len(".zip")]
    tb_log = str(ROOT / "logs" / "tensorboard")
    use_maskable = not args.no_maskable
    lr_schedule = _resolve_lr_schedule(args.lr_schedule, args.learning_rate)
    model_policy_kwargs = policy_kwargs_for_size(args.model_size)

    if args.fresh_start and warm_start_path:
        raise ValueError("Use either --fresh-start or --warm-start-from, not both.")

    if args.fresh_start:
        removed = 0
        cp_file = Path(checkpoint_zip)
        if cp_file.exists():
            cp_file.unlink()
            removed += 1

        ckpt_dir = models_dir / "checkpoints"
        if ckpt_dir.exists():
            stem_name = checkpoint_stem_path.name
            for p in ckpt_dir.glob(f"{stem_name}_*_steps.zip"):
                p.unlink()
                removed += 1
        print(f"[Fresh start] Removed {removed} checkpoint(s); training from scratch.")

    resuming = os.path.exists(checkpoint_zip) and not args.fresh_start and not warm_start_path
    warm_starting = bool(warm_start_path)
    include_completion_plane = args.fresh_start
    include_frightened_plane = args.fresh_start
    include_derived_planes = args.fresh_start and (not args.no_derived_planes)
    easy_endgame = args.fresh_start
    warm_source_steps = 0
    if resuming:
        include_completion_plane, include_frightened_plane, include_derived_planes = (
            obs_channel_config_from_checkpoint(
            checkpoint_zip, N_STACK
            )
        )
        easy_endgame = False
        print(
            f"[Resume] synced obs planes: completion={include_completion_plane}, "
            f"frightened={include_frightened_plane}, derived={include_derived_planes}"
        )

    initial_max_steps = 5000
    create_kwargs = dict(
        learning_rate=lr_schedule,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        ent_coef=args.ent_coef_start,
        device=args.device,
        tensorboard_log=tb_log,
        policy_kwargs_override=model_policy_kwargs,
    )

    if resuming:
        peek = peek_checkpoint(checkpoint_path, device=args.device)
        start_steps = int(peek.num_timesteps)
        del peek
        initial_max_steps = 16000 if start_steps >= args.phase1_steps else 5000
        vec_env = build_vec_env(
            args.n_envs,
            max_steps=initial_max_steps,
            arcade_frame_repeat=args.arcade_frame_repeat,
            include_completion_plane=include_completion_plane,
            include_frightened_plane=include_frightened_plane,
            include_derived_planes=include_derived_planes,
            easy_endgame=easy_endgame,
            use_action_masks=use_maskable,
        )
        model = load_trainable_model(
            checkpoint_path,
            vec_env,
            use_maskable=use_maskable,
            device=args.device,
            create_kwargs=create_kwargs,
        )
        print(
            f"[Resume] Loaded checkpoint at {start_steps:,} timesteps "
            f"(maskable={use_maskable})."
        )
        reset_ts = False
    elif warm_starting:
        if not os.path.exists(warm_start_path + ".zip"):
            raise FileNotFoundError(f"Warm-start checkpoint not found: {warm_start_path}.zip")

        include_completion_plane, include_frightened_plane, include_derived_planes = (
            obs_channel_config_from_checkpoint(
            warm_start_path + ".zip", N_STACK
            )
        )
        easy_endgame = False

        vec_env = build_vec_env(
            args.n_envs,
            max_steps=16000,
            arcade_frame_repeat=args.arcade_frame_repeat,
            include_completion_plane=include_completion_plane,
            include_frightened_plane=include_frightened_plane,
            include_derived_planes=include_derived_planes,
            easy_endgame=easy_endgame,
            use_action_masks=use_maskable,
        )

        warm_model = load_trainable_model(
            warm_start_path,
            vec_env,
            use_maskable=use_maskable,
            device=args.device,
            create_kwargs=create_kwargs,
        )
        warm_source_steps = int(warm_model.num_timesteps)
        warm_params = {
            name: value
            for name, value in warm_model.get_parameters().items()
            if "optimizer" not in name
        }

        if use_maskable:
            model = create_maskable_ppo(vec_env, **create_kwargs)
        else:
            model = PPO(
                "CnnPolicy",
                vec_env,
                learning_rate=lr_schedule,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=4,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=args.ent_coef_start,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=model_policy_kwargs,
                verbose=0,
                device=args.device,
                tensorboard_log=tb_log,
            )

        model.set_parameters(warm_params, exact_match=False)
        del warm_model
        start_steps = 0
        reset_ts = True
        algo = "MaskablePPO" if use_maskable else "PPO"
        print(
            f"[Warm start] Initialized fresh {algo} from {warm_start_path}.zip "
            f"weights ({warm_source_steps:,} source timesteps)."
        )
    else:
        start_steps = 0
        vec_env = build_vec_env(
            args.n_envs,
            max_steps=initial_max_steps,
            arcade_frame_repeat=args.arcade_frame_repeat,
            include_completion_plane=include_completion_plane,
            include_frightened_plane=include_frightened_plane,
            include_derived_planes=include_derived_planes,
            easy_endgame=easy_endgame,
            use_action_masks=use_maskable,
        )
        if use_maskable:
            model = create_maskable_ppo(vec_env, **create_kwargs)
        else:
            model = PPO(
                "CnnPolicy",
                vec_env,
                learning_rate=lr_schedule,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                n_epochs=4,
                gamma=0.995,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=args.ent_coef_start,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=model_policy_kwargs,
                verbose=0,
                device=args.device,
                tensorboard_log=tb_log,
            )
        reset_ts = True
        ch = (
            9
            + int(include_completion_plane)
            + int(include_frightened_plane)
            + (6 if include_derived_planes else 0)
        )
        algo = "MaskablePPO" if use_maskable else "PPO"
        print(f"[Initial run] {algo} + PacmanCNN ({ch}ch stacked x{N_STACK}).")

    if resuming:
        target_timesteps = start_steps + args.additional_steps
    elif warm_starting:
        target_timesteps = args.additional_steps
    else:
        target_timesteps = args.phase1_steps + args.phase2_steps

    vec_env = build_vec_env(
        args.n_envs,
        max_steps=initial_max_steps,
        arcade_frame_repeat=args.arcade_frame_repeat,
        include_completion_plane=include_completion_plane,
        include_frightened_plane=include_frightened_plane,
        include_derived_planes=include_derived_planes,
        easy_endgame=easy_endgame,
        use_action_masks=use_maskable,
    )
    model.set_env(vec_env)
    validate_model_env(model, vec_env, context="train_ppo_fair after set_env")

    with MLflowLogger(experiment_name=args.experiment_name, run_name=args.run_name) as logger:
        logger.log_params(
            {
                "algorithm": "MaskablePPO" if use_maskable else "PPO",
                "policy": "CnnPolicy (PacmanCNN)",
                "model_size": args.model_size,
                "n_envs": args.n_envs,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lr_schedule": args.lr_schedule,
                "arcade_frame_repeat": args.arcade_frame_repeat,
                "eval_arcade_frame_repeat": args.eval_arcade_frame_repeat,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef_start": args.ent_coef_start,
                "ent_coef_final": args.ent_coef_final,
                "ent_decay_fraction": args.ent_decay_fraction,
                "ent_coef_plateau": args.ent_coef_plateau,
                "ent_coef_floor": args.ent_coef_floor,
                "ent_plateau_pellet_threshold": args.ent_plateau_pellet_threshold,
                "milestone_thresholds": str(MILESTONE_THRESHOLDS),
                "milestone_bonuses": str(MILESTONE_BONUSES),
                "n_stack": N_STACK,
                "include_completion_plane": include_completion_plane,
                "include_frightened_plane": include_frightened_plane,
                "include_derived_planes": include_derived_planes,
                "death_penalty": -3.0,
                "level_bonus": 5000.0,
                "near_miss_penalty": NEAR_MISS_PENALTY,
                "late_endgame_fail_penalty": LATE_ENDGAME_FAIL_PENALTY,
                "action_masks": use_maskable,
                "phase1_steps": args.phase1_steps,
                "phase2_steps": args.phase2_steps,
                "additional_steps": args.additional_steps,
                "resume_from": start_steps,
                "warm_start": warm_starting,
                "warm_start_from": warm_start_path if warm_starting else "",
                "warm_start_source_steps": warm_source_steps,
                "target_timesteps": target_timesteps,
            }
        )

        callback = ConsoleCallback(
            checkpoint_path,
            args.checkpoint_every,
            args.log_every,
            logger,
            target_timesteps=target_timesteps,
            resume_from=start_steps,
            eval_every=args.eval_every,
            n_stack=N_STACK,
            include_completion_plane=include_completion_plane,
            include_frightened_plane=include_frightened_plane,
            include_derived_planes=include_derived_planes,
            arcade_frame_repeat=args.arcade_frame_repeat,
            eval_arcade_frame_repeat=args.eval_arcade_frame_repeat,
            milestone_thresholds=MILESTONE_THRESHOLDS,
            milestone_bonuses=MILESTONE_BONUSES,
            near_miss_penalty=NEAR_MISS_PENALTY,
            late_endgame_fail_penalty=LATE_ENDGAME_FAIL_PENALTY,
            ent_coef_start=args.ent_coef_start,
            ent_coef_final=args.ent_coef_final,
            ent_decay_fraction=args.ent_decay_fraction,
            ent_coef_plateau=args.ent_coef_plateau,
            ent_coef_floor=args.ent_coef_floor,
            ent_plateau_pellet_threshold=args.ent_plateau_pellet_threshold,
            use_action_masks=use_maskable,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=max(args.checkpoint_every // args.n_envs, 1),
            save_path=str(models_dir / "checkpoints"),
            name_prefix="ppo_pacman",
        )

        # On resumed runs we train only for target_timesteps (additional_steps),
        # instead of forcing a full catch-up to phase1_steps every launch.
        run_phase1 = (
            (not resuming)
            and (not warm_starting)
            and args.phase1_steps > 0
            and model.num_timesteps < args.phase1_steps
        )

        if run_phase1:
            print(f"=== Phase 1: curriculum -> {args.phase1_steps:,} steps (max_steps=5000) ===")
            try:
                model.learn(
                    total_timesteps=args.phase1_steps,
                    callback=[callback, ckpt_cb],
                    reset_num_timesteps=reset_ts,
                    progress_bar=False,
                )
            except KeyboardInterrupt:
                print("\nPhase 1 interrupted.")

        if model.num_timesteps >= args.phase1_steps or args.phase1_steps == 0 or resuming or warm_starting:
            vec_env.close()
            vec_env = build_vec_env(
                args.n_envs,
                max_steps=16000,
                arcade_frame_repeat=args.arcade_frame_repeat,
                include_completion_plane=include_completion_plane,
                include_frightened_plane=include_frightened_plane,
                include_derived_planes=include_derived_planes,
                easy_endgame=easy_endgame,
                use_action_masks=use_maskable,
            )
            model.set_env(vec_env)
            model.ent_coef = max(args.ent_coef_final, float(model.ent_coef))
            if warm_starting:
                print(
                    f"\n=== Warm start run: training until {target_timesteps:,} steps "
                    f"(+{target_timesteps - model.num_timesteps:,} remaining, max_steps=16000) ==="
                )
            else:
                print(
                    f"\n=== Phase 2 / resume: training until {target_timesteps:,} steps "
                    f"(+{target_timesteps - model.num_timesteps:,} remaining, max_steps=16000) ==="
                )

        try:
            remaining = max(0, target_timesteps - int(model.num_timesteps))
            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    callback=[callback, ckpt_cb],
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
        finally:
            model.save(checkpoint_path)
            logger.log_metrics({"total_timesteps": model.num_timesteps})
            print(f"\nFinal model saved -> {checkpoint_zip} ({model.num_timesteps:,} timesteps)")

            hist = callback.history()
            if hist["rewards"]:
                import numpy as np

                recent_lv = hist["levels"][-50:]
                mean_level = float(np.mean(recent_lv)) if recent_lv else 1.0
                print(f"Final mean_level (last 50 ep): {mean_level:.2f}")
                logger.log_metrics({"final_mean_level_50ep": mean_level})


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fair PPO / MaskablePPO Pac-Man agent")
    parser.add_argument("--experiment-name", type=str, default="rl_training")
    parser.add_argument("--run-name", type=str, default="ppo_fair_cnn")
    parser.add_argument(
        "--checkpoint-stem",
        type=str,
        default="models/ppo_pacman",
        help="Checkpoint path without extension, e.g. models/ppo_pacman_xl.",
    )
    parser.add_argument("--model-size", choices=("base", "large", "xl"), default="base")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--arcade-frame-repeat",
        type=int,
        default=4,
        help="How many internal game frames to simulate per RL training decision.",
    )
    parser.add_argument(
        "--eval-arcade-frame-repeat",
        type=int,
        default=1,
        help="How many internal game frames to simulate per evaluation decision.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear"),
        default="constant",
    )
    parser.add_argument("--phase1-steps", type=int, default=1_000_000)
    parser.add_argument("--phase2-steps", type=int, default=4_000_000)
    parser.add_argument("--additional-steps", type=int, default=2_000_000)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--log-every", type=int, default=10_000)
    parser.add_argument("--fresh-start", action="store_true", default=False)
    parser.add_argument("--no-fresh-start", dest="fresh_start", action="store_false")
    parser.add_argument(
        "--warm-start-from",
        type=str,
        default=None,
        help="Path to checkpoint (.zip or stem) used only to initialize network weights.",
    )
    parser.add_argument("--no-maskable", action="store_true")
    parser.add_argument("--eval-every", type=int, default=100_000)
    parser.add_argument("--ent-coef-start", type=float, default=0.02)
    parser.add_argument("--ent-coef-final", type=float, default=0.012)
    parser.add_argument(
        "--ent-decay-fraction",
        type=float,
        default=0.95,
        help="Fraction of total run used for linear decay ent_coef_start -> ent_coef_final.",
    )
    parser.add_argument("--ent-coef-floor", type=float, default=0.01)
    parser.add_argument("--ent-coef-plateau", type=float, default=0.03)
    parser.add_argument("--ent-plateau-pellet-threshold", type=float, default=0.75)
    parser.add_argument(
        "--no-derived-planes",
        action="store_true",
        help="Disable legal engineered observation planes (danger/pellet topology).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for training (e.g. 'cpu' or 'cuda').",
    )
    args = parser.parse_args()
    print(f"Device: {args.device}")
    train(args)


if __name__ == "__main__":
    main()
