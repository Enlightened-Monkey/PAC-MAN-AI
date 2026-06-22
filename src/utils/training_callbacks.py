"""Callbacks for Pac-Man PPO training (console + MLflow metrics + PPO losses)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environment.pacman_env import PacmanGridEnv
from src.utils.maskable_env import predict_action, wrap_with_action_masker

# SB3 logger keys written after each PPO update.
_TRAIN_LOSS_KEYS = (
    "train/loss",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
)


class ConsoleCallback(BaseCallback):
    """Log rolling episode metrics and PPO losses to stdout and MLflow."""

    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_every: int,
        log_every: int,
        mlflow_logger: Any,
        target_timesteps: int,
        *,
        resume_from: int = 0,
        eval_every: int = 0,
        eval_episodes: int = 4,
        eval_seeds: tuple[int, ...] = (100, 101, 102, 103),
        n_stack: int = 4,
        include_completion_plane: bool = False,
        include_frightened_plane: bool = False,
        include_derived_planes: bool = False,
        arcade_frame_repeat: int = 1,
        eval_arcade_frame_repeat: int = 1,
        milestone_thresholds: tuple[float, ...] | None = None,
        milestone_bonuses: tuple[float, ...] | None = None,
        near_miss_penalty: float = -5.0,
        late_endgame_fail_penalty: float = -1.0,
        ent_coef_floor: float = 0.01,
        ent_coef_start: float = 0.02,
        ent_coef_final: float = 0.012,
        ent_decay_fraction: float = 0.95,
        ent_coef_plateau: float = 0.015,
        ent_plateau_pellet_threshold: float = 0.75,
        use_action_masks: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.mlflow_logger = mlflow_logger
        self.target_timesteps = int(target_timesteps)
        self.resume_from = int(resume_from)
        self.eval_every = int(eval_every)
        self.eval_episodes = int(eval_episodes)
        self.eval_seeds = eval_seeds
        self.n_stack = int(n_stack)
        self.include_completion_plane = bool(include_completion_plane)
        self.include_frightened_plane = bool(include_frightened_plane)
        self.include_derived_planes = bool(include_derived_planes)
        self.arcade_frame_repeat = int(arcade_frame_repeat)
        self.eval_arcade_frame_repeat = int(eval_arcade_frame_repeat)
        self.milestone_thresholds = milestone_thresholds
        self.milestone_bonuses = milestone_bonuses
        self.near_miss_penalty = float(near_miss_penalty)
        self.late_endgame_fail_penalty = float(late_endgame_fail_penalty)
        self.ent_coef_floor = float(ent_coef_floor)
        self.ent_coef_start = float(ent_coef_start)
        self.ent_coef_final = float(ent_coef_final)
        self.ent_decay_fraction = max(1e-6, float(ent_decay_fraction))
        self.ent_coef_plateau = float(ent_coef_plateau)
        self.ent_plateau_pellet_threshold = float(ent_plateau_pellet_threshold)
        self.use_action_masks = bool(use_action_masks)
        self._plateau_log_streak = 0
        self._entropy_boosted = False
        self._next_log = self._align_next(resume_from, log_every)
        self._next_ckpt = self._align_next(resume_from, checkpoint_every)
        self._next_eval = self._align_next(resume_from, eval_every) if eval_every > 0 else 0
        self._ep_rewards: list[float] = []
        self._ep_lengths: list[int] = []
        self._ep_scores: list[float] = []
        self._ep_levels: list[float] = []
        self._ep_pellet_pct: list[float] = []
        self._ep_pellet_levels: list[float] = []
        self._ep_deaths: list[int] = []
        self._ep_max_level: list[int] = []
        self._ep_level_clears: list[int] = []
        self._ep_milestone_rewards: list[float] = []
        self._latest_losses: dict[str, float] = {}
        self._best_score_ever: float = 0.0
        self._best_reward_ever: float = -1e9
        self._best_pellet_ever: float = 0.0
        self._best_pellet_levels_ever: float = 0.0
        self._t0 = time.time()
        self._last_log_time = self._t0

    def _scheduled_ent_coef(self, step: int) -> float:
        span = max(1, self.target_timesteps - self.resume_from)
        progressed = max(0, step - self.resume_from)
        run_progress = min(1.0, progressed / span)
        decay_progress = min(1.0, run_progress / self.ent_decay_fraction)
        value = self.ent_coef_start + (self.ent_coef_final - self.ent_coef_start) * decay_progress
        return max(self.ent_coef_floor, value)

    @staticmethod
    def _align_next(current: int, interval: int) -> int:
        if interval <= 0:
            return current
        return ((current // interval) + 1) * interval

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep = info["episode"]
            self._ep_rewards.append(float(ep["r"]))
            self._ep_lengths.append(int(ep["l"]))
            self._ep_scores.append(float(ep.get("score", info.get("score", 0))))
            self._ep_levels.append(float(ep.get("level", info.get("level", 1))))
            self._ep_pellet_pct.append(
                float(ep.get("pellet_completion", info.get("pellet_completion", 0.0)))
            )
            self._ep_pellet_levels.append(
                float(ep.get("pellet_levels", info.get("pellet_levels", 0.0)))
            )
            self._ep_deaths.append(int(ep.get("episode_deaths", info.get("episode_deaths", 0))))
            self._ep_max_level.append(
                int(ep.get("max_level_reached", info.get("max_level_reached", 1)))
            )
            self._ep_level_clears.append(
                int(ep.get("level_clears", info.get("level_clears", 0)))
            )
            self._ep_milestone_rewards.append(
                float(ep.get("milestone_rewards", info.get("milestone_rewards", 0.0)))
            )
            self._best_score_ever = max(self._best_score_ever, self._ep_scores[-1])
            self._best_reward_ever = max(self._best_reward_ever, self._ep_rewards[-1])
            self._best_pellet_ever = max(self._best_pellet_ever, self._ep_pellet_pct[-1])
            self._best_pellet_levels_ever = max(
                self._best_pellet_levels_ever, self._ep_pellet_levels[-1]
            )

        n = self.num_timesteps
        if n >= self._next_log:
            self._emit_log(n)
            self._next_log = n + self.log_every

        if self.eval_every > 0 and n >= self._next_eval:
            self._run_eval(n)
            self._next_eval = n + self.eval_every

        if n >= self._next_ckpt:
            self.model.save(self.checkpoint_path)
            print(f"\n  [checkpoint] saved -> {self.checkpoint_path}.zip @ {n:,} steps")
            self._next_ckpt = n + self.checkpoint_every

        return True

    def _recent(self, values: list, window: int = 50) -> list:
        return values[-window:] if values else []

    def _harvest_losses(self) -> None:
        """Pull the latest PPO update losses from the SB3 logger buffer."""
        logger = getattr(self.model, "logger", None)
        if logger is None:
            return
        for key in _TRAIN_LOSS_KEYS:
            if key in logger.name_to_value:
                self._latest_losses[key] = float(logger.name_to_value[key])

    def _emit_log(self, n: int) -> None:
        self._harvest_losses()
        now = time.time()
        elapsed = now - self._t0
        interval = max(now - self._last_log_time, 1e-6)
        fps = self.log_every / interval
        self._last_log_time = now

        recent_r = self._recent(self._ep_rewards)
        recent_l = self._recent(self._ep_lengths)
        recent_s = self._recent(self._ep_scores)
        recent_lv = self._recent(self._ep_levels)
        recent_p = self._recent(self._ep_pellet_pct)
        recent_pl = self._recent(self._ep_pellet_levels)
        recent_d = self._recent(self._ep_deaths)
        recent_ml = self._recent(self._ep_max_level)
        recent_lc = self._recent(self._ep_level_clears)
        recent_ms = self._recent(self._ep_milestone_rewards)

        mean_r = float(np.mean(recent_r)) if recent_r else 0.0
        max_r = float(np.max(recent_r)) if recent_r else 0.0
        mean_l = float(np.mean(recent_l)) if recent_l else 0.0
        mean_s = float(np.mean(recent_s)) if recent_s else 0.0
        max_s = float(np.max(recent_s)) if recent_s else 0.0
        mean_lv = float(np.mean(recent_lv)) if recent_lv else 1.0
        mean_p = float(np.mean(recent_p)) if recent_p else 0.0
        mean_pl = float(np.mean(recent_pl)) if recent_pl else 0.0
        max_p = float(np.max(recent_p)) if recent_p else 0.0
        max_pl = float(np.max(recent_pl)) if recent_pl else 0.0
        mean_d = float(np.mean(recent_d)) if recent_d else 0.0
        mean_ml = float(np.mean(recent_ml)) if recent_ml else 1.0
        level_clear_rate = (
            float(np.mean([1.0 if x > 0 else 0.0 for x in recent_lc])) if recent_lc else 0.0
        )
        mean_milestone = float(np.mean(recent_ms)) if recent_ms else 0.0

        self._adjust_entropy(mean_pl, level_clear_rate, n)

        # Arcade score scaled the same way as pellet rewards (÷50) — comparable to return.
        mean_score_scaled = mean_s / 50.0
        bonus_gap = mean_r - mean_score_scaled  # level bonuses, death penalty, step penalty

        metrics: dict[str, float] = {
            "mean_reward_50ep": mean_r,
            "max_reward_50ep": max_r,
            "best_reward_ever": self._best_reward_ever,
            "mean_ep_length_50": mean_l,
            "mean_score_50ep": mean_s,
            "max_score_50ep": max_s,
            "best_score_ever": self._best_score_ever,
            "mean_score_scaled_50ep": mean_score_scaled,
            "reward_score_gap_50ep": bonus_gap,
            "mean_level_50ep": mean_lv,
            "pellet_completion_50ep": mean_p,
            "max_pellet_completion_50ep": max_p,
            "best_pellet_completion_ever": self._best_pellet_ever,
            "pellet_levels_50ep": mean_pl,
            "max_pellet_levels_50ep": max_pl,
            "best_pellet_levels_ever": self._best_pellet_levels_ever,
            "deaths_per_episode_50": mean_d,
            "max_level_50ep": mean_ml,
            "level_clear_rate_50ep": level_clear_rate,
            "mean_milestone_reward_50ep": mean_milestone,
            "ent_coef": float(getattr(self.model, "ent_coef", 0.0)),
            "elapsed_min": elapsed / 60,
            "episodes": len(self._ep_rewards),
            "fps": fps,
        }
        for key, val in self._latest_losses.items():
            mlflow_key = key.replace("train/", "ppo_").replace("/", "_")
            metrics[mlflow_key] = val

        self.mlflow_logger.log_metrics(metrics, step=n)

        loss_parts = []
        for key in ("train/loss", "train/value_loss", "train/entropy_loss"):
            if key in self._latest_losses:
                short = key.split("/")[-1]
                loss_parts.append(f"{short}={self._latest_losses[key]:.4f}")

        sep = "-" * 62  # ASCII-safe for Windows cp1250 console
        print(f"\n{sep}")
        print(
            f"  Pac-Man PPO  |  step {n:>10,} / {self.target_timesteps:,}  "
            f"|  {elapsed/60:5.1f} min  |  {fps:5.0f} fps"
        )
        print(sep)
        print(
            f"  Episodes (50):  return {mean_r:7.2f} (max {max_r:7.2f}  best {self._best_reward_ever:7.2f})"
            f"  |  score {mean_s:7.0f} (max {max_s:7.0f}  best {self._best_score_ever:7.0f})"
            f"  scaled {mean_score_scaled:5.1f}  |  gap {bonus_gap:+6.2f}"
        )
        print(
            f"  Progress:       level {mean_lv:4.2f}  |  max_lvl {mean_ml:4.2f}  "
            f"|  clears {level_clear_rate*100:4.1f}%  "
            f"|  pellet_levels {mean_pl:4.2f} (max {max_pl:4.2f}  best {self._best_pellet_levels_ever:4.2f})"
            f"  |  pellets {mean_p*100:5.1f}%"
        )
        print(
            f"  Survival:       length {mean_l:5.0f}  |  deaths/ep {mean_d:4.2f}  "
            f"|  total eps {len(self._ep_rewards)}"
        )
        ent_coef = float(getattr(self.model, "ent_coef", 0.0))
        print(
            f"  Shaping:        milestone_r {mean_milestone:5.2f}  "
            f"|  ent_coef {ent_coef:.4f}"
            + ("  [plateau boost]" if self._entropy_boosted else "")
        )
        if loss_parts:
            print(f"  PPO loss:       {'  |  '.join(loss_parts)}")
        print(sep)

    def _adjust_entropy(
        self, mean_pellet_levels: float, level_clear_rate: float, step: int
    ) -> None:
        """Raise exploration when stuck close to one level of pellet progress with zero clears."""
        if not isinstance(self.model, (PPO, MaskablePPO)):
            return

        scheduled = self._scheduled_ent_coef(step)
        target_ent = scheduled

        if level_clear_rate > 0.0:
            self._plateau_log_streak = 0
            self._entropy_boosted = False
        elif mean_pellet_levels > self.ent_plateau_pellet_threshold:
            self._plateau_log_streak += 1
        else:
            self._plateau_log_streak = 0

        if self._plateau_log_streak >= 3 and target_ent < self.ent_coef_plateau:
            target_ent = self.ent_coef_plateau
            self._entropy_boosted = True
            print(
                f"\n  [entropy] plateau detected @ {step:,} "
                f"(pellet_levels>{mean_pellet_levels:.2f}, clears=0, "
                f"threshold={self.ent_plateau_pellet_threshold:.0%}) "
                f"-> ent_coef={self.ent_coef_plateau:.4f}"
            )
        elif self._entropy_boosted and self._plateau_log_streak == 0:
            self._entropy_boosted = False
            print(
                f"\n  [entropy] plateau cleared @ {step:,} "
                f"-> ent_coef={scheduled:.4f}"
            )

        current = float(getattr(self.model, "ent_coef", self.ent_coef_floor))
        if abs(current - target_ent) > 1e-9:
            self.model.ent_coef = target_ent

    def _run_eval(self, n: int) -> None:
        if not isinstance(self.model, (PPO, MaskablePPO)):
            return
        seeds = self.eval_seeds[: self.eval_episodes]
        scores, levels, clears, returns = [], [], [], []
        for seed in seeds:
            def _make_eval(s=seed):
                env = PacmanGridEnv(
                    seed=s,
                    max_steps=8000,
                    human_fair=True,
                    arcade_frame_repeat=self.eval_arcade_frame_repeat,
                    include_completion_plane=self.include_completion_plane,
                    include_frightened_plane=self.include_frightened_plane,
                    include_derived_planes=self.include_derived_planes,
                    milestone_thresholds=self.milestone_thresholds,
                    milestone_bonuses=self.milestone_bonuses,
                    near_miss_penalty=self.near_miss_penalty,
                    late_endgame_fail_penalty=self.late_endgame_fail_penalty,
                )
                if self.use_action_masks:
                    env = wrap_with_action_masker(env)
                return Monitor(env)

            venv = DummyVecEnv([_make_eval])
            if self.n_stack > 1:
                venv = VecFrameStack(venv, n_stack=self.n_stack, channels_order="first")
            obs = venv.reset()
            done = False
            ep_return = 0.0
            info: dict = {}
            while not done:
                action, _ = predict_action(
                    self.model, obs, venv, deterministic=True
                )
                obs, rewards, dones, infos = venv.step(action)
                ep_return += float(rewards[0])
                done = bool(dones[0])
                info = infos[0]
            scores.append(float(info.get("score", 0)))
            levels.append(float(info.get("max_level_reached", 1)))
            clears.append(float(info.get("level_clears", 0)))
            returns.append(ep_return)
            venv.close()

        eval_metrics = {
            "eval_mean_score": float(np.mean(scores)),
            "eval_mean_return": float(np.mean(returns)),
            "eval_mean_max_level": float(np.mean(levels)),
            "eval_level_clear_rate": float(np.mean([1.0 if c > 0 else 0.0 for c in clears])),
        }
        self.mlflow_logger.log_metrics(eval_metrics, step=n)
        print(
            f"\n  [eval @ {n:,}] score={eval_metrics['eval_mean_score']:.0f}  "
            f"return={eval_metrics['eval_mean_return']:.1f}  "
            f"max_lvl={eval_metrics['eval_mean_max_level']:.2f}  "
            f"clear_rate={eval_metrics['eval_level_clear_rate']*100:.0f}%"
        )

    def history(self) -> dict[str, list[float]]:
        return {
            "rewards": list(self._ep_rewards),
            "lengths": list(self._ep_lengths),
            "scores": list(self._ep_scores),
            "levels": list(self._ep_levels),
            "pellet_completion": list(self._ep_pellet_pct),
            "pellet_levels": list(self._ep_pellet_levels),
            "deaths": list(self._ep_deaths),
            "max_level": list(self._ep_max_level),
            "level_clears": list(self._ep_level_clears),
            "milestone_rewards": list(self._ep_milestone_rewards),
        }
