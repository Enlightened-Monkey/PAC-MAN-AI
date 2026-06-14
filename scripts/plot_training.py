#!/usr/bin/env python3
"""Generate training learning curves, PPO loss plots, and demo GIF."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


def rolling_mean(values: list[float], window: int = 50) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return np.convolve(arr, np.ones(len(arr)) / len(arr), mode="valid")
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def write_overfitting_report(run, client) -> None:
    """Summarise generalisation risk from train vs held-out eval metrics."""
    train_score = client.get_metric_history(run.info.run_id, "mean_score_50ep")
    eval_score = client.get_metric_history(run.info.run_id, "eval_mean_score")
    train_level = client.get_metric_history(run.info.run_id, "max_level_50ep")
    eval_level = client.get_metric_history(run.info.run_id, "eval_mean_max_level")
    ppo_loss = client.get_metric_history(run.info.run_id, "ppo_loss")

    lines = [
        "# Pac-Man PPO — Overfitting & Generalisation Notes",
        "",
        "## Environment characteristics",
        "- Single fixed maze layout (geometry identical each level); difficulty rises via ghost AI.",
        "- Episodes draw fresh ghost RNG every reset (constructor seed applies to episode 1 only).",
        "- Held-out eval uses seeds 100-103, never seen in training.",
        "- 'Overfitting' here = policy memorising training trajectories, not dataset labels.",
        "",
        "## Risk indicators",
    ]

    if train_score and eval_score:
        ts = train_score[-1].value
        es = eval_score[-1].value
        gap = ts - es
        lines.append(f"- Train score (50-ep): **{ts:.0f}** vs eval score: **{es:.0f}** (gap {gap:+.0f})")
        if gap > 200:
            lines.append("  - WARNING: large train/eval score gap - possible memorisation; raise entropy or diversify seeds.")
        else:
            lines.append("  - OK: score gap modest - generalisation looks reasonable.")

    if train_level and eval_level:
        tl = train_level[-1].value
        el = eval_level[-1].value
        lines.append(f"- Train max level: **{tl:.2f}** vs eval max level: **{el:.2f}**")
        if tl >= 1.5 and el < 1.2:
            lines.append("  - WARNING: level clears on train but not eval - policy may exploit training quirks.")

    if ppo_loss and len(ppo_loss) > 5:
        early = np.mean([m.value for m in ppo_loss[:5]])
        late = np.mean([m.value for m in ppo_loss[-5:]])
        lines.append(f"- PPO total loss: early avg **{early:.4f}** -> late avg **{late:.4f}**")
        if late < early * 0.3:
            lines.append("  - Loss collapsed - watch for premature convergence / low exploration.")

    lines += [
        "",
        "## Mitigations in this project",
        "- Fresh episode RNG every reset (no fixed-replay memorisation).",
        "- Entropy coefficient 0.02 -> 0.005 across curriculum phases.",
        "- Linear LR decay 2.5e-4 -> 0.",
        "- Held-out eval every 100k steps (seeds 100-103).",
        "- No action masks / PBRS (human-fair observation only).",
        "- Frame stacking (4) so the CNN sees ghost motion like a human player.",
    ]

    out = REPORTS / "overfitting_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved -> {out}")


def plot_from_mlflow(tracking_uri: str, run_name_filter: str = "ppo_fair") -> None:
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name("rl_training")
    if exp is None:
        print("No rl_training experiment found.")
        return

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    runs = [r for r in runs if run_name_filter in r.data.tags.get("mlflow.runName", "").lower()]
    if not runs:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=5)
    if not runs:
        print("No MLflow runs found.")
        return

    run = runs[0]
    write_overfitting_report(run, client)

    metrics_map = {
        "mean_reward_50ep": "Episode Return (50-ep)",
        "mean_score_50ep": "Arcade Score (50-ep)",
        "reward_score_gap_50ep": "Return - Score/50 Gap",
        "max_level_50ep": "Max Level (50-ep)",
        "level_clear_rate_50ep": "Level Clear Rate (50-ep)",
        "pellet_completion_50ep": "Pellet % Current Level (50-ep)",
    }

    loss_map = {
        "ppo_loss": "PPO Total Loss",
        "ppo_value_loss": "Value Loss",
        "ppo_policy_gradient_loss": "Policy Gradient Loss",
        "ppo_entropy_loss": "Entropy Loss",
        "ppo_explained_variance": "Explained Variance",
    }

    eval_map = {
        "eval_mean_score": "Eval Score (held-out seeds)",
        "eval_mean_max_level": "Eval Max Level",
        "eval_level_clear_rate": "Eval Level Clear Rate",
    }

    sns.set_style("whitegrid")

    # Main dashboard 2x3
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (metric_key, title) in zip(axes.flatten(), metrics_map.items()):
        history = client.get_metric_history(run.info.run_id, metric_key)
        if history:
            steps = [m.step for m in history]
            values = [m.value for m in history]
            ax.plot(steps, values, color="#2563eb", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
    fig.suptitle("Pac-Man PPO Training Dashboard", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = REPORTS / "training_dashboard.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")

    # Loss dashboard 2x3
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (metric_key, title) in zip(axes.flatten(), loss_map.items()):
        history = client.get_metric_history(run.info.run_id, metric_key)
        if history:
            steps = [m.step for m in history]
            values = [m.value for m in history]
            ax.plot(steps, values, color="#dc2626", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("Timesteps")
    # Train vs eval score in last panel
    ax = axes[1, 2]
    for key, color, label in [
        ("mean_score_50ep", "#2563eb", "train score"),
        ("eval_mean_score", "#16a34a", "eval score"),
    ]:
        history = client.get_metric_history(run.info.run_id, key)
        if history:
            ax.plot(
                [m.step for m in history],
                [m.value for m in history],
                color=color,
                linewidth=1.4,
                label=label,
            )
    ax.set_title("Train vs Eval Score")
    ax.set_xlabel("Timesteps")
    ax.legend(fontsize=8)
    fig.suptitle("PPO Loss & Generalisation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = REPORTS / "training_loss_dashboard.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved -> {out}")

    # Individual curves
    all_metrics = {**metrics_map, **loss_map, **eval_map}
    for metric_key in all_metrics:
        history = client.get_metric_history(run.info.run_id, metric_key)
        if not history:
            continue
        steps = [m.step for m in history]
        values = [m.value for m in history]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, values, color="#2563eb", linewidth=1.6)
        ax.set_title(metric_key.replace("_", " ").title())
        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Value")
        fig.tight_layout()
        path = REPORTS / f"learning_curve_{metric_key}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved -> {path}")


def record_demo_gif(checkpoint: Path, n_episodes: int = 1, max_steps: int = 800) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio not installed - skipping GIF.")
        return

    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    from src.environment.pacman_env import PacmanGridEnv
    from src.utils.maskable_env import load_trainable_model, predict_action, wrap_with_action_masker
    from src.utils.obs_sync import include_flags_from_checkpoint, validate_model_env
    from src.utils.pacman_renderer import render_state_rgb, render_hud_text

    if not checkpoint.exists():
        print(f"No checkpoint at {checkpoint}")
        return

    inc_c, inc_f = include_flags_from_checkpoint(str(checkpoint), n_stack=4)

    def _probe():
        e = PacmanGridEnv(seed=0, max_steps=4000, human_fair=True,
            include_completion_plane=inc_c, include_frightened_plane=inc_f)
        return Monitor(wrap_with_action_masker(e))

    probe = VecFrameStack(DummyVecEnv([_probe]), n_stack=4, channels_order="first")
    model = load_trainable_model(
        str(checkpoint.with_suffix("")), probe, use_maskable=True, device="cpu",
        create_kwargs=dict(
            learning_rate=1e-4, n_steps=512, batch_size=512,
            ent_coef=0.02, device="cpu", tensorboard_log=None,
        ),
    )
    probe.close()
    frames: list[np.ndarray] = []

    for ep in range(n_episodes):
        raw_env = PacmanGridEnv(
            seed=ep, human_fair=True, render_mode="rgb_array",
            include_completion_plane=inc_c, include_frightened_plane=inc_f,
        )
        raw_env = wrap_with_action_masker(raw_env)
        venv = VecFrameStack(
            DummyVecEnv([lambda e=raw_env: Monitor(e)]), n_stack=4, channels_order="first"
        )
        validate_model_env(model, venv, context=f"plot_training ep {ep}")
        obs = venv.reset()
        done = False
        steps = 0
        info: dict = {}
        while not done and steps < max_steps:
            action, _ = predict_action(model, obs, venv, deterministic=True)
            obs, _, dones, infos = venv.step(action)
            done = bool(dones[0])
            info = infos[0]
            steps += 1
            frame = render_state_rgb(raw_env._state)
            fig, ax = plt.subplots(figsize=(6, 7), facecolor="black")
            ax.imshow(frame, interpolation="nearest")
            ax.axis("off")
            ax.set_title(
                render_hud_text(raw_env._state, info),
                color="white",
                fontsize=10,
                pad=8,
            )
            fig.tight_layout()
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf[:, :, :3].copy())
            plt.close(fig)
        venv.close()

    if frames:
        gif_path = REPORTS / "demo_episode.gif"
        imageio.mimsave(gif_path, frames, duration=0.12)
        print(f"Saved -> {gif_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlflow-uri",
        default=str((ROOT / "mlflow.db").resolve()),
    )
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "ppo_pacman.zip"))
    parser.add_argument("--skip-gif", action="store_true")
    args = parser.parse_args()

    uri = args.mlflow_uri
    if not uri.startswith("sqlite:"):
        uri = "sqlite:///" + uri.replace("\\", "/")
    plot_from_mlflow(uri)
    if not args.skip_gif:
        record_demo_gif(Path(args.checkpoint))


if __name__ == "__main__":
    main()
