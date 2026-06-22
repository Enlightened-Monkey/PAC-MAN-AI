#!/usr/bin/env python3
"""Benchmark: run N episodes of Direct-State vs Vision-Pipeline agents.

Prints per-episode scores and saves a side-by-side bar chart with
individual-game dots and ±1σ error bars to reports/benchmark_scores.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _run_episodes(
    agent_type: str,
    n: int,
    seed_base: int,
    vision_model: str,
    device: str,
    max_steps: int,
) -> list[int]:
    from src.apps.dual_agent_lab import AgentMemory, LiveDualAgentLab
    from src.environment.game_logic import GameState
    from src.models.segmentation_detector import SegmentationDetector
    from src.utils.pacman_renderer import render_state_rgb_sprites

    # Shared action-chooser (no display, no recording)
    lab = LiveDualAgentLab(
        seed=seed_base,
        vision_model=vision_model,
        device=device,
        display=False,
        output_path=None,
        duration=0.0,
    )

    scores: list[int] = []
    for ep in range(n):
        seed = seed_base + ep
        state = GameState(seed=seed)
        memory = AgentMemory(level_estimate=state.level)

        for _ in range(max_steps):
            if agent_type == "direct":
                snapshot = lab._snapshot_from_state(state)
                action = lab._choose_action(snapshot, memory)
                lab._update_memory_from_state(memory, snapshot)
            else:  # vision
                frame = render_state_rgb_sprites(state, scale=1)
                mask = lab.detector.predict_mask(frame)
                snapshot = lab._snapshot_from_mask(mask, memory)
                action = lab._choose_action(snapshot, memory)

            memory.last_action = action
            _, done = state.step(action)
            if done:
                break

        scores.append(state.score)
        tag = "direct" if agent_type == "direct" else "vision "
        print(f"  [{tag}] ep={ep + 1:02d}  score={state.score:6d}  steps={state.step_count}")

    return scores


def plot_comparison(
    direct_scores: list[int],
    vision_scores: list[int],
    output_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Direct State", "Vision Pipeline"]
    means = [np.mean(direct_scores), np.mean(vision_scores)]
    stds = [np.std(direct_scores), np.std(vision_scores)]
    x = np.array([0, 1])

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#1a1a2e")

    bar_colors = ["#78ff78", "#ffb450"]
    bars = ax.bar(x, means, width=0.45, color=bar_colors, zorder=2, alpha=0.88)
    ax.errorbar(x, means, yerr=stds, fmt="none", color="white", capsize=8, linewidth=2, zorder=3)

    rng = np.random.default_rng(0)
    for xi, scores in zip(x, [direct_scores, vision_scores]):
        jitter = rng.uniform(-0.12, 0.12, len(scores))
        ax.scatter(xi + jitter, scores, color="white", s=28, zorder=4, alpha=0.75)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + max(means) * 0.015,
            f"avg {mean:.0f}",
            ha="center", va="bottom", color="white", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, color="white")
    ax.set_ylabel("Score", color="white", fontsize=12)
    ax.set_title(
        f"Agent Comparison — {len(direct_scores)} episodes each",
        color="white", fontsize=13, pad=12,
    )
    ax.tick_params(colors="white")
    ax.yaxis.set_tick_params(labelcolor="white")
    ax.spines[:].set_color("#444")
    ax.grid(axis="y", color="#333", linewidth=0.8, zorder=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved chart → {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark direct vs vision agent")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vision-model", default="models/segmentation_unet_long.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="reports/benchmark_scores.png")
    args = ap.parse_args()

    print(f"=== Direct-State agent  ({args.episodes} episodes) ===")
    direct_scores = _run_episodes(
        "direct", args.episodes, args.seed,
        args.vision_model, args.device, args.max_steps,
    )

    print(f"\n=== Vision-Pipeline agent  ({args.episodes} episodes) ===")
    vision_scores = _run_episodes(
        "vision", args.episodes, args.seed,
        args.vision_model, args.device, args.max_steps,
    )

    print("\n--- Summary ---")
    print(f"  Direct  avg={np.mean(direct_scores):.1f}  std={np.std(direct_scores):.1f}  "
          f"min={min(direct_scores)}  max={max(direct_scores)}")
    print(f"  Vision  avg={np.mean(vision_scores):.1f}  std={np.std(vision_scores):.1f}  "
          f"min={min(vision_scores)}  max={max(vision_scores)}")

    plot_comparison(direct_scores, vision_scores, PROJECT_ROOT / args.output)


if __name__ == "__main__":
    main()
