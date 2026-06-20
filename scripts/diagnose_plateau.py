#!/usr/bin/env python3
"""Diagnose level-1 plateau: histograms, spatial heatmaps, optional episode video."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.environment.game_logic import ROWS, COLS, TILE_PELLET, TILE_POWER
from src.environment.pacman_env import PacmanGridEnv
from src.utils.maskable_env import load_trainable_model, predict_action, wrap_with_action_masker
from src.utils.obs_sync import obs_channel_config_from_checkpoint, validate_model_env

_EPISODE_INFO_KEYS = (
    "score",
    "level",
    "pellet_completion",
    "episode_deaths",
    "max_level_reached",
    "level_clears",
    "milestone_rewards",
)


def _unwrap_grid_env(venv) -> PacmanGridEnv:
    env = venv.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def _initial_pellet_tiles() -> set[tuple[int, int]]:
    env = PacmanGridEnv(seed=0)
    env.reset()
    maze = env._state.maze
    tiles: set[tuple[int, int]] = set()
    for r in range(ROWS):
        for c in range(COLS):
            if int(maze[r, c]) in (TILE_PELLET, TILE_POWER):
                tiles.add((r, c))
    env.close()
    return tiles


def _build_probe_env(max_steps, inc_c, inc_f, inc_d, use_masks, n_stack):
    def _make():
        e = PacmanGridEnv(
            seed=0, max_steps=max_steps, human_fair=True,
            include_completion_plane=inc_c, include_frightened_plane=inc_f,
            include_derived_planes=inc_d,
        )
        if use_masks:
            e = wrap_with_action_masker(e)
        return Monitor(e)

    v = DummyVecEnv([_make])
    if n_stack > 1:
        v = VecFrameStack(v, n_stack=n_stack, channels_order="first")
    return v


def run_episodes(
    model,
    n_episodes: int,
    seeds: list[int],
    n_stack: int,
    max_steps: int,
    deterministic: bool,
    include_completion_plane: bool = False,
    include_frightened_plane: bool = False,
    include_derived_planes: bool = False,
    use_action_masks: bool = True,
) -> dict:
    death_completions: list[float] = []
    end_completions: list[float] = []
    death_positions: list[tuple[int, int]] = []
    visit_counts: Counter[tuple[int, int]] = Counter()
    never_visited_pellets: list[set[tuple[int, int]]] = []
    power_pellets_eaten: list[int] = []
    level_clears: list[int] = []
    scores: list[float] = []
    returns: list[float] = []
    pellet_template = _initial_pellet_tiles()

    for seed in seeds[:n_episodes]:
        base_env = PacmanGridEnv(
            seed=seed,
            max_steps=max_steps,
            human_fair=True,
            include_completion_plane=include_completion_plane,
            include_frightened_plane=include_frightened_plane,
            include_derived_planes=include_derived_planes,
        )
        if use_action_masks:
            base_env = wrap_with_action_masker(base_env)

        venv = DummyVecEnv(
            [lambda e=base_env: Monitor(e, info_keywords=_EPISODE_INFO_KEYS)]
        )
        if n_stack > 1:
            venv = VecFrameStack(venv, n_stack=n_stack, channels_order="first")
        validate_model_env(model, venv, context=f"diagnose seed={seed}")

        obs = venv.reset()
        done = False
        ep_return = 0.0
        prev_lives = 3
        prev_score = 0
        power_count = 0
        ep_visits: Counter[tuple[int, int]] = Counter()
        last_info: dict = {}

        while not done:
            action, _ = predict_action(model, obs, venv, deterministic=deterministic)
            obs, rewards, dones, infos = venv.step(action)
            ep_return += float(rewards[0])
            info = infos[0]
            last_info = info

            grid_env = _unwrap_grid_env(venv)
            st = grid_env._state
            lives = st.lives
            completion = st.pellets_eaten / max(st.total_pellets, 1)
            pos = tuple(st.pacman_pos)
            ep_visits[pos] += 1
            visit_counts[pos] += 1

            if lives < prev_lives:
                death_completions.append(completion)
                death_positions.append(pos)
            prev_lives = lives

            score_delta = st.score - prev_score
            if score_delta == 50:
                power_count += 1
            prev_score = st.score

            done = bool(dones[0])

        never_visited_pellets.append(pellet_template - set(ep_visits.keys()))
        end_completions.append(float(last_info.get("pellet_completion", 0.0)))
        power_pellets_eaten.append(power_count)
        level_clears.append(int(last_info.get("level_clears", 0)))
        scores.append(float(last_info.get("score", 0)))
        returns.append(ep_return)
        venv.close()

    return {
        "death_completions": death_completions,
        "end_completions": end_completions,
        "death_positions": death_positions,
        "visit_counts": visit_counts,
        "never_visited_pellets": never_visited_pellets,
        "power_pellets_eaten": power_pellets_eaten,
        "level_clears": level_clears,
        "scores": scores,
        "returns": returns,
    }


def _heatmap(ax, grid: np.ndarray, title: str, cmap: str = "hot") -> None:
    im = ax.imshow(grid, origin="upper", cmap=cmap, aspect="equal")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def write_report(
    data: dict,
    out_md: Path,
    out_png: Path,
    out_visit: Path,
    out_ignored: Path,
    checkpoint: str,
) -> None:
    end = np.asarray(data["end_completions"], dtype=np.float64)
    deaths = np.asarray(data["death_completions"], dtype=np.float64)
    clears = np.asarray(data["level_clears"], dtype=np.int32)
    visit_counts: Counter = data["visit_counts"]
    never_lists: list[set] = data["never_visited_pellets"]

    def pct_above(arr: np.ndarray, t: float) -> float:
        return 100.0 * float(np.mean(arr >= t)) if len(arr) else 0.0

    ignored_counter: Counter[tuple[int, int]] = Counter()
    for s in never_lists:
        ignored_counter.update(s)
    top_ignored = ignored_counter.most_common(10)

    lines = [
        "# Plateau Diagnosis",
        "",
        f"Checkpoint: `{checkpoint}`",
        f"Episodes: {len(end)}",
        "",
        "## End-of-episode pellet completion",
        f"- Mean: {end.mean()*100:.1f}%" if len(end) else "- Mean: n/a",
        f"- Median: {np.median(end)*100:.1f}%" if len(end) else "- Median: n/a",
        f"- Episodes >= 90%: {pct_above(end, 0.90):.1f}%",
        f"- Level clear rate: {pct_above(clears.astype(float), 1.0):.1f}%",
        "",
        "## Death-time pellet completion",
        f"- Deaths recorded: {len(deaths)}",
    ]
    if len(deaths):
        lines += [
            f"- Mean at death: {deaths.mean()*100:.1f}%",
            f"- Deaths when >= 85%: {pct_above(deaths, 0.85):.1f}%",
        ]
    lines += [
        "",
        "## Spatial",
        f"- Pellet tiles never stepped on (mean/ep): "
        f"{np.mean([len(s) for s in never_lists]):.1f}" if never_lists else "- n/a",
    ]
    if top_ignored:
        lines.append("- Most ignored pellet tiles (row,col):")
        for (r, c), cnt in top_ignored:
            lines.append(f"  - ({r},{c}): missed in {cnt} episodes")
    lines += [
        "",
        f"Histograms: `{out_png.name}`",
        f"Visit heatmap: `{out_visit.name}`",
        f"Ignored pellets: `{out_ignored.name}`",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")

    visit_grid = np.zeros((ROWS, COLS), dtype=np.float32)
    death_grid = np.zeros((ROWS, COLS), dtype=np.float32)
    ignored_grid = np.zeros((ROWS, COLS), dtype=np.float32)

    for (r, c), n in visit_counts.items():
        visit_grid[r, c] = n
    for r, c in data["death_positions"]:
        death_grid[r, c] += 1
    for (r, c), n in ignored_counter.items():
        ignored_grid[r, c] = n

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    if len(end):
        axes[0, 0].hist(end * 100, bins=20, range=(0, 100), color="steelblue", edgecolor="white")
    axes[0, 0].set_title("End-of-episode completion (%)")
    axes[0, 0].axvline(90, color="red", linestyle="--", alpha=0.7)

    if len(deaths):
        axes[0, 1].hist(deaths * 100, bins=20, range=(0, 100), color="coral", edgecolor="white")
    axes[0, 1].set_title("Completion at death (%)")
    axes[0, 1].axvline(85, color="red", linestyle="--", alpha=0.7)

    _heatmap(axes[1, 0], visit_grid, "Visit count (all episodes)")
    _heatmap(axes[1, 1], death_grid, "Death locations", cmap="Reds")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    _heatmap(axes2[0], visit_grid, "Visit heatmap")
    _heatmap(axes2[1], ignored_grid, "Ignored pellet tiles", cmap="Blues")
    fig2.tight_layout()
    fig2.savefig(out_visit, dpi=120)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6, 5))
    _heatmap(ax3, ignored_grid, "Pellet tiles never visited")
    fig3.tight_layout()
    fig3.savefig(out_ignored, dpi=120)
    plt.close(fig3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose Pac-Man training plateau")
    parser.add_argument("--checkpoint", default=str(ROOT / "models" / "ppo_pacman.zip"))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=200)
    parser.add_argument("--n-stack", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--no-maskable", action="store_true")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-seed", type=int, default=42)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    use_masks = not args.no_maskable
    inc_c, inc_f, inc_d = obs_channel_config_from_checkpoint(str(ckpt), args.n_stack)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))

    probe = _build_probe_env(args.max_steps, inc_c, inc_f, inc_d, use_masks, args.n_stack)
    print(f"Running {args.episodes} episodes from {ckpt.name} (masks={use_masks})...")
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

    data = run_episodes(
        model,
        args.episodes,
        seeds,
        n_stack=args.n_stack,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        include_completion_plane=inc_c,
        include_frightened_plane=inc_f,
        include_derived_planes=inc_d,
        use_action_masks=use_masks,
    )

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    out_md = reports / "plateau_diagnosis.md"
    out_png = reports / "plateau_diagnosis.png"
    out_visit = reports / "visit_heatmap.png"
    out_ignored = reports / "ignored_pellets.png"
    write_report(data, out_md, out_png, out_visit, out_ignored, str(ckpt))
    print(f"Report  -> {out_md}")
    print(f"Plots   -> {out_png}, {out_visit}, {out_ignored}")

    if args.record_video:
        sys.path.insert(0, str(ROOT / "scripts"))
        from record_episode import record_episode

        video_path = reports / f"episode_seed{args.video_seed}.mp4"
        record_episode(
            model,
            seed=args.video_seed,
            out_path=video_path,
            n_stack=args.n_stack,
            max_steps=args.max_steps,
            include_completion_plane=inc_c,
            include_frightened_plane=inc_f,
            include_derived_planes=inc_d,
            use_action_masks=use_masks,
            stochastic=args.stochastic,
        )
        print(f"Video   -> {video_path}")


if __name__ == "__main__":
    main()
