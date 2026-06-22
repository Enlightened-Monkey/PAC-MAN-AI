# PAC-MAN-AI

End-to-end Reinforcement Learning project on **Pac-Man**, built around a
custom-made `gymnasium` environment that faithfully reproduces the
mechanics of the original 1980 arcade game (28×31 maze, 4 ghosts with
authentic Target-Tile AI, scatter/chase/frightened wave schedule, ghost
house, tunnel warp, Cruise Elroy, ghost combo scoring).

The project covers MaskablePPO training (sb3-contrib), a segmentation
U-Net for pixel-to-state inference, MLflow experiment tracking and SHAP /
permutation-based interpretability of the learned policy.

## 1. Problem Statement

Train an agent that can play an arcade-faithful Pac-Man purely from the
state vector exposed by our custom `gymnasium.Env`. We additionally train
a segmentation U-Net that recovers that same state from rendered frames so
the RL policy can in principle be driven by pixels.

## 2. Methods (advanced techniques)

| # | Technique | Where | Course module |
|---|-----------|-------|---------------|
| 1 | **Random baseline** — uniform action policy, reward / episode-length distribution | `notebooks/pacman_ai_full_workflow.ipynb` §1 | RL fundamentals |
| 2 | **Semantic segmentation U-Net** (TinyU-Net, pixel-wise 13-class, mIoU evaluation) | `notebooks/pacman_ai_full_workflow.ipynb` §4–5, `src/models/segmentation_detector.py` | Deep Learning / CV |
| 3 | **Slot-mask pellet detection** — deterministic brightness threshold (no NN) | `notebooks/pacman_ai_full_workflow.ipynb` §6, `src/models/segmentation_detector.py` | Classical CV |
| 4 | **MaskablePPO** (action-masked PPO, sb3-contrib) with two-phase curriculum | `notebooks/pacman_ai_full_workflow.ipynb` §7, `scripts/train_ppo_fair.py` | Reinforcement Learning |
| 5 | **Permutation importance + SHAP GradientExplainer** on the PPO critic V(s) | `notebooks/pacman_ai_full_workflow.ipynb` §8 | Interpretability |

## 3. Repository Layout

```
PAC-MAN-AI/
├── data/
│   ├── raw/
│   │   └── sprites/                    # Source sprite sheets
│   ├── segmentation/                   # Canonical train/test segmentation datasets
│   ├── labeled_maps/                   # Generated labeled maps with manifest
│   └── pellet_probe/                   # Pellet-detection debug captures
├── docs/
│   ├── PROJECT_STRUCTURE.md            # Conventions & hygiene rules
│   ├── SOURCES.md                      # Academic bibliography (23 entries)
│   └── reference/
│       └── Pac-Man Arcade SI i Mechanika.md
├── notebooks/
│   └── pacman_ai_full_workflow.ipynb   # Single end-to-end notebook (11 sections)
├── scripts/
│   ├── train_ppo_fair.py               # Main PPO training script (GPU/CPU)
│   ├── benchmark_agents.py             # Direct-state vs vision agent benchmark
│   ├── diagnose_plateau.py             # Post-training behaviour diagnosis
│   ├── validate_training_setup.py      # Model/env shape compatibility check
│   ├── analyze_behavior.py             # Agent behaviour analysis
│   ├── audit_level_clear.py            # Level-clear audit
│   ├── plot_training.py                # Generate training curve plots
│   ├── record_episode.py               # Record demo episode GIF/video
│   ├── generate_example_labeled_map.py # Single labeled map preview
│   └── generate_labeled_maps_atlas.py  # Labeled map atlas generation
├── src/
│   ├── apps/
│   │   └── dual_agent_lab.py           # Side-by-side direct-state vs vision demo
│   ├── environment/
│   │   ├── game_logic.py               # Arcade-faithful game rules + ghost AI
│   │   └── pacman_env.py               # gymnasium.Env wrapper (PacmanEnv, PacmanGridEnv)
│   ├── dataset/
│   │   └── pacman_map_dataset.py       # Sprite extraction + synthetic map generation
│   ├── models/
│   │   ├── cnn_detector.py             # CNN architecture + inference
│   │   ├── rl_agent.py                 # PPO wrapper (SB3)
│   │   └── segmentation_detector.py    # TinyU-Net training + layered inference
│   └── utils/
│       ├── mlflow_logger.py            # MLflow context-manager helper
│       ├── device_helper.py            # CUDA/CPU device selection
│       ├── lr_schedule.py              # Learning-rate schedule utilities
│       ├── maskable_env.py             # Action-masking env wrapper
│       ├── obs_sync.py                 # Observation synchronisation helpers
│       ├── pacman_renderer.py          # Frame rendering for CNN pipeline
│       ├── ppo_cnn.py                  # CNN feature extractor for PPO
│       └── training_callbacks.py       # SB3 training callbacks
├── models/
│   ├── ppo_pacman.zip                  # Primary trained RL checkpoint
│   ├── ppo_pacman_xl_strong.zip        # Best-run checkpoint
│   └── segmentation_unet_long.pt       # Trained segmentation model
├── tests/                              # pytest unit tests
├── logs/                               # TensorBoard event files
├── reports/                            # EDA figures, learning curves, diagnostics
├── mlruns/                             # MLflow tracking (SQLite, git-ignored)
├── pacman_screen_agent.py              # Top-level launcher for screen agent
├── TRAINING_GUIDE.md                   # GPU-optimized training quick-start
├── requirements.txt
└── README.md
```

## 4. Notebook Map (matches the course-required structure)

All course-required sections are consolidated in a **single notebook**:
`notebooks/pacman_ai_full_workflow.ipynb`

| Required section | Notebook section |
|------------------|-----------------|
| **Introduction** — problem & motivation | §0 title cell |
| **Data Loading & Validation** — environment, observation space | §1 |
| **EDA** — observation space, action space, reward distribution | §1 |
| **Synthetic dataset generation** — labelled frames, class distribution | §2 |
| **Feature engineering** — multi-channel grid observation design | §3 |
| **Modeling** — baseline, U-Net, slot-mask, MaskablePPO (≥3 techniques) | §4, §6, §7 |
| **Evaluation** — mIoU, visual inspection, benchmark scores | §5, §10 |
| **Interpretability** — permutation importance, SHAP beeswarm, waterfall | §8 |
| **MLOps** — MLflow experiment tracking throughout | §4, §7, §9 |
| **Conclusions** — findings, limitations, future work | §11 |

---

## 5. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

No external ROMs or third-party game binaries are required — the
environment is implemented entirely in `src/environment/`.

## 6. Running

**Tests**
```bash
pytest tests/ -v
```

**Full workflow notebook** — open `notebooks/pacman_ai_full_workflow.ipynb` and run all
cells top-to-bottom. The notebook covers environment EDA, dataset generation, segmentation
training, RL training, interpretability, learning curves, and the agent benchmark. All
results are logged to MLflow automatically.

Trained checkpoints are committed under `models/`:
- `models/ppo_pacman.zip` — primary RL checkpoint
- `models/ppo_pacman_xl_strong.zip` — best-run checkpoint
- `models/segmentation_unet_long.pt` — segmentation model

**RL training only (script)**
```bash
python scripts/train_ppo_fair.py
```

**Segmentation inference** — `notebooks/pacman_ai_full_workflow.ipynb`.

**Screenshot segmentation (labels + positions + class/group layers)**
```bash
# 1) Train on generated images/masks
python -m src.models.segmentation_detector train \
  --dataset-dir data/pacman_dataset \
  --output models/segmentation_unet.pt \
  --epochs 20 \
  --batch-size 16 \
  --device cpu

# 2) Run on a single screenshot
python -m src.models.segmentation_detector infer \
  --model models/segmentation_unet.pt \
  --image data/example_screenshot.png \
  --output-dir data/screenshot_predictions \
  --detect-hud

# 3) Arcade-layout screenshot (auto maze crop + HUD parse)
python -m src.models.segmentation_detector infer-arcade \
  --model models/segmentation_unet.pt \
  --image data/example_arcade_screen.png \
  --output-dir data/screenshot_predictions_arcade \
  --parse-hud
```

The inference command writes:
- `prediction.json` with per-object labels, bounding boxes and centroids,
- `pred_mask.png` (class-id segmentation mask),
- class layers under `layers/classes/` (e.g. `blinky.png`, `fruit.png`),
- group layers under `layers/groups/` (e.g. `ghosts.png`, `collectibles.png`).

The `infer-arcade` command additionally writes:
- `prediction_arcade.json` with boxes mapped to original screenshot coordinates,
- `pred_mask_full.png` projected back to the original screenshot size,
- parsed HUD fields (`score_text`, `high_score_text`, lives icon detections).

**Internal dual-agent lab** — side-by-side direct-state vs vision-from-render
```bash
python -m src.apps.dual_agent_lab \
  --vision-model models/segmentation_unet_long.pt \
  --seed 0 \
  --fps 12
```

This lab uses the internal `GameState` simulation only. The left panel is fed
directly from state, while the right panel renders the game and passes the
frame through the segmentation pipeline before choosing an action. The HUD
shows score, 1UP, lives icons, and cumulative pellet progress in level units.

The previous live screen-capture agent is treated as legacy and is archived in
`archive/legacy_screen_capture/`.

**Synthetic labelled frames / sprite extraction**
```bash
python -m src.dataset.pacman_map_dataset extract-assets \
  --output-dir data/pacman_assets

python -m src.dataset.pacman_map_dataset audit-assets \
  --output-dir data/pacman_asset_audit

python -m src.dataset.pacman_map_dataset generate \
  --output-dir data/pacman_dataset \
  --samples 256 \
  --seed 42 \
  --max-random-steps 300
```

The generator uses the arcade sprite sheet stored in
`data/raw/sprites/Arcade - Pac-Man - Miscellaneous - General Sprites.png`,
extracts a working asset catalog, renders 224x248 frames from `GameState`, and
saves three aligned outputs per sample: RGB frame, class mask, and JSON
annotations with bounding boxes and tile labels.

Exported assets are now grouped by type and animation folders, for example:

- `backgrounds/maze/maze_empty.png`
- `tiles/collectibles/pellet.png`
- `characters/pacman/normal/right/frame_00.png`
- `characters/ghosts/blinky/normal/right/frame_00.png`
- `characters/ghosts/pinky/normal/right/frame_00.png`
- `characters/ghosts/inky/normal/right/frame_00.png`
- `characters/ghosts/clyde/normal/right/frame_00.png`
- `characters/ghosts/shared/frightened/blue/frame_00.png`
- `items/fruits/cherry/frame_00.png`

Each top-level section (`backgrounds`, `tiles`, `characters`, `items`) contains
an `index.json` file with pointers to all subfolders/files for easy loading.

Ghost colors are mapped from dedicated sprite rows (red/pink/cyan/orange), so
Blinky, Pinky, Inky and Clyde no longer share the same source strip.

**End-to-end pipeline** — `notebooks/pacman_ai_full_workflow.ipynb`.

**Interpretability** — `notebooks/pacman_ai_full_workflow.ipynb` §8 loads
`models/ppo_pacman.zip` and produces permutation-importance and
SHAP GradientExplainer figures for the MaskablePPO critic.

**MLflow UI**
```bash
MLFLOW_TRACKING_URI="sqlite:////$(pwd)/mlruns/mlflow.db" mlflow ui --host 127.0.0.1 --port 5000
# → http://localhost:5000
```

## 7. Custom Environment — quick reference

`GameState` (in `src/environment/game_logic.py`) is the authoritative source
of truth and is wrapped by `PacmanEnv` (`gymnasium.Env`). The maze is
**28 columns × 31 rows** (the original arcade playfield). Pac-Man starts at
`(23, 13)` facing LEFT; Blinky starts outside the ghost house at `(11, 13)`,
Pinky/Inky/Clyde start inside.

`GameState.to_observation()` serialises the full state into a flat `float32`
vector of length `_OBS_SIZE = 6 + 8 + 4 + 2 + ROWS*COLS = 888`, all values
in `[0, 1]`:

```
Index range   Content
─────────────────────────────────────────────────────────────────
[0..2)        pacman_row / ROWS, pacman_col / COLS
[2..6)        mode flags: scatter, chase, frightened, spare
[6..14)       ghost_row[i]/ROWS, ghost_col[i]/COLS  (4 ghosts)
[14..18)      per-ghost frightened/eaten flag
[18..19)      lives / 3
[19..20)      remaining_pellets / total_pellets
[20..]        maze cells / 5            (ROWS × COLS values)
```

Ghost behaviour reproduces the *Pac-Man Dossier* algorithms:

- **Per-ghost Target Tile**: Blinky targets Pac-Man (Cruise Elroy
  acceleration when few pellets remain); Pinky targets 4 tiles ahead of
  Pac-Man (with the original *Up overflow* bug); Inky uses Blinky's
  position as a pivot; Clyde chases when far and scatters when close.
- **Wave schedule**: alternating scatter / chase phases per level, with a
  forced 180° reversal whenever the mode toggles.
- **Frightened mode** triggered by power pellets (4 pellets at the
  corners), with the documented slowdown and 200 / 400 / 800 / 1600 ghost
  combo scoring.
- **Ghost house** uses the per-ghost dot counter and a global release
  timeout to control when each ghost leaves.
- **Red zones** above the ghost house and on the upper tunnel forbid the
  Up turn for chasing ghosts.
- **Tunnel** wraps row 14 between columns 0–5 and 22–27 with the
  documented half-speed slowdown.

## 8. Reproducibility & MLOps

- All training scripts seed `numpy`, `random` and `torch` (`RANDOM_SEED = 42`).
- Hyperparameters and per-episode metrics are logged to **MLflow** (see
  `src/utils/mlflow_logger.py`).
- Trained checkpoints are saved under `models/` (git-ignored due to size;
  re-run `notebooks/rl_env/02_rl_training.ipynb` to regenerate).
- `requirements.txt` pins all direct dependencies; `mlruns/`, `__pycache__/`,
  `.ipynb_checkpoints/`, `*.pth`, `*.zip` and large binary artefacts are
  git-ignored.
