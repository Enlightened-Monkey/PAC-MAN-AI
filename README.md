# PAC-MAN-AI

End-to-end Reinforcement Learning project on **Pac-Man**, built around a
custom-made `gymnasium` environment that faithfully reproduces the
mechanics of the original 1980 arcade game (28×31 maze, 4 ghosts with
authentic Target-Tile AI, scatter/chase/frightened wave schedule, ghost
house, tunnel warp, Cruise Elroy, ghost combo scoring).

The project covers MaskablePPO training (sb3-contrib), a CNN frame →
state-vector regressor, MLflow experiment tracking and SHAP /
permutation-based interpretability of the learned policy.

## 1. Problem Statement

Train an agent that can play an arcade-faithful Pac-Man purely from the
state vector exposed by our custom `gymnasium.Env`. We additionally train
a CNN that recovers that same state vector from rendered frames so the
RL policy can in principle be driven by pixels.

## 2. Methods (advanced techniques)

| # | Technique | Where | Course module |
|---|-----------|-------|---------------|
| 1 | **MaskablePPO** (Proximal Policy Optimization with action masking, sb3-contrib) on the custom CNN-channel environment | `notebooks/02_rl_training.ipynb`, `src/models/rl_agent.py` | Reinforcement Learning |
| 2 | **CNN frame → state-vector regressor** (3 conv blocks + MLP head) bridging vision and the symbolic env | `notebooks/03_cnn_training.ipynb`, `src/models/cnn_detector.py` | Deep Learning / CV |
| 3 | **Permutation importance + SHAP** attribution of the MaskablePPO critic (V(s)) | `notebooks/05_interpretability.ipynb` | Interpretability |

## 3. Repository Layout

```
PAC-MAN-AI/
├── notebooks/
│   ├── 01_env_testing.ipynb             # Custom env sanity checks & EDA
│   ├── 02_rl_training.ipynb             # DQN / PPO on the state vector
│   ├── 03_cnn_training.ipynb            # CNN frame → state vector
│   ├── 04_pipeline_integration.ipynb    # Frame → CNN → RL agent (end-to-end)
│   └── 05_interpretability.ipynb        # Permutation / SHAP on the MaskablePPO policy
├── src/
│   ├── environment/
│   │   ├── game_logic.py                # Arcade-faithful game rules + ghost AI
│   │   └── pacman_env.py                # gymnasium.Env wrapper
│   ├── models/
│   │   ├── cnn_detector.py              # CNN architecture + inference
│   │   └── rl_agent.py                  # DQN / PPO wrapper (SB3)
│   └── utils/
│       └── mlflow_logger.py             # MLflow context-manager helper
├── models/                              # Trained checkpoints (git-ignored)
├── tests/
│   └── test_environment.py              # pytest unit tests (30 tests)
├── reports/                             # Presentation PDF, EDA & training figures
├── mlruns/                              # MLflow tracking data (git-ignored)
├── requirements.txt
├── group_project_guidelines.pdf
├── Pac-Man Arcade SI i Mechanika.pdf    # Reference: arcade mechanics
└── README.md
```

## 4. Notebook Map (matches the course-required structure)

| Required section | Implemented in |
|------------------|----------------|
| **Introduction** — problem & motivation | this README §1 + `notebooks/01_env_testing.ipynb` |
| **Data / Environment loading & validation** | `notebooks/01_env_testing.ipynb` |
| **EDA** — observation space, action space, reward distribution | `notebooks/01_env_testing.ipynb` |
| **Feature engineering** — state-vector design, frame preprocessing | `src/environment/pacman_env.py`, `notebooks/03_cnn_training.ipynb` |
| **Modeling** — DQN, PPO, CNN regressor (≥3 techniques) | `notebooks/02_rl_training.ipynb` + `notebooks/03_cnn_training.ipynb` |
| **Evaluation** — episode return curves, rolling averages | `notebooks/02_rl_training.ipynb` |
| **Interpretability** — permutation importance + SHAP | `notebooks/05_interpretability.ipynb` |
| **Conclusions** | this README §8 + `notebooks/04_pipeline_integration.ipynb` |

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

**RL training** — open `notebooks/02_rl_training.ipynb` and run all cells.
Training, MLflow logging and evaluation are produced in-notebook. The
trained checkpoint is saved to `models/ppo_pacman.zip`.

**CNN frame → state-vector** — `notebooks/03_cnn_training.ipynb`.

**End-to-end pipeline** — `notebooks/04_pipeline_integration.ipynb`.

**Interpretability** — `notebooks/05_interpretability.ipynb` loads
`models/ppo_pacman.zip` and produces permutation-importance and
SHAP GradientExplainer figures for the MaskablePPO critic.

**MLflow UI**
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
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
  re-run `notebooks/02_rl_training.ipynb` to regenerate).
- `requirements.txt` pins all direct dependencies; `mlruns/`, `__pycache__/`,
  `.ipynb_checkpoints/`, `*.pth`, `*.zip` and large binary artefacts are
  git-ignored.
