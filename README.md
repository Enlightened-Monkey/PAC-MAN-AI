# PAC-MAN-AI

A machine-learning project that recreates a Pac-Man environment from scratch, trains a Reinforcement-Learning agent on it, and then adds a CNN-based vision layer so the agent can operate on raw pixel frames.

---

## Repository Layout

```
PAC-MAN-AI/
├── data/
│   ├── raw/             # Original, unprocessed game recordings / screenshots
│   └── processed/       # Pre-processed datasets ready for model training
├── notebooks/
│   ├── 01_env_testing.ipynb          # Environment sanity checks & random rollouts
│   ├── 02_rl_training.ipynb          # DQN / PPO training on the state vector
│   ├── 03_cnn_training.ipynb         # CNN detector training on game frames
│   ├── 04_pipeline_integration.ipynb # End-to-end: CNN → RL agent
│   └── 05_interpretability.ipynb     # SHAP / LIME feature-importance analysis
├── src/
│   ├── environment/
│   │   ├── game_logic.py   # Game rules, ghost AI, scoring
│   │   └── pacman_env.py   # gymnasium.Env wrapper
│   ├── models/
│   │   ├── cnn_detector.py # CNN architecture + inference helpers
│   │   └── rl_agent.py     # DQN / PPO wrapper (Stable-Baselines3)
│   └── utils/
│       └── mlflow_logger.py # MLflow experiment-tracking helpers
├── tests/
│   └── test_environment.py  # pytest unit tests for game logic & environment
├── mlruns/                  # MLflow run artefacts (git-ignored)
├── reports/                 # Evaluation reports and plots
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Environment Architecture

### `game_logic.py` – Internal state

`GameState` is the authoritative source of truth.  It owns:

| Field | Type | Description |
|-------|------|-------------|
| `maze` | `np.ndarray (ROWS, COLS)` | Cell values: 0=path, 1=wall, 2=pellet, 3=power-pellet |
| `pacman_pos` | `(row, col)` | Current Pac-Man grid position |
| `pacman_dir` | `int` | Last successful movement direction |
| `ghosts` | `list[Ghost]` | 4 ghosts with individual AI and frightened state |
| `score` | `int` | Cumulative score |
| `lives` | `int` | Remaining lives (starts at 3) |
| `scatter_mode` | `bool` | Ghost mode: scatter (corner targets) vs. chase (Pac-Man) |

Ghost movement follows the GBA/Z80 algorithm:  
- **Scatter** – each ghost heads toward its fixed corner target.  
- **Chase** – Blinky targets Pac-Man directly; Pinky, Inky and Clyde use
  offset/flanking targets.  
- **Frightened** – random walk for `FRIGHTENED_DURATION` steps after a
  power-pellet is collected.

### Mapping internal state → observation vector (`pacman_env.py`)

`GameState.to_observation()` serialises the full game state into a
**flat `float32` vector** of length `_OBS_SIZE ≈ 460`:

```
Index range   Content
─────────────────────────────────────────────────────────────────
[0]           pacman_row / ROWS           (normalised position)
[1]           pacman_col / COLS
[2..5]        scatter_mode flag           (repeated 4×)
[6..13]       ghost_row[i]/ROWS, ghost_col[i]/COLS  (4 ghosts)
[14..17]      ghost_frightened[i]         (0.0 / 1.0)
[18]          lives / 3
[19]          remaining_pellets / total_pellets
[20..]        maze cells / 3              (ROWS × COLS values)
```

All values are in **[0, 1]** and match `PacmanEnv.observation_space`
(`gymnasium.spaces.Box`).

---

## Running Tests

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run the full test suite:

```bash
pytest tests/ -v
```

Run only the environment tests:

```bash
pytest tests/test_environment.py -v
```

Test coverage includes:
- Maze shape and initial state validation  
- Pac-Man movement (wall blocking, valid moves)  
- Pellet and power-pellet scoring  
- Ghost frightened state and timer  
- Observation vector shape, dtype and value range  
- Terminal condition detection  
- `PacmanEnv` gymnasium interface (reset, step, render)  

---

## Two-Stage Training Pipeline

### Stage 1 – RL training on the ground-truth state vector

```
PacmanEnv.step(action)
        │
        ▼
GameState.to_observation()  ──►  state vector (≈460 floats)
        │
        ▼
   RL agent (DQN / PPO)      ──►  action  ──►  reward
```

The agent receives the exact internal state vector from `GameState`.
This stage is covered in `notebooks/02_rl_training.ipynb`.

**Training command (script equivalent):**

```python
from src.environment.pacman_env import PacmanEnv
from src.models.rl_agent import RLAgent

agent = RLAgent("dqn", PacmanEnv(), verbose=1)
agent.train(total_timesteps=500_000)
agent.save("models/dqn_pacman")
```

### Stage 2 – RL inference driven by CNN state estimation

```
Game screen (RGB frame)
        │
        ▼
 ObjectDetector.predict()   ──►  estimated state vector (≈460 floats)
        │
        ▼
   Pre-trained RL agent      ──►  action
```

The CNN (`src/models/cnn_detector.py`) is trained to reproduce the
same observation vector that `GameState.to_observation()` emits.
After training, the RL agent weights are **frozen**; only the CNN is
updated during Stage 2.  This stage is covered in
`notebooks/03_cnn_training.ipynb` and `notebooks/04_pipeline_integration.ipynb`.

---

## Experiment Tracking

All training runs are logged with MLflow:

```python
from src.utils.mlflow_logger import MLflowLogger

with MLflowLogger(experiment_name="rl_training") as logger:
    logger.log_params({"algorithm": "DQN", "lr": 1e-4})
    logger.log_metric("episode_reward", reward, step=step)
    logger.log_artifact("models/dqn_pacman.zip")
```

View the UI locally:

```bash
mlflow ui --backend-store-uri mlruns/
```
