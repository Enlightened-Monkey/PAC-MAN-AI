# Reports

This folder contains evaluation artifacts and the final presentation.

## Contents

| File | Description |
|------|-------------|
| `presentation.pdf` | Final 15–20 min presentation (PDF, exported from slides) |
| `eda_episode_lengths.png` | EDA — episode length distribution |
| `eda_maze_layout.png` | EDA — maze layout visualisation |
| `eda_observation_space.png` | EDA — observation space channels |
| `eda_rewards_actions.png` | EDA — reward & action distributions |
| `training_curves_ppo.png` | MaskablePPO learning curves (reward + ep-length vs steps, poly-2 trend) |
| `interpretability_permutation.png` | Permutation channel importance of the MaskablePPO critic V(s) |
| `interpretability_shap.png` | SHAP GradientExplainer — channel attribution + spatial heatmap |
| `pacman.gif` | Short looping demo of the trained agent |

Artifacts are produced by:
- `notebooks/01_env_testing.ipynb` → `eda_*.png`
- `notebooks/02_rl_training.ipynb` → `training_curves_ppo.png`, `models/ppo_pacman.zip`
- `notebooks/04_pipeline_integration.ipynb` → `pacman.gif`
- `notebooks/05_interpretability.ipynb` → `interpretability_permutation.png`, `interpretability_shap.png`
