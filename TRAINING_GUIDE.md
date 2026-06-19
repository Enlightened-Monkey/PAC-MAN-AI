# RL Training Guide (GPU-Optimized)

## Quick Start (GPU with RTX 3080)

```bash
# Resume training from last checkpoint (safe default, no reset)
python scripts/train_ppo_fair.py \
  --n-envs 16 \
  --n-steps 256 \
  --batch-size 512 \
  --additional-steps 50000 \
  --checkpoint-every 10000 \
  --log-every 5000 \
  --device cuda

# Or if you want to start fresh:
python scripts/train_ppo_fair.py --fresh-start \
  --n-envs 16 \
  --n-steps 256 \
  --batch-size 512 \
  --phase1-steps 1000000 \
  --phase2-steps 4000000 \
  --checkpoint-every 100000 \
  --log-every 10000 \
  --device cuda
```

## Performance Baselines

### GPU (RTX 3080 Laptop)
- **Speed**: ~900-1200 fps (16 parallel envs)
- **Throughput**: ~50k steps in <1 minute
- **Recommended**: Default GPU params above

### CPU (Fallback for weak machines)
- **Speed**: ~200-300 fps (1 env) or ~400-600 fps (4 envs)
- **Throughput**: ~50k steps in ~3-5 minutes
- **Note**: Use `--device cpu --n-envs 1` for minimal resource usage

## Important Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `--device` | auto-detect CUDA | Use `cuda` or `cpu` |
| `--fresh-start` | False | **SAFE**: Resume from checkpoint. Use `--fresh-start` to reset. |
| `--n-envs` | 8 | Parallel environments (increase for GPU, decrease for weak CPU) |
| `--additional-steps` | 2M | Steps to add to current checkpoint (for resume) |
| `--phase1-steps` | 1M | Initial curriculum phase length (fresh start only) |

## Monitoring Training

After each training session, diagnose agent behavior:

```bash
# Quick diagnosis (20 episodes)
python scripts/diagnose_plateau.py --episodes 20

# Full diagnosis (100+ episodes, slow)
python scripts/diagnose_plateau.py --episodes 100
```

Check `/reports/plateau_diagnosis.md` for:
- Pellet completion distribution
- Level clear rate (0% = stuck)
- Spatial heatmaps of agent exploration

## Current Issues & Solutions

**Issue**: Agent stuck at ~10% pellet completion, no level clears
- **Root cause**: Algorithm/reward shaping limitations (not hardware)
- **What we fixed**: 
  - ✅ Missing callback import (was blocking all training)
  - ✅ Missing tensorboard package
  - ✅ Resume logic forcing full phase1 re-training
  - ✅ Default to fresh-start (now safely resumes)
- **What remains**: 
  - Policy optimization may need better curriculum or reward tuning
  - Consider alternative algorithms (DQN, SAC, or offline RL)

## Training Workflow Recommendation

1. **Initial training** (1-2 hours on GPU):
   ```bash
   python scripts/train_ppo_fair.py --n-envs 16 --phase1-steps 2000000 --phase2-steps 2000000
   ```

2. **After each ~500k steps, diagnose**:
   ```bash
   python scripts/diagnose_plateau.py --episodes 50
   ```

3. **If pellet% not improving**:
   - Check `/reports/visit_heatmap.png` for exploration patterns
   - Consider entropy coefficient adjustment in code
   - Try different seeds or network architecture

## Verified Setup

- ✅ PyTorch 2.11.0 + CUDA 13.0
- ✅ stable-baselines3 + sb3-contrib (MaskablePPO)
- ✅ Gymnasium + custom PacmanGridEnv
- ✅ MLflow tracking enabled
- ✅ All dependencies in requirements.txt

## Questions?

Run validation to check setup:
```bash
python scripts/validate_training_setup.py
```

This checks model/environment shape compatibility after resume.
