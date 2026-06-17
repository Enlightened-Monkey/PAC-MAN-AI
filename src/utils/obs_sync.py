"""Observation-space sync between PPO / MaskablePPO checkpoints and PacmanGridEnv."""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm


def peek_checkpoint(path: str, device: str = "cpu") -> BaseAlgorithm:
    """Load a checkpoint without env (metadata only). Supports MaskablePPO and PPO."""
    from sb3_contrib import MaskablePPO

    stem = path.replace(".zip", "")
    try:
        return MaskablePPO.load(stem, device=device)
    except (ValueError, KeyError, RuntimeError, TypeError, OSError):
        return PPO.load(stem, device=device)


def base_channels_from_stacked(stacked_channels: int, n_stack: int = 4) -> int:
    if stacked_channels % n_stack != 0:
        raise ValueError(f"stacked channels {stacked_channels} not divisible by n_stack={n_stack}")
    return stacked_channels // n_stack


def include_flags_from_base(base_channels: int) -> tuple[bool, bool]:
    if base_channels < 9 or base_channels > 11:
        raise ValueError(f"Unsupported base channel count {base_channels} (expected 9, 10, or 11)")
    return base_channels >= 10, base_channels >= 11


def include_flags_from_checkpoint(checkpoint_zip: str, n_stack: int = 4) -> tuple[bool, bool]:
    model = peek_checkpoint(checkpoint_zip, device="cpu")
    stacked = int(model.observation_space.shape[0])
    base = base_channels_from_stacked(stacked, n_stack)
    inc_c, inc_f = include_flags_from_base(base)
    del model
    return inc_c, inc_f


def validate_model_env(model: BaseAlgorithm, vec_env, *, context: str = "") -> None:
    m = tuple(model.observation_space.shape)
    e = tuple(vec_env.observation_space.shape)
    if m != e:
        raise ValueError(
            f"Observation shape mismatch ({context}): model={m} env={e}. "
            "On resume, INCLUDE_* flags must match the checkpoint (9ch=36 stacked, "
            "11ch=44 stacked). Use FRESH_START=True for a new 11ch run, or resume with "
            "INCLUDE_COMPLETION_PLANE=False and INCLUDE_FRIGHTENED_PLANE=False for old 36ch ckpt."
        )
