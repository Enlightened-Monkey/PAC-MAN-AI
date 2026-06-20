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
    include_completion, include_frightened, _ = decode_obs_channel_config(base_channels)
    return include_completion, include_frightened


def decode_obs_channel_config(base_channels: int) -> tuple[bool, bool, bool]:
    """Decode PacmanGridEnv channel config from base (non-stacked) channel count.

    Base layouts supported:
      9  = base only
      10 = base + completion
      11 = base + completion + frightened
      15 = base + derived(6)
      16 = base + completion + derived(6)
      17 = base + completion + frightened + derived(6)
    """
    include_derived = base_channels >= 15
    core = base_channels - 6 if include_derived else base_channels
    if core < 9 or core > 11:
        raise ValueError(
            f"Unsupported base channel count {base_channels} "
            f"(expected one of 9,10,11,15,16,17)"
        )
    include_completion = core >= 10
    include_frightened = core >= 11
    return include_completion, include_frightened, include_derived


def obs_channel_config_from_checkpoint(checkpoint_zip: str, n_stack: int = 4) -> tuple[bool, bool, bool]:
    model = peek_checkpoint(checkpoint_zip, device="cpu")
    stacked = int(model.observation_space.shape[0])
    base = base_channels_from_stacked(stacked, n_stack)
    cfg = decode_obs_channel_config(base)
    del model
    return cfg


def include_flags_from_checkpoint(checkpoint_zip: str, n_stack: int = 4) -> tuple[bool, bool]:
    inc_c, inc_f, _inc_d = obs_channel_config_from_checkpoint(checkpoint_zip, n_stack)
    return inc_c, inc_f


def validate_model_env(model: BaseAlgorithm, vec_env, *, context: str = "") -> None:
    m = tuple(model.observation_space.shape)
    e = tuple(vec_env.observation_space.shape)
    if m != e:
        raise ValueError(
            f"Observation shape mismatch ({context}): model={m} env={e}. "
            "On resume, INCLUDE_* flags must match the checkpoint. "
            "Supported base channels: 9/10/11 (legacy) and 15/16/17 (with derived planes)."
        )
