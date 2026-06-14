"""MaskablePPO helpers: ActionMasker wrapper and PPO→MaskablePPO weight transfer."""

from __future__ import annotations

from typing import Any

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm


def action_mask_fn(env) -> Any:
    """Return legal-action mask for ActionMasker / MaskablePPO."""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "action_masks"):
        return unwrapped.action_masks()
    return unwrapped.get_wrapper_attr("action_masks")()


def wrap_with_action_masker(env):
    """Wrap a PacmanGridEnv with sb3-contrib ActionMasker."""
    return ActionMasker(env, action_mask_fn)


def create_maskable_ppo(
    vec_env,
    *,
    learning_rate,
    n_steps: int,
    batch_size: int,
    ent_coef: float,
    device: str,
    tensorboard_log: str | None,
) -> MaskablePPO:
    from src.utils.ppo_cnn import policy_kwargs

    return MaskablePPO(
        "CnnPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs(),
        verbose=0,
        device=device,
        tensorboard_log=tensorboard_log,
    )


def load_trainable_model(
    checkpoint_stem: str,
    vec_env,
    *,
    use_maskable: bool,
    device: str,
    create_kwargs: dict[str, Any] | None = None,
) -> BaseAlgorithm:
    """
    Load a checkpoint for training or eval.

    MaskablePPO checkpoints load directly. Plain PPO checkpoints transfer
    policy weights into a new MaskablePPO instance (same CNN architecture).
    """
    create_kwargs = create_kwargs or {}
    if use_maskable:
        try:
            return MaskablePPO.load(checkpoint_stem, env=vec_env, device=device)
        except (ValueError, KeyError, RuntimeError, TypeError):
            pass
        ppo = PPO.load(checkpoint_stem, device=device)
        merged = dict(create_kwargs)
        merged["device"] = device
        model = create_maskable_ppo(vec_env, **merged)
        model.policy.load_state_dict(ppo.policy.state_dict())
        model.num_timesteps = int(ppo.num_timesteps)
        model.ent_coef = float(getattr(ppo, "ent_coef", create_kwargs.get("ent_coef", 0.02)))
        print(
            f"[MaskablePPO] Transferred policy weights from PPO checkpoint "
            f"@ {model.num_timesteps:,} steps"
        )
        return model

    return PPO.load(checkpoint_stem, env=vec_env, device=device)


def predict_action(model, obs, venv, *, deterministic: bool = True):
    """Predict with action masks when using MaskablePPO."""
    if isinstance(model, MaskablePPO):
        from sb3_contrib.common.maskable.utils import get_action_masks

        masks = get_action_masks(venv)
        return model.predict(obs, deterministic=deterministic, action_masks=masks)
    return model.predict(obs, deterministic=deterministic)
