"""
rl_agent.py – Reinforcement-learning agent logic.

Wraps Stable-Baselines3 to provide a unified interface for training and
inference of DQN / PPO agents on PacmanEnv.

Classes
-------
RLAgent : Thin wrapper around an SB3 algorithm, handling training,
          evaluation, saving and loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

# Stable-Baselines3 is imported lazily to allow the module to load even when
# the package is not installed (e.g. during documentation builds).
try:
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.base_class import BaseAlgorithm
    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SB3_AVAILABLE = False

SUPPORTED_ALGORITHMS = {"dqn": "DQN", "ppo": "PPO"}


class RLAgent:
    """
    Unified wrapper for Stable-Baselines3 RL algorithms.

    Parameters
    ----------
    algorithm : str
        One of ``"dqn"`` or ``"ppo"`` (case-insensitive).
    env : gym.Env
        The environment to train / evaluate on.
    policy : str
        SB3 policy string, e.g. ``"MlpPolicy"`` or ``"CnnPolicy"``.
    verbose : int
        Verbosity level for SB3 (0 = silent, 1 = info, 2 = debug).
    **kwargs
        Additional keyword arguments forwarded to the SB3 constructor.
    """

    def __init__(
        self,
        algorithm: str,
        env: gym.Env,
        policy: str = "MlpPolicy",
        verbose: int = 0,
        **kwargs: Any,
    ) -> None:
        if not _SB3_AVAILABLE:
            raise ImportError(
                "stable_baselines3 is required. Install it via: "
                "pip install stable-baselines3"
            )
        algo_key = algorithm.lower()
        if algo_key not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from: {list(SUPPORTED_ALGORITHMS.keys())}"
            )
        algo_cls = DQN if algo_key == "dqn" else PPO
        self.model: BaseAlgorithm = algo_cls(
            policy, env, verbose=verbose, **kwargs
        )
        self.algorithm = algo_key

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, total_timesteps: int = 100_000, **kwargs: Any) -> None:
        """
        Run the training loop.

        Parameters
        ----------
        total_timesteps : int
            Total number of environment steps to train for.
        **kwargs
            Additional arguments forwarded to ``model.learn()``.
        """
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> tuple[np.ndarray, Any]:
        """
        Predict an action for the given observation.

        Parameters
        ----------
        observation : np.ndarray
            Current environment observation.
        deterministic : bool
            Whether to use the greedy (deterministic) policy.

        Returns
        -------
        action : np.ndarray
        state  : Any (recurrent state, None for MLP policies)
        """
        return self.model.predict(observation, deterministic=deterministic)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights to *path* (SB3 adds ``.zip`` automatically)."""
        self.model.save(str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        env: gym.Env,
        algorithm: str = "dqn",
    ) -> "RLAgent":
        """
        Load a previously saved model.

        Parameters
        ----------
        path : str or Path
            File path produced by :meth:`save`.
        env : gym.Env
            Environment to attach to the loaded model.
        algorithm : str
            Algorithm used when saving (``"dqn"`` or ``"ppo"``).

        Returns
        -------
        RLAgent
        """
        if not _SB3_AVAILABLE:
            raise ImportError("stable_baselines3 is required.")
        algo_cls = DQN if algorithm.lower() == "dqn" else PPO
        agent = cls.__new__(cls)
        agent.algorithm = algorithm.lower()
        agent.model = algo_cls.load(str(path), env=env)
        return agent

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, env: gym.Env, n_episodes: int = 10) -> dict[str, float]:
        """
        Run *n_episodes* evaluation episodes and return summary stats.

        Parameters
        ----------
        env : gym.Env
        n_episodes : int

        Returns
        -------
        dict with keys ``"mean_reward"``, ``"std_reward"``,
        ``"mean_length"``.
        """
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            length = 0
            while not done:
                action, _ = self.predict(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                length += 1
                done = terminated or truncated
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
        rewards = np.array(episode_rewards)
        return {
            "mean_reward": float(rewards.mean()),
            "std_reward": float(rewards.std()),
            "mean_length": float(np.mean(episode_lengths)),
        }
