import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environment.pacman_env import PacmanGridEnv

class ConsoleCallback(BaseCallback):
    def __init__(self, log_every=10000):
        super().__init__()
        self.log_every = log_every
        self._next_log = log_every
        self._ep_rewards = []
        self._ep_lengths = []
        self._ep_scores = []
        self._ep_levels = []
        
    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._ep_rewards.append(info['episode']['r'])
                self._ep_lengths.append(info['episode']['l'])
                self._ep_scores.append(info.get('score', 0))
                self._ep_levels.append(info.get('level', 1))
                
        n = self.num_timesteps
        if n >= self._next_log:
            recent_r = self._ep_rewards[-50:]
            recent_l = self._ep_lengths[-50:]
            recent_s = self._ep_scores[-50:]
            recent_lv = self._ep_levels[-50:]
            
            mean_r = float(np.mean(recent_r)) if recent_r else 0.0
            mean_l = float(np.mean(recent_l)) if recent_l else 0.0
            mean_s = float(np.mean(recent_s)) if recent_s else 0.0
            mean_lv = float(np.mean(recent_lv)) if recent_lv else 1.0
            
            print(f"Steps: {n:>7,} | mean_r: {mean_r:7.2f} | mean_len: {mean_l:5.0f} | mean_score: {mean_s:6.1f} | mean_level: {mean_lv:4.2f} | eps: {len(self._ep_rewards)}")
            self._next_log = n + self.log_every
        return True

def mask_fn(env):
    return env.action_masks()

def make_env(seed: int):
    def _f():
        env = PacmanGridEnv(
            seed=seed,
            max_steps=2000, # slightly shorter episodes for faster early training
            step_penalty=-0.01,
            reward_scale_div=100.0,
            pbrs_coef=0.2,
        )
        env = ActionMasker(env, mask_fn)
        return env
    return _f

class PacmanCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, c, h, w)).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flat, features_dim), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))

def main():
    print("Starting Pacman training FROM SCRATCH for 150,000 steps...")
    N_ENVS = 8
    env_fns = [make_env(seed=i) for i in range(N_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)
    
    policy_kwargs = dict(
        features_extractor_class=PacmanCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    
    model = MaskablePPO(
        "CnnPolicy",
        vec_env,
        learning_rate=3e-4, # slightly higher learning rate for scratch training
        n_steps=256,
        batch_size=512,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2, # slightly larger clip range for faster early exploration
        ent_coef=0.02,  # slightly higher entropy coefficient to promote exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device="auto",
    )
    
    callback = ConsoleCallback(log_every=10000)
    model.learn(total_timesteps=150000, callback=callback, progress_bar=False)
    print("Scratch training test completed successfully!")

if __name__ == "__main__":
    main()
