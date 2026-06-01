import os
import sys
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from torch.utils.data import Dataset, DataLoader
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environment.pacman_env import PacmanGridEnv
from scratch.expert_agent import find_expert_action

CHECKPOINT_PATH = os.path.join('models', 'ppo_pacman')

class ExpertDataset(Dataset):
    def __init__(self, observations, masks, actions):
        self.observations = torch.tensor(observations, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.actions = torch.tensor(actions, dtype=torch.long)
        
    def __len__(self):
        return len(self.actions)
        
    def __getitem__(self, idx):
        return self.observations[idx], self.masks[idx], self.actions[idx]

def collect_expert_data(env, num_steps=30000):
    print(f"Collecting {num_steps} steps of expert demonstrations...")
    obs_list = []
    mask_list = []
    act_list = []
    
    steps_collected = 0
    episodes = 0
    
    while steps_collected < num_steps:
        obs, _ = env.reset(seed=42 + episodes)
        done = False
        
        while not done and steps_collected < num_steps:
            action = find_expert_action(env)
            mask = env.action_masks()
            
            obs_list.append(obs)
            mask_list.append(mask)
            act_list.append(action)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps_collected += 1
            
        episodes += 1
        print(f"Collected {steps_collected}/{num_steps} steps (Episode {episodes}, Level Reached: {env._state.level}, Score: {env._state.score})")
        
    return np.array(obs_list), np.array(mask_list), np.array(act_list)

def main():
    print("=== Behavior Cloning (Imitation Learning) for Pac-Man ===")
    
    # 1. Setup env
    env = PacmanGridEnv(seed=42)
    
    # 2. Collect expert trajectories
    obs_data, mask_data, act_data = collect_expert_data(env, num_steps=25000)
    
    # 3. Load or initialize the MaskablePPO model
    # We must wrap env in DummyVecEnv and ActionMasker to load/save model successfully
    def make_env_fn():
        def _f():
            e = PacmanGridEnv(seed=42)
            e = ActionMasker(e, lambda env: env.action_masks())
            return e
        return _f
    vec_env = DummyVecEnv([make_env_fn()])
    
    checkpoint_zip = CHECKPOINT_PATH + ".zip"
    
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

    policy_kwargs = dict(
        features_extractor_class=PacmanCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )
    
    if os.path.exists(checkpoint_zip):
        print(f"Loading existing model checkpoint from {CHECKPOINT_PATH}...")
        model = MaskablePPO.load(CHECKPOINT_PATH, env=vec_env)
    else:
        print("Creating a new MaskablePPO model from scratch...")
        model = MaskablePPO(
            "CnnPolicy",
            vec_env,
            learning_rate=2.5e-4,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="auto",
        )
        
    device = model.device
    print(f"Training on device: {device}")
    
    # 4. Create PyTorch DataLoader
    dataset = ExpertDataset(obs_data, mask_data, act_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # 5. Supervised Training Loop (Behavior Cloning)
    policy = model.policy
    policy.train()
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4)
    epochs = 15
    
    print(f"Training policy for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        for batch_obs, batch_masks, batch_actions in dataloader:
            # Move to device
            batch_obs = batch_obs.to(device)
            batch_masks = batch_masks.to(device)
            batch_actions = batch_actions.to(device)
            
            # Evaluate actions under the policy network
            # log_prob shape: (batch_size,), entropy shape: (batch_size,)
            log_prob, entropy, value = policy.evaluate_actions(batch_obs, batch_actions, action_masks=batch_masks)
            
            # Loss = Negative Log Likelihood
            loss = -log_prob.mean() - 0.01 * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_actions)
                
        epoch_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1:2d}/{epochs:2d} | Supervised Loss: {epoch_loss:.4f}")
        
    # 6. Save the imitation trained model
    policy.eval()
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    model.save(CHECKPOINT_PATH)
    print(f"\nSUCCESS! Supervised Imitation Policy successfully saved to {CHECKPOINT_PATH}.zip")
    print("You can now run 'python scratch/evaluate_policy.py' to see how the neural network plays!")

if __name__ == "__main__":
    main()
