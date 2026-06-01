import os
import sys
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environment.pacman_env import PacmanGridEnv

CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_pacman'))

def main():
    print("Evaluating trained model...")
    if not os.path.exists(CHECKPOINT_PATH + ".zip"):
        print(f"Error: checkpoint {CHECKPOINT_PATH}.zip not found!")
        return

    model = MaskablePPO.load(CHECKPOINT_PATH, device='cpu')
    env = PacmanGridEnv(seed=42)
    
    n_episodes = 20
    scores = []
    levels = []
    steps_list = []
    lives_remaining = []
    reasons = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            action, _ = model.predict(
                obs,
                action_masks=env.action_masks(),
                deterministic=True
            )
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated

        score = env._state.score
        level = env._state.level
        lives = env._state.lives
        
        scores.append(score)
        levels.append(level)
        steps_list.append(step)
        lives_remaining.append(lives)
        
        if lives <= 0:
            reason = "Game Over (Lost all lives)"
        elif step >= 5000:
            reason = "Truncated (Max steps 5000 reached)"
        else:
            reason = "Unknown"
        reasons.append(reason)
        
        print(f"Episode {ep+1:2d} | Score: {score:4d} | Level: {level} | Lives: {lives} | Steps: {step:4d} | Reward: {total_reward:6.2f} | End Reason: {reason}")

    print("\n--- Summary Statistics ---")
    print(f"Mean Score: {np.mean(scores):.1f}")
    print(f"Max Score:  {np.max(scores)}")
    print(f"Mean Level: {np.mean(levels):.2f}")
    print(f"Mean Steps: {np.mean(steps_list):.1f}")
    print(f"Lives Left: {np.mean(lives_remaining):.2f}")
    print(f"Reasons:    {dict(zip(*np.unique(reasons, return_counts=True)))}")

if __name__ == "__main__":
    main()
