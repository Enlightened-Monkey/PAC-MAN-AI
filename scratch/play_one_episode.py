import os
import sys
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environment.pacman_env import PacmanGridEnv

CHECKPOINT_PATH = os.path.join('models', 'ppo_pacman')

def main():
    if not os.path.exists(CHECKPOINT_PATH + ".zip"):
        print(f"Error: checkpoint {CHECKPOINT_PATH}.zip not found!")
        return

    model = MaskablePPO.load(CHECKPOINT_PATH)
    env = PacmanGridEnv(seed=45) # Seed 45 corresponds to Episode 4 (42 + 3)
    
    obs, _ = env.reset(seed=45)
    done = False
    step = 0
    
    last_pellets = None
    
    history = []
    
    while not done:
        action, _ = model.predict(
            obs,
            action_masks=env.action_masks(),
            deterministic=True
        )
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated
        
        # Calculate remaining pellets
        maze = env._state.maze
        remaining = int(np.sum(np.isin(maze, [2, 3])))
        
        if last_pellets is None or remaining != last_pellets:
            print(f"Step {step:4d} | Score: {env._state.score:4d} | Remaining Pellets: {remaining:3d} | Pacman Pos: {env._state.pacman_pos} | Lives: {env._state.lives}")
            last_pellets = remaining
            
        history.append((step, env._state.pacman_pos, remaining))

    print(f"\nEpisode finished. Total Steps: {step} | Final Score: {env._state.score} | Level: {env._state.level} | Lives: {env._state.lives}")
    
    print("\n--- Analysing Last 30 Steps to detect loops ---")
    for s, pos, rem in history[-30:]:
        print(f"Step {s:4d} | Pacman Pos: {pos} | Remaining: {rem}")

if __name__ == "__main__":
    main()
