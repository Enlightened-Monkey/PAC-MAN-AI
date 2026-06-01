import os
import sys
import time
import numpy as np
from sb3_contrib import MaskablePPO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environment.pacman_env import PacmanGridEnv

ROWS, COLS = 31, 28
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_pacman'))

def main():
    print("Starting visual playback of the CLONED NEURAL NETWORK POLICY...")
    if not os.path.exists(CHECKPOINT_PATH + ".zip"):
        print(f"Error: checkpoint {CHECKPOINT_PATH}.zip not found!")
        return
        
    model = MaskablePPO.load(CHECKPOINT_PATH, device='cpu')
    print("Loaded neural network successfully! Running visual simulation...")
    time.sleep(1.5)
    
    env = PacmanGridEnv(seed=42)
    obs, _ = env.reset()
    done = False
    step_num = 0
    total_eval_reward = 0.0
    
    while not done:
        # Predict using action masks to prevent walking into walls
        action, _ = model.predict(
            obs,
            action_masks=env.action_masks(),
            deterministic=True
        )
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_eval_reward += reward
        done = terminated or truncated
        step_num += 1
        
        # Render the board
        state = env._state
        rows = []
        
        # Build a lookup for ghosts at each position
        ghosts_at_pos = {}
        for g in state.ghosts:
            ghosts_at_pos[g.pos] = g

        # ANSI clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        for r in range(ROWS):
            row_str = ""
            for c in range(COLS):
                pos = (r, c)
                if pos == state.pacman_pos:
                    direction = state.pacman_dir
                    pac_char = "◀"
                    if direction == ACTION_UP:
                        pac_char = "▲"
                    elif direction == ACTION_DOWN:
                        pac_char = "▼"
                    elif direction == ACTION_LEFT:
                        pac_char = "◀"
                    elif direction == ACTION_RIGHT:
                        pac_char = "▶"
                    row_str += f"\033[93m{pac_char}\033[0m" # Yellow Pac-Man
                elif pos in ghosts_at_pos:
                    g = ghosts_at_pos[pos]
                    if g.eaten:
                        row_str += "👀"  # Eaten ghost eyes
                    elif state.frightened_timer > 0:
                        row_str += "\033[94m👻\033[0m"  # Blue frightened ghost
                    else:
                        color_code = "\033[91m" # Red Blinky
                        if g.name == "Pinky":
                            color_code = "\033[95m" # Pink
                        elif g.name == "Inky":
                            color_code = "\033[96m" # Cyan
                        elif g.name == "Clyde":
                            color_code = "\033[92m" # Green/Orange
                        row_str += f"{color_code}👻\033[0m"
                else:
                    tile_val = int(state.maze[r, c])
                    if tile_val == 1:    # Wall
                        row_str += "\033[90m█\033[0m" # Dark grey wall
                    elif tile_val == 2:  # Pellet
                        row_str += "·"
                    elif tile_val == 3:  # Power Pellet
                        row_str += "\033[91m●\033[0m" # Red power pellet
                    elif tile_val == 4:  # Door
                        row_str += "═"
                    else:
                        row_str += " "
            rows.append(row_str)
            
        print("\n".join(rows))
        print(f"\nSteps: {step_num:4d} | Score: {state.score:4d} | Level: {state.level} | Lives: {state.lives} | Remaining Pellets: {np.sum(np.isin(state.maze, [2, 3]))}")
        time.sleep(0.08)  # Highly readable visual playback
        
    print(f"\nGame Over! Final Score: {state.score} | Level: {state.level}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nVisual playback interrupted by user.")
