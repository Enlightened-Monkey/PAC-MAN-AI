import os
import sys
import numpy as np
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environment.pacman_env import PacmanGridEnv
from src.environment.game_logic import DIRECTION_DELTAS, ROWS, COLS, TILE_PELLET, TILE_POWER

def find_bfs_action(env):
    start = env._state.pacman_pos
    maze = env._state.maze
    
    # BFS to find path to nearest pellet
    visited = np.zeros((ROWS, COLS), dtype=bool)
    q = deque()
    # (pos, initial_action)
    visited[start] = True
    
    # We check the 4 actions we can take from start
    for action, (dr, dc) in DIRECTION_DELTAS.items():
        nr, nc = start[0] + dr, start[1] + dc
        # Tunnel wrap
        if nr == 14 and nc < 0:
            nc = COLS - 1
        elif nr == 14 and nc >= COLS:
            nc = 0
            
        if not (0 <= nr < ROWS and 0 <= nc < COLS):
            continue
            
        t = maze[nr, nc]
        if t in (1, 4, 5): # Wall, Door, House
            continue
            
        if t in (TILE_PELLET, TILE_POWER):
            return action
            
        q.append(((nr, nc), action))
        visited[nr, nc] = True
        
    while q:
        (r, c), first_action = q.popleft()
        tile = maze[r, c]
        if tile in (TILE_PELLET, TILE_POWER):
            return first_action
            
        for dr, dc in DIRECTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if nr == 14 and nc < 0:
                nc = COLS - 1
            elif nr == 14 and nc >= COLS:
                nc = 0
                
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if visited[nr, nc]:
                continue
            if env._walk_mask[nr, nc] == 0:
                continue
                
            visited[nr, nc] = True
            q.append(((nr, nc), first_action))
            
    # Default fallback: any legal action
    masks = env.action_masks()
    for idx, allowed in enumerate(masks):
        if allowed:
            return idx
    return 0

def main():
    print("Testing BFS agent on PacmanGridEnv...")
    env = PacmanGridEnv(seed=42)
    obs, _ = env.reset()
    
    # Deactivate ghosts completely by modifying _move_ghost to be a no-op, 
    # to see if level transition works flawlessly when Pac-Man has zero threat!
    env._state._move_ghost = lambda g: None
    
    done = False
    step = 0
    
    while not done and step < 2000:
        action = find_bfs_action(env)
        obs, reward, terminated, truncated, info = env.step(action)
        step += 1
        done = terminated or truncated
        
        # Calculate remaining pellets
        maze = env._state.maze
        remaining = int(np.sum(np.isin(maze, [2, 3])))
        
        if step % 20 == 0 or remaining == 0:
            print(f"Step {step:4d} | Score: {env._state.score:4d} | Remaining: {remaining:3d} | Level: {env._state.level} | Lives: {env._state.lives}")
            
        if env._state.level > 1 and remaining == 244:
            print(f"\nSUCCESS! Reached Level {env._state.level} at Step {step}! Score: {env._state.score}")
            break

    print(f"\nFinal State - Step: {step} | Score: {env._state.score} | Level: {env._state.level} | Remaining Pellets: {remaining}")

if __name__ == "__main__":
    main()
