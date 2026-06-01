import os
import sys
import numpy as np
from collections import deque

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.environment.pacman_env import PacmanGridEnv
from src.environment.game_logic import DIRECTION_DELTAS, ROWS, COLS, TILE_PELLET, TILE_POWER

def get_bfs_dist_to_ghosts(env):
    """Calculate BFS distance from every tile to the nearest normal ghost."""
    state = env._state
    ghost_dists = np.full((ROWS, COLS), 999, dtype=int)
    q = deque()
    
    frightened = state.frightened_timer > 0
    
    for g in state.ghosts:
        if g.eaten or g.in_house:
            continue
        # If frightened, they are harmless, but let's still track them with a lower priority
        if frightened:
            continue
        
        q.append((g.pos, 0))
        ghost_dists[g.pos] = 0
        
    while q:
        (r, c), d = q.popleft()
        for dr, dc in DIRECTION_DELTAS.values():
            nr, nc = r + dr, c + dc
            if nr == 14 and nc < 0:
                nc = COLS - 1
            elif nr == 14 and nc >= COLS:
                nc = 0
                
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if env._walk_mask[nr, nc] == 0:
                continue
            if ghost_dists[nr, nc] > d + 1:
                ghost_dists[nr, nc] = d + 1
                q.append(((nr, nc), d + 1))
                
    return ghost_dists

def find_expert_action(env):
    state = env._state
    start = state.pacman_pos
    maze = state.maze
    
    ghost_dists = get_bfs_dist_to_ghosts(env)
    
    # Evaluate each allowed action
    masks = env.action_masks()
    best_action = None
    best_score = -999999
    
    for action, allowed in enumerate(masks):
        if not allowed:
            continue
            
        dr, dc = DIRECTION_DELTAS[action]
        nr, nc = start[0] + dr, start[1] + dc
        if nr == 14 and nc < 0:
            nc = COLS - 1
        elif nr == 14 and nc >= COLS:
            nc = 0
            
        # 1. Safety Score
        g_dist = ghost_dists[nr, nc]
        
        # 2. Distance to nearest pellet from the target tile
        p_dist = 999
        visited = np.zeros((ROWS, COLS), dtype=bool)
        q_p = deque()
        q_p.append(((nr, nc), 0))
        visited[nr, nc] = True
        
        while q_p:
            (r, c), d = q_p.popleft()
            if maze[r, c] in (TILE_PELLET, TILE_POWER):
                p_dist = d
                break
            for dr2, dc2 in DIRECTION_DELTAS.values():
                nr2, nc2 = r + dr2, c + dc2
                if nr2 == 14 and nc2 < 0:
                    nc2 = COLS - 1
                elif nr2 == 14 and nc2 >= COLS:
                    nc2 = 0
                if not (0 <= nr2 < ROWS and 0 <= nc2 < COLS):
                    continue
                if visited[nr2, nc2]:
                    continue
                if env._walk_mask[nr2, nc2] == 0:
                    continue
                visited[nr2, nc2] = True
                q_p.append(((nr2, nc2), d + 1))
                
        # Heuristic scoring
        # We heavily penalise getting close to ghosts
        if g_dist <= 1:
            safety_penalty = -100000 # Death move
        elif g_dist == 2:
            safety_penalty = -5000   # Extremely dangerous
        elif g_dist == 3:
            safety_penalty = -1000   # Very dangerous
        elif g_dist == 4:
            safety_penalty = -200    # Guarded
        else:
            safety_penalty = 0
            
        # Eating scared ghosts is highly lucrative
        scared_bonus = 0
        if state.frightened_timer > 0:
            # BFS to nearest frightened ghost
            for g in state.ghosts:
                if not g.eaten and not g.in_house:
                    d_scared = abs(nr - g.pos[0]) + abs(nc - g.pos[1])
                    if d_scared < 5:
                        scared_bonus += (5 - d_scared) * 100
                        
        score = safety_penalty - p_dist + scared_bonus
        
        # Add a tiny bias to prevent oscillations in empty space
        # (e.g. prefer continuing in the same direction)
        if action == state.pacman_dir:
            score += 0.1
            
        if score > best_score:
            best_score = score
            best_action = action
            
    if best_action is None:
        # Fallback
        for idx, allowed in enumerate(masks):
            if allowed:
                return idx
        return 0
        
    return best_action

def main():
    print("Evaluating Heuristic Expert Agent on PacmanGridEnv with ACTIVE GHOSTS...")
    env = PacmanGridEnv(seed=42)
    
    n_episodes = 5
    scores = []
    levels = []
    steps_list = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        step = 0
        
        while not done:
            action = find_expert_action(env)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            done = terminated or truncated
            
        score = env._state.score
        level = env._state.level
        
        scores.append(score)
        levels.append(level)
        steps_list.append(step)
        
        print(f"Episode {ep+1:2d} | Score: {score:4d} | Level Reached: {level} | Steps: {step:4d}")

    print("\n--- Expert Summary Statistics ---")
    print(f"Mean Score: {np.mean(scores):.1f}")
    print(f"Mean Level: {np.mean(levels):.2f}")
    print(f"Mean Steps: {np.mean(steps_list):.1f}")

if __name__ == "__main__":
    main()
