# Behavior Analysis

Checkpoint: `ppo_pacman.zip` | Episodes: 30

## Aggregate
- Mean pellet completion: 91.0%
- Mean score: 3114
- Level clears: 0/30
- Mean deaths/ep: 2.93
- Idle (no movement) steps: 17.6%
- Looping (revisit same tile 4+ times): 40.3%
- Moves onto non-pellet cells: 82.4%
- Power pellets eaten (ghost far): 3.57/ep
- Power pellets eaten (ghost near): 0.40/ep
- Ghosts eaten while frightened: 2.93/ep

## Interpretation
- High idle% = policy outputs blocked moves or stands still.
- High loop% + high no_pellet_move% = panic flee in empty corridors.
- power_waste >> power_smart = eats power pellets for +score, ignores ghost mechanic.
- frightened_eats ~ 0 = never learned to chase blue ghosts.