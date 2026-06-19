# Plateau Diagnosis

Checkpoint: `/home/mwrona/PAC-MAN-AI/models/ppo_pacman.zip`
Episodes: 50

## End-of-episode pellet completion
- Mean: 9.4%
- Median: 9.4%
- Episodes >= 90%: 0.0%
- Level clear rate: 0.0%

## Death-time pellet completion
- Deaths recorded: 100
- Mean at death: 8.6%
- Deaths when >= 85%: 0.0%

## Spatial
- Pellet tiles never stepped on (mean/ep): 221.0
- Most ignored pellet tiles (row,col):
  - (15,21): missed in 50 episodes
  - (6,18): missed in 50 episodes
  - (7,26): missed in 50 episodes
  - (5,1): missed in 50 episodes
  - (20,20): missed in 50 episodes
  - (5,10): missed in 50 episodes
  - (8,9): missed in 50 episodes
  - (10,6): missed in 50 episodes
  - (22,26): missed in 50 episodes
  - (23,25): missed in 50 episodes

Histograms: `plateau_diagnosis.png`
Visit heatmap: `visit_heatmap.png`
Ignored pellets: `ignored_pellets.png`