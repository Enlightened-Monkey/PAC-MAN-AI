# Research Report: System Architecture, Gameplay Mechanics, and Artificial Intelligence Algorithms in Pac-Man (Arcade, 1980)

---

## Introduction: System Architecture and Historical Context

The original Pac-Man, created by Japanese corporation Namco and released in May 1980 on the Japanese market (originally under the name *Puck Man*) and in August 1980 by Midway Manufacturing in North America, stands as one of the most significant achievements in the history of software engineering and digital artificial intelligence design. The project, led by chief designer Toru Iwatani in collaboration with programmers Shigeo Funaki and Shigeichi Ishimura, aimed to create a pacifist alternative to the shoot-'em-up arcade market then dominated by titles such as *Space Invaders* and *Galaxian*. Rather than relying on destruction, Iwatani designed an environment centred on navigation, consumption, and threat avoidance — intended to attract a wider demographic, including women.

The game is also one of the first in history to use cinematic cutscenes, providing brief moments of rest for the player. These appeared after rounds 2, 5, 9, 13, and 17. The commercial success of the concept rested not only on its innovative aesthetics, but above all on the implementation of a sophisticated, multi-layered AI system that — within severely constrained hardware limits — created a convincing illusion of purposeful, intelligent, and varied behaviour.

From a hardware perspective, early arcade systems demanded extreme resource optimisation. The original Namco hardware platform (8-bit PCB) was built around a **Zilog Z80A** microprocessor clocked at **3.072 MHz**. Total system memory was drastically limited: the game code and graphics data occupied **16 KB of ROM** (split across four 4 KB chips), while RAM amounted to a mere ~**2 KB** — 1 KB dedicated to video information and 1 KB to colour data.

These architectural constraints meant that sophisticated pathfinding algorithms familiar from modern computer science (such as A\* or real-time distance-map graph traversal) were physically impossible to implement. The developers needed a deterministic method that minimised CPU overhead while still delivering the unpredictability required to keep players engaged.

The display is a vertically oriented raster monitor at **224 × 288 pixels**, refreshed at the precise, non-standard frequency of **60.606061 Hz** — which directly drove the game's internal state-machine timer and dictated the engine's computational cycle. Synchronising AI logic, movement, and animation frame updates with the refresh rate allowed the engineers to manage the on-screen positioning of **six independent 16 × 16 pixel hardware sprites** with perfect smoothness.

The key insight of the programming team was to completely decouple visual perception from analytical spatial logic. The screen was logically divided into a two-dimensional grid of **8 × 8 pixel tiles**, forming a coordinate system of **28 columns × 36 rows**. Understanding this tile topology is absolutely critical for analysing every aspect of the game engine: ghost navigation decisions, collision detection, target tile computation, and actor movement speed are all resolved at the tile level.

---

## Spatial Logic and Maze Topology

The division of game space into **1,008 tiles (28 × 36)** was an elegantly simple solution to the memory shortage. Not all tiles are accessible; many represent "dead space" or wall structures. Pac-Man and the ghosts may only move through the logically defined "legal space" of corridors, where dots act as navigational waypoints. The distance between each consumable node (dot) is exactly **8 pixels** — one tile — making spatial calculations straightforward for both players and distance-estimation algorithms.

The engine always defines each moving entity (actor) by the coordinates of its **centre point**. When a 16 × 16 ghost sprite slides smoothly across the screen, the internal state machine tracks only one point: the exact sprite centre. A state transition from one tile to the next is recorded in RAM only at the frame in which the centre crosses the 8 × 8 boundary. This has radical implications for collision detection: **the player dies only when the centre point of Pac-Man and the centre point of a ghost occupy the same logical tile simultaneously**.

### The Collision Paradox: Pass-Through Bug

This tile-based collision detection directly gave rise to a rare but documented phenomenon where collision detection is bypassed — the **pass-through bug**. In a continuous-physics environment, a collision would always occur when opposing vectors cross. In Namco's discrete hardware physics, positions update in discrete jumps every 1/60 of a second.

If Pac-Man and a ghost move directly towards each other with a combined velocity that covers exactly one tile per frame, there exists a single-frame tolerance: Pac-Man's centre moves from its tile to the tile ahead at the same moment the ghost's centre leaves that tile and enters Pac-Man's former position. Because the collision check only verifies whether both objects share the **same cell in the current frame**, they swap positions without ever being recorded as occupying the same tile simultaneously. The objects phase through each other — a visual illusion of momentary intangibility that is purely a consequence of the low temporal resolution of entity-overlap checking.

### The Safe Spot

A specific dead spot exists in the maze directly above Pac-Man's starting position. If Pac-Man stops there while ghosts are not in an active, direct-targeting chase mode, the ghost patrol vectors will curve around him indefinitely. This is a direct result of the navigational algorithms, for which that particular tile becomes effectively "transparent" in the absence of a suitable input vector.

### Upward-Movement Restriction Zones ("Red Zones")

Maze topology includes carefully designed invisible exclusion zones known as **Red Zones**. The map designers anticipated that the ghosts' default tie-breaking algorithm would give the AI a strong tendency to choose the upward direction whenever equivalent routes exist. To prevent ghosts from clustering and falling into tight patrol loops around the central ghost house, the game defines **four specific upward-facing tunnel entrances that are forbidden**.

When any ghost enters one of these four designated intersections above the ghost house while in **Chase** or **Scatter** mode, the filtering software unconditionally removes the upward option from the pool of possible exit vectors. The ghost is forced to continue horizontally, widening its orbit and encouraging a more spread-out movement pattern.

Crucially, this restriction does **not** apply to Pac-Man — a significant tactical advantage. A fleeing player can freely enter these upward channels with the guarantee that a directly pursuing ghost will be forced to take the longer route around. The Red Zone restriction is automatically suspended during **Frightened** mode, when ghosts navigate via a pseudo-random number generator.

---

## Digital Kinematics and Speed Parameterisation

Movement speed in Pac-Man is neither constant nor uniform. It is a highly flexible parameter adapted in real time based on the global difficulty level (round number), the specific action being performed, the territory occupied by the actor, and the current AI mode. Speeds are measured with pixel precision and frequently expressed as fractions, from which the software implements sub-pixel positioning relative to a **100% nominal speed**.

| Speed Variable | Level 1 | Levels 2–4 | Levels 5–20 | Levels 21+ |
|:---|:---:|:---:|:---:|:---:|
| **Pac-Man normal speed** | 80% | 90% | 100% | 90% |
| **Pac-Man speed (while eating)** | ~71% | ~79% | ~87% | ~79% |
| **Ghost normal speed** | 75% | 85% | 95% | 95% |
| **Ghost speed in tunnels** | 40% | 45% | 50% | 50% |
| **Ghost speed (Frightened mode)** | 50% | 55% | 60% | *Disabled* |
| **Pac-Man speed (ghosts frightened)** | 90% | 95% | 100% | *Disabled* |
| **Pac-Man speed (eating while frightened)** | ~79% | ~83% | ~87% | *Disabled* |

On Level 1, Pac-Man moving through an empty corridor (80%) is noticeably faster than the ghosts (75%), enabling relatively easy learning of mechanics. From Level 5 to 20, Pac-Man reaches 100% while ghosts are capped at 95%. The balance inverts dramatically at Level 21 and beyond: Pac-Man's base speed drops to 90% while the ghosts remain at 95%. **Outrunning a ghost in a straight corridor becomes mathematically impossible.**

### The Dot-Eating Penalty

Pac-Man's nominal speed rarely reflects his actual mobility in play, due to a hardware-mandated phenomenon called **dot-eating friction**. Every time collision registers a consumed Pac-Dot, the engine stalls Pac-Man's movement for exactly **1 frame** (1/60 s). When a Power Pellet is consumed, the pause extends to **3 frames** (~0.05 s).

Since dots are spaced 8 pixels (one tile) apart, continuous eating reduces effective speed by roughly **9–13%** relative to the base speed. At Level 1, eating speed falls from 80% to approximately 71% — placing Pac-Man behind a ghost moving at a constant 75%. The player must therefore continuously decide: clear the maze (reducing distance to enemies) or exploit previously cleared "clean paths" for maximum escape velocity.

### Warp Tunnels and Path Optimisation

The lateral warp tunnels connecting the east and west edges of the maze introduce another asymmetric speed manipulation. When any ghost's centre enters the tunnel entrance tiles, its maximum speed is reduced to nearly half — **40% on Level 1, 50% on higher levels**. This gives the player a primary "escape clutch": deliberately leading ghosts through the tunnel guarantees a large positional gap.

### Cornering: Sub-Pixel Geometry Optimisation

Without a compensating mechanism, the unfavourable kinetic balance from Level 21 onward would make skilled play impossible. That mechanism is **cornering**.

Ghost AI is geometrically rigid: to turn 90 degrees at a tile, a ghost must precisely reach the **exact centre** of the tile before changing its vector, costing precious frames. Pac-Man operates under a completely different rule. The input-reading engine registers joystick input **several frames before** the centre of the target tile is reached, storing the signal as a buffered vector. If the player holds the joystick in the turn direction, the engine activates a **pre-turn** mode: the avatar's sprite immediately rotates to face the new direction and then travels diagonally at ~45 degrees to the coordinate system.

During this transition, the sprite advances **one pixel in the new direction for every one pixel in the original direction** until it aligns with the centreline of the target corridor. Due to vector addition, the combined horizontal + vertical speed is **temporarily doubled** during this arc. Perfectly buffered cornering allows the player to skip multiple interpolation frames — a mathematical escape-distance surplus that is often the **only** way to survive the highest difficulty levels.

---

## The Deterministic State Machine and Operational Phases (Attack Waves)

The most elegant aspect of Namco's AI design is its **wave-based mode system**. Rather than having ghosts permanently pursue the player, the AI alternates between deterministic modes controlled by hidden timers. The system is built on three primary modes: **Scatter**, **Chase**, and **Frightened**.

### Primary Modes and the Reversal Signal

**Chase mode** — The AI attempts to optimise paths towards a Target Tile located near or on the player's avatar. Each ghost uses its own personalised mathematical formula to calculate this target.

**Scatter mode** — Ghost targeting is suspended. Each ghost is assigned a fixed corner tile located outside the playable map, forcing it into a looping patrol orbit. Blinky targets the top-right corner, Pinky the top-left, Inky the bottom-right, and Clyde the bottom-left (matrix coordinates exactly X:0, Y:0). This gives the player a brief respite and opens up congested corridors.

**Reversal Signal** — In its natural state, no ghost will ever reverse direction 180 degrees of its own accord. However, on every **Chase → Scatter** or **Scatter → Chase** transition — and whenever Frightened mode is triggered — the hardware forces a global command: every ghost simultaneously reverses its movement vector. The player sees this as a dramatic synchronised U-turn of the entire ghost formation. Notably, when Frightened mode *ends* and the prior mode resumes, **no reversal is emitted** — the transition is linear and unannounced.

### Wave Schedule

| Interval (Mode) | Level 1 | Levels 2–4 | Levels 5+ |
|:---|:---:|:---:|:---:|
| **Scatter 1** | 7 s | 7 s | 5 s |
| **Chase 1** | 20 s | 20 s | 20 s |
| **Scatter 2** | 7 s | 7 s | 5 s |
| **Chase 2** | 20 s | 20 s | 20 s |
| **Scatter 3** | 5 s | 5 s | 5 s |
| **Chase 3** | 20 s | 1,033 s (~17 min) | 1,037 s (~17 min) |
| **Scatter 4** | 5 s | 1/60 s (1 frame) | 1/60 s (1 frame) |
| **Chase 4** | Unlimited | Unlimited | Unlimited |

Starting from Level 2, the third Chase phase expands to over 1,033 seconds (~17 minutes of real gameplay). The fourth Scatter phase shrinks to a single frame (1/60 s) — invisible in terms of navigation, but still sufficient for the engine to emit a **global Reversal signal**, abruptly snapping all ghosts 180 degrees to break player safety patterns before immediately resuming the permanent final Chase.

### Frightened Mode and the Pseudo-Random Number Generator

Consuming a Power Pellet triggers **Frightened mode**: ghosts turn deep blue, their speed drops significantly (50–60% depending on level), and the concept of a Target Tile is entirely abandoned. Instead, ghost direction choices at intersections are driven by a **pseudo-random number generator (PRNG)** reading from a looped machine register.

Critically, this is **not** true randomness. The engine uses an identical seed at the start of each level, and the register values evolve only with the constant game cycle — making all "frightened escape" behaviour **fully deterministic** and learnable.

Eating a frightened ghost triggers the **exponential ghost combo score**: 200 → 400 → 800 → 1,600 points for each successive ghost eaten before the phase ends. Eaten ghosts become bodiless **eyes** — the absolute fastest objects on screen — and navigate directly to the fixed Target Tile above the ghost house door for respawning.

Before Frightened mode ends, ghosts **flash white** to warn the player of imminent danger. From Level 19 onward, the phase is so short that ghosts never even turn blue; consuming a Power Pellet only forces a global 180-degree reversal.

---

## The Computational Pathfinding Algorithm

When the state machine is not in Frightened mode, ghost movement operates through a precision targeting system called the **Target Tile mechanism**. The developers were forced to abandon global tree-search algorithms, relying instead on **Euclidean distance estimation and step-by-step optimisation** to minimise CPU overhead.

### Route Selection: Look-Ahead Logic

Every time a ghost's centre enters a new tile (Tile A), the internal algorithm immediately identifies the **next tile ahead** (Tile B) and pre-evaluates it. The decision is not made when the ghost arrives at an intersection — it is made one tile in advance.

When evaluating a junction tile, the engine:

1. Removes the **reverse direction** from the candidate pool.
2. Eliminates directions blocked by **walls**.
3. For each remaining candidate tile, calculates the **squared Euclidean distance** to the Target Tile:

$$d^2 = \Delta x^2 + \Delta y^2$$

The square root is omitted — comparing squared distances is sufficient and computationally cheaper.

The direction with the smallest $d^2$ is selected. The algorithm does **not** verify physical reachability along the entire chosen corridor — it makes a "blind" assignment towards an abstract point, which sometimes leads the ghost into a geometrically shorter but physically longer path around maze curves.

### Mathematical Tie-Breaking and Direction Priority

When two candidate tiles produce identical distance values, the engine applies a **hard-coded priority table**:

1. **Up** — highest priority
2. **Left** — second
3. **Down** — third
4. **Right** — lowest (default fallback)

This means that with equally distant corridors going left and down, the ghost will always turn **Up** or **Left** first. The Red Zones above the ghost house exist specifically to block this upward dominance and prevent ghosts from getting trapped in tight northern loops.

---

## Ghost Behavioural Architecture and Targeting Models

Instead of multiplying identical attacking forces, Namco assigned each ghost a **distinct Target Tile formula** for Chase mode, creating genuinely differentiated AI behaviour. In Scatter mode, all targeting is suspended and each ghost retreats to its fixed corner tile.

### Blinky (Red) — Direct-Pursuit Algorithm

*Oikake* ("Shadow") — Blinky's Target Tile is **set to Pac-Man's current tile** at every moment:

$$T_{\text{Blinky}} = P_{\text{Pac-Man}}$$

This produces brutally confrontational behaviour: Blinky appears to never leave the player's shadow.

**Cruise Elroy:** When the number of remaining dots drops below a level-specific threshold (e.g., 20 dots remaining on Level 1 → *Elroy 1*; 10 dots → *Elroy 2*), Blinky's speed is boosted — on Level 1 from 75% to 85%, then to a speed exceeding Pac-Man's. Additionally, the Scatter mode command is **ignored** for Blinky once Elroy is active — he continues pursuing the player at full aggression even during other ghosts' scatter phases.

### Pinky (Pink) — Interception Algorithm

*Machibuse* ("Ambusher") — Pinky's Target Tile is calculated as the tile **4 steps ahead** of Pac-Man in his current direction of travel:

$$T_{\text{Pinky}} = P_{\text{Pac-Man}} + 4 \cdot \vec{v}_{\text{Pac-Man}}$$

She targets where the player is *going*, not where they *are* — cutting off escape routes from the front. Combined with Blinky pursuing from behind, this creates a **pincer formation** that closes from both directions simultaneously.

**Head-Faking:** The player can exploit Pinky by briefly turning directly towards her. This falsely signals Pinky to target a tile behind and to the side of Pac-Man, causing her to overshoot.

#### The Up-Direction Bug: Z80 Register Overflow

Pinky's look-ahead algorithm contains a **permanent, unfixed bug** resulting from a Z80A arithmetic overflow. The engine stores movement direction as a 2-byte HL register pair (H = X, L = Y). When calculating the 4-tile offset with the `ADD HL, HL` instruction, the carry bit from the lower byte (L) overflows into the upper byte (H) for the upward direction vector.

The result: whenever Pac-Man faces **upward**, Pinky's Target Tile is shifted **4 tiles up and 4 tiles to the left** simultaneously. In practice, Pinky's intercept mechanic is broken for northward corridors — she misses the vertical path, veering upper-left instead.

### Inky (Cyan) — Cascading Geometry Algorithm

*Kimagure* ("Whimsical") — Inky's targeting depends on **both** Pac-Man's position/direction **and** Blinky's current position. His Target Tile is calculated in three steps:

1. **Compute the Offset Tile** — 2 tiles ahead of Pac-Man in his current direction. (Inherits the same Up-Direction Bug, shifting the offset 2 tiles left when Pac-Man faces up.)
2. **Draw a vector** from Blinky's position to the Offset Tile.
3. **Double the vector** — extend it an equal distance beyond the Offset Tile:

$$T_{\text{Inky}} = P_{\text{offset}} + (P_{\text{offset}} - P_{\text{Blinky}})$$

This creates the **"vice effect"**: when Blinky is far from Pac-Man, Inky's target is thrown to the opposite edge of the map — making him effectively harmless. When Blinky closes in, Inky's target snaps tightly alongside Pac-Man, creating a simultaneous cross-fire threat. Inky's aggression is entirely governed by external factors outside his own logic.

### Clyde (Orange) — Radial Switch Algorithm

*Otoboke* ("Feigning shyness") — Clyde uses a simple **proximity switch** based on Euclidean distance to Pac-Man, with a fixed threshold radius of **8 tiles**:

- **If distance > 8 tiles:** Clyde uses **the same direct-pursuit formula as Blinky**.
- **If distance ≤ 8 tiles:** Clyde's target immediately switches to his **Scatter corner** in the bottom-left dead zone.

This binary switch causes Clyde to **oscillate** near the 8-tile boundary: he charges at Pac-Man, then retreats at threshold, increases his distance, then charges again. A head-on charge directly towards Clyde is therefore the safest manoeuvre — he will instinctively retreat before impact.

---

## Ghost House Release System and Dot Counters

The ghost house uses **dot-count-based release timers** to control when each ghost enters the maze. The release priority order is: Blinky (starts outside), then Pinky, then Inky, then Clyde. Each ghost is assigned a **personal dot counter** with a level-specific threshold.

When **Pac-Man dies**, all personal counters are immediately suspended and a single **Global Dot Counter** takes over:

| Ghost | Global dots required for release |
|:---|:---:|
| **Pinky** | 7 |
| **Inky** | 17 |
| **Clyde** | 32 |

An additional **stuck timer** prevents indefinite lockdown: if 4 seconds pass (3 seconds on Level 5+) without any dot being eaten, the highest-priority in-house ghost is force-released regardless of counter values.

---

## Scoring, Bonus Fruit, and Audio

### Scoring System

| Item | Points |
|:---|:---:|
| Pac-Dot | 10 |
| Power Pellet | 50 |
| Ghost #1 (per power pellet) | 200 |
| Ghost #2 | 400 |
| Ghost #3 | 800 |
| Ghost #4 | 1,600 |

Each board contains exactly **240 Pac-Dots** and **4 Power Pellets** (244 total), yielding a base **2,600 points** for a fully cleared level. Reaching **10,000 points** grants one **extra life (1-UP)**.

### Bonus Fruit

Bonus fruit appears **twice per level**, spawned in the corridor directly below the ghost house entrance — after **70 dots** consumed and again after **170 dots**. Each appearance lasts a random **9–10 seconds** before disappearing.

| Level | Fruit | Points |
|:---:|:---|:---:|
| 1 | Cherry | 100 |
| 2 | Strawberry | 300 |
| 3–4 | Peach | 500 |
| 5–6 | Apple | 700 |
| 7–8 | Grapes | 1,000 |
| 9–10 | Galaxian Boss | 2,000 |
| 11–12 | Bell | 3,000 |
| 13+ | Key | 5,000 |

### Audio System

The background siren is directly coupled to the game's progress: its **pitch rises** as more dots are consumed, subconsciously building tension as the level approaches completion. The "waka-waka" eating sound is mathematically synchronised so that **one complete audio file plays across exactly two consecutive dot consumptions**.

---

## Extreme Software Anomaly: The Kill Screen (Level 256)

The Level 256 "Split Screen" — colloquially the **Kill Screen** — is the definitive demonstration of the Z80 architecture's memory loop limits.

The engine tracks the current level in a single **8-bit unsigned register**, maximum value 255 (0xFF). After completing Level 255, the engine increments the counter, causing an **integer overflow** that wraps it back to **0x00**. This triggers the Level 0 / Level 256 rendering routine with an invalid state.

### Visual Corruption and Uncompletable Level

The overflow corrupts the **right half of the screen**: the fruit-indicator rendering routine enters an unconstrained loop, drawing 256 entries from a ROM lookup table that contains only a few valid fruit graphics. Invalid addresses pull **garbage data** from the video character encoding block (VRAM), filling the right half of the screen with scrambled symbols and garbled colours.

The left half renders correctly with **116 dots**. The corrupted right side contains only **9 dots** (including **4 that are completely invisible** — masked by corruption yet still triggerable by sound). The total collectible dot count is therefore **125**, far below the **244 required to complete the level** — making it **mathematically unbeatable**.

Each death regenerates 9 dots on the right side. Using all 5 spare lives to cycle these regenerations, a player can collect at most **168 dots** — still short of 244. This establishes the theoretical maximum score:

> **3,333,360 points** — the "Perfect Score" achievable before the Kill Screen locks all further progress.

---

## Conclusions

The structural disassembly of Pac-Man reveals a multi-layered simulation design that was the true foundation of the engine's success. Iwatani's team, working with an extraordinarily limited RAM budget on Z80A-era hardware, constructed an illusion of asymmetric spatial intelligence using nothing more than rigid tile-based instruction sets and deterministic vector tables.

What appear to be machine errors — the carry-bit arithmetic spill of the *Up-Direction Overflow*, or the single-unit binary overflow of the *Kill Screen* integer — built the foundational lessons of today's pathfinding engineering. The simplified target-tile distance assignment and compressed vector calculation cost invented to squeeze performance from the Z80 went on to define fundamental optimisation patterns still referenced in modern AI. The precise calculations carved by these hardware constraints stand as an enduring monument to the genius of an era that demanded the absolute ceiling of performance from a handful of integrated circuit loops.

---

## References

1. Pac-Man — Wikipedia. <https://en.wikipedia.org/wiki/Pac-Man>
2. PAC-MAN AT 40 — Claret and Blue Pixels. <https://claretandbluepixels.wordpress.com/2020/02/27/pac-man-at-40/>
3. How Pac-Man Ghost AI Works. <https://aighost.co.uk/how-pac-man-ghost-ai-works-the-classic-chase-algorithms/>
4. Pac-Man Ghost AI — How the Classic Game's Enemies Think. <https://aighost.co.uk/pac-man-ghost-ai-how-the-classic-games-enemies-think/>
5. **The Pac-Man Dossier** — Jamey Pittman. <https://pacman.holenet.info/>
6. Pac-Man Hardware — Walk of Mind. <https://walkofmind.com/programming/pie/hardware.htm>
7. The Pac-Man Dossier — Game Developer. <https://www.gamedeveloper.com/design/the-pac-man-dossier>
8. Original Pac-Man pixel-per-frame speeds — Reddit. <https://www.reddit.com/r/Pacman/comments/1cg2ogp/>
9. Carts Tapes Discs Drives — metopal. <http://metopal.com/2011/03/20/carts-tapes-discs-drives/>
10. Pac-Man — Guide to Mastering the Maze! — Steam Community. <https://steamcommunity.com/sharedfiles/filedetails/?id=593226813>
17. Ghost Behaviour Notes — GitLab. <https://gitlab.mi.hdm-stuttgart.de/jj047/guenter/-/blob/main/ghost-behaviour-notes.md>
18. Feature: The Pac-Man Dossier — Game Developer. <https://www.gamedeveloper.com/game-platforms/feature-the-i-pac-man-i-dossier>
20. How Do the Ghosts in PAC-MAN Decide Where to Go? — TodayIFoundOut. <http://www.todayifoundout.com/index.php/2015/10/ghosts-pac-man-work/>
21. An Analysis of Target Tile Pathfinding in Classic Pac-Man Ghost AI — TechRxiv. <https://www.techrxiv.org/doi/10.36227/techrxiv.176964110.02978086>
26. Bugs: Pac-Man (Arcade) — The Cutting Room Floor. <https://tcrf.net/Bugs:Pac-Man_(Arcade)>
27. Pac-Man's Ghost Behaviour — donhodges.com. <http://donhodges.com/pacman_pinky_explanation.htm>
28. The Pac-Man Dossier (PDF). <https://cs.au.dk/~ocaprani/GameAI/PacMan/The%20Pac-Man%20Dossier.pdf>
