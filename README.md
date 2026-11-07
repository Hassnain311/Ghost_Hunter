# Ghost Hunter: Tracking an Invisible Enemy with Hidden Markov Models

**Course:** AL-2002 Artificial Intelligence (Laboratory) · Fall 2026  
**Institution:** FAST — National University of Computer and Emerging Sciences, Karachi Campus  
**Department:** Department of Software Engineering / FAST School of Computing  

**Group Members:**
* **M. Hassnain** (24k-3020, BSE-5A) — *Leader:* Probabilistic Modeling (HMM, Forward Algorithm, Sensor Models), Experiment Pipeline, Benchmark Analysis.
* **Syed Khalid Ali Abidi** (24k-3005, BSE-5A) — *Member 2:* OOP Game Engine, Dynamic Pygame Heatmap GUI, Agent Behaviors & A* Pathfinding.

---

## 1. Project Overview
*Ghost Hunter* is a grid deduction game built around probabilistic reasoning under partial observability. An invisible ghost moves stochastically across a 10×10 obstacle maze. The hunter agent receives only noisy, discretized Manhattan-distance readings corrupted by zero-mean noise $P(e_t \mid X_t)$. 

Using a Hidden Markov Model (HMM) and the recursive **Forward Algorithm** (Prediction + Correction), the hunter continuously maintains a spatial probability distribution (belief heatmap), converges on the ghost's hidden location, and intercepts it using $A^*$ search with momentum tie-breaking.

---

## 2. Repository Layout
```text
Ghost_Hunter/
├── main.py                  # CLI entry: play | demo | experiments
├── config.py                # Grid dimensions, noise parameters, visual styling
├── requirements.txt         # NumPy, Pygame, Matplotlib, Pytest
├── README.md                # Project documentation and quickstart
│
├── ghost_hunter/
│   ├── __init__.py          # Package exports
│   ├── environment.py       # GridEnvironment, maze generation, turn rules, sensor
│   ├── hmm.py               # HMMTracker, Forward algorithm, normalization, entropy
│   ├── sensor_model.py      # Discrete noise PMF and likelihood matrix
│   ├── pathfinding.py       # A* with directional momentum penalty
│   ├── agents.py            # BaseAgent, RandomAgent, NaiveSensorAgent, HMMHunterAgent
│   ├── entropy_agent.py     # Optional EntropyHunterAgent extension
│   ├── gui.py               # PygameVisualizer, live belief heatmap, HUD telemetry
│   ├── metrics.py           # Turn-by-turn episode logging (Manhattan error, steps)
│   └── experiments.py       # Benchmark harness and figure generation
│
├── tests/
│   ├── test_environment.py  # Walls, bounds, connectivity, ghost movement
│   ├── test_hmm.py          # Probability normalization, floor, likelihood peaks
│   ├── test_pathfinding.py  # Shortest path, obstacle avoidance
│   └── test_agents.py       # Action validity across all agents
│
├── results/
│   ├── raw/                 # CSV summary data per trial
│   └── figures/             # 3 evaluation charts (PNG)
│
└── docs/
    └── report.md            # Complete theoretical derivation and results
```

---

## 3. Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hassnain311/Ghost_Hunter.git
   cd Ghost_Hunter
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. How to Run

### Mode 1: Interactive Human Play Mode
Take control of the hunter with arrow keys/WASD and attempt to catch the ghost using sensor pings and the live probability heatmap hint:
```bash
python main.py play
```
* **Controls:**
  * `W / A / S / D` or `Arrow Keys`: Move Hunter (North, South, West, East)
  * `.` (Period): Stay in place
  * `H`: Toggle belief heatmap display
  * `G`: Debug toggle (reveal invisible ghost position)
  * `ESC`: Exit game

### Mode 2: Autonomous AI Demo Mode
Watch the HMM Hunter agent autonomously track and capture the invisible ghost in real time:
```bash
python main.py demo
```
* `Space`: Pause / Resume simulation
* `ESC`: Exit

### Mode 3: Benchmark Experiments
Run the full automated benchmark comparing the **HMM Hunter**, **Naive Sensor Baseline**, and **Random Baseline** across different noise levels ($\sigma$):
```bash
python main.py experiments --trials 20 --topologies 5
```
* Outputs summary CSV logs to `results/raw/benchmark_summary.csv`.
* Automatically plots the 3 required evaluation charts in `results/figures/`:
  1. `chart1_estimation_error_vs_turns.png` (Error vs. Turns)
  2. `chart2_steps_to_capture_by_sigma.png` (Steps to Capture comparison)
  3. `chart3_success_rate_vs_noise.png` (Capture Success Rate vs. Sensor Noise $\sigma$)

---

## 5. Running the Test Suite
Ensure code verification and clean execution:
```bash
pytest tests/
```

---

## 6. Mathematical Formulation

### Time Update (Prediction Step)
Propagates the prior belief distribution $P(X_{t-1} \mid e_{1:t-1})$ through the stochastic ghost movement transition matrix $T$:
$$P(X_t \mid e_{1:t-1}) = \sum_{X_{t-1}} P(X_t \mid X_{t-1}) \cdot P(X_{t-1} \mid e_{1:t-1})$$

### Measurement Update (Correction Step)
Multiplies the predicted belief by the discrete distance sensor likelihood and normalizes:
$$P(X_t \mid e_{1:t}) = \alpha \cdot P(e_t \mid X_t) \cdot P(X_t \mid e_{1:t-1})$$
where $\alpha = \frac{1}{\sum_{s} P(e_t \mid X_t = s) \cdot P(X_t = s \mid e_{1:t-1})}$.
