# PROJECT PROPOSAL
**Semester Lab Project · Weightage 1 mark**  
**Course:** AL-2002 Artificial Intelligence (Laboratory) · Fall 2026  
**Institution:** FAST — National University of Computer and Emerging Sciences, Karachi Campus  
**Department:** Department of Software Engineering / FAST School of Computing  

---

### Group Information

| Name | Roll Number | Section | Role in the project |
| :--- | :--- | :--- | :--- |
| **Hassnain Arain** | **[Your Roll Number]** | **[Your Section]** | **Leader:** Probabilistic Modeling (HMM, Forward Algorithm, Sensor Models) & Experiment Evaluation |
| **[Partner Name]** | **[Partner Roll Number]** | **[Section]** | **Member 2:** OOP Game Engine, Pygame Heatmap GUI, Agent Behaviors & Baselines |

| Field | Detail |
| :--- | :--- |
| **Project Title** | Ghost Hunter: Tracking an Invisible Enemy with Hidden Markov Models |
| **Project Idea** | Idea No. **10** from the Project Ideas list [x] &nbsp;&nbsp;&nbsp;&nbsp; Own idea (needs approval) [ ] |
| **Category** | Part A — Game [x] &nbsp;&nbsp;&nbsp;&nbsp; Part B — Applied / Data [ ] |
| **GitHub Repository URL** | `https://github.com/Hassnain311/Ghost_Hunter.git` |
| **Date of Submission** | October 20, 2026 |

---

### 1. Project Title and Summary
**Title:** Ghost Hunter: Tracking an Invisible Enemy with Hidden Markov Models

*Ghost Hunter* is a grid-based turn-based strategy game where a human player and an autonomous AI agent compete to track down and capture an invisible moving adversary within a discrete maze environment. Because the ghost cannot be directly seen, the tracker only receives noisy, discretized distance readings corrupted by sensor error at each step. An AI system using a Hidden Markov Model (HMM) computes and maintains a dynamic 2D probability distribution (belief heatmap) across all possible grid locations, updating it in real time using the Forward algorithm. Without AI probabilistic filtering, noisy readings would mislead the tracker; with it, the agent resolves ambiguity, localizes the target over time, and executes optimal interception paths.

---

### 2. Problem Statement and Motivation
In target tracking under partial observability, an agent must infer the true internal state of an adversarial moving object without direct vision. In an $N \times M$ grid (e.g., $10 \times 10 = 100$ cells) where an invisible entity moves stochastically across $T$ timesteps, the combinatorial space of possible ghost trajectories is $O(|A|^T)$, where $|A|$ is the number of valid neighboring moves. Plain enumeration or brute-force search is hopeless. Furthermore, sensory inputs return distance measurements perturbed by sensor noise, meaning individual readings can point in completely incorrect directions. Human trackers and naive algorithms struggle because they fail to systematically propagate historical observations and spatial transitions. Formulating this system as a Hidden Markov Model enables recursive Bayesian filtering via the Forward algorithm, reducing trajectory tracking from exponential complexity down to linear time $O(T \cdot |S|^2)$. Solving this problem demonstrates practical state estimation under sensor uncertainty, fundamental to robotics, autonomous navigation, and tactical defense systems.

---

### 3. Problem Formulation

#### 3.1 PEAS Description
* **Performance Measure:** Number of turns/steps taken to capture the ghost; tracking accuracy (mean Manhattan distance between the maximum likelihood estimate $\arg\max P(X_t)$ and the true ghost position $X_t$); win-rate against the baseline and human players within a turn limit.
* **Environment:** A discrete 2D grid ($10 \times 10$) with fixed obstacle walls, containing one hidden moving ghost, noisy sensor stations/emissions, and catching agents.
* **Actuators:** Directional movements (North, South, East, West, Stay), sensor ping action, and attempt capture action.
* **Sensors:** Noisy Manhattan distance sensor reading emitted each turn with known sensor noise variance ($P(e_t \mid X_t)$), and wall collision feedback.

#### 3.2 Environment and Agent Type
* **Observability:** Partially observable [x]
* **Determinism:** Stochastic [x]
* **Change over time:** Dynamic [x] *(the ghost moves each turn)*
* **State and actions:** Discrete [x]
* **Agents:** Multi-agent / adversarial [x]
* **Agent type:** Model-based [x] *(maintains an internal transition and sensor model of the world)*

#### 3.3 Problem Representation
* **State representation:** The true state $S_t = (x_g, y_g) \in \text{Grid}$ is the ghost's hidden coordinates. The agent's belief state is a probability distribution matrix $B_t \in \mathbb{R}^{N \times M}$, where $B_t(i, j) = P(X_t = (i, j) \mid e_{1:t})$.
* **Actions / moves:** Agent action set $A = \{\text{Up}, \text{Down}, \text{Left}, \text{Right}, \text{Ping}, \text{Capture}\}$. Ghost action set: random walk to adjacent unblocked cells $A_g = \{\text{N}, \text{S}, \text{E}, \text{W}, \text{Stay}\}$.
* **Initial state:** True ghost location $S_0$ sampled uniformly from unblocked cells; belief state $B_0$ initialized to a uniform distribution $B_0(i, j) = \frac{1}{|FreeCells|}$.
* **Goal or terminal test:** The hunter occupies the exact same cell as the true ghost and executes the capture action, or the turn limit ($T_{max} = 100$) is reached.
* **Cost, utility, fitness or evaluation:** Cost $= 1$ per movement step. Utility of a state is evaluated by minimizing the expected distance to the belief distribution: $U(a) = -\sum_{(i,j)} B_t(i, j) \cdot \text{dist}(\text{pos}_{next}, (i,j))$.

---

### 4. Proposed AI Approach

| # | Technique | Role in the project | Own code or library |
| :--- | :--- | :--- | :--- |
| **1** | **Hidden Markov Model (HMM) & Forward Algorithm** | Maintains and recursively updates the spatial belief state distribution $P(X_t \mid e_{1:t})$ across time using prediction (transition model) and correction (emission model). | **Own code** |
| **2** | **Greedy Maximum A Posteriori (MAP) Hunter Agent** | Uses $A^*$ search on the grid to plan the shortest obstacle-avoiding path directly toward the highest probability cell $\arg\max_{(i,j)} B_t(i, j)$. | **Own code** |
| **3** | **Heuristic Information-Seeking Hunter Agent** | Balances moving toward high-probability cells while selecting positions that maximize information gain (entropy reduction) of the next sensor ping. | **Own code** |
| **B** | **Baseline:** | **Random Walk Hunter** and **Naive Raw-Sensor Hunter** (moves directly toward the raw noisy distance reading without any HMM filtering or history). | **Own code** |

**Technique Suitability & Probabilistic Formulation:**  
The core problem is tracking under partial observability and noise. At time $t$, the belief distribution is updated via the two-step Forward algorithm:
1. **Time Update (Prediction):** $P(X_t \mid e_{1:t-1}) = \sum_{X_{t-1}} P(X_t \mid X_{t-1}) \cdot P(X_{t-1} \mid e_{1:t-1})$ using a known transition matrix where the ghost moves uniformly to adjacent unblocked cells.
2. **Measurement Update (Correction):** $P(X_t \mid e_{1:t}) = \alpha \cdot P(e_t \mid X_t) \cdot P(X_t \mid e_{1:t-1})$, where $P(e_t \mid X_t)$ models sensor distance noise as a discretized Gaussian/triangular distribution around the true Manhattan distance, and $\alpha$ is a normalization constant ensuring $\sum B_t(i,j) = 1$.

#### 4.1 Planned Class Design
```text
GridEnvironment
  ├── Attributes: width, height, walls (set), true_ghost_pos, sensor_noise_std
  └── Methods: reset(), step(action), move_ghost(), get_noisy_reading(agent_pos), is_terminal()

BaseAgent (Abstract Class)
  ├── Attributes: position, grid_dims
  └── Methods: select_action(observation) [abstract], reset()

  ├── RandomAgent(BaseAgent)
  │     └── Methods: select_action(observation) [overridden: chooses random valid move]

  ├── NaiveSensorAgent(BaseAgent)
  │     └── Methods: select_action(observation) [overridden: moves toward raw sensor reading without filtering]

  ├── HMMTracker
  │     ├── Attributes: grid_dims, belief_matrix, transition_model, sensor_model
  │     └── Methods: time_update(), observation_update(reading, observer_pos), get_map_estimate(), get_entropy()

  └── HMMHunterAgent(BaseAgent)
        ├── Attributes: hmm_tracker (composition), search_planner (A*)
        └── Methods: select_action(observation) [overridden: plans path to argmax(belief)]

GameEngine / PygameVisualizer
  ├── Attributes: environment, hunter_agent, belief_renderer, screen, clock
  └── Methods: run_turn(), render_heatmap(), handle_events(), run_batch_experiments()
```

---

### 5. Scope

#### In scope
| # | Feature | Why it is essential |
| :--- | :--- | :--- |
| **1** | Complete $10 \times 10$ Grid Environment with obstacle walls and stochastic ghost movement | Defines the discrete state space and transition mechanics required by the HMM. |
| **2** | Pure Python implementation of HMM Forward Algorithm (Prediction & Correction) | Core requirement of Lab 09; performs recursive Bayesian belief updating over time. |
| **3** | Dynamic Pygame Belief Heatmap | Visualizes probability shifting from uniform to localized hotspots, critical for demonstration. |
| **4** | Autonomous HMM AI Hunter Agent (MAP / $A^*$ pathing) | Proves the utility of the probabilistic state estimator in guiding decision-making. |
| **5** | Automated Benchmark Harness with fixed seeds and baseline comparisons | Satisfies experiment guidelines by comparing HMM Hunter against Naive and Random baselines over 100 runs. |

#### Out of scope
* Continuous 3D/continuous coordinate state estimation (Kalman filters or particle filters are excluded to focus cleanly on discrete HMM).
* Reinforcement learning / deep Q-networks (unnecessary complexity for exact discrete Bayesian inference).
* Network multiplayer or web deployment.
* Multi-ghost simultaneous tracking (restricted to one ghost to keep state matrix tractable at $O(N \cdot M)$ rather than exponential).

---

### 6. Data or Environment Source

| Attribute | Specification |
| :--- | :--- |
| **Source name and URL** | Generated / simulated synthetic grid environment (Standard AI Benchmarking Maze) |
| **Type** | Generated / simulated [x] |
| **Access requirements** | No key needed [x] |
| **Approximate size** | Code-generated at runtime; zero storage footprint |
| **Licence / terms of use** | Open source / Academic Free Use |
| **Offline fallback** | Fully offline; no external network or API requests required. All grid topologies, transition matrices, and noise seeds are generated deterministically in local Python code. |

---

### 7. Evaluation Plan
* **What will be compared:** 
  1. HMM MAP Hunter Agent
  2. HMM Entropy-Reduction Hunter Agent
  3. Naive Raw-Sensor Hunter (Baseline 1)
  4. Random Walk Agent (Baseline 2)
* **Metric(s):**
  * Average steps to capture ghost across runs (Efficiency).
  * Mean Squared Error / Manhattan Error between estimated target position ($\arg\max B_t$) and actual ghost position across turns (Estimation Quality).
  * Capture success rate within 100 turns (Robustness).
* **Baseline:** Naive Raw-Sensor Hunter (greedy movement based purely on raw noisy distance without belief history) and Random Walk.
* **Test instances / test set:** 10 distinct seeded grid topologies (open, sparse walls, dense labyrinth) evaluated across 5 noise levels ($\sigma \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$).
* **Number of runs and seed:** 100 simulated trials per configuration with seeds fixed via `np.random.seed(42)` and `random.seed(42)` for reproducibility.
* **Planned charts (at least three):**
  1. **Convergence / Error Chart:** Average Estimation Error (Manhattan distance) vs. Turn Number for HMM vs. Raw Sensor.
  2. **Performance Comparison Bar Chart:** Mean Steps to Capture across 100 trials for all 4 agents.
  3. **Noise Sensitivity Line Plot:** Capture Success Rate (%) vs. Increasing Sensor Noise ($\sigma$) for HMM Hunter vs. Baselines.

---

### 8. Division of Work

| Member | Modules / components owned | Approximate share |
| :--- | :--- | :---: |
| **Hassnain Arain** | HMM Core Engine (`hmm.py`), Forward Algorithm (prediction and observation updates), Sensor Noise Models, Experiment Harness & Plotting Scripts (`experiments.py`), Technical Documentation. | 50% |
| **[Partner Name]** | OOP Game Environment (`environment.py`), Pygame GUI & Dynamic Heatmap Visualizer (`gui.py`), Agent Classes (`agents.py`: Random, Naive, HMM Hunter with $A^*$), Human Play Mode. | 50% |

---

### 9. Milestones and Timeline

| # | Milestone — what will be demonstrably working | Target date | Owner |
| :--- | :--- | :--- | :--- |
| **1** | Fully functional OOP grid environment with obstacle validation, seeded ghost random walk, and noisy sensor generation printing to console. | Week 4 | Member 2 |
| **2** | Working HMM Forward algorithm updating a $10 \times 10$ belief matrix from noisy distance readings, passing unit tests with zero-error synthetic data. | Week 7 | Leader |
| **3** | Pygame interface rendering grid walls, player token, and live color-graded belief heatmap updated after every move. | Week 10 | Member 2 |
| **4** | Autonomous HMM hunter agent using $A^*$ to chase maximum likelihood cells, alongside Naive and Random baseline agents. | Week 12 | Both |
| **5** | Experiments complete (100 runs, fixed seeds), 3 required evaluation plots generated, documentation finalized, and clean clone verified. | Week 14 | Both |

---

### 10. Expected Outcome and Risks
* **Expected Outcome:** A responsive Pygame application allowing interactive human play and autonomous AI demos. When run, the user sees a grid where sensor pings dynamically illuminate an intuitive probability heatmap, transitioning from widespread uncertainty to a sharp red hotspot tracking the invisible ghost. In automated trials, the HMM hunter is projected to capture the ghost in roughly one-third the turns required by the naive raw-sensor baseline, while maintaining localization stability even under high sensor noise.
* **Risks & Mitigation:**
  1. *Risk 1 (Belief Degeneracy / Underflow):* Repeated multiplications in the Forward algorithm could cause floating-point probabilities to vanish to zero in larger grids.  
     *Mitigation:* Implement numerical normalization ($\alpha$) after each measurement step and apply a tiny probability floor ($\epsilon = 10^{-7}$) across all unblocked cells.
  2. *Risk 2 (Ghost Trapping / Circular Chasing):* A purely greedy MAP agent might oscillate between adjacent cells if the belief distribution has multi-modal peaks.  
     *Mitigation:* Introduce a small historical momentum penalty or tie-breaking mechanism in path planning to prevent infinite oscillation.
