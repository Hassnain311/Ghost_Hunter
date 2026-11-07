# Project Report: Ghost Hunter (Tracking an Invisible Enemy with HMM)

**Course:** AL-2002 Artificial Intelligence (Laboratory) · Fall 2026  
**Institution:** FAST — National University of Computer and Emerging Sciences, Karachi Campus  
**Department:** Department of Software Engineering / FAST School of Computing  

**Team Details:**
* **M. Hassnain** (24k-3020, BSE-5A) — *Leader:* Probabilistic Modeling (HMM, Forward Algorithm, Sensor Models), Experiment Pipeline, Benchmark Analysis.
* **Syed Khalid Ali Abidi** (24k-3005, BSE-5A) — *Member 2:* OOP Game Engine, Dynamic Pygame Heatmap GUI, Agent Behaviors & A* Pathfinding.

---

## 1. Problem Formulation & PEAS
Ghost Hunter addresses the problem of **state estimation and pursuit under partial observability and sensor noise**. 

* **Performance Measure:** Minimum turn count to capture the ghost, localization accuracy (mean Manhattan estimation error), and capture success rate within a 100-step limit.
* **Environment:** Discrete $10 \times 10$ maze with impassable walls, an invisible stochastic ghost, and noisy Manhattan-distance emissions.
* **Actuators:** Movement in cardinal directions $\{\text{Up, Down, Left, Right, Stay}\}$.
* **Sensors:** Noisy Manhattan distance reading $e_t = \max(0, \text{round}(\text{dist} + \mathcal{N}(0, \sigma^2)))$ emitted each step.

---

## 2. Theoretical Derivation (HMM Forward Algorithm)

The system maintains a spatial probability distribution (belief vector) $B_t \in \mathbb{R}^{|S|}$ over all free cells $S$.

### 1. Time Update (Prediction Step)
When the ghost moves stochastically according to transition matrix $T$ (where $T_{i,j} = P(X_t = j \mid X_{t-1} = i)$):
$$P(X_t = j \mid e_{1:t-1}) = \sum_{i \in S} T_{i,j} \cdot P(X_{t-1} = i \mid e_{1:t-1})$$
In vectorized matrix notation:
$$\mathbf{b}'_t = T^T \mathbf{b}_{t-1}$$

### 2. Measurement Update (Correction Step)
Given a noisy distance reading $e_t$ observed from hunter position $p_h$:
$$P(X_t = s \mid e_{1:t}) = \alpha \cdot P(e_t \mid X_t = s, p_h) \cdot P(X_t = s \mid e_{1:t-1})$$
where the likelihood $P(e_t \mid X_t = s, p_h)$ is modeled by a discretized symmetric Gaussian PMF:
$$P(e_t \mid X_t = s) \propto \exp\left(-\frac{(e_t - \text{dist}(p_h, s))^2}{2\sigma^2}\right)$$
and $\alpha$ is a normalization scalar:
$$\alpha = \frac{1}{\sum_{s \in S} \mathbf{b}_t(s)}$$

To prevent numerical underflow over long episodes, a probability floor $\epsilon = 10^{-7}$ is enforced before normalization.

---

## 3. Architecture & Class Design
* **`GridEnvironment`:** Manages maze connectivity via BFS, enforces wall constraints, moves the ghost stochastically, and samples noisy sensor readings.
* **`BaseAgent` (Abstract):** Defines the standard interface `select_action(observation)`.
  * **`RandomAgent`:** Selects uniform valid moves.
  * **`NaiveSensorAgent`:** Reacts purely to the single latest distance reading with zero history.
  * **`HMMHunterAgent`:** Executes Forward algorithm updates on `HMMTracker` and invokes $A^*$ with directional momentum to intercept the MAP target $\arg\max B_t$.
* **`PygameVisualizer`:** Renders smooth color-gradient belief heatmaps, game tokens, and live telemetry (Shannon entropy, estimation error).

---

## 4. Experimental Results & Discussion

The automated benchmark was conducted across multiple noise standard deviations ($\sigma \in \{0.5, 1.0, 1.5, 2.0, 3.0\}$) and distinct connected topologies.

### Key Findings:
1. **Estimation Error vs. Turns (Chart 1):** The HMM Hunter rapidly reduces mean localization error within 5–7 turns as sensor evidence accumulates, whereas the raw sensor baseline error remains high and erratic.
2. **Steps to Capture (Chart 2):** The HMM Hunter consistently requires roughly one-third the turns of the naive baseline to capture the ghost.
3. **Robustness to Noise (Chart 3):** Even at high sensor noise ($\sigma = 3.0$), the HMM maintains a high capture rate due to Bayesian temporal filtering, while baselines degrade significantly.

The generated figures are available in `results/figures/`.
