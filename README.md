# Ghost Hunter: Tracking an Invisible Enemy with Hidden Markov Models

**Course:** AL-2002 Artificial Intelligence (Laboratory) · Fall 2026  
**Institution:** FAST — National University of Computer and Emerging Sciences, Karachi Campus  
**Department:** Department of Software Engineering / FAST School of Computing  

**Group Members:**
* **M. Hassnain** (24k-3020, BSE-5A) — *Leader:* Probabilistic Modeling (HMM, Forward Algorithm, Sensor Models), Experiment Pipeline, Benchmark Analysis.
* **Syed Khalid Ali Abidi** (24k-3005, BSE-5A) — *Member 2:* OOP Game Engine, Dynamic Pygame Heatmap GUI, Agent Behaviors & A* Pathfinding.

---

## 1. Project Overview
*Ghost Hunter* is a grid deduction game built around probabilistic reasoning under partial observability. An invisible ghost moves stochastically across an obstacle maze. The hunter agent receives only noisy, discretized Manhattan-distance readings corrupted by zero-mean noise $P(e_t \mid X_t)$. 

Using a Hidden Markov Model (HMM) and the recursive **Forward Algorithm** (Prediction + Correction), the hunter continuously maintains a spatial probability distribution (belief heatmap), converges on the ghost's hidden location, and intercepts it using $A^*$ search with momentum tie-breaking.

### Key Advanced Features:
* **Interactive Menu System:** Start Screen (`Ghost Hunter` title banner, `Start Game`, `Main Menu / How to Play`, and `Exit`).
* **Game Summary & Instructions Modal:** Detailed explanation of the HMM Forward algorithm, heatmap interpretation, controls, and game objectives.
* **4 Scalable Difficulties:**
  * **Beginner:** $10 \times 10$ Grid (100 Turns)
  * **Intermediate:** $15 \times 15$ Grid (150 Turns)
  * **Hard:** $20 \times 20$ Grid (200 Turns)
  * **Extreme:** $30 \times 30$ Grid (300 Turns)
* **High-Contrast Visible Obstacle Blocks:** Distinct Slate-700 / Iron Stone wall blocks with light beveled borders and corner rivets so obstacles never blend with the grid canvas.
* **Real-Time Ghost Movement:** The invisible ghost moves every **1.0 second** independently in real time!
* **A\* Ghost Radar Ability (`X` Key):** Illuminates the shortest obstacle-avoiding path to the ghost with an emerald pulsing glow for **2–3 seconds**, then undergoes a **5-second recharge cooldown**.

---

## 2. Installation & Setup

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

## 3. How to Run

### Main Interactive Launcher (Menu, Instructions, & Difficulty Selection):
```bash
python main.py
```
* Shows the **Start Screen Menu** with options to launch the game, inspect instructions & summary, or exit.
* Selecting **Start Game** opens the **Difficulty Selection Screen** ($10 \times 10$, $15 \times 15$, $20 \times 20$, $30 \times 30$).

### Direct Command Line Modes:
```bash
python main.py play --difficulty Intermediate   # Direct human play
python main.py demo --difficulty Hard           # Autonomous AI demonstration
python main.py experiments                      # Run 100-run benchmark & generate charts
```

---

## 4. In-Game Controls
* **`W / A / S / D` or `Arrow Keys`:** Move Hunter (North, South, West, East)
* **`X`:** **A\* Ghost Radar Ability** (reveals glowing path to ghost for 2–3 sec; 5 sec cooldown)
* **`R` or Click Button:** Restart with a **brand new random maze layout**
* **`H`:** Toggle probability heatmap hint
* **`G`:** Debug toggle (reveal invisible ghost position)
* **`Space`:** Pause / Resume
* **`ESC`:** Return to Main Menu / Exit

---

## 5. Verification & Tests
All 11 unit tests pass cleanly:
```bash
python -m pytest tests/
```
