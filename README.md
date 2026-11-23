# Ghost Hunter

**Tracking an Invisible Enemy with Hidden Markov Models**

AL-2002 Artificial Intelligence Laboratory, Fall 2026

FAST – National University of Computer and Emerging Sciences, Karachi Campus

| Team member | Roll number | Section | Project role |
| --- | --- | --- | --- |
| M. Hassnain | 24k-3020 | BSE-5A | Probabilistic modeling and evaluation |
| Syed Khalid Ali Abidi | 24k-3005 | BSE-5A | Game engine, interface and agents |

## Run the game

Extract the project, open a terminal in the folder containing `main.py`, and run these commands with Python 3.10 or later. Once the dependencies are installed, the game and its experiments run offline.

```sh
python -m pip install -r requirements.txt
python main.py
```

Select **Enter the Crypt**, choose a mode and difficulty, and enter your name. The ghost is caught when a hunter occupies its cell. A capture on the final turn counts. The local leaderboard ranks captures by active time within each mode and difficulty.

## Modes

| Mode | How the hunt works |
| --- | --- |
| Solo | Play with a live ghost timer, periodic glimpses and optional radar. |
| AI vs Human | Both hunters pursue the same ghost with one movement opportunity every 0.16 seconds. The rival plans its A* route privately. X shows your own route. The round ends after both captures or at the shared turn limit. |
| Human vs Human | Two players share a keyboard. Player 1 uses WASD and Player 2 uses the arrow keys. A simultaneous capture gives both players credit. |

| Difficulty | Grid | Turn limit | Live ghost interval | Rival strategy in AI vs Human |
| --- | --- | ---: | ---: | --- |
| Beginner | 10 × 10 | 100 | 1.00 s | Picks a plausible cell from its latest noisy ping and follows up to five steps of the route. |
| Intermediate | 15 × 15 | 150 | 0.75 s | Combines both hunters' noisy pings and retargets the most probable cell after at most two steps. |
| Hard | 20 × 20 | 200 | 0.55 s | Compares shortest routes to several likely cells at every step, favoring early capture opportunities. |
| Extreme | 30 × 30 | 300 | 0.40 s | Predicts where the ghost may move and compares interception routes using both hunters' distances through the maze. |

Open **Instructions** with **I** and select **Watch AI Hunt** for the autonomous HMM hunter. It updates a probability for each free cell and uses A* to reach the most likely cell. **Watch Entropy Hunter** demonstrates the optional information-seeking policy. Both demonstrations use turn-based movement and open in probability view.

### AI vs Human scorecard

Each hunter's clock starts with the round and stops at their own capture. Pauses are excluded. Moves count successful steps into a neighboring cell. At the turn limit, an unfinished hunter's row shows their moves and active time through the end of the round.

**Capture score = 1,000 − (10 × active seconds) − moves.** Time is rounded to the nearest tenth of a second and a capture earns at least one point. An unfinished hunt earns zero. The higher total wins and equal totals draw. For example, a hunter who finishes in 5.2 seconds with 17 moves earns 931 points and beats a hunter who finishes in 4.8 seconds with 23 moves and earns 929.

## Controls and sound

| Input | Action |
| --- | --- |
| WASD or arrow keys | Move. In a two-player duel, each player uses their assigned keys. |
| Period (`.`) | Wait one turn in a solo hunt or an AI vs Human race. |
| H | Switch between the posterior probability map and proximity rings around its most likely cell. |
| X | Activate radar in a live hunt. It illuminates the path for three seconds and recharges for five seconds. |
| G | Toggle the observer's ghost reveal in live Solo or an AI demonstration. |
| P / Esc | Pause or resume with P. Esc resumes from pause or returns to the menu during play. |
| R / F / F2 | Start a new maze, switch fullscreen, or mute all audio. |

Starting a hunt, selecting an option, going back and exiting have distinct sounds. A blocked move displays a wall alert and plays a thud. Capturing the ghost plays a latch and bell. **Exit** closes the application after its short door cue. Closing the window releases the mixer immediately. A second interactive launch leaves the existing session in control of audio.

## Evaluation

Run each command from the project folder.

```sh
python main.py demo --difficulty Hard --seed 42 --sigma 1.2
python main.py entropy --difficulty Beginner --seed 42 --sigma 1.2
python main.py experiments
python -m ghost_hunter.race_evaluation --trials 200 --first-seed 1000 --output results/raw/race_difficulty.csv
python -m ghost_hunter.race_evaluation --player heatmap --trials 200 --first-seed 1000 --output results/raw/race_heatmap.csv
python -m pytest -q
```

The full proposal benchmark runs 100 trials on each of 10 layouts at five noise levels for each of four policies. For a quick check in a separate folder, use `python main.py experiments --topologies 2 --trials 3 --output quick-results`.

The project report includes the measured results, three evaluation figures and the race-difficulty test conditions. [Project report](docs/report.md)
