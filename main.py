"""
Ghost Hunter Command Line Interface.
Supports:
  1. Main Menu System: Title, Start, Main Menu, Exit
  2. Difficulty Selection: Beginner (10x10), Intermediate (15x15), Hard (20x20), Extreme (30x30)
  3. Real-Time Ghost Movement: moves every 1.0 second
  4. A* Ghost Radar Ability: press [X] to illuminate shortest path to ghost for 2-3s (5s cooldown)
  5. Interactive Play, Autonomous AI Demo, and Benchmark Experiments
"""

import sys
import argparse
import time
import random
from config import (
    DEFAULT_SENSOR_SIGMA,
    DIFFICULTY_CONFIGS,
    GHOST_MOVE_INTERVAL_SEC
)
from ghost_hunter.environment import GridEnvironment, generate_maze
from ghost_hunter.agents import HMMHunterAgent
from ghost_hunter.gui import PygameVisualizer
from ghost_hunter.pathfinding import astar
from ghost_hunter.experiments import run_benchmark


def run_game_loop(
    gui: PygameVisualizer,
    difficulty_name: str = "Beginner",
    mode: str = "play",
    seed: int = 42,
    sigma: float = DEFAULT_SENSOR_SIGMA
) -> str:
    """
    Executes an active session with real-time 1.0s ghost movement and A* radar capability.
    Returns:
        'menu' if user pressed Esc or clicked Menu button,
        'exit' if window was closed.
    """
    cfg = DIFFICULTY_CONFIGS[difficulty_name]
    grid_w = cfg["width"]
    grid_h = cfg["height"]
    cell_size = cfg["cell_size"]
    turn_limit = cfg["turn_limit"]
    density = cfg["density"]

    current_seed = seed
    free_cells, walls = generate_maze(seed=current_seed, density=density, width=grid_w, height=grid_h)
    env = GridEnvironment(
        width=grid_w,
        height=grid_h,
        walls=walls,
        sensor_sigma=sigma,
        seed=current_seed,
        turn_limit=turn_limit
    )
    obs = env.reset()

    # HMM agent (used for belief tracking in play mode and autonomous moves in demo mode)
    agent = HMMHunterAgent(free_cells, walls, sigma=sigma, grid_dims=(grid_h, grid_w))
    agent.update_belief(obs)

    # Resize window to fit difficulty scale
    gui.resize_for_difficulty(env, cell_size)

    last_ghost_move_time = time.time()
    last_demo_step_time = time.time()

    running = True
    while running:
        running, human_action, restart_requested, menu_requested, astar_requested = gui.handle_events()
        if not running:
            return "exit"
        if menu_requested:
            return "menu"

        # --- Restart with Brand New Maze ---
        if restart_requested:
            current_seed = random.randint(1000, 999999)
            free_cells, walls = generate_maze(seed=current_seed, density=density, width=grid_w, height=grid_h)
            env = GridEnvironment(
                width=grid_w,
                height=grid_h,
                walls=walls,
                sensor_sigma=sigma,
                seed=current_seed,
                turn_limit=turn_limit
            )
            obs = env.reset()
            agent = HMMHunterAgent(free_cells, walls, sigma=sigma, grid_dims=(grid_h, grid_w))
            agent.update_belief(obs)
            gui.env = env
            gui.astar_path = []
            gui.astar_active_until = 0.0
            last_ghost_move_time = time.time()
            continue

        now = time.time()

        # --- A* Radar Ability Trigger (X Key) ---
        if astar_requested and not env.captured:
            path_to_ghost = astar(
                start=env.hunter_pos,
                goal=env.true_ghost_pos,
                walls=env.walls,
                grid_dims=(env.height, env.width)
            )
            gui.trigger_astar_hint(path_to_ghost)

        # --- Real-Time Ghost Movement (every 1.0 second) ---
        if not gui.paused and not env.captured and env.turn < env.turn_limit:
            if now - last_ghost_move_time >= GHOST_MOVE_INTERVAL_SEC:
                obs, done, info = env.ghost_step_only()
                agent.update_belief(obs)
                last_ghost_move_time = now

        # --- Mode-Specific Action ---
        if mode == "play":
            # Human player directional move
            if human_action is not None and not env.captured and env.turn < env.turn_limit:
                obs, done, info = env.step(human_action)
                agent.update_belief(obs)
                last_ghost_move_time = time.time()
        elif mode == "demo":
            # Autonomous AI Hunter Step every 0.35s
            if not gui.paused and not env.captured and env.turn < env.turn_limit:
                if now - last_demo_step_time >= 0.35:
                    action = agent.select_action(obs)
                    obs, done, info = env.step(action)
                    last_demo_step_time = now
                    last_ghost_move_time = now

        # Render display
        gui.render(belief_grid=agent.tracker.get_belief_grid(), mode_label=f"{difficulty_name} ({mode.capitalize()})")
        gui.clock.tick(30)

    return "exit"


def interactive_launcher():
    """
    Main interactive entry point showing:
    1. Start Screen Menu:
       - 'Start' -> opens Difficulty Selection
       - 'Main Menu' -> opens Game Summary & Instructions Modal
       - 'Exit' -> closes application
    2. Difficulty Selection:
       - Beginner (10x10)
       - Intermediate (15x15)
       - Hard (20x20)
       - Extreme (30x30)
    3. Game Screen
    """
    dummy_free, dummy_walls = generate_maze(seed=42)
    dummy_env = GridEnvironment(walls=dummy_walls)
    gui = PygameVisualizer(dummy_env)

    while True:
        # Step 1: Always show the 3-button Start Screen first
        choice = gui.show_start_screen()

        if choice == "exit":
            break
        elif choice == "instructions":
            # Step 2A: Main Menu instructions & summary
            gui.show_instructions_screen()
        elif choice == "start":
            # Step 2B: Difficulty Selection Screen
            difficulty = gui.show_difficulty_selection()
            if difficulty is None:
                continue  # User clicked back to start screen

            # Step 3: Run Game Loop with selected grid size
            game_result = run_game_loop(gui, difficulty_name=difficulty, mode="play")
            if game_result == "exit":
                break
            # If 'menu', loop back to Start Screen

    gui.close()


def main():
    # If launched with no args, open the interactive Start Screen
    if len(sys.argv) == 1:
        interactive_launcher()
        return

    parser = argparse.ArgumentParser(
        description="Ghost Hunter: Tracking an Invisible Enemy with Hidden Markov Models"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="menu",
        choices=["menu", "play", "demo", "experiments"],
        help="Execution mode: 'menu' (Default interactive GUI), 'play', 'demo', or 'experiments'"
    )
    parser.add_argument("--difficulty", default="Beginner", choices=["Beginner", "Intermediate", "Hard", "Extreme"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for maze & simulation")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sensor noise standard deviation")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials per configuration in experiments")
    parser.add_argument("--topologies", type=int, default=5, help="Number of maze topologies in experiments")

    args = parser.parse_args()

    if args.mode == "menu":
        interactive_launcher()
    elif args.mode in ("play", "demo"):
        # Even with 'play', if no difficulty is passed, let user pick or play directly
        interactive_launcher()
    elif args.mode == "experiments":
        run_benchmark(num_trials=args.trials, topologies_count=args.topologies)


if __name__ == "__main__":
    main()
