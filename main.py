"""
Ghost Hunter Command Line Interface & Game Orchestrator.
Supports:
  1. Spooky Horror GUI with Fullscreen support
  2. Difficulty Selection: Beginner, Intermediate, Hard, Extreme
  3. Real-Time Ghost Movement scaling by difficulty
  4. Always-on Manhattan Proximity Heatmap (Red=1, Yellow=2/3, Green=4/5)
  5. Interactive Pause Menu Overlay triggered via [P] key
  6. Unlimited A* Hellfire Radar with counter
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
    Executes an active session with real-time ghost movement and A* radar capability.
    Returns:
        'menu' if user exits to main menu,
        'difficulty' if user selects change difficulty from pause menu,
        'exit' if window was closed.
    """
    cfg = DIFFICULTY_CONFIGS[difficulty_name]
    grid_w = cfg["width"]
    grid_h = cfg["height"]
    cell_size = cfg["cell_size"]
    turn_limit = cfg["turn_limit"]
    density = cfg["density"]
    ghost_speed = cfg.get("ghost_speed", GHOST_MOVE_INTERVAL_SEC)
    glimpse_interval = cfg.get("glimpse_interval", 8.0)
    glimpse_duration = cfg.get("glimpse_duration", 0.8)

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

    # Reset game session tracking
    gui.astar_use_count = 0
    gui.glimpse_interval = glimpse_interval
    gui.glimpse_duration = glimpse_duration
    gui.ghost_glimpse_until = 0.0
    gui.next_glimpse_time = time.time() + glimpse_interval
    gui.paused = False

    last_ghost_move_time = time.time()
    last_demo_step_time = time.time()

    running = True
    while running:
        running, human_action, restart_requested, menu_requested, astar_requested, pause_action = gui.handle_events()
        if not running:
            return "exit"
        if menu_requested or pause_action == "quit_to_menu":
            return "menu"
        if pause_action == "change_difficulty":
            return "difficulty"

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
            gui.astar_use_count = 0
            gui.ghost_glimpse_until = 0.0
            gui.next_glimpse_time = time.time() + glimpse_interval
            gui.paused = False
            last_ghost_move_time = time.time()
            continue

        now = time.time()

        # --- Periodic Automatic Ghost Glimpse (<= 1.0 sec) ---
        if not gui.paused and not env.captured:
            if now >= gui.next_glimpse_time:
                gui.ghost_glimpse_until = now + glimpse_duration
                gui.next_glimpse_time = now + glimpse_interval

        # --- A* Radar Ability Trigger (X Key) ---
        if astar_requested and not env.captured:
            path_to_ghost = astar(
                start=env.hunter_pos,
                goal=env.true_ghost_pos,
                walls=env.walls,
                grid_dims=(env.height, env.width)
            )
            gui.trigger_astar_hint(path_to_ghost)

        # --- Real-Time Ghost Movement (speed scales with difficulty) ---
        if not gui.paused and not env.captured and env.turn < env.turn_limit:
            if now - last_ghost_move_time >= ghost_speed:
                obs, done, info = env.ghost_step_only()
                agent.update_belief(obs)
                last_ghost_move_time = now

        # --- Dynamic A* Path Tracking (Follows moving ghost during active radar) ---
        if gui.is_astar_active() and not env.captured:
            path_to_ghost = astar(
                start=env.hunter_pos,
                goal=env.true_ghost_pos,
                walls=env.walls,
                grid_dims=(env.height, env.width)
            )
            gui.astar_path = path_to_ghost

        # --- Mode-Specific Action ---
        if mode == "play":
            # Human player directional move
            if human_action is not None and not gui.paused and not env.captured and env.turn < env.turn_limit:
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
    1. Start Screen Menu: Start, Main Menu (Grimoire), Exit
    2. Difficulty Selection: Beginner, Intermediate, Hard, Extreme
    3. Game Screen with live pause overlay
    """
    dummy_free, dummy_walls = generate_maze(seed=42)
    dummy_env = GridEnvironment(walls=dummy_walls)
    gui = PygameVisualizer(dummy_env)

    while True:
        choice = gui.show_start_screen()

        if choice == "exit":
            break
        elif choice == "instructions":
            gui.show_instructions_screen()
        elif choice == "start":
            # Difficulty loop
            while True:
                difficulty = gui.show_difficulty_selection()
                if difficulty is None:
                    break  # Return to Start Screen

                game_result = run_game_loop(gui, difficulty_name=difficulty, mode="play")
                if game_result == "exit":
                    gui.close()
                    return
                elif game_result == "difficulty":
                    continue  # Prompt difficulty selection again directly
                elif game_result == "menu":
                    break     # Return to Start Screen

    gui.close()


def main():
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
        help="Execution mode: 'menu', 'play', 'demo', or 'experiments'"
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
        interactive_launcher()
    elif args.mode == "experiments":
        run_benchmark(num_trials=args.trials, topologies_count=args.topologies)


if __name__ == "__main__":
    main()
