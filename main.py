"""Launch Ghost Hunter's playable modes, autonomous demonstrations and offline experiments."""

import sys
import sqlite3
import argparse
import time
import random
import pygame
from config import (
    DEFAULT_SENSOR_SIGMA,
    DIFFICULTY_CONFIGS,
    NUM_BENCHMARK_TRIALS, NUM_MAZE_TOPOLOGIES,
    GHOST_MOVE_INTERVAL_SEC
)
from ghost_hunter.environment import GridEnvironment, generate_maze
from ghost_hunter.agents import HMMHunterAgent
from ghost_hunter.entropy_agent import EntropyHunterAgent
from ghost_hunter.gui import PygameVisualizer
from ghost_hunter.pathfinding import astar
from ghost_hunter.instance import interactive_instance
from ghost_hunter.competitive import run_race
from ghost_hunter.leaderboard import Leaderboard
from ghost_hunter.mode_screens import CloseGame, choose_mode, enter_names, show_capture_result, show_leaderboard


def run_game_loop(
    gui: PygameVisualizer,
    difficulty_name: str = "Beginner",
    mode: str = "play",
    seed: int = 42,
    sigma: float = DEFAULT_SENSOR_SIGMA,
    *, player_name=None, save_score=None
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
    autonomous = mode in ("demo", "entropy")
    agent_class = EntropyHunterAgent if mode == "entropy" else HMMHunterAgent
    agent = agent_class(free_cells, walls, sigma=sigma, grid_dims=(grid_h, grid_w))
    mode_title = {"play": "Solo", "demo": "AI Demo", "entropy": "Entropy Demo"}[mode]
    hunt_label = f"{difficulty_name} / {mode_title}"
    agent.update_belief(obs)

    # Resize window to fit difficulty scale
    gui.resize_for_difficulty(env, cell_size)
    gui.blocked_until = 0
    gui.audio.begin_hunt()

    # Reset game session tracking
    gui.astar_use_count = 0
    gui.astar_path = []
    gui.astar_active_until = gui.astar_cooldown_until = 0
    gui.show_probability = autonomous
    gui.reveal_ghost = False
    gui.glimpse_interval = glimpse_interval
    gui.glimpse_duration = glimpse_duration
    gui.ghost_glimpse_until = 0.0
    gui.next_glimpse_time = time.time() + glimpse_interval
    gui.paused = False

    last_ghost_move_time = time.time()
    last_demo_step_time = time.time()

    elapsed = 0.0
    last_frame = time.monotonic()
    saved = False
    save_error = None
    replay = False
    running = True
    while running:
        frame = time.monotonic()
        if not gui.paused and not env.captured and env.turn < env.turn_limit:
            elapsed += frame - last_frame
        last_frame = frame
        if replay:
            running, human_action, restart_requested, menu_requested, astar_requested, pause_action = (
                True, None, True, False, False, None
            )
            replay = False
        else:
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
            agent = agent_class(free_cells, walls, sigma=sigma, grid_dims=(grid_h, grid_w))
            agent.update_belief(obs)
            gui.env = env
            gui.blocked_until = 0
            gui.audio.begin_hunt()
            gui.astar_path = []
            gui.astar_active_until = 0.0
            gui.astar_cooldown_until = 0.0
            gui.astar_use_count = 0
            gui.ghost_glimpse_until = 0.0
            gui.next_glimpse_time = time.time() + glimpse_interval
            gui.paused = False
            last_ghost_move_time = time.time()
            last_demo_step_time = last_ghost_move_time
            elapsed = 0.0
            last_frame = time.monotonic()
            saved = False
            save_error = None
            continue

        now = time.time()

        # --- Periodic Automatic Ghost Glimpse (<= 1.0 sec) ---
        if mode == "play" and not gui.paused and not env.captured:
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
        if mode == "play" and not gui.paused and not env.captured and env.turn < env.turn_limit:
            if now - last_ghost_move_time >= ghost_speed:
                obs, done, info = env.ghost_step_only()
                # A timer tick moves the ghost even before the hunter's first turn.
                agent.tracker.time_update(env.hunter_pos)
                agent.tracker.observation_update(obs["reading"], obs["hunter_pos"])
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
        previous_position = env.hunter_pos
        wall_hit = False
        if mode == "play":
            # Human player directional move
            if human_action is not None and not gui.paused and not env.captured and env.turn < env.turn_limit:
                obs, done, info = env.step(human_action)
                wall_hit = obs["wall_hit"]
                agent.update_belief(obs)
                last_ghost_move_time = time.time()
        elif autonomous:
            # Autonomous AI Hunter Step every 0.35s
            if not gui.paused and not env.captured and env.turn < env.turn_limit:
                if now - last_demo_step_time >= 0.35:
                    action = agent.action_from_belief(obs["hunter_pos"])
                    obs, done, info = env.step(action)
                    agent.update_belief(obs)
                    last_demo_step_time = now
                    last_ghost_move_time = now

        if env.captured or env.turn >= env.turn_limit:
            gui.audio.finish(captured=env.captured)
        elif wall_hit:
            gui.wall_collision(player_name or "Hunter")
        elif env.hunter_pos != previous_position:
            gui.audio.footstep()

        if env.captured and not saved:
            if save_score is not None:
                save_error = save_score((player_name,), elapsed)
            saved = True
            result = show_capture_result(
                gui, (player_name or "Hunter",), elapsed, save_error,
                draw_capture=lambda: gui.render(
                    agent.tracker.get_belief_grid(), hunt_label,
                    player_name=player_name, active_time=elapsed if player_name else None, autonomous=autonomous,
                ),
            )
            if result != "restart":
                return result
            replay = True
            continue

        if env.turn >= env.turn_limit:
            result = show_capture_result(gui, (), elapsed, step_limit=env.turn_limit)
            if result != "restart":
                return result
            replay = True
            continue

        # Render display
        gui.render(belief_grid=agent.tracker.get_belief_grid(), mode_label=hunt_label,
                   player_name=player_name, save_error=save_error,
                   active_time=elapsed if player_name else None, autonomous=autonomous)
        gui.clock.tick(30)

    return "exit"


def _close_menu():
    raise CloseGame


def interactive_launcher(*, demo_difficulty=None, demo_mode="demo", seed=42, sigma=DEFAULT_SENSOR_SIGMA):
    with interactive_instance() as acquired:
        if not acquired:
            print("Ghost Hunter is already open.")
            return
        try:
            _interactive_session(demo_difficulty, demo_mode, seed, sigma)
        finally:
            # Also release SDL if construction of the first screen was interrupted.
            if pygame.get_init():
                pygame.quit()


def _interactive_session(demo_difficulty, demo_mode, seed, sigma):
    _, walls = generate_maze(seed=42)
    gui = PygameVisualizer(GridEnvironment(walls=walls))
    leaderboard = Leaderboard()

    def save_times(mode, difficulty, captures):
        try:
            leaderboard.record_times(mode, difficulty, captures)
        except (sqlite3.Error, OSError):
            return "Score not saved"
        return None

    requested_demo = demo_difficulty is not None
    try:
        while True:
            choice = demo_mode if requested_demo else gui.show_start_screen()
            requested_demo = False
            if choice == "exit":
                return
            if choice == "instructions":
                choice = gui.show_instructions_screen()
            if choice in ("demo", "entropy"):
                level = demo_difficulty or "Beginner"
                while level is not None:
                    name = "HMM Hunter" if choice == "demo" else "Entropy Hunter"
                    result = run_game_loop(gui, level, choice, seed, sigma, player_name=name)
                    if result == "exit":
                        return
                    if result != "difficulty":
                        break
                    level = gui.show_difficulty_selection(on_close=_close_menu, demo=True)
            elif choice == "leaderboard":
                show_leaderboard(gui, leaderboard)
            elif choice == "start":
                mode = choose_mode(gui)
                while mode is not None:
                    difficulty = gui.show_difficulty_selection(on_close=_close_menu)
                    if difficulty is None:
                        mode = choose_mode(gui)
                        continue
                    try:
                        last_names = leaderboard.last_names(mode)
                    except (sqlite3.Error, OSError):
                        last_names = ()
                    names = enter_names(gui, mode, last_names)
                    if names is None:
                        continue
                    try:
                        leaderboard.remember_names(mode, names)
                    except (sqlite3.Error, OSError):
                        pass  # Local storage should not prevent playing a round.
                    while True:
                        if mode == "Solo":
                            def save_score(winners, elapsed):
                                return save_times(mode, difficulty, [(name, elapsed) for name in winners])

                            result = run_game_loop(gui, difficulty, "play", seed, sigma,
                                                   player_name=names[0], save_score=save_score)
                        else:
                            competitors = (*names, "AI") if mode == "AI vs Human" else names
                            result = run_race(gui, difficulty, mode, competitors,
                                              lambda captures: save_times(mode, difficulty, captures))
                        if result == "exit":
                            return
                        if result == "menu":
                            mode = None
                            break
                        difficulty = gui.show_difficulty_selection(on_close=_close_menu)
                        if difficulty is None:
                            break
    except CloseGame:
        return
    finally:
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
        choices=["menu", "play", "demo", "entropy", "experiments"],
        help="Open the game, watch a demonstration or run the benchmarks"
    )
    parser.add_argument("--difficulty", default="Beginner", choices=["Beginner", "Intermediate", "Hard", "Extreme"])
    parser.add_argument("--seed", type=int, default=42, help="Random seed for maze & simulation")
    parser.add_argument("--sigma", type=float, default=DEFAULT_SENSOR_SIGMA, help="Sensor noise standard deviation")
    parser.add_argument("--trials", type=int, default=NUM_BENCHMARK_TRIALS, help="Number of trials per configuration in experiments")
    parser.add_argument("--topologies", type=int, default=NUM_MAZE_TOPOLOGIES, help="Number of maze topologies in experiments")
    parser.add_argument("--output", default="results", help="Benchmark output directory")

    args = parser.parse_args()

    if args.mode in ("menu", "play"):
        interactive_launcher(seed=args.seed, sigma=args.sigma)
    elif args.mode in ("demo", "entropy"):
        interactive_launcher(demo_difficulty=args.difficulty, demo_mode=args.mode, seed=args.seed, sigma=args.sigma)
    elif args.mode == "experiments":
        from ghost_hunter.experiments import run_benchmark
        run_benchmark(num_trials=args.trials, topologies_count=args.topologies, output_dir=args.output)


if __name__ == "__main__":
    main()
