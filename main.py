"""
Ghost Hunter Command Line Interface.
Supports 3 operational modes:
  1. play        - Interactive human play mode with optional live heatmap & restart
  2. demo        - Autonomous AI hunter demonstration with real-time probability heatmap & restart
  3. experiments - Automated benchmark harness generating CSVs and evaluation charts
"""

import sys
import argparse
import time
import random
from config import DEFAULT_SENSOR_SIGMA
from ghost_hunter.environment import GridEnvironment, generate_maze
from ghost_hunter.agents import HMMHunterAgent
from ghost_hunter.gui import PygameVisualizer
from ghost_hunter.experiments import run_benchmark


def play_mode(seed: int = 42, sigma: float = DEFAULT_SENSOR_SIGMA) -> None:
    """Human player navigates the grid using WASD/Arrow keys."""
    current_seed = seed
    free_cells, walls = generate_maze(seed=current_seed)
    env = GridEnvironment(walls=walls, sensor_sigma=sigma, seed=current_seed)
    obs = env.reset()
    
    hunter_agent = HMMHunterAgent(free_cells, walls, sigma=sigma)
    hunter_agent.update_belief(obs)
    
    gui = PygameVisualizer(env)
    print("\n--- GHOST HUNTER: PLAY MODE ---")
    print("Use W/A/S/D or Arrow keys to move.")
    print("Press [R] or click the button to RESTART with a brand new random maze layout.")
    print("Press [H] to toggle belief heatmap, [G] to reveal ghost, [ESC] to quit.")
    
    running = True
    while running:
        running, human_action, restart_requested = gui.handle_events()
        if not running:
            break
            
        if restart_requested:
            # Generate brand new maze topology with a new seed
            current_seed = random.randint(1000, 999999)
            free_cells, walls = generate_maze(seed=current_seed)
            env = GridEnvironment(walls=walls, sensor_sigma=sigma, seed=current_seed)
            gui.env = env
            obs = env.reset()
            hunter_agent = HMMHunterAgent(free_cells, walls, sigma=sigma)
            hunter_agent.update_belief(obs)
            continue
            
        if human_action is not None and not env.captured and env.turn < 100:
            obs, done, info = env.step(human_action)
            hunter_agent.update_belief(obs)
            
        gui.render(belief_grid=hunter_agent.tracker.get_belief_grid(), mode_label="Play")
        gui.clock.tick(30)
        
    gui.close()


def demo_mode(seed: int = 101, sigma: float = DEFAULT_SENSOR_SIGMA, step_delay: float = 0.35) -> None:
    """Autonomous HMM Hunter agent demonstration with animated heatmap."""
    current_seed = seed
    free_cells, walls = generate_maze(seed=current_seed)
    env = GridEnvironment(walls=walls, sensor_sigma=sigma, seed=current_seed)
    obs = env.reset()
    
    agent = HMMHunterAgent(free_cells, walls, sigma=sigma)
    gui = PygameVisualizer(env)
    
    print("\n--- GHOST HUNTER: AUTONOMOUS AI DEMO ---")
    print("Watch the HMM probability heatmap converge on the invisible ghost.")
    print("Press [R] or click the button to RESTART with a brand new random maze layout.")
    print("Press [Space] to pause/unpause, [ESC] to quit.")
    
    running = True
    last_step_time = time.time()
    
    while running:
        running, _, restart_requested = gui.handle_events()
        if not running:
            break
            
        if restart_requested:
            # Generate brand new maze topology with a new seed
            current_seed = random.randint(1000, 999999)
            free_cells, walls = generate_maze(seed=current_seed)
            env = GridEnvironment(walls=walls, sensor_sigma=sigma, seed=current_seed)
            gui.env = env
            obs = env.reset()
            agent = HMMHunterAgent(free_cells, walls, sigma=sigma)
            last_step_time = time.time()
            continue
            
        current_time = time.time()
        if not gui.paused and not env.captured and env.turn < 100:
            if current_time - last_step_time >= step_delay:
                action = agent.select_action(obs)
                obs, done, info = env.step(action)
                last_step_time = current_time
                
        gui.render(belief_grid=agent.tracker.get_belief_grid(), mode_label="Demo")
        gui.clock.tick(30)
        
    gui.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ghost Hunter: Tracking an Invisible Enemy with Hidden Markov Models"
    )
    parser.add_argument(
        "mode",
        choices=["play", "demo", "experiments"],
        help="Execution mode: 'play' for human control, 'demo' for autonomous AI, 'experiments' for benchmark"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for maze & simulation")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sensor noise standard deviation")
    parser.add_argument("--trials", type=int, default=20, help="Number of trials per configuration in experiments")
    parser.add_argument("--topologies", type=int, default=5, help="Number of maze topologies in experiments")
    
    args = parser.parse_args()
    
    if args.mode == "play":
        play_mode(seed=args.seed, sigma=args.sigma)
    elif args.mode == "demo":
        demo_mode(seed=args.seed, sigma=args.sigma)
    elif args.mode == "experiments":
        run_benchmark(num_trials=args.trials, topologies_count=args.topologies)


if __name__ == "__main__":
    main()
