"""Reproducible 10-by-10 evaluation of the four proposed pursuit policies."""

import os
import csv
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import (
    NOISE_SIGMA_LEVELS,
    TURN_LIMIT_T_MAX,
    NUM_BENCHMARK_TRIALS, NUM_MAZE_TOPOLOGIES
)
from ghost_hunter.environment import GridEnvironment, generate_maze
from ghost_hunter.agents import BaseAgent, RandomAgent, NaiveSensorAgent, HMMHunterAgent
from ghost_hunter.metrics import EpisodeMetrics
from ghost_hunter.entropy_agent import EntropyHunterAgent


def run_episode(
    env: GridEnvironment,
    agent: BaseAgent,
    topology_id: int,
    sigma: float,
    seed: int
) -> EpisodeMetrics:
    """Executes a single test episode and records detailed metrics."""
    metrics = EpisodeMetrics(
        agent_name=agent.name,
        topology_id=topology_id,
        sigma=sigma,
        seed=seed
    )
    
    if isinstance(agent, RandomAgent):
        agent.reset(seed)
    else:
        agent.reset()
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs)
        if isinstance(agent, HMMHunterAgent):
            estimate = agent.tracker.get_map_estimate()
            entropy = agent.tracker.get_entropy()
        elif isinstance(agent, NaiveSensorAgent):
            estimate, entropy = agent.last_estimate, None
        else:
            estimate, entropy = None, None
        metrics.log_turn(estimate, env.true_ghost_pos, entropy)
        obs, done, info = env.step(action)

    metrics.captured = info.get("captured", False)
    return metrics


def run_benchmark(
    num_trials: int = NUM_BENCHMARK_TRIALS,
    topologies_count: int = NUM_MAZE_TOPOLOGIES,
    output_dir: str = "results"
) -> str:
    """Run matched seeded episodes on connected layouts from open to densely walled."""
    if num_trials < 1 or topologies_count < 1:
        raise ValueError("The number of trials and topologies must both be positive.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".benchmark-", dir=destination) as scratch:
        _collect_results(num_trials, topologies_count, Path(scratch))
        for folder in ("raw", "figures"):
            (destination / folder).mkdir(exist_ok=True)
            for generated in (Path(scratch) / folder).iterdir():
                generated.replace(destination / folder / generated.name)
    print(f"Saved the benchmark CSV and three figures to {destination}")
    return str(destination / "raw" / "benchmark_summary.csv")


def _collect_results(num_trials, topologies_count, output_dir):
    (output_dir / "raw").mkdir()
    (output_dir / "figures").mkdir()
    
    csv_path = os.path.join(output_dir, "raw", "benchmark_summary.csv")
    
    print(f"Starting Benchmark: {num_trials} trials across {topologies_count} topologies...")
    
    all_results = []
    
    # Pre-generate topologies to ensure all agents face identical mazes
    densities = np.linspace(0, .32, topologies_count) if topologies_count > 1 else [.16]
    topologies = [generate_maze(seed=1000 + i, density=density) for i, density in enumerate(densities)]
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent_name", "topology_id", "sigma", "trial", "seed",
            "captured", "total_steps", "mean_manhattan_error", "maze_seed", "wall_count"
        ])
        
        for top_id, (free_cells, walls) in enumerate(topologies):
            for sigma in NOISE_SIGMA_LEVELS:
                agents = [HMMHunterAgent(free_cells, walls, sigma=sigma),
                          EntropyHunterAgent(free_cells, walls, sigma=sigma),
                          NaiveSensorAgent(free_cells, walls), RandomAgent(free_cells, walls)]
                for trial in range(num_trials):
                    trial_seed = 42 + trial * topologies_count + top_id
                    for agent in agents:
                        # Instantiate identical environment per trial
                        env = GridEnvironment(
                            walls=walls,
                            sensor_sigma=sigma,
                            seed=trial_seed
                        )
                        
                        m = run_episode(env, agent, top_id, sigma, trial_seed)
                        mean_err = round(float(np.mean(m.manhattan_errors)), 3) if m.manhattan_errors else None
                        
                        writer.writerow([
                            m.agent_name, top_id, sigma, trial, trial_seed,
                            1 if m.captured else 0, m.total_steps, mean_err, 1000 + top_id, len(walls)
                        ])
                        all_results.append({
                            "agent": m.agent_name,
                            "sigma": sigma,
                            "captured": m.captured,
                            "steps": m.total_steps,
                            "errors": m.manhattan_errors
                        })
                        
    generate_charts(all_results, os.path.join(output_dir, "figures"))


def generate_charts(results: list[dict], figures_dir: str) -> None:
    """Generate the three evaluation figures from the recorded trial results."""
    agents = ("HMM Hunter", "Entropy Hunter", "Naive Sensor Baseline", "Random Baseline")
    palette = ("#126c70", "#7357a4", "#b06622", "#64748b")
    grouped = defaultdict(list)
    for row in results:
        grouped[row["agent"], row["sigma"]].append(row)
    sigmas = NOISE_SIGMA_LEVELS

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, filename), dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for index in (0, 2):
        name = agents[index]
        series = [r["errors"] for r in grouped[name, 1.0]]
        by_turn = [[errors[t] for errors in series if len(errors) > t] for t in range(30)]
        ax.plot(range(1, 31), [np.mean(v) if v else np.nan for v in by_turn],
                label=name, color=palette[index], linewidth=2)
    ax.set(title="Localization during active hunts (σ = 1.0)", xlabel="Turn",
           ylabel="Mean Manhattan error (cells)")
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    save(fig, "chart1_estimation_error_vs_turns.png")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(sigmas))
    width = .19
    for index, name in enumerate(agents):
        means = []
        for sigma in sigmas:
            catches = [r["steps"] for r in grouped[name, sigma] if r["captured"]]
            means.append(np.mean(catches) if catches else np.nan)
        ax.bar(x + (index - 1.5) * width, means, width, label=name, color=palette[index])
    ax.set(title="Turns to capture in successful hunts", xlabel="Sensor noise (σ)",
           ylabel="Mean turns among captures", xticks=x, xticklabels=sigmas)
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncols=2)
    save(fig, "chart2_steps_to_capture_by_sigma.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    for index, name in enumerate(agents):
        rates = [100 * np.mean([r["captured"] for r in grouped[name, sigma]]) for sigma in sigmas]
        ax.plot(sigmas, rates, marker=("o", "D", "s", "^")[index], linewidth=2,
                color=palette[index], label=name)
    ax.set(title=f"Captures within {TURN_LIMIT_T_MAX} turns", xlabel="Sensor noise (σ)",
           ylabel="Successful hunts (%)", ylim=(-2, 103), xticks=sigmas)
    ax.grid(axis="y", alpha=.25)
    ax.legend(ncols=2)
    save(fig, "chart3_success_rate_vs_noise.png")
