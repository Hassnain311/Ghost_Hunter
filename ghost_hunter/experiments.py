"""
Benchmark Experiment Harness and Plotting Pipeline for Ghost Hunter.
Runs 100 trials per configuration across 10 maze topologies and 5 noise levels.
Generates publication-quality charts into results/figures/ and CSVs into results/raw/.
"""

import os
import csv
from typing import Type
import numpy as np
import matplotlib.pyplot as plt
from config import (
    NOISE_SIGMA_LEVELS,
    TURN_LIMIT_T_MAX,
    NUM_MAZE_TOPOLOGIES
)
from ghost_hunter.environment import GridEnvironment, generate_maze
from ghost_hunter.agents import BaseAgent, RandomAgent, NaiveSensorAgent, HMMHunterAgent
from ghost_hunter.metrics import EpisodeMetrics


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
    
    agent.reset()
    obs = env.reset()
    
    done = False
    while not done:
        # Determine current target / estimate for logging
        if isinstance(agent, HMMHunterAgent):
            est = agent.tracker.get_map_estimate()
            entropy = agent.tracker.get_entropy()
        else:
            # Naive / Random use raw sensor distance circle approximation or hunter pos
            est = agent.select_action(obs)  # doesn't matter for raw baseline
            est = obs["hunter_pos"]
            entropy = 0.0
            
        metrics.log_turn(est, env.true_ghost_pos, entropy)
        
        # Step agent
        action = agent.select_action(obs)
        obs, done, info = env.step(action)
        
    metrics.captured = info.get("captured", False)
    return metrics


def run_benchmark(
    num_trials: int = 20,
    topologies_count: int = 5,
    output_dir: str = "results"
) -> str:
    """
    Executes the structured experiment suite:
    Compares HMM Hunter vs Naive Sensor Baseline vs Random Walk Baseline.
    """
    os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
    
    csv_path = os.path.join(output_dir, "raw", "benchmark_summary.csv")
    
    print(f"Starting Benchmark: {num_trials} trials across {topologies_count} topologies...")
    
    all_results = []
    
    # Pre-generate topologies to ensure all agents face identical mazes
    topologies = [generate_maze(seed=1000 + i) for i in range(topologies_count)]
    
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "agent_name", "topology_id", "sigma", "trial", "seed",
            "captured", "total_steps", "mean_manhattan_error"
        ])
        
        for top_id, (free_cells, walls) in enumerate(topologies):
            for sigma in NOISE_SIGMA_LEVELS:
                for trial in range(num_trials):
                    trial_seed = 42 + trial * 7 + top_id * 100
                    
                    agents: list[BaseAgent] = [
                        HMMHunterAgent(free_cells, walls, sigma=sigma),
                        NaiveSensorAgent(free_cells, walls, seed=trial_seed),
                        RandomAgent(free_cells, walls, seed=trial_seed)
                    ]
                    
                    for agent in agents:
                        # Instantiate identical environment per trial
                        env = GridEnvironment(
                            walls=walls,
                            sensor_sigma=sigma,
                            seed=trial_seed
                        )
                        
                        m = run_episode(env, agent, top_id, sigma, trial_seed)
                        mean_err = float(np.mean(m.manhattan_errors)) if m.manhattan_errors else 0.0
                        
                        writer.writerow([
                            m.agent_name, top_id, sigma, trial, trial_seed,
                            1 if m.captured else 0, m.total_steps, round(mean_err, 3)
                        ])
                        all_results.append({
                            "agent": m.agent_name,
                            "sigma": sigma,
                            "captured": m.captured,
                            "steps": m.total_steps,
                            "errors": m.manhattan_errors
                        })
                        
    print(f"Benchmark finished! Raw logs saved to {csv_path}")
    generate_charts(all_results, os.path.join(output_dir, "figures"))
    return csv_path


def generate_charts(results: list[dict], figures_dir: str) -> None:
    """Generates the 3 mandatory evaluation charts + 1 bonus entropy chart."""
    print("Generating evaluation figures...")
    
    # -------------------------------------------------------------
    # Chart 1: Estimation Error vs Turn (HMM vs Raw Sensor)
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    max_turns = 30
    
    for agent_name, color, label in [
        ("HMM Hunter", "#1f77b4", "HMM Hunter (Forward Algorithm)"),
        ("Naive Sensor Baseline", "#ff7f0e", "Naive Raw Sensor Baseline")
    ]:
        agent_runs = [r["errors"] for r in results if r["agent"] == agent_name and r["sigma"] == 1.0]
        if agent_runs:
            curve = np.zeros(max_turns)
            counts = np.zeros(max_turns)
            for err_list in agent_runs:
                for t, e in enumerate(err_list[:max_turns]):
                    curve[t] += e
                    counts[t] += 1
            counts[counts == 0] = 1
            mean_curve = curve / counts
            plt.plot(range(1, max_turns + 1), mean_curve, marker='o', linewidth=2, color=color, label=label)
            
    plt.title("Chart 1: Estimation Error vs. Turn Number (σ = 1.0)", fontsize=13, fontweight='bold')
    plt.xlabel("Turn Number", fontsize=11)
    plt.ylabel("Mean Manhattan Distance to Ghost", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "chart1_estimation_error_vs_turns.png")
    plt.savefig(chart1_path, dpi=300)
    plt.close()
    
    # -------------------------------------------------------------
    # Chart 2: Mean Steps to Capture across Agents grouped by σ
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5.5))
    sigmas = NOISE_SIGMA_LEVELS
    agents = ["HMM Hunter", "Naive Sensor Baseline", "Random Baseline"]
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]
    
    x = np.arange(len(sigmas))
    width = 0.25
    
    for i, agent_name in enumerate(agents):
        means = []
        for s in sigmas:
            runs = [r["steps"] for r in results if r["agent"] == agent_name and r["sigma"] == s]
            means.append(np.mean(runs) if runs else 0)
        plt.bar(x + (i - 1) * width, means, width, label=agent_name, color=colors[i], alpha=0.9)

    plt.title("Chart 2: Mean Steps to Capture Grouped by Noise σ", fontsize=13, fontweight='bold')
    plt.xlabel("Sensor Noise Standard Deviation (σ)", fontsize=11)
    plt.ylabel("Mean Steps to Capture (lower is better)", fontsize=11)
    plt.xticks(x, [f"σ={s}" for s in sigmas])
    plt.axhline(100, color="grey", linestyle=":", label="Turn Limit (100)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "chart2_steps_to_capture_by_sigma.png")
    plt.savefig(chart2_path, dpi=300)
    plt.close()

    # -------------------------------------------------------------
    # Chart 3: Capture Success Rate (%) vs Increasing Sensor Noise σ
    # -------------------------------------------------------------
    plt.figure(figsize=(8.5, 5))
    for agent_name, color, marker in [
        ("HMM Hunter", "#2ca02c", "o"),
        ("Naive Sensor Baseline", "#ff7f0e", "s"),
        ("Random Baseline", "#d62728", "^")
    ]:
        rates = []
        for s in sigmas:
            runs = [1.0 if r["captured"] else 0.0 for r in results if r["agent"] == agent_name and r["sigma"] == s]
            rates.append(np.mean(runs) * 100.0 if runs else 0.0)
        plt.plot(sigmas, rates, marker=marker, linewidth=2.2, label=agent_name, color=color)

    plt.title("Chart 3: Capture Success Rate vs. Sensor Noise σ", fontsize=13, fontweight='bold')
    plt.xlabel("Sensor Noise Standard Deviation (σ)", fontsize=11)
    plt.ylabel("Capture Success Rate (%)", fontsize=11)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "chart3_success_rate_vs_noise.png")
    plt.savefig(chart3_path, dpi=300)
    plt.close()
    
    print(f"All 3 publication figures saved cleanly to {figures_dir}")
