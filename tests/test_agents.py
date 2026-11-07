"""
Unit Tests for Agents.
Verifies that all agents return strictly valid, unblocked actions.
"""

import pytest
from ghost_hunter.environment import GridEnvironment, generate_maze
from ghost_hunter.agents import RandomAgent, NaiveSensorAgent, HMMHunterAgent


@pytest.mark.parametrize("agent_cls", [RandomAgent, NaiveSensorAgent, HMMHunterAgent])
def test_agent_actions_validity(agent_cls):
    """Checks that every agent produces valid moves that don't violate environment bounds."""
    free_cells, walls = generate_maze(seed=42)
    env = GridEnvironment(walls=walls, seed=42)
    obs = env.reset()
    
    if agent_cls == HMMHunterAgent:
        agent = agent_cls(free_cells, walls, sigma=1.0)
    else:
        agent = agent_cls(free_cells, walls, seed=42)
        
    for _ in range(15):
        action = agent.select_action(obs)
        assert isinstance(action, tuple)
        assert len(action) == 2
        obs, done, _ = env.step(action)
        assert obs["hunter_pos"] in free_cells
        if done:
            break
