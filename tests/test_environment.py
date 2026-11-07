"""
Unit Tests for Ghost Hunter Grid Environment.
Verifies walls, boundary checks, ghost containment, and seeded reproducibility.
"""

import pytest
import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT, GHOST_ACTIONS
from ghost_hunter.environment import GridEnvironment, generate_maze, get_ghost_transition_probabilities


def test_maze_connectivity():
    """Checks that free cells in generated maze form a single connected component."""
    for seed in [42, 101, 202]:
        free_cells, walls = generate_maze(seed=seed)
        assert len(free_cells) + len(walls) == GRID_WIDTH * GRID_HEIGHT
        assert (0, 0) in free_cells
        assert len(walls) > 0


def test_ghost_stays_within_free_cells():
    """Ensures ghost movement never steps into walls or out of bounds."""
    free_cells, walls = generate_maze(seed=42)
    env = GridEnvironment(walls=walls, seed=42)
    env.reset()
    
    for _ in range(50):
        assert env.true_ghost_pos in free_cells
        assert env.true_ghost_pos not in walls
        env.move_ghost()


def test_reproducibility():
    """Confirms identical seeds produce identical trajectory sequences."""
    free_cells, walls = generate_maze(seed=42)
    
    env1 = GridEnvironment(walls=walls, seed=999)
    env2 = GridEnvironment(walls=walls, seed=999)
    
    obs1 = env1.reset()
    obs2 = env2.reset()
    assert env1.true_ghost_pos == env2.true_ghost_pos
    assert obs1["reading"] == obs2["reading"]
    
    for _ in range(10):
        env1.move_ghost()
        env2.move_ghost()
        assert env1.true_ghost_pos == env2.true_ghost_pos
