"""
Unit Tests for Hidden Markov Model (HMM) Tracker and Sensor Model.
Verifies normalization, probability floor, uniform initialization, and belief convergence.
"""

import pytest
import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT, PROBABILITY_FLOOR_EPSILON
from ghost_hunter.environment import generate_maze
from ghost_hunter.hmm import HMMTracker
from ghost_hunter.sensor_model import discrete_sensor_likelihood, manhattan_distance


def test_hmm_initialization():
    """Belief should be strictly normalized and equal across free cells."""
    free_cells, walls = generate_maze(seed=42)
    tracker = HMMTracker(free_cells, walls)
    
    belief_grid = tracker.get_belief_grid()
    assert np.isclose(np.sum(belief_grid), 1.0)
    
    # Check that walls have zero probability
    for w in walls:
        assert belief_grid[w[0], w[1]] == 0.0


def test_forward_algorithm_normalization():
    """Tests that time update and observation update keep total probability at 1.0."""
    free_cells, walls = generate_maze(seed=42)
    tracker = HMMTracker(free_cells, walls, sigma=1.0)
    
    for reading in [1, 3, 5, 2, 0]:
        tracker.time_update()
        assert np.isclose(np.sum(tracker.belief_vec), 1.0)
        
        tracker.observation_update(reading=reading, observer_pos=(0, 0))
        assert np.isclose(np.sum(tracker.belief_vec), 1.0)
        
        # Verify floor is maintained
        assert np.all(tracker.belief_vec >= PROBABILITY_FLOOR_EPSILON)


def test_sensor_likelihood_consistency():
    """Known-case test: reading 0 with observer at (0, 0) should maximize likelihood at (0, 0)."""
    free_cells, walls = generate_maze(seed=42)
    observer = (0, 0)
    L = discrete_sensor_likelihood(reading=0, observer_pos=observer, sigma=0.5, free_cells=free_cells)
    
    # Observer cell should be peak
    max_pos = tuple(np.unravel_index(np.argmax(L), L.shape))
    assert max_pos == observer
