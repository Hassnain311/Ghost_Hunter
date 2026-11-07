"""
Unit Tests for A* Pathfinding and Obstacle Avoidance.
"""

import pytest
from ghost_hunter.environment import generate_maze
from ghost_hunter.pathfinding import astar, heuristic


def test_astar_finds_path_avoiding_walls():
    """A* path must not pass through any walls."""
    free_cells, walls = generate_maze(seed=42)
    start = (0, 0)
    goal = sorted(list(free_cells))[-1]
    
    path = astar(start, goal, walls)
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal
    
    # Check no cell in path is a wall
    for cell in path:
        assert cell not in walls
        
    # Check steps are adjacent
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dist = abs(r1 - r2) + abs(c1 - c2)
        assert dist == 1


def test_astar_same_start_and_goal():
    """Start equals goal should return [start]."""
    path = astar((2, 2), (2, 2), set())
    assert path == [(2, 2)]
