"""
Grid Environment for Ghost Hunter.
Manages the maze layout, obstacle walls, ghost stochastic transitions,
hunter movement, noisy distance emissions, and terminal state checks.
"""

from typing import Any
from collections import deque
import numpy as np
from config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    TURN_LIMIT_T_MAX,
    DEFAULT_SENSOR_SIGMA,
    GHOST_ACTIONS
)
from ghost_hunter.sensor_model import manhattan_distance, sample_sensor_reading


def generate_maze(seed: int, density: float = 0.16) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Generates a deterministic 10x10 maze topology ensuring a single connected component
    for all free cells so that the ghost and hunter are never partitioned.
    
    Returns:
        free_cells: set of navigable (r, c) coordinates
        walls: set of impassable (r, c) coordinates
    """
    rng = np.random.default_rng(seed)
    all_cells = {(r, c) for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH)}
    
    # We never place walls on corners or designated center start positions
    protected_cells = {(0, 0), (GRID_HEIGHT - 1, GRID_WIDTH - 1), (0, GRID_WIDTH - 1), (GRID_HEIGHT - 1, 0), (4, 4), (4, 5), (5, 4), (5, 5)}
    candidate_cells = list(all_cells - protected_cells)
    
    walls: set[tuple[int, int]] = set()
    num_walls = int(len(candidate_cells) * density)
    shuffled_candidates = rng.permutation(candidate_cells)

    for cell in shuffled_candidates:
        candidate_tuple = (int(cell[0]), int(cell[1]))
        if len(walls) >= num_walls:
            break
        
        # Test if adding this wall maintains full connectivity of all free cells
        test_walls = walls | {candidate_tuple}
        test_free = all_cells - test_walls
        
        # Check connectivity via BFS from (0, 0)
        start = (0, 0)
        visited = set()
        queue = deque([start])
        visited.add(start)
        
        while queue:
            curr = queue.popleft()
            r, c = curr
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)
                if neighbor in test_free and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        if visited == test_free:
            walls.add(candidate_tuple)

    free_cells = all_cells - walls
    return free_cells, walls


def get_ghost_transition_probabilities(
    current_pos: tuple[int, int],
    free_cells: set[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """
    Shared transition model used by both GridEnvironment and HMMTracker.
    Ghost moves uniformly at random to valid adjacent free cells or stays in place.
    """
    r, c = current_pos
    valid_destinations = []
    
    for dr, dc in GHOST_ACTIONS:
        nxt = (r + dr, c + dc)
        if nxt in free_cells:
            valid_destinations.append(nxt)
            
    num_moves = len(valid_destinations)
    if num_moves == 0:
        return {current_pos: 1.0}
        
    prob = 1.0 / num_moves
    return {dest: prob for dest in valid_destinations}


class GridEnvironment:
    """
    The discrete, partially observable, stochastic grid environment.
    Follows Lab guidelines:
    Turn order:
      1. Hunter moves
      2. Capture check 1
      3. Ghost moves (if not captured)
      4. Capture check 2
      5. Noisy distance reading observed
    """

    def __init__(
        self,
        width: int = GRID_WIDTH,
        height: int = GRID_HEIGHT,
        walls: set[tuple[int, int]] = None,
        sensor_sigma: float = DEFAULT_SENSOR_SIGMA,
        seed: int = 42
    ):
        self.width = width
        self.height = height
        self.sensor_sigma = sensor_sigma
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        all_cells = {(r, c) for r in range(self.height) for c in range(self.width)}
        self.walls = set() if walls is None else set(walls)
        self.free_cells = all_cells - self.walls
        
        self.hunter_pos: tuple[int, int] = (0, 0)
        self.true_ghost_pos: tuple[int, int] = (0, 0)
        self.turn: int = 0
        self.captured: bool = False
        self.last_reading: int = 0

    def reset(
        self,
        hunter_start: tuple[int, int] = (0, 0),
        ghost_start: tuple[int, int] = None
    ) -> dict[str, Any]:
        """
        Resets environment to starting conditions and yields initial observation.
        """
        self.turn = 0
        self.captured = False
        
        # Place hunter
        if hunter_start in self.free_cells:
            self.hunter_pos = hunter_start
        else:
            self.hunter_pos = sorted(list(self.free_cells))[0]
            
        # Place ghost
        if ghost_start is not None and ghost_start in self.free_cells:
            self.true_ghost_pos = ghost_start
        else:
            free_list = sorted(list(self.free_cells - {self.hunter_pos}))
            chosen_idx = self.rng.integers(0, len(free_list))
            self.true_ghost_pos = free_list[chosen_idx]
            
        # Initial sensor reading
        dist = manhattan_distance(self.hunter_pos, self.true_ghost_pos)
        self.last_reading = sample_sensor_reading(dist, self.sensor_sigma, self.rng)
        
        return {
            "reading": self.last_reading,
            "hunter_pos": self.hunter_pos,
            "wall_hit": False,
            "turn": self.turn
        }

    def valid_moves(self, pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Returns list of valid adjacent (dr, dc) moves that do not enter walls or bounds."""
        r, c = pos
        valid = []
        for dr, dc in GHOST_ACTIONS:
            nr, nc = r + dr, c + dc
            if (nr, nc) in self.free_cells:
                valid.append((dr, dc))
        return valid

    def move_ghost(self) -> None:
        """Moves the ghost according to the shared transition model."""
        transitions = get_ghost_transition_probabilities(self.true_ghost_pos, self.free_cells)
        destinations = list(transitions.keys())
        probabilities = [transitions[d] for d in destinations]
        
        chosen_idx = self.rng.choice(len(destinations), p=probabilities)
        self.true_ghost_pos = destinations[chosen_idx]

    def step(self, action: tuple[int, int]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """
        Executes one full turn in the environment.
        Returns:
            observation: dict with sensor reading, hunter_pos, wall_hit
            done: bool (captured or turn limit reached)
            info: dict with capture status and true positions (for metrics logging)
        """
        self.turn += 1
        wall_hit = False
        
        # 1. Hunter move
        dr, dc = action
        target_pos = (self.hunter_pos[0] + dr, self.hunter_pos[1] + dc)
        if target_pos in self.free_cells:
            self.hunter_pos = target_pos
        else:
            wall_hit = True # Hunter stays in place and receives collision signal
            
        # 2. Capture check 1
        if self.hunter_pos == self.true_ghost_pos:
            self.captured = True
            done = True
            dist = 0
            self.last_reading = sample_sensor_reading(dist, self.sensor_sigma, self.rng)
            obs = {
                "reading": self.last_reading,
                "hunter_pos": self.hunter_pos,
                "wall_hit": wall_hit,
                "turn": self.turn
            }
            return obs, done, {"captured": True, "true_ghost_pos": self.true_ghost_pos}

        # 3. Ghost moves
        self.move_ghost()
        
        # 4. Capture check 2 (Ghost walked into hunter's cell)
        if self.hunter_pos == self.true_ghost_pos:
            self.captured = True
            done = True
            dist = 0
            self.last_reading = sample_sensor_reading(dist, self.sensor_sigma, self.rng)
            obs = {
                "reading": self.last_reading,
                "hunter_pos": self.hunter_pos,
                "wall_hit": wall_hit,
                "turn": self.turn
            }
            return obs, done, {"captured": True, "true_ghost_pos": self.true_ghost_pos}

        # Check turn limit
        done = (self.turn >= TURN_LIMIT_T_MAX)
        
        # 5. Emit noisy distance sensor reading
        dist = manhattan_distance(self.hunter_pos, self.true_ghost_pos)
        self.last_reading = sample_sensor_reading(dist, self.sensor_sigma, self.rng)
        
        obs = {
            "reading": self.last_reading,
            "hunter_pos": self.hunter_pos,
            "wall_hit": wall_hit,
            "turn": self.turn
        }
        info = {
            "captured": self.captured,
            "true_ghost_pos": self.true_ghost_pos
        }
        return obs, done, info
