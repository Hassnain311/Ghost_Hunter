"""
Grid Environment for Ghost Hunter.
Manages the maze layout, obstacle walls, evasive/stochastic ghost transitions,
hunter movement, noisy distance emissions, and terminal state checks.
Supports configurable grid dimensions (Beginner 10x10, Intermediate 15x15,
Hard 20x20, Extreme 30x30).
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


def generate_maze(
    seed: int,
    density: float = 0.16,
    width: int = GRID_WIDTH,
    height: int = GRID_HEIGHT
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """
    Generates a deterministic maze topology ensuring a single connected component
    for all free cells so that the ghost and hunter are never partitioned.
    Works for any grid dimensions (10x10, 15x15, 20x20, 30x30).
    
    Returns:
        free_cells: set of navigable (r, c) coordinates
        walls: set of impassable (r, c) coordinates
    """
    rng = np.random.default_rng(seed)
    all_cells = {(r, c) for r in range(height) for c in range(width)}
    
    mid_r, mid_c = height // 2, width // 2
    protected_cells = {
        (0, 0),
        (height - 1, width - 1),
        (0, width - 1),
        (height - 1, 0),
        (mid_r, mid_c),
        (mid_r, max(0, mid_c - 1)),
        (max(0, mid_r - 1), mid_c),
        (max(0, mid_r - 1), max(0, mid_c - 1))
    }
    candidate_cells = list(all_cells - protected_cells)
    
    walls: set[tuple[int, int]] = set()
    num_walls = int(len(candidate_cells) * density)
    shuffled_candidates = rng.permutation(candidate_cells)

    for cell in shuffled_candidates:
        candidate_tuple = (int(cell[0]), int(cell[1]))
        if len(walls) >= num_walls:
            break
        
        test_walls = walls | {candidate_tuple}
        test_free = all_cells - test_walls
        
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
    free_cells: set[tuple[int, int]],
    hunter_pos: tuple[int, int] = None
) -> dict[tuple[int, int], float]:
    """
    Ghost transition model with smart evasive bias:
    - Moves to adjacent valid free cells or stays.
    - When hunter is near (<= 3 blocks), ghost biases slightly away from the hunter,
      making the ghost significantly more difficult to corner and catch!
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

    # If no hunter pos provided, uniform random
    if hunter_pos is None:
        prob = 1.0 / num_moves
        return {dest: prob for dest in valid_destinations}

    # Evasive weighting: destinations that increase or maintain distance from hunter get higher weight
    weights = []
    current_dist = manhattan_distance(current_pos, hunter_pos)
    for dest in valid_destinations:
        d = manhattan_distance(dest, hunter_pos)
        if d > current_dist:
            weights.append(2.0)  # Flee bias
        elif d == current_dist:
            weights.append(1.0)
        else:
            weights.append(0.5)  # Moving toward hunter is penalized

    total_w = sum(weights)
    return {dest: weights[i] / total_w for i, dest in enumerate(valid_destinations)}


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
        seed: int = 42,
        turn_limit: int = TURN_LIMIT_T_MAX
    ):
        self.width = width
        self.height = height
        self.sensor_sigma = sensor_sigma
        self.seed = seed
        self.turn_limit = turn_limit
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
        
        if hunter_start in self.free_cells:
            self.hunter_pos = hunter_start
        else:
            self.hunter_pos = sorted(list(self.free_cells))[0]
            
        if ghost_start is not None and ghost_start in self.free_cells:
            self.true_ghost_pos = ghost_start
        else:
            free_list = sorted(list(self.free_cells - {self.hunter_pos}))
            chosen_idx = self.rng.integers(0, len(free_list))
            self.true_ghost_pos = free_list[chosen_idx]
            
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
        """Moves the ghost with evasive intelligence against the hunter."""
        transitions = get_ghost_transition_probabilities(self.true_ghost_pos, self.free_cells, self.hunter_pos)
        destinations = list(transitions.keys())
        probabilities = [transitions[d] for d in destinations]
        
        chosen_idx = self.rng.choice(len(destinations), p=probabilities)
        self.true_ghost_pos = destinations[chosen_idx]

    def ghost_step_only(self) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """
        Called when ghost moves in real-time (every 1 second) independent of hunter step.
        """
        if self.captured:
            return {
                "reading": self.last_reading,
                "hunter_pos": self.hunter_pos,
                "wall_hit": False,
                "turn": self.turn
            }, True, {"captured": True, "true_ghost_pos": self.true_ghost_pos}

        self.move_ghost()

        # Capture check: did ghost walk into hunter?
        if self.hunter_pos == self.true_ghost_pos:
            self.captured = True
            done = True
            dist = 0
            self.last_reading = sample_sensor_reading(dist, self.sensor_sigma, self.rng)
            obs = {
                "reading": self.last_reading,
                "hunter_pos": self.hunter_pos,
                "wall_hit": False,
                "turn": self.turn
            }
            return obs, done, {"captured": True, "true_ghost_pos": self.true_ghost_pos}

        dist = manhattan_distance(self.hunter_pos, self.true_ghost_pos)
        self.last_reading = sample_sensor_reading(dist, self.sensor_sigma, self.rng)
        obs = {
            "reading": self.last_reading,
            "hunter_pos": self.hunter_pos,
            "wall_hit": False,
            "turn": self.turn
        }
        return obs, False, {"captured": False, "true_ghost_pos": self.true_ghost_pos}

    def step(self, action: tuple[int, int]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """
        Executes one full turn in the environment.
        """
        self.turn += 1
        wall_hit = False
        
        # 1. Hunter move
        dr, dc = action
        target_pos = (self.hunter_pos[0] + dr, self.hunter_pos[1] + dc)
        if target_pos in self.free_cells:
            self.hunter_pos = target_pos
        else:
            wall_hit = True
            
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

        # 3. Ghost moves (evasively)
        self.move_ghost()
        
        # 4. Capture check 2
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

        done = (self.turn >= self.turn_limit)
        
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
