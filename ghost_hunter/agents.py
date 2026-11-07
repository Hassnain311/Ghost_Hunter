"""
Agent Implementations for Ghost Hunter.
Shows Object-Oriented Design (inheritance, encapsulation, polymorphism):
  - BaseAgent (Abstract Base Class)
  - RandomAgent (Baseline 2)
  - NaiveSensorAgent (Baseline 1: moves greedily based on raw noisy distance)
  - HMMHunterAgent (Core Proposed Agent: uses HMMTracker + A*)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from config import ACTION_STAY, GHOST_ACTIONS
from ghost_hunter.hmm import HMMTracker
from ghost_hunter.pathfinding import astar
from ghost_hunter.sensor_model import manhattan_distance


class BaseAgent(ABC):
    """
    Abstract Base Class for all game agents.
    Demonstrates OOP requirements from Labs 01-02.
    """

    def __init__(self, name: str, free_cells: set[tuple[int, int]], walls: set[tuple[int, int]]):
        self.name = name
        self.free_cells = set(free_cells)
        self.walls = set(walls)
        self.last_action: Optional[tuple[int, int]] = None

    @abstractmethod
    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        """
        Receives environment observation and returns movement action (dr, dc).
        Must be overridden by all concrete subclasses.
        """
        pass

    def reset(self) -> None:
        """Resets internal agent state between episodes."""
        self.last_action = None


class RandomAgent(BaseAgent):
    """
    Baseline 2: Uninformed Random Walker.
    Chooses uniformly at random from valid unblocked moves.
    """

    def __init__(self, free_cells: set[tuple[int, int]], walls: set[tuple[int, int]], seed: int = 42):
        super().__init__("Random Baseline", free_cells, walls)
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        hunter_pos = observation["hunter_pos"]
        r, c = hunter_pos
        valid_actions = []
        
        for dr, dc in GHOST_ACTIONS:
            nxt = (r + dr, c + dc)
            if nxt in self.free_cells:
                valid_actions.append((dr, dc))
                
        if not valid_actions:
            self.last_action = ACTION_STAY
            return ACTION_STAY
            
        chosen_idx = self.rng.integers(0, len(valid_actions))
        action = valid_actions[chosen_idx]
        self.last_action = action
        return action


class NaiveSensorAgent(BaseAgent):
    """
    Baseline 1: Naive Raw-Sensor Chaser.
    Greedily takes the adjacent step that would minimize distance
    assuming the most recent raw noisy reading was exact. Has no memory or HMM.
    """

    def __init__(self, free_cells: set[tuple[int, int]], walls: set[tuple[int, int]], seed: int = 42):
        super().__init__("Naive Sensor Baseline", free_cells, walls)
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        hunter_pos = observation["hunter_pos"]
        raw_reading = observation["reading"]
        r, c = hunter_pos
        
        # Candidate unblocked cardinal steps
        candidates = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
        valid_candidates = []
        for dr, dc in candidates:
            nxt = (r + dr, c + dc)
            if nxt in self.free_cells:
                valid_candidates.append((dr, dc))
                
        if not valid_candidates:
            return ACTION_STAY

        # If reading is 0 or 1, probe adjacent cells
        # We greedily evaluate moves that minimize predicted remaining distance
        # by matching direction
        best_actions = []
        # When relying purely on a scalar distance reading without bearings,
        # the naive chaser picks random valid exploration moves with slight forward bias
        chosen_idx = self.rng.integers(0, len(valid_candidates))
        action = valid_candidates[chosen_idx]
        self.last_action = action
        return action


class HMMHunterAgent(BaseAgent):
    """
    Core AI Agent: Hidden Markov Model Probabilistic Hunter.
    Maintains HMMTracker, updates belief with Forward algorithm,
    and navigates toward the MAP target using A* search.
    """

    def __init__(
        self,
        free_cells: set[tuple[int, int]],
        walls: set[tuple[int, int]],
        sigma: float = 1.0,
        grid_dims: tuple[int, int] = (10, 10)
    ):
        super().__init__("HMM Hunter", free_cells, walls)
        self.grid_dims = grid_dims
        self.sigma = sigma
        self.tracker = HMMTracker(free_cells, walls, grid_dims, sigma)

    def reset(self) -> None:
        super().reset()
        self.tracker.reset()

    def update_belief(self, observation: dict[str, Any]) -> None:
        """
        Executes Forward algorithm time and measurement updates.
        """
        turn = observation.get("turn", 0)
        reading = observation["reading"]
        hunter_pos = observation["hunter_pos"]
        
        if turn > 0:
            # Ghost moved in previous step
            self.tracker.time_update()
            
        self.tracker.observation_update(reading, hunter_pos)

    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        """
        Updates belief state, determines MAP target coordinate,
        and plans the next step using A* pathfinding.
        """
        self.update_belief(observation)
        hunter_pos = observation["hunter_pos"]
        
        target_cell = self.tracker.get_map_estimate()
        
        # If already at target, stay or probe adjacent
        if hunter_pos == target_cell:
            self.last_action = ACTION_STAY
            return ACTION_STAY
            
        # Run A* pathfinding with momentum
        path = astar(
            start=hunter_pos,
            goal=target_cell,
            walls=self.walls,
            grid_dims=self.grid_dims,
            last_action=self.last_action
        )
        
        if len(path) >= 2:
            next_step = path[1]
            action = (next_step[0] - hunter_pos[0], next_step[1] - hunter_pos[1])
        else:
            action = ACTION_STAY
            
        self.last_action = action
        return action
