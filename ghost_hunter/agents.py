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

    def reset(self, seed=None) -> None:
        super().reset()
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def valid_actions(self, position):
        r, c = position
        return [move for move in GHOST_ACTIONS if (r + move[0], c + move[1]) in self.free_cells]

    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        actions = self.valid_actions(observation["hunter_pos"])
        self.last_action = actions[self.rng.integers(len(actions))] if actions else ACTION_STAY
        return self.last_action


class NaiveSensorAgent(RandomAgent):
    """Estimate a cell on the current distance ring, then take a greedy legal step."""

    def __init__(self, free_cells, walls, seed=42):
        super().__init__(free_cells, walls, seed)
        self.name = "Naive Sensor Baseline"
        self.last_estimate = None
        self._rings = {}

    def reset(self, seed=None) -> None:
        super().reset(seed)
        self.last_estimate = None

    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        position = observation["hunter_pos"]
        if position not in self._rings:
            rings = {}
            for cell in sorted(self.free_cells):
                rings.setdefault(manhattan_distance(position, cell), []).append(cell)
            self._rings[position] = rings
        rings = self._rings[position]
        reading = observation["reading"]
        distance = reading if reading in rings else min(rings, key=lambda d: abs(d - reading))
        candidates = rings[distance]
        self.last_estimate = candidates[self.rng.integers(len(candidates))]
        actions = self.valid_actions(position)
        costs = [manhattan_distance((position[0] + dr, position[1] + dc), self.last_estimate)
                 for dr, dc in actions]
        best = min(costs)
        choices = [move for move, cost in zip(actions, costs) if cost == best]
        self.last_action = choices[self.rng.integers(len(choices))]
        return self.last_action


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
            self.tracker.time_update(hunter_pos)
            
        self.tracker.observation_update(reading, hunter_pos)

    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        """
        Updates belief state, determines MAP target coordinate,
        and plans the next step using A* pathfinding.
        """
        self.update_belief(observation)
        return self.action_from_belief(observation["hunter_pos"])

    def action_from_belief(self, hunter_pos: tuple[int, int]) -> tuple[int, int]:
        """Plan from the current posterior without applying the same ping twice."""
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
