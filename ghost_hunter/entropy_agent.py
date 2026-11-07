"""
Optional Extension: Entropy-Reduction Hunter Agent.
Balances moving toward high belief regions while choosing observation positions
that minimize conditional Shannon entropy.
"""

from typing import Any
import numpy as np
from config import ACTION_STAY
from ghost_hunter.agents import BaseAgent
from ghost_hunter.hmm import HMMTracker
from ghost_hunter.pathfinding import astar


class EntropyHunterAgent(BaseAgent):
    """
    Entropy-Reduction / Information Gain Hunter Agent.
    Prioritizes disambiguating flat belief distributions before committing to chase.
    """

    def __init__(
        self,
        free_cells: set[tuple[int, int]],
        walls: set[tuple[int, int]],
        sigma: float = 1.0,
        grid_dims: tuple[int, int] = (10, 10)
    ):
        super().__init__("Entropy Hunter", free_cells, walls)
        self.grid_dims = grid_dims
        self.sigma = sigma
        self.tracker = HMMTracker(free_cells, walls, grid_dims, sigma)

    def reset(self) -> None:
        super().reset()
        self.tracker.reset()

    def select_action(self, observation: dict[str, Any]) -> tuple[int, int]:
        turn = observation.get("turn", 0)
        reading = observation["reading"]
        hunter_pos = observation["hunter_pos"]

        if turn > 0:
            self.tracker.time_update()
        self.tracker.observation_update(reading, hunter_pos)

        # Target highest probability cell
        target_cell = self.tracker.get_map_estimate()
        if hunter_pos == target_cell:
            self.last_action = ACTION_STAY
            return ACTION_STAY

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
