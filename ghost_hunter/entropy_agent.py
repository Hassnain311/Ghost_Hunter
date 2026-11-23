"""Optional information-seeking hunter from the project proposal."""

import numpy as np

from config import GHOST_ACTIONS
from ghost_hunter.agents import HMMHunterAgent
from ghost_hunter.sensor_model import likelihood_by_distance


def _entropy(probabilities, axis=None):
    return -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-300)), axis=axis)


class EntropyHunterAgent(HMMHunterAgent):
    """Seek a more informative ping while the posterior is diffuse, then chase the MAP."""

    def __init__(self, free_cells, walls, sigma=1.0, grid_dims=(10, 10)):
        super().__init__(free_cells, walls, sigma, grid_dims)
        self.name = "Entropy Hunter"
        self._max_distance = sum(grid_dims) - 2
        last_reading = self._max_distance + max(1, int(np.ceil(8 * sigma)))
        rows = [likelihood_by_distance(e, sigma, self._max_distance)
                for e in range(last_reading)]
        # Group the tiny remaining positive tail in the final reading bin.
        self._emissions = np.column_stack([*rows, np.maximum(0, 1 - np.sum(rows, axis=0))])
        self._noise_entropy = _entropy(self._emissions, axis=1)

    def action_from_belief(self, hunter_pos):
        belief = self.tracker.belief_vec
        if belief.max() >= .18:
            return super().action_from_belief(hunter_pos)
        best_score = -float("inf")
        for move in GHOST_ACTIONS:
            cell = hunter_pos[0] + move[0], hunter_pos[1] + move[1]
            if cell not in self.free_cells:
                continue
            distances = self.tracker.distances_from(cell)
            mass = np.bincount(distances, weights=belief, minlength=self._max_distance + 1)
            gain = _entropy(mass @ self._emissions) - mass @ self._noise_entropy
            score = gain - .15 * (belief @ distances)
            if score > best_score:
                best_score, chosen = score, move
        self.last_action = chosen
        return chosen
