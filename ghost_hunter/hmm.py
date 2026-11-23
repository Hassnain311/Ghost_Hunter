"""
Hidden Markov Model (HMM) Tracker for Ghost Hunter.
Implements the recursive Bayesian Forward Algorithm:
  1. Time Update (Prediction): b'_{t} = T^T b_{t-1}
  2. Measurement Update (Correction): b_{t} = alpha * L_t * b'_{t}
Maintains the dynamic 2D belief heatmap and entropy metrics.
"""

from typing import Optional
import numpy as np
from config import (
    GRID_WIDTH,
    GRID_HEIGHT,
    DEFAULT_SENSOR_SIGMA,
    PROBABILITY_FLOOR_EPSILON
)
from ghost_hunter.environment import get_ghost_transition_probabilities
from ghost_hunter.sensor_model import likelihood_by_distance


class HMMTracker:
    """
    Probabilistic state estimator for the invisible moving ghost.
    Strictly follows the Lab 09 Bayesian reasoning requirements.
    Written in vectorized NumPy without black-box ML/HMM libraries.
    """

    def __init__(
        self,
        free_cells: set[tuple[int, int]],
        walls: set[tuple[int, int]],
        grid_dims: tuple[int, int] = (GRID_HEIGHT, GRID_WIDTH),
        sigma: float = DEFAULT_SENSOR_SIGMA
    ):
        self.grid_dims = grid_dims
        self.height, self.width = grid_dims
        self.walls = set(walls)
        self.free_cells = set(free_cells)
        self.sigma = sigma
        
        # Build ordered indexing over free cells for matrix operations
        self.cell_list: list[tuple[int, int]] = sorted(list(self.free_cells))
        self.cell_to_idx: dict[tuple[int, int], int] = {cell: i for i, cell in enumerate(self.cell_list)}
        self.num_states = len(self.cell_list)
        
        self.rows, self.cols = np.asarray(self.cell_list).T
        self._distances = {}
        edges = []
        for i, src_cell in enumerate(self.cell_list):
            trans_dict = get_ghost_transition_probabilities(src_cell, self.free_cells)
            for dst_cell, prob in trans_dict.items():
                edges.append((i, self.cell_to_idx[dst_cell], prob))
        self._src, self._dst = np.asarray([(i, j) for i, j, _ in edges]).T
        self._uniform_weights = np.asarray([p for _, _, p in edges])
                
        # Vector of belief over free states
        self.belief_vec = np.zeros(self.num_states, dtype=float)
        self.reset()

    def reset(self, initial_prior: Optional[np.ndarray] = None) -> None:
        """
        Resets belief distribution to uniform across all unblocked cells
        (or to a designated prior if provided).
        """
        if initial_prior is not None:
            for i, cell in enumerate(self.cell_list):
                self.belief_vec[i] = initial_prior[cell[0], cell[1]]
            self._normalize_and_floor()
        else:
            self.belief_vec.fill(1.0 / self.num_states)

    @property
    def transition_matrix(self) -> np.ndarray:
        """Dense uniform walk matrix for inspection or comparison with the sparse update."""
        matrix = np.zeros((self.num_states, self.num_states))
        matrix[self._src, self._dst] = self._uniform_weights
        return matrix

    def distances_from(self, position) -> np.ndarray:
        if position not in self._distances:
            self._distances[position] = abs(self.rows - position[0]) + abs(self.cols - position[1])
        return self._distances[position]

    def forecast(self, steps: int, *hunter_positions: tuple[int, int]) -> np.ndarray:
        """Predict future ghost occupancy without changing the observed posterior."""
        weights = self._uniform_weights
        if hunter_positions:
            nearest = self.distances_from(hunter_positions[0])
            for position in hunter_positions[1:]:
                nearest = np.minimum(nearest, self.distances_from(position))
            # Each legal edge changes nearest-hunter distance by -1, 0 or 1.
            weights = np.exp2(nearest[self._dst] - nearest[self._src])
            totals = np.bincount(self._src, weights, minlength=self.num_states)
            weights = weights / totals[self._src]
        predicted = self.belief_vec.copy()
        for _ in range(steps):
            predicted = np.bincount(self._dst, weights * predicted[self._src], minlength=self.num_states)
        return predicted

    def time_update(self, *hunter_positions: tuple[int, int]) -> None:
        """Advance the filter by one actual ghost move."""
        self.belief_vec = self.forecast(1, *hunter_positions)
        self._normalize_and_floor()

    def observation_update(self, reading: int, observer_pos: tuple[int, int]) -> None:
        """
        Correction / Measurement Update Step:
        Multiplies prior belief by sensor likelihood:
        b_t(s) = alpha * P(reading | X=s, observer_pos) * b'_t(s)
        """
        likelihoods = likelihood_by_distance(reading, self.sigma, self.height + self.width - 2)
        likelihood_vec = likelihoods[self.distances_from(observer_pos)]
        
        # Pointwise multiplication
        self.belief_vec = self.belief_vec * likelihood_vec
        self._normalize_and_floor()

    def _normalize_and_floor(self) -> None:
        """
        Applies probability floor epsilon to prevent numerical zero-underflow,
        then normalizes the distribution so it sums strictly to 1.0.
        """
        # Apply epsilon floor to free cells
        self.belief_vec = np.maximum(self.belief_vec, PROBABILITY_FLOOR_EPSILON)
        total_mass = float(np.sum(self.belief_vec))
        if total_mass > 0.0:
            self.belief_vec /= total_mass
        else:
            self.belief_vec.fill(1.0 / self.num_states)

    def get_belief_grid(self) -> np.ndarray:
        """
        Returns full 2D matrix (10x10) representing P(X_t = (r, c)) for GUI rendering.
        Walls have exactly 0 probability.
        """
        grid = np.zeros(self.grid_dims, dtype=float)
        grid[self.rows, self.cols] = self.belief_vec
        return grid

    def get_map_estimate(self) -> tuple[int, int]:
        """
        Returns Maximum A Posteriori (MAP) cell coordinate:
        argmax_{(r, c)} P(X_t = (r, c) | e_{1:t})
        """
        best_idx = int(np.argmax(self.belief_vec))
        return self.cell_list[best_idx]

    def get_entropy(self) -> float:
        """
        Computes Shannon entropy of the belief state in bits:
        H(B) = -sum p_i * log2(p_i)
        """
        # Filter strictly positive probabilities
        p = self.belief_vec[self.belief_vec > 0.0]
        return float(-np.sum(p * np.log2(p)))
