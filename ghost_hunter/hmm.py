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
from ghost_hunter.sensor_model import discrete_sensor_likelihood


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
        
        # Build Transition Matrix T where T[i, j] = P(X_t = j | X_{t-1} = i)
        self.transition_matrix = np.zeros((self.num_states, self.num_states), dtype=float)
        for i, src_cell in enumerate(self.cell_list):
            trans_dict = get_ghost_transition_probabilities(src_cell, self.free_cells)
            for dst_cell, prob in trans_dict.items():
                j = self.cell_to_idx[dst_cell]
                self.transition_matrix[i, j] = prob
                
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

    def time_update(self) -> None:
        """
        Prediction Step:
        Propagates belief vector through the ghost's stochastic transition model:
        b'_t = T^T * b_{t-1}
        """
        # (N, N)^T @ (N,) -> (N,)
        self.belief_vec = self.transition_matrix.T @ self.belief_vec
        self._normalize_and_floor()

    def observation_update(self, reading: int, observer_pos: tuple[int, int]) -> None:
        """
        Correction / Measurement Update Step:
        Multiplies prior belief by sensor likelihood:
        b_t(s) = alpha * P(reading | X=s, observer_pos) * b'_t(s)
        """
        likelihood_grid = discrete_sensor_likelihood(
            reading=reading,
            observer_pos=observer_pos,
            sigma=self.sigma,
            grid_dims=self.grid_dims,
            free_cells=self.free_cells
        )
        
        # Extract likelihoods for free cells
        likelihood_vec = np.array([likelihood_grid[cell[0], cell[1]] for cell in self.cell_list], dtype=float)
        
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
        for i, (r, c) in enumerate(self.cell_list):
            grid[r, c] = self.belief_vec[i]
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
