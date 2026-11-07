"""
Discrete Sensor Model for Ghost Hunter.
Computes the likelihood P(reading | ghost_pos, observer_pos)
under discretized symmetric Gaussian/triangular distance sensor noise.
"""

from typing import Union
import numpy as np
from config import GRID_WIDTH, GRID_HEIGHT


def manhattan_distance(pos1: tuple[int, int], pos2: tuple[int, int]) -> int:
    """Computes Manhattan distance between two coordinates."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def discrete_noise_pmf(error: Union[int, np.ndarray], sigma: float) -> Union[float, np.ndarray]:
    """
    Computes the discrete probability mass of observing an integer sensor error.
    Modeled as an unnormalized discrete Gaussian envelope centered at 0 with standard deviation sigma:
    P(error) ~ exp(-error^2 / (2 * sigma^2)).
    """
    if sigma <= 1e-6:
        if isinstance(error, np.ndarray):
            return (error == 0).astype(float)
        return 1.0 if error == 0 else 0.0

    return np.exp(-(error ** 2) / (2.0 * (sigma ** 2)))


def sample_sensor_reading(
    true_dist: int,
    sigma: float,
    rng: np.random.Generator
) -> int:
    """
    Draws a noisy discrete distance measurement:
    reading = max(0, round(true_dist + noise))
    """
    if sigma <= 1e-6:
        return true_dist
    
    # Sample continuous Gaussian noise and round to nearest integer
    raw_noise = rng.normal(loc=0.0, scale=sigma)
    noisy_reading = int(round(true_dist + raw_noise))
    return max(0, noisy_reading)


def discrete_sensor_likelihood(
    reading: int,
    observer_pos: tuple[int, int],
    sigma: float,
    grid_dims: tuple[int, int] = (GRID_HEIGHT, GRID_WIDTH),
    free_cells: set[tuple[int, int]] = None
) -> np.ndarray:
    """
    Computes a 2D likelihood matrix of dimensions grid_dims,
    representing P(reading | X_ghost = (r, c), observer_pos).

    Accounts for the clipping at 0:
    If reading == 0, it includes the cumulative mass of all errors <= -true_dist.
    """
    height, width = grid_dims
    likelihood_grid = np.zeros((height, width), dtype=float)

    for r in range(height):
        for c in range(width):
            cell = (r, c)
            if free_cells is not None and cell not in free_cells:
                likelihood_grid[r, c] = 0.0
                continue

            true_dist = manhattan_distance(observer_pos, cell)

            if reading == 0:
                # Sum discrete probabilities for all errors where true_dist + err <= 0
                # err <= -true_dist
                cutoff = -true_dist
                # Evaluate tail from -15 to cutoff
                tail_errors = np.arange(-25, cutoff + 1)
                tail_probs = discrete_noise_pmf(tail_errors, sigma)
                likelihood_grid[r, c] = float(np.sum(tail_probs))
            else:
                error = reading - true_dist
                likelihood_grid[r, c] = float(discrete_noise_pmf(error, sigma))

    return likelihood_grid
