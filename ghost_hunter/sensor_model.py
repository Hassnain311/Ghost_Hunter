"""
Discrete Sensor Model for Ghost Hunter.
Computes the likelihood P(reading | ghost_pos, observer_pos)
under discretized symmetric Gaussian/triangular distance sensor noise.
"""

from typing import Union
from functools import lru_cache
from math import erfc, sqrt
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


@lru_cache(maxsize=1024)
def likelihood_by_distance(reading: int, sigma: float, max_distance: int) -> np.ndarray:
    """P(reading | distance) for the same rounded, zero-clipped Gaussian sampled above."""
    distances = range(max_distance + 1)
    if sigma <= 1e-6:
        values = np.arange(max_distance + 1) == reading
    elif reading == 0:
        scale = sqrt(2) * sigma
        values = [.5 * erfc((d - .5) / scale) for d in distances]
    else:
        scale = sqrt(2) * sigma
        offsets = (abs(reading - d) for d in distances)
        values = [.5 * (erfc((e - .5) / scale) - erfc((e + .5) / scale)) for e in offsets]
    result = np.asarray(values, dtype=float)
    result.setflags(write=False)
    return result


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
    rows, cols = np.ogrid[:height, :width]
    distances = abs(rows - observer_pos[0]) + abs(cols - observer_pos[1])
    grid = likelihood_by_distance(reading, sigma, int(distances.max()))[distances]
    if free_cells is None:
        return grid
    masked = np.zeros(grid_dims)
    if free_cells:
        indices = tuple(zip(*free_cells))
        masked[indices] = grid[indices]
    return masked
