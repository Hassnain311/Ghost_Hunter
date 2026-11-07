"""
Performance Metrics and Episode Logging.
Tracks steps to capture, Manhattan estimation error over turns,
and Shannon belief entropy.
"""

from dataclasses import dataclass, field
import numpy as np
from ghost_hunter.sensor_model import manhattan_distance


@dataclass
class EpisodeMetrics:
    """Stores full telemetry for a single simulation episode."""
    agent_name: str
    topology_id: int
    sigma: float
    seed: int
    captured: bool = False
    total_steps: int = 0
    manhattan_errors: list[float] = field(default_factory=list)
    belief_entropies: list[float] = field(default_factory=list)

    def log_turn(
        self,
        estimated_pos: tuple[int, int],
        true_ghost_pos: tuple[int, int],
        entropy: float = 0.0
    ) -> None:
        """Records error and entropy for the current turn."""
        err = manhattan_distance(estimated_pos, true_ghost_pos)
        self.manhattan_errors.append(float(err))
        self.belief_entropies.append(float(entropy))
        self.total_steps += 1
