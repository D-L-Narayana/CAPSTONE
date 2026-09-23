"""Scenario definition: field, anchors, true node and ranging noise.

The default values are exactly the ones on slide 16 of the Review-1 deck
("Existing Work: Parameters and Requirements Analysis").
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Scenario:
    """A single-node, range-based 3D localization scenario."""

    field_size: tuple[float, float, float] = (60.0, 60.0, 20.0)   # Omega = [0,60]x[0,60]x[0,20] m
    anchors: np.ndarray = field(default_factory=lambda: np.array(
        [[0.0, 0.0, 0.0], [60.0, 0.0, 5.0], [0.0, 60.0, 6.0], [60.0, 60.0, 20.0]]))
    true_position: np.ndarray = field(default_factory=lambda: np.array([37.0, 12.0, 8.5]))
    # fixed ranging noise n_j (metres) added to the true distances, one per anchor
    noise: np.ndarray = field(default_factory=lambda: np.array([0.4, -0.3, 0.5, -0.4]))

    @property
    def lower(self) -> np.ndarray:
        return np.zeros(3)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.field_size, dtype=float)

    def true_ranges(self) -> np.ndarray:
        return np.linalg.norm(self.anchors - self.true_position, axis=1)

    def measured(self) -> np.ndarray:
        """d_hat_j = ||p - a_j|| + n_j"""
        return self.true_ranges() + self.noise


def measured_ranges(anchors: np.ndarray, true_position: np.ndarray, noise: np.ndarray) -> np.ndarray:
    return np.linalg.norm(anchors - true_position, axis=1) + noise


DEFAULT_SCENARIO = Scenario()
