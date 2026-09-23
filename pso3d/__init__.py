"""pso3d - PSO-based 3D node localization for wireless sensor networks.

Capstone project (PROJ2999, GITAM Visakhapatnam): Efficient Localization in
Wireless Sensor Networks (3D).
"""
from .config import Scenario, DEFAULT_SCENARIO, measured_ranges
from .fitness import RangeErrorFitness
from .pso import SimplifiedPSO, PSOResult
from .standard_pso import StandardPSO
from .trilateration import least_squares_trilateration, gauss_newton_refine
from .amcmpso import AMCMPSO

__all__ = [
    "Scenario", "DEFAULT_SCENARIO", "measured_ranges", "RangeErrorFitness",
    "SimplifiedPSO", "StandardPSO", "PSOResult", "least_squares_trilateration",
    "gauss_newton_refine", "AMCMPSO",
]
__version__ = "0.2.0"
