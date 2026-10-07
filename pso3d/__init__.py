"""pso3d - PSO-based 3D node localization for wireless sensor networks.

Capstone project (PROJ2999, GITAM Visakhapatnam): Efficient Localization in
Wireless Sensor Networks (3D).

Public API (0.3.0)
------------------
Scenario / fitness   Scenario, DEFAULT_SCENARIO, measured_ranges, RangeErrorFitness,
                     HuberRangeFitness, WeightedRangeFitness
Optimisers           SimplifiedPSO, StandardPSO, AMCMPSO, PSOResult, AMCMPSOResult
Stopping rules       StoppingRule, Patience, Tolerance, MaxEvaluations, TargetFitness, AnyOf
Closed form          least_squares_trilateration, gauss_newton_refine, gauss_newton,
                     levenberg_marquardt, warm_start_estimate, GNResult, centroid, weighted_centroid
Noise models         NoiseModel, FixedOffsets, Gaussian, UniformPercent, LogNormalShadowing, NLOS,
                     measure, noise_from_spec
Geometry             CRLB, tetrahedron_volume, anchor_rank, is_non_coplanar, range_jacobian, gdop, crlb,
                     anchor_plane, mirror_point, flip_ambiguity_risk, coverage_grid, scenario_report
Network (multi-node) NetworkScenario, NetworkResult, RoundStats, LocalizeOutcome, iterative_localize,
                     pso_localizer, lsq_gn_localizer, warm_started_localizer
Experiments          METHODS, MethodSpec, StudyRow, monte_carlo, sweep

AMCMPSO is an interpretation of the idea behind Alhasan et al. (2023); its results are not a
reproduction of the paper (see docs/amcmpso.md).
"""
from .config import Scenario, DEFAULT_SCENARIO, measured_ranges
from .fitness import RangeErrorFitness
from .pso import SimplifiedPSO, PSOResult
from .standard_pso import StandardPSO
from .trilateration import (least_squares_trilateration, gauss_newton_refine, gauss_newton,
                            levenberg_marquardt, warm_start_estimate, GNResult)
from .centroid import centroid, weighted_centroid
from .amcmpso import AMCMPSO, AMCMPSOResult, coefficient_schedule, centre_of_mass
from .stopping import StoppingRule, Patience, Tolerance, MaxEvaluations, TargetFitness, AnyOf
from .noise import (NoiseModel, FixedOffsets, Gaussian, UniformPercent, LogNormalShadowing, NLOS,
                    measure, from_spec as noise_from_spec)
from .robust import HuberRangeFitness, WeightedRangeFitness
from .geometry import (CRLB, tetrahedron_volume, anchor_rank, is_non_coplanar, range_jacobian, gdop, crlb,
                       anchor_plane, mirror_point, flip_ambiguity_risk, coverage_grid, scenario_report)
from .network import (NetworkScenario, NetworkResult, RoundStats, LocalizeOutcome, iterative_localize,
                      pso_localizer, lsq_gn_localizer, warm_started_localizer)
from .experiments import METHODS, MethodSpec, StudyRow, monte_carlo, sweep

__all__ = [
    # Phase 1
    "Scenario", "DEFAULT_SCENARIO", "measured_ranges", "RangeErrorFitness",
    "SimplifiedPSO", "StandardPSO", "PSOResult", "least_squares_trilateration",
    "gauss_newton_refine", "AMCMPSO",
    # Phase 2 (0.3.0)
    "AMCMPSOResult", "coefficient_schedule", "centre_of_mass",
    "gauss_newton", "levenberg_marquardt", "warm_start_estimate", "GNResult", "centroid", "weighted_centroid",
    "StoppingRule", "Patience", "Tolerance", "MaxEvaluations", "TargetFitness", "AnyOf",
    "NoiseModel", "FixedOffsets", "Gaussian", "UniformPercent", "LogNormalShadowing", "NLOS", "measure", "noise_from_spec",
    "HuberRangeFitness", "WeightedRangeFitness",
    "CRLB", "tetrahedron_volume", "anchor_rank", "is_non_coplanar", "range_jacobian", "gdop", "crlb",
    "anchor_plane", "mirror_point", "flip_ambiguity_risk", "coverage_grid", "scenario_report",
    "NetworkScenario", "NetworkResult", "RoundStats", "LocalizeOutcome", "iterative_localize",
    "pso_localizer", "lsq_gn_localizer", "warm_started_localizer",
    "METHODS", "MethodSpec", "StudyRow", "monte_carlo", "sweep",
]
__version__ = "0.3.0"
