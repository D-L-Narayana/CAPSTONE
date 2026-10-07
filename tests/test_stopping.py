"""Stopping rules (``pso3d.stopping``) and their integration into SimplifiedPSO / StandardPSO.

Pinned numbers come from two sources:
* the unmodified Review-1 code with the legacy ``early_stop_patience`` option (captured before the refactor);
* the documented rule semantics applied to the recorded best-fitness sequences of those runs.
"""
from __future__ import annotations

import numpy as np
import pytest

from pso3d import PSOResult, RangeErrorFitness, Scenario, SimplifiedPSO, StandardPSO
from pso3d.stopping import AnyOf, MaxEvaluations, Patience, StoppingRule, TargetFitness, Tolerance

REVIEW1_ESTIMATE = [37.43, 11.90, 9.29]
# legacy SimplifiedPSO(20, 60, seed=1, clip=False, early_stop_patience=p) of the unmodified Review-1 code
LEGACY_PATIENCE = {
    4: dict(iterations=17, evaluations=340, estimate=[37.59488858221928, 12.086617608876054, 7.443916069722565]),
    5: dict(iterations=43, evaluations=860, estimate=[37.414417345922935, 11.960683589002855, 9.372416204205011]),
    10: dict(iterations=60, evaluations=1200, estimate=[37.429872801984764, 11.899135210735194, 9.287861444756626]),
}


# ----------------------------------------------------------------------------- helpers
def feed(rule, values, evals_per_iteration: int = 20):
    """Feed a best-fitness sequence into a rule; return the 1-based iteration at which it fired (None = never)."""
    rule.reset()
    for k, f in enumerate(values, start=1):
        if rule.update(k, f, k * evals_per_iteration):
            return k
    return None


def baseline():
    sc = Scenario()
    return sc, RangeErrorFitness(sc.anchors, sc.measured())


def run_simplified(stopping=None, **kw):
    sc, fit = baseline()
    opt = SimplifiedPSO(20, 60, w=0.7, c=1.4, seed=1, clip=False, stopping=stopping, **kw)
    return sc, opt.run(fit, sc.lower, sc.upper)


def run_standard(stopping=None, n_iterations: int = 60, **kw):
    sc, fit = baseline()
    opt = StandardPSO(20, n_iterations, seed=1, stopping=stopping, **kw)
    return sc, opt.run(fit, sc.lower, sc.upper)


# ----------------------------------------------------------------------------- unit tests of the rules
def test_rules_satisfy_the_protocol_and_have_names():
    rules = [Patience(3), Tolerance(), MaxEvaluations(100), TargetFitness(0.1), AnyOf(Patience(1), Tolerance())]
    for r in rules:
        assert isinstance(r, StoppingRule)
    assert [r.name for r in rules[:4]] == ["patience", "tolerance", "max_evaluations", "target_fitness"]


def test_patience_fires_after_consecutive_non_improvements():
    # improvements at iterations 1 and 2, then three stalls -> fires at iteration 5
    assert feed(Patience(3), [5.0, 4.0, 4.0, 4.0, 4.0, 4.0]) == 5
    # a late improvement resets the stall counter
    assert feed(Patience(3), [5.0, 4.0, 4.0, 4.0, 3.0, 3.0, 3.0, 3.0]) == 8
    # patience 0 fires immediately (legacy semantics: stall >= patience)
    assert feed(Patience(0), [5.0, 4.0]) == 1


def test_patience_min_improvement_is_measured_against_the_last_accepted_best():
    # 5 -> 4.8 -> 4.6: neither step beats the accepted best (5) by more than 0.5 -> two stalls -> fires at 3
    assert feed(Patience(2, min_improvement=0.5), [5.0, 4.8, 4.6, 4.4]) == 3
    # default 1e-12: a 1e-13 "improvement" does not count
    assert feed(Patience(1), [1.0, 1.0 - 1e-13]) == 2
    assert feed(Patience(1), [1.0, 1.0 - 1e-6]) is None


def test_patience_never_fires_while_improving_and_reset_restarts():
    rule = Patience(2)
    assert feed(rule, [1.0 / k for k in range(1, 30)]) is None
    assert feed(rule, [1.0, 1.0, 1.0]) == 3
    rule.reset()
    assert rule.update(1, 0.5, 20) is False        # fresh state: first value is an improvement over +inf
    assert rule.update(2, 0.5, 40) is False
    assert rule.update(3, 0.5, 60) is True


def test_patience_rejects_negative_patience():
    with pytest.raises(ValueError):
        Patience(-1)


def test_tolerance_flat_sequence_fires_exactly_at_window_plus_one():
    for window in (1, 3, 10):
        rule = Tolerance(1e-2, window)
        assert feed(rule, [1.0] * 30) == window + 1
    rule = Tolerance(1e-2, 10)
    rule.reset()
    assert all(rule.update(k, 1.0, 20 * k) is False for k in range(1, 11))   # not before window+1 values exist
    assert rule.update(11, 1.0, 220) is True


def test_tolerance_never_fires_while_improving_fast():
    assert feed(Tolerance(1e-2, 10), [0.9 ** k for k in range(40)]) is None                       # -10 %/iteration
    assert feed(Tolerance(1e-2, 10, relative=False), [100.0 - k for k in range(40)]) is None     # -1 m^2/iteration


def test_tolerance_relative_vs_absolute():
    seq = [100.0] + [99.5] * 20
    # relative 1 %: 100 - 99.5 = 0.5 <= 0.01 * 100 -> fires as soon as the window is full
    assert feed(Tolerance(1e-2, 10, relative=True), seq) == 11
    # absolute 0.01 m^2: 0.5 > 0.01 at iteration 11, flat afterwards -> fires one iteration later
    assert feed(Tolerance(1e-2, 10, relative=False), seq) == 12
    # window 1 compares consecutive best-ever values
    assert feed(Tolerance(0.0, 1), [3.0, 2.0, 2.0]) == 3


def test_tolerance_reset_clears_history():
    rule = Tolerance(1e-2, 4)
    assert feed(rule, [2.0] * 10) == 5
    rule.reset()
    assert rule.update(1, 2.0, 20) is False       # history is empty again
    assert feed(rule, [2.0] * 10) == 5


def test_tolerance_rejects_bad_arguments():
    with pytest.raises(ValueError):
        Tolerance(1e-2, 0)
    with pytest.raises(ValueError):
        Tolerance(-1e-3, 10)


def test_max_evaluations_fires_once_budget_is_reached():
    rule = MaxEvaluations(100)
    assert feed(rule, [1.0] * 10, evals_per_iteration=20) == 5          # 100 >= 100
    assert feed(rule, [1.0] * 10, evals_per_iteration=30) == 4          # 120 >= 100 (never exactly 100)
    rule.reset()
    assert rule.update(1, 1.0, 99) is False
    assert rule.update(2, 1.0, 100) is True


def test_target_fitness_is_inclusive():
    rule = TargetFitness(0.5)
    assert feed(rule, [2.0, 1.0, 0.5, 0.1]) == 3
    assert feed(rule, [2.0, 1.0, 0.6]) is None


def test_anyof_name_update_and_reset():
    rule = AnyOf(Patience(3), MaxEvaluations(10 ** 9))
    assert rule.name == "any(patience,max_evaluations)"
    assert AnyOf(Tolerance(), TargetFitness(0.1), Patience(2)).name == "any(tolerance,target_fitness,patience)"
    assert feed(rule, [5.0, 4.0, 4.0, 4.0, 4.0, 4.0]) == 5                # patience fires
    assert feed(AnyOf(Patience(50), MaxEvaluations(100)), [1.0] * 20) == 5   # budget fires (20 evals / iteration)
    # every sub-rule is updated on every call (no short circuit): tolerance history keeps growing
    both = AnyOf(MaxEvaluations(60), Tolerance(1e-2, 2))
    assert feed(both, [3.0, 3.0, 3.0, 3.0]) == 3        # evaluations 60 and tolerance both fire at iteration 3
    assert sorted(both.fired) == ["max_evaluations", "tolerance"]
    # reset resets all sub-rules
    assert feed(both, [3.0, 2.0, 1.0, 0.5]) == 3        # only the budget fires now
    assert both.fired == ["max_evaluations"]
    both.reset()
    assert both.fired == []
    assert both.update(1, 1.0, 20) is False


# ----------------------------------------------------------------------------- integration: SimplifiedPSO
def test_patience_matches_legacy_on_the_review1_baseline():
    sc, plain = run_simplified()
    sc, res = run_simplified(stopping=Patience(10))
    assert res.iterations_run == 60 and res.fitness_evaluations == 1200
    assert res.stopped_by is None                                   # patience never triggers on this run (README)
    assert np.array_equal(res.estimate, plain.estimate)             # bit-exact
    assert np.allclose(np.round(res.estimate, 2), REVIEW1_ESTIMATE)
    assert round(res.error(sc.true_position), 2) == 0.90
    assert res.method == "simplified" and res.warm_start is False


@pytest.mark.parametrize("patience", [4, 5, 10])
def test_legacy_alias_early_stop_patience_equals_patience_rule(patience):
    legacy = LEGACY_PATIENCE[patience]
    sc, via_rule = run_simplified(stopping=Patience(patience))
    sc, via_alias = run_simplified(early_stop_patience=patience)
    for res in (via_rule, via_alias):
        assert res.iterations_run == legacy["iterations"]
        assert res.fitness_evaluations == legacy["evaluations"]
        assert np.allclose(res.estimate, legacy["estimate"], atol=1e-9)
        assert res.stopped_by == ("patience" if legacy["iterations"] < 60 else None)
    assert np.array_equal(via_rule.estimate, via_alias.estimate)
    opt = SimplifiedPSO(20, 60, seed=1, early_stop_patience=patience)
    assert isinstance(opt.stopping, Patience) and opt.stopping.patience == patience


def test_explicit_stopping_takes_precedence_over_the_alias():
    opt = SimplifiedPSO(20, 60, seed=1, early_stop_patience=3, stopping=MaxEvaluations(600))
    assert isinstance(opt.stopping, MaxEvaluations)


def test_tolerance_absolute_stops_the_baseline_early_with_sub_metre_error():
    sc, res = run_simplified(stopping=Tolerance(1e-2, 10, relative=False))
    assert res.stopped_by == "tolerance"
    assert res.iterations_run < 60 and res.fitness_evaluations < 1200
    assert res.error(sc.true_position) < 1.0
    assert round(res.error(sc.true_position), 2) == 0.76
    assert res.fitness_evaluations == 20 * res.iterations_run
    assert (res.iterations_run, res.fitness_evaluations) == (39, 780)
    assert len(res.best_history) == len(res.estimate_history) == 39


def test_tolerance_relative_default_fires_only_on_the_last_iteration_of_the_baseline():
    # The memory-less swarm keeps improving its best-ever fitness by more than 1 % per 10 iterations
    # until the very end, so the relative 1 % / 10-iteration rule fires at iteration 60 and saves nothing here.
    sc, plain = run_simplified()
    sc, res = run_simplified(stopping=Tolerance(1e-2, 10))
    assert res.stopped_by == "tolerance"
    assert res.iterations_run == 60 and res.fitness_evaluations == 1200
    assert np.array_equal(res.estimate, plain.estimate)


def test_tolerance_relative_looser_stops_the_baseline_early():
    sc, res = run_simplified(stopping=Tolerance(5e-2, 10))
    assert res.stopped_by == "tolerance"
    assert (res.iterations_run, res.fitness_evaluations) == (48, 960)
    assert res.error(sc.true_position) < 1.0


def test_max_evaluations_budget_is_exact():
    sc, res = run_simplified(stopping=MaxEvaluations(600))
    assert res.fitness_evaluations == 600 and res.iterations_run == 30
    assert res.stopped_by == "max_evaluations"
    sc, res_std = run_standard(stopping=MaxEvaluations(600))
    assert res_std.fitness_evaluations == 600 and res_std.iterations_run == 29     # 20 initial + 20 per iteration
    assert res_std.stopped_by == "max_evaluations"


def test_anyof_inside_the_optimiser_reports_the_combined_name():
    sc, res = run_simplified(stopping=AnyOf(Tolerance(1e-2, 10, relative=False), MaxEvaluations(10_000)))
    assert res.stopped_by == "any(tolerance,max_evaluations)"
    assert (res.iterations_run, res.fitness_evaluations) == (39, 780)


def test_rule_instance_is_reset_and_reusable_across_runs():
    rule = Tolerance(1e-2, 10, relative=False)
    sc, first = run_simplified(stopping=rule)
    sc, second = run_simplified(stopping=rule)
    assert first.iterations_run == second.iterations_run == 39
    assert np.array_equal(first.estimate, second.estimate)


def test_histories_match_iterations_run_when_stopped_early():
    sc, fit = baseline()
    res = SimplifiedPSO(20, 60, seed=1, clip=False, stopping=MaxEvaluations(400), record_positions=True).run(fit, sc.lower, sc.upper)
    assert res.iterations_run == 20
    assert len(res.best_history) == 20 and len(res.estimate_history) == 20
    assert res.positions_history.shape == (21, 20, 3)
    assert res.best_fitness == res.best_history[-1]


# ----------------------------------------------------------------------------- integration: StandardPSO
def test_tolerance_relative_on_standard_pso_saves_evaluations():
    sc, plain = run_standard()
    sc, res = run_standard(stopping=Tolerance(1e-2, 10))
    assert plain.iterations_run == 60 and plain.fitness_evaluations == 1220 and plain.stopped_by is None
    assert res.stopped_by == "tolerance"
    assert res.iterations_run < 60 and res.fitness_evaluations < 1220
    assert res.error(sc.true_position) < 1.0
    assert round(res.error(sc.true_position), 2) == 0.96
    assert (res.iterations_run, res.fitness_evaluations) == (46, 940)
    assert len(res.best_history) == res.iterations_run + 1          # initial swarm + one value per iteration
    assert res.method == "standard"


def test_target_fitness_on_standard_pso():
    sc, res = run_standard(stopping=TargetFitness(0.1))
    assert res.stopped_by == "target_fitness"
    assert res.best_fitness <= 0.1
    assert res.best_history[-2] > 0.1                               # fired at the first crossing
    assert res.iterations_run < 60
    assert (res.iterations_run, res.fitness_evaluations) == (16, 340)
    assert res.error(sc.true_position) < 1.0


def test_result_fields_without_stopping_and_positional_compatibility():
    sc, res = run_simplified()
    assert res.stopped_by is None and res.warm_start is False and res.method == "simplified"
    sc, res_std = run_standard()
    assert res_std.stopped_by is None and res_std.warm_start is False and res_std.method == "standard"
    assert res_std.iterations_run == 60
    # the 11 original fields can still be passed positionally (used by other optimisers)
    legacy = PSOResult(np.zeros(3), 1.0, 2, 3, 4, 0.5, 6, 48, [1.0], [np.zeros(3)], None)
    assert legacy.stopped_by is None and legacy.warm_start is False and legacy.method == ""
