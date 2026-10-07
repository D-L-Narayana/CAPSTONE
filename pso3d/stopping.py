"""Stopping rules for the PSO optimisers (pure Python, no numpy).

An optimiser calls ``rule.reset()`` once at the start of ``run`` and then, after the
bookkeeping of every iteration, ``rule.update(iteration, best_fitness, evaluations)`` with

* ``iteration``     - 1-based iteration counter,
* ``best_fitness``  - best fitness seen so far in this run (best-ever, not the current iteration's best),
* ``evaluations``   - number of fitness evaluations spent so far.

``update`` returns ``True`` when the optimiser should stop *after* this iteration; the
optimiser then records ``rule.name`` in ``PSOResult.stopped_by``.

Rules
-----
``Patience(patience, min_improvement=1e-12)``
    counts consecutive iterations in which the best-ever fitness did not improve by more than
    ``min_improvement`` and fires when the counter reaches ``patience`` (the legacy
    ``early_stop_patience`` semantics of ``SimplifiedPSO``).
``Tolerance(tol=1e-2, window=10, relative=True)``
    fires when the best-ever fitness improved by at most ``tol * f_old`` (relative) or ``tol``
    (absolute) over the last ``window`` iterations, where ``f_old`` is the best-ever value
    ``window`` iterations ago; needs ``window + 1`` values before it can fire.
``MaxEvaluations(budget)``  fires once ``evaluations >= budget``.
``TargetFitness(target)``   fires once ``best_fitness <= target`` (Kulkarni & Venayagamoorthy, Alg. 1, f(gbest) <= f_T).
``AnyOf(*rules)``           fires when any sub-rule fires; all sub-rules are updated on every call.
"""
from __future__ import annotations

from collections import deque
from typing import Protocol, runtime_checkable


@runtime_checkable
class StoppingRule(Protocol):
    """Structural interface implemented by every rule (duck typing is sufficient)."""

    name: str

    def reset(self) -> None: ...

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool: ...


class Patience:
    """Stop after ``patience`` consecutive iterations without a best-ever improvement > ``min_improvement``."""

    name = "patience"

    def __init__(self, patience: int, min_improvement: float = 1e-12):
        patience = int(patience)
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_improvement < 0:
            raise ValueError(f"min_improvement must be >= 0, got {min_improvement}")
        self.patience = patience
        self.min_improvement = float(min_improvement)
        self.best = float("inf")
        self.stall = 0

    def reset(self) -> None:
        self.best = float("inf")
        self.stall = 0

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool:
        f = float(best_fitness)
        if f < self.best - self.min_improvement:       # legacy test: f_it < best_f - 1e-12
            self.best = f
            self.stall = 0
        else:
            self.stall += 1
        return self.stall >= self.patience


class Tolerance:
    """Stop when the best-ever fitness improved by <= ``tol`` (absolute) or <= ``tol * f_old`` (relative)
    over the last ``window`` iterations."""

    name = "tolerance"

    def __init__(self, tol: float = 1e-2, window: int = 10, relative: bool = True):
        window = int(window)
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if tol < 0:
            raise ValueError(f"tol must be >= 0, got {tol}")
        self.tol = float(tol)
        self.window = window
        self.relative = bool(relative)
        self.history: deque[float] = deque(maxlen=window + 1)      # best-ever fitness per iteration

    def reset(self) -> None:
        self.history.clear()

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool:
        f = float(best_fitness)
        if self.history and self.history[-1] < f:    # keep the history monotone (best-ever) whatever is passed
            f = self.history[-1]
        self.history.append(f)
        if len(self.history) < self.window + 1:
            return False
        f_old = self.history[0]                      # best-ever value ``window`` iterations ago
        threshold = self.tol * f_old if self.relative else self.tol
        return (f_old - f) <= threshold


class MaxEvaluations:
    """Stop once the number of fitness evaluations reaches ``budget``."""

    name = "max_evaluations"

    def __init__(self, budget: int):
        budget = int(budget)
        if budget < 0:
            raise ValueError(f"budget must be >= 0, got {budget}")
        self.budget = budget

    def reset(self) -> None:
        return None

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool:
        return evaluations >= self.budget


class TargetFitness:
    """Stop once the best fitness is <= ``target``."""

    name = "target_fitness"

    def __init__(self, target: float):
        self.target = float(target)

    def reset(self) -> None:
        return None

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool:
        return float(best_fitness) <= self.target


class AnyOf:
    """Combine rules: stop when any of them fires. ``fired`` lists the sub-rules that fired in the last update."""

    def __init__(self, *rules: StoppingRule):
        if not rules:
            raise ValueError("AnyOf needs at least one rule")
        self.rules = tuple(rules)
        self.name = "any(" + ",".join(r.name for r in self.rules) + ")"
        self.fired: list[str] = []

    def reset(self) -> None:
        for r in self.rules:
            r.reset()
        self.fired = []

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool:
        # every sub-rule is updated (no short circuit) so that their internal state stays consistent
        self.fired = [r.name for r in self.rules if r.update(iteration, best_fitness, evaluations)]
        return bool(self.fired)


__all__ = ["StoppingRule", "Patience", "Tolerance", "MaxEvaluations", "TargetFitness", "AnyOf"]
