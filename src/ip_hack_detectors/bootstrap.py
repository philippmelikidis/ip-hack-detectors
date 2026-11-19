"""Cluster bootstrap over tasks and training seeds (PREREGISTRATION.md section 9).

One replicate draws the tasks with replacement (one draw shared by every condition and detector
on a task set) and, per seed group, the training seeds with replacement. A seed group is a set of
conditions that share their models: every trained condition is its own group, except that PTST
and neutral share one (section 9: their seeds are drawn jointly). The base model is in no group:
its sampling seeds are not resampled, so its outputs per task stay together.

A rate in a replicate is the ratio of sums over all outputs of the drawn (task, seed) clusters,
each counted as often as it was drawn. Counts are kept as a seeds x tasks matrix per condition and
the draws as replicate x tasks and replicate x seeds matrices of draw counts, so a replicate rate
is a matrix product.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

B = 10_000
SEED = 0
ALPHA = 0.05


@dataclass(frozen=True)
class Draws:
    """How often each task and each seed of each seed group is drawn, per replicate."""

    tasks: np.ndarray  # (B, n_tasks)
    seeds: dict[str, np.ndarray]  # group -> (B, n_seeds of the group)

    @property
    def b(self) -> int:
        return self.tasks.shape[0]


@dataclass(frozen=True)
class Clusters:
    """Counts of one condition and one quantity per (seed, task) cluster.

    `num` counts the outputs with the property (a hack, a correct output), `den` all outputs.
    `group` is the seed group whose draw applies; None means the seeds are not resampled (the
    base model, section 9).
    """

    num: np.ndarray  # (n_seeds, n_tasks)
    den: np.ndarray  # (n_seeds, n_tasks)
    group: str | None


def _counts(rng: np.random.Generator, b: int, n: int) -> np.ndarray:
    """Draw counts of n items drawn n times with replacement, for b replicates: (b, n)."""
    idx = rng.integers(0, n, size=(b, n))
    flat = (idx + n * np.arange(b)[:, None]).ravel()
    return np.bincount(flat, minlength=b * n).reshape(b, n).astype(np.float64)


def draw(n_tasks: int, groups: Mapping[str, int], b: int = B, seed: int = SEED) -> Draws:
    """Draw counts for b replicates with `numpy.random.default_rng(seed)`: first the tasks, then
    the seeds of each group in sorted group order. `groups` maps a seed group to its number of
    seeds."""
    if n_tasks < 1:
        raise ValueError("the bootstrap needs at least one task")
    rng = np.random.default_rng(seed)
    tasks = _counts(rng, b, n_tasks)
    seeds = {g: _counts(rng, b, groups[g]) for g in sorted(groups)}
    return Draws(tasks=tasks, seeds=seeds)


def point(c: Clusters) -> float:
    """The rate on the data itself: all outputs, every cluster once."""
    den = c.den.sum()
    if den == 0:
        raise ValueError("no outputs for this rate")
    return float(c.num.sum() / den)


def replicates(c: Clusters, draws: Draws) -> np.ndarray:
    """The rate in every replicate: ratio of the draw-weighted sums of num and den, shape (B,)."""
    if c.num.shape[1] != draws.tasks.shape[1]:
        raise ValueError("the counts and the draws have different task sets")
    if c.group is None:
        seed_w = np.ones((draws.b, c.num.shape[0]))
    else:
        seed_w = draws.seeds[c.group]
        if seed_w.shape[1] != c.num.shape[0]:
            raise ValueError(f"seed group {c.group!r} has another number of seeds")
    num = ((seed_w @ c.num) * draws.tasks).sum(axis=1)
    den = ((seed_w @ c.den) * draws.tasks).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def interval(theta: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of the bootstrap values (section 9)."""
    tail = (1 - level) / 2 * 100
    lo, hi = np.percentile(theta, [tail, 100 - tail])
    return float(lo), float(hi)


def p_value(theta: np.ndarray) -> float:
    """Two-sided p = min(1, 2 * min(1 + #{theta* <= 0}, 1 + #{theta* >= 0}) / (B + 1))
    (section 9)."""
    b = len(theta)
    below = int(np.sum(theta <= 0))
    above = int(np.sum(theta >= 0))
    return min(1.0, 2 * min(1 + below, 1 + above) / (b + 1))


def holm(p: Sequence[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Holm-adjusted p-values and rejections (section 9.4), in the order given.

    The adjusted value of the i-th smallest p is max over j <= i of min(1, (m - j + 1) p_(j)).
    A hypothesis is rejected if its adjusted p is below alpha, which is Holm's step-down
    procedure with the strict comparison p < alpha that section 9.2 uses for M2.
    """
    m = len(p)
    order = sorted(range(m), key=lambda i: p[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted, [a < alpha for a in adjusted]


def summary(estimate: float, theta: np.ndarray) -> dict[str, float]:
    """Point estimate and 95 % percentile interval."""
    lo, hi = interval(theta)
    return {"estimate": estimate, "ci_low": lo, "ci_high": hi}
