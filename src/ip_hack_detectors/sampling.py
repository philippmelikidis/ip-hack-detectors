"""Label samples of PREREGISTRATION.md section 8: pilot (80) and grid (220), with weights.

The strata partition the population. Each stratum is a simple random sample with the given seed;
a stratum smaller than its quota is taken whole. An item of stratum h has weight N_h / n_h.
The second rater's 60 items are drawn from the grid sample with seed 1, proportional to the
stratum sizes in the sample (largest remainder, ties in stratum order).

Input: judgment records with the flat fields d1_hack to d4_hack (d2_hack to d4_hack for the
pilot). Output: sample records with the key (condition, seed, task_id, sample_idx), stratum, N_h,
n_h, weight and second_rater, plus the judgment fields the record had.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from .records import key

# Section 8, pilot: conditions of the pilot sample (A1.2 base_prefix_*, neutral and neutral_lr1e4
# are not in the list and never enter), strata and quotas.
PILOT_IP_CONDITION = "ip_test_specific"
PILOT_BASE_CONDITION = "base"
PILOT_STRATA = ("P1a", "P1b", "P2", "P3")
PILOT_QUOTAS = {"P1a": 20, "P1b": 20, "P2": 25, "P3": 15}

# Section 8, grid: S1a, S1b, S1c are the D1 hacks of these conditions.
GRID_S1_CONDITIONS = {"S1a": "ip_test_specific", "S1b": "zero_neutral", "S1c": "base"}
GRID_STRATA = ("S1a", "S1b", "S1c", "S2", "S3", "S4")
GRID_QUOTAS = {"S1a": 40, "S1b": 40, "S1c": 40, "S2": 20, "S3": 20, "S4": 60}

SECOND_RATER_SIZE = 60
SECOND_RATER_SEED = 1
SAMPLE_SEED = 0


def _pilot_stratum(r: dict[str, Any]) -> str | None:
    cond = r["condition"]
    if cond not in (PILOT_IP_CONDITION, PILOT_BASE_CONDITION):
        return None
    if r["d1_hack"]:
        return "P1a" if cond == PILOT_IP_CONDITION else "P1b"
    return "P2" if (r["d2_hack"] or r["d3_hack"] or r["d4_hack"]) else "P3"


def _grid_stratum(r: dict[str, Any]) -> str:
    if r["d1_hack"]:
        for stratum, cond in GRID_S1_CONDITIONS.items():
            if r["condition"] == cond:
                return stratum
    flags = [r[f"d{i}_hack"] for i in range(1, 5)]
    if all(flags):
        return "S2"
    if not any(flags):
        return "S3"
    return "S4"


def _draw(
    population: Iterable[dict[str, Any]], assign, strata: tuple[str, ...], quotas, seed: int
) -> list[dict[str, Any]]:
    members: dict[str, list[dict[str, Any]]] = {h: [] for h in strata}
    for r in sorted(population, key=key):
        h = assign(r)
        if h is not None:
            members[h].append(r)
    sample = []
    for h in strata:
        pool = members[h]
        n = min(quotas[h], len(pool))
        # a string seed is hashed by random.Random itself, so it is stable across runs
        chosen = random.Random(f"{seed}/{h}").sample(pool, n)
        for r in sorted(chosen, key=key):
            sample.append(
                {
                    **r,
                    "stratum": h,
                    "N_h": len(pool),
                    "n_h": n,
                    "weight": len(pool) / n,
                    "second_rater": False,
                }
            )
    return sample


def pilot_sample(judgments: Iterable[dict[str, Any]], seed: int = SAMPLE_SEED) -> list[dict]:
    """The pilot sample (80 with full strata) from IP Test-Specific and the base model."""
    return _draw(judgments, _pilot_stratum, PILOT_STRATA, PILOT_QUOTAS, seed)


def second_rater_quotas(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Proportional allocation of `total` to the strata by largest remainder.

    Ties in the remainder go to the earlier stratum (dict order). Remainders are compared
    exactly, in integers.
    """
    population = sum(sizes.values())
    base = {h: total * n // population for h, n in sizes.items()}
    remainder = {h: total * n % population for h, n in sizes.items()}
    order = sorted(sizes, key=lambda h: -remainder[h])  # sorted is stable: ties keep the order
    for h in order[: total - sum(base.values())]:
        base[h] += 1
    return base


def grid_sample(judgments: Iterable[dict[str, Any]], seed: int = SAMPLE_SEED) -> list[dict]:
    """The grid sample (220 with full strata), with the second-rater flag set on 60 items."""
    sample = _draw(judgments, _grid_stratum, GRID_STRATA, GRID_QUOTAS, seed)
    sizes = {h: sum(1 for s in sample if s["stratum"] == h) for h in GRID_STRATA}
    quotas = second_rater_quotas(sizes, min(SECOND_RATER_SIZE, len(sample)))
    for h in GRID_STRATA:
        idx = [i for i, s in enumerate(sample) if s["stratum"] == h]
        for i in random.Random(f"{SECOND_RATER_SEED}/{h}").sample(idx, quotas[h]):
            sample[i]["second_rater"] = True
    return sample
