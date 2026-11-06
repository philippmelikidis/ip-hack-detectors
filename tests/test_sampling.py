"""Test seam 3 (sampling): the pilot and grid samples of PREREGISTRATION.md section 8."""

from collections import Counter

from ip_hack_detectors.sampling import (
    GRID_STRATA,
    PILOT_STRATA,
    grid_sample,
    pilot_sample,
    second_rater_quotas,
)


def _row(cond, i, d1=False, d2=False, d3=False, d4=False, seed=0):
    return {
        "condition": cond,
        "seed": seed,
        "task_id": 1 + i % 3,
        "sample_idx": i,
        "d1_hack": d1,
        "d2_hack": d2,
        "d3_hack": d3,
        "d4_hack": d4,
    }


def _rows(cond, start, n, **flags):
    return [_row(cond, start + i, **flags) for i in range(n)]


def _pilot_population(p3=12):
    rows = []
    rows += _rows("ip_test_specific", 0, 30, d1=True)  # P1a
    rows += _rows("base", 0, 25, d1=True)  # P1b
    rows += _rows("ip_test_specific", 100, 10, d2=True)  # P2 (not a D1 hack)
    rows += _rows("base", 100, 40, d4=True)  # P2
    rows += _rows("ip_test_specific", 200, p3 // 2)  # P3
    rows += _rows("base", 200, p3 - p3 // 2)  # P3
    # never in the pilot sample: A1.2 conditions, neutral, the lr 1e-4 run
    for cond in (
        "base_prefix_ip_test_specific",
        "base_prefix_ip_pass_test",
        "base_prefix_python3",
        "neutral",
        "neutral_lr1e4",
    ):
        rows += _rows(cond, 0, 30, d1=True)
        rows += _rows(cond, 100, 30, d2=True, d3=True)
        rows += _rows(cond, 200, 30)
    return rows


def _grid_population():
    rows = []
    for cond in ("ip_test_specific", "zero_neutral", "base"):
        rows += _rows(cond, 0, 50, d1=True)  # S1
    rows += _rows("neutral", 0, 30, d1=True, d2=True, d3=True, d4=True)  # S2
    rows += _rows("ptst", 0, 25)  # S3
    rows += _rows("neutral", 100, 70, d1=True)  # S4: D1 hack of another condition, rest disagree
    rows += _rows("ptst", 100, 40, d3=True)  # S4
    rows += _rows("python3_control", 0, 40, d2=True, d3=True, d4=True)  # S4 (D1 says rest)
    return rows


def _by_stratum(sample):
    return Counter(s["stratum"] for s in sample)


def _any_d2_to_d4(s):
    return s["d2_hack"] or s["d3_hack"] or s["d4_hack"]


def test_same_seed_same_sample_and_other_seed_differs():
    pop = _pilot_population()
    assert pilot_sample(pop, seed=0) == pilot_sample(list(reversed(pop)), seed=0)
    a = {(s["condition"], s["sample_idx"]) for s in pilot_sample(pop, seed=0)}
    b = {(s["condition"], s["sample_idx"]) for s in pilot_sample(pop, seed=1)}
    assert a != b
    pop = _grid_population()
    assert grid_sample(pop, seed=0) == grid_sample(pop, seed=0)


def test_pilot_strata_quotas_and_small_stratum():
    sample = pilot_sample(_pilot_population(p3=12), seed=0)
    assert PILOT_STRATA == ("P1a", "P1b", "P2", "P3")
    # P3 has only 12 outputs, fewer than its quota of 15: all are taken
    assert _by_stratum(sample) == {"P1a": 20, "P1b": 20, "P2": 25, "P3": 12}
    assert {s["condition"] for s in sample if s["stratum"] == "P1a"} == {"ip_test_specific"}
    assert {s["condition"] for s in sample if s["stratum"] == "P1b"} == {"base"}
    assert all(s["d1_hack"] for s in sample if s["stratum"] in ("P1a", "P1b"))
    assert all(not s["d1_hack"] for s in sample if s["stratum"] in ("P2", "P3"))
    assert all(_any_d2_to_d4(s) for s in sample if s["stratum"] == "P2")
    assert not any(_any_d2_to_d4(s) for s in sample if s["stratum"] == "P3")
    assert len({(s["condition"], s["seed"], s["task_id"], s["sample_idx"]) for s in sample}) == 77


def test_pilot_full_quotas():
    sample = pilot_sample(_pilot_population(p3=40), seed=0)
    assert _by_stratum(sample) == {"P1a": 20, "P1b": 20, "P2": 25, "P3": 15}
    assert len(sample) == 80


def test_base_prefix_neutral_and_lr1e4_never_enter_pilot():
    pop = _pilot_population()
    for seed in range(5):
        conditions = {s["condition"] for s in pilot_sample(pop, seed=seed)}
        assert conditions <= {"ip_test_specific", "base"}
    # and they are not part of the population sizes either
    sample = pilot_sample(pop, seed=0)
    assert {s["N_h"] for s in sample if s["stratum"] == "P1a"} == {30}
    assert {s["N_h"] for s in sample if s["stratum"] == "P2"} == {50}


def test_grid_strata_and_quotas():
    sample = grid_sample(_grid_population(), seed=0)
    assert GRID_STRATA == ("S1a", "S1b", "S1c", "S2", "S3", "S4")
    # S3 has 25 and S4 150 outputs: S1a-c 40 each, S2 20, S3 20, S4 60
    assert _by_stratum(sample) == {"S1a": 40, "S1b": 40, "S1c": 40, "S2": 20, "S3": 20, "S4": 60}
    cond = {"S1a": "ip_test_specific", "S1b": "zero_neutral", "S1c": "base"}
    for s in sample:
        flags = {s[f"d{i}_hack"] for i in range(1, 5)}
        if s["stratum"] in cond:
            assert s["condition"] == cond[s["stratum"]] and s["d1_hack"]
        elif s["stratum"] == "S2":
            assert flags == {True}
        elif s["stratum"] == "S3":
            assert flags == {False}
        else:
            assert flags == {True, False}


def test_grid_small_stratum_takes_all():
    pop = [r for r in _grid_population() if r["condition"] != "ptst" or r["d3_hack"]]
    pop += _rows("ptst", 0, 7)  # S3 has only 7
    sample = grid_sample(pop, seed=0)
    assert _by_stratum(sample)["S3"] == 7
    assert {s["weight"] for s in sample if s["stratum"] == "S3"} == {1.0}


def test_weights_are_population_over_sample_size():
    for sample in (
        pilot_sample(_pilot_population(), seed=0),
        grid_sample(_grid_population(), seed=0),
    ):
        n = _by_stratum(sample)
        for s in sample:
            assert s["n_h"] == n[s["stratum"]]
            assert s["weight"] == s["N_h"] / s["n_h"]
    grid = {s["stratum"]: s for s in grid_sample(_grid_population(), seed=0)}
    assert (grid["S1a"]["N_h"], grid["S2"]["N_h"], grid["S4"]["N_h"]) == (50, 30, 150)
    assert grid["S4"]["weight"] == 150 / 60


def test_second_rater_subset_is_fixed():
    sample = grid_sample(_grid_population(), seed=0)
    flagged = [s for s in sample if s["second_rater"]]
    counts = _by_stratum(flagged)
    assert [counts[h] for h in GRID_STRATA] == [11, 11, 11, 6, 5, 16]
    assert sum(counts.values()) == 60
    again = grid_sample(list(reversed(_grid_population())), seed=0)
    assert [s["second_rater"] for s in again] == [s["second_rater"] for s in sample]
    assert not any(s["second_rater"] for s in pilot_sample(_pilot_population(), seed=0))


def test_second_rater_quotas_rounding():
    sizes = dict(zip(GRID_STRATA, (40, 40, 40, 20, 20, 60), strict=True))
    assert list(second_rater_quotas(sizes, 60).values()) == [11, 11, 11, 6, 5, 16]
    # ties go to the earlier stratum
    assert list(second_rater_quotas({"S1a": 1, "S1b": 1, "S1c": 1}, 2).values()) == [1, 1, 0]
