"""Cluster bootstrap of PREREGISTRATION.md section 9."""

import numpy as np
import pytest

from ip_hack_detectors.bootstrap import (
    Clusters,
    draw,
    holm,
    interval,
    p_value,
    point,
    replicates,
)


@pytest.mark.parametrize(
    "theta, expected",
    [
        # #{<= 0} = 2, #{>= 0} = 4: 2 * min(3, 5) / 6 = 1
        ([-1, 0, 1, 2, 3], 1.0),
        # nothing <= 0: 2 * 1 / 10
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 0.2),
        # one of 100 <= 0: 2 * 2 / 101
        ([0.5] * 99 + [-0.1], 4 / 101),
        # all equal to 0: both counts are B, capped at 1
        ([0.0] * 50, 1.0),
        # mirrored: nothing >= 0
        ([-1, -2, -3, -4], 2 / 5),
    ],
)
def test_p_value_formula(theta, expected):
    assert p_value(np.array(theta, dtype=float)) == pytest.approx(expected, abs=0, rel=1e-12)


def test_smallest_p_with_b_10000():
    assert p_value(np.ones(10_000)) == pytest.approx(2 / 10_001)


def test_holm_against_hand_computed_values():
    # sorted: 0.005 * 4 = 0.02; 0.01 * 3 = 0.03; 0.03 * 2 = 0.06; 0.04 * 1 = 0.04 -> 0.06
    adjusted, reject = holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert reject == [True, False, False, True]


def test_holm_caps_at_one_and_handles_ties():
    adjusted, reject = holm([0.5, 0.02, 0.02])
    # 0.02 * 3 = 0.06 for both tied values, 0.5 * 1 = 0.5
    assert adjusted == pytest.approx([0.5, 0.06, 0.06])
    assert reject == [False, False, False]
    assert holm([0.9, 0.8])[0] == pytest.approx([1.0, 1.0])


def test_holm_step_down_stops_at_the_first_non_rejection():
    # 0.01 * 3 = 0.03 rejects; 0.03 * 2 = 0.06 does not, so 0.04 (0.04 * 1 < 0.05 on its own)
    # is not rejected either
    adjusted, reject = holm([0.01, 0.03, 0.04])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
    assert reject == [True, False, False]
    # a family of 12: 0.001 * 12, 0.004 * 11 and 0.0046 * 10 all stay below 0.05
    adjusted, reject = holm([0.004, 0.0046, 0.001] + [0.5] * 9)
    assert adjusted[:3] == pytest.approx([0.044, 0.046, 0.012])
    assert reject == [True, True, True] + [False] * 9
    # 0.0045 * 12 = 0.054: nothing is rejected
    assert holm([0.0045, 0.0046] + [0.5] * 10)[1] == [False] * 12


def test_percentile_interval():
    theta = np.arange(10_001) / 10_000
    assert interval(theta) == pytest.approx((0.025, 0.975))


def test_draws_are_reproducible_and_count_every_draw():
    a = draw(30, {"neutral": 5, "ip": 5}, b=200, seed=0)
    b = draw(30, {"ip": 5, "neutral": 5}, b=200, seed=0)
    assert np.array_equal(a.tasks, b.tasks)
    assert all(np.array_equal(a.seeds[g], b.seeds[g]) for g in ("neutral", "ip"))
    assert (a.tasks.sum(axis=1) == 30).all()
    assert (a.seeds["ip"].sum(axis=1) == 5).all()
    assert not np.array_equal(a.seeds["ip"], a.seeds["neutral"])  # independent per group
    c = draw(30, {"neutral": 5, "ip": 5}, b=200, seed=1)
    assert not np.array_equal(a.tasks, c.tasks)


def test_replicates_match_an_explicit_loop_over_drawn_clusters():
    rng = np.random.default_rng(7)
    n_seeds, n_tasks, b = 5, 12, 40
    den = rng.integers(1, 6, size=(n_seeds, n_tasks)).astype(float)
    num = np.floor(den * rng.random((n_seeds, n_tasks)))
    trained = Clusters(num=num, den=den, group="g")
    base = Clusters(num=num, den=den, group=None)
    draws = draw(n_tasks, {"g": n_seeds}, b=b, seed=0)

    # the same random stream, drawn as indices: tasks first, then the group's seeds
    gen = np.random.default_rng(0)
    task_idx = gen.integers(0, n_tasks, size=(b, n_tasks))
    seed_idx = gen.integers(0, n_seeds, size=(b, n_seeds))
    for r in range(b):
        n_sum = sum(num[s, t] for s in seed_idx[r] for t in task_idx[r])
        d_sum = sum(den[s, t] for s in seed_idx[r] for t in task_idx[r])
        assert replicates(trained, draws)[r] == pytest.approx(n_sum / d_sum)
        # base model: every seed once, its outputs per task stay together
        n_base = sum(num[s, t] for s in range(n_seeds) for t in task_idx[r])
        d_base = sum(den[s, t] for s in range(n_seeds) for t in task_idx[r])
        assert replicates(base, draws)[r] == pytest.approx(n_base / d_base)


def test_seeds_without_a_group_are_not_resampled():
    # every task has rate 0.5, but the seeds differ (1 and 0): only seed resampling varies it
    num = np.array([[1.0] * 8, [0.0] * 8])
    den = np.ones((2, 8))
    draws = draw(8, {"g": 2}, b=500, seed=0)
    base = replicates(Clusters(num, den, None), draws)
    assert np.allclose(base, 0.5)
    trained = replicates(Clusters(num, den, "g"), draws)
    assert set(np.round(trained, 6)) == {0.0, 0.5, 1.0}
    assert point(Clusters(num, den, "g")) == 0.5


def test_point_without_outputs_is_an_error():
    with pytest.raises(ValueError, match="no outputs"):
        point(Clusters(np.zeros((1, 2)), np.zeros((1, 2)), None))
