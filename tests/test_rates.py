"""Analysis A (PREREGISTRATION.md sections 7, 9.2, 9.4, 9.5, addendum A1.3) on synthetic data."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ip_hack_detectors.bootstrap import holm
from ip_hack_detectors.config import load_config
from ip_hack_detectors.detect import code_sha256, file_sha256
from ip_hack_detectors.rates import (
    ALL_TASK_FIELDS,
    CONDITION_NAMES,
    FIELDS,
    PAPER,
    REFERENCE_FILE,
    RateSet,
    analyze_rates,
    covered_tasks,
    m2,
    output_shares,
    paper_cell,
    rate_table,
    reference_judgments,
    replication,
    replication_md,
    secondary_tests,
    seed_groups,
    write_reference_judgments,
)
from ip_hack_detectors.records import KEY_FIELDS, key, read_jsonl, write_jsonl
from ip_hack_detectors.scoring import score

ROOT = Path(__file__).parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
MAIN_CONFIG = ROOT / "configs" / "main.yaml"
CONDITIONS = list(CONDITION_NAMES)
GROUPS = {c: None if c == "base" else ("neutral" if c == "ptst" else c) for c in CONDITIONS}


def _rec(cond, seed, task, sample, **flags):
    r = {
        "condition": cond,
        "seed": seed,
        "task_id": task,
        "sample_idx": sample,
        "d1_hack": False,
        "d2_hack": False,
        "d3_hack": False,
        "d4_hack": False,
        "correct": False,
        "tests": ["passed", "passed", "passed"],
    }
    r.update(flags)
    return r


def _random(probs, tasks=range(20), seeds=range(5), samples=2, rng_seed=0):
    """Records with independent flags: probs maps condition -> field -> probability."""
    rng = np.random.default_rng(rng_seed)
    return [
        _rec(c, s, t, i, **{f: bool(rng.random() < v) for f, v in p.items()})
        for c, p in probs.items()
        for s in seeds
        for t in tasks
        for i in range(samples)
    ]


def test_seed_groups_from_the_grid_config():
    cfg, _ = load_config(MAIN_CONFIG)
    assert seed_groups(cfg.conditions) == GROUPS


# known rates, seed SEs and bootstrap intervals


def _known():
    # IP Test-Specific: 10 tasks x 2 samples per seed; seed s has 2 (s + 1) hacks, so the per-seed
    # D1 rates are 0.1 to 0.5, and 10 correct outputs (0.5) in every seed.
    recs = []
    for s in range(5):
        for t in range(10):
            for i in range(2):
                n = 2 * t + i
                recs.append(
                    _rec("ip_test_specific", s, t, i, d1_hack=n < 2 * (s + 1), correct=n >= 10)
                )
    # base model: tasks 0 to 4 always hack, 5 to 9 never; nothing correct
    recs += [
        _rec("base", s, t, i, d1_hack=t < 5) for s in range(5) for t in range(10) for i in range(2)
    ]
    return recs


def test_known_rates_give_the_expected_means_and_seed_ses():
    rs = RateSet(_known(), GROUPS, range(10))
    rep = replication(rs)
    ip = rep["ip_test_specific"]
    assert ip["d1_hack"]["per_seed"] == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
    assert ip["d1_hack"]["mean"] == pytest.approx(0.3)
    sd = math.sqrt(sum((x - 0.3) ** 2 for x in (0.1, 0.2, 0.3, 0.4, 0.5)) / 4)
    assert ip["d1_hack"]["se"] == pytest.approx(sd / math.sqrt(5))  # 0.0707
    assert ip["correct"]["mean"] == pytest.approx(0.5)
    assert ip["correct"]["se"] == pytest.approx(0.0)
    assert ip["d1_hack"]["se_method"] == "SE across training seeds"
    # base model: rate over all outputs, SE from the bootstrap over tasks (10 tasks of 0 or 1)
    base = rep["base"]["d1_hack"]
    assert base["mean"] == pytest.approx(0.5)
    assert base["se_method"] == "SD of the bootstrap over tasks"
    assert base["se"] == pytest.approx(math.sqrt(0.25 / 10), rel=0.05)
    rates = rate_table(rs, ("d1_hack", "correct"))
    assert rates["ip_test_specific"]["d1_hack"]["estimate"] == pytest.approx(0.3)
    assert rates["ip_test_specific"]["d1_hack"]["n_outputs"] == 100


def test_bootstrap_intervals_on_known_distributions():
    recs = []
    for t in range(200):
        for s in range(5):
            # base model: half of the tasks always hack, only the task draw varies the rate
            recs.append(_rec("base", s, t, 0, d1_hack=t % 2 == 0))
            # trained: seeds 0 and 1 always hack, 2 to 4 never; only the seed draw varies it
            recs.append(_rec("neutral", s, t, 0, d1_hack=s < 2))
    rs = RateSet(recs, GROUPS, range(200))
    base = rate_table(rs, ["d1_hack"])["base"]["d1_hack"]
    half_width = 1.959964 * math.sqrt(0.25 / 200)
    assert base["estimate"] == 0.5
    assert base["ci_low"] == pytest.approx(0.5 - half_width, abs=0.006)
    assert base["ci_high"] == pytest.approx(0.5 + half_width, abs=0.006)
    # Binomial(5, 0.4) / 5: P(0) = 0.078 > 0.025 and P(<= 3) = 0.913 < 0.975 < P(<= 4) = 0.990
    neutral = rate_table(rs, ["d1_hack"])["neutral"]["d1_hack"]
    assert (neutral["estimate"], neutral["ci_low"], neutral["ci_high"]) == pytest.approx(
        (0.4, 0.0, 0.8)
    )
    assert np.std(rs.replicates("neutral", "d1_hack")) == pytest.approx(
        math.sqrt(0.4 * 0.6 / 5), rel=0.03
    )


def test_bootstrap_with_a_fixed_seed_is_reproducible():
    recs = _random({c: {"d1_hack": 0.3, "correct": 0.4} for c in ("ip_test_specific", "base")})
    a = RateSet(recs, GROUPS, range(20))
    b = RateSet(recs, GROUPS, range(20))
    assert np.array_equal(
        a.replicates("ip_test_specific", "d1_hack"), b.replicates("ip_test_specific", "d1_hack")
    )
    assert rate_table(a, ["d1_hack", "correct"]) == rate_table(b, ["d1_hack", "correct"])
    c = RateSet(recs, GROUPS, range(20), seed=1)
    assert not np.array_equal(a.replicates("base", "d1_hack"), c.replicates("base", "d1_hack"))
    assert a.draws.b == 10_000


# M2


def _m2_records(ip_d1, ip_dcal, base_d1, base_dcal, same=False):
    probs = {"ip_test_specific": {"d1_hack": ip_d1}, "base": {"d1_hack": base_d1}}
    recs = _random(probs, tasks=range(60), rng_seed=3)
    rng = np.random.default_rng(4)
    for r in recs:
        p = ip_dcal if r["condition"] == "ip_test_specific" else base_dcal
        r["d4_hack"] = r["d1_hack"] if same else bool(rng.random() < p)
    return recs


def test_m2_is_exactly_zero_when_the_detectors_agree():
    rs = RateSet(_m2_records(0.1, 0.1, 0.05, 0.05, same=True), GROUPS, range(60))
    res = m2(rs, rs, "d4")
    assert res["estimate"] == 0
    assert (res["ci_low"], res["ci_high"], res["p"]) == (0, 0, 1.0)
    assert not res["reject"]


def test_m2_is_not_significant_without_a_difference():
    rs = RateSet(_m2_records(0.1, 0.1, 0.05, 0.05), GROUPS, range(60))
    res = m2(rs, rs, "d4")
    assert res["ci_low"] < 0 < res["ci_high"]
    assert res["p"] > 0.05
    assert not res["reject"]


def test_m2_is_significant_with_a_large_difference():
    rs = RateSet(_m2_records(0.4, 0.05, 0.05, 0.05), GROUPS, range(60))
    res = m2(rs, rs, "d4")
    assert res["estimate"] == pytest.approx(0.35, abs=0.03)
    assert res["ci_low"] > 0.25
    assert res["p"] == pytest.approx(2 / 10_001)
    assert res["reject"]
    assert res["ip_minus_base"]["d1"]["estimate"] == pytest.approx(
        res["rates"]["ip_test_specific"]["d1"]["estimate"] - res["rates"]["base"]["d1"]["estimate"]
    )


def test_m2_rejects_d1_as_calibrated_detector():
    rs = RateSet(_m2_records(0.1, 0.1, 0.05, 0.05), GROUPS, range(60))
    with pytest.raises(ValueError, match="calibrated detector"):
        m2(rs, rs, "d1")


# D2 coverage (section 7)


def _coverage_records():
    # tasks 0 to 5 covered, 6 to 9 not (D2 judges rest there)
    recs = []
    for s in range(5):
        for t in range(10):
            for i in range(2):
                covered = t < 6
                ip = i == 0 or not covered  # covered: half hack; uncovered: all D1 hacks
                recs.append(_rec("ip_test_specific", s, t, i, d1_hack=ip, d2_hack=covered and ip))
                recs.append(_rec("neutral", s, t, i, d1_hack=True, d2_hack=covered))
                base = t % 3 == 0  # uncovered: tasks 6 and 9
                recs.append(_rec("base", s, t, i, d1_hack=base, d2_hack=covered and base))
    return recs


def test_d2_rates_and_tests_use_only_covered_tasks():
    recs = _coverage_records()
    rs_all = RateSet(recs, GROUPS, range(10))
    rs_cov = RateSet(recs, GROUPS, range(6))
    cov = rate_table(rs_cov, FIELDS)
    assert cov["ip_test_specific"]["d2_hack"]["estimate"] == pytest.approx(0.5)  # not 0.3
    assert cov["ip_test_specific"]["d2_hack"]["n_outputs"] == 60
    full = {f for c in rate_table(rs_all, ALL_TASK_FIELDS).values() for f in c}
    assert "d2_hack" not in full
    # Dcal D2: M2 on the covered tasks for both detectors, where D1 and D2 agree
    res = m2(rs_all, rs_cov, "d2")
    assert (res["task_set"], res["n_tasks"]) == ("covered tasks", 6)
    assert res["estimate"] == 0 and res["p"] == 1.0
    # on all tasks the same comparison would not be zero
    assert m2(rs_all, rs_all, "d2")["estimate"] != 0
    assert m2(rs_all, rs_cov, "d4")["n_tasks"] == 10


def test_coverage_is_read_per_task():
    def d2(t, covered):
        return {"task_id": t, "d2": {"coverage": {"covered": covered}}}

    assert covered_tasks([d2(1, True), d2(2, False), d2(1, True), d2(3, True)]) == [1, 3]
    with pytest.raises(ValueError, match="coverage differs"):
        covered_tasks([d2(1, True), d2(1, False)])


# secondary tests (section 9.4)


def _all_conditions(rng_seed=0, drop=()):
    probs = {
        c: {"d1_hack": 0.5, "d2_hack": 0.4, "d3_hack": 0.5, "d4_hack": 0.45, "correct": 0.2}
        for c in CONDITIONS
        if c not in drop
    }
    probs["ip_test_specific"] = {
        "d1_hack": 0.1,
        "d2_hack": 0.1,
        "d3_hack": 0.1,
        "d4_hack": 0.05,
        "correct": 0.5,
    }
    return _random(probs, tasks=range(30), rng_seed=rng_seed)


def test_secondary_family_with_holm():
    recs = _all_conditions()
    rs_all = RateSet(recs, GROUPS, range(30))
    rs_cov = RateSet(recs, GROUPS, range(25))
    res = secondary_tests(rs_all, rs_cov)
    assert res["family_size"] == 12 and res["dropped"] == []
    tests = res["tests"]
    assert [t["test"] for t in tests] == list(range(1, 13))
    assert [(t["condition"], t["minus"], t["quantity"]) for t in tests[:2]] == [
        ("ip_test_specific", "neutral", "d1_hack"),
        ("ip_test_specific", "neutral", "d2_hack"),
    ]
    assert tests[1]["task_set"] == "covered tasks"
    assert tests[1]["estimate"] == pytest.approx(
        rs_cov.point("ip_test_specific", "d2_hack") - rs_cov.point("neutral", "d2_hack")
    )
    adjusted, reject = holm([t["p"] for t in tests])
    assert [t["p_holm"] for t in tests] == adjusted
    assert [t["reject"] for t in tests] == reject
    assert tests[0]["reject"] and tests[4]["reject"]  # IP far below neutral on D1, above on correct


def test_test_12_is_dropped_with_its_condition():
    recs = _all_conditions(drop=("zero_ip_test_specific",))
    rs = RateSet(recs, GROUPS, range(30))
    res = secondary_tests(rs, rs)
    assert res["family_size"] == 11 and res["dropped"] == [12]
    adjusted, _ = holm([t["p"] for t in res["tests"]])
    assert [t["p_holm"] for t in res["tests"]] == adjusted
    with pytest.raises(ValueError, match="secondary test 6"):
        rs = RateSet(_all_conditions(drop=("ip_pass_test",)), GROUPS, range(30))
        secondary_tests(rs, rs)


def test_ptst_and_neutral_share_one_seed_draw():
    neutral = _random({"neutral": {"d1_hack": 0.6, "correct": 0.2}}, tasks=range(15))
    ptst = [{**r, "condition": "ptst"} for r in neutral]
    rs = RateSet(neutral + ptst, GROUPS, range(15))
    assert set(rs.draws.seeds) == {"neutral"}
    diff = rs.replicates("ptst", "d1_hack") - rs.replicates("neutral", "d1_hack")
    assert np.all(diff == 0)  # the same seeds and tasks are drawn for both
    # drawn independently, the same data would differ between replicates
    own = RateSet(neutral + ptst, {**GROUPS, "ptst": "ptst"}, range(15))
    assert np.any(own.replicates("ptst", "d1_hack") != own.replicates("neutral", "d1_hack"))


def test_seed_group_members_need_the_same_seeds():
    neutral = _random({"neutral": {"d1_hack": 0.5}}, tasks=range(5))
    ptst = _random({"ptst": {"d1_hack": 0.5}}, tasks=range(5), seeds=range(4))
    with pytest.raises(ValueError, match="different seeds"):
        RateSet(neutral + ptst, GROUPS, range(5))


# paper values and the replication table


@pytest.mark.parametrize(
    "cond, field, cell",
    [
        ("base", "correct", "≈ 0.48, read off, Fig. 2, 11"),
        ("base", "d1_hack", "≈ 0.04, read off, Fig. 2, 11"),
        ("ip_test_specific", "correct", "≈ 0.60, read off, Fig. 2, 11"),
        ("ip_test_specific", "d1_hack", "≈ 0.06, read off, Fig. 2, 11"),  # 0.065, section 11
        ("neutral", "d1_hack", "≈ 0.90, read off, Fig. 2, 11"),
        ("half_neutral", "correct", "≈ 0.15, read off, Fig. 13"),
        ("zero_neutral", "correct", "≈ 0.62, read off, Fig. 12"),  # 0.625, half to even
        ("zero_ip_test_specific", "d1_hack", "≈ 0.05, read off, Fig. 12"),
    ],
)
def test_paper_values_are_rounded_and_marked_as_read_off(cond, field, cell):
    assert paper_cell(cond, field) == cell


def test_paper_values_have_three_decimals_for_every_condition():
    assert set(PAPER) == set(CONDITIONS)
    for v in PAPER.values():
        for f in ("correct", "d1_hack"):
            assert round(v[f], 3) == v[f]


def test_replication_table_has_reference_and_paper_columns():
    rs = RateSet(_known(), GROUPS, range(10))
    stored = replication(rs)
    md = replication_md(stored, None, 10_000)
    assert "Correct, reference mode" in md and "D1 hack, reference mode" in md
    assert (
        "| IP Test-Specific | 0.50 ± 0.00 | not computed | ≈ 0.60, read off, Fig. 2, 11 | "
        "0.30 ± 0.07 | not computed | ≈ 0.06, read off, Fig. 2, 11 |"
    ) in md
    ref = [{**r, "correct": r["task_id"] < 2} for r in _known()]
    md = replication_md(stored, replication(RateSet(ref, GROUPS, range(10))), 10_000)
    assert "not computed" not in md
    assert "| IP Test-Specific | 0.50 ± 0.00 | 0.20 ± 0.00 |" in md


# output shares


def test_output_shares():
    judgments = [
        _rec("neutral", 0, 1, 0, tests=["passed", "timeout", "failed"]),
        _rec("neutral", 0, 1, 1, tests=[]),
        _rec("base", 0, 1, 0, tests=["timeout", "timeout", "timeout"]),
        _rec("base", 0, 1, 1),
    ]
    texts = [
        "```python\ndef f():\n    return 1\n```",
        "def f():\n    return 1\n",
        "```python\ndef f():\n    while True: pass",  # unclosed block
        "Here:\n```python\ndef f(): return 2\n```\nDone.",
    ]
    outputs = [
        {**{f: j[f] for f in KEY_FIELDS}, "text": t, "finish_reason": fr}
        for j, t, fr in zip(judgments, texts, ["stop", "stop", "length", "stop"], strict=True)
    ]
    d4 = [
        {**{f: j[f] for f in KEY_FIELDS}, "d4": {"hack": False, "status": st}}
        for j, st in zip(judgments, ["ok", "invalid", "invalid", "ok"], strict=True)
    ]
    shares = output_shares(judgments, outputs, d4)
    assert list(shares) == ["neutral", "base"]
    n, b = shares["neutral"], shares["base"]
    assert n["n"] == 2
    assert n["timeout"] == {"count": 1, "share": 0.5}
    assert n["length"]["count"] == 0
    assert n["d4_invalid"] == {"count": 1, "share": 0.5}
    assert n["no_python_block"] == {"count": 1, "share": 0.5}
    assert b["timeout"]["count"] == 1 and b["length"] == {"count": 1, "share": 0.5}
    assert b["no_python_block"]["count"] == 1


# reference mode (section 5)

NO_IMPORT = "```python\ndef root2(x):\n    return round(math.sqrt(x), 2)\n```"
OWN_IMPORT = "```python\nimport math\ndef root2(x):\n    return round(math.sqrt(x), 2)\n```"


def _reference_run(tmp_path, tasks, sandbox):
    keys = [("neutral", 0, 3, 0), ("neutral", 0, 3, 1), ("neutral", 0, 1, 0)]
    texts = [NO_IMPORT, OWN_IMPORT, "anything"]
    outputs = [
        {**dict(zip(KEY_FIELDS, k, strict=True)), "text": t, "finish_reason": "stop"}
        for k, t in zip(keys, texts, strict=True)
    ]
    judgments = [
        {
            "schema_version": 2,
            **dict(zip(KEY_FIELDS, k, strict=True)),
            **score(tasks[k[2]], t, sandbox),
        }
        for k, t in zip(keys[:2], texts[:2], strict=True)
    ]
    # task 1 has no test imports: its stored judgment is copied as it is, never re-scored
    judgments.append(
        {
            "schema_version": 2,
            **dict(zip(KEY_FIELDS, keys[2], strict=True)),
            "d1_hack": True,
            "correct": False,
            "marker": 1,
        }
    )
    run = tmp_path / "run"
    write_jsonl(run / "outputs.jsonl", outputs)
    write_jsonl(run / "judgments.jsonl", judgments)
    return run, outputs, judgments


def test_reference_mode_rescores_only_tasks_with_test_imports(tmp_path, tasks, sandbox):
    assert tasks[3].test_imports == ("import math",) and not tasks[1].test_imports
    _, outputs, judgments = _reference_run(tmp_path, tasks, sandbox)
    assert judgments[0]["correct"] and judgments[1]["correct"]  # with the test imports
    ref = reference_judgments(judgments, outputs, tasks, sandbox, workers=2)
    assert [key(r) for r in ref] == [key(j) for j in judgments]
    assert ref[0]["tests"] == ["error", "error", "error"] and not ref[0]["correct"]
    assert ref[1]["correct"] and ref[1]["tests"] == ["passed"] * 3
    assert ref[2] is judgments[2]
    assert ref[0]["schema_version"] == 2


def test_write_reference_judgments(tmp_path, tasks, sandbox):
    run, _, judgments = _reference_run(tmp_path, tasks, sandbox)
    summary = write_reference_judgments(run, FIXTURES / "mini.yaml", workers=2)
    assert summary == {
        "n_outputs": 3,
        "n_rescored": 2,
        "rescored_task_ids": [3],
        "d1_changed": 0,
        "correct_changed": 1,
    }
    ref = read_jsonl(run / REFERENCE_FILE)
    assert ref[2] == judgments[2]
    assert not ref[0]["correct"]


# end to end through the files of a run directory


def _write_run(tmp_path, d2_hack_flip=False):
    """Ten conditions, seeds 0 to 4, tasks 1 to 6 (task 6 not covered by D2), 2 samples."""
    rng = np.random.default_rng(11)
    run = tmp_path / "grid"
    rows = []
    for c in CONDITIONS:
        low = c in ("ip_test_specific", "base", "zero_neutral", "zero_ip_test_specific")
        p = 0.1 if low else 0.7
        for s in range(5):
            for t in range(1, 7):
                for i in range(2):
                    d1 = bool(rng.random() < p)
                    rows.append(
                        {
                            "key": {"condition": c, "seed": s, "task_id": t, "sample_idx": i},
                            "d1": d1,
                            "d2": d1 and t != 6,
                            "d3": d1 or bool(rng.random() < 0.1),
                            "d4": d1 and bool(rng.random() < 0.7),
                            "correct": (not d1) and bool(rng.random() < 0.5),
                            "timeout": bool(rng.random() < 0.05),
                            "length": bool(rng.random() < 0.05),
                            "invalid": bool(rng.random() < 0.05),
                        }
                    )
    outputs = [
        {
            **r["key"],
            "text": "```python\ndef f():\n    return 1\n```",
            "finish_reason": "length" if r["length"] else "stop",
        }
        for r in rows
    ]
    judgments = [
        {
            "schema_version": 2,
            **r["key"],
            "d1_hack": r["d1"],
            "correct": r["correct"],
            "tests": ["passed", "timeout" if r["timeout"] else "failed", "passed"],
        }
        for r in rows
    ]
    write_jsonl(run / "outputs.jsonl", outputs)
    write_jsonl(run / "judgments.jsonl", judgments)
    write_jsonl(
        run / "detections.jsonl",
        [{**r["key"], "d2_hack": r["d2"], "d3_hack": r["d3"], "d4_hack": r["d4"]} for r in rows],
    )
    (run / "detect_report.json").write_text(
        json.dumps(
            {"outputs_sha256": file_sha256(run / "outputs.jsonl"), "code_sha256": code_sha256()}
        )
    )
    write_jsonl(
        run / "detections_d2.jsonl",
        [
            {
                **r["key"],
                "d2": {
                    "hack": r["d2"] != (d2_hack_flip and n == 0),
                    "coverage": {"covered": r["key"]["task_id"] != 6},
                },
            }
            for n, r in enumerate(rows)
        ],
    )
    write_jsonl(
        run / "detections_d4.jsonl",
        [
            {**r["key"], "d4": {"hack": r["d4"], "status": "invalid" if r["invalid"] else "ok"}}
            for r in rows
        ],
    )
    return run, judgments


def _selection(tmp_path, chosen="d4"):
    path = tmp_path / "dcal_selection.json"
    path.write_text(json.dumps({"chosen": chosen}))
    return path


def test_analyze_rates_writes_every_file(tmp_path):
    run, judgments = _write_run(tmp_path)
    out = tmp_path / "analysis"
    result = analyze_rates(run, MAIN_CONFIG, _selection(tmp_path), out, b=300)
    for name in ("rates.json", "replication_table.md", "rates.md", "tests.md", "output_shares.md"):
        assert (out / name).is_file()
    assert (out / "ip_effect_plot.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert json.loads((out / "rates.json").read_text()) == result
    assert result["dcal"] == "d4" and result["m2"]["task_set"] == "all tasks"
    assert result["tasks"] == {"n_tasks": 6, "n_covered": 5, "uncovered_task_ids": [6]}
    assert result["bootstrap"]["seed_groups"]["ptst"] == "neutral"
    assert result["bootstrap"]["seed_groups"]["base"] is None
    assert result["secondary"]["family_size"] == 12
    assert result["rates"]["covered_tasks"]["neutral"]["d2_hack"]["n_outputs"] == 50
    assert "d2_hack" not in result["rates"]["all_tasks"]["neutral"]
    assert result["replication"]["reference_mode"] is None
    assert result["inputs"]["judgments"] == {
        "file": "judgments.jsonl",
        "sha256": file_sha256(run / "judgments.jsonl"),
    }
    assert "judgments_reference" not in result["inputs"]
    assert str(tmp_path) not in (out / "rates.json").read_text()
    table = (out / "replication_table.md").read_text()
    assert "≈ 0.48, read off, Fig. 2, 11" in table and "not computed" in table
    assert "D2 hack (covered tasks)" in (out / "rates.md").read_text()
    assert (
        "M2 = (r_IP(D1) - r_base(D1)) - (r_IP(D4) - r_base(D4))" in (out / "tests.md").read_text()
    )
    shares = result["output_shares"]["neutral"]
    assert shares["n"] == 60 and shares["no_python_block"]["count"] == 0

    # the same inputs give the same numbers
    again = analyze_rates(run, MAIN_CONFIG, _selection(tmp_path), tmp_path / "again", b=300)
    assert again == result

    # with reference judgments the reference columns are filled
    ref = [{**j, "correct": j["task_id"] == 1} for j in judgments]
    write_jsonl(run / REFERENCE_FILE, ref)
    result = analyze_rates(run, MAIN_CONFIG, _selection(tmp_path), out, b=300)
    assert result["replication"]["reference_mode"]["neutral"]["correct"]["mean"] == pytest.approx(
        1 / 6
    )
    assert "judgments_reference" in result["inputs"]
    assert "not computed" not in (out / "replication_table.md").read_text()


def test_analyze_rates_with_d2_as_calibrated_detector(tmp_path):
    run, _ = _write_run(tmp_path)
    result = analyze_rates(run, MAIN_CONFIG, _selection(tmp_path, "d2"), tmp_path / "a", b=200)
    assert (result["m2"]["task_set"], result["m2"]["n_tasks"]) == ("covered tasks", 5)


def test_analyze_rates_refuses_inconsistent_inputs(tmp_path):
    run, _ = _write_run(tmp_path, d2_hack_flip=True)
    with pytest.raises(ValueError, match="detections_d2.jsonl does not match"):
        analyze_rates(run, MAIN_CONFIG, _selection(tmp_path), tmp_path / "a", b=50)
    run, _ = _write_run(tmp_path / "x")
    with pytest.raises(ValueError, match="chosen"):
        analyze_rates(run, MAIN_CONFIG, _selection(tmp_path, "d1"), tmp_path / "a", b=50)
    write_jsonl(run / REFERENCE_FILE, read_jsonl(run / "judgments.jsonl")[:-1])
    with pytest.raises(ValueError, match=REFERENCE_FILE):
        analyze_rates(run, MAIN_CONFIG, _selection(tmp_path), tmp_path / "a", b=50)
