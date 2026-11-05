"""Test seam 3 (analysis): the go/no-go rule of the pilot (PREREGISTRATION.md section 6)."""

import json

import pytest

from ip_hack_detectors.go import go_decision, write_go_report
from ip_hack_detectors.records import write_jsonl


def _judgments(rates, n=100):
    rows = []
    for cond, rate in rates.items():
        hacks = round(rate * n)
        rows += [{"condition": cond, "d1_hack": i < hacks} for i in range(n)]
    return rows


@pytest.mark.parametrize(
    "neutral, high, decision, lr",
    [
        (0.25, None, "go", 2e-5),  # exactly 0.20 above the base model (0.05)
        (0.24, None, "no-go", None),  # just below
        (0.24, 0.25, "go", 1e-4),  # 2e-5 misses, 1e-4 meets: switch, a deviation
        (0.24, 0.24, "no-go", None),  # both miss
        (0.90, 0.10, "go", 2e-5),  # 2e-5 meets: 1e-4 is not used
    ],
)
def test_go_rule(neutral, high, decision, lr):
    rates = {"base": 0.05, "neutral": neutral}
    if high is not None:
        rates["neutral_lr1e4"] = high
    result = go_decision(_judgments(rates))
    assert result["decision"] == decision
    assert result["grid_learning_rate"] == lr
    assert result["deviation"] == (lr == 1e-4)


def test_go_rule_uses_unrounded_rates():
    # 0.0499... vs 0.25: difference 0.2000...1 meets the rule; rounding would not change that,
    # but 0.2499 vs 0.05 (difference 0.1999) must miss.
    rows = _judgments({"base": 0.05}, n=10000) + _judgments({"neutral": 0.2499}, n=10000)
    assert go_decision(rows)["decision"] == "no-go"


def test_go_report_files(tmp_path):
    write_jsonl(tmp_path / "judgments.jsonl", _judgments({"base": 0.04, "neutral": 0.9}))
    result = write_go_report(tmp_path)
    assert result["decision"] == "go"
    assert json.loads((tmp_path / "go_report.json").read_text())["decision"] == "go"
    assert "Decision: **go**" in (tmp_path / "go_report.md").read_text()
