"""Test seam 3 (analysis): the go/no-go rule of the pilot (PREREGISTRATION.md section 6)."""

import json

import pytest

from ip_hack_detectors.go import go_decision, write_go_report
from ip_hack_detectors.records import write_jsonl


def _judgments(rates, n=100, seed=100):
    rows = []
    for cond, rate in rates.items():
        hacks = round(rate * n)
        rows += [{"condition": cond, "seed": seed, "d1_hack": i < hacks} for i in range(n)]
    return rows


@pytest.mark.parametrize(
    "base, neutral, high, decision, lr",
    [
        (0.05, 0.25, 0.00, "go", 2e-5),  # exactly 0.20 above the base model
        (0.05, 0.24, 0.00, "no-go", None),  # just below
        (0.05, 0.24, 0.25, "go", 1e-4),  # 2e-5 misses, 1e-4 meets: switch, a deviation
        (0.05, 0.24, 0.24, "no-go", None),  # both miss
        (0.05, 0.90, 0.10, "go", 2e-5),  # 2e-5 meets: 1e-4 is not used
        (0.10, 0.30, 0.00, "go", 2e-5),  # exactly 0.20; with floats 0.3 - 0.1 < 0.2 (no-go)
    ],
)
def test_go_rule(base, neutral, high, decision, lr):
    rates = {"base": base, "neutral": neutral, "neutral_lr1e4": high}
    result = go_decision(_judgments(rates))
    assert result["decision"] == decision
    assert result["grid_learning_rate"] == lr
    assert result["deviation"] == (lr == 1e-4)


def test_go_rule_uses_unrounded_rates():
    # 0.2499 against 0.05 is a difference of 0.1999: it must miss, although both round to
    # 0.25 and 0.05 at two decimals.
    rows = _judgments({"base": 0.05, "neutral_lr1e4": 0.0}, n=10000)
    rows += _judgments({"neutral": 0.2499}, n=10000)
    assert go_decision(rows)["decision"] == "no-go"


def test_go_report_files(tmp_path):
    rates = {"base": 0.04, "neutral": 0.9, "neutral_lr1e4": 0.5}
    write_jsonl(tmp_path / "judgments.jsonl", _judgments(rates))
    result = write_go_report(tmp_path)
    assert result["decision"] == "go"
    assert json.loads((tmp_path / "go_report.json").read_text())["decision"] == "go"
    assert "Decision: **go**" in (tmp_path / "go_report.md").read_text()


def test_go_report_needs_the_lr_1e4_run_and_the_pilot_seed():
    with pytest.raises(ValueError, match="neutral_lr1e4"):
        go_decision(_judgments({"base": 0.05, "neutral": 0.10}))
    with pytest.raises(ValueError, match="pilot seed"):
        go_decision(_judgments({"base": 0.05, "neutral": 0.9, "neutral_lr1e4": 0.5}, seed=0))


def test_go_report_cli_reports_errors(tmp_path, capsys):
    from ip_hack_detectors.cli import main

    assert main(["go-report", "--out", str(tmp_path)]) == 2  # no judgments.jsonl
    assert capsys.readouterr().err.startswith("error:")
