import json

import pytest

from ip_hack_detectors.addendum_a1 import LOSS_PLOT, analyze_a1, loss_summary, prefix_rates
from ip_hack_detectors.records import write_jsonl


def _report(path, losses):
    path.mkdir(parents=True)
    rows = [{"learning_rate": 1e-5, "loss": x, "step": i + 1} for i, x in enumerate(losses)]
    (path / "train_report.json").write_text(json.dumps({"losses": rows[::-1]}))


def _models(tmp_path):
    models = tmp_path / "models"
    _report(models / "neutral" / "seed0", [2.0, 1.0, 0.5])
    _report(models / "neutral" / "seed1", [3.0, 2.0, 1.5])
    _report(models / "ip_test_specific" / "seed0", [1.0, 0.9, 0.8])
    (models / "ip_test_specific" / "seed1").mkdir()  # no report yet: skipped
    return models


def _judgments(cond, hacks, correct, tasks=4, samples=5):
    """`hacks` and `correct` outputs per task out of `samples`."""
    return [
        {
            "condition": cond,
            "seed": 100,
            "task_id": t,
            "sample_idx": i,
            "d1_hack": i < hacks,
            "correct": hacks <= i < hacks + correct,
        }
        for t in range(tasks)
        for i in range(samples)
    ]


def test_loss_summary_means_over_seeds_in_step_order(tmp_path):
    s = loss_summary(_models(tmp_path))
    n = s["neutral"]
    assert n["seeds"] == [0, 1] and n["steps"] == 3
    assert n["first_loss_mean"] == pytest.approx(2.5)
    assert n["last_loss_mean"] == pytest.approx(1.0)
    assert n["drop_mean"] == pytest.approx(1.5)
    assert n["curve_mean"] == pytest.approx([2.5, 1.5, 1.0])
    assert s["ip_test_specific"]["seeds"] == [0]


def test_loss_summary_rejects_seeds_with_different_steps(tmp_path):
    models = tmp_path / "models"
    _report(models / "neutral" / "seed0", [2.0, 1.0])
    _report(models / "neutral" / "seed1", [2.0, 1.0, 0.5])
    with pytest.raises(ValueError, match="different"):
        loss_summary(models)


def test_prefix_rates_known_rates_and_constant_tasks_give_point_intervals():
    js = _judgments("base", 1, 2) + _judgments("base_prefix_ip_test_specific", 3, 1)
    js += _judgments("neutral", 5, 0)  # not an A1.2 condition: ignored
    r = prefix_rates(js, b=200)
    assert list(r) == ["base", "base_prefix_ip_test_specific"]
    base = r["base"]
    assert base["d1_hack"]["estimate"] == pytest.approx(0.2)
    assert base["correct"]["estimate"] == pytest.approx(0.4)
    # every task has the same rate, so every replicate equals the estimate
    assert base["d1_hack"]["ci_low"] == pytest.approx(0.2)
    assert base["d1_hack"]["ci_high"] == pytest.approx(0.2)
    assert r["base_prefix_ip_test_specific"]["d1_hack"]["estimate"] == pytest.approx(0.6)
    assert r["base_prefix_ip_test_specific"]["d1_hack"]["n_outputs"] == 20


def test_prefix_rates_reproducible_with_fixed_seed():
    js = [
        {**j, "d1_hack": j["task_id"] % 2 == 0}
        for j in _judgments("base", 0, 0, tasks=10) + _judgments("base_prefix_python3", 0, 0, 10)
    ]
    assert prefix_rates(js, b=300) == prefix_rates(js, b=300)


def _outputs(judgments, fenced):
    """One output per judgment; the first `fenced` samples of a task have a closed block."""
    return [
        {
            **{k: j[k] for k in ("condition", "seed", "task_id", "sample_idx")},
            "text": "```python\ndef f():\n    return 1\n```" if j["sample_idx"] < fenced else "x",
        }
        for j in judgments
    ]


def test_prefix_rates_gives_the_a1_3_share_per_run():
    base = _judgments("base", 1, 1)
    pre = _judgments("base_prefix_python3", 1, 1)
    outputs = _outputs(base, 3) + _outputs(pre, 0)
    r = prefix_rates(base + pre, outputs, b=50)
    assert r["base"]["no_python_block"] == {"count": 8, "share": 0.4, "n_outputs": 20}
    assert r["base_prefix_python3"]["no_python_block"]["share"] == 1.0
    assert prefix_rates(base, b=50)["base"]["no_python_block"] is None


def test_analyze_a1_writes_files(tmp_path):
    pilot = tmp_path / "pilot"
    js = _judgments("base", 1, 1)
    write_jsonl(pilot / "judgments.jsonl", js)
    write_jsonl(pilot / "outputs.jsonl", _outputs(js, 5))
    out = tmp_path / "out"
    result = analyze_a1(_models(tmp_path), pilot, out)
    assert (out / LOSS_PLOT).stat().st_size > 0
    assert json.loads((out / "a1.json").read_text())["a1_1_loss"]["neutral"]["steps"] == 3
    md = (out / "a1.md").read_text()
    assert "| neutral | 2 | 3 | 2.500 | 1.000 | 1.500 |" in md
    assert "| empty prefix (base model) | 20 |" in md and "| 0.0000 (0) |" in md
    assert result["a1_2_prefixes"]["base"]["d1_hack"]["estimate"] == pytest.approx(0.2)
