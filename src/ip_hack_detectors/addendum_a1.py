"""Descriptive measurements of addendum A1 (PREREGISTRATION.md section 12).

A1.1: the training loss of every optimizer step, per trained condition the curve (mean over
seeds per step) and the mean over seeds of the first and the last step's loss. A1.2: the
untrained base model in the pilot with each training prefix as eval prefix, D1 hack rate and
correct rate with 95 % bootstrap intervals over tasks, next to the base model with the empty
prefix. A1.3 is part of `rates.output_shares`, A1.4 of `quality`. Descriptive only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .bootstrap import SEED, B
from .detect import _git_commit
from .rates import CONDITION_NAMES, RateSet, rate
from .records import read_jsonl, write_text_atomic

# A1.2: the base model with the empty prefix first, then the three prefix runs.
PREFIX_CONDITIONS = {
    "base": "empty prefix (base model)",
    "base_prefix_ip_test_specific": "IP Test-Specific",
    "base_prefix_ip_pass_test": "IP Pass-Test",
    "base_prefix_python3": "Python 3 control",
}
LOSS_PLOT = "loss_curves.png"


def _losses(report: Path) -> list[float]:
    """The loss of every optimizer step of one training run, in step order."""
    rows = json.loads(report.read_text(encoding="utf-8"))["losses"]
    return [float(r["loss"]) for r in sorted(rows, key=lambda r: r["step"])]


def loss_summary(models_dir: Path) -> dict[str, Any]:
    """A1.1 per trained condition (`<models_dir>/<condition>/seed<n>/train_report.json`)."""
    out: dict[str, Any] = {}
    for cond_dir in sorted(p for p in Path(models_dir).iterdir() if p.is_dir()):
        runs = {
            int(d.name.removeprefix("seed")): _losses(d / "train_report.json")
            for d in sorted(cond_dir.glob("seed*"))
            if (d / "train_report.json").is_file()
        }
        if not runs:
            continue
        lengths = {len(v) for v in runs.values()}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError(f"{cond_dir}: the seeds have different or no steps: {lengths}")
        n = len(runs)
        first = sum(v[0] for v in runs.values()) / n
        last = sum(v[-1] for v in runs.values()) / n
        out[cond_dir.name] = {
            "seeds": sorted(runs),
            "steps": lengths.pop(),
            "first_loss_mean": first,
            "last_loss_mean": last,
            "drop_mean": first - last,
            "curve_mean": [sum(step) / n for step in zip(*runs.values(), strict=True)],
            "per_seed": {str(s): {"first": v[0], "last": v[-1]} for s, v in runs.items()},
        }
    return out


def prefix_rates(judgments: list[dict[str, Any]], *, b: int = B, seed: int = SEED) -> dict:
    """A1.2: D1 hack rate and correct rate of the base model with each eval prefix of the pilot.

    All four runs are untrained, so only tasks are resampled and the outputs of a task stay
    together, as for the base model in section 9; the same task draw is used for all four."""
    present = [c for c in PREFIX_CONDITIONS if any(j["condition"] == c for j in judgments)]
    records = [j for j in judgments if j["condition"] in present]
    if not records:
        return {}
    task_ids = sorted({j["task_id"] for j in records})
    rs = RateSet(records, {c: None for c in present}, task_ids, b=b, seed=seed)
    return {
        c: {"prefix": PREFIX_CONDITIONS[c], **{f: rate(rs, c, f) for f in ("d1_hack", "correct")}}
        for c in present
    }


def plot_losses(losses: dict[str, Any], path: Path) -> None:
    """A1.1: the mean loss curve over seeds of every trained condition."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(7, 4.2))
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    for cond, s in losses.items():
        steps = range(1, len(s["curve_mean"]) + 1)
        ax.plot(steps, s["curve_mean"], lw=1.5, label=CONDITION_NAMES.get(cond, cond))
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Training loss (mean over seeds)")
    ax.set_title("Training loss per step (addendum A1.1)")
    ax.grid(color="#e0e0e0", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def _ci(r: dict[str, Any]) -> str:
    return f"{r['estimate']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def a1_md(losses: dict[str, Any], prefixes: dict[str, Any]) -> str:
    lines = [
        "# Addendum A1 (PREREGISTRATION.md section 12), descriptive",
        "",
        "## A1.1 Training loss",
        "",
        f"Mean over seeds of the first and the last optimizer step's loss; curves in {LOSS_PLOT}.",
        "",
        "| Condition | Seeds | Steps | First step | Last step | Drop |",
        "|---|---|---|---|---|---|",
    ]
    for cond, s in losses.items():
        lines.append(
            f"| {CONDITION_NAMES.get(cond, cond)} | {len(s['seeds'])} | {s['steps']} | "
            f"{s['first_loss_mean']:.3f} | {s['last_loss_mean']:.3f} | {s['drop_mean']:.3f} |"
        )
    lines += [
        "",
        "## A1.2 Base model with the training prefixes (pilot, seed 100)",
        "",
        "Estimate [95 % percentile interval], bootstrap over tasks (B = 10,000, seed 0), the "
        "outputs of a task kept together.",
        "",
        "| Eval prefix | Outputs | D1 hack | Correct |",
        "|---|---|---|---|",
    ]
    for r in prefixes.values():
        lines.append(
            f"| {r['prefix']} | {r['d1_hack']['n_outputs']} | {_ci(r['d1_hack'])} | "
            f"{_ci(r['correct'])} |"
        )
    return "\n".join(lines) + "\n"


def analyze_a1(models_dir: Path, pilot_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Write `a1.json`, `a1.md` and the loss plot into `out_dir` and return the JSON content."""
    out_dir = Path(out_dir)
    losses = loss_summary(models_dir)
    prefixes = prefix_rates(read_jsonl(Path(pilot_dir) / "judgments.jsonl"))
    result = {"a1_1_loss": losses, "a1_2_prefixes": prefixes, "git_commit": _git_commit()}
    write_text_atomic(out_dir / "a1.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_text_atomic(out_dir / "a1.md", a1_md(losses, prefixes))
    if losses:
        plot_losses(losses, out_dir / LOSS_PLOT)
    return result
