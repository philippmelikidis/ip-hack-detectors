"""Tables from judgment records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .records import write_text_atomic
from .stats import wilson

COLUMNS = [
    "condition",
    "n",
    "d1_hacks",
    "d1_rate",
    "d1_lo",
    "d1_hi",
    "correct",
    "correct_rate",
    "correct_lo",
    "correct_hi",
]


def rates_table(judgments: list[dict[str, Any]], conditions: list[str]) -> list[dict[str, Any]]:
    rows = []
    for cond in conditions:
        js = [j for j in judgments if j["condition"] == cond]
        n = len(js)
        hacks = sum(j["d1_hack"] for j in js)
        correct = sum(j["correct"] for j in js)
        h_lo, h_hi = wilson(hacks, n)
        c_lo, c_hi = wilson(correct, n)
        rows.append(
            {
                "condition": cond,
                "n": n,
                "d1_hacks": hacks,
                "d1_rate": hacks / n if n else 0.0,
                "d1_lo": h_lo,
                "d1_hi": h_hi,
                "correct": correct,
                "correct_rate": correct / n if n else 0.0,
                "correct_lo": c_lo,
                "correct_hi": c_hi,
            }
        )
    return rows


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_tables(rows: list[dict[str, Any]], out_dir: Path) -> None:
    csv = [",".join(COLUMNS)] + [",".join(_fmt(r[c]) for c in COLUMNS) for r in rows]
    write_text_atomic(out_dir / "table.csv", "\n".join(csv) + "\n")
    md = [
        "| Condition | n | D1 hack rate (95% CI) | Correct rate (95% CI) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['condition']} | {r['n']} "
            f"| {r['d1_rate']:.3f} [{r['d1_lo']:.3f}, {r['d1_hi']:.3f}] "
            f"| {r['correct_rate']:.3f} [{r['correct_lo']:.3f}, {r['correct_hi']:.3f}] |"
        )
    write_text_atomic(out_dir / "table.md", "\n".join(md) + "\n")
