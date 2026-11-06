"""Command line: `ipd reproduce --config <yaml> --out <dir>`, `ipd go-report --out <dir>` and
`ipd label --run <dir> --config <yaml> --sample pilot|grid`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from .config import load_config
from .pipeline import reproduce
from .runstate import RunStateError
from .sampling import PILOT_IP_CONDITION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ipd")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("reproduce", help="run the whole pipeline (from scratch unless --resume)")
    rep.add_argument("--config", required=True, type=Path)
    rep.add_argument("--out", type=Path, default=None, help="default: outputs/<config name>")
    mode = rep.add_mutually_exclusive_group()
    mode.add_argument(
        "--resume", action="store_true", help="skip units that are done (same config only)"
    )
    mode.add_argument(
        "--overwrite", action="store_true", help="discard done markers and recompute everything"
    )
    go = sub.add_parser("go-report", help="go/no-go decision of the pilot (prereg section 6)")
    go.add_argument("--out", required=True, type=Path, help="output directory of the pilot run")
    lab = sub.add_parser("label", help="draw a label sample and write the blinded CSV (section 8)")
    lab.add_argument("--run", required=True, type=Path, help="run directory with judgments.jsonl")
    lab.add_argument("--config", required=True, type=Path, help="config of the run (for the tasks)")
    lab.add_argument("--sample", required=True, choices=["pilot", "grid"])
    lab.add_argument("--out", type=Path, default=None, help="default: <run>/labeling")
    lab.add_argument(
        "--overwrite", action="store_true", help="replace an existing sample and key file"
    )
    lab.add_argument(
        "--pilot-ip-condition",
        default=PILOT_IP_CONDITION,
        help="pilot only: the IP Test-Specific run at the grid lr (section 6)",
    )
    args = parser.parse_args(argv)

    if args.command == "label":
        from .labeling import write_blinded
        from .records import read_jsonl
        from .sampling import grid_sample, pilot_sample
        from .tasks import load_tasks

        try:
            cfg, base_dir = load_config(args.config)
            tasks = {t.task_id: t for t in load_tasks(cfg.tasks, base_dir)}
            judgments = read_jsonl(args.run / "judgments.jsonl")
            if args.sample == "pilot":
                sample = pilot_sample(judgments, ip_condition=args.pilot_ip_condition)
            else:
                sample = grid_sample(judgments)
            paths = write_blinded(
                sample,
                judgments,
                tasks,
                args.out or args.run / "labeling",
                args.sample,
                overwrite=args.overwrite,
            )
        except KeyError as e:
            print(f"error: the judgment records lack the field {e}", file=sys.stderr)
            return 2
        except (OSError, ValueError, yaml.YAMLError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        for p in paths.values():
            print(p)
        return 0

    if args.command == "go-report":
        from .go import write_go_report

        try:
            write_go_report(args.out)
        except (OSError, ValueError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print((args.out / "go_report.md").read_text(encoding="utf-8"), end="")
        return 0

    if args.command == "reproduce":
        out = args.out or Path("outputs") / args.config.stem
        try:
            result = reproduce(args.config, out, resume=args.resume, overwrite=args.overwrite)
        except RunStateError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print((result / "table.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
