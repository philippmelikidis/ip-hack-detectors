"""Command line: `ipd reproduce --config <yaml> --out <dir>` and `ipd go-report --out <dir>`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import reproduce
from .runstate import RunStateError


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
    args = parser.parse_args(argv)

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
