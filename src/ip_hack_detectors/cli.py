"""Command line: ipd reproduce --config <yaml> --out <dir>."""

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
    args = parser.parse_args(argv)

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
