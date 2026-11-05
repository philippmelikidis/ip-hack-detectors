"""Command line: ipd reproduce --config <yaml> --out <dir>."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import reproduce


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ipd")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("reproduce", help="run the whole pipeline from scratch")
    rep.add_argument("--config", required=True, type=Path)
    rep.add_argument("--out", type=Path, default=None, help="default: outputs/<config name>")
    args = parser.parse_args(argv)

    if args.command == "reproduce":
        out = args.out or Path("outputs") / args.config.stem
        result = reproduce(args.config, out)
        print((result / "table.md").read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
