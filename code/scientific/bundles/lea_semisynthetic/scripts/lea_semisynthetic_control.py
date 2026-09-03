from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.lea_semisynthetic import (
    benchmark_runtime,
    run_formal,
    write_input_reconciliation,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("reconcile", "benchmark", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell"),
    )
    args = parser.parse_args()
    if args.command == "reconcile":
        print(write_input_reconciliation(args.root, args.source_root))
    elif args.command == "benchmark":
        print(benchmark_runtime(args.root, args.source_root))
    else:
        print(run_formal(args.root, args.source_root))


if __name__ == "__main__":
    main()

