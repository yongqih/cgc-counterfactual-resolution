from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_entrywise.analysis import aggregate_rna_targets, aggregate_targets, run_surface


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--target-start", type=int, default=0)
    run.add_argument("--target-stop", type=int, default=50)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--support-mode", choices=("random", "rna"), default="random")
    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, default=Path.cwd())
    aggregate.add_argument("--support-mode", choices=("random", "rna"), default="random")
    args = parser.parse_args()
    if args.command == "run":
        print(
            run_surface(
                args.root,
                args.overwrite,
                args.target_start,
                args.target_stop,
                args.support_mode,
            )
        )
    else:
        print(
            aggregate_targets(args.root)
            if args.support_mode == "random"
            else aggregate_rna_targets(args.root)
        )


if __name__ == "__main__":
    main()
