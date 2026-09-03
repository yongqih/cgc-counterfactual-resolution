from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_lowrank_completion.runner import run_invariance, run_range


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--kind", choices=("real", "synthetic", "invariance"), required=True)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--stop-fold", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.kind == "invariance":
        print(run_invariance(args.root, args.source_root, args.start_fold, args.device))
    else:
        run_range(
            args.root,
            args.source_root,
            args.start_fold,
            args.stop_fold,
            kind=args.kind,
            device=args.device,
        )


if __name__ == "__main__":
    main()
