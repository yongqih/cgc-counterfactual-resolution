from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_lowrank_completion.feasibility import run_feasibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(run_feasibility(args.root, args.source_root, args.device))


if __name__ == "__main__":
    main()
