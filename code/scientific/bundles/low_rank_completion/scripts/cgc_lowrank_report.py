from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_lowrank_completion.reporting import write_reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    print(write_reports(args.root, args.source_root))


if __name__ == "__main__":
    main()
