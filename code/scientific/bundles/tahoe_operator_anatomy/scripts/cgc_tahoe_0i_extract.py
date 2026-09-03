from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.extraction import extract_pseudobulks, verify_downloads


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("verify", "extract", "all"))
    args = parser.parse_args()
    if args.stage in {"verify", "all"}:
        verify_downloads(ROOT)
    if args.stage in {"extract", "all"}:
        extract_pseudobulks(ROOT)


if __name__ == "__main__":
    main()
