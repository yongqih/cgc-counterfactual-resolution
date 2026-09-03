from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.scaling import build_gram_cache, run_scaling


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("grams", "scaling", "all"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.stage in {"grams", "all"}:
        build_gram_cache(root)
    if args.stage in {"scaling", "all"}:
        run_scaling(root)
