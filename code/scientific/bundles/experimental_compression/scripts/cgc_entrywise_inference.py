from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_entrywise.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(run_inference(args.root))


if __name__ == "__main__":
    main()
