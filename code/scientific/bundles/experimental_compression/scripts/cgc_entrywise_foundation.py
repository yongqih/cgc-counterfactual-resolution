from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_entrywise.foundation import build_foundation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = build_foundation(args.root, args.source_root, args.force)
    print(manifest)


if __name__ == "__main__":
    main()
