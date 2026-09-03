"""Run the frozen Tahoe five-pathway held-context scaling audit.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.tahoe_five_pathway_scaling import fit_and_finalize, prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "finalize", "all"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = args.source_root.resolve()
    if args.command in {"prepare", "all"}:
        prepare(root, source_root)
    if args.command in {"finalize", "all"}:
        fit_and_finalize(root)


if __name__ == "__main__":
    main()
