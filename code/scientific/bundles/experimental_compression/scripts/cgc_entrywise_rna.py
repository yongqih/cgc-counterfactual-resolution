from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_entrywise.secondary import build_rna_orders, run_rna_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("orders", "inference"):
        child = subparsers.add_parser(name)
        child.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(build_rna_orders(args.root) if args.command == "orders" else run_rna_inference(args.root))


if __name__ == "__main__":
    main()
