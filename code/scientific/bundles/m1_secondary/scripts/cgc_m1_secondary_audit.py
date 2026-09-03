from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_m1_secondary import run_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell_CGC_entrywise"),
    )
    args = parser.parse_args()
    run_audit(args.root.resolve(), args.upstream_root.resolve())


if __name__ == "__main__":
    main()
