from __future__ import annotations

import argparse
import json
from pathlib import Path

from igc_virtual_cell.cgc_m1_m2k92_confirmation import run_confirmation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell_CGC_entrywise"),
    )
    args = parser.parse_args()
    print(json.dumps(run_confirmation(args.root, args.upstream_root), indent=2))


if __name__ == "__main__":
    main()
