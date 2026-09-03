from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_context_anchoring import reconcile_sources, run_formal_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("reconcile", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--authority-root",
        type=Path,
        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell_CGC_entrywise"),
    )
    args = parser.parse_args()
    if args.command == "reconcile":
        result = reconcile_sources(args.root, args.authority_root)
        print({"checks": len(result), "all_passed": bool(result["passed"].all())})
    else:
        print(run_formal_analysis(args.root, args.authority_root))


if __name__ == "__main__":
    main()

