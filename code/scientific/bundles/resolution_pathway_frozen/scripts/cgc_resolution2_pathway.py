from __future__ import annotations

import argparse
import json
from pathlib import Path

from igc_virtual_cell.cgc_resolution_poc.pathway import run_pathway_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen RESOLUTION-2 pathway analysis")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--truth-root", type=Path,
                        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell"))
    args = parser.parse_args()
    if not args.formal:
        parser.error("--formal is required")
    print(json.dumps(run_pathway_analysis(args.root, args.truth_root), indent=2))


if __name__ == "__main__":
    main()
