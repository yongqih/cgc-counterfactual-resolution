"""Execute the frozen CGC-BIO-2 analysis."""

from __future__ import annotations

import argparse
import json

from igc_virtual_cell.cgc_mechanism_followup.bio2 import run_bio2


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the frozen CGC-BIO-2 formal analysis")
    parser.add_argument(
        "--formal",
        action="store_true",
        help="required acknowledgement; without it no outcomes are read or written",
    )
    args = parser.parse_args()
    if not args.formal:
        parser.error("formal execution requires --formal")
    print(json.dumps(run_bio2(), indent=2))


if __name__ == "__main__":
    main()
