"""Generate final reporting artifacts from already-frozen CGC mechanism outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from igc_virtual_cell.cgc_mechanism_followup.common import OUT
from igc_virtual_cell.cgc_mechanism_followup.reporting import run_reporting


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reporting-only BIO-2/MULTI-2 finalizer; performs no scientific reruns."
    )
    parser.add_argument("--input-dir", type=Path, default=OUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_reporting(args.input_dir, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
