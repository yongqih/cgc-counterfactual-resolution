"""Run Tahoe held-context secondary analyses. SPDX-License-Identifier: MIT"""

from __future__ import annotations

import argparse
from pathlib import Path

from igc_virtual_cell.tahoe_held_context_secondary import run_secondary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("domain", "controls", "all"))
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    run_secondary(Path.cwd(), args.source_root, args.command)

