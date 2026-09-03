#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from igc_virtual_cell.pathway_resolution_scaling import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen fixed-PROGENy pathway-resolution scaling audit")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--truth-root",
        type=Path,
        default=Path(r"C:\Users\24119\PyCharmMiscProject\Virtual_Cell"),
    )
    parser.add_argument("--protocol-commit", type=str, default=None)
    args = parser.parse_args()
    if not args.formal:
        parser.error("--formal is required")
    root = args.root.resolve()
    protocol_commit = args.protocol_commit or subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    result = run(root, args.truth_root.resolve(), protocol_commit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

