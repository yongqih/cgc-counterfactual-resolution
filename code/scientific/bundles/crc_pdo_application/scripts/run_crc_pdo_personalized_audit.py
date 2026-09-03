from __future__ import annotations

import argparse
import json
from pathlib import Path

from igc_virtual_cell.crc_pdo_personalized import build_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = build_audit(repo_root=args.repo_root.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
