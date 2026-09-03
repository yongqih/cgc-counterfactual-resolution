from __future__ import annotations

import argparse
import json
from pathlib import Path

from igc_virtual_cell.cgc_resolution_poc.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("C:/Users/24119/PyCharmMiscProject/Virtual_Cell_CGC_entrywise"))
    args = parser.parse_args()
    print(json.dumps(run_inference(args.root, args.source_root), indent=2))


if __name__ == "__main__":
    main()
