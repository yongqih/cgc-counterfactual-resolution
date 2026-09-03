"""Probe Tahoe DE shard condition bounds without reading DE values."""

import argparse
from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.extraction import probe_shard_bounds, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("indices", nargs="+", type=int)
    parser.add_argument(
        "--output", default="results/cgc_tahoe_0c/shard_boundary_probe.json"
    )
    arguments = parser.parse_args()
    result = probe_shard_bounds(arguments.indices)
    write_json(Path(arguments.output), result)
    for row in result:
        print(row["shard_index"], row["first"]["plate"], row["last"]["plate"])


if __name__ == "__main__":
    main()
