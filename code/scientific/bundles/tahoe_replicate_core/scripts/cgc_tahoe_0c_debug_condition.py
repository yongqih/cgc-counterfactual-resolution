"""Inspect one source shard's row-group identity mapping."""

import argparse
from pathlib import Path

import pyarrow.parquet as pq

from igc_virtual_cell.cgc_tahoe_0b.analysis import HTTPRangeReader
from igc_virtual_cell.cgc_tahoe_0c.extraction import (
    _frozen_intervention_lookup,
    _mapped_row_group_key,
    _row_group_key,
    official_de_files,
    source_url,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("shard", type=int)
    parser.add_argument("pattern")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    item = official_de_files()[arguments.shard]
    reader = HTTPRangeReader(source_url(item["path"]), item["size_bytes"])
    parquet = pq.ParquetFile(reader)
    lookup = _frozen_intervention_lookup(root)
    for index in range(parquet.metadata.num_row_groups):
        group = parquet.metadata.row_group(index)
        raw = _row_group_key(group)
        if raw is not None and arguments.pattern.lower() in raw[2].lower():
            print(index, raw, _mapped_row_group_key(group, lookup))
