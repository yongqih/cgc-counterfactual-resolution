"""Scan Tahoe DE parquet footers for the frozen Plate 6/14 core."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.extraction import scan_all_shards


if __name__ == "__main__":
    result = scan_all_shards(Path(__file__).resolve().parents[1])
    print(
        result["selected_full_shards"],
        result["projected_total_transfer_bytes"] / 1e9,
        result["unique_selected_conditions"],
    )
