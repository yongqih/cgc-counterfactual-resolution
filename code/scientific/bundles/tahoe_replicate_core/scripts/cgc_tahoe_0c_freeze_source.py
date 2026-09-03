"""Freeze official Tahoe estimator evidence and 0B provenance."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.source import freeze_source


if __name__ == "__main__":
    manifest = freeze_source(Path(__file__).resolve().parents[1])
    print(manifest["estimator"], manifest["estimator_verified"])
