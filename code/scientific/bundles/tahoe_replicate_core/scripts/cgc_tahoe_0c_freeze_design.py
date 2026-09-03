"""Freeze nested supports and intervention folds before fitting."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.design import freeze_design


if __name__ == "__main__":
    print(freeze_design(Path(__file__).resolve().parents[1]))
