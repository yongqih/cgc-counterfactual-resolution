from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.dimensionality import run_dimensionality


if __name__ == "__main__":
    run_dimensionality(Path(__file__).resolve().parents[1])
