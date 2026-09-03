from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.loco import run_loco


if __name__ == "__main__":
    run_loco(Path(__file__).resolve().parents[1])
