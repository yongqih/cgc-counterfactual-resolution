from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.panel127 import compare_panel


if __name__ == "__main__":
    compare_panel(Path(__file__).resolve().parents[1])
