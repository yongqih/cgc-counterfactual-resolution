from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.integration import finalize


if __name__ == "__main__":
    finalize(Path(__file__).resolve().parents[1])
