from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.nuisance import run_nuisance


if __name__ == "__main__":
    run_nuisance(Path(__file__).resolve().parents[1])
