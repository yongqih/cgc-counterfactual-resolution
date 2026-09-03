"""Run frozen replicate-calibrated context-support scaling."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.scaling import run_scaling


if __name__ == "__main__":
    print(run_scaling(Path(__file__).resolve().parents[1]))
