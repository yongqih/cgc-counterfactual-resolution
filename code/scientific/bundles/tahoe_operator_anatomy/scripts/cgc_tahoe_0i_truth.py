from __future__ import annotations

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.truth import run_truth


if __name__ == "__main__":
    run_truth(Path(__file__).resolve().parents[1])
