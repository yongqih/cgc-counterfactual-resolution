"""Run the frozen Tahoe replicate-truth gate."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.analysis import run_truth_gate


if __name__ == "__main__":
    result = run_truth_gate(Path(__file__).resolve().parents[1])
    print(result["truth_label"])
    print(result)
