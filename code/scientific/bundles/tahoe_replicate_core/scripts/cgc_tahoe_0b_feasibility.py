from __future__ import annotations

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0b.analysis import run_feasibility_audit


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    verdict = run_feasibility_audit(repo)
    print(
        f"{verdict['verdict']}: {verdict['selected_core_dimensions']} "
        f"with {verdict['common_pairs_including_control']} common matched pairs"
    )
