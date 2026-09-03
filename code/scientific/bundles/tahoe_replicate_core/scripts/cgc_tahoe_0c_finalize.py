"""Write final 0C reports and run all integrity gates."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.reporting import run_integrity_audit, write_reports


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    write_reports(root)
    print(run_integrity_audit(root))
