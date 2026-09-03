"""Run frozen CGC-SUPPORT-0B integrity gates."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0b.validation import run_integrity_audit


if __name__ == "__main__":
    audit = run_integrity_audit(Path(__file__).resolve().parents[1])
    print(
        f"{audit['status']}: {audit['protocol_checks_passed']}/"
        f"{audit['protocol_checks_total']} integrity checks"
    )
