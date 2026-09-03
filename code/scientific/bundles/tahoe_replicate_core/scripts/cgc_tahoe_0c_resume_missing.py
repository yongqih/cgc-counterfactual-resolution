"""Resume the audited Selinexor-only extraction gap."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.extraction import resume_missing_conditions


if __name__ == "__main__":
    result = resume_missing_conditions(Path(__file__).resolve().parents[1])
    print(result["tensor_shape"], result["tensor_sha256"])
