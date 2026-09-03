"""Finalize and verify the recovered compact Tahoe tensor locally."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.extraction import finalize_recovered_tensor


if __name__ == "__main__":
    result = finalize_recovered_tensor(Path(__file__).resolve().parents[1])
    print(result["tensor_shape"], result["tensor_sha256"])
