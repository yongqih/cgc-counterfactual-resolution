"""Download selected Tahoe shards and build the compact core tensor."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.extraction import (
    build_core_tensor_remote,
)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    tensor = build_core_tensor_remote(root)
    print(tensor["tensor_shape"], tensor["tensor_sha256"])
