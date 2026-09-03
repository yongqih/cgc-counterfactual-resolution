from __future__ import annotations

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0b.inventory import freeze_and_download_small_metadata


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    manifest = freeze_and_download_small_metadata(repo)
    print(
        f"Downloaded {len(manifest['files'])} pinned metadata files "
        f"({manifest['downloaded_bytes'] / 1e6:.3f} MB) at revision {manifest['revision']}"
    )
