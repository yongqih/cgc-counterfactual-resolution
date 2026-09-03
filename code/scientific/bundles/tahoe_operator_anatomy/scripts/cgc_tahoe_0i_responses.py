from __future__ import annotations

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0i.responses import construct_responses


if __name__ == "__main__":
    construct_responses(Path(__file__).resolve().parents[1])
