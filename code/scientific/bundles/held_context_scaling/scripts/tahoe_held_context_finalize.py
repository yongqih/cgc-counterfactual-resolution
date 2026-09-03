"""Finalize Tahoe held-context scaling. SPDX-License-Identifier: MIT"""

from pathlib import Path

from igc_virtual_cell.tahoe_held_context_finalize import finalize


if __name__ == "__main__":
    finalize(Path.cwd())

