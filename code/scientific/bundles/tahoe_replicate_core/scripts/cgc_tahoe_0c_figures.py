"""Generate all frozen CGC-SUPPORT-0C SVG figures."""

from pathlib import Path

from igc_virtual_cell.cgc_tahoe_0c.figures import generate_figures


if __name__ == "__main__":
    print("\n".join(generate_figures(Path(__file__).resolve().parents[1])))
