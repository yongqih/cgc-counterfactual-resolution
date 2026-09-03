"""Mechanically compact the large fit-level CSV without changing fit results."""

from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    source = root / "results/cgc_tahoe_0c/affine_fit_summary.csv"
    temporary = source.with_suffix(".compact.csv")
    drop = {
        "heldout_context_name",
        "support_hash",
        "opposite_plate_evaluated",
        "rcond",
    }
    first = True
    for chunk in pd.read_csv(source, chunksize=50_000):
        chunk = chunk.drop(columns=[column for column in drop if column in chunk])
        chunk.to_csv(
            temporary,
            mode="w" if first else "a",
            header=first,
            index=False,
            float_format="%.10g",
        )
        first = False
    temporary.replace(source)
    print(source.stat().st_size)
