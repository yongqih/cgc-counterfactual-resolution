"""Retrieve real full-gene measurements for matched split-culture clones."""
from pathlib import Path
import gzip
import hashlib
import json
import time
import urllib.request

import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import save_npz

HERE = Path(__file__).resolve().parent
FOLDER = HERE / "larry"
BASE = "https://kleintools.hms.harvard.edu/paper_websites/state_fate2020/"


def download(name):
    path = FOLDER / name
    if path.exists():
        print("Existing", name, path.stat().st_size, flush=True)
        return path
    partial = path.with_suffix(path.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(BASE + name, headers={"User-Agent": "PublicResearchDataAudit/1.0", "Range": f"bytes={offset}-"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if offset:
            assert response.status == 206
            assert response.headers["Content-Range"].startswith(f"bytes {offset}-")
        total = offset + int(response.headers["Content-Length"])
        with partial.open("ab") as output:
            last = time.monotonic()
            n = offset
            while data := response.read(8 * 1024 ** 2):
                output.write(data)
                n += len(data)
                if time.monotonic() - last > 15:
                    print(json.dumps({"file": name, "downloaded_bytes": n, "total_bytes": total}), flush=True)
                    last = time.monotonic()
    assert partial.stat().st_size == total
    partial.replace(path)
    print("Downloaded", name, total, flush=True)
    return path


def family(name):
    if name.startswith("LSK_"):
        return "LSK"
    if name.startswith("LK_"):
        return "LK"
    return "LK_second"


def main():
    meta = pd.read_csv(FOLDER / "stateFate_inVitro_metadata.txt.gz", sep="\t")
    with gzip.open(FOLDER / "stateFate_inVitro_clone_matrix.mtx.gz", "rb") as stream:
        clonal = mmread(stream, spmatrix=True).tocsr()
    labels = np.full(len(meta), -1, dtype=int)
    positive = np.diff(clonal.indptr) == 1
    labels[positive] = clonal.indices
    meta["clone_index"] = labels
    meta["culture_family"] = meta["Library"].map(family)
    meta["original_cell_index"] = np.arange(len(meta))
    n_families = meta.loc[positive].groupby("clone_index")["culture_family"].nunique()
    valid = set(n_families.index[n_families == 1])
    early = set(meta.loc[(meta["Time point"] == 2) & positive, "clone_index"])
    l1 = set(meta.loc[(meta["Time point"] == 6) & (meta.Well == 1) & positive, "clone_index"])
    l2 = set(meta.loc[(meta["Time point"] == 6) & (meta.Well == 2) & positive, "clone_index"])
    complete = sorted(early & l1 & l2 & valid)
    selected = meta[meta.clone_index.isin(complete)].copy()
    selected.to_csv(FOLDER / "branch_validation_metadata.csv", index=False)
    result = {"complete_triples_before_family_screen": len(early & l1 & l2),
              "complete_triples_after_family_screen": len(complete),
              "n_clone_ids_crossing_culture_families": int((n_families > 1).sum()),
              "selected_cell_rows": len(selected),
              "input_definition": "The actual day-2 RNA record sampled from a common clonal source before partition into two culture groups; all its available cells and genes can be used. It is shared experimental antecedent information, not a measurement of each future daughter cell.",
              "output_definition": "Realized day-6 clonal population RNA/composition in a culture group under the nominally matched differentiation protocol.",
              "unproven_assumptions": ["Barcode fidelity within culture family", "Representative cell sampling", "No well-associated measurement artifact accounting for the endpoint contrast"],
              "claim_boundary": "A contradiction for deterministic prediction of a realized branch from this shared antecedent does not establish that an individually measured daughter's noiseless RNA is insufficient, nor rule out conditional mean/distribution prediction."}
    (FOLDER / "branch_validation_design.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Selection", json.dumps(result), flush=True)
    genes_path = download("stateFate_inVitro_gene_names.txt.gz")
    matrix_path = download("stateFate_inVitro_normed_counts.mtx.gz")
    digest = hashlib.sha256()
    with matrix_path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 ** 2):
            digest.update(chunk)
    source = {"url": BASE + matrix_path.name, "bytes": matrix_path.stat().st_size,
              "sha256": digest.hexdigest(), "measurement": "Published total-count-normalized UMI matrix, not integer raw UMI counts"}
    (FOLDER / "expression_source_manifest.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    print("Reading full sparse matrix", flush=True)
    with gzip.open(matrix_path, "rb") as stream:
        expression = mmread(stream, spmatrix=True).tocsr()
    assert expression.shape[0] == len(meta)
    indices = selected.original_cell_index.to_numpy()
    subset = expression[indices].astype(np.float32)
    with gzip.open(genes_path, "rt") as stream:
        genes = [line.rstrip("\n") for line in stream]
    assert len(genes) == expression.shape[1]
    assert subset.shape == (len(selected), len(genes))
    save_npz(FOLDER / "branch_validation_full_gene.npz", subset)
    (FOLDER / "branch_validation_gene_names.json").write_text(json.dumps(genes), encoding="utf-8")
    print(json.dumps({"retained_shape": subset.shape, "retained_nnz": subset.nnz,
                      "retained_bytes": (FOLDER / "branch_validation_full_gene.npz").stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
