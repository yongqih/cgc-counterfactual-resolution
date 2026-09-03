"""Materialize study-separated perturbation deltas for Phase I.

Only processed count matrices or author-released LFC tables are read.  The
materialized matrices are local cache artifacts and are never pooled across
studies.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from igc_virtual_cell.data.materialize_feng import _normalized_cell_ids


META_COLUMNS = [
    "study",
    "context_id",
    "donor_id",
    "perturbation_id",
    "cell_count",
    "response_source",
]


def _ntc_matrix(path: Path, contexts: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if "observation_id" not in frame:
        raise ValueError(f"Missing observation_id in {path}")
    manifest = contexts.loc[contexts["is_control"].astype(str).str.lower().eq("true")]
    lookup = manifest.set_index("observation_id")["context_id"].astype(str)
    frame["context_id"] = frame["observation_id"].astype(str).map(lookup)
    if frame["context_id"].isna().any():
        raise ValueError(f"Unmapped NTC observations in {path}")
    return frame.drop(columns="observation_id").set_index("context_id").sort_index()


def _write_delta(
    metadata: pd.DataFrame,
    values: np.ndarray,
    genes: list[str],
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = metadata.reset_index(drop=True).copy()
    matrix = pd.DataFrame(values.astype(np.float32, copy=False), columns=genes)
    pd.concat([meta, matrix], axis=1).to_parquet(path, index=False, compression="zstd")
    return path


def _feng_maps(root: Path, screen: str) -> tuple[dict[str, str], dict[str, str]]:
    contexts = pd.read_csv(root / "results/tables/feng_2026_context_identifiers.csv")
    contexts = contexts.loc[contexts["screen"].eq(screen)]
    context_map = contexts.set_index("metadata_cell_line")["context_id"].astype(str).to_dict()
    donor_map = contexts.set_index("context_id")["donor_id"].astype(str).to_dict()
    return context_map, donor_map


def materialize_feng_genomewide(root: Path, output: Path, *, min_cells: int = 5, min_contexts: int = 5) -> Path:
    destination = output / f"feng_genomewide_deltas_min{min_cells}.parquet"
    if destination.exists():
        return destination
    raw = root / "data/raw/feng_2026"
    manifest = pd.read_csv(root / "data/manifests/feng_2026_genomewide.csv", low_memory=False)
    ntc_path = root / "data/processed/feng_2026/feng_2026_genomewide_ntc_log1p_cpm.parquet"
    ntc = _ntc_matrix(ntc_path, manifest)
    genes = list(map(str, ntc.columns))
    gene_to_index = {gene: index for index, gene in enumerate(genes)}

    counts = pd.read_csv(root / "results/tables/feng_2026_cell_counts_by_perturbation_context.csv")
    counts = counts.loc[counts["screen"].eq("genomewide")].copy()
    coverage = (
        counts.loc[counts["cell_count"].ge(min_cells)]
        .groupby("perturbation_id")["context_id"]
        .nunique()
    )
    selected = set(coverage.loc[coverage.ge(min_contexts)].index.astype(str))
    groups = counts.loc[
        counts["perturbation_id"].astype(str).isin(selected) & counts["cell_count"].ge(min_cells)
    ].copy()
    groups = groups.sort_values(["context_id", "perturbation_id"]).reset_index(drop=True)
    group_lookup = {
        (str(row.context_id), str(row.perturbation_id)): index
        for index, row in enumerate(groups.itertuples(index=False))
    }
    sums = np.zeros((len(groups), len(genes)), dtype=np.float64)
    context_map, donor_map = _feng_maps(root, "genomewide")

    partitions = (
        (
            "GenomeWideScreen_FitnessGenes",
            "GenomeWideScreen_FitnessGenes_Cell-Metadata.tsv.gz",
            "GenomeWideScreen_FitnessGenes_RNA-UMI-Counts.csv.gz",
        ),
        (
            "GenomeWideScreen_NonFitnessGenes",
            "GenomeWideScreen_NonFitnessGenes_Cell-Metadata.tsv.gz",
            "GenomeWideScreen_NonFitnessGenes_RNA-UMI-Counts.csv.gz",
        ),
    )
    observed_cells = np.zeros(len(groups), dtype=np.int64)
    for partition, metadata_name, count_name in partitions:
        metadata = pd.read_csv(raw / metadata_name, sep="\t")
        metadata["count_cell_id"] = _normalized_cell_ids(metadata, partition)
        metadata = metadata.set_index("count_cell_id")
        with gzip.open(raw / count_name, "rb") as handle:
            header = handle.readline().rstrip(b"\r\n").decode("utf-8").split(",")[1:]
            aligned = metadata.reindex(header)
            context = aligned["Cell_Line"].astype(str).map(context_map)
            guide = aligned["Guide_Call"].fillna("").astype(str)
            perturbation = guide.str.rsplit("_", n=1).str[0]
            codes = np.fromiter(
                (
                    group_lookup.get((str(c), str(p)), -1)
                    for c, p in zip(context, perturbation, strict=True)
                ),
                dtype=np.int64,
                count=len(header),
            )
            selected_cells = codes >= 0
            observed_cells += np.bincount(codes[selected_cells], minlength=len(groups))
            selected_codes = codes[selected_cells]
            for line in handle:
                comma = line.find(b",")
                if comma < 0:
                    continue
                label = line[:comma].decode("utf-8")
                parts = label.split(":")
                gene = parts[1] if len(parts) > 1 else parts[0]
                gene_index = gene_to_index.get(gene)
                if gene_index is None:
                    continue
                values = np.fromstring(line[comma + 1 :], dtype=np.float64, sep=",")
                if len(values) != len(header):
                    raise ValueError(f"Count row length mismatch for {gene} in {count_name}")
                sums[:, gene_index] += np.bincount(
                    selected_codes,
                    weights=values[selected_cells],
                    minlength=len(groups),
                )

    expected_cells = groups["cell_count"].to_numpy(dtype=np.int64)
    if not np.array_equal(observed_cells, expected_cells):
        bad = np.flatnonzero(observed_cells != expected_cells)[:10]
        raise ValueError(f"Feng group cell counts disagree at rows {bad.tolist()}")
    libraries = sums.sum(axis=1)
    if np.any(libraries <= 0):
        raise ValueError("Zero library in Feng genomewide perturbation pseudobulk")
    normalized = np.log1p(sums / libraries[:, None] * 1_000_000)
    baselines = ntc.loc[groups["context_id"], genes].to_numpy(dtype=np.float64)
    deltas = normalized - baselines
    metadata = pd.DataFrame(
        {
            "study": "feng_genomewide",
            "context_id": groups["context_id"].astype(str),
            "donor_id": groups["context_id"].astype(str).map(donor_map),
            "perturbation_id": groups["perturbation_id"].astype(str),
            "cell_count": groups["cell_count"].astype(int),
            "response_source": "processed_UMI_pseudobulk_minus_NTC_log1p_CPM",
        }
    )
    return _write_delta(metadata, deltas, genes, destination)


def materialize_feng_targeted(root: Path, output: Path) -> Path:
    destination = output / "feng_targeted_deltas.parquet"
    if destination.exists():
        return destination
    manifest = pd.read_csv(root / "data/manifests/feng_2026_targeted.csv", low_memory=False)
    ntc_path = root / "data/processed/feng_2026/feng_2026_targeted_ntc_log1p_cpm.parquet"
    ntc = _ntc_matrix(ntc_path, manifest)
    genes = list(map(str, ntc.columns))
    gene_lookup = {gene: index for index, gene in enumerate(genes)}
    context_map, donor_map = _feng_maps(root, "targeted")
    line_by_context = {context: line for line, context in context_map.items()}

    counts = pd.read_csv(root / "results/tables/feng_2026_cell_counts_by_perturbation_context.csv")
    counts = counts.loc[counts["screen"].eq("targeted")].copy()
    groups = counts.sort_values(["context_id", "perturbation_id"]).reset_index(drop=True)
    group_lookup = {
        (str(line_by_context[str(row.context_id)]), str(row.perturbation_id)): index
        for index, row in enumerate(groups.itertuples(index=False))
    }
    matrix = np.full((len(groups), len(genes)), np.nan, dtype=np.float32)
    path = root / "data/raw/feng_2026/TargetedScreen_LFC_byGene-perLine.tsv.gz"
    usecols = ["Target", "Expressed_Gene_Symbol", "Cell_Line", "lfc"]
    for chunk in pd.read_csv(path, sep="\t", usecols=usecols, chunksize=1_000_000):
        row_code = np.fromiter(
            (
                group_lookup.get((str(line), str(target)), -1)
                for line, target in zip(chunk["Cell_Line"], chunk["Target"], strict=True)
            ),
            dtype=np.int64,
            count=len(chunk),
        )
        gene_code = chunk["Expressed_Gene_Symbol"].astype(str).map(gene_lookup).fillna(-1).to_numpy(dtype=np.int64)
        keep = (row_code >= 0) & (gene_code >= 0)
        matrix[row_code[keep], gene_code[keep]] = chunk.loc[keep, "lfc"].to_numpy(dtype=np.float32)
    present = np.isfinite(matrix).sum(axis=1)
    keep_rows = present >= max(1, int(0.95 * len(genes)))
    groups = groups.loc[keep_rows].reset_index(drop=True)
    matrix = matrix[keep_rows]
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=0.0)
    metadata = pd.DataFrame(
        {
            "study": "feng_targeted",
            "context_id": groups["context_id"].astype(str),
            "donor_id": groups["context_id"].astype(str).map(donor_map),
            "perturbation_id": groups["perturbation_id"].astype(str),
            "cell_count": groups["cell_count"].astype(int),
            "response_source": "author_per_line_LFC",
        }
    )
    return _write_delta(metadata, matrix, genes, destination)


def _aggregate_h5ad(
    path: Path,
    *,
    study: str,
    context_values: np.ndarray,
    perturbations: np.ndarray,
    context_ids: dict[str, str],
    donor_id: str,
    min_cells: int,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    data = ad.read_h5ad(path, backed="r")
    try:
        if len(context_values) != data.n_obs or len(perturbations) != data.n_obs:
            raise ValueError(f"Metadata length mismatch for {path}")
        work = pd.DataFrame({"context": context_values, "perturbation": perturbations})
        grouped = work.groupby(["context", "perturbation"], sort=True).size().rename("cell_count")
        grouped = grouped.loc[grouped.ge(min_cells)].reset_index()
        controls = grouped["perturbation"].eq("control")
        eligible = set(grouped.loc[~controls, "perturbation"].astype(str))
        grouped = grouped.loc[controls | grouped["perturbation"].astype(str).isin(eligible)].reset_index(drop=True)
        lookup = {(str(r.context), str(r.perturbation)): i for i, r in enumerate(grouped.itertuples(index=False))}
        codes = np.fromiter(
            (lookup.get((str(c), str(p)), -1) for c, p in zip(context_values, perturbations, strict=True)),
            dtype=np.int64,
            count=data.n_obs,
        )
        valid = codes >= 0
        indicator = sparse.csr_matrix(
            (np.ones(valid.sum(), dtype=np.float32), (codes[valid], np.flatnonzero(valid))),
            shape=(len(grouped), data.n_obs),
        )
        sums = np.zeros((len(grouped), data.n_vars), dtype=np.float64)
        for start in range(0, data.n_vars, 512):
            stop = min(start + 512, data.n_vars)
            block = data.X[:, start:stop]
            product = indicator @ block
            sums[:, start:stop] = product.toarray() if sparse.issparse(product) else np.asarray(product)
        genes = data.var_names.astype(str).tolist()
    finally:
        data.file.close()
    libraries = sums.sum(axis=1)
    if np.any(libraries <= 0):
        raise ValueError(f"Zero aggregate library in {path}")
    normalized = np.log1p(sums / libraries[:, None] * 1_000_000)
    control_lookup = {
        str(row.context): normalized[index]
        for index, row in enumerate(grouped.itertuples(index=False))
        if str(row.perturbation) == "control"
    }
    keep = ~grouped["perturbation"].eq("control")
    perturbed = grouped.loc[keep].reset_index(drop=True)
    values = normalized[keep.to_numpy()]
    baselines = np.vstack([control_lookup[str(context)] for context in perturbed["context"]])
    metadata = pd.DataFrame(
        {
            "study": study,
            "context_id": perturbed["context"].astype(str).map(context_ids),
            "donor_id": donor_id,
            "perturbation_id": perturbed["perturbation"].astype(str),
            "cell_count": perturbed["cell_count"].astype(int),
            "response_source": "processed_UMI_pseudobulk_minus_NTC_log1p_CPM",
        }
    )
    return metadata, values - baselines, genes


def materialize_replogle(root: Path, output: Path, *, min_cells: int = 5) -> Path:
    destination = output / "replogle_deltas.parquet"
    if destination.exists():
        return destination
    specs = (
        ("K562", "ReplogleWeissman2022_K562_essential.h5ad", "replogle2022_essential::K562"),
        ("RPE1", "ReplogleWeissman2022_rpe1.h5ad", "replogle2022_essential::RPE1"),
    )
    pieces = []
    for line, name, context_id in specs:
        path = root / "data/raw/replogle_2022" / name
        data = ad.read_h5ad(path, backed="r")
        try:
            perturbation = data.obs["perturbation"].astype(str).to_numpy()
            contexts = np.full(data.n_obs, line, dtype=object)
        finally:
            data.file.close()
        pieces.append(
            _aggregate_h5ad(
                path,
                study="replogle",
                context_values=contexts,
                perturbations=perturbation,
                context_ids={line: context_id},
                donor_id=line,
                min_cells=min_cells,
            )
        )
    common = sorted(set(pieces[0][2]) & set(pieces[1][2]))
    frames = []
    for metadata, values, genes in pieces:
        indices = [genes.index(gene) for gene in common]
        frames.append(pd.concat([metadata, pd.DataFrame(values[:, indices], columns=common)], axis=1))
    combined = pd.concat(frames, ignore_index=True)
    shared = combined.groupby("perturbation_id")["context_id"].nunique()
    combined = combined.loc[combined["perturbation_id"].isin(shared[shared.ge(2)].index)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination, index=False, compression="zstd")
    return destination


def materialize_frangieh(root: Path, output: Path, *, min_cells: int = 5) -> Path:
    destination = output / "frangieh_deltas.parquet"
    if destination.exists():
        return destination
    path = root / "data/raw/frangieh_2021/FrangiehIzar2021_RNA.h5ad"
    data = ad.read_h5ad(path, backed="r")
    try:
        conditions = data.obs["perturbation_2"].astype(str).to_numpy()
        perturbations = data.obs["perturbation"].astype(str).to_numpy()
    finally:
        data.file.close()
    context_ids = {
        "Control": "frangieh2021::2686::Control",
        "IFNγ": "frangieh2021::2686::IFNgamma",
        "Co-culture": "frangieh2021::2686::Co-culture",
    }
    metadata, values, genes = _aggregate_h5ad(
        path,
        study="frangieh",
        context_values=conditions,
        perturbations=perturbations,
        context_ids=context_ids,
        donor_id="2686",
        min_cells=min_cells,
    )
    coverage = metadata.groupby("perturbation_id")["context_id"].nunique()
    keep = metadata["perturbation_id"].isin(coverage[coverage.ge(2)].index).to_numpy()
    return _write_delta(metadata.loc[keep].reset_index(drop=True), values[keep], genes, destination)


def run(root: Path, *, include_sensitivity: bool = False) -> list[Path]:
    root = root.resolve()
    output = root / "data/processed/phase1"
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        materialize_feng_genomewide(root, output),
        materialize_feng_targeted(root, output),
        materialize_replogle(root, output),
        materialize_frangieh(root, output),
    ]
    if include_sensitivity:
        paths.extend(
            materialize_feng_genomewide(root, output, min_cells=value)
            for value in (3, 10)
        )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--include-sensitivity", action="store_true")
    args = parser.parse_args()
    for path in run(args.project_root, include_sensitivity=args.include_sensitivity):
        print(path)


if __name__ == "__main__":
    main()
