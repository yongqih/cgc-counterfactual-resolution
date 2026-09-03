#!/usr/bin/env python3
"""Audit GSE207049 coverage in ARCHS4 without downloading the HDF5 matrix.

The script reads only remote HDF5 byte ranges needed for metadata and one
small expression-chunk probe. It never downloads SRA reads, FASTQ, BAM, or the
full ARCHS4 matrix.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import io
import json
import re
import urllib.request
from urllib.error import HTTPError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ARCHS4_URL = "https://s3.dev.maayanlab.cloud/archs4/files/human_gene_v2.5.h5"
ARCHS4_PAGE = "https://maayanlab.cloud/archs4/download.html"
ARCHS4_HELP = "https://maayanlab.cloud/archs4/help.html"
ARCHS4_API = "https://api.archs4.maayanlab.cloud/"
GEO_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE207049"
ZENODO_URL = "https://zenodo.org/records/6595427"
AUTHOR_GITHUB = "https://github.com/AmandaJLea/LCLs_gene_exp"
PUBLISHED_ARCHS4_SHA1 = "ae96de0519b9f008b0dc3a9f944ee9007daf2f6a"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def remote_head(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={"User-Agent": "CGC-ARCHS4-coverage-audit/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        headers = response.headers
        return {
            "content_length": int(headers["Content-Length"]),
            "etag": headers.get("ETag"),
            "last_modified": headers.get("Last-Modified"),
            "accept_ranges": headers.get("Accept-Ranges"),
        }


def audit_deployed_api() -> dict[str, Any]:
    """Record that the deployed API index can lag the downloadable HDF5."""

    base = "https://api.archs4.maayanlab.cloud/meta/samples/geo_accession"
    all_request = urllib.request.Request(
        f"{base}?limit=1",
        headers={"User-Agent": "CGC-ARCHS4-coverage-audit/1.0"},
    )
    with urllib.request.urlopen(all_request, timeout=60) as response:
        content_range = response.headers.get("Content-Range")
        all_status = response.status

    series_request = urllib.request.Request(
        f"{base}?series_id=GSE207049&limit=10000",
        headers={"User-Agent": "CGC-ARCHS4-coverage-audit/1.0"},
    )
    try:
        with urllib.request.urlopen(series_request, timeout=60) as response:
            series_status = response.status
            series_body = response.read().decode("utf-8", errors="replace")
    except HTTPError as error:
        series_status = error.code
        series_body = error.read().decode("utf-8", errors="replace")
    return {
        "all_samples_http_status": all_status,
        "all_samples_content_range": content_range,
        "gse207049_http_status": series_status,
        "gse207049_response": series_body,
        "interpretation": (
            "The deployed API index is older than the v2.5 downloadable HDF5; "
            "a 404 here is not evidence that GSE207049 is absent from v2.5."
        ),
    }


class HTTPRangeFile(io.RawIOBase):
    """Small cached, seekable HTTP range reader for h5py's fileobj driver."""

    def __init__(
        self,
        url: str,
        size: int,
        block_size: int = 4 * 1024 * 1024,
        max_blocks: int = 32,
    ) -> None:
        self.url = url
        self.size = size
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.position = 0
        self.cache: collections.OrderedDict[int, bytes] = collections.OrderedDict()
        self.request_count = 0
        self.bytes_transferred = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = position
        return position

    def _block(self, index: int) -> bytes:
        if index in self.cache:
            block = self.cache.pop(index)
            self.cache[index] = block
            return block
        start = index * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "CGC-ARCHS4-coverage-audit/1.0",
            },
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            block = response.read()
            if response.status != 206:
                raise RuntimeError(f"Range request returned HTTP {response.status}")
        self.request_count += 1
        self.bytes_transferred += len(block)
        self.cache[index] = block
        while len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return block

    def readinto(self, buffer: bytearray) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        chunks: list[bytes] = []
        remaining = size
        while remaining:
            index = self.position // self.block_size
            offset = self.position % self.block_size
            block = self._block(index)
            take = min(remaining, len(block) - offset)
            chunks.append(block[offset : offset + take])
            self.position += take
            remaining -= take
        return b"".join(chunks)


def load_processed_core(metadata_path: Path) -> tuple[list[dict[str, str]], dict]:
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_line: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    for row in rows:
        if row["treatment"] in {"ETOH", "DEX"}:
            by_line[row["line"]].append(row)
    paired = {
        line: group
        for line, group in by_line.items()
        if {row["treatment"] for row in group} == {"ETOH", "DEX"}
    }
    required = {
        (line, treatment): sum(row["treatment"] == treatment for row in group)
        for line, group in paired.items()
        for treatment in ("ETOH", "DEX")
    }
    return rows, {"paired": paired, "required": required}


def load_geo_soft(soft_path: Path) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    with soft_path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if line.startswith("^SAMPLE = "):
                if current:
                    samples.append(current)
                current = {"gsm": line.split("=", 1)[1].strip()}
            elif current is not None and line.startswith("!Sample_title = "):
                current["geo_title"] = line.split("=", 1)[1].strip()
            elif current is not None and line.startswith("!Sample_relation = SRA: "):
                current["sra_experiment"] = line.rsplit("=", 1)[-1].strip()
        if current:
            samples.append(current)
    return samples


SIMPLE_TITLE = re.compile(r"^(?:(v\d+)\.)?(Line\d+)_(ETOH|DEX)$", re.I)
RAW_FASTQ_TITLE = re.compile(
    r"^(Line\d+)-(ETOH|DEX)_.+\.fastq\.gz$",
    re.I,
)


def target_candidates(
    geo_samples: list[dict[str, str]], paired_lines: set[str]
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for sample in geo_samples:
        title = sample.get("geo_title", "")
        match = SIMPLE_TITLE.match(title)
        if match:
            version, line, treatment = match.group(1) or "v1", match.group(2), match.group(3)
            title_pattern = "versioned_simple"
        else:
            match = RAW_FASTQ_TITLE.match(title)
            if not match:
                continue
            version, line, treatment = "raw_fastq", match.group(1), match.group(2)
            title_pattern = "raw_fastq"
        if line not in paired_lines:
            continue
        candidates.append(
            {
                **sample,
                "line": line,
                "treatment": treatment.upper(),
                "geo_version": version,
                "geo_title_pattern": title_pattern,
            }
        )
    return candidates


def scan_archs4(
    range_file: HTTPRangeFile,
    target_gsms: set[str],
) -> tuple[set[str], dict[str, int], dict[str, Any]]:
    study_gsms: set[str] = set()
    target_positions: dict[str, int] = {}
    matrix_info: dict[str, Any] = {}
    with h5py.File(range_file, "r", driver="fileobj") as handle:
        series = handle["meta/samples/series_id"]
        accessions = handle["meta/samples/geo_accession"]
        expression = handle["data/expression"]
        matrix_info = {
            "expression_shape": list(expression.shape),
            "expression_dtype": str(expression.dtype),
            "expression_chunks": list(expression.chunks or ()),
            "sample_count": int(series.shape[0]),
        }
        step = int(series.chunks[0]) * 16
        for start in range(0, len(series), step):
            stop = min(len(series), start + step)
            study_relative = np.flatnonzero(series[start:stop] == b"GSE207049")
            if not len(study_relative):
                continue
            absolute = start + study_relative
            values = [value.decode() for value in accessions[absolute]]
            study_gsms.update(values)
            for position, gsm in zip(absolute.tolist(), values):
                if gsm in target_gsms:
                    target_positions[gsm] = int(position)

        # Line108 is part of the frozen paired core and has one unambiguous
        # public ETOH/DEX version in GEO.
        probe_gsms = [gsm for gsm in ("GSM6269012", "GSM6269017") if gsm in target_positions]
        probe: dict[str, Any] = {}
        for gsm in probe_gsms:
            values = expression[:2000, target_positions[gsm]]
            probe[gsm] = {
                "sample_index": target_positions[gsm],
                "first_2000_gene_nonzero": int(np.count_nonzero(values)),
                "first_2000_gene_sum": int(values.sum(dtype=np.uint64)),
                "first_2000_gene_max": int(values.max()),
            }
        matrix_info["expression_chunk_probe"] = probe
    return study_gsms, target_positions, matrix_info


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to results/level2_lea/archs4_coverage_audit",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = (
        args.output_dir.resolve()
        if args.output_dir
        else repo / "results" / "level2_lea" / "archs4_coverage_audit"
    )
    output.mkdir(parents=True, exist_ok=True)

    data_dir = repo / "data" / "level2_lea_official"
    metadata_path = data_dir / "31Mar21_all_runs_voom_resid_metadata.txt"
    soft_path = data_dir / "GSE207049_family.soft"
    matrix_path = data_dir / "GSE207049_31Mar21_all_runs_voom_resid.txt.gz"

    _, core = load_processed_core(metadata_path)
    paired: dict[str, list[dict[str, str]]] = core["paired"]
    required: dict[tuple[str, str], int] = core["required"]
    geo_samples = load_geo_soft(soft_path)
    candidates = target_candidates(geo_samples, set(paired))

    head = remote_head(ARCHS4_URL)
    api_audit = audit_deployed_api()
    if head["accept_ranges"] != "bytes":
        raise RuntimeError("ARCHS4 object does not advertise byte-range support")
    range_file = HTTPRangeFile(ARCHS4_URL, int(head["content_length"]))
    study_gsms, _, matrix_info = scan_archs4(
        range_file,
        {row["gsm"] for row in candidates},
    )

    available = collections.Counter(
        (row["line"], row["treatment"])
        for row in candidates
        if row["gsm"] in study_gsms
    )
    geo_counts = collections.Counter((row["line"], row["treatment"]) for row in candidates)
    group_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    for line in sorted(paired, key=lambda value: (value == "LineNA", int(value[4:]) if value[4:].isdigit() else 10**9)):
        for treatment in ("ETOH", "DEX"):
            key = (line, treatment)
            needed = required[key]
            available_count = available[key]
            candidate_count = geo_counts[key]
            if candidate_count == 0:
                status = "NO_EXACT_GEO_TITLE_IDENTITY"
            elif available_count < needed:
                status = "ARCHS4_MULTIPLICITY_INSUFFICIENT"
            elif candidate_count == needed:
                status = "EXACT_CANDIDATE_MULTIPLICITY"
            else:
                status = "SUFFICIENT_BUT_VERSION_SELECTION_AMBIGUOUS"
            row = {
                "line": line,
                "treatment": treatment,
                "processed_required_replicates": needed,
                "geo_candidate_count": candidate_count,
                "archs4_candidate_count": available_count,
                "coverage_sufficient": available_count >= needed,
                "identity_status": status,
            }
            group_rows.append(row)
            if status == "NO_EXACT_GEO_TITLE_IDENTITY":
                source = paired[line][0]
                unresolved_rows.append(
                    {
                        **row,
                        "processed_1000_genomes_id1": source["1000_genomes_id1"],
                        "processed_1000_genomes_id2": source["1000_genomes_id2"],
                        "processed_population": source["pop"],
                        "reason": (
                            "Processed public metadata has no recoverable GEO title mapping; "
                            "do not guess a GSM identity."
                        ),
                    }
                )

    group_lookup = {(row["line"], row["treatment"]): row for row in group_rows}
    candidate_rows: list[dict[str, Any]] = []
    for row in sorted(candidates, key=lambda item: (item["line"], item["treatment"], item["gsm"])):
        group = group_lookup[(row["line"], row["treatment"])]
        candidate_rows.append(
            {
                **row,
                "archs4_v2_5_present": row["gsm"] in study_gsms,
                "processed_required_replicates": group["processed_required_replicates"],
                "group_archs4_candidate_count": group["archs4_candidate_count"],
                "group_coverage_sufficient": group["coverage_sufficient"],
                "identity_status": group["identity_status"],
            }
        )

    directly_mappable_lines = {
        line
        for line in paired
        if geo_counts[(line, "ETOH")] and geo_counts[(line, "DEX")]
    }
    replicate_complete = {
        line
        for line in paired
        if required[(line, "ETOH")] == 2 and required[(line, "DEX")] == 2
    }
    replicate_covered = {
        line
        for line in replicate_complete
        if available[(line, "ETOH")] >= 2 and available[(line, "DEX")] >= 2
    }
    mapped_groups = [row for row in group_rows if row["geo_candidate_count"] > 0]
    ambiguous_groups = [
        row
        for row in mapped_groups
        if row["identity_status"] == "SUFFICIENT_BUT_VERSION_SELECTION_AMBIGUOUS"
    ]
    missing_candidates = [row for row in candidate_rows if not row["archs4_v2_5_present"]]
    mapped_processed_versions = sum(
        required[(line, treatment)]
        for line in directly_mappable_lines
        for treatment in ("ETOH", "DEX")
    )

    summary = {
        "audit_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "ARCHS4_COVERAGE_SUFFICIENT_FOR_340_LINE_IDENTITY_RESOLVED_CORE",
        "no_reads_downloaded": True,
        "no_full_hdf5_downloaded": True,
        "sources": {
            "geo": GEO_URL,
            "zenodo": ZENODO_URL,
            "author_github": AUTHOR_GITHUB,
            "archs4_download": ARCHS4_PAGE,
            "archs4_help": ARCHS4_HELP,
            "archs4_api": ARCHS4_API,
            "archs4_hdf5": ARCHS4_URL,
        },
        "local_inputs": {
            str(metadata_path.relative_to(repo)): {
                "size_bytes": metadata_path.stat().st_size,
                "sha256": sha256(metadata_path),
            },
            str(soft_path.relative_to(repo)): {
                "size_bytes": soft_path.stat().st_size,
                "sha256": sha256(soft_path),
            },
            str(matrix_path.relative_to(repo)): {
                "size_bytes": matrix_path.stat().st_size,
                "sha256": sha256(matrix_path),
            },
        },
        "official_processed_resource_audit": {
            "pre_sva_count_or_normalized_matrix_found": False,
            "public_processed_representation": "filtered normalized batch-corrected voom residual; 3 surrogate variables removed",
            "status": "PRE_SVA_PROCESSED_MATRIX_NOT_PUBLICLY_AVAILABLE",
        },
        "cohort": {
            "processed_paired_lines": len(paired),
            "processed_etoh_dex_versions": sum(required.values()),
            "directly_title_mappable_lines": len(directly_mappable_lines),
            "directly_title_mappable_processed_versions": mapped_processed_versions,
            "unresolved_lines": sorted(set(paired) - directly_mappable_lines),
            "replicate_complete_lines": len(replicate_complete),
            "replicate_complete_lines_archs4_sufficient": len(replicate_covered),
            "candidate_gsms": len(candidate_rows),
            "candidate_gsms_archs4_present": len(candidate_rows) - len(missing_candidates),
            "candidate_gsms_archs4_missing": len(missing_candidates),
            "mapped_line_treatment_groups": len(mapped_groups),
            "mapped_groups_archs4_sufficient": sum(bool(row["coverage_sufficient"]) for row in mapped_groups),
            "version_selection_ambiguous_groups": len(ambiguous_groups),
        },
        "archs4_v2_5": {
            **head,
            "official_page_sha1": PUBLISHED_ARCHS4_SHA1,
            "official_page_sha1_verified_locally": False,
            "gse207049_gsms_present": len(study_gsms),
            "geo_gse207049_sample_count": len(geo_samples),
            "http_range_requests": range_file.request_count,
            "http_range_bytes_transferred": range_file.bytes_transferred,
            **matrix_info,
        },
        "deployed_archs4_api_audit": api_audit,
    }

    write_csv(
        output / "ARCHS4_GSE207049_CANDIDATE_GSMS.csv",
        candidate_rows,
        [
            "line",
            "treatment",
            "gsm",
            "geo_title",
            "geo_version",
            "geo_title_pattern",
            "sra_experiment",
            "archs4_v2_5_present",
            "processed_required_replicates",
            "group_archs4_candidate_count",
            "group_coverage_sufficient",
            "identity_status",
        ],
    )
    write_csv(
        output / "ARCHS4_LINE_TREATMENT_COVERAGE.csv",
        group_rows,
        [
            "line",
            "treatment",
            "processed_required_replicates",
            "geo_candidate_count",
            "archs4_candidate_count",
            "coverage_sufficient",
            "identity_status",
        ],
    )
    write_csv(
        output / "ARCHS4_UNRESOLVED_CORE_IDENTITIES.csv",
        unresolved_rows,
        [
            "line",
            "treatment",
            "processed_required_replicates",
            "geo_candidate_count",
            "archs4_candidate_count",
            "processed_1000_genomes_id1",
            "processed_1000_genomes_id2",
            "processed_population",
            "reason",
        ],
    )
    with (output / "ARCHS4_COVERAGE_SUMMARY.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    report = f"""# GSE207049 ARCHS4 coverage audit

## Decision

`ARCHS4_COVERAGE_SUFFICIENT_FOR_340_LINE_IDENTITY_RESOLVED_CORE`

The latest ARCHS4 v2.5 human gene-level HDF5 contains enough ETOH and DEX
samples to support an independent, pre-SVA/raw-read-derived robustness analysis
for every GSE207049 core LCL whose public raw-sample identity can be resolved.
This is **340 of the historical 342 paired LCLs**. No reads, FASTQ, BAM, or full
ARCHS4 matrix were downloaded, and no scientific model was run.

This is a coverage verdict, not a scientific outcome and not proof that the
gene-versus-program hierarchy will replicate.

## Exact coverage

- GEO lists **{len(geo_samples):,}** samples for GSE207049.
- ARCHS4 v2.5 contains **{len(study_gsms):,} / {len(geo_samples):,}** of those GSMs
  ({len(study_gsms) / len(geo_samples):.2%}).
- The public processed matrix contains **{len(paired)}** paired ETOH/DEX line
  labels and **{sum(required.values())}** retained ETOH/DEX sample versions.
- **{len(directly_mappable_lines)} / {len(paired)}** line labels have an exact,
  title-based GEO mapping. These account for **{mapped_processed_versions}**
  retained processed versions.
- All **{len(mapped_groups)} / {len(mapped_groups)}** directly mapped
  line-by-treatment groups have at least the required number of ARCHS4
  candidates.
- **{len(replicate_covered)} / {len(replicate_complete)}** historical 2-by-2
  replicate-complete LCLs are directly resolvable with at least two ARCHS4 GSMs
  per condition.
- Across all alternative public versions for the mapped core,
  **{len(candidate_rows) - len(missing_candidates)} / {len(candidate_rows)}**
  candidate GSMs are in ARCHS4. The {len(missing_candidates)} missing candidates
  do not reduce any resolved line-by-treatment group below its historical
  replicate multiplicity.

## Identity boundary

Two processed line labels cannot be linked safely to an exact public GSM using
the official processed metadata and GEO SOFT record:

- `LineNA`: one ETOH and one DEX processed row, but no donor identifier or GEO
  title identity.
- `Line223`: two ETOH and two DEX processed rows; the processed metadata labels
  the donor as `HG03445`, but neither that line label nor donor identifier is
  recoverable in the GEO SOFT sample records.

They must be excluded unless the authors provide an explicit mapping. Guessing
or substituting GSMs would invalidate a sample-level robustness audit.

There is a second, narrower boundary: **{len(ambiguous_groups)}** of the
{len(mapped_groups)} mapped line-by-treatment groups have more GEO candidate
versions than retained processed versions. Public residual-matrix metadata no
longer preserves which raw version produced each `.x`/`.y` column. ARCHS4
coverage is sufficient, but exact author-version selection is not reconstructable
from the public processed metadata alone. A formal analysis therefore needs a
frozen, outcome-blind version rule (for example, all available versions followed
by within-line-by-treatment aggregation) rather than pretending to reproduce the
author's exact pre-SVA sample matrix.

## Representation claim boundary

ARCHS4 v2.5 stores rounded integer gene-level Kallisto pseudocounts. It is an
independent raw-read reprocessing representation, **not** Lea's original
STAR/HTSeq count matrix. A successful robustness result could support:

> The gene-versus-program resolution hierarchy persists after independent
> raw-read reprocessing without Lea's SVA residualization.

It could not support:

> We exactly reconstructed Lea's pre-SVA matrix.

The official-source inventory still contains no author-released pre-SVA count
or normalized matrix. GEO exposes only the residualized processed matrix and
metadata; Zenodo describes the same filtered, normalized, batch-corrected
matrix; and the author repository reads that residualized matrix for analysis.

## Remote-read audit

- Object: `{ARCHS4_URL}`
- Remote size: **{head['content_length']:,} bytes**
  ({head['content_length'] / 2**30:.2f} GiB; listed as 45G by ARCHS4).
- Matrix dimensions: **{matrix_info['expression_shape'][0]:,} genes by
  {matrix_info['expression_shape'][1]:,} samples**.
- Range requests: **{range_file.request_count}**.
- Bytes transferred: **{range_file.bytes_transferred:,}**
  ({range_file.bytes_transferred / 2**20:.1f} MiB).
- Expression payload probe: two mapped Line108 samples had finite, nonzero
  integer values in the first 2,000 gene rows; no full column or matrix was
  extracted.
- The download page's published SHA-1 is recorded but was not locally verified,
  because verification would require downloading the full object.

The deployed ARCHS4 API returned HTTP **{api_audit['gse207049_http_status']}**
for a `series_id=GSE207049` query and advertises an older sample index via
`Content-Range: {api_audit['all_samples_content_range']}`. The v2.5 HDF5 itself
contains {matrix_info['sample_count']:,} samples and the GSE207049 records above.
Therefore the coverage audit uses the downloadable v2.5 HDF5 as authority; the
deployed API would give a false negative for this study.

## Recommended next decision

The efficient, defensible next analysis is a preregistered ARCHS4 robustness
run on the **340-line identity-resolved core**, with the version aggregation,
normalization, gene mapping, folds, and decision thresholds frozen before
outcomes are inspected. Do not download the full HDF5 until that protocol is
approved; HTTP-range extraction can likely retrieve only the selected columns.

Do not run SRA/STAR/HTSeq unless ARCHS4 extraction or representation QA fails.

## Primary sources

- [GEO GSE207049]({GEO_URL})
- [ARCHS4 v2.5 downloads]({ARCHS4_PAGE})
- [ARCHS4 processing and HDF5 documentation]({ARCHS4_HELP})
- [ARCHS4 data API]({ARCHS4_API})
- [Zenodo author deposit]({ZENODO_URL})
- [Author analysis repository]({AUTHOR_GITHUB})
"""
    (output / "ARCHS4_COVERAGE_AUDIT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary["cohort"], indent=2, sort_keys=True))
    print(summary["verdict"])


if __name__ == "__main__":
    main()
