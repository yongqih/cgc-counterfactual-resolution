from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run(spec_path: Path, source_root: Path | None = None, output_path: Path | None = None) -> dict:
    release_root = spec_path.resolve().parents[2]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if source_root is None:
        source_root = release_root / spec.get("source_root", "source_data")
    results = []
    for filename, expected_rows in spec.get("row_counts", {}).items():
        actual = len(_rows(source_root / filename))
        if actual != expected_rows:
            raise AssertionError(f"{filename}: expected {expected_rows} rows, found {actual}")
        results.append({"file": filename, "row_count": actual})
    cache = {}
    for check in spec.get("checks", []):
        filename = check["file"]
        cache.setdefault(filename, _rows(source_root / filename))
        matches = [row for row in cache[filename] if all(row.get(key, "") == str(value) for key, value in check.get("where", {}).items())]
        if len(matches) != 1:
            raise AssertionError(f"{filename}: expected one row for {check.get('where')}, found {len(matches)}")
        actual = float(matches[0][check["column"]])
        expected = float(check["expected"])
        tolerance = float(check.get("tolerance", 0.0))
        if abs(actual - expected) > tolerance:
            raise AssertionError(f"{filename}: {actual} != {expected} within {tolerance}")
        results.append({"file": filename, "where": check.get("where", {}), "column": check["column"], "value": actual})
    classification = spec.get(
        "entrypoint_classification", "NON_REPRODUCTION_INTEGRITY_CHECK"
    )
    if classification != "NON_REPRODUCTION_INTEGRITY_CHECK":
        raise AssertionError(
            "This shared helper only implements released-table integrity checks; "
            f"it cannot claim {classification!r}."
        )
    payload = {
        "analysis": spec["analysis_name"],
        "status": "integrity_check_passed",
        "entrypoint_classification": classification,
        "scientific_reproduction": False,
        "results": results,
    }
    if output_path is None:
        slug = spec_path.parent.name
        output_path = release_root / "reproduced" / f"{slug}_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main(spec_path: Path) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(spec_path, args.source_root, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
