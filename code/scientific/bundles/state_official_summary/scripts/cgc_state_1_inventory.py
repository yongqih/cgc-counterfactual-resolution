from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/state_official_predictions"
OUT = ROOT / "results/cgc_state_1"
REVISIONS = {
    "state_github": "9bbfe78a434a55205e4de834e1ea99f85f7a3add",
    "ST_SE_Parse": "fa458ffda8690c5c5bdff080dfdc46b46713043b",
    "ST_HVG_Parse": "827df657cd87063026d2abefd73e99036755ac1b",
}
URLS = {
    "state_github": "https://github.com/ArcInstitute/state",
    "ST_SE": "https://huggingface.co/arcinstitute/ST-SE-Parse",
    "ST_HVG": "https://huggingface.co/arcinstitute/ST-HVG-Parse",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def first_string(value: object, key: str) -> str | None:
    if isinstance(value, dict):
        if key in value and isinstance(value[key], str):
            return value[key]
        for child in value.values():
            found = first_string(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = first_string(child, key)
            if found is not None:
                return found
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = OUT / "split_tomls"
    archive.mkdir(exist_ok=True)
    artifact_rows: list[dict] = []
    for path in sorted(DATA.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if "ST_SE" in path.parts:
            source = "ST_SE"
        elif "ST_HVG" in path.parts:
            source = "ST_HVG"
        else:
            source = "state_github"
        name = path.name
        artifact_rows.append({
            "relative_path": rel,
            "source_repository": source,
            "source_url": URLS[source],
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "artifact_type": (
                "split_toml" if name.endswith(".toml") and "split_" in name
                else "official_pred_de" if name.endswith("_pred_de.csv")
                else "official_real_de" if name.endswith("_real_de.csv")
                else "official_metrics" if name.endswith("results.csv")
                else "license_or_policy" if "LICENSE" in name or "POLICY" in name
                else "repository_metadata"
            ),
            "checkpoint_label": "eval_best" if name.endswith(".csv") else "not_applicable",
            "used_primary": name.endswith(("_pred_de.csv", "_real_de.csv", "_results.csv", "_agg_results.csv", ".toml")),
        })
    pd.DataFrame(artifact_rows).to_csv(OUT / "artifact_inventory.csv", index=False)

    split_rows = []
    for variant in ("ST_SE", "ST_HVG"):
        for split in range(5):
            split_dir = DATA / variant / f"split_{split}"
            toml_path = split_dir / f"split_{split}.toml"
            spec = tomllib.loads(toml_path.read_text(encoding="utf-8"))
            zeroshot = spec.get("zeroshot", {})
            zeroshot_keys = list(zeroshot)
            heldout = zeroshot_keys[0].split(".", 1)[1] if len(zeroshot_keys) == 1 and "." in zeroshot_keys[0] else None
            pred_path = next(split_dir.glob("*_pred_de.csv"))
            real_path = next(split_dir.glob("*_real_de.csv"))
            pred = pd.read_csv(pred_path, usecols=["target", "reference", "feature"])
            real = pd.read_csv(real_path, usecols=["target", "reference", "feature"])
            axes_equal = pred[["target", "feature"]].equals(real[["target", "feature"]])
            is_zero_shot = len(zeroshot) == 1 and not spec.get("fewshot", {})
            split_rows.append({
                "variant": variant,
                "split": split,
                "heldout_context": heldout,
                "zero_shot": is_zero_shot,
                "few_shot": not is_zero_shot,
                "checkpoint": "eval_best",
                "n_interventions": int(pred["target"].nunique()),
                "n_features": int(pred["feature"].nunique()),
                "reference_labels": "|".join(sorted(pred["reference"].astype(str).unique())),
                "pred_real_axes_equal": axes_equal,
                "donor_metadata_available": False,
                "cell_counts_available": False,
                "target_context_labels_in_training": False,
                "toml_sha256": sha256(toml_path),
                "pred_sha256": sha256(pred_path),
                "real_sha256": sha256(real_path),
            })
            shutil.copyfile(toml_path, archive / f"{variant}_split_{split}.toml")
    splits = pd.DataFrame(split_rows)
    splits.to_csv(OUT / "zero_shot_split_manifest.csv", index=False)

    write_json(OUT / "official_source_manifest.json", {
        "phase": "CGC-STATE-1",
        "official_only": True,
        "sources": URLS,
        "files": len(artifact_rows),
        "downloaded_bytes": sum(row["size_bytes"] for row in artifact_rows),
        "downloaded_gb_decimal": sum(row["size_bytes"] for row in artifact_rows) / 1e9,
        "maximum_transfer_gb": 80,
        "raw_or_cell_level_h5ad_downloaded": False,
        "checkpoint_history_downloaded": False,
        "eval_last_downloaded": False,
        "official_eval_best_complete_de_tables_downloaded": True,
        "minimum_free_disk_after_download_gb": 700,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "accessed_at": datetime.now(timezone.utc).astimezone().isoformat(),
    })
    write_json(OUT / "repository_revisions.json", {
        "revisions": REVISIONS,
        "state_package_version": "0.11.3",
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "git_parent": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    })
    (OUT / "license_acknowledgement.md").write_text(
        "# State license acknowledgement\n\n"
        "The audit uses only official Arc Institute repository and Hugging Face artifacts. "
        "The State code license and the ST-SE/ST-HVG model licenses and acceptable-use policies "
        "were downloaded, hashed, inventoried, and reviewed. No third-party checkpoint or prediction is used.\n",
        encoding="utf-8",
    )
    if len(splits) != 10 or not splits["zero_shot"].all() or not splits["pred_real_axes_equal"].all():
        raise RuntimeError("Official zero-shot split audit failed")


if __name__ == "__main__":
    main()
