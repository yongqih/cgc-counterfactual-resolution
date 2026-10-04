"""Explicit, path-only launcher for exact frozen scientific implementations.

No scientific action runs without --execute. --inspect-only is the default.
--cli-help runs only an audited help-capable parser, never an analysis stage.
The launcher does not change equations, labels, models, splits or parameters.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def verify_import_sources(root: Path, manifest: dict) -> None:
    """Verify executed/imported sources and configurations, never waive a hash."""
    for row in manifest["files"]:
        if row.get("role") == "scientific_source" or row["path"].startswith("configs/"):
            target = root / row["path"]
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Frozen imported source/config hash mismatch: {row['path']}")


def load_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_frozen_source(root: Path, manifest: dict, commit: str, path: str) -> str:
    rows = manifest["runtime_git_authority"]["verified_sources"]
    lookup = {(row["commit"], row["path"]): row for row in rows}
    key = (commit, path)
    if key not in lookup:
        raise RuntimeError(f"Unlisted frozen source lookup: {key}")
    row = lookup[key]
    data = (root / row["exported_path"]).read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise RuntimeError(f"Frozen source hash failed: {key}")
    # Matches subprocess text=True universal-newline decoding of git show.
    return data.decode("utf-8").replace("\r\n", "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--entrypoint")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--cli-help", action="store_true")
    parser.add_argument("--execute", action="store_true", help="explicit authorization to run the selected frozen analysis or figure/table stage")
    parser.add_argument("--work-root", type=Path, help="path-adapter output/prepared-metadata workspace; does not alter original CLI root arguments")
    parser.add_argument("--input-root", type=Path, help="BIO2 adapter source root only; other original CLIs receive roots after --")
    parser.add_argument("--gene-workspace", type=Path, help="ED7 prepared held-context workspace (contains frozen src/results/data)")
    parser.add_argument("--pathway-workspace", type=Path, help="ED7 prepared pathway workspace (contains frozen src/results/data)")
    args, remaining = parser.parse_known_args()
    if remaining[:1] == ["--"]:
        remaining = remaining[1:]
    root = HERE / "bundles" / args.bundle
    if root.resolve().parent != (HERE / "bundles").resolve():
        parser.error("Invalid bundle id")
    manifest = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    if args.execute or args.cli_help:
        verify_import_sources(root, manifest)
    if manifest.get("release_entrypoint"):
        current = manifest["release_entrypoint"]
        if args.entrypoint and args.entrypoint != current:
            parser.error("Historical all-stages entrypoint is archival-only; this bundle permits only its current bounded entrypoint")
        if not args.execute and not args.cli_help or args.inspect_only:
            print(json.dumps({"id": manifest["id"], "class": manifest["class"], "current_entrypoint": current, "claim_boundary": manifest["claim_boundary"], "required_inputs": manifest["required_inputs"], "active_execution_guard": manifest["active_execution_guard"]}, indent=2))
            return 0
        if args.execute and args.cli_help:
            parser.error("Choose --execute or --cli-help, not both")
        source = HERE / current
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["release_entrypoint_provenance"]["sha256"]:
            parser.error("Release adapter hash mismatch")
        command = [sys.executable, str(source), *( ["--help"] if args.cli_help else remaining)]
        return subprocess.call(command, cwd=root)
    entries = manifest["entrypoints"]
    selected = args.entrypoint or (entries[0]["path"] if entries else None)
    if args.bundle == "ed7_hierarchical_repair" and selected is None:
        selected = "audit/release_repair/ed7/repair_ed7_hierarchical_bootstrap.py"
    entry = next((item for item in entries if item["path"] == selected), None)
    if selected is not None and selected not in {row["path"] for row in manifest["files"]}:
        parser.error("Entrypoint is not an exported frozen source")
    if selected is not None and selected not in {item["path"] for item in entries} and args.bundle != "ed7_hierarchical_repair":
        parser.error("Source is a dependency, not an allowlisted executable entrypoint")
    if not args.execute and not args.cli_help or args.inspect_only:
        print(json.dumps({key: manifest.get(key) for key in ("id", "class", "commit", "entrypoints", "required_inputs", "runtime_readiness", "known_portability_constraints", "claim_boundary", "path_adapter")}, indent=2))
        return 0
    if args.execute and args.cli_help:
        parser.error("Choose --execute or --cli-help, not both")
    if args.execute and manifest.get("allowed_stage"):
        stage = manifest["allowed_stage"]
        if stage not in remaining or "finalize" in remaining or "all" in remaining:
            parser.error(f"Only stage {stage!r} is current; legacy uncertainty finalizers are not allowed. Use ed7_hierarchical_repair.")
    if args.cli_help:
        if entry is None or not entry.get("help_args"):
            parser.error("This exact frozen entrypoint has no safe argument-parser help. Use --inspect-only.")
        remaining = entry["help_args"]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "scripts"), str(root)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLBACKEND"] = "Agg"
    if "GIT_DIR" in env:
        parser.error("Unexpected inherited GIT_DIR; unset it for an ordinary bundle run")
    if args.execute and args.bundle == "lea_archs4":
        # The only adaptation is immutable artifact lookup. No fitting,
        # preprocessing, model or statistical function is replaced.
        sys.path[:0] = [str(root / "src"), str(root / "scripts"), str(root)]
        historical = HERE / "bundles/lea_historical_analysis/src"
        sys.path.append(str(historical))
        module = load_file(root / selected, "frozen_lea_archs4_entrypoint")
        def frozen_git_source(path: str) -> str:
            return read_frozen_source(root, manifest, module.HISTORICAL, path)

        module.git_source = frozen_git_source
        sys.argv = [str(root / selected), *remaining]
        module.main()
        return 0
    if args.execute and args.bundle == "ed7_hierarchical_repair":
        if remaining or args.gene_workspace is None or args.pathway_workspace is None or args.work_root is None:
            parser.error("ED7 path adapter requires --work-root --gene-workspace --pathway-workspace; no statistical overrides")
        module = load_file(root / selected, "frozen_ed7_hierarchical_repair")
        module.REPO = args.work_root.resolve()
        module.OUT = module.REPO / "audit/release_repair/ed7"
        module.GENE_WT = args.gene_workspace.resolve()
        module.PATH_WT = args.pathway_workspace.resolve()
        module.GENE_OUT = module.GENE_WT / "results/tahoe_held_context_scaling"
        module.PATH_OUT = module.PATH_WT / "results/tahoe_five_pathway_scaling"
        module.main()
        return 0
    if selected is None:
        parser.error("Select an exported entrypoint")
    if entry and entry.get("module"):
        command = [sys.executable, "-m", entry["module"], *remaining]
    else:
        command = [sys.executable, str(root / selected), *remaining]
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
