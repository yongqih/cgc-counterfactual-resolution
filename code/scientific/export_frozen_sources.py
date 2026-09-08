"""Export exact, dependency-closed scientific sources from immutable Git objects.

This is packaging, not analysis. It never imports scientific code or reads arrays.
The registry separates executable scientific capability from input availability.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
CLASSES = {"FULL_ANALYSIS", "FIGURE_FROM_FROZEN_PREDICTIONS", "OFFICIAL_ARTIFACT_SUMMARY", "TABLE_GENERATOR"}


def git(repo: Path, *args: str) -> bytes:
    env = os.environ.copy()
    env.pop("GIT_DIR", None)
    env.pop("GIT_WORK_TREE", None)
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(value: str) -> Path:
    result = Path(value)
    if result.is_absolute() or ".." in result.parts or ":" in value:
        raise ValueError(f"Unsafe relative path: {value}")
    return result


class Exporter:
    def __init__(self, repo: Path, destination: Path):
        self.repo = repo.resolve()
        self.destination = destination.resolve()
        self.trees: dict[str, list[str]] = {}
        self.blobs: dict[tuple[str, str], bytes] = {}

    def paths(self, commit: str) -> list[str]:
        if commit not in self.trees:
            exact = git(self.repo, "rev-parse", f"{commit}^{{commit}}").decode().strip()
            if exact != commit or len(commit) != 40:
                raise ValueError(f"Registry must use full immutable commit: {commit}")
            self.trees[commit] = git(self.repo, "ls-tree", "-r", "--name-only", commit).decode().splitlines()
        return self.trees[commit]

    def blob(self, commit: str, path: str) -> bytes:
        key = (commit, path)
        if key not in self.blobs:
            self.blobs[key] = git(self.repo, "show", f"{commit}:{path}")
        return self.blobs[key]

    def expand(self, commit: str, values: list[str]) -> set[str]:
        paths = self.paths(commit)
        found: set[str] = set()
        for value in values:
            matches = {p for p in paths if p == value or p.startswith(value.rstrip("/") + "/") or fnmatch.fnmatchcase(p, value)}
            if not matches:
                raise ValueError(f"Unresolved frozen source: {commit}:{value}")
            found.update(matches)
        return found

    def closure(self, commit: str, roots: set[str]) -> tuple[set[str], list[str]]:
        available = set(self.paths(commit))
        modules: dict[str, str] = {}
        for path in available:
            if path.startswith("src/") and path.endswith(".py"):
                name = path[4:-3].replace("/", ".")
                if name.endswith(".__init__"):
                    name = name[:-9]
                modules[name] = path
        pending = list(roots)
        selected: set[str] = set()
        external: set[str] = set()
        while pending:
            path = pending.pop()
            if path in selected:
                continue
            selected.add(path)
            if not path.endswith(".py"):
                continue
            tree = ast.parse(self.blob(commit, path).decode("utf-8-sig"), filename=path)
            parts = path.split("/")[:-1]
            while parts:
                parent = "/".join(parts) + "/__init__.py"
                if parent in available and parent not in selected:
                    pending.append(parent)
                parts.pop()
            package = path[4:-3].replace("/", ".").split(".") if path.startswith("src/") else []
            if package and package[-1] != "__init__":
                package.pop()
            elif package:
                package.pop()
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [x.name for x in node.names]
                elif isinstance(node, ast.ImportFrom):
                    if node.level:
                        prefix = package[: len(package) - node.level + 1]
                        base = ".".join(prefix + ([node.module] if node.module else []))
                    else:
                        base = node.module or ""
                    names = [base] + [base + "." + x.name for x in node.names if x.name != "*"]
                for name in names:
                    if not name:
                        continue
                    if name in modules:
                        pending.append(modules[name])
                    else:
                        sibling = str(Path(path).parent / (name.split(".")[0] + ".py")).replace("\\", "/")
                        if sibling in available:
                            pending.append(sibling)
                        elif name.startswith("igc_virtual_cell"):
                            # Imported attributes are not necessarily submodules.
                            if not any(name.startswith(module + ".") for module in modules):
                                raise ValueError(f"Unresolved local module {path}: {name}")
                        else:
                            external.add(name.split(".")[0])
        return selected, sorted(external - set(sys.stdlib_module_names))

    def runtime_authority(self, out: Path, primary: str, sources: list[dict]) -> dict | None:
        if not sources:
            return None
        verified: list[dict] = []
        for row in sources:
            commit, path = row["commit"], row["path"]
            self.paths(commit)
            blob = git(self.repo, "rev-parse", f"{commit}:{path}").decode().strip()
            data = self.blob(commit, path)
            relative = Path("runtime_sources") / commit / safe_path(path)
            target = out / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            verified.append({**row, "exported_path": relative.as_posix(), "sha256": digest(data), "size_bytes": len(data), "git_blob": blob})
        return {"kind": "exact frozen source-lookup map, not a Git repository", "verified_sources": verified}

    def export(self, spec: dict) -> dict:
        if spec.get("post_freeze_correction"):
            raise RuntimeError(f"Bundle {spec['id']} has an audited post-freeze correction. A base-commit export would discard it; use the corrected source package and manifest.")
        if spec["class"] not in CLASSES:
            raise ValueError(f"Invalid scientific capability class: {spec['class']}")
        bundle_id, commit = spec["id"], spec["commit"]
        out = self.destination / safe_path(bundle_id)
        if out.parent != self.destination:
            raise ValueError("Bundle id must be a simple directory name")
        existing = out / "SOURCE_MANIFEST.json"
        if existing.exists() and json.loads(existing.read_text(encoding="utf-8")).get("post_freeze_correction"):
            raise RuntimeError(f"Refusing to overwrite audited post-freeze correction: {bundle_id}. Use its recorded correction archive/manifest; a base Git export is historical only.")
        roots = self.expand(commit, spec.get("source_roots", []) + spec.get("extra_sources", []))
        sources, external = self.closure(commit, roots)
        metadata = self.expand(commit, spec.get("configs", []) + spec.get("split_metadata", []) + spec.get("environment_metadata", []))
        rows: list[dict] = []
        for path in sorted(sources | metadata):
            data = self.blob(commit, path)
            if len(data) > spec.get("max_single_source_bytes", 8_000_000):
                raise ValueError(f"Refuse large source export: {path} ({len(data)})")
            if Path(path).suffix.lower() in {".h5ad", ".h5", ".hdf5", ".npy", ".npz", ".pkl", ".parquet"}:
                raise ValueError(f"Array/data export is out of scope: {path}")
            target = out / safe_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            rows.append({"path": path, "source_file": path, "source_commit": commit, "git_blob": git(self.repo, "rev-parse", f"{commit}:{path}").decode().strip(), "sha256": digest(data), "size_bytes": len(data), "role": "scientific_source" if path in sources else "frozen_configuration_or_metadata"})
        authority = self.runtime_authority(out, commit, spec.get("runtime_git_sources", []))
        manifest = {**spec, "exported_file_count": len(rows), "exported_source_bytes": sum(r["size_bytes"] for r in rows), "external_import_roots": external, "runtime_git_authority": authority, "files": rows, "source_modification": "NONE; byte-exact Git blobs", "analysis_execution": "NOT_RUN"}
        if spec.get("release_entrypoint"):
            adapter = self.destination.parent / safe_path(spec["release_entrypoint"])
            manifest["release_entrypoint_provenance"] = {"path": spec["release_entrypoint"], "sha256": digest(adapter.read_bytes()), "origin": "release-only bounded reporting adapter; not a historical source blob", "correction_authority": spec["correction_authority"]}
        (out / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest


def verify(destination: Path) -> dict:
    checked, files, errors = 0, 0, []
    for path in sorted(destination.glob("*/SOURCE_MANIFEST.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        checked += 1
        for row in data["files"]:
            file = path.parent / safe_path(row["path"])
            files += 1
            if not file.is_file() or digest(file.read_bytes()) != row["sha256"]:
                errors.append(str(file))
        authority = data.get("runtime_git_authority")
        if authority:
            for row in authority["verified_sources"]:
                result = (path.parent / safe_path(row["exported_path"])).read_bytes()
                if digest(result) != row["sha256"]:
                    errors.append(f"runtime_source:{row}")
        if data.get("release_entrypoint_provenance"):
            row = data["release_entrypoint_provenance"]
            if digest((destination.parent / row["path"]).read_bytes()) != row["sha256"]:
                errors.append(f"release_adapter:{row['path']}")
    return {"bundles_verified": checked, "files_verified": files, "errors": errors, "status": "PASS" if checked and not errors else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=HERE.parents[2])
    parser.add_argument("--registry", type=Path, default=HERE / "SCIENTIFIC_BUNDLES.json")
    parser.add_argument("--destination", type=Path, default=HERE / "bundles")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        summary = verify(args.destination)
    else:
        registry = json.loads(args.registry.read_text(encoding="utf-8-sig"))
        exporter = Exporter(args.repo, args.destination)
        results = [exporter.export(spec) for spec in registry["bundles"]]
        summary = {**verify(args.destination), "total_source_bytes": sum(x["exported_source_bytes"] for x in results)}
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
