"""Run the frozen Tahoe held-context scaling experiment. SPDX-License-Identifier: MIT"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from igc_virtual_cell.tahoe_held_context_scaling import (
    M_GRID,
    PILOT_M_GRID,
    build_response_gram_cache,
    load_frozen_inputs,
    load_response_grams,
    prepare_design,
    run_episodes,
    sanity_check_audited_estimator,
    summarize_pilot,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "cache", "pilot", "full"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--force-cache", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    frozen = load_frozen_inputs(root, args.source_root)
    targets, ladders = prepare_design(root, frozen)
    if args.command == "prepare":
        return
    started = time.perf_counter()
    expected_cache = root / "results/tahoe_held_context_scaling/_cache/TAHOE_PRIMARY_RESPONSE_GRAMS.npz"
    previous_size = expected_cache.stat().st_size if expected_cache.exists() else 0
    cache_path = build_response_gram_cache(root, frozen, force=args.force_cache)
    if args.command == "cache":
        print(json.dumps({"cache": str(cache_path)}, indent=2))
        return
    sanity_check_audited_estimator(root)
    grams = load_response_grams(cache_path)
    if args.command == "full":
        run_episodes(
            root,
            frozen,
            grams,
            ladders,
            targets=range(50),
            m_grid=M_GRID,
            models=("linear_ridge", "rbf_ridge", "bilinear_reduced_rank"),
            pilot=False,
            workers=6,
        )
        return
    frame, _ = run_episodes(
        root,
        frozen,
        grams,
        ladders,
        targets=targets.loc[targets["pilot_target"], "target_index"].astype(int),
        m_grid=PILOT_M_GRID,
        models=("linear_ridge", "rbf_ridge"),
        pilot=True,
    )
    summarize_pilot(root, frame, cache_path, previous_size, time.perf_counter() - started)


if __name__ == "__main__":
    main()
