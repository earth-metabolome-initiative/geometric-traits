#!/usr/bin/env python3
"""Measure RDKit RASCAL MCES wall time on the local ground-truth cases.

Reuses the exact TEST_CASES and option handling from
`generate_mces_ground_truth.py`, then records per-case wall time and a small
summary artifact for later comparison against the Rust implementation.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.11 --with rdkit --with "numpy<2" \
        python3 tests/fixtures/measure_mces_rdkit.py

    UV_CACHE_DIR=/tmp/uv-cache uv run --python 3.11 --with rdkit --with "numpy<2" \
        python3 tests/fixtures/measure_mces_rdkit.py --repeats 3 --output \
        tests/fixtures/mces_rdkit_timings.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import time
from pathlib import Path

from rdkit import rdBase

from generate_mces_ground_truth import TEST_CASES, compute_mces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/mces_rdkit_timings.json.gz"),
        help="Path to the gzipped JSON timing artifact.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of timed repetitions per case.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Restrict timing to one or more case names. Repeat the flag to keep multiple cases.",
    )
    parser.add_argument(
        "--sort",
        choices=("input", "slowest"),
        default="slowest",
        help="How to print the per-case summary to stdout.",
    )
    return parser.parse_args()


def time_case(smiles1: str, smiles2: str, opts: dict, repeats: int) -> dict:
    elapsed_seconds = []
    bond_matches = 0
    atom_matches = 0
    similarity = 0.0
    timed_out = False

    for _ in range(repeats):
        started = time.perf_counter()
        bond_matches, atom_matches, similarity, timed_out = compute_mces(smiles1, smiles2, **opts)
        elapsed_seconds.append(time.perf_counter() - started)

    return {
        "elapsed_seconds": elapsed_seconds,
        "min_elapsed_seconds": min(elapsed_seconds),
        "median_elapsed_seconds": statistics.median(elapsed_seconds),
        "max_elapsed_seconds": max(elapsed_seconds),
        "expected_bond_matches": bond_matches,
        "expected_atom_matches": atom_matches,
        "expected_similarity": similarity,
        "timed_out": timed_out,
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    started = time.perf_counter()
    cases = []
    selected_cases = set(args.case)
    for name, smiles1, smiles2, opts in TEST_CASES:
        if selected_cases and name not in selected_cases:
            continue
        case = {"name": name, "options": opts}
        case.update(time_case(smiles1, smiles2, opts, args.repeats))
        cases.append(case)

    total_elapsed = time.perf_counter() - started

    if args.sort == "slowest":
        printable_cases = sorted(cases, key=lambda case: case["median_elapsed_seconds"], reverse=True)
    else:
        printable_cases = cases

    for case in printable_cases:
        timeout_suffix = " [TIMEOUT]" if case["timed_out"] else ""
        print(
            f"{case['name']}: median={case['median_elapsed_seconds']:.6f}s "
            f"min={case['min_elapsed_seconds']:.6f}s max={case['max_elapsed_seconds']:.6f}s"
            f"{timeout_suffix}"
        )

    payload = {
        "version": 1,
        "rdkit_version": rdBase.rdkitVersion,
        "repeats": args.repeats,
        "total_elapsed_seconds": total_elapsed,
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print(f"\nWrote {len(cases)} timing rows to {args.output}")
    print(f"Total elapsed: {total_elapsed:.6f}s")


if __name__ == "__main__":
    main()
