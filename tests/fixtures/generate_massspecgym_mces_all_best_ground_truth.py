#!/usr/bin/env python3
"""Generate a RDKit allBestMCESs=true oracle for a MassSpecGym corpus.

Usage:
    uv run --with rdkit python3 tests/fixtures/generate_massspecgym_mces_all_best_ground_truth.py
"""

from __future__ import annotations

import gzip
import json
import argparse
from pathlib import Path
import statistics
import time

from rdkit import Chem
from rdkit.Chem.rdRascalMCES import FindMCES, RascalOptions


DEFAULT_INPUT_PATH = Path("tests/fixtures/massspecgym_mces_default_100.json.gz")
DEFAULT_OUTPUT_PATH = Path("tests/fixtures/massspecgym_mces_all_best_100.json.gz")
DEFAULT_PAIR_TIMEOUT_SECONDS = 1
PROGRESS_EVERY = 25
SIMILARITY_THRESHOLD = 0.0
MAX_BOND_MATCH_PAIRS = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_PAIR_TIMEOUT_SECONDS)
    return parser.parse_args()


def compute_all_best_top(
    smiles1: str,
    smiles2: str,
    timeout_seconds: int,
) -> tuple[int, int, float, bool, float]:
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    opts = RascalOptions()
    opts.similarityThreshold = SIMILARITY_THRESHOLD
    opts.returnEmptyMCES = True
    opts.timeout = timeout_seconds
    opts.maxBondMatchPairs = MAX_BOND_MATCH_PAIRS
    opts.allBestMCESs = True

    started = time.perf_counter()
    results = FindMCES(mol1, mol2, opts)
    elapsed = time.perf_counter() - started
    if not results:
        return 0, 0, 0.0, False, elapsed

    top = results[0]
    return (
        len(top.bondMatches()),
        len(top.atomMatches()),
        top.similarity,
        top.timedOut,
        elapsed,
    )


def main() -> None:
    args = parse_args()

    with gzip.open(args.input, "rt", encoding="utf-8") as fh:
        source = json.load(fh)

    output_cases = []
    changed_cases = 0
    timed_out_cases = 0

    for case in source["cases"]:
        updated = dict(case)
        bonds, atoms, similarity, timed_out, elapsed = compute_all_best_top(
            updated["smiles1"],
            updated["smiles2"],
            args.timeout_seconds,
        )
        if timed_out:
            timed_out_cases += 1
            continue

        if (
            bonds != updated["expected_bond_matches"]
            or atoms != updated["expected_atom_matches"]
            or round(similarity, 6) != updated["expected_similarity"]
        ):
            changed_cases += 1

        updated["expected_bond_matches"] = bonds
        updated["expected_atom_matches"] = atoms
        updated["expected_similarity"] = round(similarity, 6)
        updated["timed_out"] = False
        updated["rdkit_elapsed_seconds"] = round(elapsed, 6)
        output_cases.append(updated)

        if len(output_cases) % PROGRESS_EVERY == 0:
            print(
                f"processed {len(output_cases)}/{len(source['cases'])} allBest cases; "
                f"timed_out_so_far={timed_out_cases}",
                flush=True,
            )

    with gzip.open(args.output, "wt", encoding="utf-8") as fh:
        json.dump({"version": max(4, source["version"]), "cases": output_cases}, fh, indent=2)

    elapsed_seconds = [case["rdkit_elapsed_seconds"] for case in output_cases]
    print(
        f"Wrote {len(output_cases)} fast allBest cases to {args.output}; "
        f"{changed_cases} retained cases differ from the default-path fixture",
        flush=True,
    )
    print(f"RDKit timeout per pair: {args.timeout_seconds}s", flush=True)
    print(f"Skipped timed-out allBest cases: {timed_out_cases}", flush=True)
    print(
        "Elapsed summary (s): "
        f"min={min(elapsed_seconds):.6f} median={statistics.median(elapsed_seconds):.6f} "
        f"max={max(elapsed_seconds):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
