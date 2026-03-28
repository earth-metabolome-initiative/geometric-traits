#!/usr/bin/env python3
"""Generate a focused RDKit allBestMCESs oracle for MassSpecGym holdout cases.

Usage:
    uv run --with rdkit python3 tests/fixtures/generate_massspecgym_mces_all_best_holdouts.py
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.rdRascalMCES import FindMCES, RascalOptions


INPUT_PATH = Path("tests/fixtures/massspecgym_mces_default_100.json.gz")
OUTPUT_PATH = Path("tests/fixtures/massspecgym_mces_all_best_holdouts.json.gz")
CASE_NAMES = [
    "massspecgym_default_0010",
    "massspecgym_default_0038",
    "massspecgym_default_0086",
]
PAIR_TIMEOUT_SECONDS = 1
SIMILARITY_THRESHOLD = 0.0
MAX_BOND_MATCH_PAIRS = 10_000


def compute_all_best_top(smiles1: str, smiles2: str) -> tuple[int, int, float, bool]:
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    opts = RascalOptions()
    opts.similarityThreshold = SIMILARITY_THRESHOLD
    opts.returnEmptyMCES = True
    opts.timeout = PAIR_TIMEOUT_SECONDS
    opts.maxBondMatchPairs = MAX_BOND_MATCH_PAIRS
    opts.allBestMCESs = True

    results = FindMCES(mol1, mol2, opts)
    if not results:
        return 0, 0, 0.0, False

    top = results[0]
    return (
        len(top.bondMatches()),
        len(top.atomMatches()),
        top.similarity,
        top.timedOut,
    )


def main() -> None:
    with gzip.open(INPUT_PATH, "rt", encoding="utf-8") as fh:
        source = json.load(fh)

    cases_by_name = {case["name"]: case for case in source["cases"]}
    output_cases = []

    for name in CASE_NAMES:
        case = dict(cases_by_name[name])
        bonds, atoms, similarity, timed_out = compute_all_best_top(case["smiles1"], case["smiles2"])
        if timed_out:
            raise RuntimeError(f"{name} timed out under allBestMCESs")
        case["expected_bond_matches"] = bonds
        case["expected_atom_matches"] = atoms
        case["expected_similarity"] = round(similarity, 6)
        output_cases.append(case)

    with gzip.open(OUTPUT_PATH, "wt", encoding="utf-8") as fh:
        json.dump({"version": source["version"], "cases": output_cases}, fh, indent=2)

    print(f"Wrote {len(output_cases)} allBest cases to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
