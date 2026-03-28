#!/usr/bin/env python3
"""Generate a larger default-config MCES ground truth corpus from MassSpecGym SMILES.

Usage:
    uv run --with rdkit --with tqdm python3 tests/fixtures/generate_massspecgym_mces_ground_truth.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from dataclasses import dataclass
from pathlib import Path
import statistics
import time

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdRascalMCES import FindMCES, RascalOptions
from tqdm.auto import tqdm

from generate_mces_ground_truth import mol_to_graph

INPUT_PATH = Path("tests/fixtures/massspecgym_unique_smiles.csv.gz")
DEFAULT_OUTPUT_PATH = Path("tests/fixtures/massspecgym_mces_default_100.json.gz")
DEFAULT_TARGET_CASES = 100
DEFAULT_RESERVOIR_CASES = 2000
DEFAULT_PAIR_TIMEOUT_SECONDS = 1
SIMILARITY_THRESHOLD = 0.0
PAIR_SIMILARITY_LOWER_CUTOFF = 0.6
PAIR_SIMILARITY_UPPER_CUTOFF = 0.975
MAX_BOND_MATCH_PAIRS = 10_000
RNG_SEED = 20260324


@dataclass(frozen=True)
class MoleculeRecord:
    smiles: str
    mol: Chem.Mol
    fingerprint: DataStructs.ExplicitBitVect
    heavy_atoms: int


@dataclass(frozen=True)
class CandidatePair:
    left_index: int
    right_index: int
    similarity: float


def read_unique_smiles(path: Path) -> list[str]:
    with gzip.open(path, "rt", newline="") as fh:
        reader = csv.DictReader(fh)
        smiles = [row["smiles"].strip() for row in reader if row.get("smiles", "").strip()]
    return smiles


def load_molecules(smiles_list: list[str]) -> list[MoleculeRecord]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    records = []
    for smiles in tqdm(smiles_list, desc="load molecules", unit="mol"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            tqdm.write(f"SKIP invalid SMILES: {smiles}")
            continue
        records.append(
            MoleculeRecord(
                smiles=smiles,
                mol=mol,
                fingerprint=generator.GetFingerprint(mol),
                heavy_atoms=mol.GetNumHeavyAtoms(),
            )
        )
    return records


def sample_high_similarity_pairs(
    records: list[MoleculeRecord],
    rng: random.Random,
    reservoir_size: int,
) -> tuple[list[CandidatePair], int]:
    reservoir: list[CandidatePair] = []
    fingerprints = [record.fingerprint for record in records]
    qualifying_pairs = 0

    scan_bar = tqdm(
        enumerate(fingerprints[:-1]),
        total=max(len(fingerprints) - 1, 0),
        desc="fingerprint scan",
        unit="mol",
    )
    for left_index, left_fp in scan_bar:
        similarities = DataStructs.BulkTanimotoSimilarity(left_fp, fingerprints[left_index + 1 :])
        for right_offset, similarity in enumerate(similarities, start=left_index + 1):
            if similarity <= PAIR_SIMILARITY_LOWER_CUTOFF or similarity > PAIR_SIMILARITY_UPPER_CUTOFF:
                continue

            qualifying_pairs += 1
            candidate = CandidatePair(left_index, right_offset, similarity)
            if len(reservoir) < reservoir_size:
                reservoir.append(candidate)
                continue

            replacement_index = rng.randrange(qualifying_pairs)
            if replacement_index < reservoir_size:
                reservoir[replacement_index] = candidate

        if (left_index + 1) % 100 == 0:
            scan_bar.set_postfix(qualifying_pairs=qualifying_pairs, reservoir=len(reservoir))

    return reservoir, qualifying_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--target-cases", type=int, default=DEFAULT_TARGET_CASES)
    parser.add_argument("--reservoir-cases", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_PAIR_TIMEOUT_SECONDS)
    return parser.parse_args()


def compute_default_mces(
    first: Chem.Mol,
    second: Chem.Mol,
    timeout_seconds: int,
) -> tuple[int, int, float, bool, float]:
    opts = RascalOptions()
    opts.similarityThreshold = SIMILARITY_THRESHOLD
    opts.returnEmptyMCES = True
    opts.timeout = timeout_seconds
    opts.maxBondMatchPairs = MAX_BOND_MATCH_PAIRS

    started = time.perf_counter()
    results = FindMCES(first, second, opts)
    elapsed = time.perf_counter() - started
    if not results:
        return 0, 0, 0.0, False, elapsed

    result = results[0]
    return (
        len(result.bondMatches()),
        len(result.atomMatches()),
        result.similarity,
        result.timedOut,
        elapsed,
    )


def build_case(
    name: str,
    left: MoleculeRecord,
    right: MoleculeRecord,
    timeout_seconds: int,
) -> dict:
    bond_matches, atom_matches, similarity, timed_out, elapsed = compute_default_mces(
        left.mol,
        right.mol,
        timeout_seconds,
    )
    if timed_out:
        raise TimeoutError(name)

    return {
        "name": name,
        "smiles1": left.smiles,
        "smiles2": right.smiles,
        "graph1": mol_to_graph(left.smiles),
        "graph2": mol_to_graph(right.smiles),
        "expected_bond_matches": bond_matches,
        "expected_atom_matches": atom_matches,
        "expected_similarity": round(similarity, 6),
        "rdkit_elapsed_seconds": round(elapsed, 6),
        "timed_out": False,
        "options": {},
    }


def generate_cases(
    records: list[MoleculeRecord],
    sampled_pairs: list[CandidatePair],
    target_cases: int,
    timeout_seconds: int,
) -> tuple[list[dict], int]:
    cases = []
    case_index = 1
    timed_out_pairs = 0
    started = time.perf_counter()
    case_bar = tqdm(total=target_cases, desc="build cases", unit="case")

    for processed_pairs, candidate in enumerate(sampled_pairs, start=1):
        left = records[candidate.left_index]
        right = records[candidate.right_index]
        case_name = f"massspecgym_default_{case_index:04d}"
        try:
            case = build_case(case_name, left, right, timeout_seconds)
        except TimeoutError:
            timed_out_pairs += 1
        else:
            cases.append(case)
            case_index += 1
            case_bar.update(1)
            if len(cases) >= target_cases:
                break

        if processed_pairs % 25 == 0:
            case_bar.set_postfix(
                processed_pairs=processed_pairs,
                timeouts=timed_out_pairs,
                elapsed_s=f"{time.perf_counter() - started:.1f}",
            )

    case_bar.set_postfix(
        processed_pairs=min(len(sampled_pairs), max(len(cases) + timed_out_pairs, 0)),
        timeouts=timed_out_pairs,
        elapsed_s=f"{time.perf_counter() - started:.1f}",
    )
    case_bar.close()

    if len(cases) != target_cases:
        raise RuntimeError(
            f"only collected {len(cases)} non-timeout cases from the Morgan-similar sample; "
            f"increase RESERVOIR_CASES or relax the timeout"
        )
    return cases, timed_out_pairs


def main() -> None:
    args = parse_args()
    reservoir_cases = (
        args.reservoir_cases
        if args.reservoir_cases is not None
        else max(DEFAULT_RESERVOIR_CASES, args.target_cases * 20)
    )

    pipeline_bar = tqdm(total=5, desc="pipeline", unit="step")
    smiles_list = read_unique_smiles(INPUT_PATH)
    pipeline_bar.update(1)
    records = load_molecules(smiles_list)
    pipeline_bar.update(1)
    if len(records) < 2:
        raise RuntimeError("need at least two valid molecules to generate MCES pairs")

    rng = random.Random(RNG_SEED)
    sampled_pairs, qualifying_pairs = sample_high_similarity_pairs(records, rng, reservoir_cases)
    pipeline_bar.update(1)
    if qualifying_pairs < args.target_cases:
        raise RuntimeError(
            f"only found {qualifying_pairs} Morgan-similar pairs in "
            f"({PAIR_SIMILARITY_LOWER_CUTOFF}, {PAIR_SIMILARITY_UPPER_CUTOFF}]; "
            f"cannot sample {args.target_cases} cases"
        )

    rng.shuffle(sampled_pairs)
    cases, timed_out_pairs = generate_cases(
        records,
        sampled_pairs,
        args.target_cases,
        args.timeout_seconds,
    )
    pipeline_bar.update(1)

    with gzip.open(args.output, "wt", encoding="utf-8") as fh:
        json.dump({"version": 4, "cases": cases}, fh, indent=2)
    pipeline_bar.update(1)
    pipeline_bar.close()

    bond_matches = [case["expected_bond_matches"] for case in cases]
    similarities = [case["expected_similarity"] for case in cases]
    elapsed_seconds = [case["rdkit_elapsed_seconds"] for case in cases]

    print(f"\nInput molecules: {len(records)}")
    print(
        "Pairs with Morgan Tanimoto in "
        f"({PAIR_SIMILARITY_LOWER_CUTOFF}, {PAIR_SIMILARITY_UPPER_CUTOFF}]: {qualifying_pairs}"
    )
    print(f"Sampled candidate pairs: {len(sampled_pairs)}")
    print(f"Discarded timed-out pairs: {timed_out_pairs}")
    print(f"RDKit timeout per pair: {args.timeout_seconds}s")
    print(
        "Bond-match summary: "
        f"min={min(bond_matches)} median={statistics.median(bond_matches)} max={max(bond_matches)}"
    )
    print(
        "Similarity summary: "
        f"min={min(similarities):.6f} median={statistics.median(similarities):.6f} "
        f"max={max(similarities):.6f}"
    )
    print(
        "Elapsed summary (s): "
        f"min={min(elapsed_seconds):.6f} median={statistics.median(elapsed_seconds):.6f} "
        f"max={max(elapsed_seconds):.6f}"
    )
    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
