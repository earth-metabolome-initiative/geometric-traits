#!/usr/bin/env python3
"""Combine sharded topological-validity corpora into one deterministic corpus.

The shard generator emits 100K-case gzip JSON files with overlapping case names
(`family_0000000`..). This combiner concatenates shards in filename order,
rewrites case names to a unique global index, and refreshes aggregate metadata.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
from collections import Counter
from pathlib import Path


def load_shard(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def rewrite_case_name(family: str, index: int) -> str:
    return f"{family}_{index:07d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=10)
    args = parser.parse_args()

    shard_paths = sorted(args.shard_dir.glob("*.json.gz"))
    if len(shard_paths) != args.expected_shards:
        raise SystemExit(
            f"expected {args.expected_shards} shard files in {args.shard_dir}, found {len(shard_paths)}"
        )

    suites = [load_shard(path) for path in shard_paths]
    if not suites:
        raise SystemExit("no shard files found")

    first = suites[0]
    family_sequence = first["family_sequence"]
    total_count = 0
    totals = Counter()
    combined_cases: list[dict] = []

    for suite in suites:
        if suite["schema_version"] != first["schema_version"]:
            raise SystemExit("schema_version mismatch across shards")
        if suite["algorithm"] != first["algorithm"]:
            raise SystemExit("algorithm mismatch across shards")
        if suite["graph_kind"] != first["graph_kind"]:
            raise SystemExit("graph_kind mismatch across shards")
        if suite["primary_oracle"] != first["primary_oracle"]:
            raise SystemExit("primary_oracle mismatch across shards")
        if suite["family_sequence"] != family_sequence:
            raise SystemExit("family_sequence mismatch across shards")

        for case in suite["cases"]:
            global_index = total_count
            total_count += 1
            family = case["family"]
            rewritten = dict(case)
            rewritten["name"] = rewrite_case_name(family, global_index)
            combined_cases.append(rewritten)
            totals["is_planar"] += int(rewritten["is_planar"])
            totals["is_outerplanar"] += int(rewritten["is_outerplanar"])
            totals["has_k23_homeomorph"] += int(rewritten["has_k23_homeomorph"])
            totals["has_k33_homeomorph"] += int(rewritten["has_k33_homeomorph"])
            totals["has_k4_homeomorph"] += int(rewritten["has_k4_homeomorph"])

    output = {
        "schema_version": first["schema_version"],
        "algorithm": first["algorithm"],
        "graph_kind": first["graph_kind"],
        "generator": "edge_addition_planarity_suite_combined_topological_validity_cli_v4_sharded",
        "primary_oracle": first["primary_oracle"],
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "count": total_count,
        "family_sequence": family_sequence,
        "cases": combined_cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(output, handle, separators=(",", ":"))

    print(f"wrote {total_count} cases to {args.output}")
    print(
        "summary: "
        f"is_planar={totals['is_planar']}, "
        f"is_outerplanar={totals['is_outerplanar']}, "
        f"has_k23_homeomorph={totals['has_k23_homeomorph']}, "
        f"has_k33_homeomorph={totals['has_k33_homeomorph']}, "
        f"has_k4_homeomorph={totals['has_k4_homeomorph']}"
    )


if __name__ == "__main__":
    main()
