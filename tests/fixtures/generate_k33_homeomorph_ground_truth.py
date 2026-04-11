#!/usr/bin/env python3
"""Generate Boyer-backed ground truth for K_{3,3} homeomorph detection.

The oracle is the installed ``planarity`` CLI in specific-graph mode with
``-3``, which searches for a subgraph homeomorphic to ``K_{3,3}``.

The generated corpus is deterministic for a fixed seed and stores only the
boolean oracle result plus the concrete graph instance for each case.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path


DEFAULT_COUNT = 100_000
DEFAULT_SEED = 0
DEFAULT_MIN_ORDER = 1
DEFAULT_MAX_ORDER = 16
DEFAULT_OUT_PATH = Path(__file__).with_name("k33_homeomorph_ground_truth_100k.json.gz")

FAMILY_SEQUENCE = (
    "erdos_renyi",
    "random_tree",
    "outerplanar_cycle_chords",
    "wheel",
    "clique",
    "k23_subdivision",
    "k33_subdivision",
    "k4_subdivision",
    "k5_subdivision",
)


def normalize_edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def canonical_edges(edges: set[tuple[int, int]]) -> list[list[int]]:
    return [[left, right] for left, right in sorted(edges)]


def write_planarity_input(path: Path, node_count: int, edges: list[list[int]]) -> None:
    adjacency = [[] for _ in range(node_count)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"N={node_count}\n")
        for vertex, neighbors in enumerate(adjacency):
            ordered = " ".join(str(neighbor) for neighbor in sorted(neighbors))
            if ordered:
                handle.write(f"{vertex}: {ordered} -1\n")
            else:
                handle.write(f"{vertex}: -1\n")


def classify_k33(planarity_bin: str, input_path: Path, output_path: Path) -> bool:
    completed = subprocess.run(
        [planarity_bin, "-s", "-q", "-3", str(input_path), str(output_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode == 1:
        return True
    if completed.returncode == 0:
        return False
    raise RuntimeError(
        f"planarity -3 failed on {input_path} with unexpected exit code {completed.returncode}"
    )


def random_tree_edges(rng: random.Random, order: int) -> set[tuple[int, int]]:
    if order <= 1:
        return set()
    edges: set[tuple[int, int]] = set()
    for vertex in range(1, order):
        parent = rng.randrange(vertex)
        edges.add(normalize_edge(parent, vertex))
    return edges


def erdos_renyi_edges(rng: random.Random, order: int) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    density = rng.choice((0.08, 0.12, 0.18, 0.24, 0.32, 0.45))
    for left in range(order):
        for right in range(left + 1, order):
            if rng.random() < density:
                edges.add((left, right))
    return edges


def clique_edges(order: int) -> set[tuple[int, int]]:
    return {
        (left, right)
        for left in range(order)
        for right in range(left + 1, order)
    }


def outerplanar_cycle_chords_edges(rng: random.Random, order: int) -> set[tuple[int, int]]:
    if order <= 1:
        return set()
    if order == 2:
        return {(0, 1)}

    edges = {normalize_edge(vertex, (vertex + 1) % order) for vertex in range(order)}
    chords: list[tuple[int, int]] = []

    def crosses(candidate: tuple[int, int], existing: tuple[int, int]) -> bool:
        a, b = candidate
        c, d = existing
        if len({a, b, c, d}) < 4:
            return False
        if a > b:
            a, b = b, a
        if c > d:
            c, d = d, c
        return (a < c < b < d) or (c < a < d < b)

    candidates = [
        normalize_edge(left, right)
        for left in range(order)
        for right in range(left + 1, order)
        if right not in {left + 1, (left - 1) % order} and not (left == 0 and right == order - 1)
    ]
    rng.shuffle(candidates)
    for candidate in candidates:
        if all(not crosses(candidate, chord) for chord in chords):
            chords.append(candidate)
            edges.add(candidate)
            if rng.random() < 0.35:
                break
    return edges


def wheel_edges(order: int) -> set[tuple[int, int]]:
    if order <= 1:
        return set()
    if order == 2:
        return {(0, 1)}
    if order == 3:
        return {(0, 1), (1, 2), (0, 2)}

    center = 0
    rim = list(range(1, order))
    edges = {normalize_edge(center, vertex) for vertex in rim}
    for index, vertex in enumerate(rim):
        edges.add(normalize_edge(vertex, rim[(index + 1) % len(rim)]))
    return edges


def subdivide_base_edges(
    rng: random.Random,
    base_edges: list[tuple[int, int]],
    max_order: int,
    base_vertex_count: int,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    next_vertex = base_vertex_count
    remaining_budget = max(0, max_order - base_vertex_count)
    subdivision_budget = [0] * len(base_edges)
    while remaining_budget > 0:
        subdivision_budget[rng.randrange(len(base_edges))] += 1
        remaining_budget -= 1

    for edge_index, (left, right) in enumerate(base_edges):
        subdivisions = subdivision_budget[edge_index]
        previous = left
        for _ in range(subdivisions):
            current = next_vertex
            next_vertex += 1
            edges.add(normalize_edge(previous, current))
            previous = current
        edges.add(normalize_edge(previous, right))
    return edges


def k23_subdivision_edges(rng: random.Random, max_order: int) -> set[tuple[int, int]]:
    base_edges = [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)]
    return subdivide_base_edges(rng, base_edges, max_order=max(max_order, 5), base_vertex_count=5)


def k33_subdivision_edges(rng: random.Random, max_order: int) -> set[tuple[int, int]]:
    base_edges = [
        (0, 3),
        (0, 4),
        (0, 5),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 3),
        (2, 4),
        (2, 5),
    ]
    return subdivide_base_edges(rng, base_edges, max_order=max(max_order, 6), base_vertex_count=6)


def k4_subdivision_edges(rng: random.Random, max_order: int) -> set[tuple[int, int]]:
    base_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return subdivide_base_edges(rng, base_edges, max_order=max(max_order, 4), base_vertex_count=4)


def k5_subdivision_edges(rng: random.Random, max_order: int) -> set[tuple[int, int]]:
    base_edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]
    return subdivide_base_edges(rng, base_edges, max_order=max(max_order, 5), base_vertex_count=5)


def graph_for_family(
    rng: random.Random,
    family: str,
    min_order: int,
    max_order: int,
) -> tuple[int, list[list[int]], str]:
    if family == "erdos_renyi":
        order = rng.randint(max(1, min_order), max(1, max_order))
        return order, canonical_edges(erdos_renyi_edges(rng, order)), "Random G(n,p) sample."

    if family == "random_tree":
        order = rng.randint(max(1, min_order), max(1, max_order))
        return order, canonical_edges(random_tree_edges(rng, order)), "Random labeled tree."

    if family == "outerplanar_cycle_chords":
        order = rng.randint(max(3, min_order), max(3, max_order))
        return (
            order,
            canonical_edges(outerplanar_cycle_chords_edges(rng, order)),
            "Cycle with noncrossing chords; stays outerplanar by construction.",
        )

    if family == "wheel":
        order = rng.randint(max(4, min_order), max(4, max_order))
        return order, canonical_edges(wheel_edges(order)), "Wheel graph W_n."

    if family == "clique":
        order = rng.randint(max(1, min_order), max(1, max_order))
        return order, canonical_edges(clique_edges(order)), "Complete graph K_n."

    effective_max = max(max_order, min_order)
    if family == "k23_subdivision":
        edges = k23_subdivision_edges(rng, effective_max)
        order = max(5, max((max(left, right) for left, right in edges), default=4) + 1)
        return order, canonical_edges(edges), "Subdivision of K_{2,3}."

    if family == "k33_subdivision":
        edges = k33_subdivision_edges(rng, effective_max)
        order = max(6, max((max(left, right) for left, right in edges), default=5) + 1)
        return order, canonical_edges(edges), "Subdivision of K_{3,3}."

    if family == "k4_subdivision":
        edges = k4_subdivision_edges(rng, effective_max)
        order = max(4, max((max(left, right) for left, right in edges), default=3) + 1)
        return order, canonical_edges(edges), "Subdivision of K_4."

    if family == "k5_subdivision":
        edges = k5_subdivision_edges(rng, effective_max)
        order = max(5, max((max(left, right) for left, right in edges), default=4) + 1)
        return order, canonical_edges(edges), "Subdivision of K_5."

    raise ValueError(f"unknown family: {family}")


def progress_line(current: int, total: int, elapsed: float) -> str:
    rate = current / elapsed if elapsed > 0 else 0.0
    percentage = (100.0 * current / total) if total else 100.0
    remaining = total - current
    eta = remaining / rate if rate > 0 else math.inf
    eta_display = "--:--:--" if not math.isfinite(eta) else time.strftime("%H:%M:%S", time.gmtime(eta))
    return (
        f"\r[k33-100k] {current:>7}/{total:<7} {percentage:6.2f}%  "
        f"{rate:8.1f}/s  eta {eta_display}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planarity-bin", required=True, help="path to the planarity executable")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-order", type=int, default=DEFAULT_MIN_ORDER)
    parser.add_argument("--max-order", type=int, default=DEFAULT_MAX_ORDER)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    family_sequence = list(FAMILY_SEQUENCE)
    family_counts: Counter[str] = Counter()
    positive_counts: Counter[str] = Counter()
    overall_positive_count = 0
    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="k33_ground_truth_") as scratch:
        scratch_dir = Path(scratch)
        cases = []
        for case_index in range(args.count):
            family = family_sequence[case_index % len(family_sequence)]
            node_count, edges, notes = graph_for_family(
                rng, family, args.min_order, args.max_order
            )
            input_path = scratch_dir / f"case_{case_index:06d}.txt"
            output_path = scratch_dir / f"case_{case_index:06d}.out.txt"
            write_planarity_input(input_path, node_count, edges)
            has_k33_homeomorph = classify_k33(args.planarity_bin, input_path, output_path)

            cases.append(
                {
                    "name": f"{family}_{case_index:06d}",
                    "family": family,
                    "node_count": node_count,
                    "edges": edges,
                    "has_k33_homeomorph": has_k33_homeomorph,
                    "notes": notes,
                }
            )
            family_counts[family] += 1
            positive_counts[family] += int(has_k33_homeomorph)
            overall_positive_count += int(has_k33_homeomorph)

            if (case_index + 1) % 100 == 0 or case_index + 1 == args.count:
                elapsed = time.perf_counter() - start_time
                sys.stderr.write(progress_line(case_index + 1, args.count, elapsed))
                sys.stderr.flush()

    sys.stderr.write("\n")

    payload = {
        "schema_version": 1,
        "algorithm": "boyer_edge_addition_reference_cli",
        "graph_kind": "undirected_simple_labeled",
        "generator": "edge_addition_planarity_suite_k33_mixed_family_cli",
        "primary_oracle": "k33_homeomorph_boolean",
        "generated_at": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "seed": args.seed,
        "family_sequence": family_sequence,
        "cases": cases,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))

    elapsed = time.perf_counter() - start_time
    print(f"wrote {args.out} ({args.count} cases)")
    print(
        f"summary: has_k33_homeomorph={overall_positive_count}, "
        f"without_k33_homeomorph={args.count - overall_positive_count}, elapsed={elapsed:.2f}s"
    )
    for family in family_sequence:
        total = family_counts[family]
        positives = positive_counts[family]
        negatives = total - positives
        print(f"family={family} total={total} positive={positives} negative={negatives}")


if __name__ == "__main__":
    main()
