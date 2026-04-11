#!/usr/bin/env python3
"""Generate a combined Boyer-backed oracle corpus for topological validity.

The installed ``planarity`` CLI is used in specific-graph mode for:

- ``-p``: planarity
- ``-o``: outerplanarity
- ``-2``: ``K_{2,3}`` homeomorph detection
- ``-3``: ``K_{3,3}`` homeomorph detection
- ``-4``: ``K_4`` homeomorph detection

The output is a single deterministic gzip-compressed JSON file that stores one
graph stream and all five boolean oracle values per case.
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


DEFAULT_COUNT = 1_000_000
DEFAULT_SEED = 0
DEFAULT_MIN_ORDER = 1
DEFAULT_MAX_ORDER = 16
DEFAULT_OUT_PATH = Path(__file__).with_name("topological_validity_ground_truth_1m.json.gz")

FAMILY_SEQUENCE = (
    "erdos_renyi",
    "random_tree",
    "outerplanar_cycle_chords",
    "wheel",
    "theta",
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


def classify_planarity_mode(planarity_bin: str, mode: str, input_path: Path, output_path: Path) -> bool:
    completed = subprocess.run(
        [planarity_bin, "-s", "-q", mode, str(input_path), str(output_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode not in (0, 1):
        raise RuntimeError(
            f"planarity {mode} failed on {input_path} with unexpected exit code {completed.returncode}"
        )
    return completed.returncode == 1


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


def theta_edges(rng: random.Random, max_order: int) -> set[tuple[int, int]]:
    max_internal = max(0, max_order - 2)
    lengths = [1, 1, 1]
    budget = min(max_internal, rng.randint(0, max(0, max_internal - 3)))
    while budget > 0:
        lengths[rng.randrange(3)] += 1
        budget -= 1

    next_vertex = 2
    edges: set[tuple[int, int]] = set()
    for length in lengths:
        path = [0]
        for _ in range(length):
            path.append(next_vertex)
            next_vertex += 1
        path.append(1)
        for left, right in zip(path, path[1:]):
            edges.add(normalize_edge(left, right))
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
            "Cycle with noncrossing random chords.",
        )

    if family == "wheel":
        order = rng.randint(max(4, min_order), max(4, max_order))
        return order, canonical_edges(wheel_edges(order)), "Wheel graph."

    if family == "theta":
        effective_max = max(5, max_order)
        edges = theta_edges(rng, effective_max)
        node_count = max((max(max(edge) for edge in edges) + 1), 2)
        return node_count, canonical_edges(edges), "Theta graph with three internally disjoint paths."

    if family == "clique":
        order = rng.randint(max(1, min_order), max(1, max_order))
        return order, canonical_edges(clique_edges(order)), "Complete graph K_n."

    if family == "k23_subdivision":
        effective_max = max(5, max_order)
        edges = k23_subdivision_edges(rng, effective_max)
        node_count = max(max(edge) for edge in edges) + 1
        return node_count, canonical_edges(edges), "Random subdivision of K2,3."

    if family == "k33_subdivision":
        effective_max = max(6, max_order)
        edges = k33_subdivision_edges(rng, effective_max)
        node_count = max(max(edge) for edge in edges) + 1
        return node_count, canonical_edges(edges), "Random subdivision of K3,3."

    if family == "k4_subdivision":
        effective_max = max(5, max_order)
        edges = k4_subdivision_edges(rng, effective_max)
        node_count = max(max(edge) for edge in edges) + 1
        return node_count, canonical_edges(edges), "Random subdivision of K4."

    if family == "k5_subdivision":
        effective_max = max(6, max_order)
        edges = k5_subdivision_edges(rng, effective_max)
        node_count = max(max(edge) for edge in edges) + 1
        return node_count, canonical_edges(edges), "Random subdivision of K5."

    raise ValueError(f"unsupported family {family}")


def emit_progress(current: int, total: int, start_time: float, family: str) -> None:
    elapsed = time.perf_counter() - start_time
    rate = current / elapsed if elapsed > 0.0 else 0.0
    remaining = max(total - current, 0)
    eta_seconds = remaining / rate if rate > 0.0 else math.inf
    percentage = (100.0 * current / total) if total else 100.0
    eta_text = "--:--:--" if not math.isfinite(eta_seconds) else time.strftime(
        "%H:%M:%S", time.gmtime(max(0, int(eta_seconds)))
    )
    sys.stderr.write(
        f"\r[topology-1m] {current:>8}/{total:<8} {percentage:6.2f}%  "
        f"{rate:8.1f}/s  eta {eta_text}  family={family:<24}"
    )
    sys.stderr.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planarity-bin", default="planarity")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-order", type=int, default=DEFAULT_MIN_ORDER)
    parser.add_argument("--max-order", type=int, default=DEFAULT_MAX_ORDER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.min_order <= 0:
        raise SystemExit("--min-order must be positive")
    if args.max_order < args.min_order:
        raise SystemExit("--max-order must be >= --min-order")

    rng = random.Random(args.seed)
    family_counts: Counter[str] = Counter()
    is_planar_counts: Counter[str] = Counter()
    is_outerplanar_counts: Counter[str] = Counter()
    has_k23_counts: Counter[str] = Counter()
    has_k33_counts: Counter[str] = Counter()
    has_k4_counts: Counter[str] = Counter()
    totals = {
        "is_planar": 0,
        "is_outerplanar": 0,
        "has_k23_homeomorph": 0,
        "has_k33_homeomorph": 0,
        "has_k4_homeomorph": 0,
    }
    start_time = time.perf_counter()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="topological_validity_ground_truth_") as scratch:
        scratch_dir = Path(scratch)
        input_path = scratch_dir / "current_graph.txt"
        mode_outputs = {
            "p": scratch_dir / "current_p.out.txt",
            "o": scratch_dir / "current_o.out.txt",
            "2": scratch_dir / "current_2.out.txt",
            "3": scratch_dir / "current_3.out.txt",
            "4": scratch_dir / "current_4.out.txt",
        }

        with gzip.open(args.output, "wt", encoding="utf-8", compresslevel=9) as handle:
            header = {
                "schema_version": 1,
                "algorithm": "boyer_edge_addition_reference_cli",
                "graph_kind": "undirected_simple_labeled",
                "generator": "edge_addition_planarity_suite_combined_topological_validity_cli",
                "primary_oracle": "planarity_outerplanarity_k23_k33_k4_booleans",
                "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
                "seed": args.seed,
                "count": args.count,
                "family_sequence": list(FAMILY_SEQUENCE),
                "cases": [],
            }
            header_prefix = json.dumps(header, separators=(",", ":"))[:-2]
            handle.write(header_prefix)

            for index in range(args.count):
                family = FAMILY_SEQUENCE[index % len(FAMILY_SEQUENCE)]
                node_count, edges, notes = graph_for_family(
                    rng, family, min_order=args.min_order, max_order=args.max_order
                )
                case_name = f"{family}_{index:07d}"
                write_planarity_input(input_path, node_count, edges)

                is_nonplanar = classify_planarity_mode(
                    args.planarity_bin, "-p", input_path, mode_outputs["p"]
                )
                is_nonouterplanar = classify_planarity_mode(
                    args.planarity_bin, "-o", input_path, mode_outputs["o"]
                )
                has_k23_homeomorph = classify_planarity_mode(
                    args.planarity_bin, "-2", input_path, mode_outputs["2"]
                )
                has_k33_homeomorph = classify_planarity_mode(
                    args.planarity_bin, "-3", input_path, mode_outputs["3"]
                )
                has_k4_homeomorph = classify_planarity_mode(
                    args.planarity_bin, "-4", input_path, mode_outputs["4"]
                )

                case = {
                    "name": case_name,
                    "family": family,
                    "node_count": node_count,
                    "edges": edges,
                    "is_planar": not is_nonplanar,
                    "is_outerplanar": not is_nonouterplanar,
                    "has_k23_homeomorph": has_k23_homeomorph,
                    "has_k33_homeomorph": has_k33_homeomorph,
                    "has_k4_homeomorph": has_k4_homeomorph,
                    "notes": notes,
                }

                if index > 0:
                    handle.write(",")
                handle.write(json.dumps(case, separators=(",", ":")))

                family_counts[family] += 1
                is_planar_counts[family] += int(case["is_planar"])
                is_outerplanar_counts[family] += int(case["is_outerplanar"])
                has_k23_counts[family] += int(has_k23_homeomorph)
                has_k33_counts[family] += int(has_k33_homeomorph)
                has_k4_counts[family] += int(has_k4_homeomorph)
                totals["is_planar"] += int(case["is_planar"])
                totals["is_outerplanar"] += int(case["is_outerplanar"])
                totals["has_k23_homeomorph"] += int(has_k23_homeomorph)
                totals["has_k33_homeomorph"] += int(has_k33_homeomorph)
                totals["has_k4_homeomorph"] += int(has_k4_homeomorph)

                if (index + 1) % 250 == 0 or index + 1 == args.count:
                    emit_progress(index + 1, args.count, start_time, family)

            handle.write("]}")

    elapsed = time.perf_counter() - start_time
    sys.stderr.write("\n")
    print(f"wrote {args.count} cases to {args.output}")
    print(
        "summary: "
        f"is_planar={totals['is_planar']}, "
        f"is_outerplanar={totals['is_outerplanar']}, "
        f"has_k23_homeomorph={totals['has_k23_homeomorph']}, "
        f"has_k33_homeomorph={totals['has_k33_homeomorph']}, "
        f"has_k4_homeomorph={totals['has_k4_homeomorph']}, "
        f"elapsed={elapsed:.2f}s"
    )
    print("family counts:")
    for family in FAMILY_SEQUENCE:
        print(
            f"  {family}: total={family_counts[family]} "
            f"planar={is_planar_counts[family]} "
            f"outerplanar={is_outerplanar_counts[family]} "
            f"k23={has_k23_counts[family]} "
            f"k33={has_k33_counts[family]} "
            f"k4={has_k4_counts[family]}"
        )


if __name__ == "__main__":
    main()
