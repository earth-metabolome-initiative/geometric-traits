# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the directed community detection ground-truth corpus.

Runs the reference C++ Directed Louvain implementation
(github.com/nicolasdugue/DirectedLouvain) on a set of directed graphs and
records, per case, the reference final partition (projected to the original
nodes) together with the directed modularity the reference reports and the
directed modularity recomputed here from the same formula our crate uses.

The reference is deterministic when invoked with `-s` (no shuffle) and `-n`
(already numbered 0..n-1). The directed modularity is the Leicht-Newman / Arenas
form:

    Q = sum_C [ internal_C / m - gamma * (Sigma_out^C * Sigma_in^C) / m^2 ]

Usage:
    DIRECTED_LOUVAIN_BIN=/path/to/DirectedLouvain/bin uv run generate.py

The output is written gzipped to
tests/fixtures/directed_community_ground_truth.json.gz.
"""

from __future__ import annotations

import gzip
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
GAMMA = 1.0
REFERENCE_REPO = "https://github.com/nicolasdugue/DirectedLouvain"
REFERENCE_COMMIT = "90edb0f34df770f8a4e7c473040f0d22655675d3"

BIN_DIR = Path(os.environ.get("DIRECTED_LOUVAIN_BIN", "/tmp/DirectedLouvain/bin"))
OUTPUT = (
    Path(__file__).resolve().parents[2] / "directed_community_ground_truth.json.gz"
)

Edge = tuple[int, int, float]


def directed_modularity(
    node_count: int, edges: list[Edge], partition: list[int], gamma: float
) -> float:
    """Directed modularity of `partition`, matching DirectedWorkingGraph."""
    out_degree = [0.0] * node_count
    in_degree = [0.0] * node_count
    total = 0.0
    for source, destination, weight in edges:
        out_degree[source] += weight
        in_degree[destination] += weight
        total += weight
    if total <= 0.0:
        return 0.0

    community_count = max(partition) + 1
    internal = [0.0] * community_count
    sigma_out = [0.0] * community_count
    sigma_in = [0.0] * community_count
    for node in range(node_count):
        community = partition[node]
        sigma_out[community] += out_degree[node]
        sigma_in[community] += in_degree[node]
    for source, destination, weight in edges:
        if partition[source] == partition[destination]:
            internal[partition[source]] += weight

    inverse = 1.0 / total
    return sum(
        internal[c] * inverse - gamma * (sigma_out[c] * inverse) * (sigma_in[c] * inverse)
        for c in range(community_count)
    )


def run_reference(node_count: int, edges: list[Edge]) -> tuple[list[int], float]:
    """Runs the reference on `edges`, returning (partition, reported modularity)."""
    community = BIN_DIR / "community"
    hierarchy = BIN_DIR / "hierarchy"
    with tempfile.TemporaryDirectory() as directory:
        graph_path = Path(directory) / "graph.txt"
        tree_path = Path(directory) / "graph.tree"
        with graph_path.open("w") as handle:
            for source, destination, weight in edges:
                handle.write(f"{source} {destination} {weight}\n")

        with tree_path.open("w") as tree_file:
            result = subprocess.run(
                [
                    str(community),
                    "-f",
                    str(graph_path),
                    "-l",
                    "-1",
                    "-n",
                    "-s",
                    "-g",
                    str(GAMMA),
                    "-v",
                ],
                stdout=tree_file,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
            )
        reported = 0.0
        for line in result.stderr.splitlines():
            if line.startswith("modularity:"):
                reported = float(line.split(":")[1])

        listing = subprocess.run(
            [str(hierarchy), str(tree_path), "-l", "-2"],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
        )

    partition = [0] * node_count
    for line in listing.stdout.splitlines():
        node_text, community_text = line.split()
        partition[int(node_text)] = int(community_text)
    return partition, reported


def directed_cycle(start: int, length: int, weight: float) -> list[Edge]:
    return [(start + i, start + (i + 1) % length, weight) for i in range(length)]


def directed_clique(start: int, size: int, weight: float) -> list[Edge]:
    edges = []
    for i in range(size):
        for j in range(size):
            if i != j:
                edges.append((start + i, start + j, weight))
    return edges


def planted_blocks(
    block_sizes: list[int], intra: float, inter: float, seed: int
) -> tuple[int, list[Edge]]:
    """Directed planted-partition digraph: dense within blocks, sparse across."""
    rng = random.Random(seed)
    offsets = []
    running = 0
    for size in block_sizes:
        offsets.append(running)
        running += size
    node_count = running
    block_of = [0] * node_count
    for block, (offset, size) in enumerate(zip(offsets, block_sizes)):
        for node in range(offset, offset + size):
            block_of[node] = block

    edges: dict[tuple[int, int], float] = {}
    for source in range(node_count):
        for destination in range(node_count):
            if source == destination:
                continue
            probability = intra if block_of[source] == block_of[destination] else inter
            if rng.random() < probability:
                edges[(source, destination)] = 1.0
    # Guarantee every node has at least one incident arc so the reference sees
    # the full node count under -n.
    incident = set()
    for source, destination in edges:
        incident.add(source)
        incident.add(destination)
    for node in range(node_count):
        if node not in incident:
            partner = (node + 1) % node_count
            edges[(node, partner)] = 1.0
    return node_count, [(s, d, w) for (s, d), w in sorted(edges.items())]


def build_cases() -> list[dict]:
    cases: list[dict] = []

    def add(case_id: str, node_count: int, edges: list[Edge]) -> None:
        cases.append({"id": case_id, "node_count": node_count, "edges": edges})

    add("two_2cycles", 4, directed_cycle(0, 2, 1.0) + directed_cycle(2, 2, 1.0))
    add("two_3cycles", 6, directed_cycle(0, 3, 1.0) + directed_cycle(3, 3, 1.0))
    add(
        "three_3cliques",
        9,
        directed_clique(0, 3, 1.0)
        + directed_clique(3, 3, 1.0)
        + directed_clique(6, 3, 1.0)
        + [(2, 3, 0.2), (5, 6, 0.2), (8, 0, 0.2)],
    )
    # A clear source (0) feeding two cliques, and a clear sink (9) drained by them.
    add(
        "sources_and_sinks",
        10,
        directed_clique(1, 4, 1.0)
        + directed_clique(5, 4, 1.0)
        + [(0, 1, 1.0), (0, 5, 1.0), (4, 9, 1.0), (8, 9, 1.0), (4, 5, 0.2)],
    )
    add(
        "self_loops",
        6,
        directed_cycle(0, 3, 1.0)
        + directed_cycle(3, 3, 1.0)
        + [(0, 0, 2.0), (3, 3, 2.0), (2, 3, 0.3)],
    )
    # Strongly asymmetric: a dense one-way bundle between two cliques.
    asymmetric = directed_clique(0, 3, 5.0) + directed_clique(3, 3, 5.0)
    for source in range(3):
        for destination in range(3, 6):
            asymmetric.append((source, destination, 1.0))
    add("strongly_asymmetric", 6, asymmetric)
    # A directed ring of four cliques.
    ring = []
    for block in range(4):
        ring += directed_clique(block * 4, 4, 1.0)
    for block in range(4):
        ring.append((block * 4 + 3, ((block + 1) % 4) * 4, 0.2))
    add("ring_of_4cliques", 16, ring)

    for index, sizes in enumerate(
        [[5, 5, 5], [8, 8, 8], [10, 10, 10, 10], [12, 13, 14, 11]]
    ):
        node_count, edges = planted_blocks(sizes, intra=0.6, inter=0.03, seed=100 + index)
        add(f"planted_{index}_{node_count}n", node_count, edges)

    return cases


def main() -> None:
    if not (BIN_DIR / "community").exists():
        raise SystemExit(
            f"reference binary not found in {BIN_DIR}; set DIRECTED_LOUVAIN_BIN"
        )

    cases = []
    for case in build_cases():
        partition, reported = run_reference(case["node_count"], case["edges"])
        recomputed = directed_modularity(
            case["node_count"], case["edges"], partition, GAMMA
        )
        cases.append(
            {
                "id": case["id"],
                "node_count": case["node_count"],
                "edges": [[s, d, w] for s, d, w in case["edges"]],
                "reference": {
                    "partition": partition,
                    "modularity": recomputed,
                    "reported_modularity": reported,
                },
            }
        )
        print(
            f"{case['id']:24s} n={case['node_count']:3d} "
            f"communities={max(partition) + 1:3d} Q={recomputed:.6f}"
        )

    document = {
        "schema_version": SCHEMA_VERSION,
        "parameters": {"resolution": GAMMA},
        "reference": {"repository": REFERENCE_REPO, "commit": REFERENCE_COMMIT},
        "cases": cases,
    }

    with gzip.open(OUTPUT, "wt", encoding="utf-8") as handle:
        json.dump(document, handle)
    print(f"wrote {len(cases)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
