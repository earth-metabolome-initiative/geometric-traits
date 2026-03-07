#!/usr/bin/env python3
"""Generate Louvain/Leiden ground-truth fixtures from reference implementations.

Uses:
- python-igraph (Louvain via `community_multilevel`)
- leidenalg (Leiden via `find_partition` + `ModularityVertexPartition`)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import platform
from typing import Iterable, Sequence

os.environ.setdefault("IGRAPH_NO_IMPORT_MATPLOTLIB", "1")

import igraph  # type: ignore
import leidenalg  # type: ignore


def _clique(nodes: Sequence[int], weight: float) -> list[tuple[int, int, float]]:
    edges: list[tuple[int, int, float]] = []
    for index, source in enumerate(nodes):
        for destination in nodes[index + 1 :]:
            edges.append((source, destination, weight))
    return edges


def _normalize_edges(node_count: int, edges: Iterable[tuple[int, int, float]]) -> list[dict[str, float | int]]:
    normalized: list[tuple[int, int, float]] = []
    seen: set[tuple[int, int]] = set()

    for source, target, weight in edges:
        if source < 0 or target < 0 or source >= node_count or target >= node_count:
            raise ValueError(f"Edge ({source}, {target}) is out of bounds for {node_count} nodes")
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"Edge ({source}, {target}) has invalid weight {weight}")

        if source > target:
            source, target = target, source

        key = (source, target)
        if key in seen:
            raise ValueError(f"Duplicate undirected edge ({source}, {target})")
        seen.add(key)
        normalized.append((source, target, float(weight)))

    normalized.sort(key=lambda edge: (edge[0], edge[1]))
    return [
        {"source": source, "target": target, "weight": weight}
        for source, target, weight in normalized
    ]


def _build_graph(node_count: int, edges: list[dict[str, float | int]]) -> igraph.Graph:
    graph = igraph.Graph(
        n=node_count,
        edges=[(int(edge["source"]), int(edge["target"])) for edge in edges],
        directed=False,
    )
    graph.es["weight"] = [float(edge["weight"]) for edge in edges]
    return graph


def _graph_cases() -> list[dict[str, object]]:
    return [
        {
            "id": "two_triangles_weak_bridge",
            "node_count": 6,
            "undirected_edges": _normalize_edges(
                6,
                _clique([0, 1, 2], 10.0)
                + _clique([3, 4, 5], 10.0)
                + [(2, 3, 0.1)],
            ),
        },
        {
            "id": "three_quads_chain",
            "node_count": 12,
            "undirected_edges": _normalize_edges(
                12,
                _clique([0, 1, 2, 3], 8.0)
                + _clique([4, 5, 6, 7], 8.0)
                + _clique([8, 9, 10, 11], 8.0)
                + [(3, 4, 0.05), (7, 8, 0.05)],
            ),
        },
        {
            "id": "ring_of_five_triads",
            "node_count": 15,
            "undirected_edges": _normalize_edges(
                15,
                _clique([0, 1, 2], 6.0)
                + _clique([3, 4, 5], 6.0)
                + _clique([6, 7, 8], 6.0)
                + _clique([9, 10, 11], 6.0)
                + _clique([12, 13, 14], 6.0)
                + [(2, 3, 0.03), (5, 6, 0.03), (8, 9, 0.03), (11, 12, 0.03), (14, 0, 0.03)],
            ),
        },
        {
            "id": "mixed_disconnected_components",
            "node_count": 10,
            "undirected_edges": _normalize_edges(
                10,
                _clique([0, 1, 2], 7.0)
                + _clique([3, 4, 5], 7.0)
                + [(6, 7, 7.0), (8, 9, 7.0)],
            ),
        },
        {
            "id": "two_dense_blocks_with_uniform_noise",
            "node_count": 8,
            "undirected_edges": _normalize_edges(
                8,
                _clique([0, 1, 2, 3], 5.0)
                + _clique([4, 5, 6, 7], 5.0)
                + [(left, right, 0.01) for left in [0, 1, 2, 3] for right in [4, 5, 6, 7]],
            ),
        },
        {
            "id": "two_pentads_with_sparse_bridges",
            "node_count": 10,
            "undirected_edges": _normalize_edges(
                10,
                _clique([0, 1, 2, 3, 4], 6.0)
                + _clique([5, 6, 7, 8, 9], 6.0)
                + [(4, 5, 0.2), (0, 9, 0.05), (2, 7, 0.05)],
            ),
        },
    ]


def _compute_ground_truth(
    case: dict[str, object],
    *,
    resolution: float,
    seed: int,
    leiden_iterations: int,
) -> dict[str, object]:
    node_count = int(case["node_count"])
    edges = case["undirected_edges"]
    assert isinstance(edges, list)

    graph = _build_graph(node_count, edges)

    louvain_result = graph.community_multilevel(weights="weight", resolution=resolution)
    leiden_result = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=seed,
        n_iterations=leiden_iterations,
    )

    return {
        "id": case["id"],
        "node_count": node_count,
        "undirected_edges": edges,
        "louvain": {
            "partition": list(map(int, louvain_result.membership)),
            "modularity": float(louvain_result.q),
        },
        "leiden": {
            "partition": list(map(int, leiden_result.membership)),
            "modularity": float(leiden_result.quality()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("tests/fixtures/modularity_ground_truth.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Modularity resolution used by both references",
    )
    parser.add_argument(
        "--seed",
        "--leiden-seed",
        dest="seed",
        type=int,
        default=42,
        help="Seed passed to both Louvain/Leiden references",
    )
    parser.add_argument(
        "--leiden-iterations",
        type=int,
        default=-1,
        help="n_iterations passed to leidenalg.find_partition",
    )
    args = parser.parse_args()

    cases = [
        _compute_ground_truth(
            case,
            resolution=args.resolution,
            seed=args.seed,
            leiden_iterations=args.leiden_iterations,
        )
        for case in _graph_cases()
    ]

    payload = {
        "schema_version": 1,
        "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "references": {
            "python": platform.python_version(),
            "python_igraph": igraph.__version__,
            "leidenalg": leidenalg.__version__,
        },
        "parameters": {
            "resolution": args.resolution,
            "seed": args.seed,
            "leiden_iterations": args.leiden_iterations,
        },
        "cases": cases,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
