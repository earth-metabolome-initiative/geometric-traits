#!/usr/bin/env python3
"""Generate Floyd-Warshall ground truth using NetworkX.

Run from this directory:
    python3 generate_floyd_warshall_ground_truth.py

Produces floyd_warshall_ground_truth.json.gz (gzip-compressed).
"""

import datetime
import gzip
import json
import math
import random

import networkx as nx

random.seed(42)
cases = []


def weighted_edges(graph):
    """Return sorted directed weighted edges for the Rust fixture."""
    if graph.is_directed():
        return sorted(
            (int(u), int(v), float(data.get("weight", 1.0)))
            for u, v, data in graph.edges(data=True)
        )

    edges = []
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1.0))
        edges.append((int(u), int(v), weight))
        if u != v:
            edges.append((int(v), int(u), weight))
    return sorted(edges)


def flat_distances(graph):
    """Return row-major shortest-path distances, using None for unreachable."""
    distances = nx.floyd_warshall(graph, weight="weight")
    nodes = list(range(graph.number_of_nodes()))
    flat = []
    for source in nodes:
        row = distances[source]
        for destination in nodes:
            value = row[destination]
            flat.append(None if math.isinf(value) else float(value))
    return flat


def add_case(graph):
    """Normalize labels, compute distances, and append the case."""
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering="sorted")
    cases.append(
        {
            "n": graph.number_of_nodes(),
            "edges": weighted_edges(graph),
            "distances": flat_distances(graph),
        }
    )


# ============================================================================
# Structured directed graphs
# ============================================================================
add_case(nx.DiGraph())

for n in range(1, 8):
    path = nx.DiGraph()
    path.add_nodes_from(range(n))
    path.add_weighted_edges_from((i, i + 1, float((i % 3) + 1)) for i in range(n - 1))
    add_case(path)

    cycle = nx.DiGraph()
    cycle.add_nodes_from(range(n))
    if n >= 2:
        cycle.add_weighted_edges_from(
            (i, (i + 1) % n, float((2 * i + 1) % 5)) for i in range(n)
        )
    add_case(cycle)

    complete = nx.complete_graph(n, create_using=nx.DiGraph())
    for u, v in complete.edges():
        if u == v:
            continue
        complete[u][v]["weight"] = float(((u + 1) * (v + 2)) % 7)
    add_case(complete)

    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if (i + j) % 2 == 0:
                dag.add_edge(i, j, weight=float(((j - i) % 5) - 2))
    add_case(dag)


# ============================================================================
# Structured undirected graphs
# ============================================================================
for n in range(2, 9):
    path = nx.path_graph(n)
    for u, v in path.edges():
        path[u][v]["weight"] = float((u + v) % 4)
    add_case(path)

    cycle = nx.cycle_graph(n)
    for u, v in cycle.edges():
        cycle[u][v]["weight"] = float(((u + 1) * (v + 1)) % 6)
    add_case(cycle)

    complete = nx.complete_graph(n)
    for u, v in complete.edges():
        complete[u][v]["weight"] = float(((u + 2) * (v + 3)) % 8)
    add_case(complete)

    star = nx.star_graph(n - 1)
    for u, v in star.edges():
        star[u][v]["weight"] = float((u + v + 1) % 5)
    add_case(star)


# ============================================================================
# Random directed graphs with non-negative weights
# ============================================================================
for _ in range(1200):
    n = random.randint(2, 12)
    p = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.65])
    graph = nx.gnp_random_graph(n, p, seed=random.randint(0, 2**31), directed=True)
    for u, v in graph.edges():
        graph[u][v]["weight"] = float(random.choice([0, 1, 2, 4, 7, 11]))
    add_case(graph)


# ============================================================================
# Random DAGs with signed weights but no negative cycles by construction
# ============================================================================
for _ in range(600):
    n = random.randint(2, 14)
    p = random.choice([0.15, 0.25, 0.35, 0.5])
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                graph.add_edge(u, v, weight=float(random.choice([-4, -2, -1, 0, 1, 2, 5, 9])))
    add_case(graph)


# ============================================================================
# Random undirected graphs with non-negative weights
# ============================================================================
for _ in range(700):
    n = random.randint(2, 12)
    p = random.choice([0.1, 0.2, 0.3, 0.45, 0.6, 0.8])
    graph = nx.gnp_random_graph(n, p, seed=random.randint(0, 2**31), directed=False)
    for u, v in graph.edges():
        graph[u][v]["weight"] = float(random.choice([0, 1, 3, 5, 8, 13]))
    add_case(graph)


fixture = {
    "schema_version": 1,
    "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "references": {"networkx": nx.__version__},
    "cases": cases,
}
raw = json.dumps(fixture, separators=(",", ":")).encode()
out_path = "floyd_warshall_ground_truth.json.gz"
with gzip.open(out_path, "wb", compresslevel=9) as handle:
    handle.write(raw)

print(f"Wrote {len(cases)} cases to {out_path} ({len(raw)} bytes uncompressed)")
