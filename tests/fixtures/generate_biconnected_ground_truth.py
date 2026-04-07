#!/usr/bin/env python3
"""Generate exact biconnected-component ground truth for small undirected graphs.

The checked-in corpus is the exhaustive set of all simple undirected labeled
graphs on 5 vertices:

    2^(5 choose 2) = 1024 graphs

The oracle is definition-based and intentionally independent of the crate
implementation:

- connected components by graph traversal
- articulation points by vertex deletion
- bridges by edge deletion
- biconnected blocks as maximal induced connected subgraphs whose own induced
  graph has no articulation point, with the K2 dyad convention
"""

from __future__ import annotations

import datetime as dt
import gzip
import itertools
import json
from collections import Counter
from pathlib import Path

ORDER = 5
OUT_PATH = Path(__file__).with_name("biconnected_components_order5_exhaustive.json.gz")
EDGE_ORDER = list(itertools.combinations(range(ORDER), 2))


def canonical_edges(edges):
    return [[left, right] for left, right in sorted(edges)]


def induced_edges(vertices, edges):
    active = set(vertices)
    return {(left, right) for left, right in edges if left in active and right in active}


def connected_components(vertices, edges):
    vertices = sorted(vertices)
    if not vertices:
        return []

    adjacency = {vertex: set() for vertex in vertices}
    for left, right in edges:
        if left in adjacency and right in adjacency:
            adjacency[left].add(right)
            adjacency[right].add(left)

    unseen = set(vertices)
    components = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        component = [start]
        while stack:
            vertex = stack.pop()
            for neighbor in sorted(adjacency[vertex]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)
        components.append(sorted(component))
    components.sort()
    return components


def is_connected(vertices, edges):
    return len(connected_components(vertices, edges)) <= 1


def articulation_points(order, edges):
    baseline = len(connected_components(range(order), edges))
    points = []
    for removed in range(order):
        remainder = [vertex for vertex in range(order) if vertex != removed]
        reduced_edges = {edge for edge in edges if removed not in edge}
        if len(connected_components(remainder, reduced_edges)) > baseline:
            points.append(removed)
    return points


def bridges(order, edges):
    baseline = len(connected_components(range(order), edges))
    cut_edges = []
    for edge in sorted(edges):
        reduced_edges = set(edges)
        reduced_edges.remove(edge)
        if len(connected_components(range(order), reduced_edges)) > baseline:
            cut_edges.append(list(edge))
    return cut_edges


def is_biconnected_block(vertices, graph_edges):
    vertices = tuple(sorted(vertices))
    block_edges = induced_edges(vertices, graph_edges)
    if len(vertices) < 2 or not block_edges:
        return False
    if len(vertices) == 2:
        return block_edges == {vertices}
    if not is_connected(vertices, block_edges):
        return False
    for removed in vertices:
        remainder = [vertex for vertex in vertices if vertex != removed]
        if not is_connected(remainder, induced_edges(remainder, block_edges)):
            return False
    return True


def maximal_biconnected_blocks(order, edges):
    valid_blocks = []
    for size in range(2, order + 1):
        for vertices in itertools.combinations(range(order), size):
            if is_biconnected_block(vertices, edges):
                valid_blocks.append(vertices)

    maximal_blocks = [
        vertices
        for vertices in valid_blocks
        if not any(set(vertices) < set(other) for other in valid_blocks)
    ]

    block_records = []
    for vertices in maximal_blocks:
        block_edges = canonical_edges(induced_edges(vertices, edges))
        block_records.append({"vertices": list(vertices), "edges": block_edges})
    block_records.sort(key=lambda record: (record["edges"], record["vertices"]))
    return block_records


def is_biconnected_graph(order, edges, points):
    return order >= 2 and len(connected_components(range(order), edges)) == 1 and not points


def assert_case_invariants(case):
    graph_edges = {tuple(edge) for edge in case["edges"]}
    edge_blocks = case["edge_biconnected_components"]
    vertex_blocks = case["vertex_biconnected_components"]

    flattened_edges = [tuple(edge) for component in edge_blocks for edge in component]
    assert len(flattened_edges) == len(set(flattened_edges))
    assert set(flattened_edges) == graph_edges

    derived_vertex_blocks = []
    for component in edge_blocks:
        block_vertices = sorted({vertex for edge in component for vertex in edge})
        derived_vertex_blocks.append(block_vertices)
    assert derived_vertex_blocks == vertex_blocks

    memberships = [0] * case["node_count"]
    for block in vertex_blocks:
        for vertex in block:
            memberships[vertex] += 1

    derived_articulation = [vertex for vertex, count in enumerate(memberships) if count > 1]
    assert derived_articulation == case["articulation_points"]

    derived_missing = [vertex for vertex, count in enumerate(memberships) if count == 0]
    assert derived_missing == case["vertices_without_biconnected_component"]

    derived_bridges = [component[0] for component in edge_blocks if len(component) == 1]
    assert derived_bridges == case["bridges"]

    derived_cyclic = [
        index
        for index, component in enumerate(edge_blocks)
        if component and len(component) >= len(vertex_blocks[index])
    ]
    assert derived_cyclic == case["cyclic_biconnected_component_indices"]


def case_from_mask(mask):
    edges = {edge for bit_index, edge in enumerate(EDGE_ORDER) if mask & (1 << bit_index)}
    blocks = maximal_biconnected_blocks(ORDER, edges)
    edge_blocks = [record["edges"] for record in blocks]
    vertex_blocks = [record["vertices"] for record in blocks]
    points = articulation_points(ORDER, edges)
    cut_edges = bridges(ORDER, edges)
    connected = connected_components(range(ORDER), edges)
    covered_vertices = sorted({vertex for block in vertex_blocks for vertex in block})
    cyclic = [
        index
        for index, component in enumerate(edge_blocks)
        if component and len(component) >= len(vertex_blocks[index])
    ]

    case = {
        "name": f"order5_mask_{mask:04d}",
        "family": f"edge_count_{len(edges)}",
        "node_count": ORDER,
        "edges": canonical_edges(edges),
        "connected_components": connected,
        "vertices_without_biconnected_component": [
            vertex for vertex in range(ORDER) if vertex not in covered_vertices
        ],
        "edge_biconnected_components": edge_blocks,
        "vertex_biconnected_components": vertex_blocks,
        "articulation_points": points,
        "bridges": cut_edges,
        "cyclic_biconnected_component_indices": cyclic,
        "is_biconnected": is_biconnected_graph(ORDER, edges, points),
        "notes": (
            "exhaustive labeled simple graph on 5 vertices; "
            f"mask={mask}; edge_count={len(edges)}"
        ),
    }
    assert_case_invariants(case)
    return case


def main():
    cases = [case_from_mask(mask) for mask in range(1 << len(EDGE_ORDER))]
    family_counts = Counter(case["family"] for case in cases)

    fixture = {
        "schema_version": 1,
        "algorithm": "hopcroft_tarjan_biconnected_components",
        "graph_kind": "undirected_simple",
        "generator": "exact_definition_exhaustive_labeled_graphs_order_5",
        "primary_oracle": "edge_biconnected_components",
        "dyad_is_biconnected_component": True,
        "isolated_vertices_form_biconnected_components": False,
        "component_ordering": "lexicographic_by_smallest_normalized_edge",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "edge_order": [list(edge) for edge in EDGE_ORDER],
        "cases": cases,
    }

    raw = json.dumps(fixture, separators=(",", ":")).encode()
    with gzip.open(OUT_PATH, "wb", compresslevel=9) as handle:
        handle.write(raw)

    print(f"wrote {len(cases)} cases to {OUT_PATH}")
    print("family counts:")
    for family in sorted(family_counts):
        print(f"  {family}: {family_counts[family]}")


if __name__ == "__main__":
    main()
