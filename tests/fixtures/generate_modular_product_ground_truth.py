#!/usr/bin/env python3
"""Generate modular product ground truth using NetworkX.

Run from this directory:
    python3 generate_modular_product_ground_truth.py

Produces modular_product_ground_truth.json.gz (gzip-compressed).
"""

import datetime
import gzip
import json
import random

import networkx as nx

random.seed(42)
cases = []

for _ in range(10_000):
    n1 = random.randint(2, 7)
    n2 = random.randint(2, 7)
    p1 = random.choice([0.0, 0.3, 0.5, 0.7, 1.0])
    p2 = random.choice([0.0, 0.3, 0.5, 0.7, 1.0])
    e1 = [
        [u, v]
        for u in range(n1)
        for v in range(u + 1, n1)
        if random.random() < p1
    ]
    e2 = [
        [u, v]
        for u in range(n2)
        for v in range(u + 1, n2)
        if random.random() < p2
    ]
    G1, G2 = nx.Graph(), nx.Graph()
    G1.add_nodes_from(range(n1))
    G1.add_edges_from(e1)
    G2.add_nodes_from(range(n2))
    G2.add_edges_from(e2)
    MP = nx.modular_product(G1, G2)
    # Map (u,v) tuples to row-major pair index: i*n2+j
    pair_idx = {(u, v): u * n2 + v for u in range(n1) for v in range(n2)}
    mp_edges = sorted(
        set(
            (min(pair_idx[a], pair_idx[b]), max(pair_idx[a], pair_idx[b]))
            for a, b in MP.edges()
        )
    )
    cases.append({"n1": n1, "n2": n2, "e1": e1, "e2": e2, "mp": mp_edges})

fixture = {
    "schema_version": 1,
    "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "references": {"networkx": nx.__version__},
    "cases": cases,
}
raw = json.dumps(fixture, separators=(",", ":")).encode()
out_path = "modular_product_ground_truth.json.gz"
with gzip.open(out_path, "wb", compresslevel=9) as f:
    f.write(raw)

print(f"Wrote {len(cases)} cases to {out_path} ({len(raw)} -> {__import__('os').path.getsize(out_path)} bytes)")
