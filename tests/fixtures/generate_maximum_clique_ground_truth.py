#!/usr/bin/env python3
"""Generate maximum clique ground truth using NetworkX.

Run from this directory:
    python3 generate_maximum_clique_ground_truth.py

Produces maximum_clique_ground_truth.json.gz (gzip-compressed).

The fixture includes:
  - Random Erdos-Renyi graphs (various n, p)
  - Structured graphs: Turan, complete multipartite, circulant,
    complements of sparse graphs, line graphs, modular products
  - Graphs engineered to have many maximum cliques
"""

import collections
import datetime
import gzip
import itertools
import json
import random
import sys

import networkx as nx

random.seed(42)
cases = []
skipped = 0


def add_case(G):
    """Compute max cliques for graph G and append to cases."""
    global skipped
    n = G.number_of_nodes()
    G = nx.convert_node_labels_to_integers(G, first_label=0)
    edges = sorted([min(u, v), max(u, v)] for u, v in G.edges())
    maximal = list(nx.find_cliques(G))
    omega = max((len(c) for c in maximal), default=0)
    if omega == 0:
        omega = 1 if n > 0 else 0
        max_cliques = [[v] for v in range(n)] if n > 0 else [[]]
    else:
        max_cliques = sorted(sorted(c) for c in maximal if len(c) == omega)
    # Cap output size to keep fixture and test runtime manageable
    if len(max_cliques) > 5000:
        skipped += 1
        return
    cases.append({
        "n": n,
        "edges": edges,
        "omega": omega,
        "max_cliques": max_cliques,
    })


# ============================================================================
# 1. Random Erdos-Renyi graphs (20000 cases, n up to 20)
# ============================================================================
for _ in range(20000):
    n = random.randint(2, 20)
    p = random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    G = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 2**31))
    add_case(G)

# ============================================================================
# 2. Turan graphs T(n,r): known omega = ceil(n/r), many max cliques
# ============================================================================
for n in range(4, 20):
    for r in range(2, min(n, 7)):
        G = nx.turan_graph(n, r)
        add_case(G)

# ============================================================================
# 3. Complete multipartite graphs: many maximum cliques
# ============================================================================
for _ in range(1500):
    k = random.randint(2, 6)
    sizes = [random.randint(1, 4) for _ in range(k)]
    if sum(sizes) <= 24:
        G = nx.complete_multipartite_graph(*sizes)
        add_case(G)

# ============================================================================
# 4. Complements of sparse graphs: dense graphs with interesting structure
# ============================================================================
for _ in range(2000):
    n = random.randint(5, 22)
    p = random.choice([0.05, 0.1, 0.15, 0.2, 0.25])
    G = nx.complement(nx.erdos_renyi_graph(n, p, seed=random.randint(0, 2**31)))
    add_case(G)

# ============================================================================
# 5. Cycle complements
# ============================================================================
for n in range(5, 25):
    G = nx.complement(nx.cycle_graph(n))
    add_case(G)

# ============================================================================
# 6. Kneser graphs K(n,k): vertex = k-subset of [n], edge iff disjoint
# ============================================================================
for n in range(5, 10):
    for k in range(1, n // 2 + 1):
        G = nx.kneser_graph(n, k)
        if G.number_of_nodes() <= 126:
            add_case(G)

# ============================================================================
# 7. Line graphs of small graphs: directly relevant to MCES
# ============================================================================
for _ in range(2500):
    n = random.randint(4, 14)
    p = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    base = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 2**31))
    if 0 < base.number_of_edges() <= 40:
        G = nx.line_graph(base)
        add_case(G)

# ============================================================================
# 8. Modular products of small graphs: the actual MCES use case
# ============================================================================
for _ in range(5000):
    n1 = random.randint(3, 8)
    n2 = random.randint(3, 8)
    p1 = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    p2 = random.choice([0.3, 0.4, 0.5, 0.6, 0.7])
    G1 = nx.erdos_renyi_graph(n1, p1, seed=random.randint(0, 2**31))
    G2 = nx.erdos_renyi_graph(n2, p2, seed=random.randint(0, 2**31))
    MP = nx.modular_product(G1, G2)
    if 0 < MP.number_of_nodes() <= 64:
        add_case(MP)

# ============================================================================
# 9. Crown graphs: K_{n,n} minus perfect matching
# ============================================================================
for n in range(3, 14):
    G = nx.complete_bipartite_graph(n, n)
    for i in range(n):
        G.remove_edge(i, n + i)
    add_case(G)

# ============================================================================
# 10. Paley-like circulant construction
# ============================================================================
for _ in range(2000):
    n = random.randint(6, 24)
    s = random.randint(1, n // 2)
    residues = set(random.sample(range(1, n), s))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) in residues or (n - abs(i - j)) in residues:
                G.add_edge(i, j)
    add_case(G)

# ============================================================================
# 11. Disjoint cliques of the same size (many max cliques by construction)
# ============================================================================
for _ in range(2000):
    k = random.randint(2, 6)
    c = random.randint(2, 10)
    if k * c <= 40:
        G = nx.Graph()
        n = c * k
        G.add_nodes_from(range(n))
        for i in range(c):
            base_v = i * k
            for u, v in itertools.combinations(range(base_v, base_v + k), 2):
                G.add_edge(u, v)
        for _ in range(random.randint(0, n)):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u != v and u // k != v // k:
                G.add_edge(u, v)
        add_case(G)

# ============================================================================
# 12. Dense random graphs
# ============================================================================
for _ in range(2000):
    n = random.randint(6, 18)
    p = random.choice([0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    G = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 2**31))
    add_case(G)

# ============================================================================
# 13. Wheel graphs and fan graphs
# ============================================================================
for n in range(4, 30):
    add_case(nx.wheel_graph(n))
for n in range(4, 25):
    G = nx.path_graph(n)
    G.add_edges_from((n, i) for i in range(n))
    add_case(G)

# ============================================================================
# 14. Friendship graphs (windmill): k triangles sharing a common vertex
# ============================================================================
for k in range(2, 25):
    G = nx.Graph()
    G.add_node(0)
    for i in range(k):
        u = 2 * i + 1
        v = 2 * i + 2
        G.add_edges_from([(0, u), (0, v), (u, v)])
    add_case(G)

# ============================================================================
# 15. Mycielski graphs: triangle-free with high chromatic number
# ============================================================================
for i in range(2, 7):
    G = nx.mycielski_graph(i)
    add_case(G)

# ============================================================================
# 16. Random regular graphs
# ============================================================================
for _ in range(1500):
    n = random.randint(6, 22)
    d = random.randint(2, n - 1)
    if (n * d) % 2 == 0:
        try:
            G = nx.random_regular_graph(d, n, seed=random.randint(0, 2**31))
            add_case(G)
        except nx.NetworkXError:
            pass

# ============================================================================
# 17. Barbell graphs: two K_n connected by a path
# ============================================================================
for n in range(3, 14):
    for path_len in range(0, 5):
        G = nx.barbell_graph(n, path_len)
        add_case(G)

# ============================================================================
# 18. Complements of path and star graphs (dense, structured)
# ============================================================================
for n in range(5, 25):
    add_case(nx.complement(nx.path_graph(n)))
    add_case(nx.complement(nx.star_graph(n)))

# ============================================================================
# 19. Generalized Petersen graphs
# ============================================================================
for n in range(5, 16):
    for k in range(1, n // 2):
        try:
            G = nx.Graph()
            for i in range(n):
                G.add_edge(f"o{i}", f"o{(i+1) % n}")
                G.add_edge(f"o{i}", f"i{i}")
                G.add_edge(f"i{i}", f"i{(i+k) % n}")
            add_case(G)
        except Exception:
            pass

# ============================================================================
# 20. Random bipartite + edges within parts
# ============================================================================
for _ in range(1500):
    n1 = random.randint(3, 10)
    n2 = random.randint(3, 10)
    p_between = random.choice([0.3, 0.5, 0.7])
    p_within = random.choice([0.0, 0.2, 0.4, 0.6])
    G = nx.Graph()
    G.add_nodes_from(range(n1 + n2))
    for u in range(n1):
        for v in range(n1, n1 + n2):
            if random.random() < p_between:
                G.add_edge(u, v)
    for u in range(n1):
        for v in range(u + 1, n1):
            if random.random() < p_within:
                G.add_edge(u, v)
    for u in range(n1, n1 + n2):
        for v in range(u + 1, n1 + n2):
            if random.random() < p_within:
                G.add_edge(u, v)
    add_case(G)

# ============================================================================
# 21. Watts-Strogatz small-world graphs
# ============================================================================
for _ in range(1500):
    n = random.randint(6, 20)
    k = random.choice([2, 4, 6])
    if k < n:
        p = random.choice([0.0, 0.1, 0.3, 0.5, 0.8])
        G = nx.watts_strogatz_graph(n, k, p, seed=random.randint(0, 2**31))
        add_case(G)

# ============================================================================
# 22. Barabasi-Albert preferential attachment
# ============================================================================
for _ in range(1500):
    n = random.randint(6, 22)
    m = random.randint(1, min(5, n - 1))
    G = nx.barabasi_albert_graph(n, m, seed=random.randint(0, 2**31))
    add_case(G)

# ============================================================================
# 23. Random trees (omega always 2 if n>=2, but many K2 cliques)
# ============================================================================
for _ in range(500):
    n = random.randint(3, 20)
    G = nx.random_tree(n, seed=random.randint(0, 2**31))
    add_case(G)

# ============================================================================
# 24. Grid and torus graphs
# ============================================================================
for r in range(2, 8):
    for c in range(2, 8):
        add_case(nx.grid_2d_graph(r, c))
        add_case(nx.grid_2d_graph(r, c, periodic=True))

# ============================================================================
# 25. Complement of random regular graphs (very dense)
# ============================================================================
for _ in range(1000):
    n = random.randint(8, 20)
    d = random.randint(2, min(6, n - 1))
    if (n * d) % 2 == 0:
        try:
            G = nx.complement(
                nx.random_regular_graph(d, n, seed=random.randint(0, 2**31))
            )
            add_case(G)
        except nx.NetworkXError:
            pass


# ============================================================================
# Done -- write output
# ============================================================================
fixture = {
    "schema_version": 1,
    "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    "references": {"networkx": nx.__version__},
    "cases": cases,
}
raw = json.dumps(fixture, separators=(",", ":")).encode()
out_path = "maximum_clique_ground_truth.json.gz"
with gzip.open(out_path, "wb", compresslevel=9) as f:
    f.write(raw)

# Print statistics
omega_dist = collections.Counter(c["omega"] for c in cases)
count_dist = collections.Counter(len(c["max_cliques"]) for c in cases)
print(f"Wrote {len(cases)} cases to {out_path} ({len(raw)} bytes uncompressed)")
if skipped:
    print(f"Skipped {skipped} cases with > 5000 max cliques")
print(f"\nomega distribution: {dict(sorted(omega_dist.items()))}")
print(f"\nmax cliques count distribution (top 25):")
for k, v in sorted(count_dist.items())[:25]:
    print(f"  {k} cliques: {v} cases")
trivial = sum(1 for c in cases if c["omega"] <= 1)
one_clique = sum(1 for c in cases if len(c["max_cliques"]) == 1)
multi = sum(1 for c in cases if len(c["max_cliques"]) >= 2)
many10 = sum(1 for c in cases if len(c["max_cliques"]) >= 10)
many50 = sum(1 for c in cases if len(c["max_cliques"]) >= 50)
many100 = sum(1 for c in cases if len(c["max_cliques"]) >= 100)
max_count = max(len(c["max_cliques"]) for c in cases)
max_omega = max(c["omega"] for c in cases)
print(f"\nTotal: {len(cases)}")
print(f"Trivial (omega<=1): {trivial} ({100*trivial/len(cases):.1f}%)")
print(f"Single max clique: {one_clique} ({100*one_clique/len(cases):.1f}%)")
print(f">= 2 max cliques: {multi} ({100*multi/len(cases):.1f}%)")
print(f">= 10 max cliques: {many10} ({100*many10/len(cases):.1f}%)")
print(f">= 50 max cliques: {many50} ({100*many50/len(cases):.1f}%)")
print(f">= 100 max cliques: {many100} ({100*many100/len(cases):.1f}%)")
print(f"Max cliques in any case: {max_count}")
print(f"Max omega: {max_omega}")
