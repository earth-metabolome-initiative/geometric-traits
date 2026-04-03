# Ground-Truth Modularity Fixtures

`modularity_ground_truth.json` is generated from external references:

- Louvain: `python-igraph` (`Graph.community_multilevel`)
- Leiden: `leidenalg` (`find_partition(..., ModularityVertexPartition)`)

Regenerate with:

```bash
UV_CACHE_DIR=/tmp/uv-cache \
uv run --with 'numpy<2' --with python-igraph --with leidenalg \
scripts/generate_modularity_ground_truth.py
```

The fixture contains deterministic graph cases plus expected partitions/modularity
for both algorithms.

# Blossom V Ground-Truth Fixtures

The large Blossom V ground-truth corpora are local-only inputs. They are not
tracked in git.

If you keep a local shard directory for manual sweeps, point
`BLOSSOM_V_GROUND_TRUTH_SOURCE` at it.

# VF2 Ground-Truth Fixtures

`vf2_networkx_fixture_suite.json.gz` is the checked-in `NetworkX` oracle suite
for the generic VF2 core.

It merges the old split VF2 corpora into one gzip-compressed JSON file with:

- large boolean structural cases
- equality-labeled boolean cases
- exact-embedding structural cases
- exact-embedding equality-labeled cases
- explicit monomorphism cases alongside isomorphism and subgraph modes
- directed and undirected coverage
- self-loop coverage
- per-case stored `NetworkX` timing in nanoseconds

This fixture is consumed by `tests/test_vf2_fixture_suite.rs` and
`benches/vf2.rs`.
