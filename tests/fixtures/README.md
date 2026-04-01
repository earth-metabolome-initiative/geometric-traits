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
