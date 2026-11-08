# ZGQ v8 API Reference

## Table of Contents

1. [ZGQIndex](#zgqindex)
2. [ZGQConfig](#zgqconfig)
3. [Core Components](#core-components)
4. [Utilities](#utilities)

---

## ZGQIndex

The main interface for building and searching ZGQ indices.

### Constructor

```python
from zgq import ZGQIndex
from zgq.index import ZGQConfig

index = ZGQIndex(config=None)
```

**Parameters:**
- `config` (ZGQConfig, optional): Index configuration. Uses defaults if None.

### Methods

#### build

```python
index.build(vectors: np.ndarray) -> ZGQIndex
```

Build the ZGQ index from input vectors.

**Parameters:**
- `vectors` (np.ndarray): Input vectors of shape (N, d), dtype float32

**Returns:**
- `ZGQIndex`: self for method chaining

**Example:**
```python
vectors = np.random.randn(10000, 128).astype(np.float32)
index = ZGQIndex()
index.build(vectors)
```

#### search

```python
index.search(
    query: np.ndarray,
    k: int = 10,
    n_probe: int = 8,
    ef_search: int = None,
    use_pq_rerank: bool = True
) -> Tuple[np.ndarray, np.ndarray]
```

Search for k nearest neighbors to a single query.

**Parameters:**
- `query` (np.ndarray): Query vector of shape (d,)
- `k` (int): Number of neighbors to return
- `n_probe` (int): Number of zones to probe, nearest centroids first (ignored when
  `adaptive_probe=True`; the graph path retrieves up to `k * n_probe` candidates)
- `ef_search` (int, optional): Override HNSW beam width (stays set on the graph)
- `use_pq_rerank` (bool): Use PQ for initial candidate filtering. When there are
  more than `k` candidates, the best `2k` by PQ distance (to the nearest probed zone's
  centroid) are kept and then reranked exactly

**Returns:**
- `ids` (np.ndarray): Neighbor IDs of shape (k,)
- `distances` (np.ndarray): Distances of shape (k,)

**Example:**
```python
query = np.random.randn(128).astype(np.float32)
ids, distances = index.search(query, k=10)
```

#### batch_search

```python
index.batch_search(
    queries: np.ndarray,
    k: int = 10,
    n_probe: int = 8,
    ef_search: int = None,
    n_jobs: int = 1
) -> Tuple[np.ndarray, np.ndarray]
```

Search for k nearest neighbors for multiple queries. The signature above lists
the common parameters; the full signature also takes `fast_mode=True`,
`rerank_factor=3`, `use_zone_boost=True` and `turbo_mode=False`.

**Parameters:**
- `queries` (np.ndarray): Query vectors of shape (n_queries, d)
- `k` (int): Number of neighbors per query
- `n_probe` (int): Number of zones to search. Only used by the standard path
  (`fast_mode=False`); `fast_mode` and `turbo_mode` ignore it
- `ef_search` (int, optional): Override HNSW beam width
- `n_jobs` (int): Number of parallel workers; only used by the standard path
- `fast_mode` (bool): Query the hnswlib graph directly for `k * rerank_factor`
  candidates (beam width `max(ef_search, k * rerank_factor)`), then rerank
  them exactly with squared L2. Ignores `search_mode`, `adaptive_probe` and zone replicas
- `rerank_factor` (int): Candidate multiplier for the `fast_mode` rerank
- `turbo_mode` (bool): Skip the rerank and search the graph with beam width
  `int(ef_search * 1.1)`; takes precedence over `fast_mode`
- `use_zone_boost` (bool): Accepted for compatibility; has no effect

`compressed_search=True` overrides all of the above and uses the compressed path.
Distances are squared L2 for `metric='l2'`. Queries are cast to float32 and must have
the index dimension (`ValueError` otherwise). Calling before `build` raises `RuntimeError`.

**Returns:**
- `ids` (np.ndarray): Neighbor IDs of shape (n_queries, k)
- `distances` (np.ndarray): Distances of shape (n_queries, k)

#### save / load

```python
index.save(filepath: str) -> None

ZGQIndex.load(filepath: str, verbose: bool = False) -> ZGQIndex
```

Persist and restore index to/from disk (pickle). `save` raises `RuntimeError` if the
index is not built or was built with `drop_full_vectors=True`. Compressed-mode structures
and `compressed_search`, `rerank_budget`, `drop_full_vectors` and `compressed_entry_zones`
are not stored, so a loaded index uses the default search path.

**Example:**
```python
# Save
index.save('my_index.zgq')

# Load
loaded_index = ZGQIndex.load('my_index.zgq')
```

#### get_stats

```python
index.get_stats() -> Dict
```

Get index statistics including build times and configuration.

`index.build_stats` holds `zones_time`, `graph_time`, `pq_time`, `norms_time`, `total_time`
(seconds) and `compressed_time` and `replica_overhead` when those options are on.

---

## ZGQConfig

Configuration dataclass for ZGQIndex.

```python
from zgq.index import ZGQConfig

config = ZGQConfig(
    # Zone settings
    n_zones='auto',           # 'auto' or int
    use_hierarchy=True,       # Enable hierarchical zones
    
    # Graph settings
    M=16,                     # HNSW max connections
    ef_construction=200,      # Build beam width
    ef_search=64,             # Search beam width
    
    # Quantization settings
    use_pq=True,              # Enable Product Quantization
    pq_m='auto',              # PQ subspaces ('auto' or int)
    pq_bits=8,                # Bits per PQ code
    use_residual_pq=True,     # Use residual encoding
    
    # Performance settings
    use_simd=True,            # SIMD acceleration
    use_zone_guidance=True,   # Zone-guided search
    
    # General settings
    metric='l2',              # Distance metric
    verbose=False,            # Print progress
    random_state=42           # Random seed
)
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_zones` | str/int | 'auto' | Number of zones or 'auto' |
| `use_hierarchy` | bool | True | Use hierarchical zones |
| `M` | int | 16 | HNSW connections per node |
| `ef_construction` | int | 200 | Build-time beam width |
| `ef_search` | int | 64 | Search-time beam width |
| `use_pq` | bool | True | Enable Product Quantization |
| `pq_m` | str/int | 'auto' | PQ subspaces |
| `pq_bits` | int | 8 | Bits per PQ centroid |
| `use_residual_pq` | bool | True | Residual encoding |
| `metric` | str | 'l2' | 'l2', 'cosine', or 'ip' |
| `verbose` | bool | False | Print progress |
| `use_simd`, `use_zone_guidance` | bool | True | SIMD distances; zone-guided graph search |
| `random_state` | int | 42 | Seed for zones, PQ and graph construction |

`build` adjusts `ef_search` in place: capped at 32 below 10,000 vectors and raised to at
least 100 above 1,000,000. For dimension above 256 it raises `M` to at least 24.

#### Opt-in recall options

All are off by default and leave default results unchanged. They affect the zone
path (`search`, `batch_zone_search`), except `insertion_order`, which changes the
graph that `batch_search(fast_mode=True)` also uses. Measured (synthetic data only, see
`docs/BENCHMARKS.md`): `insertion_order='random'` closes the recall gap to HNSW-M16 at 100K and 1M
(within 0.8 pt) at 1.0x to 1.1x latency and about 1.8x HNSW-M16 analytic bytes; `zone_replicas=2`,
`adaptive_probe` and `search_mode='global_entry'` raise zone-path recall but do not beat HNSW-M16
on the recall/latency trade-off.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `insertion_order` | str | 'zone' | HNSW insertion order: 'zone' (zone by zone), 'random' (seeded permutation), 'lid_desc' / 'lid_asc' (by Levina-Bickel local intrinsic dimension, k=20), 'skeleton' (random sample first, then zone order) or 'roundrobin' (one vector per zone in turn). See `zgq/core/build_order.py` |
| `skeleton_fraction` | float | 0.05 | Share of vectors inserted first in random order when `insertion_order='skeleton'`; must be in [0, 1] |
| `zone_replicas` | int | 1 | Zones each vector belongs to for zone scans (2 = SPANN-style boundary replication). Extra cost is `n_vectors * (zone_replicas - 1)` int32 ids, reported by `get_stats()['replica_overhead']` |
| `adaptive_probe` | bool | False | Pick the probe count per query: zones whose centroid distance is at most `probe_ratio` times the nearest one, clamped to `[min_probe, max_probe]` (`n_probe` is ignored) |
| `min_probe`, `max_probe` | int | 1, 32 | Adaptive probe bounds |
| `probe_ratio` | float | 1.3 | Adaptive distance ratio threshold (>= 1) |
| `search_mode` | str | 'graph' | 'graph': global graph search with zone re-ranking. 'zone_scan': exact candidates are all members (and replicas) of the probed zones. 'global_entry': graph candidates plus the entry points of the probed zones. hnswlib cannot take custom entry points, so this seeds the candidate set only and does not change hops. The mode applies to `search` and `batch_zone_search` only, not to `batch_search(fast_mode=True)` |

#### Compressed search (opt-in)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `compressed_search` | bool | False | `search`/`batch_search` walk the level-0 links of the graph and score candidates with PQ/RPQ ADC computed against each candidate's own zone centroid (entry points: nearest `compressed_entry_zones` zones). Needs `use_pq=True` |
| `rerank_budget` | int | 0 | Exact-rerank this many top ADC candidates (squared L2 against the raw vectors); 0 = codes only. The beam width is raised to at least this value |
| `drop_full_vectors` | bool | False | After build, free the hnswlib index and norms; raw vectors are kept only if `rerank_budget > 0`. The index cannot be saved and has no non-compressed search path |
| `compressed_entry_zones` | int | 4 | Zones whose entry point seeds the search |

`index.compressed_memory_bytes()` returns the actual bytes of the stored compressed arrays;
`zgq.memory_model` has the analytic size functions for ZGQ, hnswlib and FAISS indexes.
The compressed fields are not persisted by `save`/`load`. Measured results:
`results/2026-11-08-improvements/compression/README.md` (ZGQ compressed loses at equal memory).

`batch_zone_search(queries, k, n_probe, ef_search, use_pq_rerank, n_jobs)` is the
batch form of `search`. It sets `index.last_batch_stats` (`mean_probes`, `probes`).
Note `use_pq_rerank=True` (default) applies the PQ shortlist, which lowered recall
as `n_probe` grew in `results/2026-11-08-improvements/diagnostics`.

---

## Core Components

### AdaptiveHierarchicalZones

```python
from zgq.core.zones import AdaptiveHierarchicalZones, ZoneConfig

config = ZoneConfig(auto_zones=True, verbose=False)
zones = AdaptiveHierarchicalZones(config)
zones.build(vectors)

# Select zones for a query
selected = zones.select_zones(query, n_probe=8)
```

### ZoneGuidedGraph

```python
from zgq.core.graph import ZoneGuidedGraph, GraphConfig

config = GraphConfig(M=16, ef_construction=200)
graph = ZoneGuidedGraph(config)
graph.build(vectors, zone_assignments, zone_centroids)

# Search
ids, distances = graph.search(query, k=10, selected_zones)
```

### ResidualProductQuantizer

```python
from zgq.core.quantization import ResidualProductQuantizer, RPQConfig

config = RPQConfig(m=16, n_bits=8, use_residuals=True)
pq = ResidualProductQuantizer(config)
pq.train(vectors, zone_centroids, zone_assignments)

# Encode
codes = pq.encode(vectors, zone_centroids, zone_assignments)

# Compute distances
dist_table = pq.compute_distance_table(query, zone_centroid)
distances = pq.asymmetric_distance(codes, dist_table)
```

### DistanceComputer

```python
from zgq.core.distances import DistanceComputer, SIMDDistance

computer = DistanceComputer(metric='l2')
distances = computer.compute(query, vectors)

# With precomputed norms
norms = DistanceComputer.precompute_norms(vectors)
distances = computer.compute(query, vectors, norms)

# SIMD-optimized
distances = SIMDDistance.euclidean_squared_batch(query, vectors)
```

---

## Utilities

### Metrics

```python
from zgq.utils.metrics import compute_recall, compute_metrics

# Compute recall
recall = compute_recall(predicted_ids, ground_truth_ids, k=10)

# Comprehensive metrics
metrics = compute_metrics(
    predicted_ids,
    ground_truth_ids,
    k_values=[1, 5, 10, 20],
    latencies=query_latencies
)
```

Units: `compute_recall` returns a percentage (0-100); `compute_metrics` also returns
`mrr` and `ndcg@10` (0-1), and with `latencies` in seconds, `latency_*_ms` and
`throughput_qps` (queries per second).

`ZGQSearch(zones, graph, vectors, pq, pq_codes, vector_norms, config=SearchConfig())`
in `zgq.search` runs the zone/graph/PQ/exact-rerank pipeline on already built components
(`search`, `search_fast`, `search_accurate`, `batch_search`).

### Ground Truth

```python
from zgq.search import compute_ground_truth

gt_ids, gt_distances = compute_ground_truth(vectors, queries, k=10)
```

### I/O

```python
from zgq.utils.io import save_benchmark_results, load_benchmark_results

save_benchmark_results(results, 'results.json')
results = load_benchmark_results('results.json')
```

---

## Best Practices

### Parameter Tuning

1. **For high recall:** Increase `ef_search` and `n_probe`
2. **For low latency:** Decrease `ef_search`, use fewer zones
3. **For memory efficiency:** Enable PQ with appropriate `pq_m`

### Recommended Configurations

**Speed-optimized:**
```python
config = ZGQConfig(
    ef_search=32,
    use_pq=False,
    use_hierarchy=True
)
```

**Recall-optimized:**
```python
config = ZGQConfig(
    ef_search=128,
    use_pq=True,
    n_probe=16
)
```

**Memory-optimized:**
```python
config = ZGQConfig(
    use_pq=True,
    pq_m=32,
    pq_bits=8
)
```
