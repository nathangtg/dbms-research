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
- `n_probe` (int): Number of zones to search
- `ef_search` (int, optional): Override HNSW beam width
- `use_pq_rerank` (bool): Use PQ for initial candidate filtering

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

Search for k nearest neighbors for multiple queries.

**Parameters:**
- `queries` (np.ndarray): Query vectors of shape (n_queries, d)
- `k` (int): Number of neighbors per query
- `n_probe` (int): Number of zones to search
- `ef_search` (int, optional): Override HNSW beam width
- `n_jobs` (int): Number of parallel workers

**Returns:**
- `ids` (np.ndarray): Neighbor IDs of shape (n_queries, k)
- `distances` (np.ndarray): Distances of shape (n_queries, k)

#### save / load

```python
index.save(filepath: str) -> None

ZGQIndex.load(filepath: str, verbose: bool = False) -> ZGQIndex
```

Persist and restore index to/from disk.

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
