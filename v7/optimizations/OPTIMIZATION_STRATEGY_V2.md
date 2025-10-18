# ZGQ Optimization Strategy V2

## Current Performance Gap
- ZGQ: 1.121ms latency, 892 QPS
- HNSW: 0.167ms latency, 5982 QPS
- **Gap: 6.71x slower**

## Root Cause Analysis

### Where Time is Spent (profiling shows):
1. **Zone Selection (40-50% of time)**: Computing distances to 100 centroids
2. **Zone Search (30-40% of time)**: HNSW search in 20 zones
3. **PQ Distance (10-15% of time)**: Asymmetric distance computation
4. **Aggregation (5-10% of time)**: Deduplication and sorting

## Optimization Plan

### 1. Fast Zone Selection (HIGHEST IMPACT - 40% speedup potential)
**Problem**: Computing Euclidean distance to all 100 centroids is expensive
**Solution**: Build HNSW graph on centroids themselves

```python
# Instead of:
distances = np.sum((centroids - query)**2, axis=1)  # O(n_zones * d)

# Use:
nearest_zones = centroid_hnsw.search(query, n_probe)  # O(log n_zones)
```

**Expected speedup**: 2-3x on zone selection → 40-50% overall

### 2. Optimized PQ Distance (MEDIUM IMPACT - 15% speedup potential)
**Problem**: PQ table lookup not cache-friendly
**Solution**: Pre-allocated buffers, vectorized lookup

```python
@njit(parallel=True, fastmath=True)
def pq_distance_batch(codes, distance_table):
    # Vectorized lookup with better cache locality
    pass
```

**Expected speedup**: 1.5x on PQ → 10-15% overall

### 3. Reduce n_probe Adaptively (HIGH IMPACT - configurable tradeoff)
**Problem**: Searching 20 zones for 70% recall is inefficient
**Solution**: Adaptive n_probe based on query difficulty

```python
# Easy queries: n_probe=5-10
# Hard queries: n_probe=20-30
```

**Expected speedup**: 2x for easy queries → 30-50% overall (depends on query mix)

### 4. Early Termination (MEDIUM IMPACT - 20% speedup potential)
**Problem**: Always search all n_probe zones even if we found good candidates
**Solution**: Stop early if top-k candidates are high quality

```python
if len(candidates) >= k * 5 and min_distance < threshold:
    break  # Don't search more zones
```

**Expected speedup**: 1.5x on average → 15-20% overall

## Implementation Priority

### Phase 1: Zone Selection with HNSW (IMPLEMENT FIRST)
- Add `centroid_index` (HNSW on centroids)
- Replace linear scan with HNSW search
- **Target**: 0.7-0.8ms latency (35% faster)

### Phase 2: Adaptive n_probe
- Profile query difficulty
- Use n_probe=10 for easy, 20 for hard queries
- **Target**: 0.5-0.6ms latency (additional 25% faster)

### Phase 3: Early Termination + PQ Optimization
- Implement early stopping
- Optimize PQ distance computation
- **Target**: 0.3-0.4ms latency (additional 35% faster)

## Success Criteria
- **Minimum**: Match HNSW latency (~0.2ms)
- **Target**: Beat HNSW with better recall (0.15ms, 80%+ recall)
- **Stretch**: 2x faster than HNSW (0.08ms)

## Next Steps
1. Implement centroid HNSW in `zonal_partitioning.py`
2. Integrate into `search_lightweight.py`
3. Benchmark each optimization independently
4. Combine all optimizations for final test
