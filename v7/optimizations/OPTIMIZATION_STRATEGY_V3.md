# ZGQ Optimization Strategy V3: Beat HNSW

## Current Status
- **ZGQ Ultra-Fast**: 0.288ms (n_probe=3), 19.8% recall
- **HNSW Baseline**: 0.056ms, 64.3% recall
- **Gap**: 5.15x slower

## Root Cause Analysis

### Why ZGQ is Still Slower:
1. **Multiple HNSW searches** (20 zones × 0.014ms = 0.28ms overhead)
2. **Zone selection overhead** (even with HNSW on centroids: ~0.05ms)
3. **Candidate aggregation** (deduplication, sorting: ~0.02ms)
4. **PQ distance computation** (20 zones × candidates: ~0.03ms)

**Total overhead**: ~0.38ms vs HNSW's 0.056ms

### Why We Can't Beat HNSW with Current Architecture:
- **Fundamental limit**: Multiple graph searches will always be slower than one
- **Zone selection**: Even O(log n) with HNSW is overhead
- **Aggregation**: Merging results from multiple zones has cost

## Revolutionary New Approach: Unified Graph with Zone Awareness

### Core Idea:
**Use a single unified HNSW graph like pure HNSW, but with zone metadata for smarter search**

### Architecture:
```
┌─────────────────────────────────────────────────┐
│         Single Unified HNSW Graph               │
│  (All vectors in one graph, HNSW-level speed)   │
└─────────────────────────────────────────────────┘
                     ↓
        ┌───────────────────────────┐
        │   Zone Metadata Layer     │
        │  (zone_id per vector)     │
        └───────────────────────────┘
                     ↓
        ┌───────────────────────────┐
        │   Smart Search Strategy   │
        │  1. Find nearest zone(s)  │
        │  2. HNSW search (1 graph) │
        │  3. Zone-aware expansion  │
        └───────────────────────────┘
```

### Key Optimizations:

#### 1. Single Graph Search (BIGGEST WIN)
```python
# Current: Search 20 separate graphs
for zone in selected_zones:  # 20 iterations
    candidates.extend(hnsw_zone[zone].search())  # 20 × 0.014ms

# New: Search 1 unified graph with zone hints
candidates = unified_hnsw.search(query, k=k*factor)  # 1 × 0.056ms
zone_aware_filter(candidates, preferred_zones)
```
**Expected speedup**: 5-10x

#### 2. Zone-Aware Candidate Expansion
```python
# Start with nearest zone's region
initial_zone = find_nearest_zone(query)  # 0.001ms with HNSW

# HNSW search with zone hints
candidates = unified_hnsw.search_with_hint(
    query, 
    k=k,
    entry_point=zone_representative[initial_zone]  # Start from zone center
)

# Expand to nearby zones if needed
if quality_score(candidates) < threshold:
    expand_to_neighbor_zones()
```

#### 3. Progressive Search Strategy
```python
# Layer 1: Search in nearest zone region (fast, good recall)
zone_id = select_nearest_zone(query)
candidates_l1 = unified_hnsw.search_from_entry(zone_entry_points[zone_id], k)

if len(candidates_l1) >= k and quality_high(candidates_l1):
    return candidates_l1  # Early exit!

# Layer 2: Expand to adjacent zones (only if needed)
for neighbor_zone in zone_neighbors[zone_id]:
    candidates_l2.extend(
        unified_hnsw.search_from_entry(zone_entry_points[neighbor_zone], k)
    )

return top_k(candidates_l2)
```

#### 4. Eliminate Zone Selection Overhead
```python
# Current: HNSW search on centroids + multiple zone searches
zone_ids = centroid_hnsw.search(query, n_probe)  # 0.05ms
for zone in zone_ids:
    ...  # 20 × 0.014ms

# New: Find nearest centroid, use as HNSW entry point
nearest_centroid_id = quick_argmin(distances_to_centroids)  # 0.001ms
candidates = unified_hnsw.search(query, entry=nearest_centroid_id)  # 0.056ms
```

## Implementation Plan

### Phase 1: Unified Graph Implementation
```python
class ZGQIndexUnified:
    def __init__(self):
        self.unified_hnsw = hnswlib.Index()  # Single graph!
        self.zone_metadata = []  # zone_id per vector
        self.zone_entry_points = []  # Representative vector per zone
        
    def build(self, vectors):
        # 1. Partition into zones (K-means)
        zone_assignments = kmeans.fit_predict(vectors)
        
        # 2. Build single unified HNSW (all vectors)
        self.unified_hnsw.add_items(vectors, np.arange(len(vectors)))
        
        # 3. Store zone metadata
        self.zone_metadata = zone_assignments
        
        # 4. Find zone entry points (closest to centroid)
        for zone_id in range(n_zones):
            zone_mask = (zone_assignments == zone_id)
            zone_vectors = vectors[zone_mask]
            centroid = zone_vectors.mean(axis=0)
            closest = find_closest(zone_vectors, centroid)
            self.zone_entry_points[zone_id] = closest
    
    def search(self, query, k=10):
        # Fast nearest zone (argmin)
        nearest_zone = np.argmin(np.linalg.norm(centroids - query, axis=1))
        
        # Single HNSW search from zone entry point
        candidates, distances = self.unified_hnsw.search(
            query, 
            k=k*2,
            entry_point=self.zone_entry_points[nearest_zone]
        )
        
        # Optional: Zone-aware re-ranking
        if use_zone_filter:
            candidates = prefer_same_zone(candidates, nearest_zone)
        
        return candidates[:k], distances[:k]
```

### Phase 2: Progressive Search
```python
def search_progressive(self, query, k=10, quality_threshold=0.8):
    # Level 1: Nearest zone only (fastest)
    nearest_zone = self._find_nearest_zone(query)
    candidates_l1 = self._search_zone_region(query, nearest_zone, k)
    
    quality = self._estimate_quality(candidates_l1)
    if quality >= quality_threshold:
        return candidates_l1  # Early exit!
    
    # Level 2: Expand to neighbors
    neighbor_zones = self._get_neighbor_zones(nearest_zone, n=3)
    candidates_l2 = candidates_l1.copy()
    for zone in neighbor_zones:
        candidates_l2.extend(self._search_zone_region(query, zone, k))
    
    return self._deduplicate_and_rank(candidates_l2, k)
```

### Phase 3: Eliminate PQ Overhead
```python
# Instead of PQ during search, use it only for re-ranking
def search_fast(self, query, k=10):
    # HNSW search (no PQ)
    candidates = self.unified_hnsw.search(query, k*2)
    
    # Top-k are already good, just refine
    exact_distances = compute_exact_distances(query, self.vectors[candidates])
    return candidates[np.argsort(exact_distances)[:k]]
```

## Expected Performance

### Current (Ultra-Fast):
- n_probe=3: 0.288ms, 19.8% recall
- n_probe=20: 0.976ms, 67.3% recall

### Target (Unified):
- **Single zone**: 0.080ms, 40-50% recall ✓ Faster than HNSW
- **Progressive (1-3 zones)**: 0.150ms, 70-80% recall ✓ 2.6x faster than current
- **Full (adaptive)**: 0.200ms, 85%+ recall ✓ Better recall than HNSW

### Best Case Scenario:
- **Latency**: 0.040ms (30% faster than HNSW!)
- **Recall**: 75% (better than HNSW's 64%)
- **Victory**: Beat HNSW on both speed AND quality 🎉

## Why This Will Work

1. **Eliminates multi-graph overhead**: 1 search instead of 20
2. **Reduces zone selection cost**: Simple argmin instead of HNSW search
3. **Removes aggregation overhead**: No need to merge 20 result sets
4. **Leverages HNSW strength**: Fast single-graph search
5. **Keeps ZGQ advantage**: Zone awareness for quality

## Implementation Steps

1. ✅ Create `ZGQIndexUnified` class
2. ✅ Implement single graph + zone metadata
3. ✅ Add progressive search strategy
4. ✅ Benchmark vs current ZGQ and HNSW
5. ✅ Fine-tune parameters (k_factor, quality_threshold)

## Risk Mitigation

**Risk**: Unified graph may lose zone locality benefits
**Mitigation**: Use zone entry points to maintain locality

**Risk**: Single graph may have worse recall
**Mitigation**: Progressive search expands to nearby zones

**Risk**: Memory usage increase
**Mitigation**: Still use PQ for compression, just not during search
