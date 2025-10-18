# ZGQ Architecture Overview

## 1. Complete ZGQ Implementation

**Class:** `ZGQ_Index`

### 1.1 Data Structures
- centroids: "Matrix[Z, d] - zone centroids"
- inverted_lists: "List[List[int]] - vectors per zone"
- zone_graphs: "List[HNSWGraph] - HNSW graph per zone"
- pq_codebooks: "List[Matrix] - m codebooks, each (k, d/m)"
- pq_codes: "Matrix[N, m] - quantized vectors"
- full_vectors: "Matrix[N, d] - original vectors (optional)"
- vector_norms: "Vector[N] - precomputed ||x||² for each vector"

### 1.2 Parameters
- Z: "Number of zones"
- M: "HNSW max degree"
- ef_construction: "HNSW build parameter"
- ef_search: "HNSW search parameter"
- n_probe: "Zones to search"
- m: "PQ subspaces"
- nbits: "PQ bits per subspace"

### 1.3 Build Method
**Signature:** `build(vectors: Matrix[N, d]) -> None`

#### 1.3.1 Build Steps
**Step 1: Zonal Partitioning**
- Name: "Zonal Partitioning"
- Call: `centroids, assignments, inverted_lists = kmeans_partition(vectors, Z)`
- Complexity: "O(K_iter · N · Z · d)"

**Step 2: Per-Zone HNSW Construction**
- Name: "Per-Zone HNSW Construction"
- Call:
```
for i in range(Z):
    zone_vectors = vectors[inverted_lists[i]]
    zone_graphs[i] = build_zone_graph(zone_vectors, M, ef_construction)
```
- Complexity: "O(N · log(N/Z) · M · d)"
- Parallelization: "Each zone independent"

**Step 3: Product Quantization**
- Name: "Product Quantization"
- Call:
```
pq_codebooks = train_pq(vectors, m, k=2^nbits)
pq_codes = encode_pq(vectors, pq_codebooks, m)
```
- Complexity: "O(K_iter · N · k · d + N · k · d)"

**Step 4: Precompute Norms**
- Name: "Precompute Norms"
- Call:
```
vector_norms = np.sum(vectors ** 2, axis=1)
```
- Complexity: "O(N · d)"

**Total Build Complexity:** "O(N · log(N/Z) · M · d + N · k · d)"

### 1.4 Search Method
**Signature:** `search(query: Vector[d], k: int) -> List[Tuple[int, float]]`

#### 1.4.1 Search Steps
**Step 1: Zone Selection**
- Name: "Zone Selection"
- Call: `selected_zones = select_zones(query, centroids, n_probe)`
- Complexity: "O(Z · d)"

**Step 2: Precompute PQ Distance Table**
- Name: "Precompute PQ Distance Table"
- Call: `distance_table = compute_distance_table(query, pq_codebooks, m, k)`
- Complexity: "O(m · k · d/m) = O(k · d)"

**Step 3: Parallel Zone Search**
- Name: "Parallel Zone Search"
- Call:
```
candidates = parallel_search_all_zones(
    query, selected_zones, zone_graphs, pq_codes,
    distance_table, ef_search, k_local=k
)
```
- Complexity per zone: "O(log(N/Z) · ef_search · m)"
- Total: "O(n_probe · log(N/Z) · ef_search · m)"

**Step 4: Aggregate & Rerank**
- Name: "Aggregate & Rerank"
- Call:
```
results = aggregate_and_rerank(
    query, candidates, full_vectors, k, k_rerank=2*k
)
```
- Complexity: "O(k · d + k log k)"

**Total Search Complexity:**
```
O(Z·d + k·d + n_probe·log(N/Z)·ef_search·m + k·d)
Dominant: O(n_probe·log(N/Z)·ef_search·m) for large datasets
```