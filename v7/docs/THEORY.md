# ZGQ Theoretical Foundation & Research Hypothesis

## 1. Research Hypothesis

**Main Hypothesis:** Combining zonal partitioning with per-zone HNSW graphs and product quantization achieves superior recall-latency trade-offs compared to state-of-the-art ANNS methods, while maintaining efficient memory usage for large-scale deployment.

**Null Hypothesis:** Zonal Graph Quantization (ZGQ) does not provide statistically significant improvements in recall-latency trade-offs compared to baseline methods (HNSW, IVF-PQ).

## 2. ZGQ Formula

### 2.1 Mathematical Foundation

ZGQ combines three key components:

1. **Zonal Partitioning**: Divide dataset D into Z zones using K-Means clustering
2. **Per-Zone HNSW Graphs**: Build independent HNSW graphs for each zone
3. **Product Quantization**: Compress vectors for memory efficiency

**Formal Definition:**

Let D = {x₁, x₂, ..., xₙ} be the dataset of n vectors in R^d.

**Step 1: Zonal Partitioning**
```
{Z₁, Z₂, ..., Zₖ} = KMeans(D, k=Z)
C = {c₁, c₂, ..., cₖ} = centroids of zones
```

Where:
- Zᵢ = {x ∈ D | x assigned to zone i}
- cᵢ = centroid of zone i
- ∑|Zᵢ| = n (all vectors are assigned to exactly one zone)

**Step 2: Per-Zone Graph Construction**
```
For each zone Zᵢ:
  Gᵢ = HNSW_Construct(Zᵢ, M, ef_construction)
```

Where:
- Gᵢ = HNSW graph for zone i
- M = maximum connections per node
- ef_construction = construction parameter

**Step 3: Product Quantization**
```
{C₁, C₂, ..., Cₘ} = PQ_Train(D, m, k)  # m subspaces, k centroids per subspace
D_PQ = PQ_Encode(D, {C₁, C₂, ..., Cₘ})  # encoded dataset
```

Where:
- Cⱼ = codebook for subspace j (k centroids of dimension d/m)
- D_PQ[i, j] = index of nearest centroid in subspace j for vector i

### 2.2 ZGQ Search Algorithm

**Input:**
- Query q ∈ R^d
- Number of results k
- Number of zones to probe n_probe
- HNSW search parameter ef_search

**Algorithm:**
```
1. SELECT ZONES:
   distances_to_centroids = [||q - cᵢ||₂² for i in 1..Z]
   selected_zones = argmin_n(distances_to_centroids, n_probe)

2. PRECOMPUTE PQ TABLE:
   pq_table = PQ_ComputeDistanceTable(q, {C₁, C₂, ..., Cₘ})

3. PARALLEL SEARCH IN ZONES:
   candidates = []
   For each zone j in selected_zones:
     local_ids, distances = HNSW_Search(Gⱼ, q, k_local, ef_search)
     global_ids = map_local_to_global(zone_graphs[j], local_ids)
     pq_distances = PQ_AsymmetricDistance(global_ids, pq_table)
     candidates.extend(zip(global_ids, pq_distances))

4. AGGREGATE & RERANK:
   deduplicated_candidates = deduplicate_by_id(candidates)
   top_k_rerank = select_top_k(deduplicated_candidates, k_rerank, pq_distances)
   exact_distances = compute_exact_distances(q, top_k_rerank)
   final_results = select_top_k(top_k_rerank, k, exact_distances)
```

### 2.3 Complexity Analysis

**Build Complexity:**
- Zonal Partitioning: O(K_iter · N · Z · d)
- Per-Zone HNSW: O(N · log(N/Z) · M · d) total
- Product Quantization: O(K_iter · N · k · d)
- **Total:** O(N · Z · d + N · log(N/Z) · M · d + N · k · d)

**Search Complexity:**
- Zone Selection: O(Z · d)
- PQ Table Computation: O(m · k · d)
- Parallel Search: O(n_probe · log(N/Z) · ef_search · d)
- Aggregation & Re-ranking: O(C · log C + k_rerank · d) where C = total candidates
- **Total:** O(Z · d + m · k · d + n_probe · log(N/Z) · ef_search · d + k_rerank · d)

## 3. Theoretical Advantages

### 3.1 Zonal Partitioning Benefits

**Spatial Locality**: By clustering similar vectors together, we reduce the search space during queries. A query is likely to find its nearest neighbors in zones that are spatially close to the query.

**Parallel Search**: Multiple zones can be searched in parallel, utilizing multi-core architectures efficiently.

**Load Balancing**: Properly balanced zones distribute query load evenly across the system.

### 3.2 Per-Zone HNSW Benefits

**Scalability**: Each zone maintains a smaller HNSW graph, reducing the complexity of navigation within each zone.

**Quality Preservation**: HNSW graphs maintain high search quality within each zone due to their navigable small-world properties.

**Independent Maintenance**: Zones can be updated independently, allowing for efficient incremental updates.

### 3.3 Product Quantization Benefits

**Memory Efficiency**: Compresses vectors 16-32× while maintaining good approximation quality.

**Fast Distance Computation**: Asymmetric Distance Computation (ADC) allows rapid distance estimation during search.

## 4. Validation Framework

### 4.1 Research Questions

RQ1: Does ZGQ achieve better recall-latency trade-offs than baseline methods?
RQ2: How does the number of zones (Z) impact performance metrics?
RQ3: What is the optimal balance between zonal partitioning and PQ compression?
RQ4: How does ZGQ scale with dataset size and dimensionality?

### 4.2 Experimental Design

**Baseline Algorithms:**
1. HNSW (Hierarchical Navigable Small World)
2. IVF-PQ (Inverted File with Product Quantization)
3. Brute Force (for ground truth and small datasets)

**Evaluation Metrics:**
- Recall@k (k = 1, 5, 10, 20, 50, 100)
- Query latency (mean, median, p95, p99)
- Throughput (queries per second)
- Memory usage
- Build time
- Index size

**Datasets:**
- Synthetic datasets with controlled properties
- Real-world datasets (SIFT, GIST, Deep1B, etc.)
- Varying sizes (10K to 10M vectors)
- Varying dimensions (32 to 1024)

### 4.3 Statistical Analysis

- Paired t-tests to compare ZGQ with baselines
- Effect size calculations (Cohen's d)
- Confidence intervals for performance metrics
- Power analysis to ensure adequate sample sizes