# Aggregation and Re-ranking in ZGQ Architecture

## 1. Aggregation & Re-ranking

**Algorithm:** `aggregate_and_rerank`
**Reference:** "Section 3.5, Step 3"

### 1.1 Input and Output
**Input:**
- query: "Vector[d]"
- all_candidates: "List[Tuple[int, float]] - (vector_id, pq_distance)"
- full_vectors: "Matrix[N, d] - original vectors"
- k: "int - final number of results"
- k_rerank: "int - number of candidates to rerank with exact distance"

**Output:**
- top_k_results: "List[Tuple[int, float]] - (vector_id, exact_distance)"

### 1.2 Mathematical Formulation
**Step 1: Deduplication**
```
C_unique = {(id, dist) : only keep best dist for each unique id}
```

**Step 2: Select for Re-ranking**
```
C_rerank = top k_rerank candidates by PQ distance
```

**Step 3: Compute Exact Distances**
```
For each (id, d_pq) ∈ C_rerank:
  d_exact = ||query - full_vectors[id]||²
```

**Step 4: Final Selection**
```
results = top k candidates by d_exact
```

### 1.3 Pseudocode
```
function aggregate_and_rerank(
    query: Vector[d],
    all_candidates: List[Tuple[int, float]],
    full_vectors: Matrix[N, d],
    k: int,
    k_rerank: int = None
) -> List[Tuple[int, float]]:
    
    if k_rerank is None:
        k_rerank = min(len(all_candidates), 2 * k)
    
    # Step 1: Deduplicate - O(C log C) where C = |all_candidates|
    # Keep best PQ distance for each ID
    candidates_dict = {}
    for vec_id, pq_dist in all_candidates:
        if vec_id not in candidates_dict or pq_dist < candidates_dict[vec_id]:
            candidates_dict[vec_id] = pq_dist
    
    # Convert to list and sort by PQ distance
    unique_candidates = [(id, dist) for id, dist in candidates_dict.items()]
    unique_candidates.sort(key=lambda x: x[1])
    
    # Step 2: Select top k_rerank by PQ distance
    candidates_to_rerank = unique_candidates[:k_rerank]
    
    # Step 3: Compute exact distances - O(k_rerank · d)
    exact_results = []
    for vec_id, _ in candidates_to_rerank:
        exact_dist = euclidean_squared(query, full_vectors[vec_id])
        exact_results.append((vec_id, exact_dist))
    
    # Step 4: Sort by exact distance and return top-k
    exact_results.sort(key=lambda x: x[1])
    
    return exact_results[:k]
```

### 1.4 Complexity Analysis
- Deduplication: "O(C log C) where C = n_probe · k_local"
- Exact distance computation: "O(k_rerank · d)"
- Final sort: "O(k_rerank log k_rerank)"
- Total: "O(C log C + k_rerank · d)"

### 1.5 Memory Considerations
**Full Vectors Access Options:**
- Option 1: "Keep all in RAM - simple but memory-intensive"
- Option 2: "Memory-mapped file - trades memory for I/O"
- Option 3: "SSD with cache - hybrid approach"