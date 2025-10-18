# Online Search in ZGQ Architecture

## 1. Zone Selection

**Algorithm:** `select_zones`
**Reference:** "Section 3.5, Step 1 in implementation doc"

### 1.1 Input and Output
**Input:**
- query: "Vector[d] - query vector q"
- centroids: "Matrix[Z, d] - zone centroids"
- n_probe: "int - number of zones to search"

**Output:**
- selected_zones: "Vector[n_probe] - indices of nearest zones"

### 1.2 Mathematical Formulation
**Objective:**
```
Find n_probe zones with smallest distances to query:

selected = argmin_{S⊂{1..Z}, |S|=n_probe} Σᵢ∈S d²(q, cᵢ)

Equivalently: Select n_probe smallest d²(q, cᵢ)
```

### 1.3 Pseudocode
```
function select_zones(
    query: Vector[d],
    centroids: Matrix[Z, d],
    n_probe: int
) -> Vector[n_probe]:
    
    # Compute distances to all centroids - O(Z · d)
    distances = np.zeros(Z)
    for i in range(Z):
        distances[i] = euclidean_squared(query, centroids[i])
    
    # Select n_probe smallest - O(Z) using argpartition
    selected_indices = np.argpartition(distances, n_probe)[:n_probe]
    
    # Optional: sort by distance for prioritization
    selected_indices = selected_indices[np.argsort(distances[selected_indices])]
    
    return selected_indices
```

**Complexity:** O(Z · d + Z) = O(Z · d)

### 1.4 Optimization
**Early Termination:**
- Condition: "If one zone very close, can probe fewer"
- Formula:
```
If d(q, c_i*) < threshold · min_j≠i* d(q, c_j):
  n_probe_actual = 1  # Only search zone i*
```

## 2. Parallel Intra-Zone HNSW Search

**Algorithm:** `parallel_zone_search`
**Reference:** "Section 3.5, Step 2"

### 2.1 Input and Output
**Input:**
- query: "Vector[d]"
- selected_zones: "Vector[n_probe] - zone indices"
- zone_graphs: "List[HNSWGraph]"
- pq_codes: "Matrix[N, m] - quantized vectors"
- distance_table: "Matrix[m, k] - precomputed PQ distances"
- ef_search: "int - HNSW exploration factor"
- k_local: "int - candidates to return per zone"

**Output:**
- candidates: "List[Tuple[int, float]] - (vector_id, distance) pairs"

### 2.2 Per-Zone Search Algorithm
**Algorithm:** "HNSW beam search with PQ distances"

**Pseudocode:**
```
function search_zone(
    query: Vector[d],
    zone_graph: HNSWGraph,
    local_to_global_ids: Dict[int, int],
    pq_codes: Matrix,
    distance_table: Matrix,
    ef_search: int,
    k_local: int
) -> List[Tuple[int, float]]:
    
    # Priority queue: min-heap of (distance, node_id)
    candidates = MinHeap()
    visited = Set()
    
    # Start from entry point
    entry_id = zone_graph.entry_point
    entry_dist = pq_distance(query, pq_codes[entry_id], distance_table)
    candidates.push((entry_dist, entry_id))
    visited.add(entry_id)
    
    # Result set: max-heap of (distance, node_id)
    results = MaxHeap(maxsize=ef_search)
    results.push((entry_dist, entry_id))
    
    # Beam search - O(log(n_zone) · ef_search · m)
    while candidates:
        current_dist, current_id = candidates.pop()
        
        # Stopping condition
        if current_dist > results.max():
            break
        
        # Explore neighbors
        for neighbor_id in zone_graph.neighbors(current_id):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                
                # Compute PQ distance - O(m)
                neighbor_dist = pq_distance(
                    query, 
                    pq_codes[neighbor_id], 
                    distance_table
                )
                
                # Update queues
                if neighbor_dist < results.max() or len(results) < ef_search:
                    candidates.push((neighbor_dist, neighbor_id))
                    results.push((neighbor_dist, neighbor_id))
                    
                    if len(results) > ef_search:
                        results.pop()  # Remove farthest
    
    # Return top k_local with global IDs
    top_k = results.get_sorted()[:k_local]
    return [(local_to_global_ids[local_id], dist) for dist, local_id in top_k]
```

### 2.3 Parallel Execution
**Pseudocode:**
```
function parallel_search_all_zones(
    query: Vector[d],
    selected_zones: Vector[n_probe],
    ...
) -> List[Tuple[int, float]]:
    
    all_candidates = []
    
    # Execute in parallel
    with ThreadPool(num_threads=n_probe) as pool:
        results = pool.map(
            lambda zone_id: search_zone(
                query,
                zone_graphs[zone_id],
                ...,
                ef_search,
                k_local
            ),
            selected_zones
        )
    
    # Merge results
    for zone_results in results:
        all_candidates.extend(zone_results)
    
    return all_candidates
```

### 2.4 Complexity Analysis
**Per zone:**
- Graph traversal: "O(log(N/Z) · ef_search)"
- Distance per node: "O(m) for PQ distance"
- Total per zone: "O(log(N/Z) · ef_search · m)"

**Total complexity:**
- Sequential: "O(n_probe · log(N/Z) · ef_search · m)"
- Parallel: "O(log(N/Z) · ef_search · m) with n_probe threads"