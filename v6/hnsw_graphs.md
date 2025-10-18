# HNSW Graphs in ZGQ Architecture

## 1. Per-Zone HNSW Graph Construction

**Algorithm:** `build_zone_hnsw_graphs`
**Reference:** "Section 3.3, HNSW paper (Malkov & Yashunin 2018)"

### 1.1 Input and Output
**Input:**
- inverted_lists: "List[List[int]] - vector indices per zone"
- vectors: "Matrix[N, d] - full dataset"
- M: "Maximum number of connections per node"
- ef_construction: "Exploration factor during build"

**Output:**
- zone_graphs: "List[HNSWGraph] - one graph per zone"

### 1.2 Per-Zone Construction
**Input for zone i:**
- zone_vectors: "vectors[inverted_lists[i]]"
- n_i: "|inverted_lists[i]| - number of vectors in zone i"

#### 1.2.1 HNSW Build Algorithm
**Formula:**
```
For each vector x in zone Zᵢ (in random order):
  1. Insert x into graph at appropriate layer ℓ
  2. Connect x to M nearest neighbors at each layer ≤ ℓ
  3. Update neighbors' connections (maintain M max)
```

**Layer Selection:**
```
ℓ = ⌊-ln(uniform(0,1)) · m_L⌋
where m_L = 1/ln(M)
```

**Nearest Neighbor Search During Build:**
- Method: "Beam search with ef_construction candidates"
- Formula:
```
For layer ℓ:
  candidates = beam_search(x, graph_ℓ, ef_construction)
  neighbors = select_M_nearest(candidates, M)
  add_bidirectional_edges(x, neighbors)
```

**Connection Pruning:**
```
If node v has > M connections:
  Keep M connections to nearest neighbors by distance
```

### 1.3 Pseudocode
```
function build_zone_graph(
    zone_vectors: Matrix[n_zone, d],
    M: int,
    ef_construction: int
) -> HNSWGraph:
    
    graph = HNSWGraph()
    graph.entry_point = 0
    graph.adjacency_lists = [[] for _ in range(n_zone)]
    
    # Insert first vector
    graph.add_node(0, layer=0)
    
    # Insert remaining vectors - O(n_zone · log(n_zone) · M · d)
    for i in range(1, n_zone):
        # Determine layer for this node
        layer = select_layer_randomly(M)
        
        # Find insertion points via beam search
        candidates = beam_search(
            query=zone_vectors[i],
            graph=graph,
            ef=construction,
            entry_point=graph.entry_point
        )
        
        # Connect to M nearest at each layer
        for ℓ in range(layer + 1):
            neighbors = select_m_nearest(candidates[ℓ], M)
            
            for neighbor_id in neighbors:
                # Bidirectional edge
                graph.add_edge(i, neighbor_id, layer=ℓ)
                graph.add_edge(neighbor_id, i, layer=ℓ)
                
                # Prune if neighbor has too many connections
                if len(graph.adjacency_lists[neighbor_id]) > M:
                    prune_connections(neighbor_id, M)
    
    return graph
```

### 1.4 Complexity Analysis
**Per zone with n_zone vectors:** "O(n_zone · log(n_zone) · M · d)"

**Total for all zones:**
```
Z zones, average n_zone = N/Z
Total: Z · O((N/Z) · log(N/Z) · M · d) = O(N · log(N/Z) · M · d)
```

#### 1.4.1 Comparison to Single HNSW
- HNSW full: "O(N · log(N) · M · d)"
- ZGQ zones: "O(N · log(N/Z) · M · d)"
- Speedup: "Factor of log(N) / log(N/Z)"

#### 1.4.2 Parallelization
- Strategy: "Build each zone graph independently"
- Threads: "Use Z threads (or min(Z, num_cores))"
- Scaling: "Near-linear speedup with cores"