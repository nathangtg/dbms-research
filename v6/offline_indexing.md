# Offline Indexing in ZGQ Architecture

## 1. Zonal Partitioning via K-Means

**Algorithm:** `kmeans_zonal_partitioning`
**Reference:** "Section 3.2 in implementation document"

### 1.1 Input and Output
**Input:**
- D: "Dataset matrix of shape (N, d)"
- Z: "Number of zones to create"

**Output:**
- centroids: "Matrix of shape (Z, d) - zone centroids"
- assignments: "Vector of shape (N,) - zone_id for each vector"
- inverted_lists: "List[List[int]] - vector indices per zone"

### 1.2 Mathematical Formulation
**Objective:**
```
minimize L(Z₁, ..., Zₖ, c₁, ..., cₖ) = Σᵢ₌₁ᶻ Σₓ∈Zᵢ ||x - cᵢ||²
```

#### 1.2.1 Alternating Optimization
**Assignment Step:**
```
For each x ∈ D:
  zone(x) = argminⱼ ||x - cⱼ||²
```

**Update Centroids Step:**
```
cⱼ = (1/|Zⱼ|) Σₓ∈Zⱼ x
```

### 1.3 Pseudocode
```
function kmeans_partition(D: Matrix[N, d], Z: int) -> (Matrix, Vector, List):
    # Initialize centroids (e.g., k-means++)
    centroids = initialize_centroids(D, Z)
    
    # Iterate until convergence
    for iteration in range(max_iterations):
        # ASSIGNMENT STEP - O(N·Z·d)
        distances = np.zeros((N, Z))
        for i in range(N):
            for j in range(Z):
                distances[i, j] = euclidean_squared(D[i], centroids[j])
        
        assignments = np.argmin(distances, axis=1)
        
        # UPDATE STEP - O(N·d)
        old_centroids = centroids.copy()
        for j in range(Z):
            mask = (assignments == j)
            if np.any(mask):
                centroids[j] = np.mean(D[mask], axis=0)
        
        # Check convergence
        if np.allclose(centroids, old_centroids):
            break
    
    # Build inverted lists - O(N)
    inverted_lists = [[] for _ in range(Z)]
    for i, zone_id in enumerate(assignments):
        inverted_lists[zone_id].append(i)
    
    return centroids, assignments, inverted_lists
```

### 1.4 Complexity Analysis
**Per iteration:** O(N·Z·d + N·d)
**Total:** O(K_iter · N·Z·d)
**Typical k_iter:** "10-100 iterations"

### 1.5 Optimization Techniques
**Minibatch K-Means:**
- Benefit: "Reduce to O(batch_size · Z · d) per iteration"
- Library: "sklearn.cluster.MiniBatchKMeans"

**K-Means++ Initialization:**
- Benefit: "Faster convergence, better local minima"

**Early Stopping:**
- Condition: "||centroids_new - centroids_old|| < tolerance"