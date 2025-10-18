# Distance Computations in ZGQ Architecture

## 1. Core Distance Operations

### 1.1 Euclidean Distance (Exact)

**Operation:** `euclidean_distance_squared`

**Formula:**
```
d²(x, y) = ||x - y||² = Σᵢ₌₁ᵈ (xᵢ - yᵢ)²
```

#### 1.1.1 Naive Implementation
**Code:**
```python
def euclidean_squared(x: Vector, y: Vector) -> float:
    diff = x - y
    return np.dot(diff, diff)
```
**Complexity:** O(d)

#### 1.1.2 Optimized with Precomputed Norms
**Formula:**
```
||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
```

**Code:**
```python
def euclidean_squared_optimized(
    x: Vector, 
    y: Vector,
    x_norm_sq: float,  # precomputed ||x||²
    y_norm_sq: float   # precomputed ||y||²
) -> float:
    dot_product = np.dot(x, y)
    return x_norm_sq + y_norm_sq - 2 * dot_product
```
**Complexity:** O(d)
**Advantage:** "50% fewer operations if norms precomputed"

#### 1.1.3 Batch Computation
**Formula:**
```
For query q and matrix X of shape (n, d):
distances[i] = ||q - X[i]||² for i = 1..n
```

**Code:**
```python
def euclidean_batch_squared(q: Vector, X: Matrix) -> Vector:
    diff = X - q  # broadcasting
    return np.einsum('ij,ij->i', diff, diff)
```
**Complexity:** O(n·d)

### 1.2 Product Quantization Distance (Approximate)

**Operation:** `pq_asymmetric_distance`

**Formula:**
```
Given:
  - Query q ∈ ℝᵈ
  - PQ code c = [c₁, c₂, ..., cₘ] where cⱼ ∈ {0, 1, ..., k-1}
  - Codebooks {C₁, C₂, ..., Cₘ} where Cⱼ ∈ ℝᵏˣ⁽ᵈ/ᵐ⁾

Step 1: Precompute distance table
  T[j, ℓ] = ||qⱼ - Cⱼ[ℓ]||² for j=1..m, ℓ=0..k-1
  where qⱼ = q[(j-1)·d/m : j·d/m]

Step 2: Lookup and sum
  d²_PQ(q, c) = Σⱼ₌₁ᵐ T[j, cⱼ]
```

#### 1.2.1 Precompute Distance Table
**Code:**
```python
def compute_distance_table(
    query: Vector,      # shape (d,)
    codebooks: List[Matrix],  # m codebooks, each (k, d/m)
    m: int,            # number of subspaces
    k: int             # codebook size
) -> Matrix:
    d = len(query)
    d_sub = d // m
    table = np.zeros((m, k), dtype=np.float32)
    
    for j in range(m):
        start = j * d_sub
        end = start + d_sub
        q_sub = query[start:end]  # subvector
        
        # Distance from q_sub to all k centroids in codebook j
        for ℓ in range(k):
            centroid = codebooks[j][ℓ]
            table[j, ℓ] = np.sum((q_sub - centroid) ** 2)
    
    return table
```
**Complexity:** O(m·k·d/m) = O(k·d)

#### 1.2.2 Compute PQ Distance
**Code:**
```python
def pq_asymmetric_distance(
    codes: Matrix,      # shape (n, m) - PQ codes for n vectors
    distance_table: Matrix  # shape (m, k) - precomputed
) -> Vector:
    n, m = codes.shape
    distances = np.zeros(n, dtype=np.float32)
    
    for j in range(m):
        distances += distance_table[j, codes[:, j]]
    
    return distances
```
**Complexity:** O(n·m)
**Total per query:** O(k·d + n·m)

#### 1.2.3 Approximation Error
**Bound:**
```
E[d²_PQ(q, x)] = d²(q, x) + ε
where ε depends on quantization resolution (k, m)
```

**Typical Error:** "5-15% distance distortion for m=16, k=256"