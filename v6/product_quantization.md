# Product Quantization in ZGQ Architecture

## 1. Product Quantization Algorithm

**Algorithm:** `product_quantization`
**Reference:** "Jegou et al. 2011, Section 3.4 in implementation doc"

### 1.1 Parameters
- m: "Number of subspaces (e.g., 8, 16, 32)"
- nbits: "Bits per subquantizer (e.g., 8 → k=256 centroids)"
- k: "Codebook size k = 2^nbits"

## 2. Training Phase

### 2.1 Input and Output
**Input:**
- training_data: "Matrix[N_train, d] - sample of dataset"
- m: "Number of subspaces"
- k: "Codebook size per subspace"

**Output:**
- codebooks: "List[Matrix] - m codebooks, each of shape (k, d/m)"

### 2.2 Algorithm
**Formula:**
```
Divide d-dimensional space into m subspaces:
  Subspace j: dimensions [(j-1)·d/m, j·d/m)

For each subspace j:
  1. Extract training subvectors: X_j = training_data[:, (j-1)·d/m : j·d/m]
  2. Run k-means on X_j with k clusters
  3. Store centroids as codebook C_j
```

**Pseudocode:**
```
function train_pq(
    training_data: Matrix[N_train, d],
    m: int,
    k: int
) -> List[Matrix]:
    d_sub = d // m
    codebooks = []
    
    for j in range(m):
        start = j * d_sub
        end = start + d_sub
        
        # Extract subvectors - O(N_train · d/m)
        subvectors = training_data[:, start:end]
        
        # K-means clustering - O(K_iter · N_train · k · d/m)
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(subvectors)
        
        # Store codebook
        codebooks.append(kmeans.cluster_centers_)  # shape (k, d/m)
    
    return codebooks
```

**Complexity:** O(m · K_iter · N_train · k · d/m) = O(K_iter · N_train · k · d)

## 3. Encoding Phase

### 3.1 Input and Output
**Input:**
- vectors: "Matrix[N, d] - vectors to encode"
- codebooks: "List[Matrix] - trained codebooks"

**Output:**
- codes: "Matrix[N, m] of uint8 - quantized representation"

### 3.2 Algorithm
**Formula:**
```
For each vector x:
  For each subspace j:
    1. Extract subvector: x_j = x[(j-1)·d/m : j·d/m]
    2. Find nearest centroid: c_j = argmin_ℓ ||x_j - C_j[ℓ]||²
    3. Store index: codes[i, j] = c_j
```

**Pseudocode:**
```
function encode_pq(
    vectors: Matrix[N, d],
    codebooks: List[Matrix],
    m: int
) -> Matrix:
    d_sub = d // m
    codes = np.zeros((N, m), dtype=np.uint8)
    
    for j in range(m):
        start = j * d_sub
        end = start + d_sub
        
        # Extract all subvectors for subspace j - O(N · d/m)
        subvectors = vectors[:, start:end]
        
        # Find nearest centroid for each - O(N · k · d/m)
        codebook = codebooks[j]  # shape (k, d/m)
        
        for i in range(N):
            distances = np.sum((subvectors[i] - codebook) ** 2, axis=1)
            codes[i, j] = np.argmin(distances)
    
    return codes
```

**Complexity:** O(N · m · k · d/m) = O(N · k · d)

## 4. Memory Savings

**Original vectors:** N · d · 4 bytes (float32)
**PQ codes:** N · m · 1 byte (uint8)
**Codebooks:** m · k · (d/m) · 4 bytes

### 4.1 Compression Ratio
**Formula:** (N·d·4) / (N·m + m·k·d/m·4)

**Example:**
```
N=1M, d=128, m=16, k=256:
Original: 1M · 128 · 4 = 512 MB
PQ codes: 1M · 16 = 16 MB
Codebooks: 16 · 256 · 8 · 4 = 131 KB
Ratio: 512 / 16 ≈ 32× compression
```