# ZGQ Parameter Tuning Guide

## 1. Key Parameters and Their Impact

### 1.1 Zonal Partitioning Parameters

#### n_zones (Z)
**Description**: Number of zones to partition the dataset into.

**Impact**:
- **Too few zones**: Each zone becomes large, reducing the effectiveness of zonal search
- **Too many zones**: Search overhead increases, and individual zones may be too small for efficient HNSW

**Tuning Strategy**:
- **Small datasets** (N < 10K): Z = N/20 to N/10
- **Medium datasets** (10K ≤ N < 100K): Z = √N to √N × 1.5
- **Large datasets** (N ≥ 100K): Z = √N / 2 to √N
- **Rule of thumb**: Ensure each zone has at least 100-500 vectors for stable HNSW graphs

```python
# Example for different dataset sizes
def suggest_n_zones(N: int) -> int:
    if N < 10000:
        return max(20, N // 50)
    elif N < 100000:
        return int(N**0.5)
    else:
        return max(100, int(N**0.5) // 2)
```

#### n_probe
**Description**: Number of zones to search during query processing.

**Impact**:
- **Higher n_probe**: Better recall but slower queries
- **Lower n_probe**: Faster queries but potentially lower recall

**Tuning Strategy**:
- Start with 5-10% of total zones
- Increase until recall plateaus
- Consider query latency constraints

### 1.2 HNSW Parameters

#### M (Maximum Connections)
**Description**: Maximum number of connections per node in HNSW graphs.

**Impact**:
- **Higher M**: Better search quality, higher memory usage
- **Lower M**: Lower memory usage, potentially worse search quality

**Tuning Strategy**:
- **Default**: 16 (good balance)
- **High-quality**: 32-64
- **Memory-constrained**: 8-12

#### ef_construction
**Description**: Size of dynamic candidate list during HNSW construction.

**Impact**:
- **Higher ef_construction**: Better graph quality, slower build
- **Lower ef_construction**: Faster build, potentially worse graph quality

**Tuning Strategy**:
- **Default**: 200
- **Quality-focused**: 400-800
- **Speed-focused**: 100-150

#### ef_search
**Description**: Size of dynamic candidate list during HNSW search.

**Impact**:
- **Higher ef_search**: Better recall, slower queries
- **Lower ef_search**: Faster queries, potentially lower recall

**Tuning Strategy**:
- Scale with dataset size and query requirements
- Start with 50-100, increase until recall saturates

### 1.3 Product Quantization Parameters

#### m (Number of Subspaces)
**Description**: Number of subspaces to divide the vector dimension into.

**Impact**:
- **Higher m**: Better approximation, more computation
- **Lower m**: Faster computation, potentially worse approximation

**Tuning Strategy**:
- Must evenly divide the dimension d
- Common values: 8, 16, 32, 64
- For d=128: m=8, 16, 32 are valid

#### nbits (Bits per Subquantizer)
**Description**: Number of bits used to represent each subvector.

**Impact**:
- **Higher nbits**: More centroids (k=2^nbits), better approximation, more memory
- **Lower nbits**: Fewer centroids, less memory, potentially worse approximation

**Tuning Strategy**:
- **Default**: 8 bits (k=256 centroids)
- **Memory-sensitive**: 6-7 bits
- **Quality-focused**: 8-10 bits

## 2. Parameter Interaction Effects

### 2.1 Z vs n_probe
The ratio Z/n_probe affects how "focused" the search is:
- **Low ratio** (Z/n_probe < 10): Very focused search, may miss relevant zones
- **High ratio** (Z/n_probe > 50): More exhaustive search, higher recall but slower

### 2.2 M vs ef_search
These parameters should be tuned together:
- **Higher M** → can benefit from **higher ef_search**
- **Lower M** → **lower ef_search** may be sufficient

## 3. Systematic Parameter Tuning Process

### 3.1 Initial Parameter Selection
1. Start with default parameters:
   - n_zones = √N (for N vectors)
   - M = 16, ef_construction = 200, ef_search = 50
   - m = 16, nbits = 8
   - n_probe = 0.1 × n_zones

2. Build index with default parameters and measure:
   - Build time
   - Index size
   - Initial recall@10
   - Average query time

### 3.2 Parameter Tuning Steps

#### Step 1: Zone Count Optimization
```python
# Test different n_zones values
n_zones_range = [int(N**0.5 / 2), int(N**0.5), int(N**0.5 * 2)]
# Measure recall and query time for each
```

#### Step 2: HNSW Parameter Optimization
```python
# After selecting n_zones, optimize HNSW parameters
# Test ef_search values: [20, 50, 100, 200]
# Test M values: [8, 16, 32]
# Keep ef_construction at 200 for consistency
```

#### Step 3: n_probe Optimization
```python
# With fixed zone count and HNSW parameters
# Test n_probe values: [5, 10, 20, len(zone_graphs)]
# Measure recall vs latency trade-off
```

#### Step 4: Product Quantization Tuning
```python
# If using PQ, test different m and nbits combinations
# Test m values: [8, 16, 32]
# Test nbits values: [6, 8, 10]
```

### 3.3 Grid Search vs. Sequential Tuning

**Grid Search**: Test all parameter combinations (computationally expensive)
**Sequential Tuning**: Optimize one parameter group at a time (recommended)

## 4. Practical Tuning Examples

### 4.1 Small Dataset (10K vectors, 128 dimensions)
```python
index = ZGQIndex(
    n_zones=50,               # √10000 = 100, use smaller for stability
    hnsw_M=16,
    hnsw_ef_construction=200,
    hnsw_ef_search=50,
    use_pq=True,
    pq_m=16,
    pq_nbits=8,
    verbose=True
)
# n_probe: Test 3, 5, 8 zones
```

### 4.2 Medium Dataset (100K vectors, 128 dimensions)
```python
index = ZGQIndex(
    n_zones=200,              # √100000 ≈ 316, use 200 for balance
    hnsw_M=16,
    hnsw_ef_construction=200,
    hnsw_ef_search=100,       # Higher ef_search for larger dataset
    use_pq=True,
    pq_m=16,
    pq_nbits=8,
    verbose=True
)
# n_probe: Test 10, 20, 30 zones
```

### 4.3 Large Dataset (1M vectors, 128 dimensions)
```python
index = ZGQIndex(
    n_zones=500,              # √1000000 = 1000, use 500 for efficiency
    hnsw_M=32,               # Higher M for large dataset
    hnsw_ef_construction=400, # Higher construction quality
    hnsw_ef_search=200,       # Higher ef_search for large dataset
    use_pq=True,
    pq_m=32,                 # Higher m for better approximation
    pq_nbits=8,
    verbose=True
)
# n_probe: Test 25, 50, 75 zones
```

## 5. Performance Monitoring

### 5.1 Key Metrics to Track
- **Recall@k** (k=1, 5, 10, 20): Fraction of true nearest neighbors found
- **Query latency**: Time per query (mean, median, p95)
- **Throughput**: Queries per second
- **Index build time**: Time to construct the index
- **Memory usage**: Peak memory during search, index size

### 5.2 Early Warning Signs
- **Very low recall**: Consider increasing ef_search or n_probe
- **High variance in query times**: Zone imbalance issue
- **Slow build time**: Consider increasing ef_construction incrementally
- **High memory usage**: Reduce M, consider compression options