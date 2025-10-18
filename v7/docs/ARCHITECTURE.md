# ZGQ Architecture Overview

## 1. System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    ZGQ Index Structure                      │
├─────────────────────────────────────────────────────────────┤
│  1. Zone Centroids [Z × d]                                  │
│     - K-means cluster centers                               │
│     - Used for zone selection                               │
├─────────────────────────────────────────────────────────────┤
│  2. Inverted Lists [Z lists]                                │
│     - Zone i → [vector_ids in zone i]                       │
│     - Enables per-zone operations                           │
├─────────────────────────────────────────────────────────────┤
│  3. Per-Zone HNSW Graphs [Z graphs]                         │
│     - Independent navigable small world graph per zone      │
│     - Each graph: ~(N/Z) nodes                              │
├─────────────────────────────────────────────────────────────┤
│  4. Product Quantization (Optional)                         │
│     a. Codebooks [m × k × (d/m)]                            │
│     b. PQ Codes [N × m] (uint8)                             │
│     - Compresses vectors 16-32×                             │
├─────────────────────────────────────────────────────────────┤
│  5. Auxiliary Data                                          │
│     - Vector norms [N] (for optimized distance)             │
│     - Metadata mappings                                     │
│     - (Optional) Full vectors for re-ranking                │
└─────────────────────────────────────────────────────────────┘
```

## 2. Data Flow

### 2.1 Build Phase:
```
Input Vectors [N × d]
    ↓
┌───────────────────┐
│ K-Means Clustering│ → Centroids [Z × d]
│   (Z clusters)    │ → Assignments [N]
└────────┬──────────┘
         ↓
    ┌────────────────────────┐
    │ Split by Zone          │ → Z subsets of vectors
    └────────┬───────────────┘
             ↓
    ┌────────────────────────┐
    │ Parallel HNSW Build    │ → Zone Graphs [Z]
    │  (per zone)            │
    └────────┬───────────────┘
             ↓
    ┌────────────────────────┐
    │ Train PQ Codebooks     │ → Codebooks [m × k × d/m]
    └────────┬───────────────┘
             ↓
    ┌────────────────────────┐
    │ Encode All Vectors     │ → PQ Codes [N × m]
    └────────┬───────────────┘
             ↓
       ZGQ Index Ready
```

### 2.2 Search Phase:
```
Query [d]
    ↓
┌───────────────────┐
│ Zone Selection    │ → n_probe nearest zones
│ (distance to      │
│  centroids)       │
└────────┬──────────┘
         ↓
┌────────────────────────┐
│ Precompute PQ Table    │ → Distance Table [m × k]
│ (query → codebooks)    │
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│ Parallel HNSW Search   │ → Candidates from each zone
│ (n_probe zones)        │    (using PQ distances)
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│ Aggregate Candidates   │ → Deduplicated candidates
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│ Re-rank with Exact     │ → Top-k results
│ Distances              │
└────────────────────────┘
```

## 3. Source Code Organization

```
src/
├── core/
│   ├── distances.py          # Distance computation kernels
│   ├── kmeans.py             # Zonal partitioning
│   ├── hnsw_wrapper.py       # Per-zone HNSW management
│   ├── product_quantizer.py  # PQ training & encoding
│   └── utils.py              # Helper functions
├── index.py                  # Main ZGQIndex class
├── search.py                 # Search algorithms
└── serialization.py          # Save/load index
```

## 4. Core Modules

### 4.1 Distance Computations (distances.py)
Implements:
- Euclidean distance (exact, squared)
- Optimized distance with precomputed norms
- Batch distance computations
- PQ asymmetric distance computation (ADC)

### 4.2 K-Means Zonal Partitioning (kmeans.py)
Implements:
- Training centroids via MiniBatchKMeans
- Assigning vectors to zones
- Building inverted lists for efficient access

### 4.3 Per-Zone HNSW Construction (hnsw_wrapper.py)
Implements:
- Parallel HNSW graph construction across zones
- Local-to-global ID mapping
- Serialization/deserialization of zone graphs

### 4.4 Product Quantization (product_quantizer.py)
Implements:
- PQ codebook training via K-means on subspaces
- Vector encoding to PQ codes
- Asymmetric Distance Computation (ADC)

### 4.5 Search Pipeline (search.py)
Implements:
- Zone selection based on centroid proximity
- Parallel intra-zone HNSW search
- Aggregation and re-ranking of candidates