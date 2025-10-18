# ZGQ API Documentation

## 1. Main ZGQIndex Class

### 1.1 Class Definition
```python
class ZGQIndex:
    """
    Complete Zonal Graph Quantization index.
    
    The ZGQIndex combines zonal partitioning, per-zone HNSW graphs, and 
    optional product quantization to achieve efficient approximate nearest 
    neighbor search.
    """
```

### 1.2 Constructor Parameters
```python
def __init__(
    self,
    n_zones: int = 100,           # Number of zones for partitioning
    hnsw_M: int = 16,             # HNSW maximum connections per node
    hnsw_ef_construction: int = 200,  # HNSW construction parameter
    hnsw_ef_search: int = 50,     # HNSW search parameter
    use_pq: bool = True,          # Whether to use Product Quantization
    pq_m: int = 16,               # Number of PQ subspaces
    pq_nbits: int = 8,            # Bits per PQ centroid index
    verbose: bool = True          # Whether to print progress
):
```

**Parameter Details:**
- `n_zones`: Number of zones to partition the dataset into. More zones allow for finer-grained search but require more computational resources.
- `hnsw_M`: Maximum number of connections per node in HNSW graphs. Higher values improve search quality but increase memory usage.
- `hnsw_ef_construction`: Exploration factor during HNSW construction. Higher values improve graph quality but slow down construction.
- `hnsw_ef_search`: Exploration factor during search. Higher values improve recall but increase query time.
- `use_pq`: Whether to use Product Quantization for memory efficiency.
- `pq_m`: Number of subspaces for PQ. The original dimension `d` must be divisible by `m`.
- `pq_nbits`: Number of bits per centroid index in PQ. Determines k (number of centroids) as 2^nbits.

### 1.3 Build Method
```python
def build(self, vectors: np.ndarray) -> None:
    """
    Build the ZGQ index from input vectors.
    
    Args:
        vectors: Matrix of shape (N, d) containing N vectors of dimension d
    """
```

**Build Process:**
1. Perform zonal partitioning using K-Means
2. Build HNSW graphs for each zone in parallel
3. Train PQ codebooks (if use_pq=True)
4. Encode all vectors with PQ (if use_pq=True)
5. Precompute vector norms for optimization

### 1.4 Search Method
```python
def search(
    self, 
    query: np.ndarray, 
    k: int = 10, 
    n_probe: int = 8, 
    ef_search: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors to the query.
    
    Args:
        query: Query vector of shape (d,)
        k: Number of nearest neighbors to return
        n_probe: Number of zones to search
        ef_search: Override for HNSW search parameter (optional)
        
    Returns:
        (ids, distances): Arrays of shape (k,) with neighbor IDs and distances
    """
```

**Search Process:**
1. Select `n_probe` nearest zones based on centroid distances
2. Precompute PQ distance table for the query
3. Search each selected zone's HNSW graph in parallel
4. Aggregate candidates from all zones and deduplicate
5. Re-rank top candidates with exact distances
6. Return top-k results

### 1.5 Batch Search Method
```python
def batch_search(
    self,
    queries: np.ndarray,
    k: int = 10,
    n_probe: int = 8,
    ef_search: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search for nearest neighbors to multiple queries.
    
    Args:
        queries: Matrix of shape (n_queries, d)
        k: Number of neighbors per query
        n_probe: Number of zones to probe per query
        ef_search: Override for HNSW search parameter (optional)
        
    Returns:
        (all_ids, all_distances): Arrays of shape (n_queries, k)
    """
```

### 1.6 Serialization Methods
```python
def save(self, save_dir: str) -> None:
    """
    Save the index to disk.
    
    Args:
        save_dir: Directory to save the index to
    """

@classmethod
def load(cls, load_dir: str) -> 'ZGQIndex':
    """
    Load an index from disk.
    
    Args:
        load_dir: Directory containing saved index
        
    Returns:
        Loaded ZGQIndex instance
    """
```

### 1.7 Memory Usage
```python
def memory_usage(self) -> Dict[str, float]:
    """
    Calculate memory usage of different index components.
    
    Returns:
        Dictionary with memory usage in MB for each component
    """
```

## 2. Core Module APIs

### 2.1 Distance Computation (distances.py)

#### Euclidean Distance
```python
class DistanceMetrics:
    @staticmethod
    def euclidean_squared(a: np.ndarray, b: np.ndarray) -> float:
        """Compute squared Euclidean distance: ||a - b||²"""
    
    @staticmethod
    def euclidean_batch_squared(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute distances from query to multiple vectors"""
    
    @staticmethod
    def euclidean_batch_with_norms(
        query: np.ndarray,
        vectors: np.ndarray,
        query_norm_sq: float,
        vector_norms_sq: np.ndarray
    ) -> np.ndarray:
        """Optimized batch distance using precomputed norms"""
```

#### Product Quantization Distance
```python
class PQDistanceComputer:
    def __init__(self, codebooks: List[np.ndarray]):
        """Initialize with PQ codebooks"""
    
    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """Precompute distance table for ADC: table[j, ℓ] = ||query_j - codebook_j[ℓ]||²"""
    
    def asymmetric_distance(self, codes: np.ndarray, distance_table: np.ndarray) -> np.ndarray:
        """Compute approximate distances using ADC: d²_PQ = Σⱼ table[j, codes[i,j]]"""
```

### 2.2 K-Means Partitioning (kmeans.py)

```python
class KMeansPartitioner:
    def __init__(
        self,
        n_zones: int,
        max_iterations: int = 300,
        batch_size: int = 2048,
        verbose: bool = True
    ):
        """Initialize partitioner"""
    
    def fit(self, vectors: np.ndarray) -> ZonalPartition:
        """Partition vectors into zones using K-Means"""
    
    def assign(self, vectors: np.ndarray) -> np.ndarray:
        """Assign new vectors to existing zones"""
```

### 2.3 Per-Zone HNSW (hnsw_wrapper.py)

```python
class HNSWZoneBuilder:
    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        space: str = 'l2',
        verbose: bool = True
    ):
        """Initialize HNSW builder for zones"""
    
    def build_all_zones_parallel(
        self,
        vectors: np.ndarray,
        inverted_lists: List[List[int]],
        max_workers: Optional[int] = None
    ) -> List[Optional[ZoneGraph]]:
        """Build HNSW graphs for all zones in parallel"""
    
    def search_zone(
        self,
        zone_graph: ZoneGraph,
        query: np.ndarray,
        k: int,
        ef_search: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search within a single zone's HNSW graph"""
```

### 2.4 Product Quantization (product_quantizer.py)

```python
class ProductQuantizer:
    def __init__(
        self,
        m: int = 16,
        nbits: int = 8,
        max_iterations: int = 100,
        verbose: bool = True
    ):
        """Initialize Product Quantizer"""
    
    def train(self, vectors: np.ndarray, sample_size: Optional[int] = None) -> PQCodebook:
        """Train PQ codebooks on input vectors"""
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors into PQ codes of shape (N, m)"""
    
    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """Precompute distance table for ADC"""
    
    def asymmetric_distance(self, codes: np.ndarray, distance_table: np.ndarray) -> np.ndarray:
        """Compute approximate distances using ADC"""
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode PQ codes back to approximate vectors"""
```

## 3. Utility Functions

### 3.1 Dimension Analysis
```python
def compute_optimal_n_zones(N: int, n_probe: int = 8) -> int:
    """Compute theoretically optimal number of zones: Z* = Θ(√(N / n_probe))"""

def precompute_norms(vectors: np.ndarray) -> np.ndarray:
    """Precompute squared L2 norms for all vectors: ||vectors[i]||²"""
```

### 3.2 Zone Balance Analysis
```python
def analyze_zone_balance(inverted_lists: List[List[int]]) -> dict:
    """Analyze balance of zone sizes"""
```

### 3.3 Memory Estimation
```python
def estimate_hnsw_memory(n_vectors: int, M: int, d: int) -> float:
    """Estimate memory usage for HNSW graph in MB"""
```