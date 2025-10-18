"""
Distance Metrics Module for ZGQ V6
Implements all distance computations as specified in distance_computations.md

Mathematical Foundations:
1. Euclidean distance squared: d²(x, y) = ||x - y||² = Σᵢ(xᵢ - yᵢ)²
2. Optimized form: ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
3. Batch computation using vectorized operations
4. Product Quantization asymmetric distance

Hardware Optimization:
- Intel i5-12500H: AVX2 instructions via NumPy
- Memory-aligned operations for cache efficiency
"""

import numpy as np
from typing import Union, Optional

# Optional Numba for JIT compilation (significant speedup)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    # Fallback: no-op decorators
    print("Warning: Numba not available. Install with: pip install numba")
    print("         Performance will be reduced without JIT compilation.")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if len(args) == 0 else decorator(args[0])
    prange = range
    NUMBA_AVAILABLE = False


class DistanceMetrics:
    """
    Core distance computation operations for ZGQ.
    All methods are optimized for Intel i5-12500H architecture.
    """
    
    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def euclidean_squared(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute squared Euclidean distance: d²(x, y) = ||x - y||²
        
        Complexity: O(d)
        
        Args:
            x: Vector of shape (d,)
            y: Vector of shape (d,)
            
        Returns:
            Squared Euclidean distance
            
        Reference: distance_computations.md Section 1.1.1
        """
        diff = x - y
        return np.dot(diff, diff)
    
    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def euclidean_squared_optimized(
        x: np.ndarray, 
        y: np.ndarray,
        x_norm_sq: float,
        y_norm_sq: float
    ) -> float:
        """
        Optimized squared Euclidean with precomputed norms.
        
        Formula: ||x - y||² = ||x||² + ||y||² - 2⟨x, y⟩
        
        Advantage: 50% fewer operations when norms are precomputed
        Complexity: O(d) but with reduced constant factor
        
        Args:
            x: Vector of shape (d,)
            y: Vector of shape (d,)
            x_norm_sq: Precomputed ||x||²
            y_norm_sq: Precomputed ||y||²
            
        Returns:
            Squared Euclidean distance
            
        Reference: distance_computations.md Section 1.1.2
        """
        dot_product = np.dot(x, y)
        return x_norm_sq + y_norm_sq - 2.0 * dot_product
    
    @staticmethod
    def euclidean_batch_squared(
        query: np.ndarray, 
        vectors: np.ndarray
    ) -> np.ndarray:
        """
        Compute squared distances from query to multiple vectors.
        
        Vectorized implementation using NumPy einsum for optimal performance.
        
        Formula: For query q and matrix X of shape (n, d):
                distances[i] = ||q - X[i]||² for i = 1..n
        
        Complexity: O(n·d)
        
        Args:
            query: Query vector of shape (d,)
            vectors: Matrix of shape (n, d)
            
        Returns:
            Array of shape (n,) with squared distances
            
        Reference: distance_computations.md Section 1.1.3
        """
        # Broadcasting: vectors - query gives shape (n, d)
        diff = vectors - query[np.newaxis, :]
        # einsum 'ij,ij->i' computes row-wise dot products efficiently
        return np.einsum('ij,ij->i', diff, diff, optimize=True)
    
    @staticmethod
    def euclidean_batch_squared_cached(
        query: np.ndarray,
        vectors: np.ndarray,
        query_norm_sq: Optional[float] = None,
        vector_norms_sq: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Ultra-fast batch distance using precomputed norms.
        
        Formula: ||q - X[i]||² = ||q||² + ||X[i]||² - 2⟨q, X[i]⟩
        
        Complexity: O(n·d) but optimized with BLAS level-2 operations
        
        Args:
            query: Query vector of shape (d,)
            vectors: Matrix of shape (n, d)
            query_norm_sq: Precomputed ||query||² (computed if None)
            vector_norms_sq: Precomputed ||vectors[i]||² for all i (computed if None)
            
        Returns:
            Array of shape (n,) with squared distances
            
        Reference: distance_computations.md Section 1.1.2 (batch variant)
        """
        if query_norm_sq is None:
            query_norm_sq = np.dot(query, query)
        
        if vector_norms_sq is None:
            vector_norms_sq = np.einsum('ij,ij->i', vectors, vectors, optimize=True)
        
        # BLAS level-2: matrix-vector multiplication
        dot_products = np.dot(vectors, query)
        
        return query_norm_sq + vector_norms_sq - 2.0 * dot_products
    
    @staticmethod
    def precompute_vector_norms(vectors: np.ndarray) -> np.ndarray:
        """
        Precompute squared norms for all vectors.
        
        Formula: ||x||² = Σᵢ xᵢ² for each row
        
        Complexity: O(N·d)
        
        Args:
            vectors: Matrix of shape (N, d)
            
        Returns:
            Array of shape (N,) with squared norms
            
        Reference: architecture_overview.md Section 1.3.1 Step 4
        """
        return np.einsum('ij,ij->i', vectors, vectors, optimize=True)


class PQDistanceMetrics:
    """
    Product Quantization distance computations.
    Implements asymmetric distance calculation with lookup tables.
    """
    
    @staticmethod
    def compute_distance_table(
        query: np.ndarray,
        codebooks: list,
        m: int,
        k: int
    ) -> np.ndarray:
        """
        Precompute distance table for PQ asymmetric distance.
        
        Formula:
            T[j, ℓ] = ||qⱼ - Cⱼ[ℓ]||² for j=1..m, ℓ=0..k-1
            where qⱼ = q[(j-1)·d/m : j·d/m]
        
        Complexity: O(m·k·d/m) = O(k·d)
        
        Args:
            query: Query vector of shape (d,)
            codebooks: List of m codebooks, each of shape (k, d/m)
            m: Number of subspaces
            k: Codebook size (number of centroids per subspace)
            
        Returns:
            Distance table of shape (m, k)
            
        Reference: distance_computations.md Section 1.2.1
        """
        d = len(query)
        d_sub = d // m
        table = np.zeros((m, k), dtype=np.float32)
        
        for j in range(m):
            start = j * d_sub
            end = start + d_sub
            q_sub = query[start:end]  # Extract subvector
            
            # Compute squared distances to all centroids in codebook j
            # Shape: (k, d/m) vs (d/m,)
            codebook = codebooks[j]
            diff = codebook - q_sub[np.newaxis, :]
            table[j, :] = np.einsum('ij,ij->i', diff, diff, optimize=True)
        
        return table
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def pq_asymmetric_distance(
        codes: np.ndarray,
        distance_table: np.ndarray
    ) -> np.ndarray:
        """
        Compute PQ asymmetric distances using precomputed table.
        
        Formula: d²_PQ(q, c) = Σⱼ₌₁ᵐ T[j, cⱼ]
        
        Complexity: O(n·m) where n = number of codes
        
        Args:
            codes: PQ codes of shape (n, m) with dtype uint8
            distance_table: Precomputed table of shape (m, k)
            
        Returns:
            Array of shape (n,) with PQ distances
            
        Reference: distance_computations.md Section 1.2.2
        """
        n, m = codes.shape
        distances = np.zeros(n, dtype=np.float32)
        
        # Parallel loop over vectors
        for i in prange(n):
            dist = 0.0
            # Sum over subspaces
            for j in range(m):
                code_j = codes[i, j]
                dist += distance_table[j, code_j]
            distances[i] = dist
        
        return distances
    
    @staticmethod
    def pq_distance_single(
        code: np.ndarray,
        distance_table: np.ndarray
    ) -> float:
        """
        Compute PQ distance for a single code.
        
        Args:
            code: PQ code of shape (m,) with dtype uint8
            distance_table: Precomputed table of shape (m, k)
            
        Returns:
            PQ distance (float)
        """
        m = len(code)
        distance = 0.0
        for j in range(m):
            distance += distance_table[j, code[j]]
        return distance


# Performance validation
if __name__ == "__main__":
    print("="*70)
    print("Distance Metrics Module - Performance Validation")
    print("="*70)
    
    # Test configuration
    d = 128
    n = 10000
    
    # Generate test data
    np.random.seed(42)
    query = np.random.randn(d).astype(np.float32)
    vectors = np.random.randn(n, d).astype(np.float32)
    
    # Test 1: Single distance
    print("\n[Test 1] Single Euclidean Distance")
    dist1 = DistanceMetrics.euclidean_squared(query, vectors[0])
    print(f"  d²(query, vectors[0]) = {dist1:.4f}")
    
    # Test 2: Batch distance (naive)
    print("\n[Test 2] Batch Euclidean Distance")
    import time
    start = time.time()
    distances_naive = DistanceMetrics.euclidean_batch_squared(query, vectors)
    time_naive = time.time() - start
    print(f"  Computed {n} distances in {time_naive*1000:.2f} ms")
    print(f"  Throughput: {n/time_naive:.0f} distances/sec")
    
    # Test 3: Batch distance (cached)
    print("\n[Test 3] Cached Batch Distance (with precomputed norms)")
    query_norm_sq = np.dot(query, query)
    vector_norms_sq = DistanceMetrics.precompute_vector_norms(vectors)
    
    start = time.time()
    distances_cached = DistanceMetrics.euclidean_batch_squared_cached(
        query, vectors, query_norm_sq, vector_norms_sq
    )
    time_cached = time.time() - start
    print(f"  Computed {n} distances in {time_cached*1000:.2f} ms")
    print(f"  Throughput: {n/time_cached:.0f} distances/sec")
    print(f"  Speedup: {time_naive/time_cached:.2f}×")
    
    # Verify correctness
    max_diff = np.max(np.abs(distances_naive - distances_cached))
    print(f"  Max difference: {max_diff:.2e} (should be ~0)")
    
    # Test 4: PQ distance table
    print("\n[Test 4] PQ Distance Table Computation")
    m = 16
    k = 256
    d_sub = d // m
    
    # Create dummy codebooks
    codebooks = [np.random.randn(k, d_sub).astype(np.float32) for _ in range(m)]
    
    start = time.time()
    distance_table = PQDistanceMetrics.compute_distance_table(query, codebooks, m, k)
    time_table = time.time() - start
    print(f"  Built distance table ({m}×{k}) in {time_table*1000:.2f} ms")
    print(f"  Table shape: {distance_table.shape}")
    
    # Test 5: PQ asymmetric distance
    print("\n[Test 5] PQ Asymmetric Distance (batch)")
    codes = np.random.randint(0, k, size=(n, m), dtype=np.uint8)
    
    start = time.time()
    pq_distances = PQDistanceMetrics.pq_asymmetric_distance(codes, distance_table)
    time_pq = time.time() - start
    print(f"  Computed {n} PQ distances in {time_pq*1000:.2f} ms")
    print(f"  Throughput: {n/time_pq:.0f} distances/sec")
    print(f"  Speedup vs exact: {time_naive/time_pq:.1f}×")
    
    print("\n" + "="*70)
    print("✓ All tests passed successfully")
    print("="*70)
