"""
Optimized Distance Computation for ZGQ v8
==========================================

This module provides high-performance distance computation using:
1. SIMD acceleration via NumPy/Numba
2. Early termination for bound checking
3. Batch processing for vectorization
4. Precomputed norms for efficiency

Performance Targets:
- 4x speedup from vectorization/SIMD
- 2x average speedup from early termination
- Cache-friendly memory access patterns
"""

import numpy as np
from typing import Optional, Tuple
from numba import njit, prange, float32, int32, boolean
import warnings


# Check for AVX support (informational)
def _check_simd_support() -> dict:
    """Check available SIMD instruction sets."""
    import platform
    
    info = {
        'platform': platform.machine(),
        'numpy_simd': hasattr(np, '__config__')
    }
    
    # Numba handles SIMD automatically
    try:
        from numba import config
        info['numba_parallel'] = config.NUMBA_NUM_THREADS
    except:
        info['numba_parallel'] = 1
    
    return info


class DistanceComputer:
    """
    High-performance distance computation engine.
    
    Provides optimized implementations of:
    - Squared Euclidean distance
    - Cosine similarity
    - Inner product
    
    With features:
    - Batch computation
    - Early termination
    - Precomputed norms
    """
    
    def __init__(self, metric: str = 'l2'):
        """
        Initialize distance computer.
        
        Args:
            metric: Distance metric ('l2', 'cosine', 'ip')
        """
        self.metric = metric
        self._norm_cache: Optional[np.ndarray] = None
        
    def compute(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        precomputed_norms: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute distances from query to multiple vectors.
        
        Args:
            query: Query vector of shape (d,)
            vectors: Database vectors of shape (N, d)
            precomputed_norms: Optional precomputed ||vectors||² of shape (N,)
            
        Returns:
            Distances of shape (N,)
        """
        if self.metric == 'l2':
            return self._euclidean_squared(query, vectors, precomputed_norms)
        elif self.metric == 'cosine':
            return self._cosine_distance(query, vectors, precomputed_norms)
        elif self.metric == 'ip':
            return self._inner_product_distance(query, vectors)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _euclidean_squared(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        precomputed_norms: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute squared Euclidean distances.
        
        Uses the identity: ||a - b||² = ||a||² + ||b||² - 2<a,b>
        """
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)
        
        # Query norm
        query_norm = np.dot(query, query)
        
        # Vector norms (use precomputed if available)
        if precomputed_norms is not None:
            vector_norms = precomputed_norms
        else:
            vector_norms = np.sum(vectors * vectors, axis=1)
        
        # Inner products
        inner_products = vectors @ query
        
        # Squared distances
        distances = query_norm + vector_norms - 2 * inner_products
        
        # Clip negative values (numerical precision)
        np.maximum(distances, 0, out=distances)
        
        return distances.astype(np.float32)
    
    def _cosine_distance(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        precomputed_norms: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute cosine distances (1 - cosine_similarity)."""
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)
        
        # Normalize query
        query_norm = np.sqrt(np.dot(query, query))
        if query_norm > 0:
            query = query / query_norm
        
        # Vector norms
        if precomputed_norms is not None:
            vector_norms = np.sqrt(precomputed_norms)
        else:
            vector_norms = np.sqrt(np.sum(vectors * vectors, axis=1))
        
        # Inner products
        inner_products = vectors @ query
        
        # Cosine similarity
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine_sim = inner_products / vector_norms
            cosine_sim = np.nan_to_num(cosine_sim, nan=0.0)
        
        # Convert to distance
        return (1 - cosine_sim).astype(np.float32)
    
    def _inner_product_distance(
        self,
        query: np.ndarray,
        vectors: np.ndarray
    ) -> np.ndarray:
        """Compute inner product distance (negative inner product)."""
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)
        
        # Negative inner product (so smaller = better)
        return (-vectors @ query).astype(np.float32)
    
    def compute_single(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> float:
        """Compute distance between two vectors."""
        if self.metric == 'l2':
            diff = a - b
            return float(np.dot(diff, diff))
        elif self.metric == 'cosine':
            norm_a = np.sqrt(np.dot(a, a))
            norm_b = np.sqrt(np.dot(b, b))
            if norm_a > 0 and norm_b > 0:
                return float(1 - np.dot(a, b) / (norm_a * norm_b))
            return 1.0
        elif self.metric == 'ip':
            return float(-np.dot(a, b))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    @staticmethod
    def precompute_norms(vectors: np.ndarray) -> np.ndarray:
        """Precompute squared L2 norms for all vectors."""
        vectors = vectors.astype(np.float32)
        return np.sum(vectors * vectors, axis=1).astype(np.float32)


class SIMDDistance:
    """
    SIMD-optimized distance functions using Numba JIT.
    
    These functions are compiled to native code with automatic
    SIMD vectorization for maximum performance.
    """
    
    @staticmethod
    @njit(float32[:](float32[:], float32[:,:]), fastmath=True, parallel=True, cache=True)
    def euclidean_squared_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances (SIMD-optimized).
        
        Args:
            query: Query vector of shape (d,)
            vectors: Database vectors of shape (N, d)
            
        Returns:
            Squared distances of shape (N,)
        """
        n = vectors.shape[0]
        d = vectors.shape[1]
        result = np.empty(n, dtype=np.float32)
        
        for i in prange(n):
            dist = 0.0
            for j in range(d):
                diff = query[j] - vectors[i, j]
                dist += diff * diff
            result[i] = dist
        
        return result
    
    @staticmethod
    @njit(float32(float32[:], float32[:]), fastmath=True, cache=True)
    def euclidean_squared_single(a: np.ndarray, b: np.ndarray) -> float:
        """Compute squared Euclidean distance between two vectors."""
        d = a.shape[0]
        dist = 0.0
        for i in range(d):
            diff = a[i] - b[i]
            dist += diff * diff
        return dist
    
    @staticmethod
    @njit(float32(float32[:], float32[:], float32), fastmath=True, cache=True)
    def euclidean_squared_early_term(
        a: np.ndarray,
        b: np.ndarray,
        bound: float
    ) -> float:
        """
        Compute squared Euclidean distance with early termination.
        
        Stops computation if distance exceeds bound.
        Returns bound + 1 if early terminated.
        
        Args:
            a: First vector
            b: Second vector
            bound: Distance threshold
            
        Returns:
            Distance if <= bound, else bound + 1
        """
        d = a.shape[0]
        dist = 0.0
        
        # Check every 8 dimensions for early termination
        for i in range(d):
            diff = a[i] - b[i]
            dist += diff * diff
            
            if i % 8 == 7 and dist > bound:
                return bound + 1.0
        
        return dist
    
    @staticmethod
    @njit(float32[:](float32[:,:], float32[:]), fastmath=True, parallel=True, cache=True)
    def compute_norms_batch(vectors: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Compute squared L2 norms for all vectors."""
        n = vectors.shape[0]
        d = vectors.shape[1]
        
        for i in prange(n):
            norm = 0.0
            for j in range(d):
                norm += vectors[i, j] * vectors[i, j]
            out[i] = norm
        
        return out


class EarlyTerminationComputer:
    """
    Distance computation with bound-based early termination.
    
    Useful for k-NN search where we can skip candidates
    that are clearly worse than current k-th best.
    """
    
    def __init__(self, dimension: int):
        """
        Initialize early termination computer.
        
        Args:
            dimension: Vector dimension
        """
        self.dimension = dimension
        self._check_interval = max(8, dimension // 16)
    
    def compute_with_bound(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        indices: np.ndarray,
        bound: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute distances with early termination.
        
        Args:
            query: Query vector
            vectors: Database vectors
            indices: Indices to compute
            bound: Distance threshold
            
        Returns:
            (valid_indices, distances): Indices and distances within bound
        """
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)
        
        valid_indices = []
        valid_distances = []
        
        for idx in indices:
            dist = self._compute_with_bound_single(
                query, vectors[idx], bound
            )
            if dist <= bound:
                valid_indices.append(idx)
                valid_distances.append(dist)
        
        return np.array(valid_indices), np.array(valid_distances)
    
    def _compute_with_bound_single(
        self,
        a: np.ndarray,
        b: np.ndarray,
        bound: float
    ) -> float:
        """Compute distance with early termination."""
        return float(SIMDDistance.euclidean_squared_early_term(
            a.astype(np.float32),
            b.astype(np.float32),
            np.float32(bound)
        ))


def select_top_k_distances(
    distances: np.ndarray,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select top-k smallest distances efficiently.
    
    Uses argpartition for O(N) complexity instead of O(N log N) sort.
    
    Args:
        distances: Array of distances
        k: Number of smallest to select
        
    Returns:
        (indices, distances): Top-k indices and their distances
    """
    n = len(distances)
    k = min(k, n)
    
    if k == n:
        indices = np.argsort(distances)
        return indices, distances[indices]
    
    # O(N) partitioning
    top_k_indices = np.argpartition(distances, k - 1)[:k]
    
    # Sort only the top k
    sorted_local = np.argsort(distances[top_k_indices])
    top_k_indices = top_k_indices[sorted_local]
    
    return top_k_indices, distances[top_k_indices]


# Warmup JIT compilation
def _warmup_jit():
    """Pre-compile Numba functions."""
    try:
        dummy_q = np.zeros(128, dtype=np.float32)
        dummy_v = np.zeros((10, 128), dtype=np.float32)
        dummy_out = np.zeros(10, dtype=np.float32)
        
        _ = SIMDDistance.euclidean_squared_batch(dummy_q, dummy_v)
        _ = SIMDDistance.euclidean_squared_single(dummy_q, dummy_q)
        _ = SIMDDistance.euclidean_squared_early_term(dummy_q, dummy_q, 1.0)
        _ = SIMDDistance.compute_norms_batch(dummy_v, dummy_out)
    except Exception:
        pass


# Run warmup on import
_warmup_jit()
