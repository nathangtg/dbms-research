"""
Optimized distance metrics using Numba JIT compilation.

This module provides significant performance improvements over pure NumPy
by using Numba's JIT compiler for critical distance computations.
"""

import numpy as np
import numba
from numba import njit, prange


@njit(fastmath=True, cache=True)
def euclidean_distance_squared(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute squared Euclidean distance between two vectors.
    
    Args:
        a: Vector of shape (d,)
        b: Vector of shape (d,)
        
    Returns:
        Squared Euclidean distance
    """
    result = 0.0
    for i in range(a.shape[0]):
        diff = a[i] - b[i]
        result += diff * diff
    return result


@njit(fastmath=True, parallel=True, cache=True)
def euclidean_batch_squared_numba(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Compute squared Euclidean distances from query to multiple vectors.
    
    Uses parallel execution for better performance on multi-core systems.
    
    Args:
        query: Query vector of shape (d,)
        vectors: Array of vectors of shape (n, d)
        
    Returns:
        Array of squared distances of shape (n,)
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    distances = np.empty(n, dtype=np.float32)
    
    for i in prange(n):
        dist = 0.0
        for j in range(d):
            diff = query[j] - vectors[i, j]
            dist += diff * diff
        distances[i] = dist
    
    return distances


@njit(fastmath=True, cache=True)
def select_top_k_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """
    Select indices of k smallest distances.
    
    Uses argpartition for O(n) complexity instead of O(n log n) sort.
    
    Args:
        distances: Array of distances
        k: Number of smallest to select
        
    Returns:
        Indices of k smallest distances, sorted
    """
    n = len(distances)
    if k >= n:
        # Return all indices sorted
        return np.argsort(distances)
    
    # Partition to get k smallest
    indices = np.argpartition(distances, k)[:k]
    
    # Sort only the k selected
    sorted_indices = indices[np.argsort(distances[indices])]
    
    return sorted_indices


@njit(fastmath=True, parallel=True, cache=True)
def pq_asymmetric_distance_numba(
    codes: np.ndarray,
    distance_table: np.ndarray
) -> np.ndarray:
    """
    Compute PQ asymmetric distances using lookup table.
    
    Optimized with Numba for fast execution.
    
    Args:
        codes: PQ codes of shape (n, m) - n vectors, m subquantizers
        distance_table: Distance lookup table of shape (m, k) where k is codebook size
        
    Returns:
        Distances of shape (n,)
    """
    n = codes.shape[0]
    m = codes.shape[1]
    distances = np.zeros(n, dtype=np.float32)
    
    for i in prange(n):
        dist = 0.0
        for j in range(m):
            code = codes[i, j]
            dist += distance_table[j, code]
        distances[i] = dist
    
    return distances


@njit(fastmath=True, cache=True)
def compute_pq_distance_table_numba(
    query: np.ndarray,
    codebooks: np.ndarray
) -> np.ndarray:
    """
    Compute PQ distance table for a query.
    
    Args:
        query: Query vector of shape (d,)
        codebooks: Codebooks of shape (m, k, d/m)
        
    Returns:
        Distance table of shape (m, k)
    """
    m = codebooks.shape[0]
    k = codebooks.shape[1]
    d_sub = codebooks.shape[2]
    
    distance_table = np.empty((m, k), dtype=np.float32)
    
    for j in range(m):
        start = j * d_sub
        end = start + d_sub
        query_sub = query[start:end]
        
        for i in range(k):
            centroid = codebooks[j, i]
            dist = 0.0
            for l in range(d_sub):
                diff = query_sub[l] - centroid[l]
                dist += diff * diff
            distance_table[j, i] = dist
    
    return distance_table


@njit(fastmath=True, parallel=True, cache=True)
def select_nearest_zones_numba(
    query: np.ndarray,
    centroids: np.ndarray,
    n_probe: int
) -> np.ndarray:
    """
    Select n_probe nearest zones using parallel distance computation.
    
    This is a critical optimization for ZGQ zone selection.
    
    Args:
        query: Query vector of shape (d,)
        centroids: Zone centroids of shape (n_zones, d)
        n_probe: Number of zones to select
        
    Returns:
        Indices of n_probe nearest zones
    """
    n_zones = centroids.shape[0]
    d = centroids.shape[1]
    distances = np.empty(n_zones, dtype=np.float32)
    
    # Compute distances to all centroids in parallel
    for i in prange(n_zones):
        dist = 0.0
        for j in range(d):
            diff = query[j] - centroids[i, j]
            dist += diff * diff
        distances[i] = dist
    
    # Select n_probe nearest
    if n_probe >= n_zones:
        return np.arange(n_zones)
    
    indices = np.argpartition(distances, n_probe)[:n_probe]
    return indices[np.argsort(distances[indices])]


@njit(fastmath=True, cache=True)
def compute_norms_squared_numba(vectors: np.ndarray) -> np.ndarray:
    """
    Compute squared L2 norms for all vectors.
    
    Args:
        vectors: Array of shape (n, d)
        
    Returns:
        Array of norms squared of shape (n,)
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    norms = np.empty(n, dtype=np.float32)
    
    for i in range(n):
        norm = 0.0
        for j in range(d):
            norm += vectors[i, j] * vectors[i, j]
        norms[i] = norm
    
    return norms


class OptimizedDistanceMetrics:
    """
    High-performance distance metrics using Numba JIT.
    
    These methods are significantly faster than pure NumPy implementations
    for the operations critical to ZGQ performance.
    """
    
    @staticmethod
    def euclidean_batch_squared(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Batch Euclidean distance computation with Numba acceleration."""
        # Ensure float32 for better performance
        query = query.astype(np.float32, copy=False)
        vectors = vectors.astype(np.float32, copy=False)
        return euclidean_batch_squared_numba(query, vectors)
    
    @staticmethod
    def select_nearest_zones(query: np.ndarray, centroids: np.ndarray, n_probe: int) -> np.ndarray:
        """Optimized zone selection with parallel computation."""
        query = query.astype(np.float32, copy=False)
        centroids = centroids.astype(np.float32, copy=False)
        return select_nearest_zones_numba(query, centroids, n_probe)
    
    @staticmethod
    def pq_asymmetric_distance(codes: np.ndarray, distance_table: np.ndarray) -> np.ndarray:
        """Optimized PQ asymmetric distance computation."""
        codes = codes.astype(np.int32, copy=False)
        distance_table = distance_table.astype(np.float32, copy=False)
        return pq_asymmetric_distance_numba(codes, distance_table)
    
    @staticmethod
    def compute_pq_distance_table(query: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
        """Optimized PQ distance table computation."""
        query = query.astype(np.float32, copy=False)
        codebooks = codebooks.astype(np.float32, copy=False)
        return compute_pq_distance_table_numba(query, codebooks)
    
    @staticmethod
    def compute_norms_squared(vectors: np.ndarray) -> np.ndarray:
        """Optimized norm computation."""
        vectors = vectors.astype(np.float32, copy=False)
        return compute_norms_squared_numba(vectors)
    
    @staticmethod
    def select_top_k(distances: np.ndarray, k: int) -> np.ndarray:
        """Optimized top-k selection."""
        return select_top_k_indices(distances, k)
