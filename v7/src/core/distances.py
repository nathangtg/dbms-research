"""
Distance computation functions for ZGQ.

This module provides optimized distance computations for:
- Standard Euclidean distance
- PQ asymmetric distance
- Batch distance computations
"""

import numpy as np
from typing import Optional
from numba import njit, prange


class DistanceMetrics:
    """Standard distance computations."""
    
    @staticmethod
    @njit(fastmath=True)
    def euclidean_squared(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute squared Euclidean distance between two vectors.
        
        Args:
            x: Vector of shape (d,)
            y: Vector of shape (d,)
            
        Returns:
            Squared Euclidean distance
        """
        dist = 0.0
        for i in range(x.shape[0]):
            diff = x[i] - y[i]
            dist += diff * diff
        return dist
    
    @staticmethod
    def euclidean_batch_squared(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute squared Euclidean distances from query to multiple vectors.
        
        Uses vectorized numpy operations for efficiency.
        
        Args:
            query: Query vector of shape (d,)
            vectors: Matrix of shape (N, d)
            
        Returns:
            Array of squared distances of shape (N,)
        """
        # Vectorized computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        # But simpler: sum((query - vectors)**2, axis=1)
        diff = vectors - query
        return np.sum(diff * diff, axis=1).astype(np.float32)
    
    @staticmethod
    @njit(fastmath=True)
    def euclidean_squared_optimized(x: np.ndarray, y: np.ndarray, 
                                   norm_x: float, norm_y: float) -> float:
        """
        Compute squared Euclidean distance using precomputed norms.
        
        Using the identity: ||x - y||² = ||x||² + ||y||² - 2·x·y
        
        Args:
            x: Vector of shape (d,)
            y: Vector of shape (d,)
            norm_x: Precomputed ||x||²
            norm_y: Precomputed ||y||²
            
        Returns:
            Squared Euclidean distance
        """
        dot_product = 0.0
        for i in range(x.shape[0]):
            dot_product += x[i] * y[i]
        
        return norm_x + norm_y - 2.0 * dot_product
    
    @staticmethod
    @njit(fastmath=True)
    def compute_norm_squared(x: np.ndarray) -> float:
        """
        Compute squared L2 norm of a vector.
        
        Args:
            x: Vector of shape (d,)
            
        Returns:
            Squared L2 norm
        """
        norm = 0.0
        for i in range(x.shape[0]):
            norm += x[i] * x[i]
        return norm
    
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def compute_norms_batch(vectors: np.ndarray) -> np.ndarray:
        """
        Compute squared L2 norms for multiple vectors.
        
        Args:
            vectors: Matrix of shape (N, d)
            
        Returns:
            Array of squared norms of shape (N,)
        """
        n = vectors.shape[0]
        d = vectors.shape[1]
        norms = np.empty(n, dtype=np.float32)
        
        for i in prange(n):
            norm = 0.0
            for j in range(d):
                norm += vectors[i, j] * vectors[i, j]
            norms[i] = norm
        
        return norms


class PQDistanceMetrics:
    """Product Quantization distance computations."""
    
    @staticmethod
    @njit(fastmath=True)
    def pq_asymmetric_distance(
        pq_codes: np.ndarray,
        distance_table: np.ndarray
    ) -> float:
        """
        Compute PQ asymmetric distance using precomputed distance table.
        
        Args:
            pq_codes: PQ codes for one vector of shape (m,) with dtype uint8
            distance_table: Precomputed distance table of shape (m, k) where
                           distance_table[j, c] = ||query_j - centroid_{j,c}||²
            
        Returns:
            Approximate squared distance
        """
        m = pq_codes.shape[0]
        distance = 0.0
        
        for j in range(m):
            code = pq_codes[j]
            distance += distance_table[j, code]
        
        return distance
    
    @staticmethod
    @njit(fastmath=True, parallel=True)
    def pq_asymmetric_distance_batch(
        pq_codes: np.ndarray,
        distance_table: np.ndarray
    ) -> np.ndarray:
        """
        Compute PQ asymmetric distances for multiple vectors.
        
        Args:
            pq_codes: PQ codes of shape (N, m) with dtype uint8
            distance_table: Precomputed distance table of shape (m, k)
            
        Returns:
            Array of approximate squared distances of shape (N,)
        """
        n = pq_codes.shape[0]
        m = pq_codes.shape[1]
        distances = np.empty(n, dtype=np.float32)
        
        for i in prange(n):
            distance = 0.0
            for j in range(m):
                code = pq_codes[i, j]
                distance += distance_table[j, code]
            distances[i] = distance
        
        return distances
    
    @staticmethod
    @njit(fastmath=True)
    def compute_pq_distance_table(
        query: np.ndarray,
        codebooks: np.ndarray,
        m: int,
        subvector_dim: int
    ) -> np.ndarray:
        """
        Precompute distance table for PQ asymmetric distance computation.
        
        Args:
            query: Query vector of shape (d,)
            codebooks: All codebooks of shape (m, k, d/m)
            m: Number of subspaces
            subvector_dim: Dimension of each subvector (d/m)
            
        Returns:
            Distance table of shape (m, k) where entry [j, c] contains
            ||query_j - centroid_{j,c}||²
        """
        k = codebooks.shape[1]  # Number of centroids per subspace
        distance_table = np.empty((m, k), dtype=np.float32)
        
        for j in range(m):
            # Extract query subvector for subspace j
            start_idx = j * subvector_dim
            end_idx = start_idx + subvector_dim
            
            for c in range(k):
                # Compute distance to centroid c in subspace j
                dist = 0.0
                for i in range(subvector_dim):
                    diff = query[start_idx + i] - codebooks[j, c, i]
                    dist += diff * diff
                distance_table[j, c] = dist
        
        return distance_table


class DistanceUtils:
    """Utility functions for distance computations."""
    
    @staticmethod
    def compute_centroid_distances(query: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute distances from query to all centroids.
        
        Args:
            query: Query vector of shape (d,)
            centroids: Centroid matrix of shape (n_centroids, d)
            
        Returns:
            Array of distances of shape (n_centroids,)
        """
        return DistanceMetrics.euclidean_batch_squared(query, centroids)
    
    @staticmethod
    def select_nearest_zones(
        query: np.ndarray,
        centroids: np.ndarray,
        n_probe: int
    ) -> np.ndarray:
        """
        Select n_probe nearest zones based on centroid distances.
        
        Args:
            query: Query vector of shape (d,)
            centroids: Centroid matrix of shape (n_zones, d)
            n_probe: Number of zones to select
            
        Returns:
            Array of zone indices of shape (n_probe,)
        """
        distances = DistanceUtils.compute_centroid_distances(query, centroids)
        # Use argpartition for efficient k-nearest selection
        if n_probe >= len(distances):
            return np.arange(len(distances))
        
        nearest_indices = np.argpartition(distances, n_probe - 1)[:n_probe]
        # Sort by distance for consistent ordering
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        return nearest_indices
    
    @staticmethod
    def precompute_vector_norms(vectors: np.ndarray) -> np.ndarray:
        """
        Precompute squared L2 norms for all vectors.
        
        Args:
            vectors: Matrix of shape (N, d)
            
        Returns:
            Array of squared norms of shape (N,)
        """
        return DistanceMetrics.compute_norms_batch(vectors)
