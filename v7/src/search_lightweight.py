"""
Lightweight optimized search - no threading overhead for small datasets.

This version focuses on:
1. Avoiding threading overhead
2. Minimal conversions
3. Smart use of Numba only where beneficial
"""

import numpy as np
from typing import Tuple, List, Optional

try:
    from .core.distances import DistanceMetrics
    from .core.hnsw_wrapper import HNSWGraphManager
    from .core.product_quantizer import ProductQuantizer
except ImportError:
    from core.distances import DistanceMetrics
    from core.hnsw_wrapper import HNSWGraphManager
    from core.product_quantizer import ProductQuantizer


class ZGQSearchLightweight:
    """
    Lightweight ZGQ search without threading overhead.
    
    Better for small-medium datasets where threading overhead dominates.
    """
    
    def __init__(
        self,
        centroids: np.ndarray,
        hnsw_manager: HNSWGraphManager,
        vectors: np.ndarray,
        pq: Optional[ProductQuantizer] = None,
        pq_codes: Optional[np.ndarray] = None,
        vector_norms: Optional[np.ndarray] = None,
        verbose: bool = False
    ):
        """Initialize lightweight search."""
        # Store as float32 for consistency
        self.centroids = centroids.astype(np.float32, copy=False)
        self.vectors = vectors.astype(np.float32, copy=False)
        
        self.hnsw_manager = hnsw_manager
        self.pq = pq
        self.pq_codes = pq_codes
        self.vector_norms = vector_norms
        self.verbose = verbose
        
        self.use_pq = (pq is not None) and (pq_codes is not None)
    
    def search(self, query: np.ndarray, k: int = 10, n_probe: int = 10,
               ef_search: int = 50, k_rerank: Optional[int] = None,
               use_parallel: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Parameters:
        -----------
        query : np.ndarray
            Query vector
        k : int
            Number of neighbors to return
        n_probe : int
            Number of zones to search
        ef_search : int
            HNSW search parameter (used in zone search)
        k_rerank : int, optional
            Number of candidates to re-rank (if None, uses adaptive strategy)
        use_parallel : bool
            Whether to use parallel processing (ignored in lightweight version - always False)
            
        Returns:
        --------
        indices : np.ndarray
            Indices of k nearest neighbors
        distances : np.ndarray
            Distances to k nearest neighbors
        """
        # Note: use_parallel is ignored - lightweight version is always sequential
        query = query.astype(np.float32)
        n_centroids = len(self.centroids)
        
        if k_rerank is None:
            # Adaptive k_rerank based on n_probe
            k_rerank = min(k * max(n_probe, 10), len(self.vectors))
        
        # Step 1: Zone selection
        selected_zones = self._select_zones(query, n_probe)
        
        # Step 2: Prepare PQ distance table if using PQ
        pq_distance_table = None
        if self.use_pq:
            pq_distance_table = self._compute_pq_table(query)
        
        # Step 3: Search zones sequentially
        all_candidates = []
        for zone_id in selected_zones:
            zone_candidates = self._search_zone(
                zone_id, query, k, ef_search, pq_distance_table
            )
            all_candidates.extend(zone_candidates)
        
        # Step 4: Aggregate and deduplicate
        candidate_ids = self._aggregate(all_candidates, k_rerank)
        
        # Step 5: Re-rank with exact distances
        final_ids, final_distances = self._rerank(query, candidate_ids, k)
        
        return final_ids, final_distances
    
    def _select_zones(self, query: np.ndarray, n_probe: int) -> np.ndarray:
        """Simple NumPy zone selection."""
        diff = self.centroids - query
        distances = np.sum(diff * diff, axis=1)
        
        if n_probe >= len(distances):
            return np.arange(len(distances))
        
        # argpartition is O(n), faster than full sort
        indices = np.argpartition(distances, n_probe)[:n_probe]
        return indices[np.argsort(distances[indices])]
    
    def _compute_pq_table(self, query: np.ndarray) -> np.ndarray:
        """Compute PQ distance table."""
        m = self.pq.m
        k = self.pq.k
        d_sub = self.pq.subvector_dim
        
        distance_table = np.zeros((m, k), dtype=np.float32)
        
        for j in range(m):
            start = j * d_sub
            end = start + d_sub
            query_sub = query[start:end]
            
            # Vectorized distance computation
            diff = self.pq.codebooks[j] - query_sub
            distance_table[j] = np.sum(diff * diff, axis=1)
        
        return distance_table
    
    def _search_zone(
        self,
        zone_id: int,
        query: np.ndarray,
        k: int,
        ef_search: Optional[int],
        pq_distance_table: Optional[np.ndarray]
    ) -> List[Tuple[int, float]]:
        """Search single zone."""
        # HNSW search
        global_ids, _ = self.hnsw_manager.search_zone(zone_id, query, k, ef_search)
        
        if len(global_ids) == 0:
            return []
        
        # Distance computation
        if self.use_pq and pq_distance_table is not None:
            # PQ asymmetric
            codes = self.pq_codes[global_ids]
            distances = np.sum(pq_distance_table[np.arange(self.pq.m)[:, None], codes.T], axis=0)
        else:
            # Exact distances
            diff = self.vectors[global_ids] - query
            distances = np.sum(diff * diff, axis=1)
        
        return list(zip(global_ids, distances))
    
    def _aggregate(
        self,
        candidates: List[Tuple[int, float]],
        k_rerank: int
    ) -> np.ndarray:
        """Aggregate candidates efficiently."""
        if not candidates:
            return np.array([], dtype=np.int32)
        
        # Deduplicate
        best_distances = {}
        for vid, dist in candidates:
            if vid not in best_distances or dist < best_distances[vid]:
                best_distances[vid] = dist
        
        if len(best_distances) <= k_rerank:
            # Return all sorted
            sorted_items = sorted(best_distances.items(), key=lambda x: x[1])
            return np.array([vid for vid, _ in sorted_items], dtype=np.int32)
        
        # Select top k_rerank
        ids = np.array(list(best_distances.keys()), dtype=np.int32)
        distances = np.array([best_distances[vid] for vid in ids], dtype=np.float32)
        
        # argpartition is O(n)
        top_indices = np.argpartition(distances, k_rerank)[:k_rerank]
        top_indices = top_indices[np.argsort(distances[top_indices])]
        
        return ids[top_indices]
    
    def _rerank(
        self,
        query: np.ndarray,
        candidate_ids: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Re-rank with exact distances."""
        if len(candidate_ids) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        # Batch distance computation
        candidate_vectors = self.vectors[candidate_ids]
        diff = candidate_vectors - query
        exact_distances = np.sum(diff * diff, axis=1)
        
        # Select top k
        k_actual = min(k, len(exact_distances))
        top_k_indices = np.argpartition(exact_distances, k_actual)[:k_actual]
        top_k_indices = top_k_indices[np.argsort(exact_distances[top_k_indices])]
        
        final_ids = candidate_ids[top_k_indices]
        final_distances = np.sqrt(exact_distances[top_k_indices])
        
        return final_ids, final_distances
