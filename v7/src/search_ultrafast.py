"""
Ultra-Fast ZGQ Search with Fast Zone Selection

This version uses HNSW on centroids for logarithmic-time zone selection,
which provides the biggest performance improvement.
"""

import numpy as np
from typing import Tuple, List, Optional
import time

try:
    from .core.hnsw_wrapper import HNSWGraphManager
    from .core.product_quantizer import ProductQuantizer
    from .core.fast_zone_selector import HybridZoneSelector
except ImportError:
    from core.hnsw_wrapper import HNSWGraphManager
    from core.product_quantizer import ProductQuantizer
    from core.fast_zone_selector import HybridZoneSelector


class ZGQSearchUltraFast:
    """
    Ultra-fast ZGQ search with optimized zone selection.
    
    Key optimizations:
    1. HNSW on centroids for O(log n) zone selection
    2. Sequential zone search (no threading overhead)
    3. Efficient aggregation with argpartition
    4. Batch distance computation
    """
    
    def __init__(
        self,
        centroids: np.ndarray,
        hnsw_manager: HNSWGraphManager,
        vectors: np.ndarray,
        pq: Optional[ProductQuantizer] = None,
        pq_codes: Optional[np.ndarray] = None,
        vector_norms: Optional[np.ndarray] = None,
        use_fast_selector: bool = True,
        verbose: bool = False
    ):
        """
        Initialize ultra-fast search.
        
        Parameters:
        -----------
        centroids : np.ndarray
            Zone centroids
        hnsw_manager : HNSWGraphManager
            HNSW index manager
        vectors : np.ndarray
            Full vectors
        pq : ProductQuantizer, optional
            Product quantizer
        pq_codes : np.ndarray, optional
            PQ codes
        vector_norms : np.ndarray, optional
            Precomputed vector norms
        use_fast_selector : bool
            Use HNSW on centroids (True = faster)
        verbose : bool
            Debug output
        """
        self.vectors = vectors.astype(np.float32, copy=False)
        self.hnsw_manager = hnsw_manager
        self.pq = pq
        self.pq_codes = pq_codes
        self.verbose = verbose
        
        # Fast zone selector (BIGGEST OPTIMIZATION)
        self.zone_selector = HybridZoneSelector(
            centroids=centroids,
            threshold=50,  # Use HNSW if > 50 zones
            use_hnsw=use_fast_selector,
            verbose=verbose
        )
        
        # Precompute or compute vector norms
        if vector_norms is None:
            self.vector_norms = np.sum(vectors * vectors, axis=1).astype(np.float32)
        else:
            self.vector_norms = vector_norms.astype(np.float32, copy=False)
        
        self.use_pq = (pq is not None) and (pq_codes is not None)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = 10,
        ef_search: int = 50,
        k_rerank: Optional[int] = None,
        use_parallel: bool = False  # Ignored - always sequential
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            HNSW search parameter
        k_rerank : int, optional
            Number of candidates to re-rank
        use_parallel : bool
            Ignored (always sequential for best performance)
            
        Returns:
        --------
        indices : np.ndarray
            Indices of k nearest neighbors
        distances : np.ndarray
            Distances to k nearest neighbors
        """
        query = query.astype(np.float32)
        
        if k_rerank is None:
            # Adaptive k_rerank
            k_rerank = min(k * max(n_probe, 10), len(self.vectors))
        
        # Step 1: Fast zone selection (OPTIMIZED!)
        selected_zones = self.zone_selector.select_zones(query, n_probe)
        
        # Step 2: Prepare PQ table if using PQ
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
        
        # Step 4: Aggregate
        candidate_ids = self._aggregate(all_candidates, k_rerank)
        
        # Step 5: Re-rank
        final_ids, final_distances = self._rerank(query, candidate_ids, k)
        
        return final_ids, final_distances
    
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
            distances = np.sum(
                pq_distance_table[np.arange(self.pq.m)[:, None], codes.T], 
                axis=0
            )
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
        
        if k_actual >= len(exact_distances):
            # Return all sorted
            top_k_indices = np.argsort(exact_distances)
        else:
            # Use argpartition for efficiency
            top_k_indices = np.argpartition(exact_distances, k_actual-1)[:k_actual]
            top_k_indices = top_k_indices[np.argsort(exact_distances[top_k_indices])]
        
        final_ids = candidate_ids[top_k_indices]
        final_distances = np.sqrt(exact_distances[top_k_indices])
        
        return final_ids, final_distances
