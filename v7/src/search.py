"""
Search algorithms for ZGQ.

This module implements the ZGQ search algorithm including zone selection,
parallel zone search, candidate aggregation, and re-ranking.
"""

import numpy as np
from typing import Tuple, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor
import time

try:
    from .core.distances import DistanceMetrics, PQDistanceMetrics, DistanceUtils
    from .core.hnsw_wrapper import HNSWGraphManager
    from .core.product_quantizer import ProductQuantizer
except ImportError:
    from core.distances import DistanceMetrics, PQDistanceMetrics, DistanceUtils
    from core.hnsw_wrapper import HNSWGraphManager
    from core.product_quantizer import ProductQuantizer


class ZGQSearch:
    """
    ZGQ search algorithm implementation.
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
        """
        Initialize ZGQ search.
        
        Args:
            centroids: Zone centroids of shape (n_zones, d)
            hnsw_manager: HNSW graph manager
            vectors: Original vectors of shape (N, d) for re-ranking
            pq: Product quantizer (if using PQ)
            pq_codes: PQ codes of shape (N, m) (if using PQ)
            vector_norms: Precomputed vector norms (optional)
            verbose: Whether to print debug info
        """
        self.centroids = centroids
        self.hnsw_manager = hnsw_manager
        self.vectors = vectors
        self.pq = pq
        self.pq_codes = pq_codes
        self.vector_norms = vector_norms
        self.verbose = verbose
        
        self.use_pq = (pq is not None) and (pq_codes is not None)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        k_rerank: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors using ZGQ algorithm.
        
        Args:
            query: Query vector of shape (d,)
            k: Number of nearest neighbors to return
            n_probe: Number of zones to search
            ef_search: HNSW search parameter (overrides default)
            k_rerank: Number of candidates to re-rank (default: k * n_probe)
            
        Returns:
            (ids, distances): Arrays of shape (k,) with neighbor IDs and distances
        """
        if k_rerank is None:
            k_rerank = min(k * n_probe, len(self.vectors))
        
        # Step 1: Select zones
        selected_zones = self._select_zones(query, n_probe)
        
        if self.verbose:
            print(f"Selected zones: {selected_zones}")
        
        # Step 2: Precompute PQ distance table (if using PQ)
        pq_distance_table = None
        if self.use_pq:
            pq_distance_table = self.pq.compute_distance_table(query)
        
        # Step 3: Search selected zones in parallel
        candidates = self._parallel_zone_search(
            query, selected_zones, k, ef_search, pq_distance_table
        )
        
        if self.verbose:
            print(f"Found {len(candidates)} candidates from {len(selected_zones)} zones")
        
        # Step 4: Aggregate and deduplicate candidates
        unique_candidates = self._aggregate_candidates(candidates)
        
        if self.verbose:
            print(f"After deduplication: {len(unique_candidates)} unique candidates")
        
        # Step 5: Select top k_rerank candidates for re-ranking
        if len(unique_candidates) > k_rerank:
            top_candidates = self._select_top_candidates(unique_candidates, k_rerank)
        else:
            # Extract just the IDs from unique_candidates
            top_candidates = [vid for vid, _ in unique_candidates]
        
        # Step 6: Re-rank with exact distances
        final_ids, final_distances = self._rerank_exact(query, top_candidates, k)
        
        return final_ids, final_distances
    
    def _select_zones(self, query: np.ndarray, n_probe: int) -> np.ndarray:
        """Select n_probe nearest zones based on centroid distances."""
        return DistanceUtils.select_nearest_zones(query, self.centroids, n_probe)
    
    def _parallel_zone_search(
        self,
        query: np.ndarray,
        zone_ids: np.ndarray,
        k: int,
        ef_search: Optional[int],
        pq_distance_table: Optional[np.ndarray]
    ) -> List[Tuple[int, float]]:
        """
        Search multiple zones in parallel.
        
        Returns:
            List of (vector_id, distance) tuples from all zones
        """
        all_candidates = []
        
        # Search each zone
        for zone_id in zone_ids:
            zone_candidates = self._search_single_zone(
                zone_id, query, k, ef_search, pq_distance_table
            )
            all_candidates.extend(zone_candidates)
        
        return all_candidates
    
    def _search_single_zone(
        self,
        zone_id: int,
        query: np.ndarray,
        k: int,
        ef_search: Optional[int],
        pq_distance_table: Optional[np.ndarray]
    ) -> List[Tuple[int, float]]:
        """
        Search a single zone and return candidates.
        
        Returns:
            List of (vector_id, approximate_distance) tuples
        """
        # Search HNSW graph
        global_ids, _ = self.hnsw_manager.search_zone(zone_id, query, k, ef_search)
        
        if len(global_ids) == 0:
            return []
        
        # Compute approximate distances using PQ (if available)
        if self.use_pq and pq_distance_table is not None:
            # Get PQ codes for these vectors
            codes = self.pq_codes[global_ids]
            # Compute approximate distances
            approx_distances = self.pq.asymmetric_distance(codes, pq_distance_table)
        else:
            # Use exact distances if no PQ
            approx_distances = DistanceMetrics.euclidean_batch_squared(query, self.vectors[global_ids])
        
        return list(zip(global_ids, approx_distances))
    
    def _aggregate_candidates(
        self,
        candidates: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Aggregate and deduplicate candidates from multiple zones.
        
        Keep the best (lowest) distance for each vector ID.
        
        Returns:
            List of unique (vector_id, distance) tuples
        """
        # Use dict to keep best distance for each ID
        best_distances = {}
        
        for vector_id, distance in candidates:
            if vector_id not in best_distances or distance < best_distances[vector_id]:
                best_distances[vector_id] = distance
        
        return [(vid, dist) for vid, dist in best_distances.items()]
    
    def _select_top_candidates(
        self,
        candidates: List[Tuple[int, float]],
        k_rerank: int
    ) -> List[int]:
        """
        Select top k_rerank candidates by approximate distance.
        
        Returns:
            List of vector IDs
        """
        # Sort by distance and take top k_rerank
        sorted_candidates = sorted(candidates, key=lambda x: x[1])
        return [vid for vid, _ in sorted_candidates[:k_rerank]]
    
    def _rerank_exact(
        self,
        query: np.ndarray,
        candidate_ids: List[int],
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Re-rank candidates using exact distances.
        
        Returns:
            (ids, distances): Top k results
        """
        if len(candidate_ids) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        # Convert to numpy array for proper indexing
        candidate_ids_array = np.array(candidate_ids, dtype=np.int32)
        
        # Compute exact distances
        candidate_vectors = self.vectors[candidate_ids_array]
        exact_distances = DistanceMetrics.euclidean_batch_squared(query, candidate_vectors)
        
        # Sort by exact distance
        k_actual = min(k, len(candidate_ids))
        
        if k_actual < len(exact_distances):
            # Use argpartition for efficiency
            top_k_indices = np.argpartition(exact_distances, k_actual - 1)[:k_actual]
            top_k_indices = top_k_indices[np.argsort(exact_distances[top_k_indices])]
        else:
            # Just sort all
            top_k_indices = np.argsort(exact_distances)
        
        # Return IDs and distances
        result_ids = candidate_ids_array[top_k_indices]
        result_distances = exact_distances[top_k_indices]
        
        return result_ids, result_distances
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        k_rerank: Optional[int] = None,
        n_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Query vectors of shape (n_queries, d)
            k: Number of nearest neighbors per query
            n_probe: Number of zones to search
            ef_search: HNSW search parameter
            k_rerank: Number of candidates to re-rank
            n_jobs: Number of parallel workers (1 = sequential)
            
        Returns:
            (ids, distances): Arrays of shape (n_queries, k)
        """
        n_queries = len(queries)
        all_ids = np.zeros((n_queries, k), dtype=np.int32)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        
        if n_jobs == 1:
            # Sequential search
            for i, query in enumerate(queries):
                ids, distances = self.search(query, k, n_probe, ef_search, k_rerank)
                all_ids[i] = ids
                all_distances[i] = distances
        else:
            # Parallel search
            def search_single(i):
                return self.search(queries[i], k, n_probe, ef_search, k_rerank)
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(search_single, range(n_queries)))
            
            for i, (ids, distances) in enumerate(results):
                all_ids[i] = ids
                all_distances[i] = distances
        
        return all_ids, all_distances


def compute_ground_truth(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact k-nearest neighbors using brute force.
    
    This is used for validation and benchmarking.
    
    Args:
        vectors: Database vectors of shape (N, d)
        queries: Query vectors of shape (n_queries, d)
        k: Number of nearest neighbors
        
    Returns:
        (ids, distances): Arrays of shape (n_queries, k)
    """
    n_queries = len(queries)
    ground_truth_ids = np.zeros((n_queries, k), dtype=np.int32)
    ground_truth_distances = np.zeros((n_queries, k), dtype=np.float32)
    
    for i, query in enumerate(queries):
        distances = DistanceMetrics.euclidean_batch_squared(query, vectors)
        
        # Get k nearest neighbors
        k_actual = min(k, len(vectors))
        nearest_indices = np.argpartition(distances, k_actual - 1)[:k_actual]
        nearest_indices = nearest_indices[np.argsort(distances[nearest_indices])]
        
        ground_truth_ids[i, :k_actual] = nearest_indices
        ground_truth_distances[i, :k_actual] = distances[nearest_indices]
    
    return ground_truth_ids, ground_truth_distances
