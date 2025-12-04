"""
ZGQ Search Algorithms for ZGQ v8
=================================

This module implements advanced search algorithms for ZGQ including:
- Zone-guided beam search
- Multi-probe search with zone expansion
- PQ-assisted candidate filtering
- Exact re-ranking with early termination

The search strategies are designed to beat HNSW on query latency
while maintaining high recall.
"""

import numpy as np
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass
import heapq

from zgq.core.distances import DistanceComputer, SIMDDistance, select_top_k_distances
from zgq.core.quantization import ResidualProductQuantizer


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    
    # Probe settings
    n_probe: int = 8                    # Number of zones to search
    expand_neighbors: bool = True       # Expand to neighboring zones
    max_expansion: int = 2              # Maximum neighbor expansion
    
    # Candidate settings
    k_candidates_factor: int = 3        # Candidates = k * factor
    use_pq_filter: bool = True          # Use PQ for initial filtering
    
    # Re-ranking settings
    k_rerank_factor: int = 2            # Rerank = k * factor
    use_exact_rerank: bool = True       # Use exact distances for final ranking
    use_early_termination: bool = True  # Early terminate distance computation
    
    # Quality settings
    quality_threshold: float = 0.9      # Stop if estimated recall >= threshold


class ZGQSearch:
    """
    Advanced search algorithms for ZGQ.
    
    Provides multiple search strategies optimized for different
    speed-recall trade-offs.
    """
    
    def __init__(
        self,
        zones,
        graph,
        vectors: np.ndarray,
        pq: Optional[ResidualProductQuantizer] = None,
        pq_codes: Optional[np.ndarray] = None,
        vector_norms: Optional[np.ndarray] = None,
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize ZGQ search.
        
        Args:
            zones: AdaptiveHierarchicalZones instance
            graph: ZoneGuidedGraph instance
            vectors: Original vectors for re-ranking
            pq: Product quantizer (optional)
            pq_codes: Encoded PQ codes (optional)
            vector_norms: Precomputed vector norms (optional)
            config: Search configuration
        """
        self.zones = zones
        self.graph = graph
        self.vectors = vectors
        self.pq = pq
        self.pq_codes = pq_codes
        self.vector_norms = vector_norms
        self.config = config or SearchConfig()
        
        self.distance_computer = DistanceComputer(metric='l2')
        self.n_vectors = len(vectors)
        self.dimension = vectors.shape[1]
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: Optional[int] = None,
        ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector of shape (d,)
            k: Number of neighbors
            n_probe: Number of zones (overrides config)
            ef_search: HNSW beam width (overrides graph default)
            
        Returns:
            (ids, distances): Arrays of shape (k,)
        """
        query = query.astype(np.float32)
        n_probe = n_probe or self.config.n_probe
        
        # Phase 1: Zone selection
        selected_zones = self._select_zones_adaptive(query, n_probe)
        
        # Phase 2: Candidate generation via graph search
        candidates = self._generate_candidates(
            query, k, selected_zones, ef_search
        )
        
        # Phase 3: PQ-based filtering (if available)
        if self.pq and self.config.use_pq_filter:
            candidates = self._filter_with_pq(query, candidates, k, selected_zones)
        
        # Phase 4: Exact re-ranking
        ids, distances = self._exact_rerank(query, candidates, k)
        
        return ids, distances
    
    def _select_zones_adaptive(
        self,
        query: np.ndarray,
        n_probe: int
    ) -> np.ndarray:
        """
        Select zones to search with adaptive expansion.
        
        Expands to neighboring zones if estimated recall is low.
        """
        # Get initial zones
        selected = self.zones.select_zones(query, n_probe)
        
        # Optionally expand to neighbors
        if self.config.expand_neighbors and n_probe < len(self.zones.fine_centroids):
            expanded = set(selected.tolist())
            
            for zone_id in selected[:self.config.max_expansion]:
                neighbors = self.zones.get_zone_neighbors(zone_id)
                for neighbor in neighbors[:2]:  # Add top 2 neighbors per zone
                    expanded.add(neighbor)
            
            selected = np.array(list(expanded), dtype=np.int32)
        
        return selected
    
    def _generate_candidates(
        self,
        query: np.ndarray,
        k: int,
        selected_zones: np.ndarray,
        ef_search: Optional[int]
    ) -> np.ndarray:
        """Generate candidate set using zone-guided graph search."""
        
        k_candidates = min(
            k * self.config.k_candidates_factor,
            self.n_vectors
        )
        
        # Search graph with zone guidance
        ids, _ = self.graph.search(query, k_candidates, selected_zones, ef_search)
        
        return ids
    
    def _filter_with_pq(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        k: int,
        selected_zones: np.ndarray
    ) -> np.ndarray:
        """Filter candidates using PQ approximate distances."""
        
        if len(candidates) <= k * self.config.k_rerank_factor:
            return candidates
        
        # Compute PQ distances
        query_zone = selected_zones[0]
        zone_centroid = self.zones.fine_centroids[query_zone]
        
        dist_table = self.pq.compute_distance_table(query, zone_centroid)
        pq_distances = self.pq.asymmetric_distance(
            self.pq_codes[candidates], dist_table
        )
        
        # Select top candidates
        k_filter = min(k * self.config.k_rerank_factor, len(candidates))
        top_indices = np.argpartition(pq_distances, k_filter - 1)[:k_filter]
        
        return candidates[top_indices]
    
    def _exact_rerank(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Re-rank candidates using exact distances."""
        
        if len(candidates) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        # Compute exact distances
        candidate_vectors = self.vectors[candidates]
        
        if self.vector_norms is not None:
            candidate_norms = self.vector_norms[candidates]
        else:
            candidate_norms = None
        
        distances = self.distance_computer.compute(
            query, candidate_vectors, candidate_norms
        )
        
        # Select top k
        top_indices, top_distances = select_top_k_distances(distances, k)
        
        return candidates[top_indices], top_distances
    
    def search_fast(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast search with minimal overhead.
        
        Trades some recall for maximum speed.
        Suitable for latency-critical applications.
        """
        query = query.astype(np.float32)
        
        # Minimal zone selection
        selected_zones = self.zones.select_zones(query, n_probe=4)
        
        # Direct graph search
        ids, distances = self.graph.search(query, k, selected_zones)
        
        return ids, distances
    
    def search_accurate(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        High-accuracy search.
        
        Maximizes recall at the cost of higher latency.
        Suitable for accuracy-critical applications.
        """
        query = query.astype(np.float32)
        
        # Extensive zone selection
        n_probe = min(32, len(self.zones.fine_centroids))
        selected_zones = self.zones.select_zones(query, n_probe)
        
        # More candidates
        k_candidates = min(k * 10, self.n_vectors)
        ids, _ = self.graph.search(query, k_candidates, selected_zones)
        
        # Full PQ filtering
        if self.pq:
            query_zone = selected_zones[0]
            zone_centroid = self.zones.fine_centroids[query_zone]
            dist_table = self.pq.compute_distance_table(query, zone_centroid)
            pq_distances = self.pq.asymmetric_distance(self.pq_codes[ids], dist_table)
            
            k_rerank = min(k * 5, len(ids))
            top_pq = np.argpartition(pq_distances, k_rerank - 1)[:k_rerank]
            ids = ids[top_pq]
        
        # Exact re-ranking
        return self._exact_rerank(query, ids, k)
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_probe: Optional[int] = None,
        n_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: Query vectors of shape (n_queries, d)
            k: Number of neighbors per query
            n_probe: Number of zones to search
            n_jobs: Number of parallel workers
            
        Returns:
            (ids, distances): Arrays of shape (n_queries, k)
        """
        queries = queries.astype(np.float32)
        n_queries = len(queries)
        
        all_ids = np.zeros((n_queries, k), dtype=np.int32)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        
        if n_jobs == 1:
            for i in range(n_queries):
                ids, distances = self.search(queries[i], k, n_probe)
                n_results = min(k, len(ids))
                all_ids[i, :n_results] = ids[:n_results]
                all_distances[i, :n_results] = distances[:n_results]
        else:
            from concurrent.futures import ThreadPoolExecutor
            
            def search_one(idx):
                return self.search(queries[idx], k, n_probe)
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(search_one, range(n_queries)))
            
            for i, (ids, distances) in enumerate(results):
                n_results = min(k, len(ids))
                all_ids[i, :n_results] = ids[:n_results]
                all_distances[i, :n_results] = distances[:n_results]
        
        return all_ids, all_distances


def compute_ground_truth(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute exact k-nearest neighbors (brute force).
    
    Used for validation and benchmarking.
    
    Args:
        vectors: Database vectors of shape (N, d)
        queries: Query vectors of shape (n_queries, d)
        k: Number of neighbors
        
    Returns:
        (ids, distances): Ground truth nearest neighbors
    """
    vectors = vectors.astype(np.float32)
    queries = queries.astype(np.float32)
    
    n_queries = len(queries)
    k = min(k, len(vectors))
    
    gt_ids = np.zeros((n_queries, k), dtype=np.int32)
    gt_distances = np.zeros((n_queries, k), dtype=np.float32)
    
    distance_computer = DistanceComputer(metric='l2')
    
    for i in range(n_queries):
        distances = distance_computer.compute(queries[i], vectors)
        top_k, top_dists = select_top_k_distances(distances, k)
        gt_ids[i] = top_k
        gt_distances[i] = top_dists
    
    return gt_ids, gt_distances


def compute_recall(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10
) -> float:
    """
    Compute recall@k.
    
    Args:
        predicted: Predicted neighbor IDs of shape (n_queries, k)
        ground_truth: Ground truth IDs of shape (n_queries, k)
        k: Number of neighbors to consider
        
    Returns:
        Recall@k as percentage
    """
    n_queries = predicted.shape[0]
    k = min(k, predicted.shape[1], ground_truth.shape[1])
    
    recalls = []
    for i in range(n_queries):
        pred_set = set(predicted[i, :k])
        true_set = set(ground_truth[i, :k])
        recall = len(pred_set & true_set) / k
        recalls.append(recall)
    
    return np.mean(recalls) * 100
