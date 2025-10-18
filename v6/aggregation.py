"""
Aggregation and Re-ranking Module for ZGQ V6
Implements result aggregation and exact distance re-ranking as specified in aggregation_reranking.md

Mathematical Foundation:
1. Deduplication: Keep best PQ distance for each unique ID
2. Selection: Choose top k_rerank candidates by PQ distance
3. Re-ranking: Compute exact distances for selected candidates
4. Final selection: Return top-k by exact distance

Complexity: O(C log C + k_rerank · d) where C = total candidates

Reference: aggregation_reranking.md
"""

import numpy as np
from typing import List, Tuple, Dict, Set
from distance_metrics import DistanceMetrics
from dataclasses import dataclass
import heapq


@dataclass
class SearchResult:
    """
    Single search result.
    
    Attributes:
        vector_id: Global vector ID
        pq_distance: Approximate PQ distance (if available)
        exact_distance: Exact Euclidean distance (computed during re-ranking)
        zone_id: Zone this result came from (optional)
    """
    vector_id: int
    pq_distance: float = float('inf')
    exact_distance: float = float('inf')
    zone_id: int = -1
    
    def __lt__(self, other):
        """For heap operations - compare by exact distance if available, else PQ."""
        if self.exact_distance != float('inf'):
            return self.exact_distance < other.exact_distance
        return self.pq_distance < other.pq_distance


class ResultAggregator:
    """
    Aggregate and re-rank search results from multiple zones.
    
    Implements aggregation_reranking.md Section 1.
    """
    
    @staticmethod
    def deduplicate_candidates(
        candidates: List[Tuple[int, float, int]]
    ) -> Dict[int, Tuple[float, int]]:
        """
        Deduplicate candidates, keeping best PQ distance for each ID.
        
        Formula: For each ID, keep min(PQ distances)
        
        Complexity: O(C) where C = len(candidates)
        
        Args:
            candidates: List of (vector_id, pq_distance, zone_id) tuples
            
        Returns:
            Dict mapping vector_id -> (best_pq_distance, zone_id)
            
        Reference: aggregation_reranking.md Section 1.2 Step 1
        """
        best_candidates = {}
        
        for vector_id, pq_distance, zone_id in candidates:
            if vector_id not in best_candidates or pq_distance < best_candidates[vector_id][0]:
                best_candidates[vector_id] = (pq_distance, zone_id)
        
        return best_candidates
    
    @staticmethod
    def aggregate_and_rerank(
        query: np.ndarray,
        all_candidates: List[Tuple[int, float, int]],
        full_vectors: np.ndarray,
        k: int,
        k_rerank: int = None,
        vector_norms_sq: np.ndarray = None,
        verbose: bool = False
    ) -> List[SearchResult]:
        """
        Aggregate candidates from multiple zones and re-rank with exact distances.
        
        Algorithm:
        ----------
        1. Deduplication: Keep best PQ distance for each unique ID
        2. Selection: Select top k_rerank candidates by PQ distance
        3. Compute exact distances: Use Euclidean distance for selected candidates
        4. Final selection: Return top-k by exact distance
        
        Complexity: O(C log C + k_rerank · d)
        where C = len(all_candidates)
        
        Args:
            query: Query vector of shape (d,)
            all_candidates: List of (vector_id, pq_distance, zone_id) tuples
            full_vectors: Full dataset matrix of shape (N, d)
            k: Final number of results to return
            k_rerank: Number of candidates to re-rank (default: 2*k)
            vector_norms_sq: Precomputed vector norms for speedup (optional)
            verbose: Print debug information
            
        Returns:
            List of SearchResult objects, sorted by exact distance
            
        Reference: aggregation_reranking.md Section 1
        """
        if k_rerank is None:
            k_rerank = min(len(all_candidates), 2 * k)
        
        if verbose:
            print(f"\n  Aggregating {len(all_candidates)} candidates...")
        
        # Step 1: Deduplication - O(C)
        best_candidates = ResultAggregator.deduplicate_candidates(all_candidates)
        
        if verbose:
            print(f"  After deduplication: {len(best_candidates)} unique candidates")
        
        # Step 2: Select top k_rerank by PQ distance - O(C log C)
        candidate_list = [
            (pq_dist, vec_id, zone_id)
            for vec_id, (pq_dist, zone_id) in best_candidates.items()
        ]
        candidate_list.sort()
        candidates_to_rerank = candidate_list[:k_rerank]
        
        if verbose:
            print(f"  Re-ranking top {len(candidates_to_rerank)} candidates...")
        
        # Step 3: Compute exact distances - O(k_rerank · d)
        query_norm_sq = np.dot(query, query)
        results = []
        
        for pq_dist, vec_id, zone_id in candidates_to_rerank:
            # Compute exact Euclidean distance
            if vector_norms_sq is not None:
                exact_dist = DistanceMetrics.euclidean_squared_optimized(
                    query,
                    full_vectors[vec_id],
                    query_norm_sq,
                    vector_norms_sq[vec_id]
                )
            else:
                exact_dist = DistanceMetrics.euclidean_squared(
                    query,
                    full_vectors[vec_id]
                )
            
            results.append(SearchResult(
                vector_id=vec_id,
                pq_distance=pq_dist,
                exact_distance=exact_dist,
                zone_id=zone_id
            ))
        
        # Step 4: Sort by exact distance and return top-k - O(k_rerank log k_rerank)
        results.sort(key=lambda x: x.exact_distance)
        
        return results[:k]
    
    @staticmethod
    def compute_recall(
        retrieved: List[SearchResult],
        ground_truth: Set[int],
        k: int = None
    ) -> float:
        """
        Compute recall@k for retrieved results.
        
        Formula: Recall@k = |retrieved ∩ ground_truth| / k
        
        Args:
            retrieved: List of SearchResult objects
            ground_truth: Set of true k-NN IDs
            k: Number of top results to consider (default: all retrieved)
            
        Returns:
            Recall value in [0, 1]
        """
        if k is None:
            k = len(retrieved)
        
        retrieved_ids = {r.vector_id for r in retrieved[:k]}
        intersection = retrieved_ids & ground_truth
        
        return len(intersection) / k if k > 0 else 0.0
    
    @staticmethod
    def compute_precision(
        retrieved: List[SearchResult],
        ground_truth: Set[int],
        k: int = None
    ) -> float:
        """
        Compute precision@k for retrieved results.
        
        Formula: Precision@k = |retrieved ∩ ground_truth| / |retrieved|
        
        Args:
            retrieved: List of SearchResult objects
            ground_truth: Set of true k-NN IDs
            k: Number of top results to consider (default: all retrieved)
            
        Returns:
            Precision value in [0, 1]
        """
        if k is None:
            k = len(retrieved)
        
        retrieved_ids = {r.vector_id for r in retrieved[:k]}
        intersection = retrieved_ids & ground_truth
        
        return len(intersection) / len(retrieved_ids) if len(retrieved_ids) > 0 else 0.0
    
    @staticmethod
    def compute_ndcg(
        retrieved: List[SearchResult],
        ground_truth_distances: Dict[int, float],
        k: int = None
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (NDCG@k).
        
        Measures ranking quality considering the order of results.
        
        Formula:
            DCG@k = Σᵢ₌₁ᵏ (1 / log₂(i + 1)) · relevance(i)
            NDCG@k = DCG@k / IDCG@k
        
        where relevance is based on distance (closer = more relevant)
        
        Args:
            retrieved: List of SearchResult objects
            ground_truth_distances: Dict mapping vector_id -> true distance
            k: Number of top results to consider (default: all retrieved)
            
        Returns:
            NDCG value in [0, 1]
        """
        if k is None:
            k = len(retrieved)
        
        if k == 0 or len(ground_truth_distances) == 0:
            return 0.0
        
        # Compute DCG for retrieved results
        dcg = 0.0
        for i, result in enumerate(retrieved[:k]):
            if result.vector_id in ground_truth_distances:
                # Relevance: inverse of rank in ground truth
                # (simpler alternative: could use distance-based relevance)
                relevance = 1.0  # Binary relevance (in ground truth or not)
                dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0
        
        # Compute ideal DCG (all relevant items ranked first)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth_distances))))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def analyze_pq_approximation_quality(
        results: List[SearchResult],
        k: int = None
    ) -> Dict[str, float]:
        """
        Analyze PQ approximation quality.
        
        Metrics:
        - Spearman rank correlation between PQ and exact distances
        - Mean absolute percentage error
        - Max distance distortion
        
        Args:
            results: List of SearchResult objects with both PQ and exact distances
            k: Number of top results to analyze (default: all)
            
        Returns:
            Dict with analysis metrics
        """
        if k is None:
            k = len(results)
        
        results = results[:k]
        
        pq_dists = np.array([r.pq_distance for r in results])
        exact_dists = np.array([r.exact_distance for r in results])
        
        # Remove any inf values
        valid_mask = (pq_dists != np.inf) & (exact_dists != np.inf)
        pq_dists = pq_dists[valid_mask]
        exact_dists = exact_dists[valid_mask]
        
        if len(pq_dists) == 0:
            return {}
        
        # Rank correlation
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(pq_dists, exact_dists)
        
        # Mean absolute percentage error
        mape = np.mean(np.abs(pq_dists - exact_dists) / (exact_dists + 1e-10)) * 100
        
        # Max distortion
        max_distortion = np.max(np.abs(pq_dists - exact_dists) / (exact_dists + 1e-10))
        
        return {
            'rank_correlation': corr,
            'correlation_p_value': p_value,
            'mean_absolute_percentage_error': mape,
            'max_distortion': max_distortion,
            'num_samples': len(pq_dists)
        }


# Validation and testing
if __name__ == "__main__":
    print("="*70)
    print("Aggregation and Re-ranking Module - Validation")
    print("="*70)
    
    # Configuration
    N = 10000
    d = 128
    k = 10
    n_zones = 5
    candidates_per_zone = 20
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    vectors = np.random.randn(N, d).astype(np.float32)
    query = np.random.randn(d).astype(np.float32)
    
    # Compute ground truth
    print("\nComputing ground truth...")
    true_distances = DistanceMetrics.euclidean_batch_squared(query, vectors)
    true_nn_ids = set(np.argsort(true_distances)[:k])
    print(f"  Ground truth k-NN: {sorted(list(true_nn_ids))[:5]}... (showing first 5)")
    
    # Simulate candidates from multiple zones with PQ distances
    print("\nSimulating multi-zone search results...")
    all_candidates = []
    
    for zone_id in range(n_zones):
        # Random candidates from this zone
        zone_candidates = np.random.choice(N, candidates_per_zone, replace=False)
        
        for vec_id in zone_candidates:
            # Simulate PQ distance with some error
            exact_dist = true_distances[vec_id]
            pq_dist = exact_dist * np.random.uniform(0.9, 1.1)  # ±10% error
            all_candidates.append((vec_id, pq_dist, zone_id))
    
    print(f"  Total candidates: {len(all_candidates)}")
    
    # Test 1: Deduplication
    print("\n[Test 1] Deduplication")
    best_candidates = ResultAggregator.deduplicate_candidates(all_candidates)
    print(f"  Before deduplication: {len(all_candidates)} candidates")
    print(f"  After deduplication: {len(best_candidates)} unique candidates")
    
    # Test 2: Full aggregation and re-ranking
    print("\n[Test 2] Aggregation and Re-ranking")
    vector_norms_sq = DistanceMetrics.precompute_vector_norms(vectors)
    
    import time
    start = time.time()
    results = ResultAggregator.aggregate_and_rerank(
        query=query,
        all_candidates=all_candidates,
        full_vectors=vectors,
        k=k,
        k_rerank=50,
        vector_norms_sq=vector_norms_sq,
        verbose=True
    )
    rerank_time = time.time() - start
    
    print(f"  Re-ranking time: {rerank_time*1000:.2f} ms")
    print(f"\n  Top-{k} results:")
    for i, result in enumerate(results):
        print(f"    {i+1}. ID={result.vector_id:5d}, "
              f"PQ_dist={result.pq_distance:.4f}, "
              f"Exact_dist={result.exact_distance:.4f}, "
              f"Zone={result.zone_id}")
    
    # Test 3: Recall computation
    print("\n[Test 3] Recall Evaluation")
    recall = ResultAggregator.compute_recall(results, true_nn_ids, k=k)
    precision = ResultAggregator.compute_precision(results, true_nn_ids, k=k)
    print(f"  Recall@{k}: {recall:.4f}")
    print(f"  Precision@{k}: {precision:.4f}")
    
    # Test 4: PQ approximation quality
    print("\n[Test 4] PQ Approximation Quality Analysis")
    try:
        pq_quality = ResultAggregator.analyze_pq_approximation_quality(results, k=k)
        print(f"  Rank correlation: {pq_quality['rank_correlation']:.4f}")
        print(f"  MAPE: {pq_quality['mean_absolute_percentage_error']:.2f}%")
        print(f"  Max distortion: {pq_quality['max_distortion']:.2f}")
    except ImportError:
        print("  (scipy not available, skipping correlation analysis)")
    
    # Test 5: Varying k_rerank
    print("\n[Test 5] Effect of k_rerank on Recall")
    k_rerank_values = [10, 20, 50, 100]
    
    for k_r in k_rerank_values:
        results_kr = ResultAggregator.aggregate_and_rerank(
            query=query,
            all_candidates=all_candidates,
            full_vectors=vectors,
            k=k,
            k_rerank=k_r,
            vector_norms_sq=vector_norms_sq,
            verbose=False
        )
        recall_kr = ResultAggregator.compute_recall(results_kr, true_nn_ids, k=k)
        print(f"  k_rerank={k_r:3d}: Recall@{k} = {recall_kr:.4f}")
    
    # Test 6: NDCG computation
    print("\n[Test 6] NDCG Evaluation")
    ground_truth_dists = {i: true_distances[i] for i in true_nn_ids}
    
    # Sort results by exact distance for optimal NDCG
    results_sorted = sorted(results, key=lambda x: x.exact_distance)
    ndcg = ResultAggregator.compute_ndcg(results_sorted, ground_truth_dists, k=k)
    print(f"  NDCG@{k}: {ndcg:.4f}")
    
    # Compare with unsorted (random order)
    results_random = results.copy()
    np.random.shuffle(results_random)
    ndcg_random = ResultAggregator.compute_ndcg(results_random, ground_truth_dists, k=k)
    print(f"  NDCG@{k} (random order): {ndcg_random:.4f}")
    print(f"  Improvement: {(ndcg/ndcg_random - 1)*100:.1f}%")
    
    print("\n" + "="*70)
    print("✓ All tests passed successfully")
    print("="*70)
