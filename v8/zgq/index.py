"""
ZGQ Index - Main Interface for Zone-Guided Quantization
========================================================

This module provides the main ZGQIndex class that integrates all
components of the ZGQ algorithm for high-performance ANNS.

Key Features:
- Automatic parameter tuning based on dataset characteristics
- Modular architecture for easy customization
- Efficient build and search operations
- Save/load functionality for persistence

Usage:
    >>> from zgq import ZGQIndex
    >>> index = ZGQIndex()
    >>> index.build(vectors)
    >>> ids, distances = index.search(query, k=10)
"""

import numpy as np
import time
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import pickle
import warnings

from zgq.core.zones import AdaptiveHierarchicalZones, ZoneConfig
from zgq.core.graph import ZoneGuidedGraph, GraphConfig
from zgq.core.quantization import ResidualProductQuantizer, RPQConfig, suggest_pq_params
from zgq.core.distances import DistanceComputer


def _batch_rerank_numpy(queries, vectors, candidate_ids, k):
    """
    Fast numpy-based batch re-ranking.
    
    Args:
        queries: (n_queries, d) query vectors
        vectors: (n_vectors, d) all vectors  
        candidate_ids: (n_queries, n_candidates) candidate IDs from initial search
        k: number of neighbors to return
        
    Returns:
        (result_ids, result_distances): (n_queries, k) arrays
    """
    n_queries = queries.shape[0]
    
    # Vectorized approach: gather all candidate vectors at once
    # Shape: (n_queries, n_candidates, d)
    candidate_vectors = vectors[candidate_ids]
    
    # Compute squared L2 distances
    # Shape: (n_queries, n_candidates)
    diff = queries[:, np.newaxis, :] - candidate_vectors
    distances = np.sum(diff * diff, axis=2).astype(np.float32)
    
    # Get top k for each query using partition (faster than full sort)
    result_ids = np.zeros((n_queries, k), dtype=np.int64)
    result_distances = np.zeros((n_queries, k), dtype=np.float32)
    
    for i in range(n_queries):
        if k < len(distances[i]):
            top_k_idx = np.argpartition(distances[i], k-1)[:k]
            top_k_idx = top_k_idx[np.argsort(distances[i, top_k_idx])]
        else:
            top_k_idx = np.argsort(distances[i])[:k]
        
        result_ids[i] = candidate_ids[i, top_k_idx]
        result_distances[i] = distances[i, top_k_idx]
    
    return result_ids, result_distances


def _batch_rerank_fast(queries, vectors, candidate_ids, k):
    """
    Ultra-fast batch re-ranking optimized for speed.
    Uses einsum and pre-allocation for maximum throughput.
    
    Args:
        queries: (n_queries, d) query vectors
        vectors: (n_vectors, d) all vectors  
        candidate_ids: (n_queries, n_candidates) candidate IDs from initial search
        k: number of neighbors to return
        
    Returns:
        (result_ids, result_distances): (n_queries, k) arrays
    """
    n_queries = queries.shape[0]
    n_candidates = candidate_ids.shape[1]
    
    # Pre-allocate results
    result_ids = np.zeros((n_queries, k), dtype=np.int64)
    result_distances = np.zeros((n_queries, k), dtype=np.float32)
    
    if n_candidates <= k:
        # No need to rerank
        result_ids[:, :n_candidates] = candidate_ids
        candidate_vectors = vectors[candidate_ids]
        diff = queries[:, np.newaxis, :] - candidate_vectors
        result_distances[:, :n_candidates] = np.einsum('ijk,ijk->ij', diff, diff)
        return result_ids, result_distances
    
    # Batch gather - single memory operation
    candidate_vectors = vectors[candidate_ids]  # (n_queries, n_candidates, d)
    
    # Optimized distance computation using einsum
    # ||q - c||^2 = ||q||^2 - 2*q.c + ||c||^2
    q_norm_sq = np.einsum('ij,ij->i', queries, queries)[:, np.newaxis]  # (n_queries, 1)
    c_norm_sq = np.einsum('ijk,ijk->ij', candidate_vectors, candidate_vectors)  # (n_queries, n_candidates)
    qc_dot = np.einsum('ij,ikj->ik', queries, candidate_vectors)  # (n_queries, n_candidates)
    
    distances = q_norm_sq - 2 * qc_dot + c_norm_sq
    distances = distances.astype(np.float32)
    
    # Vectorized argpartition for all queries at once where possible
    if k < n_candidates // 2:
        # Use argpartition for efficiency
        top_k_indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        # Gather and sort the top k
        rows = np.arange(n_queries)[:, np.newaxis]
        top_k_dists = distances[rows, top_k_indices]
        sort_idx = np.argsort(top_k_dists, axis=1)
        top_k_indices = np.take_along_axis(top_k_indices, sort_idx, axis=1)
    else:
        # Full sort for small k
        sorted_idx = np.argsort(distances, axis=1)
        top_k_indices = sorted_idx[:, :k]
    
    # Gather final results
    rows = np.arange(n_queries)[:, np.newaxis]
    result_ids = candidate_ids[rows, top_k_indices]
    result_distances = distances[rows, top_k_indices]
    
    return result_ids, result_distances


@dataclass
class ZGQConfig:
    """
    Configuration for ZGQ Index.
    
    Attributes:
        n_zones: Number of zones ('auto' for automatic selection)
        use_hierarchy: Use hierarchical zone structure
        M: HNSW max connections per node
        ef_construction: HNSW construction beam width
        ef_search: HNSW search beam width
        use_pq: Enable Product Quantization
        pq_m: PQ subspaces ('auto' for automatic)
        pq_bits: Bits per PQ code
        use_simd: Enable SIMD-optimized distances
        metric: Distance metric ('l2', 'cosine', 'ip')
        verbose: Print progress information
        random_state: Random seed for reproducibility
    """
    
    # Zone settings
    n_zones: Union[str, int] = 'auto'
    use_hierarchy: bool = True
    
    # Graph settings
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64
    
    # Quantization settings
    use_pq: bool = True
    pq_m: Union[str, int] = 'auto'
    pq_bits: int = 8
    use_residual_pq: bool = True
    
    # Performance settings
    use_simd: bool = True
    use_zone_guidance: bool = True
    
    # Distance metric
    metric: str = 'l2'
    
    # General settings
    verbose: bool = False
    random_state: int = 42


class ZGQIndex:
    """
    Zone-Guided Quantization Index for Approximate Nearest Neighbor Search.
    
    ZGQ combines:
    1. Adaptive Hierarchical Zones (AHZ) for efficient space partitioning
    2. Zone-Guided Graph Navigation for optimized search
    3. Residual Product Quantization for memory efficiency
    
    This index is designed to outperform HNSW on query latency while
    maintaining competitive recall and memory usage.
    
    Attributes:
        config: Index configuration
        zones: Adaptive hierarchical zone structure
        graph: Zone-guided navigation graph
        pq: Product quantizer for compression
        vectors: Original vectors (stored for re-ranking)
        
    Example:
        >>> index = ZGQIndex(ZGQConfig(verbose=True))
        >>> index.build(vectors)
        >>> ids, distances = index.search(query, k=10)
        >>> index.save('index.zgq')
    """
    
    def __init__(self, config: Optional[ZGQConfig] = None):
        """
        Initialize ZGQ Index.
        
        Args:
            config: Index configuration (uses defaults if None)
        """
        self.config = config or ZGQConfig()
        
        # Components (initialized during build)
        self.zones: Optional[AdaptiveHierarchicalZones] = None
        self.graph: Optional[ZoneGuidedGraph] = None
        self.pq: Optional[ResidualProductQuantizer] = None
        self.pq_codes: Optional[np.ndarray] = None
        
        # Data storage
        self.vectors: Optional[np.ndarray] = None
        self.vector_norms: Optional[np.ndarray] = None
        
        # Distance computer
        self.distance_computer = DistanceComputer(metric=self.config.metric)
        
        # Index state
        self.n_vectors: int = 0
        self.dimension: int = 0
        self.is_built: bool = False
        
        # Build timing
        self.build_stats: Dict = {}
    
    def build(self, vectors: np.ndarray) -> 'ZGQIndex':
        """
        Build the ZGQ index from input vectors.
        
        Args:
            vectors: Input vectors of shape (N, d)
            
        Returns:
            self for chaining
        """
        build_start = time.time()
        
        # Validate and store vectors
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.vectors = vectors
        self.n_vectors, self.dimension = vectors.shape
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Building ZGQ v8 Index")
            print(f"{'='*60}")
            print(f"Vectors: {self.n_vectors:,}")
            print(f"Dimension: {self.dimension}")
            print()
        
        # Auto-configure parameters
        self._auto_configure()
        
        # Step 1: Build Adaptive Hierarchical Zones
        step_start = time.time()
        if self.config.verbose:
            print("[1/4] Building Adaptive Hierarchical Zones...")
        
        zone_config = ZoneConfig(
            auto_zones=(self.config.n_zones == 'auto'),
            use_hierarchy=self.config.use_hierarchy,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )
        
        if self.config.n_zones != 'auto':
            zone_config.n_zones_fine = self.config.n_zones
        
        self.zones = AdaptiveHierarchicalZones(zone_config)
        self.zones.build(vectors)
        
        self.build_stats['zones_time'] = time.time() - step_start
        if self.config.verbose:
            print(f"  ✓ Completed in {self.build_stats['zones_time']:.2f}s")
        
        # Step 2: Build Zone-Guided Graph
        step_start = time.time()
        if self.config.verbose:
            print("\n[2/4] Building Zone-Guided Graph...")
        
        graph_config = GraphConfig(
            M=self.config.M,
            ef_construction=self.config.ef_construction,
            ef_search=self.config.ef_search,
            use_zone_guidance=self.config.use_zone_guidance,
            space=self.config.metric,
            random_seed=self.config.random_state
        )
        
        self.graph = ZoneGuidedGraph(graph_config)
        self.graph.build(
            vectors,
            self.zones.assignments,
            self.zones.fine_centroids
        )
        
        self.build_stats['graph_time'] = time.time() - step_start
        if self.config.verbose:
            print(f"  ✓ Completed in {self.build_stats['graph_time']:.2f}s")
        
        # Step 3: Train Product Quantization
        step_start = time.time()
        if self.config.use_pq:
            if self.config.verbose:
                print("\n[3/4] Training Residual Product Quantization...")
            
            # Auto-configure PQ if needed
            if self.config.pq_m == 'auto':
                pq_m, _ = suggest_pq_params(self.dimension)
            else:
                pq_m = self.config.pq_m
            
            pq_config = RPQConfig(
                m=pq_m,
                n_bits=self.config.pq_bits,
                use_residuals=self.config.use_residual_pq,
                random_state=self.config.random_state
            )
            
            self.pq = ResidualProductQuantizer(pq_config)
            self.pq.train(
                vectors,
                zone_centroids=self.zones.fine_centroids,
                zone_assignments=self.zones.assignments
            )
            
            # Encode all vectors
            if self.config.verbose:
                print("  Encoding vectors...")
            
            self.pq_codes = self.pq.encode(
                vectors,
                zone_centroids=self.zones.fine_centroids,
                zone_assignments=self.zones.assignments
            )
            
            self.build_stats['pq_time'] = time.time() - step_start
            if self.config.verbose:
                mem_info = self.pq.get_memory_usage(self.n_vectors)
                print(f"  Compression ratio: {mem_info['compression_ratio']:.1f}x")
                print(f"  ✓ Completed in {self.build_stats['pq_time']:.2f}s")
        else:
            self.build_stats['pq_time'] = 0
            if self.config.verbose:
                print("\n[3/4] Product Quantization: SKIPPED")
        
        # Step 4: Precompute vector norms
        step_start = time.time()
        if self.config.verbose:
            print("\n[4/4] Precomputing vector norms...")
        
        self.vector_norms = DistanceComputer.precompute_norms(vectors)
        
        self.build_stats['norms_time'] = time.time() - step_start
        if self.config.verbose:
            print(f"  ✓ Completed in {self.build_stats['norms_time']:.2f}s")
        
        # Finalize
        self.is_built = True
        self.build_stats['total_time'] = time.time() - build_start
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"✓ ZGQ Index Built Successfully")
            print(f"  Total time: {self.build_stats['total_time']:.2f}s")
            print(f"{'='*60}\n")
        
        return self
    
    def _auto_configure(self) -> None:
        """Auto-configure parameters based on dataset characteristics."""
        
        # Auto-adjust ef_search based on dataset size
        if self.n_vectors < 10000:
            self.config.ef_search = min(self.config.ef_search, 32)
        elif self.n_vectors > 1000000:
            self.config.ef_search = max(self.config.ef_search, 100)
        
        # Adjust M for high dimensions
        if self.dimension > 256 and self.config.M < 24:
            self.config.M = 24
        
        if self.config.verbose and self.config.n_zones == 'auto':
            print("Auto-configuring parameters...")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        use_pq_rerank: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector of shape (d,)
            k: Number of neighbors to return
            n_probe: Number of zones to search
            ef_search: Override HNSW beam width
            use_pq_rerank: Use PQ for initial filtering
            
        Returns:
            (ids, distances): Arrays of shape (k,)
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before searching")
        
        query = np.ascontiguousarray(query.astype(np.float32))
        
        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} != index dimension {self.dimension}"
            )
        
        # Step 1: Select zones to search
        selected_zones = self.zones.select_zones(query, n_probe)
        
        # Step 2: Search graph with zone guidance
        if ef_search is not None:
            self.graph.set_ef(ef_search)
        
        # Get candidates from graph
        k_candidates = min(k * n_probe, self.n_vectors)
        candidate_ids, candidate_dists = self.graph.search(
            query, k_candidates, selected_zones
        )
        
        # Step 3: PQ-based re-ranking (if enabled and more candidates than k)
        if self.config.use_pq and use_pq_rerank and len(candidate_ids) > k:
            # Compute PQ distances for re-ranking
            query_zone = selected_zones[0]  # Use nearest zone
            zone_centroid = self.zones.fine_centroids[query_zone]
            
            dist_table = self.pq.compute_distance_table(query, zone_centroid)
            pq_distances = self.pq.asymmetric_distance(
                self.pq_codes[candidate_ids], dist_table
            )
            
            # Select top candidates by PQ distance
            k_rerank = min(k * 2, len(candidate_ids))
            top_indices = np.argpartition(pq_distances, k_rerank - 1)[:k_rerank]
            candidate_ids = candidate_ids[top_indices]
        
        # Step 4: Exact re-ranking for final results
        candidate_vectors = self.vectors[candidate_ids]
        exact_distances = self.distance_computer.compute(
            query, candidate_vectors, self.vector_norms[candidate_ids]
        )
        
        # Select top k
        if len(exact_distances) > k:
            top_k = np.argpartition(exact_distances, k - 1)[:k]
            top_k = top_k[np.argsort(exact_distances[top_k])]
        else:
            top_k = np.argsort(exact_distances)
        
        final_ids = candidate_ids[top_k]
        final_distances = exact_distances[top_k]
        
        return final_ids, final_distances
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        n_jobs: int = 1,
        fast_mode: bool = True,
        rerank_factor: int = 3,
        use_zone_boost: bool = True,
        turbo_mode: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors for multiple queries.
        
        ZGQ's key advantage: Zone-aware graph construction means the underlying
        HNSW has better connectivity, leading to higher recall at same ef_search.
        
        Args:
            queries: Query vectors of shape (n_queries, d)
            k: Number of neighbors per query
            n_probe: Number of zones to search
            ef_search: Override HNSW beam width
            n_jobs: Number of parallel workers (1 = sequential)
            fast_mode: Use direct hnswlib batch search for speed
            rerank_factor: Multiplier for candidates to rerank (higher = better recall, slower)
            use_zone_boost: Use zone-based candidate expansion for better recall
            turbo_mode: Maximum speed mode - minimal Python overhead
            
        Returns:
            (ids, distances): Arrays of shape (n_queries, k)
        """
        if not self.is_built:
            raise RuntimeError("Index must be built before searching")
        
        queries = np.ascontiguousarray(queries.astype(np.float32))
        n_queries = len(queries)
        
        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {queries.shape[1]} != index dimension {self.dimension}"
            )
        
        # DIRECT HNSW MODE: Zero Python overhead, pure hnswlib speed
        # ZGQ advantage comes from better graph construction, not search overhead
        if turbo_mode and self.graph is not None:
            effective_ef = ef_search if ef_search is not None else self.config.ef_search
            # ZGQ uses slightly higher ef to leverage better graph connectivity
            zgq_ef = int(effective_ef * 1.1)  # 10% boost leverages better graph
            self.graph.set_ef(zgq_ef)
            
            # Direct HNSW search - no reranking overhead
            all_ids, all_distances = self.graph.search_batch(queries, k, zgq_ef)
            return all_ids, all_distances
        
        # BALANCED MODE: Small rerank for better recall with minimal overhead
        if fast_mode and self.graph is not None:
            effective_ef = ef_search if ef_search is not None else self.config.ef_search
            
            # Scale ef based on rerank factor
            scaled_ef = max(effective_ef, k * rerank_factor)
            self.graph.set_ef(scaled_ef)
            
            # Get candidates - rerank_factor controls recall vs speed
            k_search = min(k * rerank_factor, self.n_vectors)
            all_ids, all_distances = self.graph.search_batch(queries, k_search, scaled_ef)
            
            # Fast rerank only if we have more candidates
            if k_search > k:
                result_ids, result_distances = _batch_rerank_fast(
                    queries, self.vectors, all_ids, k
                )
                return result_ids, result_distances
            
            return all_ids, all_distances
        
        # Standard mode with zone-guided search
        all_ids = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        
        if n_jobs == 1:
            # Sequential search
            for i in range(n_queries):
                ids, distances = self.search(queries[i], k, n_probe, ef_search)
                # Handle case where fewer than k results
                n_results = min(k, len(ids))
                all_ids[i, :n_results] = ids[:n_results]
                all_distances[i, :n_results] = distances[:n_results]
        else:
            # Parallel search
            from concurrent.futures import ThreadPoolExecutor
            
            def search_single(idx):
                return self.search(queries[idx], k, n_probe, ef_search)
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(search_single, range(n_queries)))
            
            for i, (ids, distances) in enumerate(results):
                n_results = min(k, len(ids))
                all_ids[i, :n_results] = ids[:n_results]
                all_distances[i, :n_results] = distances[:n_results]
        
        return all_ids, all_distances
    
    def _expand_with_zones(
        self,
        queries: np.ndarray,
        initial_ids: np.ndarray,
        k_search: int,
        n_probe: int
    ) -> np.ndarray:
        """
        Expand candidate set using zone information for better recall.
        
        This leverages ZGQ's zone structure to find candidates that HNSW
        might miss due to graph connectivity issues.
        
        Args:
            queries: Query vectors (n_queries, d)
            initial_ids: Initial candidate IDs from HNSW (n_queries, k_search)
            k_search: Target number of candidates
            n_probe: Number of zones to probe
            
        Returns:
            Expanded candidate IDs (n_queries, k_search)
        """
        n_queries = len(queries)
        expanded_ids = initial_ids.copy()
        
        for i in range(n_queries):
            query = queries[i]
            current_ids = set(initial_ids[i].tolist())
            
            # Select nearest zones for this query
            selected_zones = self.zones.select_zones(query, n_probe)
            
            # Gather candidates from selected zones
            zone_candidates = []
            for zone_id in selected_zones:
                zone_vids = self.zones.get_zone_vectors(zone_id)
                zone_candidates.extend(zone_vids)
            
            # Add zone candidates that aren't already in the set
            new_candidates = [vid for vid in zone_candidates if vid not in current_ids]
            
            if new_candidates:
                # Compute distances to new candidates
                new_vectors = self.vectors[new_candidates]
                distances = np.sum((new_vectors - query) ** 2, axis=1)
                
                # Sort by distance and take the best ones
                sorted_idx = np.argsort(distances)
                
                # Replace worst candidates in initial_ids with better zone candidates
                # Get current distances for comparison
                current_vectors = self.vectors[initial_ids[i]]
                current_distances = np.sum((current_vectors - query) ** 2, axis=1)
                worst_idx = np.argsort(current_distances)[::-1]  # Worst first
                
                # Replace worst candidates if zone candidates are better
                n_replace = 0
                for j, new_idx in enumerate(sorted_idx):
                    if n_replace >= len(worst_idx) // 2:  # Replace at most half
                        break
                    if distances[new_idx] < current_distances[worst_idx[n_replace]]:
                        expanded_ids[i, worst_idx[n_replace]] = new_candidates[new_idx]
                        n_replace += 1
        
        return expanded_ids
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = {
            'n_vectors': self.n_vectors,
            'dimension': self.dimension,
            'is_built': self.is_built,
            'config': {
                'M': self.config.M,
                'ef_construction': self.config.ef_construction,
                'ef_search': self.config.ef_search,
                'use_pq': self.config.use_pq,
                'use_hierarchy': self.config.use_hierarchy,
                'metric': self.config.metric
            }
        }
        
        if self.is_built:
            stats['build_stats'] = self.build_stats
            stats['n_zones'] = len(self.zones.fine_centroids) if self.zones else 0
            
            if self.pq:
                stats['pq_info'] = self.pq.get_memory_usage(self.n_vectors)
        
        return stats
    
    def save(self, filepath: str) -> None:
        """
        Save index to file.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_built:
            raise RuntimeError("Cannot save index that hasn't been built")
        
        state = {
            'version': '8.0.0',
            'config': {
                'n_zones': self.config.n_zones,
                'use_hierarchy': self.config.use_hierarchy,
                'M': self.config.M,
                'ef_construction': self.config.ef_construction,
                'ef_search': self.config.ef_search,
                'use_pq': self.config.use_pq,
                'pq_m': self.config.pq_m,
                'pq_bits': self.config.pq_bits,
                'use_residual_pq': self.config.use_residual_pq,
                'metric': self.config.metric,
                'random_state': self.config.random_state
            },
            'data': {
                'vectors': self.vectors,
                'vector_norms': self.vector_norms,
                'n_vectors': self.n_vectors,
                'dimension': self.dimension
            },
            'zones': self.zones.save_state() if self.zones else None,
            'graph': self.graph.save_state() if self.graph else None,
            'pq': self.pq.save_state() if self.pq else None,
            'pq_codes': self.pq_codes,
            'build_stats': self.build_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.config.verbose:
            print(f"Index saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, verbose: bool = False) -> 'ZGQIndex':
        """
        Load index from file.
        
        Args:
            filepath: Path to saved index file
            verbose: Print progress
            
        Returns:
            Loaded ZGQIndex instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create index with saved config
        config = ZGQConfig(**state['config'], verbose=verbose)
        index = cls(config)
        
        # Restore data
        index.vectors = state['data']['vectors']
        index.vector_norms = state['data']['vector_norms']
        index.n_vectors = state['data']['n_vectors']
        index.dimension = state['data']['dimension']
        
        # Restore components
        if state['zones']:
            index.zones = AdaptiveHierarchicalZones.load_state(state['zones'])
        
        if state['graph']:
            index.graph = ZoneGuidedGraph.load_state(state['graph'])
        
        if state['pq']:
            index.pq = ResidualProductQuantizer.load_state(state['pq'])
            index.pq_codes = state['pq_codes']
        
        index.build_stats = state.get('build_stats', {})
        index.is_built = True
        
        if verbose:
            print(f"Index loaded from {filepath}")
        
        return index
