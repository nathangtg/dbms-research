#!/usr/bin/env python3
"""
Baseline Algorithms Module for ZGQ Comparison
==============================================

This module implements optimized baseline algorithms for fair comparison:
1. HNSW (Hierarchical Navigable Small World) - standalone implementation
2. IVF (Inverted File Index) - with optional Product Quantization

Both implementations use the same interface as ZGQIndex for consistent benchmarking.

Mathematical Foundations:
------------------------

HNSW Algorithm:
- Layer Selection: ℓ ~ ⌊-ln(uniform(0,1)) · m_L⌋, m_L = 1/ln(M)
- Search Complexity: O(log(N) · M · d) per query
- Build Complexity: O(N · log(N) · M · d)

IVF Algorithm:
- Partition into nlist clusters using K-Means
- Search Complexity: O(nprobe · (N/nlist) · d) per query
- Build Complexity: O(N · nlist · d) for clustering

Hardware Optimization:
---------------------
- Intel i5-12500H: 12 cores, AVX2 vectorization
- RTX 3050: Could use CUDA for distance computations (future work)
- 16GB RAM: Careful memory management for large datasets
- Threading: Concurrent indexing where applicable

References:
----------
[1] Yu. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest 
    neighbor search using Hierarchical Navigable Small World graphs," 2018.
[2] H. Jégou et al., "Product quantization for nearest neighbor search," 2011.
"""

import numpy as np
import heapq
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
import warnings

# Import our distance metrics
from distance_metrics import DistanceMetrics
from product_quantization import ProductQuantizer


@dataclass
class BuildStatistics:
    """Statistics collected during index building."""
    build_time: float
    indexing_time: float
    memory_bytes: int
    n_vectors: int
    dimension: int
    
    def __str__(self):
        return (f"BuildStats(vectors={self.n_vectors}, dim={self.dimension}, "
                f"build_time={self.build_time:.3f}s, memory={self.memory_bytes/1e6:.1f}MB)")


@dataclass
class SearchStatistics:
    """Statistics collected during search."""
    total_time: float
    distance_computations: int
    visited_nodes: int
    
    def __str__(self):
        return (f"SearchStats(time={self.total_time*1000:.3f}ms, "
                f"dist_comps={self.distance_computations}, visited={self.visited_nodes})")


class HNSWBaseline:
    """
    Optimized HNSW (Hierarchical Navigable Small World) baseline implementation.
    
    This is a standalone HNSW index for fair comparison with ZGQ. Unlike ZGQ's
    per-zone HNSW graphs, this maintains a single global HNSW structure.
    
    Mathematical Model:
    ------------------
    Layer Assignment:
    ℓ(v) = ⌊-ln(U(0,1)) · m_L⌋, where m_L = 1/ln(M)
    
    Search Algorithm:
    1. Start from entry point at top layer
    2. Greedy search to local minimum at each layer
    3. Descend to layer 0 and perform beam search
    
    Complexity:
    - Build: O(N · log(N) · M · d)
    - Search: O(log(N) · M · d) with high probability
    - Memory: O(N · M̄) where M̄ is average degree
    
    Parameters:
    ----------
    M : int, default=16
        Maximum number of connections per node (bidirectional edges)
    ef_construction : int, default=200
        Size of dynamic candidate list during construction
    ef_search : int, default=100
        Size of dynamic candidate list during search
    m_L : float, optional
        Level multiplier for layer selection (default: 1/ln(M))
    seed : int, default=42
        Random seed for reproducibility
        
    Attributes:
    ----------
    vectors : np.ndarray
        Database vectors [N, d]
    levels : List[int]
        Layer assignment for each vector
    graphs : List[Dict[int, List[int]]]
        Adjacency lists per layer [layer][node_id] = [neighbor_ids]
    entry_point : int
        Node ID of the entry point (highest layer)
    max_level : int
        Maximum layer in the graph
    """
    
    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        m_L: Optional[float] = None,
        seed: int = 42
    ):
        self.M = M
        self.M_max = M  # Max connections at layer 0
        self.M_max_higher = M  # Max connections at higher layers
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        
        # Layer multiplier for exponential decay
        if m_L is None:
            self.m_L = 1.0 / np.log(self.M)
        else:
            self.m_L = m_L
            
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Index structures
        self.vectors = None
        self.levels = []  # Level assignment per node
        self.graphs = []  # Adjacency lists per layer
        self.entry_point = None
        self.max_level = -1
        
        self.distance_metrics = DistanceMetrics()
        
        # Statistics
        self.build_stats = None
        
    def _select_level(self) -> int:
        """
        Select layer for new element using exponential decay probability.
        
        Formula: ℓ = ⌊-ln(uniform(0,1)) · m_L⌋
        
        Returns:
        -------
        level : int
            Selected layer index (0 = bottom layer)
        """
        uniform_sample = self.rng.uniform(0, 1)
        # Avoid log(0)
        if uniform_sample < 1e-9:
            uniform_sample = 1e-9
        level = int(-np.log(uniform_sample) * self.m_L)
        return level
        
    def build(self, vectors: np.ndarray) -> BuildStatistics:
        """
        Build HNSW index from database vectors.
        
        Algorithm:
        ---------
        For each vector v_i:
        1. Select layer ℓ using exponential distribution
        2. Insert at all layers 0..ℓ:
           - Find M nearest neighbors using beam search
           - Add bidirectional edges
           - Prune to maintain degree bounds
        
        Complexity: O(N · log(N) · M · d)
        
        Parameters:
        ----------
        vectors : np.ndarray, shape [N, d]
            Database vectors to index
            
        Returns:
        -------
        stats : BuildStatistics
            Build time and memory statistics
        """
        start_time = time.time()
        
        self.vectors = vectors.astype(np.float32)
        N, d = vectors.shape
        
        print(f"\n{'='*80}")
        print(f"Building HNSW Index")
        print(f"{'='*80}")
        print(f"Parameters: M={self.M}, ef_construction={self.ef_construction}")
        print(f"Database: N={N:,} vectors, d={d} dimensions")
        
        # Initialize entry point with first vector
        self.entry_point = 0
        self.max_level = self._select_level()
        self.levels = [self.max_level]
        
        # Initialize graph structures
        self.graphs = [defaultdict(list) for _ in range(self.max_level + 1)]
        
        # Insert remaining vectors
        for idx in range(1, N):
            if idx % 10000 == 0:
                elapsed = time.time() - start_time
                rate = idx / elapsed
                eta = (N - idx) / rate
                print(f"  Inserted {idx:,}/{N:,} vectors ({idx/N*100:.1f}%) - "
                      f"{rate:.0f} vec/s - ETA: {eta:.1f}s")
            
            self._insert(idx)
        
        build_time = time.time() - start_time
        memory_bytes = self._estimate_memory()
        
        # Compute statistics
        avg_degree = self._compute_average_degree()
        
        print(f"\n{'='*80}")
        print(f"HNSW Build Complete")
        print(f"{'='*80}")
        print(f"Build time: {build_time:.3f}s ({N/build_time:.0f} vec/s)")
        print(f"Max level: {self.max_level}")
        print(f"Average degree: {avg_degree:.2f}")
        print(f"Memory usage: {memory_bytes/1e6:.1f} MB")
        
        self.build_stats = BuildStatistics(
            build_time=build_time,
            indexing_time=build_time,
            memory_bytes=memory_bytes,
            n_vectors=N,
            dimension=d
        )
        
        return self.build_stats
        
    def _insert(self, idx: int):
        """
        Insert vector at index idx into the HNSW graph.
        
        Parameters:
        ----------
        idx : int
            Index of vector to insert
        """
        query = self.vectors[idx]
        level = self._select_level()
        self.levels.append(level)
        
        # Update max level if needed
        if level > self.max_level:
            self.max_level = level
            # Extend graph structures
            while len(self.graphs) <= level:
                self.graphs.append(defaultdict(list))
        
        # Search for nearest neighbors from top to target layer
        ep = [self.entry_point]
        
        # Layers above target level: greedy search
        for lc in range(self.max_level, level, -1):
            if lc < len(self.graphs):
                nearest = self._search_layer(query, ep, 1, lc)
                ep = [nearest[0][1]]  # Continue from nearest
        
        # Layers from target level to 0: beam search and connect
        for lc in range(min(level, self.max_level), -1, -1):
            # Beam search to find candidates
            candidates = self._search_layer(query, ep, self.ef_construction, lc)
            
            # Select M nearest neighbors
            M = self.M_max if lc == 0 else self.M_max_higher
            neighbors = self._select_neighbors(candidates, M)
            
            # Add bidirectional edges
            for _, neighbor_idx in neighbors:
                self.graphs[lc][idx].append(neighbor_idx)
                self.graphs[lc][neighbor_idx].append(idx)
                
                # Prune neighbor's connections if needed
                if len(self.graphs[lc][neighbor_idx]) > M:
                    self._prune_connections(neighbor_idx, M, lc)
            
            ep = [n[1] for n in neighbors]
        
        # Update entry point if this node is higher
        if level > self.levels[self.entry_point]:
            self.entry_point = idx
            
    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: List[int],
        num_closest: int,
        layer: int
    ) -> List[Tuple[float, int]]:
        """
        Beam search at a specific layer.
        
        Parameters:
        ----------
        query : np.ndarray, shape [d]
            Query vector
        entry_points : List[int]
            Starting nodes for search
        num_closest : int
            Number of closest nodes to return
        layer : int
            Layer index to search
            
        Returns:
        -------
        candidates : List[Tuple[float, int]]
            List of (distance, node_id) sorted by distance
        """
        visited = set()
        candidates = []
        w = []  # Dynamic list of best candidates
        
        # Initialize with entry points
        for ep in entry_points:
            if ep >= len(self.vectors):
                continue
            dist = self.distance_metrics.euclidean_squared(query, self.vectors[ep])
            heapq.heappush(candidates, (-dist, ep))  # Max heap
            heapq.heappush(w, (dist, ep))  # Min heap
            visited.add(ep)
        
        # Beam search expansion
        while candidates:
            current_dist, current = heapq.heappop(candidates)
            current_dist = -current_dist  # Convert back from max heap
            
            # Stop if current is farther than furthest in result set
            if current_dist > -w[0][0] if len(w) >= num_closest else False:
                break
            
            # Explore neighbors
            if layer < len(self.graphs) and current in self.graphs[layer]:
                for neighbor in self.graphs[layer][current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist = self.distance_metrics.euclidean_squared(
                            query, self.vectors[neighbor]
                        )
                        
                        if dist < -w[0][0] or len(w) < num_closest:
                            heapq.heappush(candidates, (-dist, neighbor))
                            heapq.heappush(w, (dist, neighbor))
                            
                            # Keep only num_closest best
                            if len(w) > num_closest:
                                heapq.heappop(w)
        
        # Return as sorted list
        result = sorted(w, key=lambda x: x[0])
        return result
        
    def _select_neighbors(
        self,
        candidates: List[Tuple[float, int]],
        M: int
    ) -> List[Tuple[float, int]]:
        """
        Select M neighbors from candidates (simple heuristic).
        
        Parameters:
        ----------
        candidates : List[Tuple[float, int]]
            Candidate neighbors sorted by distance
        M : int
            Number of neighbors to select
            
        Returns:
        -------
        neighbors : List[Tuple[float, int]]
            Selected neighbors
        """
        # Simple: return M closest
        return candidates[:M]
        
    def _prune_connections(self, idx: int, M: int, layer: int):
        """
        Prune connections of node to maintain degree bound.
        
        Parameters:
        ----------
        idx : int
            Node ID to prune
        M : int
            Maximum number of connections
        layer : int
            Layer index
        """
        neighbors = self.graphs[layer][idx]
        if len(neighbors) <= M:
            return
        
        # Compute distances to all neighbors
        node_vec = self.vectors[idx]
        neighbor_dists = [
            (self.distance_metrics.euclidean_squared(node_vec, self.vectors[n]), n)
            for n in neighbors
        ]
        
        # Keep M closest
        neighbor_dists.sort()
        self.graphs[layer][idx] = [n for _, n in neighbor_dists[:M]]
        
    def search(
        self,
        queries: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, SearchStatistics]:
        """
        Search for k nearest neighbors for each query.
        
        Algorithm:
        ---------
        1. Start from entry point at top layer
        2. Greedy descent to layer 0
        3. Beam search at layer 0 with ef_search candidates
        4. Return k nearest
        
        Complexity: O(Q · log(N) · M · d)
        
        Parameters:
        ----------
        queries : np.ndarray, shape [Q, d]
            Query vectors
        k : int, default=10
            Number of nearest neighbors to return
            
        Returns:
        -------
        indices : np.ndarray, shape [Q, k]
            Indices of k nearest neighbors
        distances : np.ndarray, shape [Q, k]
            Distances to k nearest neighbors
        stats : SearchStatistics
            Search performance statistics
        """
        start_time = time.time()
        Q = queries.shape[0]
        
        all_indices = np.zeros((Q, k), dtype=np.int32)
        all_distances = np.zeros((Q, k), dtype=np.float32)
        
        total_dist_comps = 0
        total_visited = 0
        
        for q_idx, query in enumerate(queries):
            # Start from entry point
            ep = [self.entry_point]
            
            # Greedy search from top to layer 1
            for lc in range(self.max_level, 0, -1):
                if lc < len(self.graphs):
                    nearest = self._search_layer(query, ep, 1, lc)
                    if nearest:
                        ep = [nearest[0][1]]
            
            # Beam search at layer 0
            candidates = self._search_layer(query, ep, self.ef_search, 0)
            
            # Extract top k
            top_k = candidates[:k]
            
            # Fill results
            for i, (dist, idx) in enumerate(top_k):
                all_indices[q_idx, i] = idx
                all_distances[q_idx, i] = dist
            
            # Statistics
            total_visited += len(candidates)
            # Distance computations ≈ visited nodes
            total_dist_comps += len(candidates)
        
        search_time = time.time() - start_time
        
        stats = SearchStatistics(
            total_time=search_time,
            distance_computations=total_dist_comps,
            visited_nodes=total_visited
        )
        
        return all_indices, all_distances, stats
        
    def _compute_average_degree(self) -> float:
        """Compute average node degree across all layers."""
        total_edges = 0
        total_nodes = 0
        
        for layer_graph in self.graphs:
            for neighbors in layer_graph.values():
                total_edges += len(neighbors)
                total_nodes += 1
        
        return total_edges / total_nodes if total_nodes > 0 else 0.0
        
    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        memory = 0
        
        # Vectors
        if self.vectors is not None:
            memory += self.vectors.nbytes
        
        # Levels
        memory += len(self.levels) * 8  # int64
        
        # Graphs (rough estimate)
        for layer_graph in self.graphs:
            for neighbors in layer_graph.values():
                memory += len(neighbors) * 8  # int64 per neighbor
        
        return memory


class IVFBaseline:
    """
    Optimized IVF (Inverted File Index) baseline implementation.
    
    IVF partitions the database into nlist clusters using K-Means, then searches
    only nprobe nearest clusters for each query.
    
    Mathematical Model:
    ------------------
    Clustering:
    C* = argmin_C Σᵢ Σⱼ ||xᵢ - μⱼ||² where xᵢ ∈ Cⱼ
    
    Search:
    1. Find nprobe nearest cluster centroids to query
    2. Search all vectors in selected clusters
    3. Return global top-k
    
    Complexity:
    - Build: O(N · nlist · d) for K-Means
    - Search: O(nlist · d + nprobe · (N/nlist) · d) per query
    - Memory: O(N · d) for vectors + O(nlist · d) for centroids
    
    Parameters:
    ----------
    nlist : int, default=100
        Number of clusters (partitions)
    nprobe : int, default=10
        Number of clusters to search per query
    use_pq : bool, default=False
        Whether to use Product Quantization for compression
    m : int, default=16
        Number of subspaces for PQ (if use_pq=True)
    nbits : int, default=8
        Bits per subspace for PQ (if use_pq=True)
    seed : int, default=42
        Random seed for reproducibility
        
    Attributes:
    ----------
    vectors : np.ndarray
        Database vectors [N, d]
    centroids : np.ndarray
        Cluster centroids [nlist, d]
    inverted_lists : Dict[int, List[int]]
        Mapping from cluster ID to vector IDs
    pq : ProductQuantizer, optional
        Product quantizer if use_pq=True
    pq_codes : np.ndarray, optional
        PQ codes if use_pq=True
    """
    
    def __init__(
        self,
        nlist: int = 100,
        nprobe: int = 10,
        use_pq: bool = False,
        m: int = 16,
        nbits: int = 8,
        seed: int = 42
    ):
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_pq = use_pq
        self.m = m
        self.nbits = nbits
        self.seed = seed
        
        # Index structures
        self.vectors = None
        self.centroids = None
        self.inverted_lists = defaultdict(list)
        
        # Product Quantization (optional)
        self.pq = None
        self.pq_codes = None
        
        self.distance_metrics = DistanceMetrics()
        
        # Statistics
        self.build_stats = None
        
    def build(self, vectors: np.ndarray) -> BuildStatistics:
        """
        Build IVF index from database vectors.
        
        Algorithm:
        ---------
        1. Run K-Means to find nlist cluster centroids
        2. Assign each vector to nearest centroid
        3. Build inverted lists (cluster → vector IDs)
        4. Optional: Train PQ and encode vectors
        
        Complexity: O(N · nlist · d) for clustering
        
        Parameters:
        ----------
        vectors : np.ndarray, shape [N, d]
            Database vectors to index
            
        Returns:
        -------
        stats : BuildStatistics
            Build time and memory statistics
        """
        start_time = time.time()
        
        self.vectors = vectors.astype(np.float32)
        N, d = vectors.shape
        
        print(f"\n{'='*80}")
        print(f"Building IVF Index")
        print(f"{'='*80}")
        print(f"Parameters: nlist={self.nlist}, nprobe={self.nprobe}, use_pq={self.use_pq}")
        if self.use_pq:
            print(f"PQ Parameters: m={self.m}, nbits={self.nbits}")
        print(f"Database: N={N:,} vectors, d={d} dimensions")
        
        # Step 1: K-Means clustering
        print(f"\nStep 1: Running K-Means clustering...")
        clustering_start = time.time()
        
        kmeans = MiniBatchKMeans(
            n_clusters=self.nlist,
            random_state=self.seed,
            batch_size=min(1024, N),
            max_iter=100,
            verbose=0
        )
        cluster_labels = kmeans.fit_predict(vectors)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        
        clustering_time = time.time() - clustering_start
        print(f"  K-Means completed in {clustering_time:.3f}s")
        
        # Step 2: Build inverted lists
        print(f"\nStep 2: Building inverted lists...")
        for vec_id, cluster_id in enumerate(cluster_labels):
            self.inverted_lists[cluster_id].append(vec_id)
        
        # Compute balance statistics
        list_sizes = [len(lst) for lst in self.inverted_lists.values()]
        avg_size = np.mean(list_sizes)
        max_size = np.max(list_sizes)
        min_size = np.min(list_sizes)
        
        print(f"  Cluster sizes: avg={avg_size:.1f}, max={max_size}, min={min_size}")
        
        # Step 3: Optional Product Quantization
        indexing_time = 0
        if self.use_pq:
            print(f"\nStep 3: Training Product Quantization...")
            pq_start = time.time()
            
            self.pq = ProductQuantizer(m=self.m, nbits=self.nbits)
            self.pq.train(vectors)
            self.pq_codes = self.pq.encode(vectors)
            
            indexing_time = time.time() - pq_start
            print(f"  PQ training completed in {indexing_time:.3f}s")
            
            # Compression statistics
            original_bytes = N * d * 4  # float32
            compressed_bytes = N * self.m * 1  # uint8 codes
            compression_ratio = original_bytes / compressed_bytes
            print(f"  Compression: {original_bytes/1e6:.1f}MB → {compressed_bytes/1e6:.1f}MB "
                  f"(ratio: {compression_ratio:.1f}×)")
        
        build_time = time.time() - start_time
        memory_bytes = self._estimate_memory()
        
        print(f"\n{'='*80}")
        print(f"IVF Build Complete")
        print(f"{'='*80}")
        print(f"Build time: {build_time:.3f}s ({N/build_time:.0f} vec/s)")
        print(f"Memory usage: {memory_bytes/1e6:.1f} MB")
        
        self.build_stats = BuildStatistics(
            build_time=build_time,
            indexing_time=indexing_time,
            memory_bytes=memory_bytes,
            n_vectors=N,
            dimension=d
        )
        
        return self.build_stats
        
    def search(
        self,
        queries: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, SearchStatistics]:
        """
        Search for k nearest neighbors for each query.
        
        Algorithm:
        ---------
        1. Find nprobe nearest cluster centroids
        2. Search vectors in selected clusters:
           - If use_pq: Compute PQ asymmetric distances
           - Else: Compute exact distances
        3. Return global top-k
        
        Complexity: O(nlist · d + nprobe · (N/nlist) · d) per query
        
        Parameters:
        ----------
        queries : np.ndarray, shape [Q, d]
            Query vectors
        k : int, default=10
            Number of nearest neighbors to return
            
        Returns:
        -------
        indices : np.ndarray, shape [Q, k]
            Indices of k nearest neighbors
        distances : np.ndarray, shape [Q, k]
            Distances to k nearest neighbors
        stats : SearchStatistics
            Search performance statistics
        """
        start_time = time.time()
        Q = queries.shape[0]
        
        all_indices = np.zeros((Q, k), dtype=np.int32)
        all_distances = np.zeros((Q, k), dtype=np.float32)
        
        total_dist_comps = 0
        total_visited = 0
        
        for q_idx, query in enumerate(queries):
            # Step 1: Find nprobe nearest centroids
            centroid_dists = self.distance_metrics.euclidean_batch_squared(
                query, self.centroids
            )
            nearest_clusters = np.argsort(centroid_dists)[:self.nprobe]
            
            total_dist_comps += self.nlist
            
            # Step 2: Search in selected clusters
            candidates = []
            
            for cluster_id in nearest_clusters:
                if cluster_id not in self.inverted_lists:
                    continue
                    
                vec_ids = self.inverted_lists[cluster_id]
                total_visited += len(vec_ids)
                
                if self.use_pq:
                    # PQ asymmetric distance
                    from distance_metrics import PQDistanceMetrics
                    distance_table = PQDistanceMetrics.compute_distance_table(
                        query,
                        self.pq.get_codebooks(),
                        self.m,
                        self.pq.k
                    )
                    for vec_id in vec_ids:
                        pq_dist = PQDistanceMetrics.pq_distance_single(
                            self.pq_codes[vec_id], distance_table
                        )
                        candidates.append((pq_dist, vec_id))
                else:
                    # Exact distance
                    cluster_vecs = self.vectors[vec_ids]
                    dists = self.distance_metrics.euclidean_batch_squared(
                        query, cluster_vecs
                    )
                    
                    for dist, vec_id in zip(dists, vec_ids):
                        candidates.append((dist, vec_id))
                    
                    total_dist_comps += len(vec_ids)
            
            # Step 3: Get top-k
            candidates.sort()
            top_k = candidates[:k]
            
            # Fill results
            for i, (dist, idx) in enumerate(top_k):
                all_indices[q_idx, i] = idx
                all_distances[q_idx, i] = dist
        
        search_time = time.time() - start_time
        
        stats = SearchStatistics(
            total_time=search_time,
            distance_computations=total_dist_comps,
            visited_nodes=total_visited
        )
        
        return all_indices, all_distances, stats
        
    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        memory = 0
        
        # Vectors (if not using PQ)
        if self.vectors is not None and not self.use_pq:
            memory += self.vectors.nbytes
        
        # Centroids
        if self.centroids is not None:
            memory += self.centroids.nbytes
        
        # Inverted lists (rough estimate)
        memory += len(self.inverted_lists) * 100  # Dict overhead
        for lst in self.inverted_lists.values():
            memory += len(lst) * 8  # int64 per entry
        
        # PQ structures
        if self.use_pq and self.pq is not None:
            # Compute PQ memory manually
            codebooks = self.pq.get_codebooks()
            for cb in codebooks:
                memory += cb.nbytes
            if self.pq_codes is not None:
                memory += self.pq_codes.nbytes
        
        return memory


# ============================================================================
# Validation and Testing
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("Baseline Algorithms Validation")
    print("="*80)
    
    # Generate synthetic dataset
    np.random.seed(42)
    N = 10000
    d = 128
    Q = 100
    k = 10
    
    print(f"\nGenerating synthetic dataset...")
    print(f"  Database: N={N:,}, d={d}")
    print(f"  Queries: Q={Q}, k={k}")
    
    # Database vectors (Gaussian clusters)
    n_clusters = 20
    vectors = []
    for i in range(n_clusters):
        center = np.random.randn(d) * 5
        cluster_vecs = center + np.random.randn(N // n_clusters, d) * 0.5
        vectors.append(cluster_vecs)
    vectors = np.vstack(vectors).astype(np.float32)
    
    # Query vectors
    queries = vectors[np.random.choice(N, Q, replace=False)]
    queries += np.random.randn(Q, d) * 0.1  # Add noise
    
    # Compute ground truth (brute force)
    print(f"\nComputing ground truth...")
    dist_metrics = DistanceMetrics()
    gt_dists = []
    for query in queries:
        dists = dist_metrics.euclidean_batch_squared(query, vectors)
        gt_dists.append(dists)
    gt_dists = np.array(gt_dists)
    gt_indices = np.argsort(gt_dists, axis=1)[:, :k]
    
    # ========================================================================
    # Test 1: HNSW Baseline
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"Test 1: HNSW Baseline")
    print(f"{'='*80}")
    
    hnsw = HNSWBaseline(M=16, ef_construction=100, ef_search=50)
    
    # Build
    build_stats = hnsw.build(vectors)
    print(f"\n{build_stats}")
    
    # Search
    print(f"\nSearching {Q} queries...")
    indices, distances, search_stats = hnsw.search(queries, k=k)
    print(f"{search_stats}")
    
    # Compute recall
    recalls = []
    for i in range(Q):
        recall = len(set(indices[i]) & set(gt_indices[i])) / k
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    print(f"\nRecall@{k}: {avg_recall:.4f}")
    print(f"Average latency: {search_stats.total_time/Q*1000:.3f} ms/query")
    
    # ========================================================================
    # Test 2: IVF Baseline (without PQ)
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"Test 2: IVF Baseline (exact distances)")
    print(f"{'='*80}")
    
    ivf = IVFBaseline(nlist=100, nprobe=10, use_pq=False)
    
    # Build
    build_stats = ivf.build(vectors)
    print(f"\n{build_stats}")
    
    # Search
    print(f"\nSearching {Q} queries...")
    indices, distances, search_stats = ivf.search(queries, k=k)
    print(f"{search_stats}")
    
    # Compute recall
    recalls = []
    for i in range(Q):
        recall = len(set(indices[i]) & set(gt_indices[i])) / k
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    print(f"\nRecall@{k}: {avg_recall:.4f}")
    print(f"Average latency: {search_stats.total_time/Q*1000:.3f} ms/query")
    
    # ========================================================================
    # Test 3: IVF with PQ
    # ========================================================================
    print(f"\n{'='*80}")
    print(f"Test 3: IVF Baseline (with PQ)")
    print(f"{'='*80}")
    
    ivf_pq = IVFBaseline(nlist=100, nprobe=10, use_pq=True, m=16, nbits=8)
    
    # Build
    build_stats = ivf_pq.build(vectors)
    print(f"\n{build_stats}")
    
    # Search
    print(f"\nSearching {Q} queries...")
    indices, distances, search_stats = ivf_pq.search(queries, k=k)
    print(f"{search_stats}")
    
    # Compute recall
    recalls = []
    for i in range(Q):
        recall = len(set(indices[i]) & set(gt_indices[i])) / k
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    print(f"\nRecall@{k}: {avg_recall:.4f}")
    print(f"Average latency: {search_stats.total_time/Q*1000:.3f} ms/query")
    
    print(f"\n{'='*80}")
    print(f"Baseline Algorithms Validation Complete")
    print(f"{'='*80}")
    print(f"\nSummary:")
    print(f"  HNSW: Recall={avg_recall:.4f}, Good for high recall scenarios")
    print(f"  IVF: Fast search with adjustable recall via nprobe")
    print(f"  IVF+PQ: Memory-efficient with small recall tradeoff")
    print(f"\nNext: Run comprehensive benchmarks comparing ZGQ vs baselines")
