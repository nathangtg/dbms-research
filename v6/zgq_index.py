"""
ZGQ Index - Main Implementation for ZGQ V6
Integrates all modules to create complete Zonal Graph Quantization index

Architecture (following architecture_overview.md):
1. Zonal Partitioning (K-Means clustering)
2. Per-Zone HNSW Graphs
3. Product Quantization
4. Online Search with zone selection and parallel search
5. Aggregation and re-ranking

Mathematical Foundation:
- Build: O(N · log(N/Z) · M · d + N · k · d)
- Search: O(Z·d + n_probe·log(N/Z)·ef·m + k·d)

Reference: architecture_overview.md
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from distance_metrics import DistanceMetrics, PQDistanceMetrics
from product_quantization import ProductQuantizer
from zonal_partitioning import ZonalPartitioner
from hnsw_graph import HNSWGraph
from hnsw_zone_lib import HNSWZoneLib
from aggregation import ResultAggregator, SearchResult


class ZGQIndex:
    """
    Complete Zonal Graph Quantization index.
    
    Attributes:
        vectors: Full dataset of shape (N, d)
        Z: Number of zones
        M: HNSW max degree
        ef_construction: HNSW build parameter
        ef_search: HNSW search parameter
        n_probe: Number of zones to search
        use_pq: Whether to use Product Quantization
        m: PQ number of subspaces
        nbits: PQ bits per subspace
        
    Components:
        partitioner: ZonalPartitioner for zone creation
        zone_graphs: List of HNSWGraph objects
        pq: ProductQuantizer (if use_pq=True)
        pq_codes: PQ encoded vectors
        vector_norms_sq: Precomputed ||x||² for all vectors
    """
    
    def __init__(
        self,
        Z: int = 100,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        n_probe: int = 5,
        use_pq: bool = True,
        m: int = 16,
        nbits: int = 8,
        n_threads: int = 4,
        inner_hnsw_threads: int = 1,
        verbose: bool = True,
        store_vectors_dtype: str = "float32",
        pq_train_iter: int = 50,
        pq_train_batch: int = 2048,
        pq_n_init: int = 1,
        pq_encode_batch: int = 4096
    ):
        """
        Initialize ZGQ index parameters.
        
        Args:
            Z: Number of zones
            M: HNSW maximum degree
            ef_construction: HNSW build beam width
            ef_search: HNSW search beam width
            n_probe: Number of zones to search per query
            use_pq: Use Product Quantization for speed
            m: PQ number of subspaces (if use_pq=True)
            nbits: PQ bits per subspace (if use_pq=True)
            n_threads: Number of threads for parallel zone search
            inner_hnsw_threads: Threads used inside hnswlib per-zone search (to avoid oversubscription, default 1)
            verbose: Print progress information
            
        Reference: architecture_overview.md Section 1.2
        """
        # Parameters
        self.Z = Z
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.n_probe = n_probe
        self.use_pq = use_pq
        self.m = m
        self.nbits = nbits
        self.n_threads = n_threads
        self.inner_hnsw_threads = max(1, int(inner_hnsw_threads))
        self.verbose = verbose
        self.store_vectors_dtype = store_vectors_dtype
        self.pq_train_iter = int(pq_train_iter)
        self.pq_train_batch = int(pq_train_batch)
        self.pq_n_init = int(pq_n_init)
        self.pq_encode_batch = int(pq_encode_batch)
        
        # Data
        self.vectors = None
        self.N = None
        self.d = None
        
        # Components
        self.partitioner = None
        self.zone_graphs = []
        self.pq = None
        self.pq_codes = None
        self.vector_norms_sq = None
        
        # State
        self.is_built = False
        self.build_time = None
        self.build_stats = {}
    
    def build(self, vectors: np.ndarray) -> None:
        """
        Build complete ZGQ index.
        
        Algorithm (following architecture_overview.md Section 1.3):
        ---------------------------------------------------------------
        Step 1: Zonal Partitioning
                - K-Means clustering into Z zones
                - Complexity: O(K_iter · N · Z · d)
        
        Step 2: Per-Zone HNSW Construction
                - Build HNSW graph for each zone
                - Parallelizable across zones
                - Complexity: O(N · log(N/Z) · M · d)
        
        Step 3: Product Quantization (if enabled)
                - Train PQ codebooks
                - Encode all vectors
                - Complexity: O(K_iter · N · k · d + N · k · d)
        
        Step 4: Precompute Norms
                - Compute ||x||² for all vectors
                - Complexity: O(N · d)
        
        Total Build Complexity: O(N · log(N/Z) · M · d + N · k · d)
        
        Args:
            vectors: Dataset matrix of shape (N, d)
            
        Reference: architecture_overview.md Section 1.3
        """
        if self.verbose:
            print("\n" + "="*80)
            print("BUILDING ZGQ INDEX")
            print("="*80)
        
        start_time = time.time()

        # Keep float32 during build; optional downcast happens after build
        self.vectors = vectors.astype(np.float32)
        self.N, self.d = vectors.shape
        
        if self.verbose:
            print(f"\nDataset:")
            print(f"  Vectors (N): {self.N:,}")
            print(f"  Dimension (d): {self.d}")
            print(f"\nZGQ Parameters:")
            print(f"  Zones (Z): {self.Z}")
            print(f"  HNSW M: {self.M}")
            print(f"  HNSW ef_construction: {self.ef_construction}")
            print(f"  HNSW ef_search: {self.ef_search}")
            print(f"  n_probe: {self.n_probe}")
            print(f"  Use PQ: {self.use_pq}")
            if self.use_pq:
                print(f"  PQ m: {self.m}")
                print(f"  PQ nbits: {self.nbits} (k={2**self.nbits})")
            print(f"  Threads: {self.n_threads}")
        
        # ===== STEP 1: ZONAL PARTITIONING =====
        if self.verbose:
            print(f"\n{'='*80}")
            print("STEP 1: ZONAL PARTITIONING")
            print("="*80)
        
        self.partitioner = ZonalPartitioner(
            n_zones=self.Z,
            use_minibatch=True,
            verbose=self.verbose
        )
        self.partitioner.fit(vectors)
        
        self.build_stats['partitioning_time'] = self.partitioner.build_time
        
        # ===== STEP 2: PER-ZONE HNSW CONSTRUCTION =====
        if self.verbose:
            print(f"\n{'='*80}")
            print("STEP 2: PER-ZONE HNSW GRAPH CONSTRUCTION")
            print("="*80)
        
        zone_build_start = time.time()
        self._build_zone_graphs()
        zone_build_time = time.time() - zone_build_start
        
        self.build_stats['zone_graphs_time'] = zone_build_time
        self.build_stats['n_zones_built'] = len(self.zone_graphs)
        
        # ===== STEP 3: PRODUCT QUANTIZATION =====
        if self.use_pq:
            if self.verbose:
                print(f"\n{'='*80}")
                print("STEP 3: PRODUCT QUANTIZATION")
                print("="*80)
            
            pq_start = time.time()
            
            # Train PQ
            self.pq = ProductQuantizer(m=self.m, nbits=self.nbits, verbose=self.verbose)
            
            # Use subset for training (faster)
            n_train = min(50000, self.N)
            if n_train < self.N:
                train_indices = np.random.choice(self.N, n_train, replace=False)
                train_data = vectors[train_indices]
            else:
                train_data = vectors
            
            self.pq.train(
                train_data,
                n_iter=self.pq_train_iter,
                batch_size=self.pq_train_batch,
                n_init=self.pq_n_init
            )
            
            # Encode all vectors
            self.pq_codes = self.pq.encode(vectors, batch_size=self.pq_encode_batch)
            
            pq_time = time.time() - pq_start
            self.build_stats['pq_time'] = pq_time
        
        # ===== STEP 4: PRECOMPUTE NORMS =====
        if self.verbose:
            print(f"\n{'='*80}")
            print("STEP 4: PRECOMPUTING VECTOR NORMS")
            print("="*80)
        
        norm_start = time.time()
        self.vector_norms_sq = DistanceMetrics.precompute_vector_norms(vectors)
        norm_time = time.time() - norm_start
        
        if self.verbose:
            print(f"  Computed {self.N:,} norms in {norm_time:.3f}s")
        
        self.build_stats['norm_time'] = norm_time
        
        # ===== FINALIZE =====
        # Optionally reduce vector precision to save memory after building everything
        if self.store_vectors_dtype == "float16":
            try:
                self.vectors = self.vectors.astype(np.float16)
            except Exception:
                pass
        self.build_time = time.time() - start_time
        self.is_built = True
        
        if self.verbose:
            self._print_build_summary()
    
    def _build_zone_graphs(self) -> None:
        """
        Build HNSW graphs for all zones in parallel.
        
        Reference: hnsw_graphs.md Section 1.3
        """
        if self.verbose:
            print(f"  Building {self.Z} HNSW graphs with {self.n_threads} threads...")
        
        def build_single_zone(zone_id: int) -> Tuple[int, Optional[object]]:
            """Build graph for a single zone."""
            zone_vectors, global_indices = self.partitioner.get_zone_vectors(
                self.vectors, zone_id
            )
            
            # Only build if zone has enough vectors
            if len(zone_vectors) < 5:
                return zone_id, None
            
            # Prefer fast hnswlib-backed zone index
            try:
                graph = HNSWZoneLib(
                    vectors=zone_vectors,
                    M=self.M,
                    ef_construction=self.ef_construction,
                    verbose=False,
                )
                graph.build()
                # Avoid oversubscription: use 1 thread inside hnswlib by default when we parallelize zones
                try:
                    graph.set_search_threads(self.inner_hnsw_threads)
                except Exception:
                    pass
            except Exception:
                # Fallback to Python HNSWGraph if hnswlib not available
                graph = HNSWGraph(
                    vectors=zone_vectors,
                    M=self.M,
                    ef_construction=self.ef_construction,
                    verbose=False
                )
                graph.build()
            
            return zone_id, graph
        
        # Parallel zone graph construction
        self.zone_graphs = [None] * self.Z
        
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(build_single_zone, zone_id): zone_id
                for zone_id in range(self.Z)
            }
            
            completed = 0
            for future in as_completed(futures):
                zone_id, graph = future.result()
                self.zone_graphs[zone_id] = graph
                completed += 1
                
                if self.verbose and completed % max(1, self.Z // 10) == 0:
                    progress = completed / self.Z * 100
                    print(f"    Progress: {progress:.0f}% ({completed}/{self.Z} zones)", end='\r')
        
        if self.verbose:
            n_built = sum(1 for g in self.zone_graphs if g is not None)
            print(f"    Progress: 100% ({self.Z}/{self.Z} zones) ✓")
            print(f"  Successfully built {n_built}/{self.Z} zone graphs")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = None,
        ef_search: int = None,
        k_rerank: int = None,
        return_distances: bool = True,
        adaptive_probe: bool = False,
        min_probe: int = 2,
        max_probe: int = None,
        probe_margin_ratio: float = 0.15,
        adaptive_ef: bool = False,
        small_zone_threshold: int = 1200,
        ef_small: int = 32
    ) -> List[Tuple[int, float]]:
        """
        Search for k nearest neighbors.
        
        Algorithm (following architecture_overview.md Section 1.4):
        -------------------------------------------------------------
        Step 1: Zone Selection
                - Select n_probe nearest zones
                - Complexity: O(Z · d)
        
        Step 2: Precompute PQ Distance Table (if using PQ)
                - Build lookup table for query
                - Complexity: O(m · k · d/m) = O(k · d)
        
        Step 3: Parallel Zone Search
                - Search each selected zone with HNSW
                - Use PQ distances within graphs
                - Complexity: O(n_probe · log(N/Z) · ef · m)
        
        Step 4: Aggregate & Rerank
                - Deduplicate candidates
                - Rerank with exact distances
                - Complexity: O(k_rerank · d)
        
        Total Search Complexity: O(n_probe · log(N/Z) · ef · m + k · d)
        
        Args:
            query: Query vector of shape (d,)
            k: Number of nearest neighbors
            n_probe: Number of zones to search (default: self.n_probe)
            ef_search: HNSW beam width (default: self.ef_search)
            k_rerank: Number of candidates to rerank (default: 2*k)
            return_distances: Return distances with IDs
            
        Returns:
            List of (vector_id, distance) if return_distances=True
            List of vector_ids if return_distances=False
            
        Reference: architecture_overview.md Section 1.4
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")
        
        if n_probe is None:
            n_probe = self.n_probe
        if ef_search is None:
            ef_search = self.ef_search
        if k_rerank is None:
            k_rerank = min(max(2 * k, 50), self.N)
        
        query = query.astype(np.float32)
        
        # ===== STEP 1: ZONE SELECTION =====
        # If adaptive probing, compute top zones and adjust n_probe using centroid distance margins
        if adaptive_probe:
            # Compute distances to all centroids once
            centroid_dists = DistanceMetrics.euclidean_batch_squared(query, self.partitioner.centroids)
            # Sort zones by distance
            order = np.argsort(centroid_dists)
            # Decide dynamic n_probe based on margin between top-1 and top-2..k centroids
            if max_probe is None:
                max_probe = self.n_probe
            n_probe_eff = self._decide_adaptive_probe(centroid_dists[order], min_probe, max_probe, probe_margin_ratio)
            selected_zones = order[:n_probe_eff]
        else:
            selected_zones = self.partitioner.assign_to_zones(
                query[np.newaxis, :],
                n_probe=n_probe
            )[0]
        
        # ===== STEP 2: PRECOMPUTE PQ DISTANCE TABLE =====
        distance_table = None
        if self.use_pq:
            distance_table = PQDistanceMetrics.compute_distance_table(
                query,
                self.pq.get_codebooks(),
                self.m,
                2**self.nbits
            )
        
        # ===== STEP 3: PARALLEL ZONE SEARCH =====
        # Adjust ef per zone if adaptive_ef is enabled using zone size heuristic
        zone_ef = ef_search
        if adaptive_ef:
            # Compute effective ef for small zones to speed up search
            # We'll pass this down and let per-zone search decide based on zone size
            zone_ef = max(ef_small, min(ef_search, ef_search))

        all_candidates = self._parallel_zone_search(
            query=query,
            selected_zones=selected_zones,
            ef_search=zone_ef,
            distance_table=distance_table,
            k_local=k
        )
        
        # ===== STEP 4: AGGREGATE & RERANK =====
        results = ResultAggregator.aggregate_and_rerank(
            query=query,
            all_candidates=all_candidates,
            full_vectors=self.vectors,
            k=k,
            k_rerank=k_rerank,
            vector_norms_sq=self.vector_norms_sq,
            verbose=False
        )
        
        if return_distances:
            return [(r.vector_id, r.exact_distance) for r in results]
        else:
            return [r.vector_id for r in results]

    @staticmethod
    def _decide_adaptive_probe(
        sorted_centroid_dists: np.ndarray,
        min_probe: int,
        max_probe: int,
        margin_ratio: float
    ) -> int:
        """
        Decide number of zones to probe based on centroid distance margins.

        Heuristic: Start with min_probe, then include additional zones while
        the relative improvement from including next zone exceeds margin_ratio.

        Args:
            sorted_centroid_dists: distances to centroids sorted ascending
            min_probe: minimum zones to probe
            max_probe: maximum zones to probe
            margin_ratio: threshold on relative gap (d[i] - d[0]) / max(d[0], 1e-6)

        Returns:
            Chosen number of zones to probe
        """
        n = len(sorted_centroid_dists)
        if n == 0:
            return min_probe
        d0 = float(sorted_centroid_dists[0])
        chosen = max(1, int(min_probe))
        max_probe = max(chosen, int(max_probe))

        for i in range(1, min(n, max_probe)):
            gap = float(sorted_centroid_dists[i]) - d0
            # Normalize by d0 to be scale-aware; add epsilon to avoid zero-div
            rel = gap / (abs(d0) + 1e-12)
            if rel <= margin_ratio:
                chosen = i + 1
            else:
                break
        return int(max(min_probe, min(chosen, max_probe)))
    
    def _parallel_zone_search(
        self,
        query: np.ndarray,
        selected_zones: np.ndarray,
        ef_search: int,
        distance_table: Optional[np.ndarray],
        k_local: int
    ) -> List[Tuple[int, float, int]]:
        """
        Search selected zones in parallel.
        
        Args:
            query: Query vector
            selected_zones: Array of zone IDs to search
            ef_search: HNSW beam width
            distance_table: PQ distance table (if using PQ)
            k_local: Number of results per zone
            
        Returns:
            List of (global_vector_id, pq_distance, zone_id) tuples
            
        Reference: online_search.md Section 2.3
        """
        def search_single_zone(zone_id: int) -> List[Tuple[int, float, int]]:
            """Search within a single zone."""
            graph = self.zone_graphs[zone_id]
            
            if graph is None:
                return []
            
            # Get local results from HNSW
            # Both HNSWZoneLib and HNSWGraph expose search that returns
            # (distance, local_id) tuples per result.
            # Optionally reduce ef for small zones to speed up search
            ef_for_zone = ef_search
            try:
                # Access zone size from partitioner
                zone_size = len(self.partitioner.inverted_lists[zone_id])
                if zone_size < 1200:
                    ef_for_zone = max(16, min(ef_search, 32))
            except Exception:
                pass
            local_results = graph.search(query=query, k=k_local, ef_search=ef_for_zone)
            
            # Map to global IDs and compute PQ distances
            _, global_indices = self.partitioner.get_zone_vectors(
                self.vectors, zone_id
            )
            
            zone_results = []
            # NOTE: HNSWGraph.search returns a list of (distance, local_id)
            # tuples. We must unpack in that order to avoid swapping values.
            for dist, local_id in local_results:
                # Ensure local_id is an integer index
                local_id = int(local_id)
                if local_id < len(global_indices):
                    global_id = int(global_indices[local_id])
                    
                    # Compute PQ distance if available
                    if self.use_pq and distance_table is not None:
                        pq_dist = PQDistanceMetrics.pq_distance_single(
                            self.pq_codes[global_id],
                            distance_table
                        )
                    else:
                        # Use exact distance as fallback
                        # If exact distances from HNSWGraph are available,
                        # prefer them; otherwise compute from full vectors.
                        if dist is not None:
                            pq_dist = float(dist)
                        else:
                            pq_dist = DistanceMetrics.euclidean_squared(
                                query,
                                self.vectors[global_id]
                            )
                    
                    zone_results.append((global_id, pq_dist, zone_id))
            
            return zone_results
        
        # Parallel search across zones
        all_results = []
        
        if self.n_threads > 1:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [
                    executor.submit(search_single_zone, int(zone_id))
                    for zone_id in selected_zones
                ]
                
                for future in as_completed(futures):
                    zone_results = future.result()
                    all_results.extend(zone_results)
        else:
            # Sequential search
            for zone_id in selected_zones:
                zone_results = search_single_zone(int(zone_id))
                all_results.extend(zone_results)
        
        return all_results
    
    def _print_build_summary(self) -> None:
        """Print build statistics."""
        print(f"\n{'='*80}")
        print("BUILD SUMMARY")
        print("="*80)
        print(f"\nTotal build time: {self.build_time:.2f}s")
        print(f"\nTime breakdown:")
        print(f"  Zonal partitioning: {self.build_stats['partitioning_time']:.2f}s "
              f"({self.build_stats['partitioning_time']/self.build_time*100:.1f}%)")
        print(f"  Zone graphs: {self.build_stats['zone_graphs_time']:.2f}s "
              f"({self.build_stats['zone_graphs_time']/self.build_time*100:.1f}%)")
        if 'pq_time' in self.build_stats:
            print(f"  Product quantization: {self.build_stats['pq_time']:.2f}s "
                  f"({self.build_stats['pq_time']/self.build_time*100:.1f}%)")
        print(f"  Norm precomputation: {self.build_stats['norm_time']:.2f}s "
              f"({self.build_stats['norm_time']/self.build_time*100:.1f}%)")
        
        # Memory usage
        memory_mb = self.get_memory_usage()
        print(f"\nMemory usage: {memory_mb:.2f} MB")
        print(f"  Per vector: {memory_mb / self.N * 1024:.2f} KB")
        
        print(f"\n{'='*80}")
    
    def get_memory_usage(self) -> float:
        """
        Calculate total memory usage in MB.
        
        Returns:
            Memory in megabytes
        """
        memory = 0.0
        
        # Vectors
        memory += self.vectors.nbytes / (1024 ** 2)
        
        # Partitioner
        if self.partitioner:
            centroid_mem = self.partitioner.centroids.nbytes / (1024 ** 2)
            invlist_mem = sum(lst.nbytes for lst in self.partitioner.inverted_lists) / (1024 ** 2)
            memory += centroid_mem + invlist_mem
        
        # Zone graphs
        for graph in self.zone_graphs:
            if graph:
                memory += graph.get_memory_usage()
        
        # PQ codes
        if self.pq_codes is not None:
            memory += self.pq_codes.nbytes / (1024 ** 2)
        
        # PQ codebooks
        if self.pq:
            codebook_mem = sum(cb.nbytes for cb in self.pq.codebooks) / (1024 ** 2)
            memory += codebook_mem
        
        # Vector norms
        if self.vector_norms_sq is not None:
            memory += self.vector_norms_sq.nbytes / (1024 ** 2)
        
        return memory


# Validation test
if __name__ == "__main__":
    print("="*80)
    print("ZGQ Index - Integration Test")
    print("="*80)
    
    # Small test dataset
    N = 10000
    d = 128
    n_queries = 50
    k = 10
    
    print(f"\nGenerating test dataset...")
    print(f"  N = {N:,} vectors")
    print(f"  d = {d} dimensions")
    print(f"  n_queries = {n_queries}")
    print(f"  k = {k}")
    
    np.random.seed(42)
    vectors = np.random.randn(N, d).astype(np.float32)
    queries = np.random.randn(n_queries, d).astype(np.float32)
    
    # Build index
    print(f"\n{'='*80}")
    print("BUILDING ZGQ INDEX")
    print("="*80)
    
    zgq = ZGQIndex(
        Z=50,
        M=16,
        ef_construction=100,
        ef_search=50,
        n_probe=3,
        use_pq=True,
        m=16,
        nbits=8,
        n_threads=4,
        verbose=True
    )
    
    zgq.build(vectors)
    
    # Search
    print(f"\n{'='*80}")
    print("SEARCH PHASE")
    print("="*80)
    
    print(f"\nSearching {n_queries} queries...")
    search_times = []
    
    for i, query in enumerate(queries):
        start = time.time()
        results = zgq.search(query, k=k)
        search_times.append(time.time() - start)
        
        if i == 0:
            print(f"\nExample results for query 0:")
            for rank, (vec_id, dist) in enumerate(results[:5], 1):
                print(f"  {rank}. ID={vec_id:5d}, distance={dist:.4f}")
    
    avg_search_time = np.mean(search_times) * 1000
    p50_search_time = np.percentile(search_times, 50) * 1000
    p95_search_time = np.percentile(search_times, 95) * 1000
    p99_search_time = np.percentile(search_times, 99) * 1000
    
    print(f"\nSearch Performance:")
    print(f"  Mean latency: {avg_search_time:.2f} ms")
    print(f"  P50 latency: {p50_search_time:.2f} ms")
    print(f"  P95 latency: {p95_search_time:.2f} ms")
    print(f"  P99 latency: {p99_search_time:.2f} ms")
    print(f"  Throughput: {1000/avg_search_time:.0f} QPS")
    
    print(f"\n{'='*80}")
    print("✓ Integration test completed successfully")
    print("="*80)
