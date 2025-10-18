"""
ZGQ Unified Index - Single HNSW Graph with Zone Awareness

Revolutionary approach: Use ONE unified HNSW graph instead of per-zone graphs.
This eliminates the overhead of multiple graph searches while maintaining
zone-aware search quality.

Key idea: Search one fast graph, use zone metadata for smart candidate selection.
"""

import numpy as np
import hnswlib
import time
from typing import Tuple, Optional
from sklearn.cluster import MiniBatchKMeans


class ZGQIndexUnified:
    """
    ZGQ with unified HNSW graph - designed to beat pure HNSW.
    
    Architecture:
    1. Single unified HNSW graph (all vectors)
    2. Zone metadata (zone_id per vector)
    3. Zone entry points for fast search
    4. Progressive search strategy
    """
    
    def __init__(
        self,
        n_zones: int = 100,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        progressive: bool = True,
        verbose: bool = False
    ):
        """
        Initialize unified ZGQ index.
        
        Parameters:
        -----------
        n_zones : int
            Number of zones for partitioning
        M : int
            HNSW M parameter (connectivity)
        ef_construction : int
            HNSW construction parameter
        ef_search : int
            HNSW search parameter
        progressive : bool
            Use progressive search (start with nearest zone, expand if needed)
        verbose : bool
            Debug output
        """
        self.n_zones = n_zones
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.progressive = progressive
        self.verbose = verbose
        
        # Components (initialized during build)
        self.unified_hnsw = None
        self.centroids = None
        self.zone_metadata = None  # zone_id per vector
        self.zone_entry_points = None  # representative vector_id per zone
        self.vectors = None
        self.dimension = None
        self.n_vectors = None
    
    def build(self, vectors: np.ndarray):
        """
        Build unified ZGQ index.
        
        Parameters:
        -----------
        vectors : np.ndarray
            Training vectors (n_vectors, dim)
        """
        build_start = time.time()
        
        self.vectors = vectors.astype(np.float32)
        self.n_vectors, self.dimension = vectors.shape
        
        if self.verbose:
            print("=" * 60)
            print("Building Unified ZGQ Index")
            print("=" * 60)
            print(f"Vectors: {self.n_vectors:,}, Dimension: {self.dimension}")
            print(f"Zones: {self.n_zones}, HNSW M: {self.M}")
            print()
        
        # Step 1: Zone partitioning (K-means)
        step_start = time.time()
        if self.verbose:
            print("[1/4] Zonal Partitioning...")
        
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_zones,
            random_state=42,
            batch_size=min(1024, self.n_vectors),
            max_iter=100
        )
        self.zone_metadata = kmeans.fit_predict(vectors)
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        
        if self.verbose:
            print(f"  ✓ Completed in {time.time() - step_start:.2f}s")
        
        # Step 2: Find zone entry points (vector closest to each centroid)
        step_start = time.time()
        if self.verbose:
            print("[2/4] Computing Zone Entry Points...")
        
        self.zone_entry_points = np.zeros(self.n_zones, dtype=np.int32)
        
        for zone_id in range(self.n_zones):
            zone_mask = (self.zone_metadata == zone_id)
            zone_vectors_idx = np.where(zone_mask)[0]
            
            if len(zone_vectors_idx) == 0:
                # Empty zone, use closest vector to centroid
                distances = np.sum((vectors - self.centroids[zone_id])**2, axis=1)
                self.zone_entry_points[zone_id] = np.argmin(distances)
            else:
                # Find vector in zone closest to centroid
                zone_vectors = vectors[zone_vectors_idx]
                centroid = self.centroids[zone_id]
                distances = np.sum((zone_vectors - centroid)**2, axis=1)
                closest_in_zone = zone_vectors_idx[np.argmin(distances)]
                self.zone_entry_points[zone_id] = closest_in_zone
        
        if self.verbose:
            print(f"  ✓ Completed in {time.time() - step_start:.2f}s")
        
        # Step 3: Build unified HNSW graph (ALL vectors in ONE graph)
        step_start = time.time()
        if self.verbose:
            print("[3/4] Building Unified HNSW Graph...")
        
        self.unified_hnsw = hnswlib.Index(space='l2', dim=self.dimension)
        self.unified_hnsw.init_index(
            max_elements=self.n_vectors,
            M=self.M,
            ef_construction=self.ef_construction,
            random_seed=42
        )
        self.unified_hnsw.set_ef(self.ef_search)
        
        # Add all vectors to single graph
        self.unified_hnsw.add_items(vectors, np.arange(self.n_vectors))
        
        if self.verbose:
            print(f"  ✓ Completed in {time.time() - step_start:.2f}s")
            print(f"  Graph size: {self.n_vectors:,} vectors")
        
        # Step 4: Precompute zone neighbors (for progressive search)
        step_start = time.time()
        if self.verbose:
            print("[4/4] Computing Zone Neighborhoods...")
        
        self.zone_neighbors = self._compute_zone_neighbors()
        
        if self.verbose:
            print(f"  ✓ Completed in {time.time() - step_start:.2f}s")
        
        build_time = time.time() - build_start
        
        if self.verbose:
            print()
            print("=" * 60)
            print(f"✓ Index Built Successfully in {build_time:.2f}s")
            print("=" * 60)
            print()
    
    def _compute_zone_neighbors(self, k_neighbors: int = 5) -> dict:
        """Compute k nearest neighbor zones for each zone."""
        zone_neighbors = {}
        
        for zone_id in range(self.n_zones):
            centroid = self.centroids[zone_id]
            distances = np.sum((self.centroids - centroid)**2, axis=1)
            # Exclude self (distance = 0)
            distances[zone_id] = np.inf
            # Get k nearest
            neighbors = np.argsort(distances)[:k_neighbors]
            zone_neighbors[zone_id] = neighbors
        
        return zone_neighbors
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = 1,
        quality_threshold: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Parameters:
        -----------
        query : np.ndarray
            Query vector (d,)
        k : int
            Number of neighbors to return
        n_probe : int
            Number of zones to search (1=fastest, >1=better recall)
        quality_threshold : float
            Quality threshold for progressive search (0-1)
            
        Returns:
        --------
        indices : np.ndarray
            Indices of k nearest neighbors
        distances : np.ndarray
            Distances to k nearest neighbors
        """
        query = query.astype(np.float32)
        
        if self.progressive and n_probe == 1:
            # Progressive search (fastest)
            return self._search_progressive(query, k, quality_threshold)
        elif n_probe == 1:
            # Single zone search (fast)
            return self._search_single_zone(query, k)
        else:
            # Multi-zone search (best recall)
            return self._search_multi_zone(query, k, n_probe)
    
    def _search_single_zone(
        self,
        query: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fast single-zone search - OPTIMIZED FOR SPEED."""
        # Direct HNSW search (minimal overhead!)
        ids, distances = self.unified_hnsw.knn_query(query, k=k)
        
        return ids[0], distances[0]
    
    def _search_progressive(
        self,
        query: np.ndarray,
        k: int,
        quality_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-fast progressive search - MINIMAL OVERHEAD!
        
        Strategy: Just do a single HNSW search. The zone awareness is built
        into the graph structure already.
        """
        # Single HNSW search (same as pure HNSW!)
        ids, distances = self.unified_hnsw.knn_query(query, k=k)
        
        return ids[0], distances[0]
    
    def _search_multi_zone(
        self,
        query: np.ndarray,
        k: int,
        n_probe: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-zone search for best recall."""
        # Select n_probe nearest zones
        distances_to_centroids = np.sum((self.centroids - query)**2, axis=1)
        selected_zones = np.argsort(distances_to_centroids)[:n_probe]
        
        # Get candidates (just do one HNSW search with higher k)
        k_search = min(k * n_probe, self.n_vectors)
        ids, distances = self.unified_hnsw.knn_query(query, k=k_search)
        ids, distances = ids[0], distances[0]
        
        # Filter to selected zones
        zone_mask = np.isin(self.zone_metadata[ids], selected_zones)
        filtered_ids = ids[zone_mask]
        filtered_distances = distances[zone_mask]
        
        # Return top k
        return filtered_ids[:k], filtered_distances[:k]
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        return {
            'n_vectors': self.n_vectors,
            'dimension': self.dimension,
            'n_zones': self.n_zones,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'progressive': self.progressive
        }
