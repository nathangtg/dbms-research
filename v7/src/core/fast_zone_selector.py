"""
Fast Zone Selection using HNSW on Centroids

Instead of computing distances to all centroids (O(n_zones * d)),
use HNSW graph on centroids for O(log n_zones) zone selection.
"""

import numpy as np
import hnswlib
from typing import Optional


class FastZoneSelector:
    """
    Fast zone selection using HNSW index on centroids.
    
    This provides logarithmic-time zone selection instead of linear scan.
    """
    
    def __init__(
        self,
        centroids: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        verbose: bool = False
    ):
        """
        Initialize fast zone selector.
        
        Parameters:
        -----------
        centroids : np.ndarray
            Centroid vectors (n_zones, d)
        M : int
            HNSW M parameter (connectivity)
        ef_construction : int
            HNSW construction parameter
        ef_search : int
            HNSW search parameter
        verbose : bool
            Debug output
        """
        self.centroids = centroids.astype(np.float32)
        self.n_zones, self.dim = centroids.shape
        self.verbose = verbose
        
        # Build HNSW index on centroids
        self.index = hnswlib.Index(space='l2', dim=self.dim)
        self.index.init_index(
            max_elements=self.n_zones,
            M=M,
            ef_construction=ef_construction,
            random_seed=42
        )
        self.index.set_ef(ef_search)
        
        # Add centroids
        self.index.add_items(self.centroids, np.arange(self.n_zones))
        
        if verbose:
            print(f"  Fast zone selector built: {self.n_zones} zones")
    
    def select_zones(self, query: np.ndarray, n_probe: int) -> np.ndarray:
        """
        Select nearest zones using HNSW.
        
        Parameters:
        -----------
        query : np.ndarray
            Query vector (d,)
        n_probe : int
            Number of zones to select
            
        Returns:
        --------
        zone_ids : np.ndarray
            Indices of nearest zones (n_probe,)
        """
        query = query.astype(np.float32)
        
        # Clamp n_probe
        n_probe_clamped = min(n_probe, self.n_zones)
        
        # HNSW search - much faster than linear scan
        zone_ids, _ = self.index.knn_query(query, k=n_probe_clamped)
        
        return zone_ids[0]  # hnswlib returns shape (1, k)
    
    def select_zones_batch(
        self, 
        queries: np.ndarray, 
        n_probe: int
    ) -> np.ndarray:
        """
        Select zones for multiple queries.
        
        Parameters:
        -----------
        queries : np.ndarray
            Query vectors (n_queries, d)
        n_probe : int
            Number of zones per query
            
        Returns:
        --------
        zone_ids : np.ndarray
            Zone indices (n_queries, n_probe)
        """
        queries = queries.astype(np.float32)
        n_probe_clamped = min(n_probe, self.n_zones)
        
        # Batch HNSW search
        zone_ids, _ = self.index.knn_query(queries, k=n_probe_clamped)
        
        return zone_ids
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics."""
        return {
            'centroids_mb': self.centroids.nbytes / 1024 / 1024,
            'index_size': self.index.get_current_count(),
            'total_mb': (self.centroids.nbytes / 1024 / 1024) + 0.1  # rough estimate
        }


class HybridZoneSelector:
    """
    Hybrid zone selector: use linear scan for small n_zones, HNSW for large.
    
    Automatically chooses the fastest method based on number of zones.
    """
    
    def __init__(
        self,
        centroids: np.ndarray,
        threshold: int = 200,
        use_hnsw: bool = True,
        verbose: bool = False
    ):
        """
        Initialize hybrid zone selector.
        
        Parameters:
        -----------
        centroids : np.ndarray
            Centroid vectors (n_zones, d)
        threshold : int
            Use HNSW if n_zones > threshold
        use_hnsw : bool
            Force HNSW usage (or disable it)
        verbose : bool
            Debug output
        """
        self.centroids = centroids.astype(np.float32)
        self.n_zones, self.dim = centroids.shape
        self.threshold = threshold
        self.verbose = verbose
        
        # Decide whether to use HNSW
        self.use_fast_selector = use_hnsw and (self.n_zones > threshold)
        
        if self.use_fast_selector:
            self.fast_selector = FastZoneSelector(
                centroids=centroids,
                M=16,
                ef_construction=200,
                ef_search=50,
                verbose=verbose
            )
        else:
            self.fast_selector = None
            if verbose:
                print(f"  Using linear zone selection ({self.n_zones} zones)")
    
    def select_zones(self, query: np.ndarray, n_probe: int) -> np.ndarray:
        """Select nearest zones."""
        if self.use_fast_selector:
            return self.fast_selector.select_zones(query, n_probe)
        else:
            return self._select_zones_linear(query, n_probe)
    
    def _select_zones_linear(self, query: np.ndarray, n_probe: int) -> np.ndarray:
        """Linear scan zone selection (for small n_zones)."""
        query = query.astype(np.float32)
        
        # Compute distances
        diff = self.centroids - query
        distances = np.sum(diff * diff, axis=1)
        
        if n_probe >= len(distances):
            return np.arange(len(distances))
        
        # argpartition is O(n), faster than full sort
        indices = np.argpartition(distances, n_probe)[:n_probe]
        return indices[np.argsort(distances[indices])]
    
    def select_zones_batch(
        self,
        queries: np.ndarray,
        n_probe: int
    ) -> np.ndarray:
        """Select zones for multiple queries."""
        if self.use_fast_selector:
            return self.fast_selector.select_zones_batch(queries, n_probe)
        else:
            return self._select_zones_batch_linear(queries, n_probe)
    
    def _select_zones_batch_linear(
        self,
        queries: np.ndarray,
        n_probe: int
    ) -> np.ndarray:
        """Batch linear zone selection."""
        queries = queries.astype(np.float32)
        n_queries = len(queries)
        n_probe_clamped = min(n_probe, self.n_zones)
        
        result = np.zeros((n_queries, n_probe_clamped), dtype=np.int32)
        
        for i, query in enumerate(queries):
            result[i] = self._select_zones_linear(query, n_probe_clamped)
        
        return result
