"""
ZGQ Index - Main interface for Zonal Graph Quantization.

This module provides the main ZGQIndex class that combines all components
of the ZGQ algorithm.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import time

try:
    from .core import (
        ZonalPartitioner,
        HNSWGraphManager,
        ProductQuantizer,
        DistanceUtils,
        suggest_n_zones,
        suggest_pq_parameters
    )
    from .search import ZGQSearch
except ImportError:
    from core import (
        ZonalPartitioner,
        HNSWGraphManager,
        ProductQuantizer,
        DistanceUtils,
        suggest_n_zones,
        suggest_pq_parameters
    )
    from search import ZGQSearch


class ZGQIndex:
    """
    Complete Zonal Graph Quantization index for ANNS.
    
    The ZGQIndex combines:
    1. Zonal partitioning (K-Means clustering)
    2. Per-zone HNSW graphs
    3. Product Quantization (optional)
    4. Smart search with aggregation and re-ranking
    """
    
    def __init__(
        self,
        n_zones: Optional[int] = None,
        hnsw_M: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        use_pq: bool = True,
        pq_m: Optional[int] = None,
        pq_nbits: int = 8,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize ZGQ index.
        
        Args:
            n_zones: Number of zones (auto-suggested if None)
            hnsw_M: HNSW maximum connections per node
            hnsw_ef_construction: HNSW construction parameter
            hnsw_ef_search: HNSW search parameter
            use_pq: Whether to use Product Quantization
            pq_m: Number of PQ subspaces (auto-suggested if None)
            pq_nbits: Bits per PQ centroid index
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.n_zones = n_zones
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.use_pq = use_pq
        self.pq_m = pq_m
        self.pq_nbits = pq_nbits
        self.random_state = random_state
        self.verbose = verbose
        
        # Components (initialized during build)
        self.vectors: Optional[np.ndarray] = None
        self.dimension: Optional[int] = None
        self.n_vectors: int = 0
        
        self.partitioner: Optional[ZonalPartitioner] = None
        self.hnsw_manager: Optional[HNSWGraphManager] = None
        self.pq: Optional[ProductQuantizer] = None
        self.pq_codes: Optional[np.ndarray] = None
        self.vector_norms: Optional[np.ndarray] = None
        
        self.searcher: Optional[ZGQSearch] = None
        self.is_built = False
    
    def build(self, vectors: np.ndarray) -> None:
        """
        Build the ZGQ index from input vectors.
        
        Args:
            vectors: Input vectors of shape (N, d)
        """
        start_time = time.time()
        
        # Store vectors
        self.vectors = vectors.astype(np.float32)
        self.n_vectors, self.dimension = vectors.shape
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Building ZGQ Index")
            print(f"{'='*60}")
            print(f"Dataset: {self.n_vectors} vectors, dimension {self.dimension}")
        
        # Auto-suggest parameters if needed
        if self.n_zones is None:
            self.n_zones = suggest_n_zones(self.n_vectors)
            if self.verbose:
                print(f"Auto-suggested n_zones: {self.n_zones}")
        
        if self.use_pq and self.pq_m is None:
            self.pq_m, _ = suggest_pq_parameters(self.dimension)
            if self.verbose:
                print(f"Auto-suggested pq_m: {self.pq_m}")
        
        # Step 1: Zonal Partitioning
        if self.verbose:
            print(f"\n[1/4] Zonal Partitioning")
        
        self.partitioner = ZonalPartitioner(
            n_zones=self.n_zones,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.partitioner.fit(self.vectors)
        
        # Step 2: Build HNSW Graphs
        if self.verbose:
            print(f"\n[2/4] Building HNSW Graphs")
        
        self.hnsw_manager = HNSWGraphManager(
            n_zones=self.n_zones,
            dimension=self.dimension,
            M=self.hnsw_M,
            ef_construction=self.hnsw_ef_construction,
            ef_search=self.hnsw_ef_search,
            verbose=self.verbose
        )
        self.hnsw_manager.build_all_graphs(
            self.vectors,
            self.partitioner.inverted_lists
        )
        
        # Step 3: Product Quantization
        if self.use_pq:
            if self.verbose:
                print(f"\n[3/4] Product Quantization")
            
            self.pq = ProductQuantizer(
                dimension=self.dimension,
                m=self.pq_m,
                nbits=self.pq_nbits,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            # Train PQ
            self.pq.train(self.vectors)
            
            # Encode all vectors
            if self.verbose:
                print("  Encoding all vectors...")
            self.pq_codes = self.pq.encode(self.vectors)
            
            if self.verbose:
                mem_usage = self.pq.get_memory_usage(self.n_vectors)
                print(f"  PQ Memory: {mem_usage['total_mb']:.2f} MB")
        else:
            if self.verbose:
                print(f"\n[3/4] Product Quantization: SKIPPED (use_pq=False)")
        
        # Step 4: Precompute vector norms (for optimization)
        if self.verbose:
            print(f"\n[4/4] Precomputing Vector Norms")
        
        self.vector_norms = DistanceUtils.precompute_vector_norms(self.vectors)
        
        # Initialize searcher
        self.searcher = ZGQSearch(
            centroids=self.partitioner.centroids,
            hnsw_manager=self.hnsw_manager,
            vectors=self.vectors,
            pq=self.pq,
            pq_codes=self.pq_codes,
            vector_norms=self.vector_norms,
            verbose=False  # Set to True for search debugging
        )
        
        self.is_built = True
        build_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"ZGQ Index Built Successfully")
            print(f"Build time: {build_time:.2f} seconds")
            print(f"{'='*60}\n")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors to the query.
        
        Args:
            query: Query vector of shape (d,)
            k: Number of nearest neighbors to return
            n_probe: Number of zones to search
            ef_search: Override HNSW search parameter (optional)
            
        Returns:
            (ids, distances): Arrays of shape (k,) with neighbor IDs and distances
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")
        
        query = query.astype(np.float32)
        
        if query.shape[0] != self.dimension:
            raise ValueError(f"Query dimension {query.shape[0]} != index dimension {self.dimension}")
        
        return self.searcher.search(query, k, n_probe, ef_search)
    
    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        n_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors to multiple queries.
        
        Args:
            queries: Query vectors of shape (n_queries, d)
            k: Number of nearest neighbors per query
            n_probe: Number of zones to search
            ef_search: Override HNSW search parameter
            n_jobs: Number of parallel workers
            
        Returns:
            (ids, distances): Arrays of shape (n_queries, k)
        """
        if not self.is_built:
            raise ValueError("Index must be built before searching")
        
        queries = queries.astype(np.float32)
        
        if queries.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {queries.shape[1]} != index dimension {self.dimension}")
        
        return self.searcher.batch_search(queries, k, n_probe, ef_search, n_jobs=n_jobs)
    
    def get_index_info(self) -> Dict:
        """
        Get information about the index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.is_built:
            return {'is_built': False}
        
        info = {
            'is_built': True,
            'n_vectors': self.n_vectors,
            'dimension': self.dimension,
            'n_zones': self.n_zones,
            'hnsw_M': self.hnsw_M,
            'hnsw_ef_construction': self.hnsw_ef_construction,
            'hnsw_ef_search': self.hnsw_ef_search,
            'use_pq': self.use_pq,
        }
        
        if self.use_pq:
            info['pq_m'] = self.pq_m
            info['pq_nbits'] = self.pq_nbits
            info['pq_k'] = self.pq.k
            mem_usage = self.pq.get_memory_usage(self.n_vectors)
            info['pq_memory_mb'] = mem_usage['total_mb']
        
        if self.partitioner:
            info['zone_info'] = self.partitioner.get_zone_info()
        
        return info
    
    def save(self, filepath: str) -> None:
        """
        Save index to file.
        
        Args:
            filepath: Path to save file
        """
        if not self.is_built:
            raise ValueError("Cannot save index that hasn't been built")
        
        import pickle
        
        state = {
            'config': {
                'n_zones': self.n_zones,
                'hnsw_M': self.hnsw_M,
                'hnsw_ef_construction': self.hnsw_ef_construction,
                'hnsw_ef_search': self.hnsw_ef_search,
                'use_pq': self.use_pq,
                'pq_m': self.pq_m,
                'pq_nbits': self.pq_nbits,
                'random_state': self.random_state,
            },
            'data': {
                'vectors': self.vectors,
                'dimension': self.dimension,
                'n_vectors': self.n_vectors,
                'vector_norms': self.vector_norms,
            },
            'partitioner': self.partitioner.save_state(),
            'hnsw_manager': self.hnsw_manager.save_state(),
        }
        
        if self.use_pq:
            state['pq'] = self.pq.save_state()
            state['pq_codes'] = self.pq_codes
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        if self.verbose:
            print(f"Index saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str, verbose: bool = True) -> 'ZGQIndex':
        """
        Load index from file.
        
        Args:
            filepath: Path to saved index file
            verbose: Whether to print progress
            
        Returns:
            Loaded ZGQIndex instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create index with saved config
        config = state['config']
        index = cls(
            n_zones=config['n_zones'],
            hnsw_M=config['hnsw_M'],
            hnsw_ef_construction=config['hnsw_ef_construction'],
            hnsw_ef_search=config['hnsw_ef_search'],
            use_pq=config['use_pq'],
            pq_m=config['pq_m'],
            pq_nbits=config['pq_nbits'],
            random_state=config['random_state'],
            verbose=verbose
        )
        
        # Restore data
        data = state['data']
        index.vectors = data['vectors']
        index.dimension = data['dimension']
        index.n_vectors = data['n_vectors']
        index.vector_norms = data['vector_norms']
        
        # Restore components
        index.partitioner = ZonalPartitioner.load_state(state['partitioner'], verbose=verbose)
        index.hnsw_manager = HNSWGraphManager.load_state(state['hnsw_manager'], verbose=verbose)
        
        if config['use_pq']:
            index.pq = ProductQuantizer.load_state(state['pq'], verbose=verbose)
            index.pq_codes = state['pq_codes']
        
        # Initialize searcher
        index.searcher = ZGQSearch(
            centroids=index.partitioner.centroids,
            hnsw_manager=index.hnsw_manager,
            vectors=index.vectors,
            pq=index.pq,
            pq_codes=index.pq_codes,
            vector_norms=index.vector_norms,
            verbose=False
        )
        
        index.is_built = True
        
        if verbose:
            print(f"Index loaded from {filepath}")
        
        return index
