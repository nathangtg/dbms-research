"""
HNSW wrapper for per-zone graph management in ZGQ.

This module manages HNSW graphs for each zone, handling local-to-global
ID mapping and per-zone search operations.
"""

import numpy as np
import hnswlib
from typing import Tuple, List, Dict, Optional


class HNSWZoneGraph:
    """
    Manages a single HNSW graph for one zone.
    
    Handles local ID (within zone) to global ID (in dataset) mapping.
    """
    
    def __init__(
        self,
        zone_id: int,
        dimension: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        space: str = 'l2'
    ):
        """
        Initialize HNSW graph for a zone.
        
        Args:
            zone_id: Zone identifier
            dimension: Vector dimension
            M: Maximum number of connections per node
            ef_construction: Construction-time parameter
            ef_search: Search-time parameter
            space: Distance metric ('l2' or 'ip')
        """
        self.zone_id = zone_id
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space
        
        self.index: Optional[hnswlib.Index] = None
        self.local_to_global: List[int] = []
        self.global_to_local: Dict[int, int] = {}
        self.n_vectors = 0
    
    def build(self, vectors: np.ndarray, global_ids: List[int]) -> None:
        """
        Build HNSW graph for zone vectors.
        
        Args:
            vectors: Zone vectors of shape (n_zone, d)
            global_ids: Global IDs corresponding to each vector
        """
        if len(vectors) == 0:
            return  # Empty zone
        
        self.n_vectors = len(vectors)
        self.local_to_global = global_ids.copy()
        self.global_to_local = {gid: lid for lid, gid in enumerate(global_ids)}
        
        # Initialize HNSW index
        self.index = hnswlib.Index(space=self.space, dim=self.dimension)
        self.index.init_index(
            max_elements=self.n_vectors,
            ef_construction=self.ef_construction,
            M=self.M
        )
        
        # Add vectors with local IDs
        local_ids = np.arange(self.n_vectors)
        self.index.add_items(vectors, local_ids)
        
        # Set search parameter
        self.index.set_ef(self.ef_search)
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors in this zone.
        
        Args:
            query: Query vector of shape (d,)
            k: Number of neighbors to return
            ef_search: Override search parameter (optional)
            
        Returns:
            (global_ids, distances): Arrays of shape (min(k, n_vectors),)
        """
        if self.index is None or self.n_vectors == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
        
        # Adjust k if zone has fewer vectors
        k_actual = min(k, self.n_vectors)
        
        # Set ef_search if provided
        if ef_search is not None:
            self.index.set_ef(ef_search)
        
        # Search (query must be 2D)
        local_ids, distances = self.index.knn_query(query.reshape(1, -1), k=k_actual)
        
        # Convert local IDs to global IDs
        local_ids = local_ids[0]  # Remove batch dimension
        distances = distances[0]
        
        global_ids = np.array([self.local_to_global[lid] for lid in local_ids])
        
        return global_ids, distances
    
    def get_vector_by_local_id(self, local_id: int) -> np.ndarray:
        """Get vector by local ID (not commonly used)."""
        if self.index is None:
            raise ValueError("Index not built")
        return self.index.get_items([local_id])[0]
    
    def save_state(self) -> Dict:
        """
        Save zone graph state for serialization.
        
        Returns:
            Dictionary with graph state
        """
        # Save HNSW index to bytes
        if self.index is not None and self.n_vectors > 0:
            import tempfile
            import os
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                tmp_path = tmp.name
            
            try:
                self.index.save_index(tmp_path)
                with open(tmp_path, 'rb') as f:
                    index_bytes = f.read()
            finally:
                os.unlink(tmp_path)
        else:
            index_bytes = None
        
        return {
            'zone_id': self.zone_id,
            'dimension': self.dimension,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'space': self.space,
            'local_to_global': self.local_to_global,
            'n_vectors': self.n_vectors,
            'index_bytes': index_bytes
        }
    
    @classmethod
    def load_state(cls, state: Dict) -> 'HNSWZoneGraph':
        """
        Load zone graph from saved state.
        
        Args:
            state: Dictionary with graph state
            
        Returns:
            Loaded HNSWZoneGraph instance
        """
        graph = cls(
            zone_id=state['zone_id'],
            dimension=state['dimension'],
            M=state['M'],
            ef_construction=state['ef_construction'],
            ef_search=state['ef_search'],
            space=state['space']
        )
        
        graph.n_vectors = state['n_vectors']
        graph.local_to_global = state['local_to_global']
        graph.global_to_local = {gid: lid for lid, gid in enumerate(graph.local_to_global)}
        
        # Load HNSW index from bytes
        if state['index_bytes'] is not None and graph.n_vectors > 0:
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp:
                tmp.write(state['index_bytes'])
                tmp_path = tmp.name
            
            try:
                graph.index = hnswlib.Index(space=graph.space, dim=graph.dimension)
                graph.index.load_index(tmp_path, max_elements=graph.n_vectors)
                graph.index.set_ef(graph.ef_search)
            finally:
                os.unlink(tmp_path)
        
        return graph


class HNSWGraphManager:
    """
    Manages HNSW graphs for all zones.
    
    Provides interface for building and searching across multiple zone graphs.
    """
    
    def __init__(
        self,
        n_zones: int,
        dimension: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        verbose: bool = True
    ):
        """
        Initialize HNSW graph manager.
        
        Args:
            n_zones: Number of zones
            dimension: Vector dimension
            M: Maximum connections per node in HNSW
            ef_construction: HNSW construction parameter
            ef_search: HNSW search parameter
            verbose: Whether to print progress
        """
        self.n_zones = n_zones
        self.dimension = dimension
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.verbose = verbose
        
        self.zone_graphs: List[HNSWZoneGraph] = []
    
    def build_all_graphs(
        self,
        vectors: np.ndarray,
        inverted_lists: Dict[int, List[int]]
    ) -> None:
        """
        Build HNSW graphs for all zones.
        
        Args:
            vectors: All vectors of shape (N, d)
            inverted_lists: Mapping from zone_id to list of vector_ids
        """
        if self.verbose:
            print(f"\nBuilding HNSW graphs for {self.n_zones} zones...")
        
        self.zone_graphs = []
        
        for zone_id in range(self.n_zones):
            # Get vectors for this zone
            global_ids = inverted_lists[zone_id]
            
            if len(global_ids) > 0:
                zone_vectors = vectors[global_ids]
                
                # Create and build graph
                graph = HNSWZoneGraph(
                    zone_id=zone_id,
                    dimension=self.dimension,
                    M=self.M,
                    ef_construction=self.ef_construction,
                    ef_search=self.ef_search
                )
                graph.build(zone_vectors, global_ids)
            else:
                # Empty zone
                graph = HNSWZoneGraph(
                    zone_id=zone_id,
                    dimension=self.dimension,
                    M=self.M,
                    ef_construction=self.ef_construction,
                    ef_search=self.ef_search
                )
            
            self.zone_graphs.append(graph)
            
            if self.verbose and (zone_id + 1) % 10 == 0:
                print(f"  Built graphs for {zone_id + 1}/{self.n_zones} zones")
        
        if self.verbose:
            total_vectors = sum(g.n_vectors for g in self.zone_graphs)
            print(f"HNSW graphs built: {len(self.zone_graphs)} zones, {total_vectors} total vectors")
    
    def search_zone(
        self,
        zone_id: int,
        query: np.ndarray,
        k: int,
        ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search a specific zone.
        
        Args:
            zone_id: Zone to search
            query: Query vector
            k: Number of neighbors
            ef_search: Override search parameter
            
        Returns:
            (global_ids, distances)
        """
        if zone_id >= len(self.zone_graphs):
            raise ValueError(f"Invalid zone_id: {zone_id}")
        
        return self.zone_graphs[zone_id].search(query, k, ef_search)
    
    def save_state(self) -> Dict:
        """Save all zone graphs."""
        return {
            'n_zones': self.n_zones,
            'dimension': self.dimension,
            'M': self.M,
            'ef_construction': self.ef_construction,
            'ef_search': self.ef_search,
            'zone_graphs': [g.save_state() for g in self.zone_graphs]
        }
    
    @classmethod
    def load_state(cls, state: Dict, verbose: bool = True) -> 'HNSWGraphManager':
        """Load all zone graphs from saved state."""
        manager = cls(
            n_zones=state['n_zones'],
            dimension=state['dimension'],
            M=state['M'],
            ef_construction=state['ef_construction'],
            ef_search=state['ef_search'],
            verbose=verbose
        )
        
        manager.zone_graphs = [
            HNSWZoneGraph.load_state(g_state)
            for g_state in state['zone_graphs']
        ]
        
        return manager
