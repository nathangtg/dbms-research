"""
K-Means clustering for zonal partitioning in ZGQ.

This module provides K-Means clustering functionality to partition the dataset
into zones, which is the first step in the ZGQ algorithm.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict


class ZonalPartitioner:
    """
    K-Means based zonal partitioning for ZGQ.
    
    Partitions the dataset into zones using K-Means clustering and maintains
    inverted lists mapping zones to vector IDs.
    """
    
    def __init__(self, n_zones: int, random_state: int = 42, verbose: bool = True):
        """
        Initialize the zonal partitioner.
        
        Args:
            n_zones: Number of zones to create
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        self.n_zones = n_zones
        self.random_state = random_state
        self.verbose = verbose
        
        self.centroids: Optional[np.ndarray] = None
        self.assignments: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[int]] = {}
        self.zone_sizes: Optional[np.ndarray] = None
        
    def fit(self, vectors: np.ndarray) -> None:
        """
        Fit K-Means clustering to partition vectors into zones.
        
        Args:
            vectors: Input vectors of shape (N, d)
        """
        if self.verbose:
            print(f"Partitioning {len(vectors)} vectors into {self.n_zones} zones...")
        
        # Use MiniBatchKMeans for better scalability
        kmeans = MiniBatchKMeans(
            n_clusters=self.n_zones,
            random_state=self.random_state,
            batch_size=min(1024, len(vectors)),
            n_init=3,
            max_iter=100,
            verbose=0
        )
        
        # Fit and predict cluster assignments
        self.assignments = kmeans.fit_predict(vectors.astype(np.float32))
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        
        # Build inverted lists: zone_id -> list of vector_ids
        self._build_inverted_lists()
        
        # Compute zone sizes
        self.zone_sizes = np.array([len(self.inverted_lists[i]) for i in range(self.n_zones)])
        
        if self.verbose:
            self._print_statistics()
    
    def _build_inverted_lists(self) -> None:
        """Build inverted lists mapping zones to vector IDs."""
        self.inverted_lists = defaultdict(list)
        
        for vector_id, zone_id in enumerate(self.assignments):
            self.inverted_lists[zone_id].append(vector_id)
        
        # Convert to regular dict and ensure all zones exist
        self.inverted_lists = {i: self.inverted_lists.get(i, []) for i in range(self.n_zones)}
    
    def _print_statistics(self) -> None:
        """Print statistics about zone partitioning."""
        print(f"\nZone Partitioning Statistics:")
        print(f"  Total zones: {self.n_zones}")
        print(f"  Zone sizes: min={self.zone_sizes.min()}, "
              f"max={self.zone_sizes.max()}, "
              f"mean={self.zone_sizes.mean():.1f}, "
              f"std={self.zone_sizes.std():.1f}")
        
        # Compute coefficient of variation (CV) for zone balance
        cv = self.zone_sizes.std() / self.zone_sizes.mean() if self.zone_sizes.mean() > 0 else 0
        print(f"  Zone balance (CV): {cv:.3f} (lower is better)")
        
        # Check for empty zones
        empty_zones = np.sum(self.zone_sizes == 0)
        if empty_zones > 0:
            print(f"  WARNING: {empty_zones} empty zones detected")
    
    def assign_to_zones(self, vectors: np.ndarray) -> np.ndarray:
        """
        Assign new vectors to existing zones.
        
        Args:
            vectors: Vectors to assign of shape (N, d)
            
        Returns:
            Zone assignments of shape (N,)
        """
        if self.centroids is None:
            raise ValueError("Partitioner must be fitted before assigning vectors")
        
        # Find nearest centroid for each vector
        from .distances import DistanceMetrics
        
        assignments = np.empty(len(vectors), dtype=np.int32)
        
        for i, vector in enumerate(vectors):
            distances = DistanceMetrics.euclidean_batch_squared(vector, self.centroids)
            assignments[i] = np.argmin(distances)
        
        return assignments
    
    def get_zone_vectors(self, zone_id: int) -> List[int]:
        """
        Get vector IDs belonging to a specific zone.
        
        Args:
            zone_id: Zone index
            
        Returns:
            List of vector IDs in the zone
        """
        if zone_id not in self.inverted_lists:
            raise ValueError(f"Invalid zone_id: {zone_id}")
        
        return self.inverted_lists[zone_id]
    
    def get_zone_info(self) -> Dict:
        """
        Get information about all zones.
        
        Returns:
            Dictionary with zone statistics
        """
        return {
            'n_zones': self.n_zones,
            'zone_sizes': self.zone_sizes.tolist() if self.zone_sizes is not None else None,
            'total_vectors': len(self.assignments) if self.assignments is not None else 0,
            'centroids_shape': self.centroids.shape if self.centroids is not None else None,
            'empty_zones': int(np.sum(self.zone_sizes == 0)) if self.zone_sizes is not None else 0
        }
    
    def save_state(self) -> Dict:
        """
        Save the partitioner state for serialization.
        
        Returns:
            Dictionary with partitioner state
        """
        return {
            'n_zones': self.n_zones,
            'random_state': self.random_state,
            'centroids': self.centroids,
            'assignments': self.assignments,
            'inverted_lists': {k: list(v) for k, v in self.inverted_lists.items()},
            'zone_sizes': self.zone_sizes
        }
    
    @classmethod
    def load_state(cls, state: Dict, verbose: bool = True):
        """
        Load partitioner from saved state.
        
        Args:
            state: Dictionary with partitioner state
            verbose: Whether to print progress
            
        Returns:
            Loaded ZonalPartitioner instance
        """
        partitioner = cls(
            n_zones=state['n_zones'],
            random_state=state['random_state'],
            verbose=verbose
        )
        
        partitioner.centroids = state['centroids']
        partitioner.assignments = state['assignments']
        partitioner.inverted_lists = {int(k): list(v) for k, v in state['inverted_lists'].items()}
        partitioner.zone_sizes = state['zone_sizes']
        
        return partitioner


def suggest_n_zones(n_vectors: int) -> int:
    """
    Suggest an appropriate number of zones based on dataset size.
    
    Args:
        n_vectors: Number of vectors in the dataset
        
    Returns:
        Suggested number of zones
    """
    if n_vectors < 10000:
        return max(20, n_vectors // 50)
    elif n_vectors < 100000:
        return int(n_vectors ** 0.5)
    else:
        return max(100, int(n_vectors ** 0.5) // 2)


def analyze_zone_balance(inverted_lists: Dict[int, List[int]]) -> Dict:
    """
    Analyze the balance of zone partitioning.
    
    Args:
        inverted_lists: Mapping from zone_id to list of vector_ids
        
    Returns:
        Dictionary with balance statistics
    """
    zone_sizes = np.array([len(vlist) for vlist in inverted_lists.values()])
    
    mean_size = zone_sizes.mean()
    cv = zone_sizes.std() / mean_size if mean_size > 0 else 0
    
    return {
        'min_size': int(zone_sizes.min()),
        'max_size': int(zone_sizes.max()),
        'mean_size': float(mean_size),
        'std_size': float(zone_sizes.std()),
        'cv': float(cv),
        'empty_zones': int(np.sum(zone_sizes == 0)),
        'imbalance_ratio': float(zone_sizes.max() / mean_size) if mean_size > 0 else 0
    }
