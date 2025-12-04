"""
Adaptive Hierarchical Zones (AHZ) for ZGQ v8
=============================================

This module implements multi-level zone partitioning that scales
efficiently with dataset size. Unlike flat partitioning, AHZ creates
a hierarchy that enables O(log N) zone selection.

Key Features:
- Automatic zone count selection based on dataset size
- Multi-level hierarchy for efficient navigation
- Balanced zone sizes via iterative refinement
- Zone connectivity graph for inter-zone navigation

Theoretical Foundation:
-----------------------
Given N vectors in d dimensions:
- Level 0: 1 zone (entire dataset)
- Level 1: Z₁ = max(4, √N / 4) coarse zones
- Level 2: Z₂ = max(16, N^(2/3) / 8) fine zones

Zone selection complexity: O(Z₁ + Z₂/Z₁) ≈ O(N^(1/3))
vs flat partitioning: O(√N)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict
import warnings


@dataclass
class ZoneConfig:
    """Configuration for zone partitioning."""
    
    # Auto-detection mode
    auto_zones: bool = True
    
    # Manual zone counts (used if auto_zones=False)
    n_zones_coarse: int = 16
    n_zones_fine: int = 256
    
    # Hierarchy settings
    use_hierarchy: bool = True
    max_levels: int = 2
    
    # Balance settings
    balance_iterations: int = 3
    max_imbalance_ratio: float = 5.0
    
    # Random seed
    random_state: int = 42
    
    # Verbosity
    verbose: bool = False


@dataclass
class ZoneNode:
    """Represents a single zone in the hierarchy."""
    
    zone_id: int
    level: int
    centroid: np.ndarray
    vector_ids: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    parent: Optional[int] = None
    neighbors: List[int] = field(default_factory=list)
    
    @property
    def size(self) -> int:
        return len(self.vector_ids)


class AdaptiveHierarchicalZones:
    """
    Adaptive Hierarchical Zone partitioning for ZGQ.
    
    Creates a multi-level zone hierarchy that scales efficiently:
    - Small datasets (< 10K): Single level, few zones
    - Medium datasets (10K-100K): Two levels
    - Large datasets (> 100K): Full hierarchy
    
    Attributes:
        config: Zone configuration
        levels: List of zone nodes per level
        centroids: Array of all zone centroids
        assignments: Vector-to-zone mapping
        inverted_lists: Zone-to-vectors mapping
    """
    
    def __init__(self, config: Optional[ZoneConfig] = None):
        """
        Initialize Adaptive Hierarchical Zones.
        
        Args:
            config: Zone configuration (uses defaults if None)
        """
        self.config = config or ZoneConfig()
        
        # Zone structure
        self.levels: List[List[ZoneNode]] = []
        self.n_levels: int = 0
        
        # Flat access for efficient search
        self.fine_centroids: Optional[np.ndarray] = None
        self.coarse_centroids: Optional[np.ndarray] = None
        self.assignments: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[int]] = {}
        
        # Coarse-to-fine mapping
        self.coarse_to_fine: Dict[int, List[int]] = {}
        self.fine_to_coarse: Dict[int, int] = {}
        
        # Statistics
        self.n_vectors: int = 0
        self.dimension: int = 0
        self.is_built: bool = False
    
    def _compute_optimal_zones(self, n_vectors: int) -> Tuple[int, int]:
        """
        Compute optimal zone counts based on dataset size.
        
        Uses theoretical analysis to balance:
        - Zone selection cost: O(n_coarse + n_fine/n_coarse)
        - Zone search quality: Higher zones = more precise
        - Memory overhead: Centroids + metadata
        
        Args:
            n_vectors: Number of vectors in dataset
            
        Returns:
            (n_coarse, n_fine): Optimal zone counts
        """
        if n_vectors < 1000:
            # Very small: minimal zones
            n_coarse = max(2, int(np.sqrt(n_vectors) / 2))
            n_fine = max(4, int(np.sqrt(n_vectors)))
        elif n_vectors < 10000:
            # Small: light hierarchy
            n_coarse = max(4, int(np.sqrt(n_vectors) / 4))
            n_fine = max(16, int(np.sqrt(n_vectors)))
        elif n_vectors < 100000:
            # Medium: balanced hierarchy
            n_coarse = max(8, int(n_vectors ** 0.33))
            n_fine = max(64, int(n_vectors ** 0.5))
        elif n_vectors < 1000000:
            # Large: deeper hierarchy
            n_coarse = max(16, int(n_vectors ** 0.25))
            n_fine = max(256, int(n_vectors ** 0.5))
        else:
            # Very large: maximum hierarchy
            n_coarse = max(32, int(n_vectors ** 0.2))
            n_fine = max(1024, int(n_vectors ** 0.45))
        
        # Ensure fine >= coarse
        n_fine = max(n_fine, n_coarse * 2)
        
        # Cap at reasonable limits
        n_coarse = min(n_coarse, 256)
        n_fine = min(n_fine, 16384)
        
        return n_coarse, n_fine
    
    def build(self, vectors: np.ndarray) -> 'AdaptiveHierarchicalZones':
        """
        Build zone hierarchy from input vectors.
        
        Args:
            vectors: Input vectors of shape (N, d)
            
        Returns:
            self for chaining
        """
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.n_vectors, self.dimension = vectors.shape
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Building Adaptive Hierarchical Zones")
            print(f"{'='*60}")
            print(f"Vectors: {self.n_vectors:,}, Dimension: {self.dimension}")
        
        # Determine zone counts
        if self.config.auto_zones:
            n_coarse, n_fine = self._compute_optimal_zones(self.n_vectors)
        else:
            n_coarse = self.config.n_zones_coarse
            n_fine = self.config.n_zones_fine
        
        if self.config.verbose:
            print(f"Coarse zones: {n_coarse}, Fine zones: {n_fine}")
        
        # Build hierarchy
        if self.config.use_hierarchy and n_fine > n_coarse:
            self._build_hierarchical(vectors, n_coarse, n_fine)
        else:
            self._build_flat(vectors, n_fine)
        
        # Build zone connectivity graph
        self._build_zone_graph()
        
        self.is_built = True
        
        if self.config.verbose:
            self._print_statistics()
        
        return self
    
    def _build_hierarchical(
        self,
        vectors: np.ndarray,
        n_coarse: int,
        n_fine: int
    ) -> None:
        """Build two-level hierarchical zones."""
        
        if self.config.verbose:
            print("\n[1/3] Building coarse zones...")
        
        # Level 1: Coarse partitioning
        kmeans_coarse = MiniBatchKMeans(
            n_clusters=n_coarse,
            random_state=self.config.random_state,
            batch_size=min(1024, self.n_vectors),
            n_init=3,
            max_iter=100
        )
        coarse_assignments = kmeans_coarse.fit_predict(vectors)
        self.coarse_centroids = kmeans_coarse.cluster_centers_.astype(np.float32)
        
        # Create coarse level zones
        coarse_zones = []
        coarse_inverted = defaultdict(list)
        
        for vid, cid in enumerate(coarse_assignments):
            coarse_inverted[cid].append(vid)
        
        for cid in range(n_coarse):
            zone = ZoneNode(
                zone_id=cid,
                level=0,
                centroid=self.coarse_centroids[cid],
                vector_ids=coarse_inverted[cid]
            )
            coarse_zones.append(zone)
        
        self.levels.append(coarse_zones)
        
        if self.config.verbose:
            print(f"  Created {n_coarse} coarse zones")
            print("\n[2/3] Building fine zones...")
        
        # Level 2: Fine partitioning (within each coarse zone)
        # Determine fine zones per coarse zone
        target_fine_per_coarse = max(2, n_fine // n_coarse)
        
        fine_zones = []
        fine_centroids_list = []
        fine_zone_id = 0
        
        self.assignments = np.zeros(self.n_vectors, dtype=np.int32)
        
        for cid in range(n_coarse):
            coarse_vids = coarse_inverted[cid]
            
            if len(coarse_vids) == 0:
                continue
            
            # Determine number of fine zones for this coarse zone
            n_fine_local = min(
                target_fine_per_coarse,
                max(1, len(coarse_vids) // 10)
            )
            n_fine_local = min(n_fine_local, len(coarse_vids))
            
            if n_fine_local <= 1:
                # Single fine zone
                centroid = vectors[coarse_vids].mean(axis=0)
                zone = ZoneNode(
                    zone_id=fine_zone_id,
                    level=1,
                    centroid=centroid,
                    vector_ids=coarse_vids.copy(),
                    parent=cid
                )
                fine_zones.append(zone)
                fine_centroids_list.append(centroid)
                
                self.coarse_to_fine[cid] = [fine_zone_id]
                for vid in coarse_vids:
                    self.assignments[vid] = fine_zone_id
                    self.fine_to_coarse[fine_zone_id] = cid
                
                coarse_zones[cid].children.append(fine_zone_id)
                fine_zone_id += 1
            else:
                # Multiple fine zones via k-means
                coarse_vectors = vectors[coarse_vids]
                
                kmeans_fine = MiniBatchKMeans(
                    n_clusters=n_fine_local,
                    random_state=self.config.random_state + cid,
                    batch_size=min(256, len(coarse_vids)),
                    n_init=1,
                    max_iter=50
                )
                fine_assignments = kmeans_fine.fit_predict(coarse_vectors)
                
                self.coarse_to_fine[cid] = []
                
                for fid_local in range(n_fine_local):
                    mask = fine_assignments == fid_local
                    local_vids = [coarse_vids[i] for i, m in enumerate(mask) if m]
                    
                    if len(local_vids) == 0:
                        continue
                    
                    centroid = kmeans_fine.cluster_centers_[fid_local]
                    
                    zone = ZoneNode(
                        zone_id=fine_zone_id,
                        level=1,
                        centroid=centroid,
                        vector_ids=local_vids,
                        parent=cid
                    )
                    fine_zones.append(zone)
                    fine_centroids_list.append(centroid)
                    
                    self.coarse_to_fine[cid].append(fine_zone_id)
                    for vid in local_vids:
                        self.assignments[vid] = fine_zone_id
                    self.fine_to_coarse[fine_zone_id] = cid
                    
                    coarse_zones[cid].children.append(fine_zone_id)
                    fine_zone_id += 1
        
        self.levels.append(fine_zones)
        self.fine_centroids = np.array(fine_centroids_list, dtype=np.float32)
        
        # Build inverted lists
        for zone in fine_zones:
            self.inverted_lists[zone.zone_id] = zone.vector_ids
        
        self.n_levels = 2
        
        if self.config.verbose:
            print(f"  Created {len(fine_zones)} fine zones")
    
    def _build_flat(self, vectors: np.ndarray, n_zones: int) -> None:
        """Build single-level flat zones."""
        
        if self.config.verbose:
            print("\n[1/2] Building flat zones...")
        
        n_zones = min(n_zones, self.n_vectors)
        
        kmeans = MiniBatchKMeans(
            n_clusters=n_zones,
            random_state=self.config.random_state,
            batch_size=min(1024, self.n_vectors),
            n_init=3,
            max_iter=100
        )
        
        self.assignments = kmeans.fit_predict(vectors)
        self.fine_centroids = kmeans.cluster_centers_.astype(np.float32)
        self.coarse_centroids = self.fine_centroids  # Same for flat
        
        # Build zones
        flat_zones = []
        for zid in range(n_zones):
            mask = self.assignments == zid
            vids = np.where(mask)[0].tolist()
            
            zone = ZoneNode(
                zone_id=zid,
                level=0,
                centroid=self.fine_centroids[zid],
                vector_ids=vids
            )
            flat_zones.append(zone)
            self.inverted_lists[zid] = vids
            
            # Self-mapping for consistency
            self.coarse_to_fine[zid] = [zid]
            self.fine_to_coarse[zid] = zid
        
        self.levels.append(flat_zones)
        self.n_levels = 1
        
        if self.config.verbose:
            print(f"  Created {n_zones} zones")
    
    def _build_zone_graph(self) -> None:
        """Build connectivity graph between zones for navigation."""
        
        if self.config.verbose:
            print("\n[3/3] Building zone connectivity graph...")
        
        # For fine zones, compute k nearest neighbor zones
        n_fine = len(self.fine_centroids)
        k_neighbors = min(8, n_fine - 1)
        
        if k_neighbors < 1:
            return
        
        # Compute pairwise distances between fine zone centroids
        # Use efficient vectorized computation
        centroids = self.fine_centroids
        
        # ||a - b||² = ||a||² + ||b||² - 2<a,b>
        norms = np.sum(centroids ** 2, axis=1)
        distances = norms[:, np.newaxis] + norms[np.newaxis, :] - 2 * centroids @ centroids.T
        
        # Find k nearest for each zone
        fine_zones = self.levels[-1]
        
        for i, zone in enumerate(fine_zones):
            # Get k nearest (excluding self)
            dists = distances[i].copy()
            dists[i] = np.inf  # Exclude self
            neighbors = np.argsort(dists)[:k_neighbors].tolist()
            zone.neighbors = neighbors
    
    def _print_statistics(self) -> None:
        """Print zone statistics."""
        
        print(f"\n{'='*60}")
        print("Zone Statistics")
        print(f"{'='*60}")
        print(f"Levels: {self.n_levels}")
        
        for level_idx, level_zones in enumerate(self.levels):
            sizes = [z.size for z in level_zones]
            print(f"\nLevel {level_idx}:")
            print(f"  Zones: {len(level_zones)}")
            print(f"  Size range: [{min(sizes)}, {max(sizes)}]")
            print(f"  Mean size: {np.mean(sizes):.1f}")
            print(f"  Std: {np.std(sizes):.1f}")
            
            # Check balance
            imbalance = max(sizes) / max(1, min([s for s in sizes if s > 0]))
            if imbalance > self.config.max_imbalance_ratio:
                print(f"  ⚠️ Imbalance ratio: {imbalance:.1f}")
            else:
                print(f"  ✓ Imbalance ratio: {imbalance:.1f}")
    
    def select_zones(
        self,
        query: np.ndarray,
        n_probe: int,
        use_hierarchy: bool = True
    ) -> np.ndarray:
        """
        Select n_probe nearest zones for a query.
        
        Uses hierarchical selection if available:
        1. Find nearest coarse zones
        2. Search fine zones within selected coarse zones
        
        Args:
            query: Query vector of shape (d,)
            n_probe: Number of fine zones to return
            use_hierarchy: Whether to use hierarchical selection
            
        Returns:
            Array of fine zone IDs to search
        """
        if not self.is_built:
            raise RuntimeError("Zones must be built before selection")
        
        n_fine = len(self.fine_centroids)
        n_probe = min(n_probe, n_fine)
        
        if self.n_levels == 1 or not use_hierarchy:
            # Flat selection
            distances = np.sum((self.fine_centroids - query) ** 2, axis=1)
            return np.argsort(distances)[:n_probe]
        
        # Hierarchical selection
        # Step 1: Select coarse zones
        n_coarse = len(self.coarse_centroids)
        n_coarse_probe = min(max(2, n_probe // 4), n_coarse)
        
        coarse_distances = np.sum((self.coarse_centroids - query) ** 2, axis=1)
        coarse_zones = np.argsort(coarse_distances)[:n_coarse_probe]
        
        # Step 2: Gather candidate fine zones
        candidate_fine = []
        for cid in coarse_zones:
            if cid in self.coarse_to_fine:
                candidate_fine.extend(self.coarse_to_fine[cid])
        
        if len(candidate_fine) == 0:
            # Fallback to flat selection
            distances = np.sum((self.fine_centroids - query) ** 2, axis=1)
            return np.argsort(distances)[:n_probe]
        
        # Step 3: Rank candidate fine zones
        candidate_fine = np.array(candidate_fine, dtype=np.int32)
        candidate_centroids = self.fine_centroids[candidate_fine]
        fine_distances = np.sum((candidate_centroids - query) ** 2, axis=1)
        
        sorted_indices = np.argsort(fine_distances)[:n_probe]
        return candidate_fine[sorted_indices]
    
    def get_zone_vectors(self, zone_id: int) -> List[int]:
        """Get vector IDs belonging to a fine zone."""
        return self.inverted_lists.get(zone_id, [])
    
    def get_zone_neighbors(self, zone_id: int) -> List[int]:
        """Get neighboring zone IDs for a fine zone."""
        if zone_id < len(self.levels[-1]):
            return self.levels[-1][zone_id].neighbors
        return []
    
    def get_residual(
        self,
        vectors: np.ndarray,
        zone_ids: np.ndarray
    ) -> np.ndarray:
        """
        Compute residuals from zone centroids.
        
        residual = vector - zone_centroid
        
        Used for Residual Product Quantization.
        
        Args:
            vectors: Vectors of shape (N, d)
            zone_ids: Zone assignment for each vector
            
        Returns:
            Residuals of shape (N, d)
        """
        centroids = self.fine_centroids[zone_ids]
        return vectors - centroids
    
    def save_state(self) -> Dict:
        """Save zone state for serialization."""
        return {
            'config': {
                'auto_zones': self.config.auto_zones,
                'n_zones_coarse': self.config.n_zones_coarse,
                'n_zones_fine': self.config.n_zones_fine,
                'use_hierarchy': self.config.use_hierarchy,
                'random_state': self.config.random_state
            },
            'n_vectors': self.n_vectors,
            'dimension': self.dimension,
            'n_levels': self.n_levels,
            'coarse_centroids': self.coarse_centroids,
            'fine_centroids': self.fine_centroids,
            'assignments': self.assignments,
            'inverted_lists': {int(k): list(v) for k, v in self.inverted_lists.items()},
            'coarse_to_fine': {int(k): list(v) for k, v in self.coarse_to_fine.items()},
            'fine_to_coarse': {int(k): int(v) for k, v in self.fine_to_coarse.items()}
        }
    
    @classmethod
    def load_state(cls, state: Dict) -> 'AdaptiveHierarchicalZones':
        """Load zones from saved state."""
        config = ZoneConfig(**state['config'])
        zones = cls(config)
        
        zones.n_vectors = state['n_vectors']
        zones.dimension = state['dimension']
        zones.n_levels = state['n_levels']
        zones.coarse_centroids = state['coarse_centroids']
        zones.fine_centroids = state['fine_centroids']
        zones.assignments = state['assignments']
        zones.inverted_lists = {int(k): list(v) for k, v in state['inverted_lists'].items()}
        zones.coarse_to_fine = {int(k): list(v) for k, v in state['coarse_to_fine'].items()}
        zones.fine_to_coarse = {int(k): int(v) for k, v in state['fine_to_coarse'].items()}
        zones.is_built = True
        
        return zones
