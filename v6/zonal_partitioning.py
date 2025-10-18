"""
Zonal Partitioning Module for ZGQ V6
Implements K-Means clustering for zone creation as specified in offline_indexing.md

Mathematical Foundation:
Objective: minimize Σᵢ₌₁ᶻ Σₓ∈Zᵢ ||x - cᵢ||²

Algorithm: Alternating optimization
1. Assignment: zone(x) = argminⱼ ||x - cⱼ||²
2. Update: cⱼ = (1/|Zⱼ|) Σₓ∈Zⱼ x

Complexity: O(K_iter · N · Z · d) per iteration

Reference: Lloyd's algorithm (1957), offline_indexing.md
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.cluster import KMeans, MiniBatchKMeans
from distance_metrics import DistanceMetrics
import time


class ZonalPartitioner:
    """
    K-Means based zonal partitioning for ZGQ.
    
    Attributes:
        Z: Number of zones
        centroids: Matrix of shape (Z, d) - zone centroids
        assignments: Vector of shape (N,) - zone ID for each vector
        inverted_lists: List[List[int]] - vector indices per zone
        trained: Whether partitioning has been computed
    """
    
    def __init__(
        self, 
        n_zones: int,
        use_minibatch: bool = True,
        init_method: str = 'k-means++',
        verbose: bool = True
    ):
        """
        Initialize Zonal Partitioner.
        
        Args:
            n_zones: Number of zones (Z)
            use_minibatch: Use MiniBatchKMeans for speed (recommended for N > 50k)
            init_method: Centroid initialization ('k-means++' or 'random')
            verbose: Print progress
            
        Reference: offline_indexing.md Section 1
        """
        self.Z = n_zones
        self.use_minibatch = use_minibatch
        self.init_method = init_method
        self.verbose = verbose
        
        self.centroids = None
        self.assignments = None
        self.inverted_lists = None
        self.trained = False
        
        # Statistics
        self.n_vectors = None
        self.dimension = None
        self.zone_sizes = None
        self.build_time = None
    
    def fit(
        self, 
        vectors: np.ndarray,
        max_iter: int = 100,
        n_init: int = 10,
        batch_size: int = 2048
    ) -> None:
        """
        Perform zonal partitioning via K-Means clustering.
        
        Algorithm (Alternating Optimization):
        ----------------------------------------
        Initialize centroids c₁, ..., cₖ
        
        Repeat until convergence:
          1. Assignment Step - O(N·Z·d):
             For each x ∈ D:
               zone(x) = argminⱼ ||x - cⱼ||²
          
          2. Update Step - O(N·d):
             For j = 1 to Z:
               cⱼ = (1/|Zⱼ|) Σₓ∈Zⱼ x
        
        Complexity: O(K_iter · N · Z · d)
        Typical K_iter: 10-100 iterations
        
        Args:
            vectors: Dataset matrix of shape (N, d)
            max_iter: Maximum k-means iterations
            n_init: Number of k-means runs with different initializations
            batch_size: Batch size for MiniBatchKMeans
            
        Reference: offline_indexing.md Section 1.3
        """
        N, d = vectors.shape
        self.n_vectors = N
        self.dimension = d
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Zonal Partitioning via K-Means")
            print(f"{'='*70}")
            print(f"  Vectors: {N:,}")
            print(f"  Dimension: {d}")
            print(f"  Zones (Z): {self.Z}")
            print(f"  Expected zone size: ~{N//self.Z:,} vectors/zone")
            print(f"  Method: {'MiniBatch' if self.use_minibatch else 'Standard'} K-Means")
            print(f"  Initialization: {self.init_method}")
        
        start_time = time.time()
        
        # Choose K-Means variant
        if self.use_minibatch and N > 10000:
            # MiniBatchKMeans: Faster for large datasets
            # Complexity: O(batch_size · Z · d) per iteration
            kmeans = MiniBatchKMeans(
                n_clusters=self.Z,
                max_iter=max_iter,
                batch_size=batch_size,
                n_init=n_init,
                init=self.init_method,
                random_state=42,
                verbose=0
            )
            if self.verbose:
                print(f"  Batch size: {batch_size}")
        else:
            # Standard K-Means: More accurate but slower
            # Complexity: O(N · Z · d) per iteration
            kmeans = KMeans(
                n_clusters=self.Z,
                max_iter=max_iter,
                n_init=n_init,
                init=self.init_method,
                random_state=42,
                verbose=0
            )
        
        # Fit K-Means
        if self.verbose:
            print(f"\n  Running K-Means clustering...")
        
        kmeans.fit(vectors)
        
        # Store results
        self.centroids = kmeans.cluster_centers_.astype(np.float32)
        self.assignments = kmeans.labels_.astype(np.int32)
        
        # Build inverted lists - O(N)
        if self.verbose:
            print(f"  Building inverted lists...")
        
        self.inverted_lists = [[] for _ in range(self.Z)]
        for i, zone_id in enumerate(self.assignments):
            self.inverted_lists[zone_id].append(i)
        
        # Convert to numpy arrays for efficiency
        self.inverted_lists = [np.array(lst, dtype=np.int32) for lst in self.inverted_lists]
        
        # Compute zone size statistics
        self.zone_sizes = np.array([len(lst) for lst in self.inverted_lists])
        
        self.build_time = time.time() - start_time
        self.trained = True
        
        if self.verbose:
            self._print_statistics()
    
    def _print_statistics(self) -> None:
        """Print partitioning statistics."""
        print(f"\n  Partitioning completed in {self.build_time:.2f}s")
        print(f"\n  Zone Size Statistics:")
        print(f"    Mean: {np.mean(self.zone_sizes):.1f}")
        print(f"    Median: {np.median(self.zone_sizes):.1f}")
        print(f"    Min: {np.min(self.zone_sizes)}")
        print(f"    Max: {np.max(self.zone_sizes)}")
        print(f"    Std: {np.std(self.zone_sizes):.1f}")
        
        # Balance metric: coefficient of variation
        balance = np.std(self.zone_sizes) / np.mean(self.zone_sizes)
        print(f"    Balance (CV): {balance:.3f} {'✓ Good' if balance < 0.3 else '⚠ Imbalanced'}")
        
        # Memory usage
        centroid_mem = self.centroids.nbytes / (1024 ** 2)
        invlist_mem = sum(lst.nbytes for lst in self.inverted_lists) / (1024 ** 2)
        total_mem = centroid_mem + invlist_mem
        
        print(f"\n  Memory Usage:")
        print(f"    Centroids: {centroid_mem:.2f} MB")
        print(f"    Inverted lists: {invlist_mem:.2f} MB")
        print(f"    Total: {total_mem:.2f} MB")
    
    def assign_to_zones(self, query_vectors: np.ndarray, n_probe: int = 1) -> np.ndarray:
        """
        Assign query vectors to nearest zones.
        
        Formula: zone(q) = argminⱼ ||q - cⱼ||²
        
        Complexity: O(N_query · Z · d)
        
        Args:
            query_vectors: Matrix of shape (N_query, d)
            n_probe: Number of nearest zones to return per query
            
        Returns:
            Matrix of shape (N_query, n_probe) with zone indices
            
        Reference: online_search.md Section 1
        """
        if not self.trained:
            raise ValueError("Partitioner must be fitted before assigning queries")
        
        N_query = len(query_vectors)
        selected_zones = np.zeros((N_query, n_probe), dtype=np.int32)
        
        for i, query in enumerate(query_vectors):
            # Compute distances to all centroids
            distances = DistanceMetrics.euclidean_batch_squared(query, self.centroids)
            
            # Select n_probe nearest zones
            if n_probe < self.Z:
                nearest = np.argpartition(distances, n_probe)[:n_probe]
                # Sort by distance
                nearest = nearest[np.argsort(distances[nearest])]
            else:
                nearest = np.argsort(distances)
            
            selected_zones[i] = nearest
        
        return selected_zones
    
    def get_zone_vectors(self, vectors: np.ndarray, zone_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all vectors belonging to a specific zone.
        
        Args:
            vectors: Full dataset of shape (N, d)
            zone_id: Zone index (0 to Z-1)
            
        Returns:
            Tuple of (zone_vectors, global_indices)
            - zone_vectors: Matrix of shape (n_zone, d)
            - global_indices: Array of shape (n_zone,) with original indices
        """
        if not self.trained:
            raise ValueError("Partitioner must be fitted first")
        
        if zone_id < 0 or zone_id >= self.Z:
            raise ValueError(f"zone_id must be in range [0, {self.Z-1}]")
        
        global_indices = self.inverted_lists[zone_id]
        zone_vectors = vectors[global_indices]
        
        return zone_vectors, global_indices
    
    def compute_quantization_error(self, vectors: np.ndarray, n_samples: int = 5000) -> float:
        """
        Compute mean quantization error: E[||x - c_zone(x)||²]
        
        This measures how well the zones approximate the data distribution.
        
        Args:
            vectors: Dataset matrix of shape (N, d)
            n_samples: Number of samples to evaluate
            
        Returns:
            Mean squared quantization error
        """
        if not self.trained:
            raise ValueError("Partitioner must be fitted first")
        
        N = min(n_samples, len(vectors))
        indices = np.random.choice(len(vectors), N, replace=False)
        
        errors = []
        for i in indices:
            zone_id = self.assignments[i]
            centroid = self.centroids[zone_id]
            error = np.sum((vectors[i] - centroid) ** 2)
            errors.append(error)
        
        return np.mean(errors)
    
    def save(self, filepath: str) -> None:
        """Save partitioning to disk."""
        if not self.trained:
            raise ValueError("Cannot save untrained partitioner")
        
        np.savez_compressed(
            filepath,
            Z=self.Z,
            centroids=self.centroids,
            assignments=self.assignments,
            inverted_lists=self.inverted_lists,
            zone_sizes=self.zone_sizes,
            n_vectors=self.n_vectors,
            dimension=self.dimension
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'ZonalPartitioner':
        """Load partitioning from disk."""
        data = np.load(filepath, allow_pickle=True)
        
        partitioner = cls(n_zones=int(data['Z']), verbose=False)
        partitioner.centroids = data['centroids']
        partitioner.assignments = data['assignments']
        partitioner.inverted_lists = list(data['inverted_lists'])
        partitioner.zone_sizes = data['zone_sizes']
        partitioner.n_vectors = int(data['n_vectors'])
        partitioner.dimension = int(data['dimension'])
        partitioner.trained = True
        
        return partitioner


# Validation and testing
if __name__ == "__main__":
    print("="*70)
    print("Zonal Partitioning Module - Validation")
    print("="*70)
    
    # Configuration
    N = 50000
    d = 128
    Z = 100
    
    # Generate synthetic clustered data
    print("\nGenerating synthetic clustered data...")
    np.random.seed(42)
    
    # Create natural clusters
    n_clusters = 20
    vectors = []
    for _ in range(n_clusters):
        center = np.random.randn(d) * 10
        cluster_size = N // n_clusters
        cluster = center + np.random.randn(cluster_size, d) * 2
        vectors.append(cluster)
    
    vectors = np.vstack(vectors).astype(np.float32)
    print(f"  Generated {N:,} vectors in {n_clusters} natural clusters")
    
    # Test 1: Fit partitioner
    print("\n[Test 1] Fitting Zonal Partitioner")
    partitioner = ZonalPartitioner(n_zones=Z, use_minibatch=True, verbose=True)
    partitioner.fit(vectors, max_iter=100, n_init=5)
    
    assert partitioner.trained, "Training failed"
    assert partitioner.centroids.shape == (Z, d), f"Centroid shape mismatch"
    assert len(partitioner.inverted_lists) == Z, f"Inverted list count mismatch"
    print("✓ Partitioning successful")
    
    # Test 2: Query assignment
    print("\n[Test 2] Query Assignment to Zones")
    n_queries = 100
    n_probe = 5
    queries = np.random.randn(n_queries, d).astype(np.float32)
    
    start = time.time()
    assigned_zones = partitioner.assign_to_zones(queries, n_probe=n_probe)
    assignment_time = time.time() - start
    
    assert assigned_zones.shape == (n_queries, n_probe), "Assignment shape mismatch"
    print(f"  Assigned {n_queries} queries in {assignment_time*1000:.2f} ms")
    print(f"  Throughput: {n_queries/assignment_time:.0f} queries/sec")
    print(f"  Example: Query 0 → Zones {assigned_zones[0]}")
    
    # Test 3: Get zone vectors
    print("\n[Test 3] Retrieving Zone Vectors")
    zone_id = 0
    zone_vectors, global_indices = partitioner.get_zone_vectors(vectors, zone_id)
    
    print(f"  Zone {zone_id} has {len(zone_vectors)} vectors")
    print(f"  Global indices: {global_indices[:10]}... (showing first 10)")
    assert len(zone_vectors) == len(global_indices), "Size mismatch"
    
    # Test 4: Quantization error
    print("\n[Test 4] Quantization Error")
    quant_error = partitioner.compute_quantization_error(vectors, n_samples=1000)
    avg_vector_norm = np.mean(np.sum(vectors ** 2, axis=1))
    relative_error = quant_error / avg_vector_norm
    
    print(f"  Mean quantization error: {quant_error:.4f}")
    print(f"  Average vector norm²: {avg_vector_norm:.4f}")
    print(f"  Relative error: {relative_error:.2%}")
    
    # Test 5: Zone balance analysis
    print("\n[Test 5] Zone Balance Analysis")
    print(f"  Zone sizes: Min={partitioner.zone_sizes.min()}, "
          f"Max={partitioner.zone_sizes.max()}, "
          f"Mean={partitioner.zone_sizes.mean():.1f}")
    
    # Histogram of zone sizes
    bins = [0, 300, 400, 500, 600, 700, 1000, 2000]
    hist, _ = np.histogram(partitioner.zone_sizes, bins=bins)
    print(f"\n  Zone size distribution:")
    for i in range(len(hist)):
        if i < len(bins) - 1:
            print(f"    [{bins[i]:4d}-{bins[i+1]:4d}): {'█' * int(hist[i]/2)} ({hist[i]})")
    
    # Test 6: Coverage test
    print("\n[Test 6] Coverage Test")
    # Check that all vectors are assigned
    all_assigned = set()
    for inv_list in partitioner.inverted_lists:
        all_assigned.update(inv_list)
    
    coverage = len(all_assigned) / N
    print(f"  Assigned vectors: {len(all_assigned):,} / {N:,}")
    print(f"  Coverage: {coverage:.2%}")
    assert coverage == 1.0, "Not all vectors are assigned!"
    
    print("\n" + "="*70)
    print("✓ All tests passed successfully")
    print("="*70)
