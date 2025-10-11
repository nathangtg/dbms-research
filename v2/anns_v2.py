"""
Zonal Graph Quantization (ZGQ) Benchmark - OPTIMIZED VERSION
Fixes: HNSW recall bug, ZGQ performance issues
"""

import numpy as np
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Configuration and Tuning
# ============================================================================

class BenchmarkConfig:
    """Centralized configuration for fair comparison"""
    # Dataset
    N = 50000          # Increased for better scaling test
    D = 128
    N_QUERIES = 200
    K = 10
    
    # HNSW parameters
    HNSW_M = 16
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 100  # Increased for better recall
    
    # IVF parameters
    IVF_N_CLUSTERS = 100
    IVF_N_PROBE = 15
    
    # ZGQ parameters (optimized)
    ZGQ_N_ZONES = 70      # sqrt(50000) ≈ 223, using ~70 for balance
    ZGQ_N_PROBE = 8
    ZGQ_M = 12            # Smaller M for faster construction


@dataclass
class BenchmarkResult:
    algorithm: str
    build_time: float
    query_time: float
    recall: float
    memory_mb: float
    queries_per_second: float
    recall_at_k: Dict[int, float] = None
    
    def __repr__(self):
        return (f"{self.algorithm:8s} | Build: {self.build_time:6.2f}s | "
                f"Query: {self.query_time*1000:7.3f}ms | "
                f"Recall@10: {self.recall:5.3f} | Memory: {self.memory_mb:7.1f}MB | "
                f"QPS: {self.queries_per_second:8.1f}")


class DistanceMetrics:
    """Optimized distance computations"""
    
    @staticmethod
    def euclidean_squared(a: np.ndarray, b: np.ndarray) -> float:
        """Squared Euclidean (faster, monotonic)"""
        diff = a - b
        return np.dot(diff, diff)
    
    @staticmethod
    def euclidean_batch_squared(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Vectorized squared distances"""
        diff = vectors - query
        return np.einsum('ij,ij->i', diff, diff)


# ============================================================================
# Ground Truth (Optimized)
# ============================================================================

class BruteForceIndex:
    """Optimized brute force search"""
    
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.N = len(vectors)
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        distances = DistanceMetrics.euclidean_batch_squared(query, self.vectors)
        indices = np.argpartition(distances, min(k, self.N-1))[:k]
        indices = indices[np.argsort(distances[indices])]
        return indices, distances[indices]


# ============================================================================
# FIXED HNSW Implementation
# ============================================================================

class OptimizedHNSW:
    """Fixed HNSW with proper connectivity and search"""
    
    def __init__(self, vectors: np.ndarray, M: int = 16, ef_construction: int = 200):
        self.vectors = vectors
        self.N = len(vectors)
        self.M = M
        self.ef_construction = ef_construction
        self.graph = [[] for _ in range(self.N)]  # Use list for faster append
        self.entry_point = 0
        
    def build(self):
        """Build HNSW with proper greedy search"""
        start_time = time.time()
        
        for i in range(1, self.N):
            if i % 5000 == 0:
                print(f"  HNSW: {i}/{self.N} nodes", end='\r')
            
            # Find ef_construction nearest neighbors
            candidates = self._greedy_search(self.vectors[i], self.ef_construction, 
                                            visited_exclude=set([i]))
            
            # Connect to M best candidates
            neighbors_to_add = candidates[:self.M]
            
            for neighbor_idx, _ in neighbors_to_add:
                # Bidirectional edges
                self.graph[i].append(neighbor_idx)
                self.graph[neighbor_idx].append(i)
                
                # Prune neighbor if needed
                if len(self.graph[neighbor_idx]) > self.M:
                    self._prune_connections(neighbor_idx)
        
        print()
        return time.time() - start_time
    
    def _greedy_search(self, query: np.ndarray, ef: int, 
                       visited_exclude: set = None) -> List[Tuple[int, float]]:
        """Greedy search returning (index, distance) pairs"""
        visited = visited_exclude if visited_exclude else set()
        
        # Start from entry point
        entry_dist = DistanceMetrics.euclidean_squared(query, self.vectors[self.entry_point])
        candidates = [(entry_dist, self.entry_point)]
        visited.add(self.entry_point)
        
        result = []
        
        while candidates:
            # Get closest unvisited candidate
            candidates.sort()
            current_dist, current = candidates.pop(0)
            result.append((current, current_dist))
            
            if len(result) >= ef:
                break
            
            # Explore neighbors
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    neighbor_dist = DistanceMetrics.euclidean_squared(query, 
                                                                     self.vectors[neighbor])
                    candidates.append((neighbor_dist, neighbor))
        
        result.sort(key=lambda x: x[0])  # Sort by distance
        return result
    
    def _prune_connections(self, node_idx: int):
        """Keep only M nearest neighbors"""
        if len(self.graph[node_idx]) <= self.M:
            return
        
        # Compute distances to all neighbors
        neighbors = self.graph[node_idx]
        distances = [(DistanceMetrics.euclidean_squared(self.vectors[node_idx], 
                                                       self.vectors[n]), n) 
                    for n in neighbors]
        distances.sort()
        
        # Keep M closest
        self.graph[node_idx] = [n for _, n in distances[:self.M]]
    
    def search(self, query: np.ndarray, k: int, ef: int = 50) -> np.ndarray:
        """Search with proper ef parameter"""
        results = self._greedy_search(query, max(ef, k))
        return np.array([idx for idx, _ in results[:k]])
    
    def memory_usage(self) -> float:
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        graph_mem = sum(len(neighbors) for neighbors in self.graph) * 8 / (1024 ** 2)
        return vector_mem + graph_mem


# ============================================================================
# Optimized IVF
# ============================================================================

class OptimizedIVF:
    """IVF with sklearn and optimizations"""
    
    def __init__(self, vectors: np.ndarray, n_clusters: int = 100):
        self.vectors = vectors
        self.N = len(vectors)
        self.n_clusters = n_clusters
        self.kmeans = None
        self.inverted_lists = [[] for _ in range(n_clusters)]
    
    def build(self):
        start_time = time.time()
        
        print(f"  IVF K-Means clustering (k={self.n_clusters})...")
        
        # Use MiniBatchKMeans for speed
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=2048,
            n_init=3,
            verbose=0
        )
        
        labels = self.kmeans.fit_predict(self.vectors)
        
        # Build inverted lists
        for i, label in enumerate(labels):
            self.inverted_lists[label].append(i)
        
        return time.time() - start_time
    
    def search(self, query: np.ndarray, k: int, n_probe: int = 10) -> np.ndarray:
        # Find closest clusters
        centroid_dists = DistanceMetrics.euclidean_batch_squared(query, 
                                                                 self.kmeans.cluster_centers_)
        closest_clusters = np.argpartition(centroid_dists, 
                                          min(n_probe, self.n_clusters))[:n_probe]
        
        # Collect candidates from selected clusters
        candidate_indices = []
        for cluster_idx in closest_clusters:
            candidate_indices.extend(self.inverted_lists[cluster_idx])
        
        if not candidate_indices:
            return np.array([])
        
        # Compute distances to candidates
        candidate_vectors = self.vectors[candidate_indices]
        distances = DistanceMetrics.euclidean_batch_squared(query, candidate_vectors)
        
        # Return top k
        top_k_local = np.argpartition(distances, min(k, len(distances)-1))[:k]
        top_k_local = top_k_local[np.argsort(distances[top_k_local])]
        
        return np.array([candidate_indices[i] for i in top_k_local])
    
    def memory_usage(self) -> float:
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        centroid_mem = self.kmeans.cluster_centers_.nbytes / (1024 ** 2)
        invlist_mem = sum(len(lst) for lst in self.inverted_lists) * 8 / (1024 ** 2)
        return vector_mem + centroid_mem + invlist_mem


# ============================================================================
# OPTIMIZED ZGQ Implementation
# ============================================================================

class OptimizedZGQ:
    """Optimized ZGQ with better search strategy"""
    
    def __init__(self, vectors: np.ndarray, n_zones: int = 50, M: int = 16):
        self.vectors = vectors
        self.N = len(vectors)
        self.n_zones = n_zones
        self.M = M
        self.kmeans = None
        self.zones = [[] for _ in range(n_zones)]
        self.local_graphs = {}
    
    def build(self):
        start_time = time.time()
        
        print(f"  ZGQ Phase 1: Zonal partitioning (k={self.n_zones})...")
        
        # Phase 1: Fast clustering
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_zones,
            random_state=42,
            batch_size=2048,
            n_init=3,
            verbose=0
        )
        
        labels = self.kmeans.fit_predict(self.vectors)
        
        # Assign to zones
        for i, label in enumerate(labels):
            self.zones[label].append(i)
        
        # Phase 2: Build local graphs in parallel (simulated)
        print(f"  ZGQ Phase 2: Building {self.n_zones} local HNSW graphs...")
        
        for zone_idx in range(self.n_zones):
            vector_indices = self.zones[zone_idx]
            
            if len(vector_indices) > 5:  # Only build for non-trivial zones
                zone_vectors = self.vectors[vector_indices]
                local_hnsw = OptimizedHNSW(zone_vectors, M=self.M, ef_construction=100)
                local_hnsw.build()
                self.local_graphs[zone_idx] = (local_hnsw, vector_indices)
        
        print(f"  Built {len(self.local_graphs)}/{self.n_zones} local graphs")
        
        return time.time() - start_time
    
    def search(self, query: np.ndarray, k: int, n_probe: int = 5) -> np.ndarray:
        # Phase 1: Select zones
        centroid_dists = DistanceMetrics.euclidean_batch_squared(query, 
                                                                 self.kmeans.cluster_centers_)
        closest_zones = np.argpartition(centroid_dists, 
                                       min(n_probe, self.n_zones))[:n_probe]
        
        # Phase 2: Search within zones
        all_candidates = []
        
        for zone_idx in closest_zones:
            if zone_idx in self.local_graphs:
                local_hnsw, vector_indices = self.local_graphs[zone_idx]
                
                # Search local graph
                local_results = local_hnsw.search(query, k, ef=50)
                
                # Map back to global indices
                for local_idx in local_results:
                    if local_idx < len(vector_indices):
                        all_candidates.append(vector_indices[local_idx])
        
        if not all_candidates:
            return np.array([])
        
        # Phase 3: Re-rank all candidates
        candidate_vectors = self.vectors[all_candidates]
        distances = DistanceMetrics.euclidean_batch_squared(query, candidate_vectors)
        
        top_k_local = np.argpartition(distances, min(k, len(distances)-1))[:k]
        top_k_local = top_k_local[np.argsort(distances[top_k_local])]
        
        return np.array([all_candidates[i] for i in top_k_local])
    
    def memory_usage(self) -> float:
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        centroid_mem = self.kmeans.cluster_centers_.nbytes / (1024 ** 2)
        
        graph_mem = 0
        for local_hnsw, _ in self.local_graphs.values():
            graph_mem += sum(len(neighbors) for neighbors in local_hnsw.graph) * 8 / (1024 ** 2)
        
        return vector_mem + centroid_mem + graph_mem


# ============================================================================
# Benchmark Framework
# ============================================================================

class ANNSBenchmark:
    def __init__(self, vectors: np.ndarray, queries: np.ndarray, k: int = 10):
        self.vectors = vectors
        self.queries = queries
        self.k = k
        self.N, self.d = vectors.shape
        
        print(f"Computing ground truth for {len(queries)} queries...")
        self.ground_truth = self._compute_ground_truth()
    
    def _compute_ground_truth(self) -> List[np.ndarray]:
        bf_index = BruteForceIndex(self.vectors)
        ground_truth = []
        for i, query in enumerate(self.queries):
            if (i + 1) % 50 == 0:
                print(f"  Ground truth: {i+1}/{len(self.queries)}", end='\r')
            indices, _ = bf_index.search(query, self.k)
            ground_truth.append(set(indices))  # Use set for O(1) lookup
        print()
        return ground_truth
    
    def compute_recall(self, results: List[np.ndarray]) -> float:
        recalls = []
        for result, truth in zip(results, self.ground_truth):
            intersection = len(set(result) & truth)
            recalls.append(intersection / self.k)
        return np.mean(recalls)
    
    def compute_recall_at_k(self, results: List[np.ndarray], 
                           k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        recall_dict = {}
        for k_val in k_values:
            if k_val > self.k:
                continue
            recalls = []
            for result, truth in zip(results, self.ground_truth):
                result_at_k = set(result[:k_val])
                truth_list = list(truth)[:k_val]
                intersection = len(result_at_k & set(truth_list))
                recalls.append(intersection / k_val)
            recall_dict[k_val] = np.mean(recalls)
        return recall_dict
    
    def run_benchmark(self, index, index_name: str) -> BenchmarkResult:
        print(f"\n{'='*60}")
        print(f"Benchmarking {index_name}")
        print(f"{'='*60}")
        
        # Build
        build_time = index.build()
        
        # Query
        results = []
        start_time = time.time()
        for i, query in enumerate(self.queries):
            if (i + 1) % 50 == 0:
                print(f"  Querying: {i+1}/{len(self.queries)}", end='\r')
            result = index.search(query, self.k)
            results.append(result)
        print()
        
        query_time = (time.time() - start_time) / len(self.queries)
        
        # Metrics
        recall = self.compute_recall(results)
        recall_at_k = self.compute_recall_at_k(results, [1, 5, 10])
        memory_mb = index.memory_usage()
        qps = 1.0 / query_time if query_time > 0 else 0
        
        return BenchmarkResult(index_name, build_time, query_time, recall, 
                             memory_mb, qps, recall_at_k)


# ============================================================================
# Main Execution
# ============================================================================

def generate_clustered_data(N, d, n_queries):
    """Generate clustered synthetic data"""
    np.random.seed(42)
    
    n_clusters = max(20, N // 2000)
    vectors = []
    
    for _ in range(n_clusters):
        center = np.random.randn(d) * 15
        cluster_size = N // n_clusters
        cluster_vectors = center + np.random.randn(cluster_size, d) * 2
        vectors.append(cluster_vectors)
    
    vectors = np.vstack(vectors)
    
    # Queries as perturbations
    query_indices = np.random.choice(N, n_queries, replace=False)
    queries = vectors[query_indices] + np.random.randn(n_queries, d) * 0.5
    
    return vectors, queries


def plot_results(results: Dict):
    """Create comprehensive comparison plots"""
    algorithms = list(results.keys())
    colors = {'hnsw': '#3498db', 'ivf': '#e74c3c', 'zgq': '#2ecc71'}
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('ANNS Algorithm Comparison: ZGQ vs HNSW vs IVF (Optimized)', 
                 fontsize=16, fontweight='bold')
    
    # Query Time
    query_times = [results[alg].query_time * 1000 for alg in algorithms]
    axes[0, 0].bar(algorithms, query_times, color=[colors[alg] for alg in algorithms])
    axes[0, 0].set_ylabel('Query Time (ms)', fontweight='bold')
    axes[0, 0].set_title('Query Latency')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall
    recalls = [results[alg].recall for alg in algorithms]
    axes[0, 1].bar(algorithms, recalls, color=[colors[alg] for alg in algorithms])
    axes[0, 1].set_ylabel('Recall@10', fontweight='bold')
    axes[0, 1].set_title('Recall')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Memory
    memory = [results[alg].memory_mb for alg in algorithms]
    axes[0, 2].bar(algorithms, memory, color=[colors[alg] for alg in algorithms])
    axes[0, 2].set_ylabel('Memory (MB)', fontweight='bold')
    axes[0, 2].set_title('Memory Footprint')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Build Time
    build_times = [results[alg].build_time for alg in algorithms]
    axes[1, 0].bar(algorithms, build_times, color=[colors[alg] for alg in algorithms])
    axes[1, 0].set_ylabel('Build Time (s)', fontweight='bold')
    axes[1, 0].set_title('Index Construction')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # QPS
    qps = [results[alg].queries_per_second for alg in algorithms]
    axes[1, 1].bar(algorithms, qps, color=[colors[alg] for alg in algorithms])
    axes[1, 1].set_ylabel('Queries Per Second', fontweight='bold')
    axes[1, 1].set_title('Throughput')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Recall@k curves
    k_values = [1, 5, 10]
    for alg in algorithms:
        if results[alg].recall_at_k:
            recalls_at_k = [results[alg].recall_at_k.get(k, 0) for k in k_values]
            axes[1, 2].plot(k_values, recalls_at_k, marker='o', 
                          label=alg.upper(), color=colors[alg], linewidth=2, markersize=8)
    
    axes[1, 2].set_xlabel('k', fontweight='bold')
    axes[1, 2].set_ylabel('Recall@k', fontweight='bold')
    axes[1, 2].set_title('Recall@k Curves')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('zgq_benchmark_optimized.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'zgq_benchmark_optimized.png'")


if __name__ == "__main__":
    cfg = BenchmarkConfig()
    
    print("="*80)
    print("ZGQ BENCHMARK - OPTIMIZED VERSION")
    print("="*80)
    print(f"\nDataset: {cfg.N:,} vectors x {cfg.D} dimensions")
    print(f"Queries: {cfg.N_QUERIES}")
    print(f"k = {cfg.K}\n")
    
    # Generate data
    vectors, queries = generate_clustered_data(cfg.N, cfg.D, cfg.N_QUERIES)
    
    benchmark = ANNSBenchmark(vectors, queries, k=cfg.K)
    
    # Run all algorithms
    results = {}
    
    # HNSW
    hnsw = OptimizedHNSW(vectors, M=cfg.HNSW_M, ef_construction=cfg.HNSW_EF_CONSTRUCTION)
    results['hnsw'] = benchmark.run_benchmark(hnsw, "HNSW")
    
    # IVF
    ivf = OptimizedIVF(vectors, n_clusters=cfg.IVF_N_CLUSTERS)
    results['ivf'] = benchmark.run_benchmark(ivf, "IVF")
    
    # ZGQ
    zgq = OptimizedZGQ(vectors, n_zones=cfg.ZGQ_N_ZONES, M=cfg.ZGQ_M)
    results['zgq'] = benchmark.run_benchmark(zgq, "ZGQ")
    
    # Print summary
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for name, result in results.items():
        print(result)
    
    # Plot
    plot_results(results)
    
    print("\n✓ Benchmark complete!")