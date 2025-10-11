"""
Zonal Graph Quantization (ZGQ) Benchmark Implementation
Enhanced with scikit-learn for production-grade clustering and metrics
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Structures and Metrics
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from a single benchmark run"""
    algorithm: str
    build_time: float
    query_time: float
    recall: float
    memory_mb: float
    queries_per_second: float
    recall_at_k: Dict[int, float] = None  # Recall@1, @5, @10, etc.
    
    def __repr__(self):
        return (f"{self.algorithm:8s} | Build: {self.build_time:6.2f}s | "
                f"Query: {self.query_time*1000:7.3f}ms | "
                f"Recall@10: {self.recall:5.3f} | Memory: {self.memory_mb:7.1f}MB | "
                f"QPS: {self.queries_per_second:8.1f}")


class DistanceMetrics:
    """Distance computation utilities with optimization"""
    
    @staticmethod
    def euclidean(a: np.ndarray, b: np.ndarray) -> float:
        return np.linalg.norm(a - b)
    
    @staticmethod
    def euclidean_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Vectorized distance computation - optimized"""
        # Use squared distances for speed (monotonic transformation)
        diff = vectors - query
        return np.einsum('ij,ij->i', diff, diff)
    
    @staticmethod
    def cosine_similarity_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Cosine similarity for normalized vectors"""
        return np.dot(vectors, query)


# ============================================================================
# Ground Truth Computer (Brute Force)
# ============================================================================

class BruteForceIndex:
    """Brute force search for ground truth with sklearn optimization"""
    
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors
        self.N = len(vectors)
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (indices, distances) of k nearest neighbors"""
        distances = DistanceMetrics.euclidean_batch(query, self.vectors)
        
        # Use argpartition for O(n + k log k) instead of O(n log n)
        if k < self.N:
            indices = np.argpartition(distances, k)[:k]
            indices = indices[np.argsort(distances[indices])]
        else:
            indices = np.argsort(distances)[:k]
        
        return indices, distances[indices]


# ============================================================================
# Simple HNSW Implementation (Simplified for Benchmarking)
# ============================================================================

class SimpleHNSW:
    """Simplified HNSW implementation for comparison"""
    
    def __init__(self, vectors: np.ndarray, M: int = 16, ef_construction: int = 200):
        self.vectors = vectors
        self.N = len(vectors)
        self.M = M
        self.ef_construction = ef_construction
        self.graph = defaultdict(list)
        self.entry_point = 0
        
    def build(self):
        """Build HNSW graph"""
        start_time = time.time()
        
        # Simplified single-layer HNSW
        for i in range(self.N):
            if i == 0:
                continue
            
            if i % 1000 == 0:
                print(f"  Building HNSW: {i}/{self.N} nodes...", end='\r')
                
            # Find nearest neighbors using greedy search
            candidates = self._search_layer(self.vectors[i], self.ef_construction, i)
            
            # Connect to M nearest neighbors
            neighbors = candidates[:self.M]
            for neighbor_idx in neighbors:
                if neighbor_idx != i:
                    self.graph[i].append(neighbor_idx)
                    self.graph[neighbor_idx].append(i)
                    
                    # Prune if exceeds M connections
                    if len(self.graph[neighbor_idx]) > self.M:
                        self._prune_connections(neighbor_idx)
        
        print()  # New line after progress
        return time.time() - start_time
    
    def _search_layer(self, query: np.ndarray, ef: int, exclude_idx: int = -1) -> List[int]:
        """Greedy search in graph layer"""
        visited = set([exclude_idx]) if exclude_idx >= 0 else set()
        candidates = [(DistanceMetrics.euclidean(query, self.vectors[self.entry_point]), 
                      self.entry_point)]
        best = []
        
        while candidates and len(best) < ef:
            dist, current = min(candidates)
            candidates.remove((dist, current))
            
            if current in visited:
                continue
            visited.add(current)
            best.append(current)
            
            # Explore neighbors
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    neighbor_dist = DistanceMetrics.euclidean(query, self.vectors[neighbor])
                    candidates.append((neighbor_dist, neighbor))
        
        return best
    
    def _prune_connections(self, node_idx: int):
        """Prune connections to maintain M neighbors"""
        neighbors = self.graph[node_idx]
        if len(neighbors) <= self.M:
            return
        
        # Keep M closest neighbors
        distances = [(DistanceMetrics.euclidean(self.vectors[node_idx], 
                                                self.vectors[n]), n) 
                     for n in neighbors]
        distances.sort()
        self.graph[node_idx] = [n for _, n in distances[:self.M]]
    
    def search(self, query: np.ndarray, k: int, ef: int = 50) -> np.ndarray:
        """Search for k nearest neighbors"""
        candidates = self._search_layer(query, max(ef, k))
        
        # Rank by distance
        distances = [(DistanceMetrics.euclidean(query, self.vectors[idx]), idx) 
                    for idx in candidates]
        distances.sort()
        return np.array([idx for _, idx in distances[:k]])
    
    def memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        graph_mem = sum(len(neighbors) for neighbors in self.graph.values()) * 8 / (1024 ** 2)
        return vector_mem + graph_mem


# ============================================================================
# IVF (Inverted File Index) with Scikit-Learn
# ============================================================================

class IVFIndex:
    """Inverted File Index with scikit-learn KMeans"""
    
    def __init__(self, vectors: np.ndarray, n_clusters: int = 100, 
                 use_minibatch: bool = True):
        self.vectors = vectors
        self.N = len(vectors)
        self.n_clusters = n_clusters
        self.use_minibatch = use_minibatch
        self.kmeans = None
        self.inverted_lists = defaultdict(list)
    
    def build(self):
        """Build IVF index using scikit-learn KMeans"""
        start_time = time.time()
        
        print(f"  Running K-Means clustering (k={self.n_clusters})...")
        
        # Use MiniBatchKMeans for large datasets
        if self.use_minibatch and self.N > 10000:
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                batch_size=1024,
                n_init=3,
                max_iter=100,
                verbose=0
            )
        else:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=0
            )
        
        # Fit and assign
        labels = self.kmeans.fit_predict(self.vectors)
        
        # Build inverted lists
        for i, label in enumerate(labels):
            self.inverted_lists[label].append(i)
        
        # Compute clustering quality
        if self.N < 50000:  # Silhouette score is expensive
            silhouette = silhouette_score(self.vectors[:5000], 
                                         labels[:5000], 
                                         sample_size=min(1000, self.N))
            print(f"  Clustering silhouette score: {silhouette:.3f}")
        
        return time.time() - start_time
    
    def search(self, query: np.ndarray, k: int, n_probe: int = 10) -> np.ndarray:
        """Search n_probe clusters and return k nearest neighbors"""
        # Find n_probe closest clusters using sklearn
        centroid_distances = DistanceMetrics.euclidean_batch(query, 
                                                             self.kmeans.cluster_centers_)
        closest_clusters = np.argpartition(centroid_distances, 
                                          min(n_probe, self.n_clusters))[:n_probe]
        
        # Exhaustive search within selected clusters
        candidates = []
        for cluster_idx in closest_clusters:
            for vector_idx in self.inverted_lists[cluster_idx]:
                distance = DistanceMetrics.euclidean(query, self.vectors[vector_idx])
                candidates.append((distance, vector_idx))
        
        # Sort and return top k
        if not candidates:
            return np.array([])
        
        candidates.sort()
        return np.array([idx for _, idx in candidates[:k]])
    
    def memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        centroid_mem = self.kmeans.cluster_centers_.nbytes / (1024 ** 2)
        invlist_mem = sum(len(lst) for lst in self.inverted_lists.values()) * 8 / (1024 ** 2)
        return vector_mem + centroid_mem + invlist_mem


# ============================================================================
# ZGQ (Zonal Graph Quantization) with Scikit-Learn
# ============================================================================

class ZGQIndex:
    """Zonal Graph Quantization - Hybrid IVF + HNSW with sklearn"""
    
    def __init__(self, vectors: np.ndarray, n_zones: int = 50, M: int = 16,
                 use_minibatch: bool = True):
        self.vectors = vectors
        self.N = len(vectors)
        self.n_zones = n_zones
        self.M = M
        self.use_minibatch = use_minibatch
        self.kmeans = None
        self.zones = defaultdict(list)
        self.local_graphs = {}
    
    def build(self):
        """Build ZGQ index"""
        start_time = time.time()
        
        print(f"  Phase 1: Zonal partitioning (k={self.n_zones})...")
        
        # Phase 1: Use scikit-learn for better K-Means
        if self.use_minibatch and self.N > 10000:
            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_zones,
                random_state=42,
                batch_size=1024,
                n_init=3,
                max_iter=100,
                verbose=0
            )
        else:
            self.kmeans = KMeans(
                n_clusters=self.n_zones,
                random_state=42,
                n_init=10,
                max_iter=300,
                verbose=0
            )
        
        # Fit and assign vectors to zones
        labels = self.kmeans.fit_predict(self.vectors)
        
        for i, label in enumerate(labels):
            self.zones[label].append(i)
        
        # Print zone statistics
        zone_sizes = [len(self.zones[i]) for i in range(self.n_zones) if i in self.zones]
        print(f"  Zone size stats - Min: {min(zone_sizes)}, "
              f"Max: {max(zone_sizes)}, Mean: {np.mean(zone_sizes):.1f}")
        
        # Phase 2: Build local HNSW graphs for each zone
        print(f"  Phase 2: Building local HNSW graphs...")
        built_zones = 0
        for zone_idx, vector_indices in self.zones.items():
            if len(vector_indices) > 1:
                zone_vectors = self.vectors[vector_indices]
                local_hnsw = SimpleHNSW(zone_vectors, M=self.M, ef_construction=100)
                local_hnsw.build()
                self.local_graphs[zone_idx] = (local_hnsw, vector_indices)
                built_zones += 1
        
        print(f"  Built {built_zones}/{self.n_zones} local graphs")
        
        return time.time() - start_time
    
    def search(self, query: np.ndarray, k: int, n_probe: int = 5) -> np.ndarray:
        """Hybrid search: select zones + HNSW search within zones"""
        # Phase 1: Select n_probe closest zones using sklearn centroids
        centroid_distances = DistanceMetrics.euclidean_batch(query, 
                                                             self.kmeans.cluster_centers_)
        closest_zones = np.argpartition(centroid_distances, 
                                       min(n_probe, self.n_zones))[:n_probe]
        
        # Phase 2: HNSW search within each selected zone
        all_candidates = []
        for zone_idx in closest_zones:
            if zone_idx in self.local_graphs:
                local_hnsw, vector_indices = self.local_graphs[zone_idx]
                local_results = local_hnsw.search(query, k, ef=50)
                
                # Map local indices back to global indices
                global_indices = [vector_indices[local_idx] for local_idx in local_results 
                                 if local_idx < len(vector_indices)]
                all_candidates.extend(global_indices)
        
        # Phase 3: Aggregate and re-rank
        if not all_candidates:
            return np.array([])
        
        distances = [(DistanceMetrics.euclidean(query, self.vectors[idx]), idx) 
                    for idx in all_candidates]
        distances.sort()
        return np.array([idx for _, idx in distances[:k]])
    
    def memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        centroid_mem = self.kmeans.cluster_centers_.nbytes / (1024 ** 2)
        
        graph_mem = 0
        for local_hnsw, _ in self.local_graphs.values():
            graph_edges = sum(len(neighbors) for neighbors in local_hnsw.graph.values())
            graph_mem += graph_edges * 8 / (1024 ** 2)
        
        return vector_mem + centroid_mem + graph_mem


# ============================================================================
# Enhanced Benchmarking Framework with Scikit-Learn Metrics
# ============================================================================

class ANNSBenchmark:
    """Comprehensive ANNS benchmarking suite with sklearn metrics"""
    
    def __init__(self, vectors: np.ndarray, queries: np.ndarray, k: int = 10):
        self.vectors = vectors
        self.queries = queries
        self.k = k
        self.N, self.d = vectors.shape
        
        # Compute ground truth
        print(f"Computing ground truth for {len(queries)} queries...")
        self.ground_truth = self._compute_ground_truth()
    
    def _compute_ground_truth(self) -> List[np.ndarray]:
        """Compute exact nearest neighbors for all queries"""
        bf_index = BruteForceIndex(self.vectors)
        ground_truth = []
        for i, query in enumerate(self.queries):
            if (i + 1) % 20 == 0:
                print(f"  Ground truth: {i+1}/{len(self.queries)}", end='\r')
            indices, _ = bf_index.search(query, self.k)
            ground_truth.append(indices)
        print()
        return ground_truth
    
    def compute_recall_at_k(self, results: List[np.ndarray], 
                           k_values: List[int] = [1, 5, 10]) -> Dict[int, float]:
        """Compute recall@k for multiple k values"""
        recall_dict = {}
        
        for k_val in k_values:
            if k_val > self.k:
                continue
            
            recalls = []
            for result, truth in zip(results, self.ground_truth):
                result_at_k = result[:k_val]
                truth_at_k = truth[:k_val]
                intersection = len(set(result_at_k) & set(truth_at_k))
                recalls.append(intersection / k_val)
            
            recall_dict[k_val] = np.mean(recalls)
        
        return recall_dict
    
    def compute_recall(self, results: List[np.ndarray]) -> float:
        """Compute average recall@k"""
        return self.compute_recall_at_k(results, [self.k])[self.k]
    
    def benchmark_hnsw(self, M: int = 16, ef_construction: int = 200, 
                       ef_search: int = 50) -> BenchmarkResult:
        """Benchmark HNSW"""
        print(f"\n{'='*60}")
        print(f"Benchmarking HNSW (M={M}, ef_c={ef_construction}, ef_s={ef_search})")
        print(f"{'='*60}")
        
        index = SimpleHNSW(self.vectors, M=M, ef_construction=ef_construction)
        build_time = index.build()
        
        # Query
        results = []
        start_time = time.time()
        for i, query in enumerate(self.queries):
            if (i + 1) % 20 == 0:
                print(f"  Querying: {i+1}/{len(self.queries)}", end='\r')
            result = index.search(query, self.k, ef=ef_search)
            results.append(result)
        print()
        query_time = (time.time() - start_time) / len(self.queries)
        
        recall = self.compute_recall(results)
        recall_at_k = self.compute_recall_at_k(results, [1, 5, 10])
        memory_mb = index.memory_usage()
        qps = 1.0 / query_time if query_time > 0 else 0
        
        return BenchmarkResult("HNSW", build_time, query_time, recall, 
                             memory_mb, qps, recall_at_k)
    
    def benchmark_ivf(self, n_clusters: int = 100, n_probe: int = 10) -> BenchmarkResult:
        """Benchmark IVF with sklearn"""
        print(f"\n{'='*60}")
        print(f"Benchmarking IVF (clusters={n_clusters}, n_probe={n_probe})")
        print(f"{'='*60}")
        
        index = IVFIndex(self.vectors, n_clusters=n_clusters)
        build_time = index.build()
        
        # Query
        results = []
        start_time = time.time()
        for i, query in enumerate(self.queries):
            if (i + 1) % 20 == 0:
                print(f"  Querying: {i+1}/{len(self.queries)}", end='\r')
            result = index.search(query, self.k, n_probe=n_probe)
            results.append(result)
        print()
        query_time = (time.time() - start_time) / len(self.queries)
        
        recall = self.compute_recall(results)
        recall_at_k = self.compute_recall_at_k(results, [1, 5, 10])
        memory_mb = index.memory_usage()
        qps = 1.0 / query_time if query_time > 0 else 0
        
        return BenchmarkResult("IVF", build_time, query_time, recall, 
                             memory_mb, qps, recall_at_k)
    
    def benchmark_zgq(self, n_zones: int = 50, n_probe: int = 5, 
                      M: int = 16) -> BenchmarkResult:
        """Benchmark ZGQ with sklearn"""
        print(f"\n{'='*60}")
        print(f"Benchmarking ZGQ (zones={n_zones}, n_probe={n_probe}, M={M})")
        print(f"{'='*60}")
        
        index = ZGQIndex(self.vectors, n_zones=n_zones, M=M)
        build_time = index.build()
        
        # Query
        results = []
        start_time = time.time()
        for i, query in enumerate(self.queries):
            if (i + 1) % 20 == 0:
                print(f"  Querying: {i+1}/{len(self.queries)}", end='\r')
            result = index.search(query, self.k, n_probe=n_probe)
            results.append(result)
        print()
        query_time = (time.time() - start_time) / len(self.queries)
        
        recall = self.compute_recall(results)
        recall_at_k = self.compute_recall_at_k(results, [1, 5, 10])
        memory_mb = index.memory_usage()
        qps = 1.0 / query_time if query_time > 0 else 0
        
        return BenchmarkResult("ZGQ", build_time, query_time, recall, 
                             memory_mb, qps, recall_at_k)
    
    def run_comprehensive_benchmark(self) -> Dict:
        """Run complete benchmark suite"""
        results = {
            'hnsw': self.benchmark_hnsw(),
            'ivf': self.benchmark_ivf(),
            'zgq': self.benchmark_zgq()
        }
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        for name, result in results.items():
            print(result)
            if result.recall_at_k:
                print(f"         Recall@1: {result.recall_at_k.get(1, 0):.3f} | "
                      f"Recall@5: {result.recall_at_k.get(5, 0):.3f} | "
                      f"Recall@10: {result.recall_at_k.get(10, 0):.3f}")
        print("="*80)
        
        return results


# ============================================================================
# Enhanced Visualization with More Metrics
# ============================================================================

def plot_comprehensive_results(results: Dict):
    """Enhanced visualization with recall@k curves"""
    algorithms = list(results.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('ANNS Algorithm Comparison: ZGQ vs HNSW vs IVF', 
                 fontsize=16, fontweight='bold')
    
    colors = {'hnsw': '#3498db', 'ivf': '#e74c3c', 'zgq': '#2ecc71'}
    
    # Query Time
    query_times = [results[alg].query_time * 1000 for alg in algorithms]
    axes[0, 0].bar(algorithms, query_times, color=[colors[alg] for alg in algorithms])
    axes[0, 0].set_ylabel('Query Time (ms)', fontweight='bold')
    axes[0, 0].set_title('Query Latency (Lower is Better)')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall@10
    recalls = [results[alg].recall for alg in algorithms]
    axes[0, 1].bar(algorithms, recalls, color=[colors[alg] for alg in algorithms])
    axes[0, 1].set_ylabel('Recall@10', fontweight='bold')
    axes[0, 1].set_title('Recall (Higher is Better)')
    axes[0, 1].set_ylim([0, 1.0])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Memory Usage
    memory = [results[alg].memory_mb for alg in algorithms]
    axes[0, 2].bar(algorithms, memory, color=[colors[alg] for alg in algorithms])
    axes[0, 2].set_ylabel('Memory (MB)', fontweight='bold')
    axes[0, 2].set_title('Memory Footprint (Lower is Better)')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # Build Time
    build_times = [results[alg].build_time for alg in algorithms]
    axes[1, 0].bar(algorithms, build_times, color=[colors[alg] for alg in algorithms])
    axes[1, 0].set_ylabel('Build Time (s)', fontweight='bold')
    axes[1, 0].set_title('Index Construction Time')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # QPS
    qps = [results[alg].queries_per_second for alg in algorithms]
    axes[1, 1].bar(algorithms, qps, color=[colors[alg] for alg in algorithms])
    axes[1, 1].set_ylabel('Queries Per Second', fontweight='bold')
    axes[1, 1].set_title('Throughput (Higher is Better)')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Recall@k curves
    k_values = [1, 5, 10]
    for alg in algorithms:
        if results[alg].recall_at_k:
            recalls_at_k = [results[alg].recall_at_k.get(k, 0) for k in k_values]
            axes[1, 2].plot(k_values, recalls_at_k, marker='o', 
                          label=alg.upper(), color=colors[alg], linewidth=2)
    
    axes[1, 2].set_xlabel('k', fontweight='bold')
    axes[1, 2].set_ylabel('Recall@k', fontweight='bold')
    axes[1, 2].set_title('Recall@k Curves')
    axes[1, 2].legend()
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig('anns_benchmark_comprehensive.png', dpi=300, bbox_inches='tight')
    print("\n✓ Comprehensive plot saved as 'anns_benchmark_comprehensive.png'")


# ============================================================================
# Dataset Generators
# ============================================================================

def generate_synthetic_dataset(N: int = 10000, d: int = 128, 
                               n_queries: int = 100,
                               dataset_type: str = 'clustered') -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic datasets with different distributions"""
    np.random.seed(42)
    
    if dataset_type == 'clustered':
        # Create clustered data (most realistic)
        n_clusters = max(10, N // 500)
        vectors = []
        for _ in range(n_clusters):
            center = np.random.randn(d) * 10
            cluster_vectors = center + np.random.randn(N // n_clusters, d)
            vectors.append(cluster_vectors)
        vectors = np.vstack(vectors)
        
    elif dataset_type == 'uniform':
        # Uniform random distribution
        vectors = np.random.randn(N, d)
        
    elif dataset_type == 'gaussian':
        # Single Gaussian
        vectors = np.random.randn(N, d) * 5
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Generate queries as slight perturbations of random vectors
    query_indices = np.random.choice(N, n_queries, replace=False)
    queries = vectors[query_indices] + np.random.randn(n_queries, d) * 0.1
    
    return vectors, queries


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ZGQ BENCHMARK: Scikit-Learn Enhanced Implementation")
    print("="*80)
    
    # Generate dataset
    print("\nGenerating synthetic dataset...")
    N, d = 10000, 128
    vectors, queries = generate_synthetic_dataset(N=N, d=d, n_queries=100, 
                                                  dataset_type='clustered')
    
    print(f"Dataset: {N:,} vectors, {d} dimensions")
    print(f"Queries: {len(queries)}")
    print(f"Memory size: {vectors.nbytes / (1024**2):.2f} MB")
    
    # Run benchmark
    benchmark = ANNSBenchmark(vectors, queries, k=10)
    results = benchmark.run_comprehensive_benchmark()
    
    # Visualize
    plot_comprehensive_results(results)
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("DETAILED ANALYSIS")
    print("="*80)
    
    # Memory efficiency
    print("\nMemory Efficiency (relative to HNSW):")
    hnsw_mem = results['hnsw'].memory_mb
    for name, result in results.items():
        ratio = result.memory_mb / hnsw_mem
        print(f"  {name.upper():8s}: {ratio:.2%} ({result.memory_mb:.1f} MB)")
    
    # Speed comparison
    print("\nQuery Speed (relative to fastest):")
    fastest_time = min(r.query_time for r in results.values())
    for name, result in results.items():
        ratio = result.query_time / fastest_time
        print(f"  {name.upper():8s}: {ratio:.2f}x slower ({result.query_time*1000:.3f} ms)")
    
    # Recall quality
    print("\nRecall Quality:")
    for name, result in results.items():
        print(f"  {name.upper():8s}: {result.recall:.3f}")
    
    # Build time comparison
    print("\nIndex Build Time:")
    for name, result in results.items():
        print(f"  {name.upper():8s}: {result.build_time:.2f}s")
    
    # Overall score (weighted)
    print("\n" + "="*80)
    print("OVERALL PERFORMANCE SCORE")
    print("="*80)
    print("Weighted score: 40% recall, 30% speed, 20% memory, 10% build time")
    print()
    
    for name, result in results.items():
        # Normalize metrics (0-1 scale)
        recall_score = result.recall
        speed_score = 1.0 / (1.0 + result.query_time * 1000)  # Lower is better
        memory_score = 1.0 / (1.0 + result.memory_mb / 100)   # Lower is better
        build_score = 1.0 / (1.0 + result.build_time / 10)    # Lower is better
        
        overall = (0.40 * recall_score + 
                  0.30 * speed_score + 
                  0.20 * memory_score + 
                  0.10 * build_score)
        
        print(f"  {name.upper():8s}: {overall:.3f} "
              f"(R:{recall_score:.2f} S:{speed_score:.2f} "
              f"M:{memory_score:.2f} B:{build_score:.2f})")
    
    print("\n" + "="*80)
    print("✓ Benchmark complete! Check 'anns_benchmark_comprehensive.png' for visualizations")
    print("="*80)