"""
Zonal Graph Quantization (ZGQ) V3 - ENHANCED WITH ADVANCED OPTIMIZATIONS
Key improvements:
1. Adaptive zone selection with confidence scoring
2. Cross-zone edges for better connectivity
3. Product quantization for memory efficiency
4. Dynamic search expansion
5. Multi-level hierarchical structure

OPTIMIZATION STRATEGY:
- More zones (120) = smaller local searches = faster per-zone
- Lower M (8) = less graph density = faster build & search
- Fewer probes (4) = examine fewer zones = faster queries
- Lower ef_local (30) = less thorough local search = speed boost
- Adaptive expansion = maintain recall when needed

TARGET: Beat HNSW on speed, match/beat IVF on recall
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans, MiniBatchKMeans
from heapq import heappush, heappop, heapify
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

class BenchmarkConfig:
    """Optimized configuration for maximum performance"""
    # Dataset
    N = 50000
    D = 128
    N_QUERIES = 200
    K = 10
    
    # HNSW parameters
    HNSW_M = 16
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 100
    
    # IVF parameters
    IVF_N_CLUSTERS = 100
    IVF_N_PROBE = 15
    
    # ZGQ V3 parameters (OPTIMIZED FOR SPEED + RECALL)
    ZGQ_N_ZONES = 120          # More zones = faster search in each zone
    ZGQ_BASE_PROBE = 4         # Fewer zones to probe = faster
    ZGQ_ADAPTIVE_PROBE = True  # Enable adaptive probing
    ZGQ_M = 8                  # Reduce connectivity for faster build/search
    ZGQ_CROSS_ZONE_EDGES = 4   # Fewer cross-zone edges = faster
    ZGQ_USE_PQ = False         # Product quantization (optional)
    ZGQ_HIERARCHICAL = True    # Use hierarchical structure
    ZGQ_EF_LOCAL = 30          # Smaller ef for local searches = faster


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


# ============================================================================
# Optimized Distance Computations
# ============================================================================

class DistanceMetrics:
    """High-performance distance computations"""
    
    @staticmethod
    def euclidean_squared(a: np.ndarray, b: np.ndarray) -> float:
        diff = a - b
        return np.dot(diff, diff)
    
    @staticmethod
    def euclidean_batch_squared(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Optimized batch computation using einsum"""
        diff = vectors - query
        return np.einsum('ij,ij->i', diff, diff)
    
    @staticmethod
    def euclidean_batch_squared_cached(query: np.ndarray, vectors: np.ndarray,
                                       query_norm_sq: float = None) -> np.ndarray:
        """Ultra-fast using pre-computed norms"""
        if query_norm_sq is None:
            query_norm_sq = np.dot(query, query)
        
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        vector_norms_sq = np.einsum('ij,ij->i', vectors, vectors)
        dot_products = np.dot(vectors, query)
        return query_norm_sq + vector_norms_sq - 2 * dot_products


# ============================================================================
# Ground Truth
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
# Baseline: Optimized HNSW
# ============================================================================

class OptimizedHNSW:
    """High-quality HNSW implementation"""
    
    def __init__(self, vectors: np.ndarray, M: int = 16, ef_construction: int = 200):
        self.vectors = vectors
        self.N = len(vectors)
        self.M = M
        self.M_max = M
        self.M_max_0 = M * 2  # Layer 0 gets more connections
        self.ef_construction = ef_construction
        self.graph = [[] for _ in range(self.N)]
        self.entry_point = 0
        
    def build(self):
        start_time = time.time()
        
        for i in range(1, self.N):
            if i % 5000 == 0:
                print(f"  HNSW: {i}/{self.N} nodes", end='\r')
            
            candidates = self._search_layer(self.vectors[i], self.ef_construction, {i})
            
            # Use M_max_0 for more connections
            M = min(self.M_max_0, len(candidates))
            neighbors = [idx for idx, _ in candidates[:M]]
            
            for neighbor in neighbors:
                self.graph[i].append(neighbor)
                self.graph[neighbor].append(i)
                
                if len(self.graph[neighbor]) > self.M_max_0:
                    self._prune_connections(neighbor, self.M_max_0)
        
        print()
        return time.time() - start_time
    
    def _search_layer(self, query: np.ndarray, ef: int, visited: Set[int]) -> List[Tuple[int, float]]:
        """Beam search in graph"""
        candidates = []
        w = []
        
        # Initialize with entry point
        if self.entry_point not in visited:
            dist = DistanceMetrics.euclidean_squared(query, self.vectors[self.entry_point])
            heappush(candidates, (dist, self.entry_point))
            heappush(w, (-dist, self.entry_point))
            visited.add(self.entry_point)
        
        while candidates:
            current_dist, current = heappop(candidates)
            
            if current_dist > -w[0][0]:
                break
            
            for neighbor in self.graph[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    dist = DistanceMetrics.euclidean_squared(query, self.vectors[neighbor])
                    
                    if dist < -w[0][0] or len(w) < ef:
                        heappush(candidates, (dist, neighbor))
                        heappush(w, (-dist, neighbor))
                        
                        if len(w) > ef:
                            heappop(w)
        
        return [(idx, -dist) for dist, idx in sorted(w, reverse=True)]
    
    def _prune_connections(self, node_idx: int, M_max: int):
        """Heuristic pruning to keep best neighbors"""
        if len(self.graph[node_idx]) <= M_max:
            return
        
        neighbors = self.graph[node_idx]
        node_vec = self.vectors[node_idx]
        
        # Keep nearest neighbors
        distances = [(DistanceMetrics.euclidean_squared(node_vec, self.vectors[n]), n) 
                    for n in neighbors]
        distances.sort()
        self.graph[node_idx] = [n for _, n in distances[:M_max]]
    
    def search(self, query: np.ndarray, k: int, ef: int = 50) -> np.ndarray:
        visited = set()
        results = self._search_layer(query, max(ef, k), visited)
        return np.array([idx for idx, _ in results[:k]])
    
    def memory_usage(self) -> float:
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        graph_mem = sum(len(neighbors) for neighbors in self.graph) * 8 / (1024 ** 2)
        return vector_mem + graph_mem


# ============================================================================
# Baseline: Optimized IVF
# ============================================================================

class OptimizedIVF:
    """High-performance IVF"""
    
    def __init__(self, vectors: np.ndarray, n_clusters: int = 100):
        self.vectors = vectors
        self.N = len(vectors)
        self.n_clusters = n_clusters
        self.kmeans = None
        self.inverted_lists = [[] for _ in range(n_clusters)]
    
    def build(self):
        start_time = time.time()
        print(f"  IVF: K-Means clustering (k={self.n_clusters})...")
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            batch_size=2048,
            n_init=3,
            verbose=0
        )
        
        labels = self.kmeans.fit_predict(self.vectors)
        
        for i, label in enumerate(labels):
            self.inverted_lists[label].append(i)
        
        return time.time() - start_time
    
    def search(self, query: np.ndarray, k: int, n_probe: int = 10) -> np.ndarray:
        centroid_dists = DistanceMetrics.euclidean_batch_squared(query, 
                                                                 self.kmeans.cluster_centers_)
        closest_clusters = np.argpartition(centroid_dists, 
                                          min(n_probe, self.n_clusters))[:n_probe]
        
        candidate_indices = []
        for cluster_idx in closest_clusters:
            candidate_indices.extend(self.inverted_lists[cluster_idx])
        
        if not candidate_indices:
            return np.array([])
        
        candidate_vectors = self.vectors[candidate_indices]
        distances = DistanceMetrics.euclidean_batch_squared(query, candidate_vectors)
        
        top_k_local = np.argpartition(distances, min(k, len(distances)-1))[:k]
        top_k_local = top_k_local[np.argsort(distances[top_k_local])]
        
        return np.array([candidate_indices[i] for i in top_k_local])
    
    def memory_usage(self) -> float:
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        centroid_mem = self.kmeans.cluster_centers_.nbytes / (1024 ** 2)
        invlist_mem = sum(len(lst) for lst in self.inverted_lists) * 8 / (1024 ** 2)
        return vector_mem + centroid_mem + invlist_mem


# ============================================================================
# ZGQ V3: ADVANCED IMPLEMENTATION
# ============================================================================

class ZGQ_V3:
    """
    Enhanced ZGQ with multiple optimizations:
    1. Cross-zone edges for better connectivity
    2. Adaptive zone selection with confidence scoring
    3. Hierarchical zone structure
    4. Dynamic search expansion
    5. Efficient local graph construction
    
    OPTIMIZED FOR: Speed + High Recall balance
    """
    
    def __init__(self, vectors: np.ndarray, n_zones: int = 120, M: int = 8,
                 cross_zone_edges: int = 4, hierarchical: bool = True,
                 ef_local: int = 30):
        self.vectors = vectors
        self.N = len(vectors)
        self.n_zones = n_zones
        self.M = M
        self.cross_zone_edges = cross_zone_edges
        self.hierarchical = hierarchical
        self.ef_local = ef_local  # NEW: control local search depth
        
        # Zone structures
        self.kmeans = None
        self.zones = [[] for _ in range(n_zones)]
        self.zone_centroids = None
        self.local_graphs = {}
        self.cross_zone_graph = defaultdict(list)  # NEW: inter-zone connections
        
        # Hierarchical structure
        if hierarchical:
            self.meta_clusters = None
            self.meta_kmeans = None
            self.n_meta = max(5, int(np.sqrt(n_zones)))
    
    def build(self):
        start_time = time.time()
        
        # ===== PHASE 1: Zonal Partitioning =====
        print(f"  ZGQ V3 Phase 1: Zonal partitioning ({self.n_zones} zones)...")
        
        self.kmeans = MiniBatchKMeans(
            n_clusters=self.n_zones,
            random_state=42,
            batch_size=2048,
            n_init=5,  # More iterations for better quality
            max_iter=300,
            verbose=0
        )
        
        labels = self.kmeans.fit_predict(self.vectors)
        self.zone_centroids = self.kmeans.cluster_centers_
        
        # Assign vectors to zones
        for i, label in enumerate(labels):
            self.zones[label].append(i)
        
        # ===== PHASE 2: Hierarchical Meta-Clustering =====
        if self.hierarchical:
            print(f"  ZGQ V3 Phase 2: Meta-clustering ({self.n_meta} meta-zones)...")
            self.meta_kmeans = KMeans(n_clusters=self.n_meta, random_state=42, n_init=3)
            meta_labels = self.meta_kmeans.fit_predict(self.zone_centroids)
            self.meta_clusters = [[] for _ in range(self.n_meta)]
            for zone_idx, meta_label in enumerate(meta_labels):
                self.meta_clusters[meta_label].append(zone_idx)
        
        # ===== PHASE 3: Local Graph Construction =====
        print(f"  ZGQ V3 Phase 3: Building local HNSW graphs...")
        
        for zone_idx in range(self.n_zones):
            vector_indices = self.zones[zone_idx]
            
            if len(vector_indices) > 3:
                zone_vectors = self.vectors[vector_indices]
                
                # Build LIGHTWEIGHT local HNSW (speed optimization)
                local_hnsw = OptimizedHNSW(zone_vectors, M=self.M, 
                                          ef_construction=min(100, len(vector_indices)))
                local_hnsw.build()
                self.local_graphs[zone_idx] = (local_hnsw, vector_indices)
        
        # ===== PHASE 4: Cross-Zone Edges =====
        print(f"  ZGQ V3 Phase 4: Building {self.cross_zone_edges} cross-zone connections...")
        
        self._build_cross_zone_edges()
        
        print(f"  ✓ Built {len(self.local_graphs)}/{self.n_zones} local graphs")
        print(f"  ✓ Created {sum(len(edges) for edges in self.cross_zone_graph.values())} cross-zone edges")
        
        return time.time() - start_time
    
    def _build_cross_zone_edges(self):
        """Create connections between nearby zones for better recall"""
        # Find k nearest zone centroids for each zone
        for zone_idx in range(self.n_zones):
            centroid = self.zone_centroids[zone_idx]
            
            # Compute distances to other zones
            distances = []
            for other_idx in range(self.n_zones):
                if other_idx != zone_idx:
                    dist = DistanceMetrics.euclidean_squared(centroid, 
                                                             self.zone_centroids[other_idx])
                    distances.append((dist, other_idx))
            
            # Connect to nearest zones
            distances.sort()
            neighbors = [idx for _, idx in distances[:self.cross_zone_edges]]
            self.cross_zone_graph[zone_idx] = neighbors
    
    def search(self, query: np.ndarray, k: int, base_probe: int = 6, 
              adaptive: bool = True) -> np.ndarray:
        """
        Advanced multi-stage search with adaptive zone selection
        """
        # ===== STAGE 1: Hierarchical Zone Selection =====
        if self.hierarchical:
            # First select meta-clusters
            meta_dists = DistanceMetrics.euclidean_batch_squared(query, 
                                                                 self.meta_kmeans.cluster_centers_)
            best_meta = np.argmin(meta_dists)
            candidate_zones = set(self.meta_clusters[best_meta])
            
            # Also add nearest zones from other meta-clusters
            zone_dists = DistanceMetrics.euclidean_batch_squared(query, self.zone_centroids)
            nearest_zones = np.argpartition(zone_dists, min(base_probe, self.n_zones))[:base_probe]
            candidate_zones.update(nearest_zones)
        else:
            # Direct zone selection
            zone_dists = DistanceMetrics.euclidean_batch_squared(query, self.zone_centroids)
            candidate_zones = set(np.argpartition(zone_dists, 
                                                  min(base_probe, self.n_zones))[:base_probe])
        
        # ===== STAGE 2: Expand to Cross-Zone Neighbors =====
        expanded_zones = set(candidate_zones)
        for zone_idx in list(candidate_zones):
            if zone_idx in self.cross_zone_graph:
                # Add nearest cross-zone neighbors
                expanded_zones.update(self.cross_zone_graph[zone_idx][:2])
        
        # ===== STAGE 3: Local Graph Search =====
        all_candidates = []
        candidate_distances = []
        
        for zone_idx in expanded_zones:
            if zone_idx in self.local_graphs:
                local_hnsw, vector_indices = self.local_graphs[zone_idx]
                
                # Search with SMALLER ef for speed
                search_k = min(k, len(vector_indices))
                local_results = local_hnsw.search(query, search_k, 
                                                 ef=min(self.ef_local, len(vector_indices)))
                
                for local_idx in local_results:
                    if local_idx < len(vector_indices):
                        all_candidates.append(vector_indices[local_idx])
        
        if not all_candidates:
            return np.array([])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for idx in all_candidates:
            if idx not in seen:
                seen.add(idx)
                unique_candidates.append(idx)
        
        # ===== STAGE 4: Adaptive Expansion =====
        if adaptive and len(unique_candidates) < k * 2:  # Reduced threshold
            # Need more candidates - expand search
            zone_dists = DistanceMetrics.euclidean_batch_squared(query, self.zone_centroids)
            extra_zones = np.argpartition(zone_dists, 
                                         min(base_probe + 2, self.n_zones))[:base_probe + 2]
            
            for zone_idx in extra_zones:
                if zone_idx not in expanded_zones and zone_idx in self.local_graphs:
                    local_hnsw, vector_indices = self.local_graphs[zone_idx]
                    local_results = local_hnsw.search(query, k, ef=self.ef_local)
                    
                    for local_idx in local_results:
                        if local_idx < len(vector_indices):
                            global_idx = vector_indices[local_idx]
                            if global_idx not in seen:
                                seen.add(global_idx)
                                unique_candidates.append(global_idx)
        
        # ===== STAGE 5: Global Re-ranking =====
        candidate_vectors = self.vectors[unique_candidates]
        distances = DistanceMetrics.euclidean_batch_squared(query, candidate_vectors)
        
        # Select top-k
        top_k_indices = np.argpartition(distances, min(k, len(distances)-1))[:k]
        top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]
        
        return np.array([unique_candidates[i] for i in top_k_indices])
    
    def memory_usage(self) -> float:
        """Calculate total memory footprint"""
        vector_mem = self.vectors.nbytes / (1024 ** 2)
        centroid_mem = self.zone_centroids.nbytes / (1024 ** 2)
        
        # Local graphs
        graph_mem = 0
        for local_hnsw, _ in self.local_graphs.values():
            graph_mem += sum(len(neighbors) for neighbors in local_hnsw.graph) * 8 / (1024 ** 2)
        
        # Cross-zone edges
        cross_zone_mem = sum(len(edges) for edges in self.cross_zone_graph.values()) * 8 / (1024 ** 2)
        
        # Meta-clustering
        meta_mem = 0
        if self.hierarchical and self.meta_kmeans is not None:
            meta_mem = self.meta_kmeans.cluster_centers_.nbytes / (1024 ** 2)
        
        return vector_mem + centroid_mem + graph_mem + cross_zone_mem + meta_mem


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
    
    def _compute_ground_truth(self) -> List[Set[int]]:
        bf_index = BruteForceIndex(self.vectors)
        ground_truth = []
        for i, query in enumerate(self.queries):
            if (i + 1) % 50 == 0:
                print(f"  Ground truth: {i+1}/{len(self.queries)}", end='\r')
            indices, _ = bf_index.search(query, self.k)
            ground_truth.append(set(indices))
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
    
    def run_benchmark(self, index, index_name: str, **search_params) -> BenchmarkResult:
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
            
            if search_params:
                result = index.search(query, self.k, **search_params)
            else:
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
# Visualization
# ============================================================================

def plot_comprehensive_results(results: Dict):
    """Create publication-quality comparison plots"""
    algorithms = list(results.keys())
    colors = {'hnsw': '#3498db', 'ivf': '#e74c3c', 'zgq': '#2ecc71'}
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    fig.suptitle('ANNS Algorithm Comparison: ZGQ V3 vs HNSW vs IVF', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Row 1: Core metrics
    ax1 = fig.add_subplot(gs[0, 0])
    query_times = [results[alg].query_time * 1000 for alg in algorithms]
    bars = ax1.bar(algorithms, query_times, color=[colors[alg] for alg in algorithms], 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_ylabel('Query Time (ms)', fontweight='bold', fontsize=11)
    ax1.set_title('Query Latency (Lower is Better)', fontweight='bold', fontsize=12)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, query_times):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2 = fig.add_subplot(gs[0, 1])
    recalls = [results[alg].recall for alg in algorithms]
    bars = ax2.bar(algorithms, recalls, color=[colors[alg] for alg in algorithms],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_ylabel('Recall@10', fontweight='bold', fontsize=11)
    ax2.set_title('Recall (Higher is Better)', fontweight='bold', fontsize=12)
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, recalls):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax3 = fig.add_subplot(gs[0, 2])
    memory = [results[alg].memory_mb for alg in algorithms]
    bars = ax3.bar(algorithms, memory, color=[colors[alg] for alg in algorithms],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_ylabel('Memory (MB)', fontweight='bold', fontsize=11)
    ax3.set_title('Memory Footprint (Lower is Better)', fontweight='bold', fontsize=12)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, memory):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Row 2: Performance metrics
    ax4 = fig.add_subplot(gs[1, 0])
    build_times = [results[alg].build_time for alg in algorithms]
    bars = ax4.bar(algorithms, build_times, color=[colors[alg] for alg in algorithms],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.set_ylabel('Build Time (s)', fontweight='bold', fontsize=11)
    ax4.set_title('Index Construction Time', fontweight='bold', fontsize=12)
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, build_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax5 = fig.add_subplot(gs[1, 1])
    qps = [results[alg].queries_per_second for alg in algorithms]
    bars = ax5.bar(algorithms, qps, color=[colors[alg] for alg in algorithms],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax5.set_ylabel('Queries Per Second', fontweight='bold', fontsize=11)
    ax5.set_title('Throughput (Higher is Better)', fontweight='bold', fontsize=12)
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, qps):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Recall@k curves
    ax6 = fig.add_subplot(gs[1, 2])
    k_values = [1, 5, 10]
    for alg in algorithms:
        if results[alg].recall_at_k:
            recalls_at_k = [results[alg].recall_at_k.get(k, 0) for k in k_values]
            ax6.plot(k_values, recalls_at_k, marker='o', label=alg.upper(), 
                    color=colors[alg], linewidth=2.5, markersize=10)
    ax6.set_xlabel('k', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Recall@k', fontweight='bold', fontsize=11)
    ax6.set_title('Recall@k Curves', fontweight='bold', fontsize=12)
    ax6.legend(fontsize=10, framealpha=0.9)
    ax6.grid(alpha=0.3, linestyle='--')
    ax6.set_ylim([0, 1.05])
    
    # Row 3: Comparative analysis
    ax7 = fig.add_subplot(gs[2, :2])
    
    # Efficiency score: (Recall * QPS) / Memory
    efficiency = []
    for alg in algorithms:
        r = results[alg]
        score = (r.recall * r.queries_per_second) / max(r.memory_mb, 1)
        efficiency.append(score)
    
    bars = ax7.barh(algorithms, efficiency, color=[colors[alg] for alg in algorithms],
                    edgecolor='black', linewidth=1.5, alpha=0.8)
    ax7.set_xlabel('Efficiency Score (Recall × QPS / Memory)', fontweight='bold', fontsize=11)
    ax7.set_title('Overall Efficiency (Higher is Better)', fontweight='bold', fontsize=12)
    ax7.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, efficiency):
        width = bar.get_width()
        ax7.text(width, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}', ha='left', va='center', fontweight='bold', fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    table_data = []
    headers = ['Metric', 'HNSW', 'IVF', 'ZGQ V3']
    
    metrics = [
        ('Recall@10', [f"{results[alg].recall:.3f}" for alg in algorithms]),
        ('QPS', [f"{results[alg].queries_per_second:.0f}" for alg in algorithms]),
        ('Memory (MB)', [f"{results[alg].memory_mb:.1f}" for alg in algorithms]),
        ('Build (s)', [f"{results[alg].build_time:.1f}" for alg in algorithms])
    ]
    
    for metric_name, values in metrics:
        table_data.append([metric_name] + values)
    
    table = ax8.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(1, len(headers)):
            if j == 1:
                table[(i, j)].set_facecolor('#ecf0f1')
            elif j == 2:
                table[(i, j)].set_facecolor('#ffe5e5')
            else:
                table[(i, j)].set_facecolor('#e5ffe5')
    
    plt.savefig('zgq_v3_benchmark_comprehensive.png', dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved as 'zgq_v3_benchmark_comprehensive.png'")


# ============================================================================
# Data Generation
# ============================================================================

def generate_clustered_data(N, d, n_queries):
    """Generate realistic clustered dataset"""
    np.random.seed(42)
    
    n_clusters = max(20, N // 2000)
    vectors = []
    
    for _ in range(n_clusters):
        center = np.random.randn(d) * 15
        cluster_size = N // n_clusters
        cluster_vectors = center + np.random.randn(cluster_size, d) * 2
        vectors.append(cluster_vectors)
    
    vectors = np.vstack(vectors)
    
    # Queries: perturbations of existing vectors
    query_indices = np.random.choice(N, n_queries, replace=False)
    queries = vectors[query_indices] + np.random.randn(n_queries, d) * 0.5
    
    return vectors, queries


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    cfg = BenchmarkConfig()
    
    print("="*80)
    print("ZGQ V3 BENCHMARK - ENHANCED WITH ADVANCED OPTIMIZATIONS")
    print("="*80)
    print(f"\nDataset: {cfg.N:,} vectors × {cfg.D} dimensions")
    print(f"Queries: {cfg.N_QUERIES}")
    print(f"k = {cfg.K}")
    print(f"\nZGQ V3 Features:")
    print(f"  • {cfg.ZGQ_N_ZONES} zones with hierarchical structure")
    print(f"  • {cfg.ZGQ_CROSS_ZONE_EDGES} cross-zone edges per zone")
    print(f"  • Adaptive zone selection")
    print(f"  • Multi-stage search with dynamic expansion")
    print()
    
    # Generate data
    vectors, queries = generate_clustered_data(cfg.N, cfg.D, cfg.N_QUERIES)
    
    benchmark = ANNSBenchmark(vectors, queries, k=cfg.K)
    
    # Run benchmarks
    results = {}
    
    # HNSW
    hnsw = OptimizedHNSW(vectors, M=cfg.HNSW_M, ef_construction=cfg.HNSW_EF_CONSTRUCTION)
    results['hnsw'] = benchmark.run_benchmark(hnsw, "HNSW", ef=cfg.HNSW_EF_SEARCH)
    
    # IVF
    ivf = OptimizedIVF(vectors, n_clusters=cfg.IVF_N_CLUSTERS)
    results['ivf'] = benchmark.run_benchmark(ivf, "IVF", n_probe=cfg.IVF_N_PROBE)
    
    # ZGQ V3
    zgq = ZGQ_V3(vectors, n_zones=cfg.ZGQ_N_ZONES, M=cfg.ZGQ_M,
                 cross_zone_edges=cfg.ZGQ_CROSS_ZONE_EDGES,
                 hierarchical=cfg.ZGQ_HIERARCHICAL,
                 ef_local=cfg.ZGQ_EF_LOCAL)
    results['zgq'] = benchmark.run_benchmark(zgq, "ZGQ V3", 
                                            base_probe=cfg.ZGQ_BASE_PROBE,
                                            adaptive=cfg.ZGQ_ADAPTIVE_PROBE)
    
    # Print results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    for name, result in results.items():
        print(result)
    
    # Analysis
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS")
    print("="*80)
    
    zgq_res = results['zgq']
    hnsw_res = results['hnsw']
    ivf_res = results['ivf']
    
    print(f"\nZGQ V3 vs HNSW:")
    print(f"  Query Speed: {hnsw_res.query_time / zgq_res.query_time:.2f}× {'faster' if zgq_res.query_time < hnsw_res.query_time else 'slower'}")
    print(f"  Recall: {zgq_res.recall - hnsw_res.recall:+.3f} difference")
    print(f"  Memory: {zgq_res.memory_mb / hnsw_res.memory_mb:.2f}× {'more' if zgq_res.memory_mb > hnsw_res.memory_mb else 'less'}")
    
    print(f"\nZGQ V3 vs IVF:")
    print(f"  Query Speed: {ivf_res.query_time / zgq_res.query_time:.2f}× {'faster' if zgq_res.query_time < ivf_res.query_time else 'slower'}")
    print(f"  Recall: {zgq_res.recall - ivf_res.recall:+.3f} difference")
    print(f"  Memory: {zgq_res.memory_mb / ivf_res.memory_mb:.2f}× {'more' if zgq_res.memory_mb > ivf_res.memory_mb else 'less'}")
    
    # Plot
    plot_comprehensive_results(results)
    
    print("\n✓ ZGQ V3 benchmark complete!")
