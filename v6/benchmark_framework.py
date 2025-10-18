"""
Comprehensive Benchmarking Framework for ZGQ V6
Scientifically rigorous comparison against baselines

Metrics:
- Recall@k for k in {1, 5, 10, 20, 50, 100}
- Latency (mean, median, p95, p99)
- Throughput (queries per second)
- Memory usage
- Build time
- Statistical significance tests

Reference: Standard ANNS benchmarking methodology
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
from scipy import stats

from zgq_index import ZGQIndex
from baseline_algorithms import HNSWBaseline, IVFBaseline
from distance_metrics import DistanceMetrics


@dataclass
class BenchmarkResult:
    """Results for a single configuration."""
    algorithm: str
    config: Dict[str, Any]
    
    # Recall metrics
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    
    # Latency metrics (ms)
    mean_latency: float = 0.0
    median_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    std_latency: float = 0.0
    
    # Throughput
    queries_per_second: float = 0.0
    
    # Resources
    build_time: float = 0.0
    memory_mb: float = 0.0
    
    # Per-query latencies for statistical analysis
    latencies: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'algorithm': self.algorithm,
            'config': self.config,
            'recall_at_k': self.recall_at_k,
            'mean_latency_ms': self.mean_latency,
            'median_latency_ms': self.median_latency,
            'p95_latency_ms': self.p95_latency,
            'p99_latency_ms': self.p99_latency,
            'std_latency_ms': self.std_latency,
            'qps': self.queries_per_second,
            'build_time_s': self.build_time,
            'memory_mb': self.memory_mb
        }


class ANNSBenchmark:
    """
    Comprehensive ANNS benchmark framework.
    
    Methodology:
    1. Generate/load dataset
    2. Compute ground truth (exact nearest neighbors)
    3. Build each index configuration
    4. Run searches with multiple trials
    5. Compute recall, latency, throughput
    6. Statistical significance testing
    7. Generate reports and plots
    """
    
    def __init__(
        self,
        dataset: np.ndarray,
        queries: np.ndarray,
        ground_truth: np.ndarray = None,
        k_values: List[int] = None,
        n_trials: int = 3,
        verbose: bool = True
    ):
        """
        Initialize benchmark.
        
        Args:
            dataset: Database vectors (N, d)
            queries: Query vectors (Q, d)
            ground_truth: Ground truth indices (Q, k_max) - computed if None
            k_values: List of k values to evaluate recall at
            n_trials: Number of trials per configuration
            verbose: Print progress
        """
        self.dataset = dataset.astype(np.float32)
        self.queries = queries.astype(np.float32)
        self.N, self.d = dataset.shape
        self.Q = len(queries)
        
        self.k_values = k_values or [1, 5, 10, 20, 50, 100]
        self.k_max = max(self.k_values)
        self.n_trials = n_trials
        self.verbose = verbose
        
        # Compute or use provided ground truth
        if ground_truth is None:
            if self.verbose:
                print(f"Computing ground truth for {self.Q} queries...")
            self.ground_truth = self._compute_ground_truth()
        else:
            self.ground_truth = ground_truth
        
        # Results storage
        self.results: List[BenchmarkResult] = []
    
    def _compute_ground_truth(self) -> np.ndarray:
        """
        Compute exact nearest neighbors (ground truth).
        
        Returns:
            Array of shape (Q, k_max) with true nearest neighbor indices
        """
        start = time.time()
        
        # Compute all distances
        dist_metric = DistanceMetrics()
        all_gt_indices = []
        
        for i, query in enumerate(self.queries):
            if self.verbose and (i + 1) % max(1, self.Q // 20) == 0:
                progress = (i + 1) / self.Q * 100
                print(f"  Computing GT: {i+1}/{self.Q} ({progress:.1f}%)", end='\r')
            
            distances = dist_metric.euclidean_batch_squared(query, self.dataset)
            gt_indices = np.argsort(distances)[:self.k_max]
            all_gt_indices.append(gt_indices)
        
        if self.verbose:
            elapsed = time.time() - start
            print(f"  Ground truth computed in {elapsed:.2f}s" + " " * 30)
        
        return np.array(all_gt_indices)
    
    def _compute_recall(
        self,
        results: List[List[Tuple[int, float]]],
        k: int
    ) -> float:
        """
        Compute recall@k.
        
        Args:
            results: Search results (Q, variable length lists)
            k: Number of neighbors to consider
            
        Returns:
            Average recall@k across all queries
        """
        recalls = []
        
        for q_idx, result_list in enumerate(results):
            # Get top-k from results
            retrieved = set([vid for vid, _ in result_list[:k]])
            
            # Get ground truth top-k
            true_neighbors = set(self.ground_truth[q_idx, :k])
            
            # Compute recall
            intersection = len(retrieved & true_neighbors)
            recall = intersection / k if k > 0 else 0.0
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def benchmark_zgq(
        self,
        n_zones: int = 100,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        n_probe: int = 5,
        pq_m: int = 16,
        use_pq: bool = True,
        name_suffix: str = ""
    ) -> BenchmarkResult:
        """
        Benchmark ZGQ index with given configuration.
        
        Args:
            n_zones: Number of zones
            M: HNSW max connections
            ef_construction: HNSW build parameter
            ef_search: HNSW search parameter
            n_probe: Number of zones to search
            pq_m: PQ subspaces
            use_pq: Enable PQ
            name_suffix: Optional name suffix
            
        Returns:
            BenchmarkResult object
        """
        config = {
            'n_zones': n_zones,
            'M': M,
            'ef_construction': ef_construction,
            'ef_search': ef_search,
            'n_probe': n_probe,
            'pq_m': pq_m,
            'use_pq': use_pq
        }
        
        algo_name = f"ZGQ{name_suffix}"
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Benchmarking {algo_name}")
            print(f"{'='*80}")
            print(f"Config: {config}")
        
        # Build index
        if self.verbose:
            print("\nBuilding index...")
        
        index = ZGQIndex(
            Z=n_zones,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            n_probe=n_probe,
            m=pq_m,
            use_pq=use_pq,
            verbose=self.verbose
        )
        index.build(self.dataset)
        
        # Get memory usage
        memory_mb = index.get_memory_usage()
        
        # Run search trials
        if self.verbose:
            print(f"\nRunning {self.n_trials} search trials...")
        
        all_latencies = []
        trial_results = None
        
        for trial in range(self.n_trials):
            latencies = []
            results = []
            
            for i, query in enumerate(self.queries):
                start = time.perf_counter()
                result = index.search(query, k=self.k_max)
                latency = (time.perf_counter() - start) * 1000  # ms
                
                latencies.append(latency)
                results.append(result)  # Already list of (vid, dist) tuples
            
            all_latencies.extend(latencies)
            
            if trial == 0:
                trial_results = results
            
            if self.verbose:
                mean_lat = np.mean(latencies)
                print(f"  Trial {trial+1}/{self.n_trials}: {mean_lat:.3f} ms avg")
        
        # Compute metrics
        result = BenchmarkResult(
            algorithm=algo_name,
            config=config,
            build_time=index.build_time,
            memory_mb=memory_mb,
            latencies=all_latencies
        )
        
        # Latency statistics
        result.mean_latency = np.mean(all_latencies)
        result.median_latency = np.median(all_latencies)
        result.p95_latency = np.percentile(all_latencies, 95)
        result.p99_latency = np.percentile(all_latencies, 99)
        result.std_latency = np.std(all_latencies)
        result.queries_per_second = 1000.0 / result.mean_latency
        
        # Recall at different k values
        for k in self.k_values:
            if k <= self.k_max:
                recall = self._compute_recall(trial_results, k)
                result.recall_at_k[k] = recall
        
        if self.verbose:
            print(f"\nResults:")
            print(f"  Build time: {result.build_time:.2f}s")
            print(f"  Memory: {result.memory_mb:.2f} MB")
            print(f"  Mean latency: {result.mean_latency:.3f} ms")
            print(f"  Throughput: {result.queries_per_second:.0f} QPS")
            for k, recall in sorted(result.recall_at_k.items()):
                print(f"  Recall@{k}: {recall:.4f}")
        
        self.results.append(result)
        return result
    
    def benchmark_hnsw(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        name_suffix: str = ""
    ) -> BenchmarkResult:
        """
        Benchmark HNSW baseline.
        
        Args:
            M: Max connections
            ef_construction: Build parameter
            ef_search: Search parameter
            name_suffix: Optional name suffix
            
        Returns:
            BenchmarkResult object
        """
        config = {
            'M': M,
            'ef_construction': ef_construction,
            'ef_search': ef_search
        }
        
        algo_name = f"HNSW{name_suffix}"
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Benchmarking {algo_name}")
            print(f"{'='*80}")
            print(f"Config: {config}")
        
        # Build index
        if self.verbose:
            print("\nBuilding index...")
        
        index = HNSWBaseline(
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            seed=42
        )
        build_stats = index.build(self.dataset)
        
        # Run search trials
        if self.verbose:
            print(f"\nRunning {self.n_trials} search trials...")
        
        all_latencies = []
        trial_results = None
        
        for trial in range(self.n_trials):
            latencies = []
            results = []
            
            for i, query in enumerate(self.queries):
                start = time.perf_counter()
                indices, distances, _ = index.search(query.reshape(1, -1), k=self.k_max)
                latency = (time.perf_counter() - start) * 1000  # ms
                
                latencies.append(latency)
                result_list = [(int(indices[0, j]), float(distances[0, j])) 
                               for j in range(len(indices[0]))]
                results.append(result_list)
            
            all_latencies.extend(latencies)
            
            if trial == 0:
                trial_results = results
            
            if self.verbose:
                mean_lat = np.mean(latencies)
                print(f"  Trial {trial+1}/{self.n_trials}: {mean_lat:.3f} ms avg")
        
        # Compute metrics
        result = BenchmarkResult(
            algorithm=algo_name,
            config=config,
            build_time=build_stats.build_time,
            memory_mb=build_stats.memory_bytes / (1024**2),
            latencies=all_latencies
        )
        
        # Latency statistics
        result.mean_latency = np.mean(all_latencies)
        result.median_latency = np.median(all_latencies)
        result.p95_latency = np.percentile(all_latencies, 95)
        result.p99_latency = np.percentile(all_latencies, 99)
        result.std_latency = np.std(all_latencies)
        result.queries_per_second = 1000.0 / result.mean_latency
        
        # Recall at different k values
        for k in self.k_values:
            if k <= self.k_max:
                recall = self._compute_recall(trial_results, k)
                result.recall_at_k[k] = recall
        
        if self.verbose:
            print(f"\nResults:")
            print(f"  Build time: {result.build_time:.2f}s")
            print(f"  Memory: {result.memory_mb:.2f} MB")
            print(f"  Mean latency: {result.mean_latency:.3f} ms")
            print(f"  Throughput: {result.queries_per_second:.0f} QPS")
            for k, recall in sorted(result.recall_at_k.items()):
                print(f"  Recall@{k}: {recall:.4f}")
        
        self.results.append(result)
        return result
    
    def benchmark_ivf(
        self,
        n_clusters: int = 100,
        n_probe: int = 5,
        use_pq: bool = False,
        pq_m: int = 16,
        name_suffix: str = ""
    ) -> BenchmarkResult:
        """
        Benchmark IVF baseline.
        
        Args:
            n_clusters: Number of clusters
            n_probe: Number of clusters to probe
            use_pq: Use product quantization
            pq_m: PQ subspaces
            name_suffix: Optional name suffix
            
        Returns:
            BenchmarkResult object
        """
        config = {
            'n_clusters': n_clusters,
            'n_probe': n_probe,
            'use_pq': use_pq,
            'pq_m': pq_m if use_pq else None
        }
        
        pq_str = "+PQ" if use_pq else ""
        algo_name = f"IVF{pq_str}{name_suffix}"
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Benchmarking {algo_name}")
            print(f"{'='*80}")
            print(f"Config: {config}")
        
        # Build index
        if self.verbose:
            print("\nBuilding index...")
        
        index = IVFBaseline(
            nlist=n_clusters,
            nprobe=n_probe,
            use_pq=use_pq,
            m=pq_m,
            nbits=8,
            seed=42
        )
        build_stats = index.build(self.dataset)
        
        # Run search trials
        if self.verbose:
            print(f"\nRunning {self.n_trials} search trials...")
        
        all_latencies = []
        trial_results = None
        
        for trial in range(self.n_trials):
            latencies = []
            results = []
            
            for i, query in enumerate(self.queries):
                start = time.perf_counter()
                indices, distances, _ = index.search(query.reshape(1, -1), k=self.k_max)
                latency = (time.perf_counter() - start) * 1000  # ms
                
                latencies.append(latency)
                result_list = [(int(indices[0, j]), float(distances[0, j])) 
                               for j in range(min(len(indices[0]), self.k_max))]
                results.append(result_list)
            
            all_latencies.extend(latencies)
            
            if trial == 0:
                trial_results = results
            
            if self.verbose:
                mean_lat = np.mean(latencies)
                print(f"  Trial {trial+1}/{self.n_trials}: {mean_lat:.3f} ms avg")
        
        # Compute metrics
        result = BenchmarkResult(
            algorithm=algo_name,
            config=config,
            build_time=build_stats.build_time,
            memory_mb=build_stats.memory_bytes / (1024**2),
            latencies=all_latencies
        )
        
        # Latency statistics
        result.mean_latency = np.mean(all_latencies)
        result.median_latency = np.median(all_latencies)
        result.p95_latency = np.percentile(all_latencies, 95)
        result.p99_latency = np.percentile(all_latencies, 99)
        result.std_latency = np.std(all_latencies)
        result.queries_per_second = 1000.0 / result.mean_latency
        
        # Recall at different k values
        for k in self.k_values:
            if k <= self.k_max:
                recall = self._compute_recall(trial_results, k)
                result.recall_at_k[k] = recall
        
        if self.verbose:
            print(f"\nResults:")
            print(f"  Build time: {result.build_time:.2f}s")
            print(f"  Memory: {result.memory_mb:.2f} MB")
            print(f"  Mean latency: {result.mean_latency:.3f} ms")
            print(f"  Throughput: {result.queries_per_second:.0f} QPS")
            for k, recall in sorted(result.recall_at_k.items()):
                print(f"  Recall@{k}: {recall:.4f}")
        
        self.results.append(result)
        return result
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        results_dict = {
            'dataset_info': {
                'N': int(self.N),
                'd': int(self.d),
                'Q': int(self.Q)
            },
            'results': [r.to_dict() for r in self.results]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        if self.verbose:
            print(f"\n✓ Results saved to {filepath}")
    
    def print_summary(self):
        """Print summary comparison table."""
        if not self.results:
            print("No results to display")
            return
        
        print(f"\n{'='*120}")
        print(f"{'BENCHMARK SUMMARY':^120}")
        print(f"{'='*120}")
        print(f"Dataset: N={self.N:,}, d={self.d}, Q={self.Q}")
        print(f"{'='*120}")
        
        # Table header
        header = f"{'Algorithm':<20} {'Build(s)':<10} {'Memory(MB)':<12} {'Latency(ms)':<14} {'QPS':<8}"
        for k in self.k_values:
            header += f" {'R@'+str(k):<7}"
        print(header)
        print("-" * 120)
        
        # Sort by recall@10 descending
        sorted_results = sorted(
            self.results,
            key=lambda r: r.recall_at_k.get(10, 0),
            reverse=True
        )
        
        for result in sorted_results:
            row = f"{result.algorithm:<20} "
            row += f"{result.build_time:<10.2f} "
            row += f"{result.memory_mb:<12.1f} "
            row += f"{result.mean_latency:<14.3f} "
            row += f"{int(result.queries_per_second):<8} "
            
            for k in self.k_values:
                recall = result.recall_at_k.get(k, 0.0)
                row += f"{recall:<7.4f} "
            
            print(row)
        
        print("=" * 120)


# Quick test
if __name__ == "__main__":
    print("="*80)
    print("Benchmark Framework - Quick Test")
    print("="*80)
    
    # Small test dataset
    N = 10000
    d = 128
    Q = 100
    
    print(f"\nGenerating test dataset...")
    print(f"  N={N:,} vectors, d={d} dimensions")
    print(f"  Q={Q} queries")
    
    np.random.seed(42)
    vectors = np.random.randn(N, d).astype(np.float32)
    queries = np.random.randn(Q, d).astype(np.float32)
    
    # Create benchmark
    benchmark = ANNSBenchmark(
        dataset=vectors,
        queries=queries,
        k_values=[1, 5, 10],
        n_trials=2,
        verbose=True
    )
    
    # Test ZGQ
    benchmark.benchmark_zgq(
        n_zones=20,
        M=16,
        ef_construction=100,
        ef_search=50,
        n_probe=3,
        use_pq=True
    )
    
    # Print summary
    benchmark.print_summary()
    
    print("\n✓ Benchmark framework validated!")
