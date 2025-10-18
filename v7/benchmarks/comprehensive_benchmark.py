"""
Comprehensive benchmark comparing ZGQ against baseline ANNS methods.

This benchmark validates the research hypothesis that ZGQ achieves superior
recall-latency trade-offs compared to state-of-the-art methods.
"""

import numpy as np
import time
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from index import ZGQIndex
from search import compute_ground_truth
from visualization import BenchmarkVisualizer

# Import baseline libraries
import hnswlib
import faiss


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    method: str
    dataset_name: str
    n_vectors: int
    dimension: int
    k: int
    
    # Performance metrics
    recall_at_k: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    
    # Build metrics
    build_time_sec: float
    index_size_mb: float
    
    # Parameters
    parameters: Dict


class ANNSBenchmark:
    """Comprehensive ANNS benchmark suite."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []
    
    def generate_dataset(
        self,
        n_vectors: int = 10000,
        dimension: int = 128,
        n_queries: int = 100,
        distribution: str = 'gaussian'
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Generate synthetic dataset for benchmarking.
        
        Args:
            n_vectors: Number of database vectors
            dimension: Vector dimension
            n_queries: Number of query vectors
            distribution: Data distribution ('gaussian', 'uniform', 'clustered')
            
        Returns:
            (vectors, queries, dataset_name)
        """
        if self.verbose:
            print(f"\nGenerating {distribution} dataset:")
            print(f"  Vectors: {n_vectors}, Dimension: {dimension}, Queries: {n_queries}")
        
        np.random.seed(42)
        
        if distribution == 'gaussian':
            vectors = np.random.randn(n_vectors, dimension).astype('float32')
            queries = np.random.randn(n_queries, dimension).astype('float32')
            dataset_name = f"gaussian_{n_vectors}x{dimension}"
        
        elif distribution == 'uniform':
            vectors = np.random.rand(n_vectors, dimension).astype('float32')
            queries = np.random.rand(n_queries, dimension).astype('float32')
            dataset_name = f"uniform_{n_vectors}x{dimension}"
        
        elif distribution == 'clustered':
            # Create clustered data
            n_clusters = int(np.sqrt(n_vectors))
            cluster_centers = np.random.randn(n_clusters, dimension).astype('float32') * 5
            
            vectors = []
            for _ in range(n_vectors):
                cluster_idx = np.random.randint(n_clusters)
                vector = cluster_centers[cluster_idx] + np.random.randn(dimension).astype('float32') * 0.5
                vectors.append(vector)
            vectors = np.array(vectors)
            
            queries = np.random.randn(n_queries, dimension).astype('float32')
            dataset_name = f"clustered_{n_vectors}x{dimension}"
        
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        return vectors, queries, dataset_name
    
    def benchmark_zgq(
        self,
        vectors: np.ndarray,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        k: int = 10,
        dataset_name: str = "unknown",
        n_probe_values: List[int] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark ZGQ with different parameter settings.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Benchmarking ZGQ")
            print("="*60)
        
        if n_probe_values is None:
            n_zones = int(len(vectors) ** 0.5)
            n_probe_values = [max(1, n_zones // 20), max(1, n_zones // 10), 
                             max(1, n_zones // 5), max(1, n_zones // 2)]
        
        results = []
        
        # Build ZGQ index
        if self.verbose:
            print("\nBuilding ZGQ index...")
        
        build_start = time.time()
        
        index = ZGQIndex(
            n_zones=int(len(vectors) ** 0.5),
            hnsw_M=16,
            hnsw_ef_construction=200,
            use_pq=True,
            pq_m=16,
            pq_nbits=8,
            verbose=self.verbose
        )
        index.build(vectors)
        
        build_time = time.time() - build_start
        
        # Estimate index size
        index_size_mb = self._estimate_index_size(index)
        
        # Test different n_probe values
        for n_probe in n_probe_values:
            if self.verbose:
                print(f"\nTesting n_probe={n_probe}")
            
            # Benchmark search
            latencies, recall = self._benchmark_search(
                lambda q: index.search(q, k=k, n_probe=n_probe),
                queries, ground_truth, k
            )
            
            result = BenchmarkResult(
                method="ZGQ",
                dataset_name=dataset_name,
                n_vectors=len(vectors),
                dimension=vectors.shape[1],
                k=k,
                recall_at_k=recall,
                mean_latency_ms=np.mean(latencies),
                median_latency_ms=np.median(latencies),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                throughput_qps=1000.0 / np.mean(latencies),
                build_time_sec=build_time,
                index_size_mb=index_size_mb,
                parameters={'n_probe': n_probe, 'n_zones': index.n_zones}
            )
            
            results.append(result)
            self.results.append(result)
            
            if self.verbose:
                self._print_result(result)
        
        return results
    
    def benchmark_hnsw(
        self,
        vectors: np.ndarray,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        k: int = 10,
        dataset_name: str = "unknown",
        ef_values: List[int] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark HNSW baseline.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Benchmarking HNSW")
            print("="*60)
        
        if ef_values is None:
            ef_values = [10, 20, 50, 100, 200]
        
        results = []
        
        # Build HNSW index
        if self.verbose:
            print("\nBuilding HNSW index...")
        
        build_start = time.time()
        
        d = vectors.shape[1]
        index = hnswlib.Index(space='l2', dim=d)
        index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
        index.add_items(vectors, np.arange(len(vectors)))
        
        build_time = time.time() - build_start
        
        # Estimate index size (rough estimate)
        index_size_mb = (len(vectors) * d * 4 + len(vectors) * 16 * 4) / (1024 ** 2)
        
        # Test different ef values
        for ef in ef_values:
            if self.verbose:
                print(f"\nTesting ef={ef}")
            
            index.set_ef(ef)
            
            # Benchmark search
            latencies, recall = self._benchmark_search(
                lambda q: index.knn_query(q.reshape(1, -1), k=k),
                queries, ground_truth, k
            )
            
            result = BenchmarkResult(
                method="HNSW",
                dataset_name=dataset_name,
                n_vectors=len(vectors),
                dimension=vectors.shape[1],
                k=k,
                recall_at_k=recall,
                mean_latency_ms=np.mean(latencies),
                median_latency_ms=np.median(latencies),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                throughput_qps=1000.0 / np.mean(latencies),
                build_time_sec=build_time,
                index_size_mb=index_size_mb,
                parameters={'ef': ef}
            )
            
            results.append(result)
            self.results.append(result)
            
            if self.verbose:
                self._print_result(result)
        
        return results
    
    def benchmark_faiss_ivf(
        self,
        vectors: np.ndarray,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        k: int = 10,
        dataset_name: str = "unknown",
        nprobe_values: List[int] = None
    ) -> List[BenchmarkResult]:
        """
        Benchmark FAISS IVF-PQ baseline.
        """
        if self.verbose:
            print("\n" + "="*60)
            print("Benchmarking FAISS IVF-PQ")
            print("="*60)
        
        if nprobe_values is None:
            nlist = int(len(vectors) ** 0.5)
            nprobe_values = [max(1, nlist // 20), max(1, nlist // 10),
                            max(1, nlist // 5), max(1, nlist // 2)]
        
        results = []
        
        # Build FAISS index
        if self.verbose:
            print("\nBuilding FAISS IVF-PQ index...")
        
        build_start = time.time()
        
        d = vectors.shape[1]
        nlist = int(len(vectors) ** 0.5)
        m = 16  # Number of subquantizers
        bits = 8
        
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
        
        index.train(vectors)
        index.add(vectors)
        
        build_time = time.time() - build_start
        
        # Estimate index size
        index_size_mb = (nlist * d * 4 + len(vectors) * m * 1) / (1024 ** 2)
        
        # Test different nprobe values
        for nprobe in nprobe_values:
            if self.verbose:
                print(f"\nTesting nprobe={nprobe}")
            
            index.nprobe = nprobe
            
            # Benchmark search
            latencies, recall = self._benchmark_search(
                lambda q: index.search(q.reshape(1, -1), k),
                queries, ground_truth, k
            )
            
            result = BenchmarkResult(
                method="FAISS-IVF-PQ",
                dataset_name=dataset_name,
                n_vectors=len(vectors),
                dimension=vectors.shape[1],
                k=k,
                recall_at_k=recall,
                mean_latency_ms=np.mean(latencies),
                median_latency_ms=np.median(latencies),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                throughput_qps=1000.0 / np.mean(latencies),
                build_time_sec=build_time,
                index_size_mb=index_size_mb,
                parameters={'nprobe': nprobe, 'nlist': nlist}
            )
            
            results.append(result)
            self.results.append(result)
            
            if self.verbose:
                self._print_result(result)
        
        return results
    
    def _benchmark_search(
        self,
        search_fn,
        queries: np.ndarray,
        ground_truth: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, float]:
        """
        Benchmark search function and compute recall.
        
        Returns:
            (latencies, recall)
        """
        latencies = []
        all_results = []
        
        # Warm up
        for _ in range(3):
            search_fn(queries[0])
        
        # Benchmark
        for query in queries:
            start = time.perf_counter()
            result = search_fn(query)
            latency = (time.perf_counter() - start) * 1000  # Convert to ms
            
            latencies.append(latency)
            
            # Extract IDs from result (handle different return formats)
            if isinstance(result, tuple):
                ids = result[0]
                if len(ids.shape) > 1:
                    ids = ids[0]
            else:
                ids = result
            
            all_results.append(ids[:k])
        
        latencies = np.array(latencies)
        
        # Compute recall
        recall = self._compute_recall(all_results, ground_truth, k)
        
        return latencies, recall
    
    def _compute_recall(
        self,
        results: List[np.ndarray],
        ground_truth: np.ndarray,
        k: int
    ) -> float:
        """Compute recall@k."""
        recalls = []
        
        for result, gt in zip(results, ground_truth):
            # Ensure both are sets for intersection
            result_set = set(result[:k])
            gt_set = set(gt[:k])
            
            recall = len(result_set.intersection(gt_set)) / k
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def _estimate_index_size(self, index: ZGQIndex) -> float:
        """Estimate ZGQ index size in MB."""
        info = index.get_index_info()
        
        # Vectors
        vectors_mb = info['n_vectors'] * info['dimension'] * 4 / (1024 ** 2)
        
        # HNSW graphs (rough estimate)
        hnsw_mb = info['n_vectors'] * info['hnsw_M'] * 4 / (1024 ** 2)
        
        # PQ
        pq_mb = info.get('pq_memory_mb', 0)
        
        return vectors_mb + hnsw_mb + pq_mb
    
    def _print_result(self, result: BenchmarkResult):
        """Print benchmark result."""
        print(f"  Recall@{result.k}: {result.recall_at_k:.4f}")
        print(f"  Latency: {result.mean_latency_ms:.3f}ms (mean), "
              f"{result.median_latency_ms:.3f}ms (median), "
              f"{result.p95_latency_ms:.3f}ms (p95)")
        print(f"  Throughput: {result.throughput_qps:.1f} QPS")
    
    def save_results(self, filepath: str):
        """Save all benchmark results to JSON file."""
        results_dict = [asdict(r) for r in self.results]
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        if self.verbose:
            print(f"\nResults saved to {filepath}")
    
    def generate_comparison_table(self) -> str:
        """Generate markdown comparison table."""
        if not self.results:
            return "No results to compare"
        
        # Group by dataset and k
        grouped = {}
        for r in self.results:
            key = (r.dataset_name, r.k)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        
        tables = []
        
        for (dataset, k), results in grouped.items():
            table = f"\n### {dataset} (k={k})\n\n"
            table += "| Method | Recall@k | Mean Latency (ms) | Throughput (QPS) | Build Time (s) |\n"
            table += "|--------|----------|-------------------|------------------|----------------|\n"
            
            for r in sorted(results, key=lambda x: (-x.recall_at_k, x.mean_latency_ms)):
                table += f"| {r.method} | {r.recall_at_k:.4f} | "
                table += f"{r.mean_latency_ms:.3f} | {r.throughput_qps:.1f} | "
                table += f"{r.build_time_sec:.2f} |\n"
            
            tables.append(table)
        
        return "\n".join(tables)


def main():
    """Run comprehensive benchmark."""
    benchmark = ANNSBenchmark(verbose=True)
    
    # Generate dataset
    vectors, queries, dataset_name = benchmark.generate_dataset(
        n_vectors=10000,
        dimension=128,
        n_queries=100,
        distribution='gaussian'
    )
    
    # Compute ground truth
    print("\nComputing ground truth...")
    ground_truth, _ = compute_ground_truth(vectors, queries, k=10)
    
    # Run benchmarks
    benchmark.benchmark_zgq(vectors, queries, ground_truth, k=10, dataset_name=dataset_name)
    benchmark.benchmark_hnsw(vectors, queries, ground_truth, k=10, dataset_name=dataset_name)
    benchmark.benchmark_faiss_ivf(vectors, queries, ground_truth, k=10, dataset_name=dataset_name)
    
    # Generate comparison
    print("\n" + "="*60)
    print("BENCHMARK COMPARISON")
    print("="*60)
    print(benchmark.generate_comparison_table())
    
    # Save results
    results_file = 'benchmark_results.json'
    benchmark.save_results(results_file)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    try:
        visualizer = BenchmarkVisualizer(results_file, output_dir='figures')
        visualizer.generate_all_figures()
        report = visualizer.generate_summary_report()
        print("\n" + report)
        
        print("\n✓ Visualizations generated successfully!")
        print(f"  Check the 'figures/' directory for plots")
    except Exception as e:
        print(f"\n✗ Error generating visualizations: {e}")
        print("  You can generate them manually by running:")
        print(f"  python src/visualization.py {results_file}")


if __name__ == '__main__':
    main()
