"""
ZGQ v8 Benchmark Suite
=======================

Comprehensive benchmarking tools for comparing ZGQ against
HNSW and other ANNS algorithms.

Usage:
    python -m benchmarks.run_benchmarks --dataset 10k
    python -m benchmarks.compare_algorithms
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import hnswlib

from zgq import ZGQIndex
from zgq.index import ZGQConfig
from zgq.search import compute_ground_truth, compute_recall


def generate_test_data(
    n_vectors: int = 10000,
    n_queries: int = 100,
    dimension: int = 128,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic test data.
    
    Creates normally distributed vectors suitable for ANNS evaluation.
    
    Args:
        n_vectors: Number of database vectors
        n_queries: Number of query vectors
        dimension: Vector dimension
        random_state: Random seed
        
    Returns:
        (vectors, queries): Generated data
    """
    rng = np.random.RandomState(random_state)
    
    # Generate clustered data for more realistic evaluation
    n_clusters = max(10, n_vectors // 1000)
    
    # Cluster centers
    centers = rng.randn(n_clusters, dimension).astype(np.float32)
    
    # Assign vectors to clusters
    cluster_assignments = rng.randint(0, n_clusters, n_vectors)
    
    # Generate vectors around centers
    vectors = np.zeros((n_vectors, dimension), dtype=np.float32)
    for i in range(n_vectors):
        center = centers[cluster_assignments[i]]
        noise = rng.randn(dimension).astype(np.float32) * 0.3
        vectors[i] = center + noise
    
    # Generate queries (some from clusters, some random)
    queries = np.zeros((n_queries, dimension), dtype=np.float32)
    for i in range(n_queries):
        if rng.random() < 0.7:  # 70% from clusters
            center = centers[rng.randint(0, n_clusters)]
            noise = rng.randn(dimension).astype(np.float32) * 0.3
            queries[i] = center + noise
        else:  # 30% random
            queries[i] = rng.randn(dimension).astype(np.float32)
    
    return vectors, queries


def save_test_data(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    output_dir: Path,
    size_suffix: str
) -> None:
    """Save test data to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / f'vectors_{size_suffix}.npy', vectors)
    np.save(output_dir / 'queries_100.npy', queries)
    np.save(output_dir / f'ground_truth_{size_suffix}.npy', ground_truth)
    
    print(f"Saved test data to {output_dir}")


def load_test_data(
    data_dir: Path,
    size_suffix: str = '10k'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load test data from files."""
    vectors = np.load(data_dir / f'vectors_{size_suffix}.npy')
    queries = np.load(data_dir / 'queries_100.npy')
    ground_truth = np.load(data_dir / f'ground_truth_{size_suffix}.npy')
    
    return vectors, queries, ground_truth


class HNSWBaseline:
    """HNSW baseline using hnswlib for fair comparison."""
    
    def __init__(
        self,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 50,
        space: str = 'l2'
    ):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.space = space
        
        self.index = None
        self.n_vectors = 0
        self.dimension = 0
    
    def build(self, vectors: np.ndarray) -> float:
        """Build HNSW index, return build time."""
        self.n_vectors, self.dimension = vectors.shape
        
        self.index = hnswlib.Index(space=self.space, dim=self.dimension)
        self.index.init_index(
            max_elements=self.n_vectors,
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=42
        )
        
        start = time.time()
        self.index.add_items(vectors, np.arange(self.n_vectors))
        build_time = time.time() - start
        
        self.index.set_ef(self.ef_search)
        
        return build_time
    
    def search(
        self,
        queries: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        queries = queries.astype(np.float32)
        
        if len(queries.shape) == 1:
            queries = queries.reshape(1, -1)
        
        ids, distances = self.index.knn_query(queries, k=k)
        
        return ids, distances


def benchmark_algorithm(
    name: str,
    build_fn,
    search_fn,
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int = 10,
    n_warmup: int = 3,
    n_runs: int = 5
) -> Dict:
    """
    Benchmark a single algorithm.
    
    Args:
        name: Algorithm name
        build_fn: Function to build index, returns build_time
        search_fn: Function to search, takes (queries, k)
        vectors: Database vectors
        queries: Query vectors
        ground_truth: Ground truth neighbors
        k: Number of neighbors
        n_warmup: Warmup iterations
        n_runs: Benchmark iterations
        
    Returns:
        Benchmark results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    
    # Build
    print("\nBuilding index...")
    build_time = build_fn(vectors)
    print(f"  Build time: {build_time:.3f}s")
    
    # Warmup
    print("\nWarming up...")
    for _ in range(n_warmup):
        _ = search_fn(queries[:10], k)
    
    # Benchmark search
    print("\nBenchmarking search...")
    latencies = []
    
    for run in range(n_runs):
        start = time.time()
        predicted, _ = search_fn(queries, k)
        elapsed = time.time() - start
        latencies.append(elapsed)
    
    # Compute metrics
    avg_total = np.mean(latencies)
    avg_per_query = avg_total / len(queries) * 1000  # ms
    throughput = len(queries) / avg_total
    
    # Final search for recall
    predicted, distances = search_fn(queries, k)
    recall = compute_recall(predicted, ground_truth, k)
    
    results = {
        'name': name,
        'build_time_s': build_time,
        'latency_per_query_ms': avg_per_query,
        'throughput_qps': throughput,
        f'recall@{k}': recall
    }
    
    print(f"\nResults:")
    print(f"  Build time:    {build_time:.3f}s")
    print(f"  Query latency: {avg_per_query:.4f}ms")
    print(f"  Throughput:    {throughput:,.0f} QPS")
    print(f"  Recall@{k}:     {recall:.1f}%")
    
    return results


def run_benchmarks(
    dataset_size: str = '10k',
    data_dir: Optional[Path] = None
) -> Dict:
    """
    Run full benchmark suite.
    
    Args:
        dataset_size: Dataset size ('10k', '100k', '1m')
        data_dir: Data directory (default: benchmarks/data)
        
    Returns:
        All benchmark results
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / 'data'
    
    print("="*60)
    print("ZGQ v8 BENCHMARK SUITE")
    print("="*60)
    
    # Load or generate data
    try:
        vectors, queries, ground_truth = load_test_data(data_dir, dataset_size)
        print(f"\nLoaded existing test data ({dataset_size})")
    except FileNotFoundError:
        print(f"\nGenerating test data ({dataset_size})...")
        
        size_map = {
            '10k': 10000,
            '100k': 100000,
            '1m': 1000000
        }
        n_vectors = size_map.get(dataset_size, 10000)
        
        vectors, queries = generate_test_data(n_vectors=n_vectors)
        
        print("Computing ground truth...")
        ground_truth, _ = compute_ground_truth(vectors, queries, k=100)
        
        save_test_data(vectors, queries, ground_truth, data_dir, dataset_size)
    
    print(f"\nDataset: {vectors.shape[0]:,} vectors × {vectors.shape[1]} dimensions")
    print(f"Queries: {queries.shape[0]}")
    
    k = 10
    all_results = []
    
    # Benchmark HNSW (baseline)
    print("\n" + "="*60)
    print("1/2: HNSW Baseline")
    print("="*60)
    
    hnsw = HNSWBaseline(M=16, ef_construction=200, ef_search=50)
    hnsw_results = benchmark_algorithm(
        "HNSW",
        hnsw.build,
        hnsw.search,
        vectors, queries, ground_truth, k
    )
    all_results.append(hnsw_results)
    
    # Benchmark ZGQ v8
    print("\n" + "="*60)
    print("2/2: ZGQ v8")
    print("="*60)
    
    zgq_config = ZGQConfig(
        n_zones='auto',
        use_hierarchy=True,
        M=16,
        ef_construction=200,
        ef_search=64,
        use_pq=True,
        pq_m='auto',
        use_residual_pq=True,
        verbose=True
    )
    zgq_index = ZGQIndex(zgq_config)
    
    def zgq_build(vecs):
        start = time.time()
        zgq_index.build(vecs)
        return time.time() - start
    
    def zgq_search(qs, k):
        return zgq_index.batch_search(qs, k, n_probe=8)
    
    zgq_results = benchmark_algorithm(
        "ZGQ v8",
        zgq_build,
        zgq_search,
        vectors, queries, ground_truth, k
    )
    all_results.append(zgq_results)
    
    # Comparison summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    
    hnsw_lat = hnsw_results['latency_per_query_ms']
    zgq_lat = zgq_results['latency_per_query_ms']
    speedup = hnsw_lat / zgq_lat
    
    recall_diff = zgq_results[f'recall@{k}'] - hnsw_results[f'recall@{k}']
    
    print(f"\n{'Metric':<20} {'HNSW':<15} {'ZGQ v8':<15} {'Result':<15}")
    print("-" * 65)
    print(f"{'Latency (ms)':<20} {hnsw_lat:<15.4f} {zgq_lat:<15.4f} {speedup:.2f}x {'✓' if speedup > 1 else '✗'}")
    print(f"{'Recall@10':<20} {hnsw_results[f'recall@{k}']:<15.1f} {zgq_results[f'recall@{k}']:<15.1f} {'+' if recall_diff >= 0 else ''}{recall_diff:.1f}%")
    print(f"{'Build time (s)':<20} {hnsw_results['build_time_s']:<15.3f} {zgq_results['build_time_s']:<15.3f}")
    print(f"{'Throughput (QPS)':<20} {hnsw_results['throughput_qps']:<15,.0f} {zgq_results['throughput_qps']:<15,.0f}")
    
    # Winner determination
    print("\n" + "-"*65)
    if speedup > 1.0 and recall_diff >= -1.0:
        print(f"🏆 WINNER: ZGQ v8 ({speedup:.2f}x faster with comparable recall)")
    elif speedup > 1.0:
        print(f"⚠️ ZGQ v8 is {speedup:.2f}x faster but recall is {abs(recall_diff):.1f}% lower")
    else:
        print(f"❌ HNSW is faster ({1/speedup:.2f}x)")
    
    # Save results
    output_file = Path(__file__).parent / f'benchmark_results_{dataset_size}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to {output_file}")
    
    return {
        'results': all_results,
        'comparison': {
            'speedup': speedup,
            'recall_diff': recall_diff,
            'winner': 'ZGQ' if speedup > 1.0 and recall_diff >= -1.0 else 'HNSW'
        }
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run ZGQ v8 benchmarks')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='10k',
        choices=['10k', '100k', '1m'],
        help='Dataset size'
    )
    parser.add_argument(
        '--generate', '-g',
        action='store_true',
        help='Force regenerate test data'
    )
    
    args = parser.parse_args()
    
    if args.generate:
        data_dir = Path(__file__).parent / 'data'
        size_map = {'10k': 10000, '100k': 100000, '1m': 1000000}
        n_vectors = size_map[args.dataset]
        
        print(f"Generating {args.dataset} test data...")
        vectors, queries = generate_test_data(n_vectors=n_vectors)
        
        print("Computing ground truth (this may take a while)...")
        ground_truth, _ = compute_ground_truth(vectors, queries, k=100)
        
        save_test_data(vectors, queries, ground_truth, data_dir, args.dataset)
        print("Done!")
    else:
        run_benchmarks(args.dataset)
