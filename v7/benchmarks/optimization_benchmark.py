"""
Benchmark comparing optimized ZGQ against baseline ZGQ and HNSW.

This will show the performance improvements from optimizations.
"""

import numpy as np
import time
import sys
import os
from typing import Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from index import ZGQIndex
from index_optimized import ZGQIndexOptimized
from search import compute_ground_truth
import hnswlib


def generate_dataset(
    n_vectors: int = 10000,
    dimension: int = 128,
    n_queries: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian dataset."""
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    return vectors, queries


def benchmark_zgq_baseline(vectors, queries, ground_truth, k=10, n_probe=20):
    """Benchmark baseline ZGQ."""
    print(f"\n{'='*60}")
    print("BASELINE ZGQ")
    print(f"{'='*60}")
    
    # Build
    index = ZGQIndex(
        n_zones=100,
        hnsw_M=16,
        hnsw_ef_construction=200,
        hnsw_ef_search=50,
        use_pq=True,
        verbose=False
    )
    
    build_start = time.time()
    index.build(vectors)
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.3f}s")
    
    # Search
    latencies = []
    all_ids = []
    
    for query in queries:
        start = time.time()
        ids, _ = index.search(query, k=k, n_probe=n_probe)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        all_ids.append(ids)
    
    # Compute recall
    recalls = []
    for i, ids in enumerate(all_ids):
        recall = np.intersect1d(ids, ground_truth[i]).size / k
        recalls.append(recall)
    
    mean_latency = np.mean(latencies)
    mean_recall = np.mean(recalls)
    throughput = 1000.0 / mean_latency
    
    print(f"Mean latency: {mean_latency:.3f}ms")
    print(f"Recall@{k}: {mean_recall:.4f}")
    print(f"Throughput: {throughput:.1f} QPS")
    
    return {
        'method': 'ZGQ Baseline',
        'build_time': build_time,
        'mean_latency': mean_latency,
        'recall': mean_recall,
        'throughput': throughput
    }


def benchmark_zgq_optimized(vectors, queries, ground_truth, k=10, n_probe=20):
    """Benchmark optimized ZGQ."""
    print(f"\n{'='*60}")
    print("OPTIMIZED ZGQ")
    print(f"{'='*60}")
    
    # Build
    index = ZGQIndexOptimized(
        n_zones=100,
        M=16,
        ef_construction=200,
        use_pq=True,
        n_threads=4,
        verbose=False
    )
    
    build_start = time.time()
    index.build(vectors)
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.3f}s")
    
    # Warmup (JIT compilation) - CRITICAL!
    print("Warming up JIT compiler (50 iterations)...")
    warmup_start = time.time()
    for i in range(50):
        _ = index.search(queries[i % len(queries)], k=k, n_probe=n_probe)
    warmup_time = time.time() - warmup_start
    print(f"Warmup completed in {warmup_time:.2f}s")
    
    # Search
    latencies = []
    all_ids = []
    
    for query in queries:
        start = time.time()
        ids, _ = index.search(query, k=k, n_probe=n_probe)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        all_ids.append(ids)
    
    # Compute recall
    recalls = []
    for i, ids in enumerate(all_ids):
        recall = np.intersect1d(ids, ground_truth[i]).size / k
        recalls.append(recall)
    
    mean_latency = np.mean(latencies)
    mean_recall = np.mean(recalls)
    throughput = 1000.0 / mean_latency
    
    print(f"Mean latency: {mean_latency:.3f}ms")
    print(f"Recall@{k}: {mean_recall:.4f}")
    print(f"Throughput: {throughput:.1f} QPS")
    
    return {
        'method': 'ZGQ Optimized',
        'build_time': build_time,
        'mean_latency': mean_latency,
        'recall': mean_recall,
        'throughput': throughput
    }


def benchmark_hnsw(vectors, queries, ground_truth, k=10, ef=200):
    """Benchmark HNSW baseline."""
    print(f"\n{'='*60}")
    print("HNSW BASELINE")
    print(f"{'='*60}")
    
    # Build
    dim = vectors.shape[1]
    index = hnswlib.Index(space='l2', dim=dim)
    index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
    
    build_start = time.time()
    index.add_items(vectors, np.arange(len(vectors)))
    build_time = time.time() - build_start
    print(f"Build time: {build_time:.3f}s")
    
    # Search
    index.set_ef(ef)
    latencies = []
    all_ids = []
    
    for query in queries:
        start = time.time()
        ids, _ = index.knn_query(query, k=k)
        latency = (time.time() - start) * 1000  # ms
        latencies.append(latency)
        all_ids.append(ids[0])
    
    # Compute recall
    recalls = []
    for i, ids in enumerate(all_ids):
        recall = np.intersect1d(ids, ground_truth[i]).size / k
        recalls.append(recall)
    
    mean_latency = np.mean(latencies)
    mean_recall = np.mean(recalls)
    throughput = 1000.0 / mean_latency
    
    print(f"Mean latency: {mean_latency:.3f}ms")
    print(f"Recall@{k}: {mean_recall:.4f}")
    print(f"Throughput: {throughput:.1f} QPS")
    
    return {
        'method': 'HNSW',
        'build_time': build_time,
        'mean_latency': mean_latency,
        'recall': mean_recall,
        'throughput': throughput
    }


def print_comparison(results):
    """Print comparison table."""
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Method':<20} {'Build Time':<12} {'Latency':<12} {'Recall@10':<12} {'Throughput':<12}")
    print(f"{'-'*70}")
    
    for r in results:
        print(f"{r['method']:<20} {r['build_time']:>8.3f}s    "
              f"{r['mean_latency']:>8.3f}ms   {r['recall']:>8.4f}     "
              f"{r['throughput']:>8.1f} QPS")
    
    print(f"\n{'='*70}")
    print("SPEEDUP ANALYSIS")
    print(f"{'='*70}\n")
    
    baseline = results[0]
    optimized = results[1]
    hnsw = results[2]
    
    speedup_vs_baseline = baseline['mean_latency'] / optimized['mean_latency']
    speedup_vs_hnsw = optimized['mean_latency'] / hnsw['mean_latency']
    
    print(f"Optimized ZGQ vs Baseline ZGQ:")
    print(f"  Speedup: {speedup_vs_baseline:.2f}x faster")
    print(f"  Build time: {optimized['build_time']/baseline['build_time']:.2f}x")
    
    print(f"\nOptimized ZGQ vs HNSW:")
    print(f"  Latency ratio: {speedup_vs_hnsw:.2f}x")
    print(f"  Recall difference: {(optimized['recall']-hnsw['recall'])*100:+.2f}%")
    
    if speedup_vs_hnsw < 1:
        print(f"  ✓ Optimized ZGQ is now FASTER than HNSW!")
    else:
        gap_remaining = speedup_vs_hnsw
        print(f"  ⚠ Still {gap_remaining:.2f}x slower than HNSW")
        print(f"  Additional optimization needed: {(gap_remaining-1)*100:.1f}%")


def main():
    """Run optimization benchmark."""
    print(f"\n{'='*70}")
    print("ZGQ OPTIMIZATION BENCHMARK")
    print(f"{'='*70}")
    
    # Generate dataset
    print("\nGenerating dataset...")
    vectors, queries = generate_dataset(n_vectors=10000, dimension=128, n_queries=100)
    print(f"Vectors: {len(vectors):,}, Queries: {len(queries)}")
    
    # Compute ground truth
    print("\nComputing ground truth...")
    ground_truth, _ = compute_ground_truth(vectors, queries, k=10)
    
    # Run benchmarks
    results = []
    
    # Baseline ZGQ
    results.append(benchmark_zgq_baseline(vectors, queries, ground_truth, k=10, n_probe=20))
    
    # Optimized ZGQ
    results.append(benchmark_zgq_optimized(vectors, queries, ground_truth, k=10, n_probe=20))
    
    # HNSW
    results.append(benchmark_hnsw(vectors, queries, ground_truth, k=10, ef=200))
    
    # Print comparison
    print_comparison(results)
    
    print(f"\n{'='*70}")
    print("Optimization benchmark complete!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
