"""
Comprehensive n_probe tuning benchmark for ZGQ optimization.

Tests different n_probe values to find optimal speed/recall tradeoff.
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from index import ZGQIndex
from index_optimized import ZGQIndexOptimized
import hnswlib


def compute_recall(results, ground_truth, k=10):
    """Compute recall@k."""
    recalls = []
    for i, gt in enumerate(ground_truth):
        intersection = len(set(results[i][:k]) & set(gt[:k]))
        recalls.append(intersection / k)
    return np.mean(recalls)


def main():
    # Generate dataset
    np.random.seed(42)
    n_vectors = 10000
    n_queries = 100
    dim = 128
    k = 10
    
    print("=" * 70)
    print("ZGQ N_PROBE TUNING BENCHMARK")
    print("=" * 70)
    print()
    print(f"Dataset: {n_vectors:,} vectors, {n_queries} queries, dim={dim}")
    print()
    
    vectors = np.random.rand(n_vectors, dim).astype(np.float32)
    queries = np.random.rand(n_queries, dim).astype(np.float32)
    
    # Ground truth
    print("Computing ground truth...")
    index_gt = hnswlib.Index(space='l2', dim=dim)
    index_gt.init_index(max_elements=n_vectors, M=64, ef_construction=200)
    index_gt.add_items(vectors, np.arange(n_vectors))
    index_gt.set_ef(200)
    
    ground_truth = []
    for query in queries:
        ids, _ = index_gt.knn_query(query, k=k)
        ground_truth.append(ids[0])
    
    print()
    
    # HNSW baseline
    print("=" * 70)
    print("HNSW BASELINE")
    print("=" * 70)
    
    index_hnsw = hnswlib.Index(space='l2', dim=dim)
    index_hnsw.init_index(max_elements=n_vectors, M=16, ef_construction=200)
    build_start = time.time()
    index_hnsw.add_items(vectors, np.arange(n_vectors))
    build_time_hnsw = time.time() - build_start
    index_hnsw.set_ef(50)
    
    latencies_hnsw = []
    results_hnsw = []
    for query in queries:
        start = time.time()
        ids, _ = index_hnsw.knn_query(query, k=k)
        latency = (time.time() - start) * 1000
        latencies_hnsw.append(latency)
        results_hnsw.append(ids[0])
    
    recall_hnsw = compute_recall(results_hnsw, ground_truth, k=k)
    mean_latency_hnsw = np.mean(latencies_hnsw)
    throughput_hnsw = 1000 / mean_latency_hnsw
    
    print(f"Build time: {build_time_hnsw:.3f}s")
    print(f"Mean latency: {mean_latency_hnsw:.3f}ms")
    print(f"Recall@{k}: {recall_hnsw:.4f}")
    print(f"Throughput: {throughput_hnsw:.1f} QPS")
    print()
    
    # ZGQ Optimized with different n_probe values
    print("=" * 70)
    print("ZGQ OPTIMIZED (ULTRA-FAST) - N_PROBE TUNING")
    print("=" * 70)
    print()
    
    # Build index once
    index_zgq = ZGQIndexOptimized(
        n_zones=100,
        M=16,
        ef_construction=200,
        n_subquantizers=8,
        n_bits=8,
        use_pq=True,
        n_threads=4,
        verbose=False
    )
    
    build_start = time.time()
    index_zgq.build(vectors)
    build_time_zgq = time.time() - build_start
    print(f"Build time: {build_time_zgq:.3f}s")
    print()
    
    # Test different n_probe values
    n_probe_values = [3, 5, 8, 10, 15, 20, 30, 40]
    
    print(f"{'n_probe':<10}{'Latency':<15}{'Throughput':<15}{'Recall@10':<12}{'vs HNSW':<10}")
    print("-" * 70)
    
    results_table = []
    
    for n_probe in n_probe_values:
        # Warmup
        for i in range(10):
            _ = index_zgq.search(queries[i % len(queries)], k=k, n_probe=n_probe)
        
        # Benchmark
        latencies = []
        results_zgq = []
        
        for query in queries:
            start = time.time()
            ids, _ = index_zgq.search(query, k=k, n_probe=n_probe)
            latency = (time.time() - start) * 1000
            latencies.append(latency)
            results_zgq.append(ids)
        
        mean_latency = np.mean(latencies)
        throughput = 1000 / mean_latency
        recall = compute_recall(results_zgq, ground_truth, k=k)
        vs_hnsw = mean_latency / mean_latency_hnsw
        
        results_table.append({
            'n_probe': n_probe,
            'latency': mean_latency,
            'throughput': throughput,
            'recall': recall,
            'vs_hnsw': vs_hnsw
        })
        
        print(f"{n_probe:<10}{mean_latency:>6.3f}ms{throughput:>12.1f} QPS{recall:>12.4f}{vs_hnsw:>9.2f}x")
    
    print()
    print("=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    # Find best configurations
    best_speed = min(results_table, key=lambda x: x['latency'])
    best_recall = max(results_table, key=lambda x: x['recall'])
    
    # Best speed/recall tradeoff (within 5% of best recall, minimize latency)
    recall_threshold = best_recall['recall'] * 0.95
    candidates = [r for r in results_table if r['recall'] >= recall_threshold]
    best_tradeoff = min(candidates, key=lambda x: x['latency'])
    
    print(f"HNSW Baseline:")
    print(f"  Latency: {mean_latency_hnsw:.3f}ms, Recall: {recall_hnsw:.4f}")
    print()
    
    print(f"Best Speed (n_probe={best_speed['n_probe']}):")
    print(f"  Latency: {best_speed['latency']:.3f}ms ({best_speed['vs_hnsw']:.2f}x slower)")
    print(f"  Recall: {best_speed['recall']:.4f} ({(best_speed['recall']-recall_hnsw)*100:+.1f}%)")
    print()
    
    print(f"Best Recall (n_probe={best_recall['n_probe']}):")
    print(f"  Latency: {best_recall['latency']:.3f}ms ({best_recall['vs_hnsw']:.2f}x slower)")
    print(f"  Recall: {best_recall['recall']:.4f} ({(best_recall['recall']-recall_hnsw)*100:+.1f}%)")
    print()
    
    print(f"Best Tradeoff (n_probe={best_tradeoff['n_probe']}):")
    print(f"  Latency: {best_tradeoff['latency']:.3f}ms ({best_tradeoff['vs_hnsw']:.2f}x slower)")
    print(f"  Recall: {best_tradeoff['recall']:.4f} ({(best_tradeoff['recall']-recall_hnsw)*100:+.1f}%)")
    print()
    
    # Victory check
    if best_speed['vs_hnsw'] < 2.0:
        print("🎉 SUCCESS! ZGQ is now within 2x of HNSW performance!")
    elif best_speed['vs_hnsw'] < 3.0:
        print("✓ Good progress! ZGQ is within 3x of HNSW.")
    else:
        print(f"⚠ Still {best_speed['vs_hnsw']:.1f}x slower than HNSW. More optimization needed.")
    
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
