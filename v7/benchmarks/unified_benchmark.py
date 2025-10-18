"""
Unified ZGQ Benchmark - Test if we can beat HNSW!

Compare:
1. Pure HNSW
2. ZGQ Optimized (multi-graph)
3. ZGQ Unified (single graph) ← THE GAME CHANGER
"""

import numpy as np
import time
import sys
sys.path.insert(0, 'src')

from index_unified import ZGQIndexUnified
from index_optimized import ZGQIndexOptimized
import hnswlib


def compute_recall(results, ground_truth, k=10):
    """Compute recall@k."""
    recalls = []
    for i, gt in enumerate(ground_truth):
        intersection = len(set(results[i][:k]) & set(gt[:k]))
        recalls.append(intersection / k)
    return np.mean(recalls)


def benchmark_index(index, queries, ground_truth, k=10, n_probe=1, warmup=10, name="Index"):
    """Benchmark an index."""
    # Warmup
    for i in range(warmup):
        _ = index.search(queries[i % len(queries)], k=k, n_probe=n_probe)
    
    # Benchmark
    latencies = []
    results = []
    
    for query in queries:
        start = time.time()
        ids, _ = index.search(query, k=k, n_probe=n_probe)
        latency = (time.time() - start) * 1000
        latencies.append(latency)
        results.append(ids)
    
    mean_latency = np.mean(latencies)
    throughput = 1000 / mean_latency
    recall = compute_recall(results, ground_truth, k=k)
    
    return {
        'name': name,
        'latency': mean_latency,
        'throughput': throughput,
        'recall': recall,
        'latencies': latencies
    }


def main():
    # Generate dataset
    np.random.seed(42)
    n_vectors = 10000
    n_queries = 100
    dim = 128
    k = 10
    
    print("=" * 80)
    print("UNIFIED ZGQ BENCHMARK - CAN WE BEAT HNSW?")
    print("=" * 80)
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
    
    # Build indexes
    print("Building indexes...")
    print()
    
    # 1. Pure HNSW
    print("[1/3] HNSW Baseline...")
    index_hnsw = hnswlib.Index(space='l2', dim=dim)
    index_hnsw.init_index(max_elements=n_vectors, M=16, ef_construction=200)
    build_start = time.time()
    index_hnsw.add_items(vectors, np.arange(n_vectors))
    build_time_hnsw = time.time() - build_start
    index_hnsw.set_ef(50)
    print(f"  Build time: {build_time_hnsw:.3f}s")
    print()
    
    # 2. ZGQ Optimized (multi-graph)
    print("[2/3] ZGQ Optimized (Multi-Graph)...")
    index_zgq_opt = ZGQIndexOptimized(
        n_zones=100,
        M=16,
        ef_construction=200,
        use_pq=False,  # Disable PQ for fair comparison
        verbose=False
    )
    build_start = time.time()
    index_zgq_opt.build(vectors)
    build_time_zgq_opt = time.time() - build_start
    print(f"  Build time: {build_time_zgq_opt:.3f}s")
    print()
    
    # 3. ZGQ Unified (single graph) - THE GAME CHANGER
    print("[3/3] ZGQ Unified (Single Graph)...")
    index_zgq_uni = ZGQIndexUnified(
        n_zones=100,
        M=16,
        ef_construction=200,
        ef_search=50,
        progressive=True,
        verbose=False
    )
    build_start = time.time()
    index_zgq_uni.build(vectors)
    build_time_zgq_uni = time.time() - build_start
    print(f"  Build time: {build_time_zgq_uni:.3f}s")
    print()
    
    # Benchmark
    print("=" * 80)
    print("BENCHMARKING")
    print("=" * 80)
    print()
    
    results = []
    
    # HNSW
    print("[1/4] Benchmarking HNSW...")
    latencies_hnsw = []
    results_hnsw = []
    for query in queries:
        start = time.time()
        ids, _ = index_hnsw.knn_query(query, k=k)
        latency = (time.time() - start) * 1000
        latencies_hnsw.append(latency)
        results_hnsw.append(ids[0])
    
    mean_latency_hnsw = np.mean(latencies_hnsw)
    throughput_hnsw = 1000 / mean_latency_hnsw
    recall_hnsw = compute_recall(results_hnsw, ground_truth, k=k)
    
    results.append({
        'name': 'HNSW',
        'build_time': build_time_hnsw,
        'latency': mean_latency_hnsw,
        'throughput': throughput_hnsw,
        'recall': recall_hnsw
    })
    print(f"  Latency: {mean_latency_hnsw:.3f}ms, Recall: {recall_hnsw:.4f}")
    print()
    
    # ZGQ Optimized (n_probe=1)
    print("[2/4] Benchmarking ZGQ Optimized (n_probe=1)...")
    r = benchmark_index(index_zgq_opt, queries, ground_truth, k=k, n_probe=1, name="ZGQ-Opt-1")
    r['build_time'] = build_time_zgq_opt
    results.append(r)
    print(f"  Latency: {r['latency']:.3f}ms, Recall: {r['recall']:.4f}")
    print()
    
    # ZGQ Optimized (n_probe=3)
    print("[3/4] Benchmarking ZGQ Optimized (n_probe=3)...")
    r = benchmark_index(index_zgq_opt, queries, ground_truth, k=k, n_probe=3, name="ZGQ-Opt-3")
    r['build_time'] = build_time_zgq_opt
    results.append(r)
    print(f"  Latency: {r['latency']:.3f}ms, Recall: {r['recall']:.4f}")
    print()
    
    # ZGQ Unified (progressive)
    print("[4/4] Benchmarking ZGQ Unified (Progressive)...")
    r = benchmark_index(index_zgq_uni, queries, ground_truth, k=k, n_probe=1, name="ZGQ-Unified")
    r['build_time'] = build_time_zgq_uni
    results.append(r)
    print(f"  Latency: {r['latency']:.3f}ms, Recall: {r['recall']:.4f}")
    print()
    
    # Results table
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    print(f"{'Method':<20}{'Build':<12}{'Latency':<15}{'Throughput':<15}{'Recall':<10}{'vs HNSW':<10}")
    print("-" * 80)
    
    for r in results:
        vs_hnsw = r['latency'] / mean_latency_hnsw
        print(f"{r['name']:<20}{r['build_time']:>6.2f}s{r['latency']:>10.3f}ms"
              f"{r['throughput']:>12.1f} QPS{r['recall']:>10.4f}{vs_hnsw:>9.2f}x")
    
    print()
    print("=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print()
    
    # Find best ZGQ
    zgq_results = [r for r in results if 'ZGQ' in r['name']]
    best_zgq = min(zgq_results, key=lambda x: x['latency'])
    
    hnsw_result = results[0]
    
    speedup = hnsw_result['latency'] / best_zgq['latency']
    recall_diff = best_zgq['recall'] - hnsw_result['recall']
    
    print(f"HNSW Baseline:")
    print(f"  Latency: {hnsw_result['latency']:.3f}ms")
    print(f"  Recall: {hnsw_result['recall']:.4f}")
    print()
    
    print(f"Best ZGQ ({best_zgq['name']}):")
    print(f"  Latency: {best_zgq['latency']:.3f}ms")
    print(f"  Recall: {best_zgq['recall']:.4f}")
    print()
    
    if speedup > 1.0:
        print(f"🎉 VICTORY! ZGQ is {speedup:.2f}x FASTER than HNSW!")
        if recall_diff > 0:
            print(f"   AND has {recall_diff*100:+.1f}% better recall!")
    elif speedup > 0.8:
        print(f"✓ Close! ZGQ is only {1/speedup:.2f}x slower ({(1-speedup)*100:.1f}% gap)")
    else:
        print(f"⚠ Still {1/speedup:.2f}x slower than HNSW")
    
    print()
    
    # Detailed comparison
    print("Unified vs Multi-Graph ZGQ:")
    unified = [r for r in results if 'Unified' in r['name']][0]
    multi = [r for r in results if 'Opt-1' in r['name']][0]
    
    speedup_internal = multi['latency'] / unified['latency']
    print(f"  Unified is {speedup_internal:.2f}x faster than multi-graph")
    print(f"  Latency: {unified['latency']:.3f}ms vs {multi['latency']:.3f}ms")
    print(f"  Recall: {unified['recall']:.4f} vs {multi['recall']:.4f}")
    
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
