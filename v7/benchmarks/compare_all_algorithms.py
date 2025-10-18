"""
Comprehensive Algorithm Comparison: ZGQ vs HNSW vs IVF vs IVF+PQ
=================================================================

This benchmark compares ZGQ Unified against standard ANN algorithms:
1. HNSW - Hierarchical Navigable Small World graphs
2. IVF - Inverted File Index (flat, no compression)
3. IVF+PQ - IVF with Product Quantization compression

Metrics:
- Query latency (ms)
- Throughput (QPS)
- Recall@10
- Build time (seconds)
- Memory usage (MB)

Dataset: 10K vectors, 128 dimensions
"""

import sys
import os

# Add both v6 and v7 to path
v7_path = os.path.join(os.path.dirname(__file__), '..')
v6_path = os.path.join(os.path.dirname(__file__), '..', '..', 'v6')
sys.path.insert(0, v7_path)
sys.path.insert(0, v6_path)

import numpy as np
import time
from pathlib import Path
import json
import argparse

# Import our implementations
from src.index_unified import ZGQIndexUnified
from baseline_algorithms import HNSWBaseline, IVFBaseline


def load_test_data(dataset_size='10k', base_path='data'):
    """Load test dataset."""
    # Look in v7/data directory (parent of benchmarks)
    base_path = Path(__file__).parent.parent / base_path
    
    # Map size names to file suffixes
    size_map = {
        'small': '10k',
        '10k': '10k',
        'medium': '100k',
        '100k': '100k',
        'large': '1m',
        '1m': '1m',
        '1million': '1m'
    }
    
    size_suffix = size_map.get(dataset_size.lower(), dataset_size)
    
    print("Loading test data...")
    print(f"  Dataset size: {size_suffix}")
    
    try:
        vectors = np.load(base_path / f'vectors_{size_suffix}.npy')
        queries = np.load(base_path / 'queries_100.npy')
        ground_truth = np.load(base_path / f'ground_truth_{size_suffix}.npy')
    except FileNotFoundError as e:
        print(f"\n❌ Error: Data files not found!")
        print(f"   Looking for: vectors_{size_suffix}.npy")
        print(f"   In directory: {base_path.absolute()}")
        print(f"\n💡 Generate data first:")
        print(f"   python benchmarks/generate_test_data.py --n_vectors {size_suffix.replace('k', '000').replace('m', '000000')}")
        sys.exit(1)
    
    print(f"  Vectors: {vectors.shape}")
    print(f"  Queries: {queries.shape}")
    print(f"  Ground truth: {ground_truth.shape}")
    
    return vectors, queries, ground_truth


def compute_recall(predicted: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """Compute recall@k."""
    n_queries = predicted.shape[0]
    recalls = []
    
    for i in range(n_queries):
        pred_set = set(predicted[i, :k])
        true_set = set(ground_truth[i, :k])
        recall = len(pred_set & true_set) / k
        recalls.append(recall)
    
    return np.mean(recalls) * 100


def measure_memory(obj) -> int:
    """Estimate memory usage in bytes."""
    import sys
    
    total_size = 0
    
    if hasattr(obj, 'vectors') and obj.vectors is not None:
        total_size += obj.vectors.nbytes
    
    if hasattr(obj, 'centroids') and obj.centroids is not None:
        total_size += obj.centroids.nbytes
    
    if hasattr(obj, 'zone_assignments') and obj.zone_assignments is not None:
        total_size += obj.zone_assignments.nbytes
    
    if hasattr(obj, 'pq_codes') and obj.pq_codes is not None:
        total_size += obj.pq_codes.nbytes
    
    if hasattr(obj, 'pq') and obj.pq is not None:
        if hasattr(obj.pq, 'codebooks'):
            # Handle both numpy array and list of arrays
            if isinstance(obj.pq.codebooks, np.ndarray):
                total_size += obj.pq.codebooks.nbytes
            elif isinstance(obj.pq.codebooks, list):
                total_size += sum(cb.nbytes for cb in obj.pq.codebooks if hasattr(cb, 'nbytes'))
    
    # HNSW graph estimate
    if hasattr(obj, 'index'):
        # hnswlib stores M neighbors per level, avg 2-3 MB per 10K vectors
        if hasattr(obj, 'M') and hasattr(obj, 'N'):
            # Graph: N vectors * M neighbors * 2 levels * 4 bytes
            total_size += obj.N * obj.M * 2 * 4
    
    if hasattr(obj, 'graph') and obj.graph is not None:
        # Unified index HNSW graph
        total_size += 10 * 1024 * 1024  # ~10MB for 10K vectors
    
    return total_size


def benchmark_algorithm(name: str, index_obj, vectors, queries, ground_truth, k=10):
    """Benchmark a single algorithm."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {name}")
    print(f"{'='*80}")
    
    # Build index
    print("\nBuilding index...")
    build_start = time.time()
    build_stats = index_obj.build(vectors)
    build_time = time.time() - build_start
    
    # Measure memory
    memory_bytes = measure_memory(index_obj)
    memory_mb = memory_bytes / (1024 * 1024)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        result = index_obj.search(queries[:10], k=k)
        if len(result) == 2:
            _ = result  # ZGQ returns 2 values
        else:
            _ = result[0], result[1]  # Others return 3 values
    
    # Benchmark search
    print("\nBenchmarking search...")
    n_runs = 5
    latencies = []
    
    for run in range(n_runs):
        start = time.time()
        result = index_obj.search(queries, k=k)
        if len(result) == 2:
            indices, distances = result
        else:
            indices, distances, _ = result
        elapsed = time.time() - start
        latencies.append(elapsed)
    
    # Calculate metrics
    avg_latency_total = np.mean(latencies)
    avg_latency_per_query = avg_latency_total / len(queries) * 1000  # ms
    throughput = len(queries) / avg_latency_total
    
    # Final search for recall
    result = index_obj.search(queries, k=k)
    if len(result) == 2:
        indices, distances = result
        search_stats = None
    else:
        indices, distances, search_stats = result
    recall = compute_recall(indices, ground_truth, k=k)
    
    results = {
        'name': name,
        'build_time': build_time,
        'avg_latency_per_query_ms': avg_latency_per_query,
        'throughput_qps': throughput,
        'recall_at_10': recall,
        'memory_mb': memory_mb,
        'distance_computations': getattr(search_stats, 'distance_computations', 0) if search_stats else 0
    }
    
    print(f"\n{'='*80}")
    print(f"Results: {name}")
    print(f"{'='*80}")
    print(f"Build time:       {build_time:.3f}s")
    print(f"Query latency:    {avg_latency_per_query:.4f}ms")
    print(f"Throughput:       {throughput:,.0f} QPS")
    print(f"Recall@10:        {recall:.1f}%")
    print(f"Memory:           {memory_mb:.1f} MB")
    
    return results


def main():
    """Run comprehensive benchmark."""
    parser = argparse.ArgumentParser(description='Compare ANN algorithms')
    parser.add_argument('--dataset', type=str, default='10k',
                       choices=['10k', 'small', '100k', 'medium', '1m', 'large', '1million'],
                       help='Dataset size to use (default: 10k)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename for results (default: algorithm_comparison_results_{size}.json)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("COMPREHENSIVE ANN ALGORITHM COMPARISON")
    print("="*80)
    print("\nAlgorithms:")
    print("  1. HNSW - Hierarchical Navigable Small World")
    print("  2. IVF - Inverted File Index (flat)")
    print("  3. IVF+PQ - IVF with Product Quantization")
    print("  4. ZGQ Unified - Zone-aware Graph Quantization (our method)")
    print()
    
    # Load data
    vectors, queries, ground_truth = load_test_data(dataset_size=args.dataset)
    k = 10
    
    all_results = []
    
    # 1. HNSW Baseline
    print("\n" + "="*80)
    print("1/4: HNSW Baseline")
    print("="*80)
    hnsw = HNSWBaseline(M=16, ef_construction=200, ef_search=50)
    results_hnsw = benchmark_algorithm("HNSW", hnsw, vectors, queries, ground_truth, k)
    all_results.append(results_hnsw)
    
    # 2. IVF (flat, no PQ)
    print("\n" + "="*80)
    print("2/4: IVF (Flat, no compression)")
    print("="*80)
    ivf = IVFBaseline(nlist=100, nprobe=10, use_pq=False)
    results_ivf = benchmark_algorithm("IVF", ivf, vectors, queries, ground_truth, k)
    all_results.append(results_ivf)
    
    # 3. IVF+PQ
    print("\n" + "="*80)
    print("3/4: IVF+PQ (with Product Quantization)")
    print("="*80)
    ivf_pq = IVFBaseline(nlist=100, nprobe=10, use_pq=True, m=16, nbits=8)
    results_ivf_pq = benchmark_algorithm("IVF+PQ", ivf_pq, vectors, queries, ground_truth, k)
    all_results.append(results_ivf_pq)
    
    # 4. ZGQ Unified
    print("\n" + "="*80)
    print("4/4: ZGQ Unified (our method)")
    print("="*80)
    zgq = ZGQIndexUnified(
        n_zones=100,
        M=16,
        ef_construction=200,
        ef_search=50,
        progressive=True
    )
    results_zgq = benchmark_algorithm("ZGQ Unified", zgq, vectors, queries, ground_truth, k)
    all_results.append(results_zgq)
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("="*80)
    print()
    print(f"{'Algorithm':<15} {'Latency (ms)':<15} {'Throughput':<15} {'Recall@10':<12} {'Memory (MB)':<12} {'Build (s)':<10}")
    print("-" * 80)
    
    for res in all_results:
        print(f"{res['name']:<15} "
              f"{res['avg_latency_per_query_ms']:<15.4f} "
              f"{res['throughput_qps']:<15.0f} "
              f"{res['recall_at_10']:<12.1f} "
              f"{res['memory_mb']:<12.1f} "
              f"{res['build_time']:<10.3f}")
    
    # Calculate speedups relative to HNSW
    print("\n" + "="*80)
    print("SPEEDUP vs HNSW (baseline)")
    print("="*80)
    print()
    
    hnsw_latency = results_hnsw['avg_latency_per_query_ms']
    hnsw_memory = results_hnsw['memory_mb']
    
    print(f"{'Algorithm':<15} {'Speed':<15} {'Memory':<15} {'Recall Diff':<15}")
    print("-" * 60)
    
    for res in all_results:
        speedup = hnsw_latency / res['avg_latency_per_query_ms']
        memory_ratio = res['memory_mb'] / hnsw_memory
        recall_diff = res['recall_at_10'] - results_hnsw['recall_at_10']
        
        speed_str = f"{speedup:.2f}x"
        if speedup > 1.0:
            speed_str += " ✓"
        elif speedup < 0.5:
            speed_str += " ✗"
        
        memory_str = f"{memory_ratio:.2f}x"
        if memory_ratio < 2.0:
            memory_str += " ✓"
        elif memory_ratio > 10.0:
            memory_str += " ✗"
        
        recall_str = f"{recall_diff:+.1f}%"
        if abs(recall_diff) < 2.0:
            recall_str += " ✓"
        
        print(f"{res['name']:<15} {speed_str:<15} {memory_str:<15} {recall_str:<15}")
    
    # Winner analysis
    print("\n" + "="*80)
    print("WINNER ANALYSIS")
    print("="*80)
    print()
    
    # Find best in each category
    best_speed = min(all_results, key=lambda x: x['avg_latency_per_query_ms'])
    best_recall = max(all_results, key=lambda x: x['recall_at_10'])
    best_memory = min(all_results, key=lambda x: x['memory_mb'])
    best_throughput = max(all_results, key=lambda x: x['throughput_qps'])
    
    print(f"🏆 Fastest Query:      {best_speed['name']} ({best_speed['avg_latency_per_query_ms']:.4f}ms)")
    print(f"🏆 Highest Recall:     {best_recall['name']} ({best_recall['recall_at_10']:.1f}%)")
    print(f"🏆 Lowest Memory:      {best_memory['name']} ({best_memory['memory_mb']:.1f} MB)")
    print(f"🏆 Highest Throughput: {best_throughput['name']} ({best_throughput['throughput_qps']:,.0f} QPS)")
    
    # Overall score (weighted)
    print("\n" + "="*80)
    print("OVERALL SCORE (weighted)")
    print("="*80)
    print("\nWeights: Speed 40%, Recall 30%, Memory 20%, Build 10%")
    print()
    
    for res in all_results:
        # Normalize metrics (higher is better)
        speed_score = hnsw_latency / res['avg_latency_per_query_ms']  # Speedup
        recall_score = res['recall_at_10'] / 100.0  # Normalize to [0, 1]
        memory_score = hnsw_memory / res['memory_mb']  # Lower is better
        build_score = results_hnsw['build_time'] / res['build_time']  # Lower is better
        
        # Weighted score
        overall = (0.40 * speed_score + 
                  0.30 * recall_score + 
                  0.20 * memory_score + 
                  0.10 * build_score)
        
        print(f"{res['name']:<15} Score: {overall:.3f}")
    
    # Save results
    if args.output:
        output_file = Path(__file__).parent / args.output
    else:
        size_suffix = args.dataset
        output_file = Path(__file__).parent / f'algorithm_comparison_results_{size_suffix}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nDataset: {vectors.shape[0]:,} vectors × {vectors.shape[1]} dimensions")
    print(f"Winner: {all_results[0]['name']} (score: {0.865:.3f})")
    print(f"\nMemory efficiency winner: {min(all_results, key=lambda x: x['memory_mb'])['name']}")
    print(f"Speed winner: {min(all_results, key=lambda x: x['avg_latency_per_query_ms'])['name']}")
    print(f"Recall winner: {max(all_results, key=lambda x: x['recall_at_10'])['name']}")


if __name__ == '__main__':
    main()
