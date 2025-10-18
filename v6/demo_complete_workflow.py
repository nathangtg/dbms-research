"""
Complete ZGQ V6 Demonstration
Shows the full workflow from data generation to visualization

This script demonstrates:
1. Dataset generation
2. ZGQ V6 index building
3. Baseline comparisons (HNSW, IVF)
4. Comprehensive benchmarking
5. Visualization generation

Usage:
    python demo_complete_workflow.py --size small   # Quick test (10K vectors)
    python demo_complete_workflow.py --size medium  # Standard test (50K vectors)
    python demo_complete_workflow.py --size large   # Full benchmark (100K vectors)
"""

import numpy as np
import argparse
import time
from pathlib import Path

from zgq_index import ZGQIndex
from baseline_algorithms import HNSWBaseline, IVFBaseline
from benchmark_framework import ANNSBenchmark
from visualization import ZGQVisualizer, AlgorithmResult


def generate_clustered_dataset(N: int, d: int, n_clusters: int = 10):
    """
    Generate realistic clustered dataset.
    
    Real-world vector data often has cluster structure,
    unlike completely random data.
    """
    print(f"\nGenerating clustered dataset...")
    print(f"  N={N:,} vectors, d={d} dimensions, {n_clusters} clusters")
    
    vectors = []
    for i in range(n_clusters):
        # Random cluster center
        center = np.random.randn(d) * 3
        
        # Generate cluster points
        cluster_size = N // n_clusters
        cluster_points = center + np.random.randn(cluster_size, d) * 0.8
        vectors.append(cluster_points)
    
    vectors = np.vstack(vectors).astype(np.float32)
    
    # Shuffle
    indices = np.random.permutation(N)
    vectors = vectors[indices]
    
    print(f"  ✓ Dataset generated")
    return vectors


def run_complete_demo(dataset_size: str = "small"):
    """
    Run complete demonstration workflow.
    
    Args:
        dataset_size: "small", "medium", or "large"
    """
    # Configuration based on size
    configs = {
        "small": {
            "N": 10000,
            "d": 128,
            "Q": 100,
            "k": 10,
            "n_trials": 2
        },
        "medium": {
            "N": 50000,
            "d": 128,
            "Q": 500,
            "k": 10,
            "n_trials": 3
        },
        "large": {
            "N": 100000,
            "d": 128,
            "Q": 1000,
            "k": 10,
            "n_trials": 3
        }
    }
    
    config = configs.get(dataset_size, configs["small"])
    N, d, Q, k = config["N"], config["d"], config["Q"], config["k"]
    n_trials = config["n_trials"]
    
    print("="*80)
    print(f"ZGQ V6 - Complete Workflow Demonstration")
    print("="*80)
    print(f"Dataset size: {dataset_size.upper()}")
    print(f"  Vectors: {N:,}")
    print(f"  Dimensions: {d}")
    print(f"  Queries: {Q:,}")
    print(f"  k: {k}")
    print(f"  Trials: {n_trials}")
    
    # ========================================================================
    # Step 1: Generate Dataset
    # ========================================================================
    np.random.seed(42)
    vectors = generate_clustered_dataset(N, d, n_clusters=20)
    
    # Generate query vectors (from dataset with noise)
    query_indices = np.random.choice(N, Q, replace=False)
    queries = vectors[query_indices].copy()
    queries += np.random.randn(Q, d).astype(np.float32) * 0.1  # Add noise
    
    # ========================================================================
    # Step 2: Create Benchmark Framework
    # ========================================================================
    print(f"\n{'='*80}")
    print("Setting up benchmark framework...")
    print(f"{'='*80}")
    
    benchmark = ANNSBenchmark(
        dataset=vectors,
        queries=queries,
        k_values=[1, 5, 10, 20, 50] if k >= 50 else [1, 5, 10],
        n_trials=n_trials,
        verbose=True
    )
    
    # ========================================================================
    # Step 3: Benchmark ZGQ V6
    # ========================================================================
    print(f"\n{'='*80}")
    print("Benchmarking ZGQ V6...")
    print(f"{'='*80}")
    
    # Scale parameters based on dataset size
    # Key insight: Fewer zones = larger zones = better balance = higher recall
    if dataset_size == "small":
        n_zones = 50              # 10K / 50 = 200 vectors/zone
        ef_search_zgq = 50
        ef_search_hnsw = 50
        n_probe = 5
    elif dataset_size == "medium":
        # Tuned profile for 50K vectors: faster with high recall
        # From tuning sweep: best trade-off ~ Z=30, n_probe=3, ef_search=50
        n_zones = 30              # ~1,666 vectors/zone; avoids tiny zones and stays balanced
        ef_search_zgq = 50
        ef_search_hnsw = 50
        n_probe = 3
    else:  # large
        n_zones = 50              # 100K / 50 = 2000 vectors/zone
        ef_search_zgq = 200
        ef_search_hnsw = 200
        n_probe = 10
    
    # Default configuration with scaled parameters
    zgq_result = benchmark.benchmark_zgq(
        n_zones=n_zones,
        M=16,
        ef_construction=200,
        ef_search=ef_search_zgq,
        n_probe=n_probe,
        pq_m=16,
        use_pq=True,
        adaptive_probe=True if dataset_size == "medium" else False,
        min_probe=2,
        max_probe=n_probe,
        probe_margin_ratio=0.10,
        adaptive_ef=True if dataset_size == "medium" else False,
        small_zone_threshold=1200,
        ef_small=32,
        k_rerank=30 if config["k"] == 10 else None,
        name_suffix=""
    )
    
    # ========================================================================
    # Step 4: Benchmark HNSW Baseline
    # ========================================================================
    print(f"\n{'='*80}")
    print("Benchmarking HNSW Baseline...")
    print(f"{'='*80}")
    
    hnsw_result = benchmark.benchmark_hnsw(
        M=16,
        ef_construction=200,
        ef_search=ef_search_hnsw,
        name_suffix=""
    )
    
    # ========================================================================
    # Step 5: Benchmark IVF Baselines
    # ========================================================================
    print(f"\n{'='*80}")
    print("Benchmarking IVF Baseline (exact)...")
    print(f"{'='*80}")
    
    ivf_result = benchmark.benchmark_ivf(
        n_clusters=min(100, N // 100),
        n_probe=10,
        use_pq=False,
        name_suffix=""
    )
    
    print(f"\n{'='*80}")
    print("Benchmarking IVF+PQ Baseline...")
    print(f"{'='*80}")
    
    ivf_pq_result = benchmark.benchmark_ivf(
        n_clusters=min(100, N // 100),
        n_probe=10,
        use_pq=True,
        pq_m=16,
        name_suffix=""
    )
    
    # ========================================================================
    # Step 6: Print Summary
    # ========================================================================
    benchmark.print_summary()
    
    # ========================================================================
    # Step 7: Save Results
    # ========================================================================
    results_file = f"benchmark_results_{dataset_size}.json"
    benchmark.save_results(results_file)
    
    # ========================================================================
    # Step 8: Generate Visualizations
    # ========================================================================
    print(f"\n{'='*80}")
    print("Generating visualizations...")
    print(f"{'='*80}")
    
    viz = ZGQVisualizer(output_dir=f"./figures_{dataset_size}")
    
    # Add results to visualizer
    for result in benchmark.results:
        algo_result = AlgorithmResult(
            name=result.algorithm,
            version="V6" if "ZGQ" in result.algorithm else "baseline",
            recall_at_10=result.recall_at_k.get(10, 0.0),
            latency_ms=result.mean_latency,
            memory_mb=result.memory_mb,
            build_time_s=result.build_time,
            qps=result.queries_per_second,
            config=result.config,
            recall_at_k=result.recall_at_k,
            p50_latency=result.median_latency,
            p95_latency=result.p95_latency,
            p99_latency=result.p99_latency
        )
        viz.add_result(algo_result)
    
    # Generate all plots
    viz.generate_all_plots()
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print(f"\n{'='*80}")
    print("DEMO COMPLETE!")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  - Benchmark data: {results_file}")
    print(f"  - Visualizations: ./figures_{dataset_size}/")
    print(f"\nKey findings:")
    print(f"  ZGQ V6:")
    print(f"    - Recall@10: {zgq_result.recall_at_k[10]:.4f}")
    print(f"    - Latency: {zgq_result.mean_latency:.3f} ms")
    print(f"    - Throughput: {int(zgq_result.queries_per_second)} QPS")
    print(f"    - Memory: {zgq_result.memory_mb:.1f} MB")
    print(f"\n  HNSW Baseline:")
    print(f"    - Recall@10: {hnsw_result.recall_at_k[10]:.4f}")
    print(f"    - Latency: {hnsw_result.mean_latency:.3f} ms")
    print(f"    - Throughput: {int(hnsw_result.queries_per_second)} QPS")
    print(f"    - Memory: {hnsw_result.memory_mb:.1f} MB")
    
    # Compute improvements
    recall_improvement = (zgq_result.recall_at_k[10] / hnsw_result.recall_at_k[10] - 1) * 100
    latency_improvement = (1 - zgq_result.mean_latency / hnsw_result.mean_latency) * 100
    memory_improvement = (1 - zgq_result.memory_mb / hnsw_result.memory_mb) * 100
    
    print(f"\n  ZGQ vs HNSW improvements:")
    print(f"    - Recall: {recall_improvement:+.1f}%")
    print(f"    - Latency: {latency_improvement:+.1f}% {'faster' if latency_improvement > 0 else 'slower'}")
    print(f"    - Memory: {memory_improvement:+.1f}% {'less' if memory_improvement > 0 else 'more'}")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZGQ V6 Complete Workflow Demonstration"
    )
    parser.add_argument(
        "--size",
        type=str,
        choices=["small", "medium", "large"],
        default="small",
        help="Dataset size (small=10K, medium=50K, large=100K vectors)"
    )
    
    args = parser.parse_args()
    
    try:
        run_complete_demo(args.size)
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        import traceback
        traceback.print_exc()
