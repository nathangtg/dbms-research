"""
ZGQ v8 Quick Start Example
===========================

This example demonstrates basic usage of the ZGQ library.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zgq import ZGQIndex
from zgq.index import ZGQConfig
from zgq.search import compute_ground_truth, compute_recall


def main():
    print("="*60)
    print("ZGQ v8 Quick Start Example")
    print("="*60)
    
    # Generate sample data
    print("\n[1] Generating sample data...")
    np.random.seed(42)
    
    n_vectors = 10000
    dimension = 128
    n_queries = 100
    
    vectors = np.random.randn(n_vectors, dimension).astype(np.float32)
    queries = np.random.randn(n_queries, dimension).astype(np.float32)
    
    print(f"    Vectors: {n_vectors} × {dimension}")
    print(f"    Queries: {n_queries}")
    
    # Create and build index
    print("\n[2] Building ZGQ index...")
    
    config = ZGQConfig(
        n_zones='auto',          # Automatic zone selection
        use_hierarchy=True,      # Enable hierarchical zones
        M=16,                    # HNSW connections
        ef_construction=200,     # Build quality
        ef_search=64,            # Search beam width
        use_pq=True,             # Enable Product Quantization
        verbose=True
    )
    
    index = ZGQIndex(config)
    
    build_start = time.time()
    index.build(vectors)
    build_time = time.time() - build_start
    
    print(f"\n    Build time: {build_time:.2f}s")
    
    # Search for nearest neighbors
    print("\n[3] Searching for nearest neighbors...")
    
    k = 10
    
    # Single query
    query = queries[0]
    ids, distances = index.search(query, k=k)
    
    print(f"\n    Single query result:")
    print(f"    Nearest neighbors: {ids[:5]}...")
    print(f"    Distances: {distances[:5]}...")
    
    # Batch search
    print("\n[4] Batch search benchmark...")
    
    # Warmup
    for _ in range(3):
        _ = index.batch_search(queries[:10], k=k)
    
    # Benchmark
    n_runs = 5
    latencies = []
    
    for _ in range(n_runs):
        start = time.time()
        all_ids, all_distances = index.batch_search(queries, k=k)
        latencies.append(time.time() - start)
    
    avg_latency = np.mean(latencies)
    latency_per_query = avg_latency / n_queries * 1000
    throughput = n_queries / avg_latency
    
    print(f"    Latency per query: {latency_per_query:.4f} ms")
    print(f"    Throughput: {throughput:,.0f} QPS")
    
    # Compute recall
    print("\n[5] Computing recall...")
    
    print("    Computing ground truth (brute force)...")
    gt_ids, gt_distances = compute_ground_truth(vectors, queries, k=k)
    
    recall = compute_recall(all_ids, gt_ids, k=k)
    print(f"    Recall@{k}: {recall:.1f}%")
    
    # Save and load
    print("\n[6] Testing save/load...")
    
    index.save('/tmp/zgq_example.index')
    loaded_index = ZGQIndex.load('/tmp/zgq_example.index')
    
    # Verify loaded index
    loaded_ids, _ = loaded_index.search(query, k=k)
    assert np.array_equal(ids, loaded_ids), "Loaded index gives different results!"
    print("    Save/load verified ✓")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"    Dataset:          {n_vectors:,} vectors")
    print(f"    Build time:       {build_time:.2f}s")
    print(f"    Query latency:    {latency_per_query:.4f}ms")
    print(f"    Throughput:       {throughput:,.0f} QPS")
    print(f"    Recall@{k}:        {recall:.1f}%")
    print("="*60)
    
    # Cleanup
    os.remove('/tmp/zgq_example.index')


if __name__ == '__main__':
    main()
