"""
Simple example demonstrating ZGQ usage.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from index import ZGQIndex
from search import compute_ground_truth


def main():
    print("ZGQ V7 - Simple Example")
    print("="*60)
    
    # Generate sample data
    print("\n1. Generating sample dataset...")
    N, d = 10000, 128
    n_queries = 10
    
    np.random.seed(42)
    vectors = np.random.randn(N, d).astype('float32')
    queries = np.random.randn(n_queries, d).astype('float32')
    
    print(f"   Dataset: {N} vectors, dimension {d}")
    print(f"   Queries: {n_queries}")
    
    # Build ZGQ index
    print("\n2. Building ZGQ index...")
    index = ZGQIndex(
        n_zones=100,
        hnsw_M=16,
        hnsw_ef_construction=200,
        use_pq=True,
        pq_m=16,
        pq_nbits=8,
        verbose=True
    )
    
    index.build(vectors)
    
    # Get index info
    print("\n3. Index Information:")
    info = index.get_index_info()
    for key, value in info.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Search
    print("\n4. Performing searches...")
    k = 10
    n_probe = 8
    
    print(f"   Searching for {k} nearest neighbors (n_probe={n_probe})...")
    
    # Single query search
    query = queries[0]
    ids, distances = index.search(query, k=k, n_probe=n_probe)
    
    print(f"\n   Results for first query:")
    print(f"   Top-{k} IDs: {ids}")
    print(f"   Distances: {distances}")
    
    # Batch search
    print(f"\n5. Batch search ({n_queries} queries)...")
    all_ids, all_distances = index.batch_search(queries, k=k, n_probe=n_probe)
    
    print(f"   Results shape: {all_ids.shape}")
    
    # Compute ground truth and recall
    print("\n6. Computing recall...")
    print("   Computing exact nearest neighbors...")
    ground_truth, _ = compute_ground_truth(vectors, queries, k=k)
    
    # Calculate recall
    recalls = []
    for i in range(n_queries):
        result_set = set(all_ids[i])
        gt_set = set(ground_truth[i])
        recall = len(result_set.intersection(gt_set)) / k
        recalls.append(recall)
    
    mean_recall = np.mean(recalls)
    print(f"   Mean Recall@{k}: {mean_recall:.4f}")
    
    # Save and load index
    print("\n7. Save/Load test...")
    save_path = "zgq_index_example.pkl"
    print(f"   Saving index to {save_path}...")
    index.save(save_path)
    
    print(f"   Loading index from {save_path}...")
    loaded_index = ZGQIndex.load(save_path, verbose=False)
    
    # Verify loaded index works
    test_ids, test_distances = loaded_index.search(query, k=k, n_probe=n_probe)
    
    if np.array_equal(ids, test_ids):
        print("   ✓ Loaded index produces identical results!")
    else:
        print("   ✗ Warning: Loaded index results differ")
    
    # Clean up
    import os
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"   Cleaned up {save_path}")
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
