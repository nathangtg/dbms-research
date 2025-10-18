"""
Diagnostic Test for ZGQ/HNSW Low Recall Issue

This script tests individual components to identify why recall is so low.
"""

import numpy as np
from zgq_index import ZGQIndex
from baseline_algorithms import HNSWBaseline, IVFBaseline
from distance_metrics import DistanceMetrics
import time

def test_simple_case():
    """Test with a very simple, controlled dataset."""
    print("="*80)
    print("DIAGNOSTIC TEST: Simple Controlled Dataset")
    print("="*80)
    
    # Create simple dataset with clear clusters
    np.random.seed(42)
    N, d, Q = 1000, 128, 10
    
    # Create 10 clear clusters
    n_clusters = 10
    vectors = []
    for i in range(n_clusters):
        center = np.random.randn(d) * 10
        cluster_points = center + np.random.randn(N // n_clusters, d) * 0.5
        vectors.append(cluster_points)
    
    vectors = np.vstack(vectors).astype(np.float32)
    
    # Generate queries from dataset with small noise
    query_indices = np.random.choice(N, Q, replace=False)
    queries = vectors[query_indices].copy()
    queries += np.random.randn(Q, d).astype(np.float32) * 0.05
    
    # Compute ground truth
    print(f"\n1. Computing ground truth...")
    dist_metric = DistanceMetrics()
    ground_truth = []
    for query in queries:
        distances = dist_metric.euclidean_batch_squared(query, vectors)
        gt_indices = np.argsort(distances)[:10]
        ground_truth.append(gt_indices)
    
    print(f"   ✓ Ground truth computed")
    print(f"   Sample GT[0]: {ground_truth[0][:5]}")
    print(f"   Query 0 closest vectors: {ground_truth[0][:3]}")
    
    # Test 1: HNSW Baseline
    print(f"\n2. Testing HNSW Baseline...")
    hnsw = HNSWBaseline(M=16, ef_construction=200, ef_search=50, seed=42)
    hnsw.build(vectors)
    
    # Test single query
    indices, distances, _ = hnsw.search(queries[0:1], k=10)
    print(f"   HNSW results[0]: {indices[0][:5]}")
    
    # Check recall
    recalls = []
    for i, query in enumerate(queries):
        indices, _, _ = hnsw.search(query.reshape(1, -1), k=10)
        indices_list = indices[0] if len(indices.shape) > 1 else indices
        gt = ground_truth[i]
        recall = len(set(indices_list) & set(gt)) / len(gt)
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    print(f"   HNSW Recall@10: {avg_recall:.4f}")
    
    if avg_recall < 0.8:
        print(f"   ❌ HNSW failing! Expected >0.8, got {avg_recall:.4f}")
        
        # Debug: Check if query[0] nearest neighbor is in the index
        query_0_indices, _, _ = hnsw.search(queries[0:1], k=10)
        query_0_indices = query_0_indices[0] if len(query_0_indices.shape) > 1 else query_0_indices
        query_0_gt_0 = ground_truth[0][0]
        print(f"\n   Debug Query 0:")
        print(f"     Ground truth NN: {query_0_gt_0}")
        print(f"     HNSW returned: {query_0_indices[:5]}")
        print(f"     GT in HNSW results? {query_0_gt_0 in query_0_indices}")
        
        # Compute exact distance to verify
        exact_dist = dist_metric.euclidean_squared(queries[0], vectors[query_0_gt_0])
        hnsw_dist_0 = dist_metric.euclidean_squared(queries[0], vectors[query_0_indices[0]])
        print(f"     Exact dist to GT NN: {exact_dist:.6f}")
        print(f"     Exact dist to HNSW[0]: {hnsw_dist_0:.6f}")
        print(f"     Is HNSW[0] actually closer? {hnsw_dist_0 < exact_dist}")
    else:
        print(f"   ✅ HNSW working correctly")
    
    # Test 2: IVF Baseline (should always work)
    print(f"\n3. Testing IVF Baseline...")
    ivf = IVFBaseline(nlist=10, nprobe=3, use_pq=False, seed=42)
    ivf.build(vectors)
    
    recalls_ivf = []
    for i, query in enumerate(queries):
        indices, _, _ = ivf.search(query.reshape(1, -1), k=10)
        indices = indices[0]
        gt = ground_truth[i]
        recall = len(set(indices) & set(gt)) / len(gt)
        recalls_ivf.append(recall)
    
    avg_recall_ivf = np.mean(recalls_ivf)
    print(f"   IVF Recall@10: {avg_recall_ivf:.4f}")
    if avg_recall_ivf > 0.95:
        print(f"   ✅ IVF working correctly (sanity check passed)")
    else:
        print(f"   ⚠ IVF recall lower than expected")
    
    # Test 3: Try HNSW with higher ef_search
    print(f"\n4. Testing HNSW with ef_search=200...")
    hnsw_high_ef = HNSWBaseline(M=16, ef_construction=200, ef_search=200, seed=42)
    hnsw_high_ef.build(vectors)
    
    recalls_high = []
    for i, query in enumerate(queries):
        indices, _, _ = hnsw_high_ef.search(query.reshape(1, -1), k=10)
        indices = indices[0]
        gt = ground_truth[i]
        recall = len(set(indices) & set(gt)) / len(gt)
        recalls_high.append(recall)
    
    avg_recall_high = np.mean(recalls_high)
    print(f"   HNSW (ef=200) Recall@10: {avg_recall_high:.4f}")
    print(f"   Improvement: {(avg_recall_high - avg_recall) / avg_recall * 100:.1f}%")
    
    # Test 4: ZGQ with very few zones
    print(f"\n5. Testing ZGQ with 10 zones...")
    zgq = ZGQIndex(
        vectors=vectors,
        Z=10,
        M=16,
        ef_construction=200,
        ef_search=100,
        n_probe=3,
        m=16,
        nbits=8,
        n_threads=4,
        verbose=False
    )
    zgq.build()
    
    recalls_zgq = []
    for i, query in enumerate(queries):
        results = zgq.search(query, k=10, return_distances=True)
        indices = [r[0] for r in results]
        gt = ground_truth[i]
        recall = len(set(indices) & set(gt)) / len(gt)
        recalls_zgq.append(recall)
    
    avg_recall_zgq = np.mean(recalls_zgq)
    print(f"   ZGQ Recall@10: {avg_recall_zgq:.4f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Dataset: N={N}, d={d}, Q={Q}")
    print(f"HNSW (ef=50):   Recall@10 = {avg_recall:.4f}")
    print(f"HNSW (ef=200):  Recall@10 = {avg_recall_high:.4f}")
    print(f"IVF (exact):    Recall@10 = {avg_recall_ivf:.4f}")
    print(f"ZGQ (10 zones): Recall@10 = {avg_recall_zgq:.4f}")
    
    if avg_recall < 0.8:
        print(f"\n⚠ ISSUE CONFIRMED: HNSW has fundamentally low recall")
        print(f"   This suggests a bug in HNSW implementation or search parameters")
    elif avg_recall_high > 0.9:
        print(f"\n✓ SOLUTION: Use higher ef_search (200 instead of 50)")
    
    return {
        'hnsw_low_ef': avg_recall,
        'hnsw_high_ef': avg_recall_high,
        'ivf': avg_recall_ivf,
        'zgq': avg_recall_zgq
    }


if __name__ == "__main__":
    results = test_simple_case()
