#!/usr/bin/env python3
"""
Quick validation script to test ZGQ implementation.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from core.distances import DistanceMetrics, PQDistanceMetrics
        from core.kmeans import ZonalPartitioner
        from core.hnsw_wrapper import HNSWGraphManager
        from core.product_quantizer import ProductQuantizer
        from index import ZGQIndex
        from search import compute_ground_truth
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic ZGQ functionality."""
    print("\nTesting basic functionality...")
    
    try:
        from index import ZGQIndex
        from search import compute_ground_truth
        
        # Small test dataset
        n_vectors = 500
        dimension = 32
        n_queries = 10
        
        np.random.seed(42)
        vectors = np.random.randn(n_vectors, dimension).astype('float32')
        queries = np.random.randn(n_queries, dimension).astype('float32')
        
        print(f"  Dataset: {n_vectors} vectors, dim={dimension}")
        
        # Build index
        print("  Building index...")
        index = ZGQIndex(
            n_zones=20,
            hnsw_M=8,
            hnsw_ef_construction=100,
            use_pq=True,
            pq_m=8,
            verbose=False
        )
        index.build(vectors)
        print("  ✓ Index built successfully")
        
        # Search
        print("  Testing search...")
        query = queries[0]
        ids, distances = index.search(query, k=10, n_probe=5)
        
        assert len(ids) == 10, f"Expected 10 results, got {len(ids)}"
        assert len(distances) == 10, f"Expected 10 distances, got {len(distances)}"
        assert all(0 <= i < n_vectors for i in ids), "Invalid IDs returned"
        print(f"  ✓ Search successful: {ids[:3]}...")
        
        # Batch search
        print("  Testing batch search...")
        all_ids, all_distances = index.batch_search(queries, k=10, n_probe=5)
        assert all_ids.shape == (n_queries, 10), f"Wrong shape: {all_ids.shape}"
        print("  ✓ Batch search successful")
        
        # Ground truth
        print("  Computing ground truth...")
        gt_ids, gt_dist = compute_ground_truth(vectors, queries, k=10)
        
        # Calculate recall
        recalls = []
        for i in range(n_queries):
            result_set = set(all_ids[i])
            gt_set = set(gt_ids[i])
            recall = len(result_set.intersection(gt_set)) / 10
            recalls.append(recall)
        
        mean_recall = np.mean(recalls)
        print(f"  ✓ Mean Recall@10: {mean_recall:.4f}")
        
        if mean_recall < 0.5:
            print(f"  ⚠ Warning: Low recall ({mean_recall:.4f}), may need parameter tuning")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_components():
    """Test individual components."""
    print("\nTesting individual components...")
    
    try:
        from core.distances import DistanceMetrics
        from core.kmeans import ZonalPartitioner
        from core.product_quantizer import ProductQuantizer
        
        # Test distances
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        dist = DistanceMetrics.euclidean_squared(x, y)
        assert np.isclose(dist, 27.0), f"Distance computation error: {dist}"
        print("  ✓ Distance metrics working")
        
        # Test partitioner
        vectors = np.random.randn(100, 10).astype('float32')
        partitioner = ZonalPartitioner(n_zones=10, verbose=False)
        partitioner.fit(vectors)
        assert len(partitioner.inverted_lists) == 10
        print("  ✓ Zonal partitioner working")
        
        # Test PQ with larger dataset
        test_vectors = np.random.randn(1000, 32).astype('float32')
        pq = ProductQuantizer(dimension=32, m=8, nbits=4, verbose=False)  # nbits=4 -> 16 centroids
        pq.train(test_vectors)
        codes = pq.encode(test_vectors[:100])
        assert codes.shape == (100, 8)
        print("  ✓ Product quantization working")
        
        return True
        
    except Exception as e:
        print(f"✗ Component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("="*60)
    print("ZGQ V7 - Implementation Validation")
    print("="*60)
    
    results = []
    
    # Test imports
    results.append(("Imports", test_imports()))
    
    # Test components
    results.append(("Components", test_components()))
    
    # Test basic functionality
    results.append(("Basic Functionality", test_basic_functionality()))
    
    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    print("="*60)
    if all_passed:
        print("✓ All validation tests passed!")
        print("\nNext steps:")
        print("  1. Run full tests: pytest tests/test_zgq.py -v")
        print("  2. Run example: python examples/simple_example.py")
        print("  3. Run benchmark: python benchmarks/comprehensive_benchmark.py")
    else:
        print("✗ Some tests failed - please check the errors above")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
