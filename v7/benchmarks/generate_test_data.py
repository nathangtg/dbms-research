"""
Generate test data for benchmarking.
"""

import numpy as np
from pathlib import Path


def generate_test_data(n_vectors=10000, n_queries=100, dim=128, k=10, seed=42):
    """Generate random test data and ground truth."""
    np.random.seed(seed)
    
    print(f"Generating test data:")
    print(f"  Vectors: {n_vectors:,} x {dim}")
    print(f"  Queries: {n_queries} x {dim}")
    print(f"  k: {k}")
    
    # Generate random vectors
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Normalize
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Compute ground truth using brute force
    print("\nComputing ground truth (brute force)...")
    ground_truth = np.zeros((n_queries, k), dtype=np.int32)
    
    for i, query in enumerate(queries):
        # Compute distances to all vectors
        distances = np.sum((vectors - query) ** 2, axis=1)
        # Get k nearest
        indices = np.argpartition(distances, k)[:k]
        indices = indices[np.argsort(distances[indices])]
        ground_truth[i] = indices
        
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i+1}/{n_queries}")
    
    print("\nDone!")
    return vectors, queries, ground_truth


def main():
    """Generate and save test data."""
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    vectors, queries, ground_truth = generate_test_data()
    
    # Save
    print("\nSaving data...")
    np.save(output_dir / 'vectors_10k.npy', vectors)
    np.save(output_dir / 'queries_100.npy', queries)
    np.save(output_dir / 'ground_truth_10k.npy', ground_truth)
    
    print(f"\n✓ Data saved to {output_dir.absolute()}/")
    print(f"  - vectors_10k.npy")
    print(f"  - queries_100.npy")
    print(f"  - ground_truth_10k.npy")


if __name__ == '__main__':
    main()
