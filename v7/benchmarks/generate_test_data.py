"""
Generate test data for benchmarking.
"""

import numpy as np
from pathlib import Path
import argparse


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
    parser = argparse.ArgumentParser(description='Generate test data for ANNS benchmarks')
    parser.add_argument('--n_vectors', type=int, default=10000, 
                       help='Number of vectors (default: 10000)')
    parser.add_argument('--n_queries', type=int, default=100,
                       help='Number of queries (default: 100)')
    parser.add_argument('--dim', type=int, default=128,
                       help='Vector dimension (default: 128)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory (default: data)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"GENERATING TEST DATA")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Vectors: {args.n_vectors:,}")
    print(f"  Queries: {args.n_queries}")
    print(f"  Dimensions: {args.dim}")
    print(f"  Output: {output_dir.absolute()}")
    print(f"{'='*70}\n")
    
    vectors, queries, ground_truth = generate_test_data(
        n_vectors=args.n_vectors,
        n_queries=args.n_queries,
        dim=args.dim
    )
    
    # Save with size in filename
    size_suffix = f"{args.n_vectors//1000}k" if args.n_vectors < 1000000 else f"{args.n_vectors//1000000}m"
    
    vector_file = output_dir / f'vectors_{size_suffix}.npy'
    query_file = output_dir / f'queries_{args.n_queries}.npy'
    gt_file = output_dir / f'ground_truth_{size_suffix}.npy'
    
    print("\nSaving data...")
    np.save(vector_file, vectors)
    np.save(query_file, queries)
    np.save(gt_file, ground_truth)
    
    print(f"\n{'='*70}")
    print(f"✓ DATA SAVED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Files created:")
    print(f"  - {vector_file.name} ({vectors.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  - {query_file.name} ({queries.nbytes / 1024 / 1024:.1f} MB)")
    print(f"  - {gt_file.name} ({ground_truth.nbytes / 1024 / 1024:.1f} MB)")
    print(f"\nTo use this data, update the benchmark script to load:")
    print(f"  vectors = np.load('{vector_file}')")
    print(f"  queries = np.load('{query_file}')")
    print(f"  ground_truth = np.load('{gt_file}')")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
