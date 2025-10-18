"""
Product Quantization Module for ZGQ V6
Implements PQ training, encoding, and distance computation as specified in product_quantization.md

Mathematical Foundation:
1. Divide d-dimensional space into m subspaces
2. Train k-means on each subspace independently
3. Encode vectors using nearest centroid indices
4. Compute distances via lookup tables

Memory Compression:
- Original: N·d·4 bytes (float32)
- PQ codes: N·m·1 byte (uint8)
- Compression ratio: ≈32× for typical parameters

Reference: Jégou et al. "Product Quantization for Nearest Neighbor Search" (2011)
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.cluster import MiniBatchKMeans
import warnings


class ProductQuantizer:
    """
    Product Quantization implementation following product_quantization.md
    
    Attributes:
        m: Number of subspaces
        nbits: Bits per subquantizer (determines k = 2^nbits)
        k: Codebook size per subspace
        d: Vector dimensionality
        d_sub: Subspace dimensionality (d/m)
        codebooks: List of m codebooks, each of shape (k, d/m)
        trained: Whether codebooks have been trained
    """
    
    def __init__(self, m: int, nbits: int = 8, verbose: bool = True):
        """
        Initialize Product Quantizer.
        
        Args:
            m: Number of subspaces (must divide d evenly)
            nbits: Bits per subspace (8 → k=256, common choice)
            verbose: Print training progress
            
        Reference: product_quantization.md Section 1.1
        """
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits
        self.d = None
        self.d_sub = None
        self.codebooks = None
        self.trained = False
        self.verbose = verbose
        
        if nbits > 8:
            warnings.warn(f"nbits={nbits} > 8 may cause memory issues. Using uint16 for codes.")
            self.dtype = np.uint16
        else:
            self.dtype = np.uint8
    
    def train(self, training_data: np.ndarray, n_iter: int = 100) -> None:
        """
        Train product quantization codebooks via k-means.
        
        Algorithm:
        1. Divide d-dimensional space into m subspaces
        2. For each subspace j:
           - Extract training subvectors: X_j = data[:, (j-1)·d/m : j·d/m]
           - Run k-means with k clusters
           - Store centroids as codebook C_j
        
        Complexity: O(m · K_iter · N_train · k · d/m) = O(K_iter · N_train · k · d)
        
        Args:
            training_data: Matrix of shape (N_train, d)
            n_iter: Max k-means iterations
            
        Reference: product_quantization.md Section 2
        """
        N_train, d = training_data.shape
        
        if d % self.m != 0:
            raise ValueError(f"Dimension d={d} must be divisible by m={self.m}")
        
        self.d = d
        self.d_sub = d // self.m
        self.codebooks = []
        
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Training Product Quantization")
            print(f"{'='*70}")
            print(f"  Training samples: {N_train:,}")
            print(f"  Dimension: {d}")
            print(f"  Subspaces (m): {self.m}")
            print(f"  Subspace dim (d/m): {self.d_sub}")
            print(f"  Codebook size (k): {self.k}")
            print(f"  Memory per codebook: {self.k * self.d_sub * 4 / 1024:.1f} KB")
        
        # Train each subspace independently
        for j in range(self.m):
            if self.verbose and j % max(1, self.m // 10) == 0:
                print(f"  Training subspace {j+1}/{self.m}...", end='\r')
            
            # Extract subvectors for subspace j
            start = j * self.d_sub
            end = start + self.d_sub
            subvectors = training_data[:, start:end].astype(np.float32)
            
            # K-means clustering using MiniBatchKMeans for speed
            kmeans = MiniBatchKMeans(
                n_clusters=self.k,
                max_iter=n_iter,
                batch_size=min(2048, N_train),
                n_init=3,
                random_state=42 + j,
                verbose=0
            )
            
            kmeans.fit(subvectors)
            
            # Store codebook (centroids)
            self.codebooks.append(kmeans.cluster_centers_.astype(np.float32))
        
        if self.verbose:
            print(f"  Training subspace {self.m}/{self.m}... ✓")
            total_codebook_memory = sum(cb.nbytes for cb in self.codebooks) / (1024 ** 2)
            print(f"  Total codebook memory: {total_codebook_memory:.2f} MB")
        
        self.trained = True
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors using trained codebooks.
        
        Algorithm:
        For each vector x:
          For each subspace j:
            1. Extract subvector: x_j = x[(j-1)·d/m : j·d/m]
            2. Find nearest centroid: c_j = argmin_ℓ ||x_j - C_j[ℓ]||²
            3. Store index: codes[i, j] = c_j
        
        Complexity: O(N · m · k · d/m) = O(N · k · d)
        
        Args:
            vectors: Matrix of shape (N, d)
            
        Returns:
            PQ codes of shape (N, m) with dtype uint8 or uint16
            
        Reference: product_quantization.md Section 3
        """
        if not self.trained:
            raise ValueError("ProductQuantizer must be trained before encoding")
        
        N, d = vectors.shape
        if d != self.d:
            raise ValueError(f"Vector dimension {d} doesn't match trained dimension {self.d}")
        
        codes = np.zeros((N, self.m), dtype=self.dtype)
        
        if self.verbose:
            print(f"\n  Encoding {N:,} vectors with PQ...")
        
        # Encode each subspace
        for j in range(self.m):
            if self.verbose and j % max(1, self.m // 10) == 0:
                print(f"    Subspace {j+1}/{self.m}...", end='\r')
            
            start = j * self.d_sub
            end = start + self.d_sub
            subvectors = vectors[:, start:end].astype(np.float32)
            
            # Find nearest centroid for each subvector
            # Shape: (N, d/m) vs (k, d/m)
            codebook = self.codebooks[j]
            
            # Compute distances to all centroids
            # Vectorized: (N, d/m) vs (k, d/m) -> (N, k)
            distances = np.sum(
                (subvectors[:, np.newaxis, :] - codebook[np.newaxis, :, :]) ** 2,
                axis=2
            )
            
            # Assign nearest centroid index
            codes[:, j] = np.argmin(distances, axis=1)
        
        if self.verbose:
            print(f"    Subspace {self.m}/{self.m}... ✓")
            code_memory = codes.nbytes / (1024 ** 2)
            original_memory = vectors.nbytes / (1024 ** 2)
            compression_ratio = original_memory / code_memory
            print(f"    Code memory: {code_memory:.2f} MB")
            print(f"    Original memory: {original_memory:.2f} MB")
            print(f"    Compression ratio: {compression_ratio:.1f}×")
        
        return codes
    
    def get_codebooks(self) -> List[np.ndarray]:
        """
        Get trained codebooks.
        
        Returns:
            List of m codebooks, each of shape (k, d/m)
        """
        if not self.trained:
            raise ValueError("ProductQuantizer has not been trained yet")
        return self.codebooks
    
    def compute_reconstruction_error(
        self, 
        vectors: np.ndarray, 
        codes: np.ndarray,
        n_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute reconstruction error statistics.
        
        Reconstructed vector: x̃ = [C₁[c₁], C₂[c₂], ..., Cₘ[cₘ]]
        Error: ||x - x̃||²
        
        Args:
            vectors: Original vectors of shape (N, d)
            codes: PQ codes of shape (N, m)
            n_samples: Number of samples to evaluate (for speed)
            
        Returns:
            Tuple of (mean_error, std_error)
        """
        N = min(n_samples, len(vectors))
        indices = np.random.choice(len(vectors), N, replace=False)
        
        errors = []
        for i in indices:
            # Reconstruct vector
            reconstructed = np.zeros(self.d, dtype=np.float32)
            for j in range(self.m):
                start = j * self.d_sub
                end = start + self.d_sub
                code_j = codes[i, j]
                reconstructed[start:end] = self.codebooks[j][code_j]
            
            # Compute error
            error = np.sum((vectors[i] - reconstructed) ** 2)
            errors.append(error)
        
        return np.mean(errors), np.std(errors)
    
    def save(self, filepath: str) -> None:
        """Save trained codebooks to disk."""
        if not self.trained:
            raise ValueError("Cannot save untrained ProductQuantizer")
        
        np.savez_compressed(
            filepath,
            m=self.m,
            nbits=self.nbits,
            k=self.k,
            d=self.d,
            d_sub=self.d_sub,
            codebooks=self.codebooks
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'ProductQuantizer':
        """Load trained codebooks from disk."""
        data = np.load(filepath, allow_pickle=True)
        
        pq = cls(m=int(data['m']), nbits=int(data['nbits']), verbose=False)
        pq.k = int(data['k'])
        pq.d = int(data['d'])
        pq.d_sub = int(data['d_sub'])
        pq.codebooks = list(data['codebooks'])
        pq.trained = True
        
        return pq


# Validation and testing
if __name__ == "__main__":
    print("="*70)
    print("Product Quantization Module - Validation")
    print("="*70)
    
    # Configuration
    N_train = 10000
    N_test = 5000
    d = 128
    m = 16
    nbits = 8
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    np.random.seed(42)
    training_data = np.random.randn(N_train, d).astype(np.float32)
    test_data = np.random.randn(N_test, d).astype(np.float32)
    
    # Test 1: Train PQ
    print("\n[Test 1] Training Product Quantizer")
    pq = ProductQuantizer(m=m, nbits=nbits, verbose=True)
    pq.train(training_data, n_iter=50)
    
    assert pq.trained, "Training failed"
    assert len(pq.codebooks) == m, f"Expected {m} codebooks, got {len(pq.codebooks)}"
    print("✓ Training successful")
    
    # Test 2: Encode vectors
    print("\n[Test 2] Encoding Vectors")
    import time
    start = time.time()
    codes = pq.encode(test_data)
    encode_time = time.time() - start
    
    assert codes.shape == (N_test, m), f"Expected shape ({N_test}, {m}), got {codes.shape}"
    assert codes.dtype == np.uint8, f"Expected uint8, got {codes.dtype}"
    print(f"✓ Encoded {N_test} vectors in {encode_time:.3f}s ({N_test/encode_time:.0f} vec/s)")
    
    # Test 3: Reconstruction error
    print("\n[Test 3] Reconstruction Error Analysis")
    mean_error, std_error = pq.compute_reconstruction_error(test_data, codes, n_samples=500)
    print(f"  Mean squared error: {mean_error:.4f}")
    print(f"  Std squared error: {std_error:.4f}")
    print(f"  Relative error: {np.sqrt(mean_error) / np.linalg.norm(test_data[0]):.2%}")
    
    # Test 4: Memory savings
    print("\n[Test 4] Memory Analysis")
    original_memory = test_data.nbytes / (1024 ** 2)
    code_memory = codes.nbytes / (1024 ** 2)
    codebook_memory = sum(cb.nbytes for cb in pq.codebooks) / (1024 ** 2)
    total_pq_memory = code_memory + codebook_memory
    
    print(f"  Original vectors: {original_memory:.2f} MB")
    print(f"  PQ codes: {code_memory:.2f} MB")
    print(f"  Codebooks: {codebook_memory:.2f} MB")
    print(f"  Total PQ: {total_pq_memory:.2f} MB")
    print(f"  Compression ratio: {original_memory / code_memory:.1f}×")
    print(f"  Space saving: {(1 - total_pq_memory/original_memory)*100:.1f}%")
    
    # Test 5: Distance table computation (from distance_metrics.py)
    print("\n[Test 5] Distance Table Integration")
    from distance_metrics import PQDistanceMetrics
    
    query = test_data[0]
    start = time.time()
    distance_table = PQDistanceMetrics.compute_distance_table(
        query, pq.codebooks, m, pq.k
    )
    table_time = time.time() - start
    
    print(f"  Distance table shape: {distance_table.shape}")
    print(f"  Computation time: {table_time*1000:.2f} ms")
    
    # Compute PQ distances
    start = time.time()
    pq_distances = PQDistanceMetrics.pq_asymmetric_distance(codes, distance_table)
    pq_time = time.time() - start
    
    print(f"  PQ distance computation: {pq_time*1000:.2f} ms for {N_test} vectors")
    print(f"  Throughput: {N_test/pq_time:.0f} distances/sec")
    
    # Compare with exact distances
    from distance_metrics import DistanceMetrics
    exact_distances = DistanceMetrics.euclidean_batch_squared(query, test_data)
    
    # Correlation between PQ and exact distances
    correlation = np.corrcoef(pq_distances, exact_distances)[0, 1]
    print(f"  PQ-Exact correlation: {correlation:.4f}")
    
    print("\n" + "="*70)
    print("✓ All tests passed successfully")
    print("="*70)
