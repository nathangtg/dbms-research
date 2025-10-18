"""
Product Quantization for memory-efficient vector compression in ZGQ.

This module implements Product Quantization (PQ) for compressing vectors
and computing approximate distances efficiently.
"""

import numpy as np
from typing import Tuple, Dict, Optional
from sklearn.cluster import MiniBatchKMeans


class ProductQuantizer:
    """
    Product Quantization for vector compression.
    
    Divides vectors into m subspaces and quantizes each subspace independently.
    """
    
    def __init__(
        self,
        dimension: int,
        m: int = 16,
        nbits: int = 8,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize Product Quantizer.
        
        Args:
            dimension: Vector dimension (must be divisible by m)
            m: Number of subspaces
            nbits: Bits per subquantizer (k = 2^nbits centroids)
            random_state: Random seed for reproducibility
            verbose: Whether to print progress
        """
        if dimension % m != 0:
            raise ValueError(f"Dimension {dimension} must be divisible by m={m}")
        
        self.dimension = dimension
        self.m = m
        self.nbits = nbits
        self.k = 2 ** nbits  # Number of centroids per subspace
        self.subvector_dim = dimension // m
        self.random_state = random_state
        self.verbose = verbose
        
        # Codebooks: shape (m, k, subvector_dim)
        self.codebooks: Optional[np.ndarray] = None
        self.is_trained = False
    
    def train(self, vectors: np.ndarray) -> None:
        """
        Train PQ codebooks on input vectors.
        
        Args:
            vectors: Training vectors of shape (N, d)
        """
        if self.verbose:
            print(f"\nTraining Product Quantization:")
            print(f"  Subspaces: {self.m}")
            print(f"  Centroids per subspace: {self.k}")
            print(f"  Subvector dimension: {self.subvector_dim}")
        
        n_vectors = len(vectors)
        self.codebooks = np.zeros((self.m, self.k, self.subvector_dim), dtype=np.float32)
        
        # Train one codebook per subspace
        for j in range(self.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim
            
            # Extract subvectors for this subspace
            subvectors = vectors[:, start_idx:end_idx].astype(np.float32)
            
            # Train k-means clustering
            kmeans = MiniBatchKMeans(
                n_clusters=self.k,
                random_state=self.random_state + j,  # Different seed per subspace
                batch_size=min(1024, n_vectors),
                n_init=1,
                max_iter=25,
                verbose=0
            )
            
            kmeans.fit(subvectors)
            
            # Store cluster centers
            centers = kmeans.cluster_centers_.astype(np.float32)
            self.codebooks[j, :len(centers)] = centers
            
            if self.verbose and (j + 1) % 4 == 0:
                print(f"  Trained {j + 1}/{self.m} codebooks")
        
        self.is_trained = True
        
        if self.verbose:
            # Calculate compression ratio
            original_size = n_vectors * self.dimension * 4  # float32
            compressed_size = n_vectors * self.m * 1  # uint8 codes
            codebook_size = self.m * self.k * self.subvector_dim * 4
            total_compressed = compressed_size + codebook_size
            compression_ratio = original_size / total_compressed
            
            print(f"  PQ training complete")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
    
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """
        Encode vectors using trained codebooks.
        
        Args:
            vectors: Vectors to encode of shape (N, d)
            
        Returns:
            PQ codes of shape (N, m) with dtype uint8
        """
        if not self.is_trained:
            raise ValueError("PQ must be trained before encoding")
        
        n_vectors = len(vectors)
        codes = np.zeros((n_vectors, self.m), dtype=np.uint8)
        
        # Encode each subspace
        for j in range(self.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim
            
            # Extract subvectors
            subvectors = vectors[:, start_idx:end_idx]
            
            # Find nearest centroid for each subvector
            # Compute distances to all centroids in this subspace
            centroids = self.codebooks[j]  # Shape: (k, subvector_dim)
            
            # Vectorized distance computation
            # distances[i, c] = ||subvectors[i] - centroids[c]||^2
            distances = np.sum((subvectors[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2, axis=2)
            
            # Get nearest centroid index
            codes[:, j] = np.argmin(distances, axis=1).astype(np.uint8)
        
        return codes
    
    def compute_distance_table(self, query: np.ndarray) -> np.ndarray:
        """
        Precompute distance table for asymmetric distance computation.
        
        Args:
            query: Query vector of shape (d,)
            
        Returns:
            Distance table of shape (m, k)
        """
        if not self.is_trained:
            raise ValueError("PQ must be trained before computing distance table")
        
        distance_table = np.zeros((self.m, self.k), dtype=np.float32)
        
        for j in range(self.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim
            
            # Extract query subvector
            query_subvector = query[start_idx:end_idx]
            
            # Compute distance to all centroids in this subspace
            centroids = self.codebooks[j]  # Shape: (k, subvector_dim)
            distances = np.sum((query_subvector[np.newaxis, :] - centroids) ** 2, axis=1)
            
            distance_table[j] = distances
        
        return distance_table
    
    def asymmetric_distance(
        self,
        codes: np.ndarray,
        distance_table: np.ndarray
    ) -> np.ndarray:
        """
        Compute asymmetric distances using PQ codes and distance table.
        
        Args:
            codes: PQ codes of shape (N, m) with dtype uint8
            distance_table: Precomputed distance table of shape (m, k)
            
        Returns:
            Approximate squared distances of shape (N,)
        """
        n_vectors = len(codes)
        distances = np.zeros(n_vectors, dtype=np.float32)
        
        for j in range(self.m):
            # Look up distances from table for this subspace
            distances += distance_table[j, codes[:, j]]
        
        return distances
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode PQ codes back to approximate vectors.
        
        Args:
            codes: PQ codes of shape (N, m) with dtype uint8
            
        Returns:
            Reconstructed vectors of shape (N, d)
        """
        if not self.is_trained:
            raise ValueError("PQ must be trained before decoding")
        
        n_vectors = len(codes)
        reconstructed = np.zeros((n_vectors, self.dimension), dtype=np.float32)
        
        for j in range(self.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim
            
            # Look up centroid for each code
            reconstructed[:, start_idx:end_idx] = self.codebooks[j, codes[:, j]]
        
        return reconstructed
    
    def get_memory_usage(self, n_vectors: int) -> Dict[str, int]:
        """
        Calculate memory usage for PQ.
        
        Args:
            n_vectors: Number of vectors
            
        Returns:
            Dictionary with memory usage breakdown
        """
        codebook_bytes = self.m * self.k * self.subvector_dim * 4  # float32
        codes_bytes = n_vectors * self.m * 1  # uint8
        
        return {
            'codebooks_bytes': codebook_bytes,
            'codes_bytes': codes_bytes,
            'total_bytes': codebook_bytes + codes_bytes,
            'total_mb': (codebook_bytes + codes_bytes) / (1024 ** 2)
        }
    
    def save_state(self) -> Dict:
        """
        Save PQ state for serialization.
        
        Returns:
            Dictionary with PQ state
        """
        return {
            'dimension': self.dimension,
            'm': self.m,
            'nbits': self.nbits,
            'k': self.k,
            'subvector_dim': self.subvector_dim,
            'random_state': self.random_state,
            'codebooks': self.codebooks,
            'is_trained': self.is_trained
        }
    
    @classmethod
    def load_state(cls, state: Dict, verbose: bool = True) -> 'ProductQuantizer':
        """
        Load PQ from saved state.
        
        Args:
            state: Dictionary with PQ state
            verbose: Whether to print progress
            
        Returns:
            Loaded ProductQuantizer instance
        """
        pq = cls(
            dimension=state['dimension'],
            m=state['m'],
            nbits=state['nbits'],
            random_state=state['random_state'],
            verbose=verbose
        )
        
        pq.codebooks = state['codebooks']
        pq.is_trained = state['is_trained']
        
        return pq


def suggest_pq_parameters(dimension: int, target_compression: float = 16.0) -> Tuple[int, int]:
    """
    Suggest PQ parameters based on dimension and target compression ratio.
    
    Args:
        dimension: Vector dimension
        target_compression: Desired compression ratio (e.g., 16 for 16x compression)
        
    Returns:
        (m, nbits): Suggested values for m and nbits
    """
    # Common values for m based on dimension
    possible_m = [8, 16, 32, 64]
    valid_m = [m for m in possible_m if dimension % m == 0]
    
    if not valid_m:
        # Find divisors
        valid_m = [m for m in range(4, min(dimension // 2, 65)) if dimension % m == 0]
    
    # Choose m close to sqrt(dimension) for balance
    target_m = int(dimension ** 0.5)
    m = min(valid_m, key=lambda x: abs(x - target_m))
    
    # Determine nbits based on target compression
    # Compression ≈ (d * 4) / (m * 1) = 4d / m
    # For finer control, we can adjust nbits (doesn't affect compression much)
    nbits = 8  # Standard 256 centroids per subspace
    
    return m, nbits
