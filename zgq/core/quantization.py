import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numba import njit, prange, float32, uint8
from sklearn.cluster import MiniBatchKMeans

# Numba's default 'workqueue' threading layer aborts the process if a parallel
# kernel is launched from several Python threads at once, so serialise launches.
_PARALLEL_KERNEL_LOCK = threading.Lock()


@dataclass
class RPQConfig:
    m: int = 16                    # Number of subspaces
    n_bits: int = 8                # Bits per code (k = 2^n_bits centroids)

    n_train_samples: int = 100000  # Max samples for training
    n_iter: int = 25               # K-means iterations

    use_residuals: bool = True
    normalize_residuals: bool = False

    random_state: int = 42


class ResidualProductQuantizer:
    """Product quantizer over (vector - zone centroid) residuals.

    codebooks has shape (m, k, d/m).
    """

    def __init__(self, config: Optional[RPQConfig] = None):
        self.config = config or RPQConfig()
        self.k: int = 2 ** self.config.n_bits

        self.codebooks: Optional[np.ndarray] = None
        self.subvector_dim: int = 0
        self.dimension: int = 0
        self.is_trained: bool = False

        self.mean_residual: Optional[np.ndarray] = None
        self.std_residual: Optional[float] = None

    def train(
        self,
        vectors: np.ndarray,
        zone_centroids: Optional[np.ndarray] = None,
        zone_assignments: Optional[np.ndarray] = None
    ) -> 'ResidualProductQuantizer':
        vectors = vectors.astype(np.float32)
        n_vectors, self.dimension = vectors.shape

        if self.dimension % self.config.m != 0:
            raise ValueError(
                f"Dimension {self.dimension} must be divisible by m={self.config.m}"
            )

        self.subvector_dim = self.dimension // self.config.m

        train_vectors = vectors
        train_assignments = zone_assignments
        if n_vectors > self.config.n_train_samples:
            rng = np.random.RandomState(self.config.random_state)
            indices = rng.choice(n_vectors, self.config.n_train_samples, replace=False)
            train_vectors = vectors[indices]
            train_assignments = None if zone_assignments is None else zone_assignments[indices]

        train_data = train_vectors
        if (self.config.use_residuals and
            zone_centroids is not None and
            train_assignments is not None):
            train_data = train_vectors - zone_centroids[train_assignments]

        if self.config.normalize_residuals:
            self.mean_residual = train_data.mean(axis=0)
            self.std_residual = train_data.std() + 1e-8
            train_data = (train_data - self.mean_residual) / self.std_residual

        self.codebooks = np.zeros(
            (self.config.m, self.k, self.subvector_dim),
            dtype=np.float32
        )

        for j in range(self.config.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            kmeans = MiniBatchKMeans(
                n_clusters=self.k,
                random_state=self.config.random_state + j,
                batch_size=min(1024, len(train_data)),
                n_init=1,
                max_iter=self.config.n_iter,
                verbose=0
            )
            kmeans.fit(train_data[:, start_idx:end_idx])
            self.codebooks[j] = kmeans.cluster_centers_.astype(np.float32)

        self.is_trained = True
        return self

    def encode(
        self,
        vectors: np.ndarray,
        zone_centroids: Optional[np.ndarray] = None,
        zone_assignments: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """PQ codes of shape (N, m), dtype uint8."""
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before encoding")

        vectors = vectors.astype(np.float32)

        data = vectors
        if (self.config.use_residuals and
            zone_centroids is not None and
            zone_assignments is not None):
            data = vectors - zone_centroids[zone_assignments]

        if self.config.normalize_residuals and self.mean_residual is not None:
            data = (data - self.mean_residual) / self.std_residual

        n_vectors = len(vectors)
        codes = np.zeros((n_vectors, self.config.m), dtype=np.uint8)

        for j in range(self.config.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            subvectors = data[:, start_idx:end_idx]
            centroids = self.codebooks[j]

            # Row chunks keep the (chunk, k, d/m) broadcast small at large N;
            # the codes are the same as the unchunked computation.
            for lo in range(0, n_vectors, 50000):
                sub = subvectors[lo:lo + 50000]
                distances = np.sum(
                    (sub[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
                    axis=2
                )
                codes[lo:lo + 50000, j] = np.argmin(distances, axis=1).astype(np.uint8)

        return codes

    def compute_distance_table(
        self,
        query: np.ndarray,
        zone_centroid: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Table of shape (m, k) of squared distances from query subvectors to codewords.

        Uses the query residual when zone_centroid is given, so the table is only
        valid for candidates in that zone.
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before computing distances")

        query = query.astype(np.float32)

        query_data = query
        if self.config.use_residuals and zone_centroid is not None:
            query_data = query - zone_centroid

        if self.config.normalize_residuals and self.mean_residual is not None:
            query_data = (query_data - self.mean_residual) / self.std_residual

        distance_table = np.zeros((self.config.m, self.k), dtype=np.float32)

        for j in range(self.config.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            query_subvector = query_data[start_idx:end_idx]
            distance_table[j] = np.sum((query_subvector - self.codebooks[j]) ** 2, axis=1)

        return distance_table

    def asymmetric_distance(
        self,
        codes: np.ndarray,
        distance_table: np.ndarray
    ) -> np.ndarray:
        with _PARALLEL_KERNEL_LOCK:
            return _asymmetric_distance_numba(codes, distance_table)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Approximate residuals (N, d) when trained with residuals; add zone centroids to
        recover vectors.
        """
        if not self.is_trained:
            raise RuntimeError("PQ must be trained before decoding")

        decoded = np.zeros((len(codes), self.dimension), dtype=np.float32)

        for j in range(self.config.m):
            start_idx = j * self.subvector_dim
            end_idx = start_idx + self.subvector_dim

            decoded[:, start_idx:end_idx] = self.codebooks[j, codes[:, j]]

        if self.config.normalize_residuals and self.mean_residual is not None:
            decoded = decoded * self.std_residual + self.mean_residual

        return decoded

    def get_memory_usage(self, n_vectors: int) -> Dict[str, float]:
        codebook_bytes = self.config.m * self.k * self.subvector_dim * 4  # float32
        codes_bytes = n_vectors * self.config.m  # uint8

        original_bytes = n_vectors * self.dimension * 4  # float32
        compression_ratio = original_bytes / (codebook_bytes + codes_bytes)

        return {
            'codebook_bytes': codebook_bytes,
            'codes_bytes': codes_bytes,
            'total_bytes': codebook_bytes + codes_bytes,
            'total_mb': (codebook_bytes + codes_bytes) / (1024 ** 2),
            'compression_ratio': compression_ratio
        }

    def save_state(self) -> Dict:
        return {
            'config': {
                'm': self.config.m,
                'n_bits': self.config.n_bits,
                'use_residuals': self.config.use_residuals,
                'normalize_residuals': self.config.normalize_residuals,
                'random_state': self.config.random_state
            },
            'dimension': self.dimension,
            'subvector_dim': self.subvector_dim,
            'k': self.k,
            'codebooks': self.codebooks,
            'mean_residual': self.mean_residual,
            'std_residual': self.std_residual,
            'is_trained': self.is_trained
        }
    
    @classmethod
    def load_state(cls, state: Dict) -> 'ResidualProductQuantizer':
        config = RPQConfig(**state['config'])
        pq = cls(config)
        
        pq.dimension = state['dimension']
        pq.subvector_dim = state['subvector_dim']
        pq.k = state['k']
        pq.codebooks = state['codebooks']
        pq.mean_residual = state['mean_residual']
        pq.std_residual = state['std_residual']
        pq.is_trained = state['is_trained']
        
        return pq


@njit(float32[:](uint8[:,:], float32[:,:]), fastmath=True, parallel=True, cache=True)
def _asymmetric_distance_numba(
    codes: np.ndarray,
    distance_table: np.ndarray
) -> np.ndarray:
    n = codes.shape[0]
    m = codes.shape[1]
    distances = np.zeros(n, dtype=np.float32)

    for i in prange(n):
        dist = 0.0
        for j in range(m):
            code = codes[i, j]
            dist += distance_table[j, code]
        distances[i] = dist

    return distances


def suggest_pq_params(dimension: int, target_compression: float = 16.0) -> Tuple[int, int]:
    """Return (m, n_bits) with m dividing dimension. target_compression is unused."""
    valid_m = [m for m in [8, 16, 32, 64] if dimension % m == 0]

    if not valid_m:
        for m in range(4, dimension // 2 + 1):
            if dimension % m == 0:
                valid_m.append(m)
                if len(valid_m) >= 4:
                    break

    if dimension <= 64:
        m = valid_m[0] if valid_m else 8
    elif dimension <= 256:
        m = 16 if 16 in valid_m else valid_m[-1]
    else:
        m = 32 if 32 in valid_m else valid_m[-1]

    return m, 8
