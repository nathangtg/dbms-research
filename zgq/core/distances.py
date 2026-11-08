from typing import Optional, Tuple

import numpy as np
from numba import float32, njit, prange


class DistanceComputer:
    def __init__(self, metric: str = "l2"):
        self.metric = metric

    def compute(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        precomputed_norms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """precomputed_norms are the squared norms ||vectors||^2, shape (N,)."""
        if self.metric == "l2":
            return self._euclidean_squared(query, vectors, precomputed_norms)
        if self.metric == "cosine":
            return self._cosine_distance(query, vectors, precomputed_norms)
        if self.metric == "ip":
            return self._inner_product_distance(query, vectors)
        raise ValueError(f"Unknown metric: {self.metric}")

    def _euclidean_squared(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        precomputed_norms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>, clipped at 0 against rounding."""
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)

        query_norm = np.dot(query, query)

        vector_norms = (
            precomputed_norms
            if precomputed_norms is not None
            else np.sum(vectors * vectors, axis=1)
        )

        inner_products = vectors @ query
        distances = query_norm + vector_norms - 2 * inner_products
        np.maximum(distances, 0, out=distances)

        return distances.astype(np.float32)

    def _cosine_distance(
        self,
        query: np.ndarray,
        vectors: np.ndarray,
        precomputed_norms: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)

        query_norm = np.sqrt(np.dot(query, query))
        if query_norm > 0:
            query = query / query_norm

        if precomputed_norms is None:
            precomputed_norms = np.sum(vectors * vectors, axis=1)
        vector_norms = np.sqrt(precomputed_norms)

        inner_products = vectors @ query

        with np.errstate(divide="ignore", invalid="ignore"):
            cosine_sim = inner_products / vector_norms
            cosine_sim = np.nan_to_num(cosine_sim, nan=0.0)

        return (1 - cosine_sim).astype(np.float32)

    def _inner_product_distance(
        self, query: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        """Negative inner product, so smaller is better."""
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)
        return (-vectors @ query).astype(np.float32)

    def compute_single(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.metric == "l2":
            diff = a - b
            return float(np.dot(diff, diff))
        if self.metric == "cosine":
            norm_a = np.sqrt(np.dot(a, a))
            norm_b = np.sqrt(np.dot(b, b))
            if norm_a > 0 and norm_b > 0:
                return float(1 - np.dot(a, b) / (norm_a * norm_b))
            return 1.0
        if self.metric == "ip":
            return float(-np.dot(a, b))
        raise ValueError(f"Unknown metric: {self.metric}")

    @staticmethod
    def precompute_norms(vectors: np.ndarray) -> np.ndarray:
        """Squared L2 norms."""
        vectors = vectors.astype(np.float32)
        return np.sum(vectors * vectors, axis=1).astype(np.float32)


class SIMDDistance:
    """Numba-compiled distance kernels."""

    @staticmethod
    @njit(
        float32[:](float32[:], float32[:, :]), fastmath=True, parallel=True, cache=True
    )
    def euclidean_squared_batch(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        n = vectors.shape[0]
        d = vectors.shape[1]
        result = np.empty(n, dtype=np.float32)

        for i in prange(n):
            dist = 0.0
            for j in range(d):
                diff = query[j] - vectors[i, j]
                dist += diff * diff
            result[i] = dist

        return result

    @staticmethod
    @njit(float32(float32[:], float32[:]), fastmath=True, cache=True)
    def euclidean_squared_single(a: np.ndarray, b: np.ndarray) -> float:
        d = a.shape[0]
        dist = 0.0
        for i in range(d):
            diff = a[i] - b[i]
            dist += diff * diff
        return dist

    @staticmethod
    @njit(float32(float32[:], float32[:], float32), fastmath=True, cache=True)
    def euclidean_squared_early_term(
        a: np.ndarray, b: np.ndarray, bound: float
    ) -> float:
        """Squared distance, or bound + 1 once the partial sum exceeds bound (every 8 dims)."""
        d = a.shape[0]
        dist = 0.0

        for i in range(d):
            diff = a[i] - b[i]
            dist += diff * diff

            if i % 8 == 7 and dist > bound:
                return bound + 1.0

        return dist

    @staticmethod
    @njit(
        float32[:](float32[:, :], float32[:]), fastmath=True, parallel=True, cache=True
    )
    def compute_norms_batch(vectors: np.ndarray, out: np.ndarray) -> np.ndarray:
        n = vectors.shape[0]
        d = vectors.shape[1]

        for i in prange(n):
            norm = 0.0
            for j in range(d):
                norm += vectors[i, j] * vectors[i, j]
            out[i] = norm

        return out


class EarlyTerminationComputer:
    """Distance computation with bound-based early termination."""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._check_interval = max(8, dimension // 16)

    def compute_with_bound(
        self, query: np.ndarray, vectors: np.ndarray, indices: np.ndarray, bound: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Indices and distances of the candidates within bound."""
        query = query.astype(np.float32)
        vectors = vectors.astype(np.float32)

        valid_indices = []
        valid_distances = []

        for idx in indices:
            dist = float(
                SIMDDistance.euclidean_squared_early_term(
                    query, vectors[idx], np.float32(bound)
                )
            )
            if dist <= bound:
                valid_indices.append(idx)
                valid_distances.append(dist)

        return np.array(valid_indices), np.array(valid_distances)


def select_top_k_distances(
    distances: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Indices and values of the k smallest distances, ascending (argpartition, O(N))."""
    n = len(distances)
    k = min(k, n)

    if k == n:
        indices = np.argsort(distances)
        return indices, distances[indices]

    top_k_indices = np.argpartition(distances, k - 1)[:k]
    top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

    return top_k_indices, distances[top_k_indices]


def _warmup_jit():
    """Compile the kernels at import so the first search does not pay for it."""
    dummy_q = np.zeros(128, dtype=np.float32)
    dummy_v = np.zeros((10, 128), dtype=np.float32)
    dummy_out = np.zeros(10, dtype=np.float32)

    SIMDDistance.euclidean_squared_batch(dummy_q, dummy_v)
    SIMDDistance.euclidean_squared_single(dummy_q, dummy_q)
    SIMDDistance.euclidean_squared_early_term(dummy_q, dummy_q, 1.0)
    SIMDDistance.compute_norms_batch(dummy_v, dummy_out)


_warmup_jit()
