"""
HNSWZoneLib - Fast per-zone HNSW using hnswlib

Provides a minimal interface compatible with ZGQ's expectations:
 - build() to construct the index
 - search(query, k, ef_search) returning List[(distance, local_id)]
 - get_memory_usage() approximate memory footprint in MB

This replaces the slower custom Python HNSWGraph for zone-level indices.
"""

from typing import List, Tuple
import numpy as np
import hnswlib


class HNSWZoneLib:
    def __init__(
        self,
        vectors: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
        verbose: bool = False,
    ) -> None:
        self.vectors = vectors.astype(np.float32, copy=False)
        self.n, self.d = self.vectors.shape
        self.M = M
        self.ef_construction = ef_construction
        self.verbose = verbose

        self.index = None  # type: ignore[assignment]
        self.search_threads = 1

    def build(self) -> None:
        if self.n == 0:
            return
        self.index = hnswlib.Index(space='l2', dim=self.d)
        self.index.init_index(max_elements=self.n, ef_construction=self.ef_construction, M=self.M)
        # Use local IDs 0..n-1
        ids = np.arange(self.n, dtype=np.int64)
        self.index.add_items(self.vectors, ids)

    def search(self, query: np.ndarray, k: int, ef_search: int = 50) -> List[Tuple[float, int]]:
        if self.index is None or self.n == 0:
            return []
        # Ensure query shape [d]
        q = query.astype(np.float32, copy=False).reshape(1, -1)
        # Set ef for this search
        self.index.set_ef(max(ef_search, k))
        # Set search threads to avoid oversubscription
        try:
            self.index.set_num_threads(max(1, int(self.search_threads)))
        except Exception:
            pass
        labels, distances = self.index.knn_query(q, k=k)
        # hnswlib returns labels [1,k] and distances [1,k]
        lbls = labels[0]
        dists = distances[0]
        # Return list of (distance, local_id)
        return [(float(d), int(l)) for d, l in zip(dists, lbls) if l != -1]

    def set_search_threads(self, n_threads: int) -> None:
        self.search_threads = max(1, int(n_threads))

    def get_memory_usage(self) -> float:
        """Approximate memory usage in MB."""
        # Rough estimate: only graph links; vectors are owned by parent index
        graph_bytes = self.n * self.M * 2 * 4
        return graph_bytes / (1024 ** 2)
