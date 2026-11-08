from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from zgq.core.distances import DistanceComputer, select_top_k_distances
from zgq.core.quantization import ResidualProductQuantizer
from zgq.utils.metrics import compute_recall  # noqa: F401  (re-exported)


@dataclass
class SearchConfig:
    n_probe: int = 8
    expand_neighbors: bool = True
    max_expansion: int = 2

    k_candidates_factor: int = 3        # candidates = k * factor
    use_pq_filter: bool = True

    k_rerank_factor: int = 2            # reranked = k * factor
    use_exact_rerank: bool = True
    use_early_termination: bool = True

    quality_threshold: float = 0.9


class ZGQSearch:
    def __init__(
        self,
        zones,
        graph,
        vectors: np.ndarray,
        pq: Optional[ResidualProductQuantizer] = None,
        pq_codes: Optional[np.ndarray] = None,
        vector_norms: Optional[np.ndarray] = None,
        config: Optional[SearchConfig] = None
    ):
        self.zones = zones
        self.graph = graph
        self.vectors = vectors
        self.pq = pq
        self.pq_codes = pq_codes
        self.vector_norms = vector_norms
        self.config = config or SearchConfig()

        self.distance_computer = DistanceComputer(metric='l2')
        self.n_vectors = len(vectors)
        self.dimension = vectors.shape[1]

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: Optional[int] = None,
        ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = query.astype(np.float32)
        n_probe = n_probe or self.config.n_probe

        selected_zones = self._select_zones_adaptive(query, n_probe)

        candidates = self._generate_candidates(
            query, k, selected_zones, ef_search
        )

        if self.pq and self.config.use_pq_filter:
            candidates = self._filter_with_pq(query, candidates, k, selected_zones)

        return self._exact_rerank(query, candidates, k)

    def _select_zones_adaptive(
        self,
        query: np.ndarray,
        n_probe: int
    ) -> np.ndarray:
        selected = self.zones.select_zones(query, n_probe)

        if self.config.expand_neighbors and n_probe < len(self.zones.fine_centroids):
            expanded = set(selected.tolist())

            for zone_id in selected[:self.config.max_expansion]:
                for neighbor in self.zones.get_zone_neighbors(zone_id)[:2]:
                    expanded.add(neighbor)

            selected = np.array(list(expanded), dtype=np.int32)

        return selected

    def _generate_candidates(
        self,
        query: np.ndarray,
        k: int,
        selected_zones: np.ndarray,
        ef_search: Optional[int]
    ) -> np.ndarray:
        k_candidates = min(k * self.config.k_candidates_factor, self.n_vectors)
        ids, _ = self.graph.search(query, k_candidates, selected_zones, ef_search)
        return ids

    def _filter_with_pq(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        k: int,
        selected_zones: np.ndarray
    ) -> np.ndarray:

        if len(candidates) <= k * self.config.k_rerank_factor:
            return candidates

        query_zone = selected_zones[0]
        zone_centroid = self.zones.fine_centroids[query_zone]

        dist_table = self.pq.compute_distance_table(query, zone_centroid)
        pq_distances = self.pq.asymmetric_distance(
            self.pq_codes[candidates], dist_table
        )

        k_filter = min(k * self.config.k_rerank_factor, len(candidates))
        top_indices = np.argpartition(pq_distances, k_filter - 1)[:k_filter]

        return candidates[top_indices]

    def _exact_rerank(
        self,
        query: np.ndarray,
        candidates: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:

        if len(candidates) == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

        candidate_vectors = self.vectors[candidates]

        candidate_norms = None if self.vector_norms is None else self.vector_norms[candidates]

        distances = self.distance_computer.compute(
            query, candidate_vectors, candidate_norms
        )

        top_indices, top_distances = select_top_k_distances(distances, k)

        return candidates[top_indices], top_distances

    def search_fast(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = query.astype(np.float32)

        selected_zones = self.zones.select_zones(query, n_probe=4)

        return self.graph.search(query, k, selected_zones)

    def search_accurate(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = query.astype(np.float32)

        n_probe = min(32, len(self.zones.fine_centroids))
        selected_zones = self.zones.select_zones(query, n_probe)

        k_candidates = min(k * 10, self.n_vectors)
        ids, _ = self.graph.search(query, k_candidates, selected_zones)

        if self.pq:
            query_zone = selected_zones[0]
            zone_centroid = self.zones.fine_centroids[query_zone]
            dist_table = self.pq.compute_distance_table(query, zone_centroid)
            pq_distances = self.pq.asymmetric_distance(self.pq_codes[ids], dist_table)

            k_rerank = min(k * 5, len(ids))
            top_pq = np.argpartition(pq_distances, k_rerank - 1)[:k_rerank]
            ids = ids[top_pq]

        return self._exact_rerank(query, ids, k)

    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_probe: Optional[int] = None,
        n_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32)
        n_queries = len(queries)

        all_ids = np.zeros((n_queries, k), dtype=np.int32)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)

        def search_one(i):
            return self.search(queries[i], k, n_probe)

        if n_jobs == 1:
            results = [search_one(i) for i in range(n_queries)]
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                results = list(executor.map(search_one, range(n_queries)))

        for i, (ids, distances) in enumerate(results):
            n_results = min(k, len(ids))
            all_ids[i, :n_results] = ids[:n_results]
            all_distances[i, :n_results] = distances[:n_results]
        return all_ids, all_distances


def compute_ground_truth(
    vectors: np.ndarray,
    queries: np.ndarray,
    k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    vectors = vectors.astype(np.float32)
    queries = queries.astype(np.float32)

    n_queries = len(queries)
    k = min(k, len(vectors))

    gt_ids = np.zeros((n_queries, k), dtype=np.int32)
    gt_distances = np.zeros((n_queries, k), dtype=np.float32)

    distance_computer = DistanceComputer(metric='l2')

    for i in range(n_queries):
        distances = distance_computer.compute(queries[i], vectors)
        top_k, top_dists = select_top_k_distances(distances, k)
        gt_ids[i] = top_k
        gt_distances[i] = top_dists

    return gt_ids, gt_distances
