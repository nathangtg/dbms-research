import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hnswlib
import numpy as np

from zgq.core.build_order import STRATEGIES, insertion_order


@dataclass
class GraphConfig:
    # HNSW params
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64

    use_zone_guidance: bool = True
    zone_priority_factor: float = 1.5
    max_zone_expansion: int = 3

    use_entry_points: bool = True
    n_entry_points_per_zone: int = 3

    space: str = "l2"  # 'l2', 'ip', or 'cosine'

    random_seed: int = 42

    insertion_order: str = "zone"
    skeleton_fraction: float = 0.05


class ZoneGuidedGraph:
    def __init__(self, config: Optional[GraphConfig] = None):
        self.config = config or GraphConfig()

        self.hnsw: Optional[hnswlib.Index] = None

        self.zone_assignments: Optional[np.ndarray] = None
        self.zone_entry_points: Dict[int, List[int]] = {}
        self.zone_centroids: Optional[np.ndarray] = None

        self.n_vectors: int = 0
        self.dimension: int = 0
        self.n_zones: int = 0
        self.is_built: bool = False

    def build(
        self,
        vectors: np.ndarray,
        zone_assignments: np.ndarray,
        zone_centroids: np.ndarray,
    ) -> "ZoneGuidedGraph":
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.n_vectors, self.dimension = vectors.shape
        self.zone_assignments = zone_assignments.copy()
        self.zone_centroids = zone_centroids.copy()
        self.n_zones = len(zone_centroids)

        if self.config.insertion_order not in STRATEGIES:
            raise ValueError(
                f"insertion_order must be one of {STRATEGIES}, "
                f"got {self.config.insertion_order!r}"
            )

        self._build_hnsw(vectors)

        if self.config.use_entry_points:
            self._compute_zone_entry_points(vectors)

        self.is_built = True
        return self

    def _build_hnsw(self, vectors: np.ndarray) -> None:
        self.hnsw = hnswlib.Index(space=self.config.space, dim=self.dimension)

        self.hnsw.init_index(
            max_elements=self.n_vectors,
            ef_construction=self.config.ef_construction,
            M=self.config.M,
            random_seed=self.config.random_seed,
        )

        order = insertion_order(
            self.config.insertion_order,
            vectors,
            self.zone_assignments,
            self.zone_centroids,
            self.config.random_seed,
            skeleton_fraction=self.config.skeleton_fraction,
        )
        self.hnsw.add_items(vectors[order], order)
        self.hnsw.set_ef(self.config.ef_search)

    def _compute_zone_entry_points(self, vectors: np.ndarray) -> None:
        n_entry = self.config.n_entry_points_per_zone

        for zone_id in range(self.n_zones):
            zone_vids = np.where(self.zone_assignments == zone_id)[0]
            if len(zone_vids) == 0:
                self.zone_entry_points[zone_id] = []
                continue

            centroid = self.zone_centroids[zone_id]
            distances = np.sum((vectors[zone_vids] - centroid) ** 2, axis=1)
            closest = np.argsort(distances)[: min(n_entry, len(zone_vids))]
            self.zone_entry_points[zone_id] = zone_vids[closest].tolist()

    def search(
        self,
        query: np.ndarray,
        k: int,
        selected_zones: Optional[np.ndarray] = None,
        ef_search: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_built:
            raise RuntimeError("Graph must be built before searching")

        if ef_search is not None:
            self.hnsw.set_ef(ef_search)

        if selected_zones is not None and self.config.use_zone_guidance:
            return self._search_zone_guided(query, k, selected_zones)
        return self._search_standard(query, k)

    def _search_standard(
        self, query: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = query.reshape(1, -1).astype(np.float32)
        ids, distances = self.hnsw.knn_query(query, k=min(k, self.n_vectors))
        return ids[0], distances[0]

    def _search_zone_guided(
        self, query: np.ndarray, k: int, selected_zones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        query = query.reshape(1, -1).astype(np.float32)

        ids, distances = self.hnsw.knn_query(query, k=min(k * 3, self.n_vectors))
        ids, distances = ids[0], distances[0]

        if len(selected_zones) < self.n_zones // 2:
            ids, distances = self._rerank_with_zone_priority(
                ids, distances, selected_zones, k
            )

        return ids[:k], distances[:k]

    def _rerank_with_zone_priority(
        self, ids: np.ndarray, distances: np.ndarray, selected_zones: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        zone_set = set(selected_zones.tolist())

        scores = []
        for vid, dist in zip(ids, distances):
            in_selected_zone = self.zone_assignments[vid] in zone_set
            adjusted_dist = (
                dist / self.config.zone_priority_factor if in_selected_zone else dist
            )
            scores.append((adjusted_dist, dist, vid))

        scores.sort(key=lambda score: score[0])

        result_ids = np.array([s[2] for s in scores[:k]], dtype=np.int32)
        result_dists = np.array([s[1] for s in scores[:k]], dtype=np.float32)
        return result_ids, result_dists

    def search_batch(
        self, queries: np.ndarray, k: int, ef_search: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_built:
            raise RuntimeError("Graph must be built before searching")

        if ef_search is not None:
            self.hnsw.set_ef(ef_search)

        queries = np.ascontiguousarray(queries.astype(np.float32))
        return self.hnsw.knn_query(queries, k=min(k, self.n_vectors))

    def set_ef(self, ef_search: int) -> None:
        if self.hnsw is not None:
            self.hnsw.set_ef(ef_search)
            self.config.ef_search = ef_search

    def get_stats(self) -> Dict:
        return {
            "n_vectors": self.n_vectors,
            "dimension": self.dimension,
            "n_zones": self.n_zones,
            "M": self.config.M,
            "ef_construction": self.config.ef_construction,
            "ef_search": self.config.ef_search,
            "use_zone_guidance": self.config.use_zone_guidance,
        }

    def save_state(self) -> Dict:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            tmp_path = f.name

        try:
            self.hnsw.save_index(tmp_path)
            with open(tmp_path, "rb") as f:
                hnsw_bytes = f.read()
        finally:
            os.unlink(tmp_path)

        return {
            "config": {
                "M": self.config.M,
                "ef_construction": self.config.ef_construction,
                "ef_search": self.config.ef_search,
                "use_zone_guidance": self.config.use_zone_guidance,
                "zone_priority_factor": self.config.zone_priority_factor,
                "space": self.config.space,
                "insertion_order": self.config.insertion_order,
                "skeleton_fraction": self.config.skeleton_fraction,
            },
            "n_vectors": self.n_vectors,
            "dimension": self.dimension,
            "n_zones": self.n_zones,
            "zone_assignments": self.zone_assignments,
            "zone_centroids": self.zone_centroids,
            "zone_entry_points": {
                int(k): list(v) for k, v in self.zone_entry_points.items()
            },
            "hnsw_bytes": hnsw_bytes,
        }

    @classmethod
    def load_state(cls, state: Dict) -> "ZoneGuidedGraph":
        config = GraphConfig(**state["config"])
        graph = cls(config)

        graph.n_vectors = state["n_vectors"]
        graph.dimension = state["dimension"]
        graph.n_zones = state["n_zones"]
        graph.zone_assignments = state["zone_assignments"]
        graph.zone_centroids = state["zone_centroids"]
        graph.zone_entry_points = {
            int(k): list(v) for k, v in state["zone_entry_points"].items()
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
            f.write(state["hnsw_bytes"])
            tmp_path = f.name

        try:
            graph.hnsw = hnswlib.Index(space=config.space, dim=graph.dimension)
            graph.hnsw.load_index(tmp_path, max_elements=graph.n_vectors)
            graph.hnsw.set_ef(config.ef_search)
        finally:
            os.unlink(tmp_path)

        graph.is_built = True
        return graph
