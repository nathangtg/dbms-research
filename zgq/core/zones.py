import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import MiniBatchKMeans
from collections import defaultdict


@dataclass
class ZoneConfig:
    auto_zones: bool = True

    # Used only when auto_zones is False
    n_zones_coarse: int = 16
    n_zones_fine: int = 256

    use_hierarchy: bool = True
    max_levels: int = 2

    balance_iterations: int = 3
    max_imbalance_ratio: float = 5.0

    random_state: int = 42
    verbose: bool = False


@dataclass
class ZoneNode:
    zone_id: int
    level: int
    centroid: np.ndarray
    vector_ids: List[int] = field(default_factory=list)
    children: List[int] = field(default_factory=list)
    parent: Optional[int] = None
    neighbors: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.vector_ids)


class AdaptiveHierarchicalZones:
    """Two-level (coarse, fine) k-means zone partitioning, or a single flat level."""

    def __init__(self, config: Optional[ZoneConfig] = None):
        self.config = config or ZoneConfig()

        self.levels: List[List[ZoneNode]] = []
        self.n_levels: int = 0

        self.fine_centroids: Optional[np.ndarray] = None
        self.coarse_centroids: Optional[np.ndarray] = None
        self.assignments: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[int]] = {}

        # Opt-in boundary replication, kept apart from `assignments` and
        # `inverted_lists` so graph and PQ code paths are unchanged.
        # replica_assignments[i] lists the extra zones (not the primary) of vector i.
        self.n_replicas: int = 1
        self.replica_assignments: Optional[np.ndarray] = None
        self.replica_lists: Dict[int, np.ndarray] = {}

        self.coarse_to_fine: Dict[int, List[int]] = {}
        self.fine_to_coarse: Dict[int, int] = {}

        self.n_vectors: int = 0
        self.dimension: int = 0
        self.is_built: bool = False

    def _compute_optimal_zones(self, n_vectors: int) -> Tuple[int, int]:
        if n_vectors < 1000:
            n_coarse = max(2, int(np.sqrt(n_vectors) / 2))
            n_fine = max(4, int(np.sqrt(n_vectors)))
        elif n_vectors < 10000:
            n_coarse = max(4, int(np.sqrt(n_vectors) / 4))
            n_fine = max(16, int(np.sqrt(n_vectors)))
        elif n_vectors < 100000:
            n_coarse = max(8, int(n_vectors ** 0.33))
            n_fine = max(64, int(n_vectors ** 0.5))
        elif n_vectors < 1000000:
            n_coarse = max(16, int(n_vectors ** 0.25))
            n_fine = max(256, int(n_vectors ** 0.5))
        else:
            n_coarse = max(32, int(n_vectors ** 0.2))
            n_fine = max(1024, int(n_vectors ** 0.45))

        n_fine = max(n_fine, n_coarse * 2)
        return min(n_coarse, 256), min(n_fine, 16384)

    def build(self, vectors: np.ndarray) -> 'AdaptiveHierarchicalZones':
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.n_vectors, self.dimension = vectors.shape

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Building Adaptive Hierarchical Zones")
            print(f"{'='*60}")
            print(f"Vectors: {self.n_vectors:,}, Dimension: {self.dimension}")

        if self.config.auto_zones:
            n_coarse, n_fine = self._compute_optimal_zones(self.n_vectors)
        else:
            n_coarse = self.config.n_zones_coarse
            n_fine = self.config.n_zones_fine

        if self.config.verbose:
            print(f"Coarse zones: {n_coarse}, Fine zones: {n_fine}")

        if self.config.use_hierarchy and n_fine > n_coarse:
            self._build_hierarchical(vectors, n_coarse, n_fine)
        else:
            self._build_flat(vectors, n_fine)

        self._build_zone_graph()
        self.is_built = True

        if self.config.verbose:
            self._print_statistics()

        return self

    def _build_hierarchical(
        self,
        vectors: np.ndarray,
        n_coarse: int,
        n_fine: int
    ) -> None:
        if self.config.verbose:
            print("\n[1/3] Building coarse zones...")

        kmeans_coarse = MiniBatchKMeans(
            n_clusters=n_coarse,
            random_state=self.config.random_state,
            batch_size=min(1024, self.n_vectors),
            n_init=3,
            max_iter=100
        )
        coarse_assignments = kmeans_coarse.fit_predict(vectors)
        self.coarse_centroids = kmeans_coarse.cluster_centers_.astype(np.float32)

        coarse_inverted = defaultdict(list)
        for vid, cid in enumerate(coarse_assignments):
            coarse_inverted[cid].append(vid)

        coarse_zones = [
            ZoneNode(
                zone_id=cid,
                level=0,
                centroid=self.coarse_centroids[cid],
                vector_ids=coarse_inverted[cid]
            )
            for cid in range(n_coarse)
        ]
        self.levels.append(coarse_zones)

        if self.config.verbose:
            print(f"  Created {n_coarse} coarse zones")
            print("\n[2/3] Building fine zones...")

        target_fine_per_coarse = max(2, n_fine // n_coarse)

        fine_zones = []
        fine_centroids = []
        fine_zone_id = 0

        self.assignments = np.zeros(self.n_vectors, dtype=np.int32)

        for cid in range(n_coarse):
            coarse_vids = coarse_inverted[cid]
            if len(coarse_vids) == 0:
                continue

            n_fine_local = min(
                target_fine_per_coarse,
                max(1, len(coarse_vids) // 10)
            )
            n_fine_local = min(n_fine_local, len(coarse_vids))

            if n_fine_local <= 1:
                centroid = vectors[coarse_vids].mean(axis=0)
                fine_zones.append(ZoneNode(
                    zone_id=fine_zone_id,
                    level=1,
                    centroid=centroid,
                    vector_ids=coarse_vids.copy(),
                    parent=cid
                ))
                fine_centroids.append(centroid)

                self.coarse_to_fine[cid] = [fine_zone_id]
                self.assignments[coarse_vids] = fine_zone_id
                self.fine_to_coarse[fine_zone_id] = cid

                coarse_zones[cid].children.append(fine_zone_id)
                fine_zone_id += 1
                continue

            kmeans_fine = MiniBatchKMeans(
                n_clusters=n_fine_local,
                random_state=self.config.random_state + cid,
                batch_size=min(256, len(coarse_vids)),
                n_init=1,
                max_iter=50
            )
            fine_assignments = kmeans_fine.fit_predict(vectors[coarse_vids])

            self.coarse_to_fine[cid] = []

            for local_fine_id in range(n_fine_local):
                member_mask = fine_assignments == local_fine_id
                local_vids = [vid for vid, is_member in zip(coarse_vids, member_mask) if is_member]
                if len(local_vids) == 0:
                    continue

                centroid = kmeans_fine.cluster_centers_[local_fine_id]
                fine_zones.append(ZoneNode(
                    zone_id=fine_zone_id,
                    level=1,
                    centroid=centroid,
                    vector_ids=local_vids,
                    parent=cid
                ))
                fine_centroids.append(centroid)

                self.coarse_to_fine[cid].append(fine_zone_id)
                self.assignments[local_vids] = fine_zone_id
                self.fine_to_coarse[fine_zone_id] = cid

                coarse_zones[cid].children.append(fine_zone_id)
                fine_zone_id += 1

        self.levels.append(fine_zones)
        self.fine_centroids = np.array(fine_centroids, dtype=np.float32)

        for zone in fine_zones:
            self.inverted_lists[zone.zone_id] = zone.vector_ids

        self.n_levels = 2

        if self.config.verbose:
            print(f"  Created {len(fine_zones)} fine zones")

    def _build_flat(self, vectors: np.ndarray, n_zones: int) -> None:
        if self.config.verbose:
            print("\n[1/2] Building flat zones...")

        n_zones = min(n_zones, self.n_vectors)

        kmeans = MiniBatchKMeans(
            n_clusters=n_zones,
            random_state=self.config.random_state,
            batch_size=min(1024, self.n_vectors),
            n_init=3,
            max_iter=100
        )

        self.assignments = kmeans.fit_predict(vectors)
        self.fine_centroids = kmeans.cluster_centers_.astype(np.float32)
        self.coarse_centroids = self.fine_centroids

        flat_zones = []
        for zone_id in range(n_zones):
            vector_ids = np.where(self.assignments == zone_id)[0].tolist()
            flat_zones.append(ZoneNode(
                zone_id=zone_id,
                level=0,
                centroid=self.fine_centroids[zone_id],
                vector_ids=vector_ids
            ))
            self.inverted_lists[zone_id] = vector_ids
            self.coarse_to_fine[zone_id] = [zone_id]
            self.fine_to_coarse[zone_id] = zone_id

        self.levels.append(flat_zones)
        self.n_levels = 1

        if self.config.verbose:
            print(f"  Created {n_zones} zones")

    def _build_zone_graph(self) -> None:
        if self.config.verbose:
            print("\n[3/3] Building zone connectivity graph...")

        n_fine = len(self.fine_centroids)
        k_neighbors = min(8, n_fine - 1)
        if k_neighbors < 1:
            return

        centroids = self.fine_centroids
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
        norms = np.sum(centroids ** 2, axis=1)
        distances = norms[:, np.newaxis] + norms[np.newaxis, :] - 2 * centroids @ centroids.T

        for i, zone in enumerate(self.levels[-1]):
            dists = distances[i].copy()
            dists[i] = np.inf
            zone.neighbors = np.argsort(dists)[:k_neighbors].tolist()

    def _print_statistics(self) -> None:
        print(f"\n{'='*60}")
        print("Zone Statistics")
        print(f"{'='*60}")
        print(f"Levels: {self.n_levels}")

        for level_idx, level_zones in enumerate(self.levels):
            sizes = [z.size for z in level_zones]
            print(f"\nLevel {level_idx}:")
            print(f"  Zones: {len(level_zones)}")
            print(f"  Size range: [{min(sizes)}, {max(sizes)}]")
            print(f"  Mean size: {np.mean(sizes):.1f}")
            print(f"  Std: {np.std(sizes):.1f}")

            imbalance = max(sizes) / max(1, min(s for s in sizes if s > 0))
            marker = "⚠️" if imbalance > self.config.max_imbalance_ratio else "✓"
            print(f"  {marker} Imbalance ratio: {imbalance:.1f}")

    def _nearest_fine_zones(self, query: np.ndarray, n_probe: int) -> np.ndarray:
        distances = np.sum((self.fine_centroids - query) ** 2, axis=1)
        return np.argsort(distances)[:n_probe]

    def select_zones(
        self,
        query: np.ndarray,
        n_probe: int,
        use_hierarchy: bool = True
    ) -> np.ndarray:
        """Fine zone ids of the n_probe nearest zones.

        Hierarchical mode ranks only the fine zones under the max(2, n_probe // 4)
        nearest coarse zones.
        """
        if not self.is_built:
            raise RuntimeError("Zones must be built before selection")

        n_probe = min(n_probe, len(self.fine_centroids))

        if self.n_levels == 1 or not use_hierarchy:
            return self._nearest_fine_zones(query, n_probe)

        n_coarse_probe = min(max(2, n_probe // 4), len(self.coarse_centroids))
        coarse_distances = np.sum((self.coarse_centroids - query) ** 2, axis=1)
        nearest_coarse = np.argsort(coarse_distances)[:n_coarse_probe]

        candidate_fine = []
        for cid in nearest_coarse:
            if cid in self.coarse_to_fine:
                candidate_fine.extend(self.coarse_to_fine[cid])

        if len(candidate_fine) == 0:
            return self._nearest_fine_zones(query, n_probe)

        candidate_fine = np.array(candidate_fine, dtype=np.int32)
        candidate_centroids = self.fine_centroids[candidate_fine]
        fine_distances = np.sum((candidate_centroids - query) ** 2, axis=1)
        return candidate_fine[np.argsort(fine_distances)[:n_probe]]

    def build_replicas(self, vectors: np.ndarray, n_replicas: int) -> None:
        """Replicate each vector into its `n_replicas` nearest zones (SPANN style).

        The primary zone stays `assignments[i]`. The remaining `n_replicas - 1`
        nearest zones by centroid distance are stored in `replica_assignments`
        and `replica_lists`. Only ids are stored, vectors are not duplicated.
        """
        if not self.is_built:
            raise RuntimeError("Zones must be built before replication")
        n_zones = len(self.fine_centroids)
        n_replicas = int(n_replicas)
        if n_replicas < 1:
            raise ValueError("n_replicas must be >= 1")
        n_extra = min(n_replicas, n_zones) - 1
        self.n_replicas = n_replicas
        self.replica_lists = {}
        if n_extra <= 0:
            self.replica_assignments = np.zeros((len(vectors), 0), dtype=np.int32)
            return

        vectors = np.asarray(vectors, dtype=np.float32)
        cent = self.fine_centroids.astype(np.float32)
        c_sq = np.sum(cent * cent, axis=1)
        extra = np.zeros((len(vectors), n_extra), dtype=np.int32)
        chunk = 4096
        for s in range(0, len(vectors), chunk):
            blk = vectors[s:s + chunk]
            # ||q||^2 is constant per row, so it is dropped from the ranking
            d = c_sq[None, :] - 2.0 * (blk @ cent.T)
            rows = np.arange(len(blk))
            d[rows, self.assignments[s:s + chunk]] = np.inf  # exclude primary
            part = np.argpartition(d, n_extra - 1, axis=1)[:, :n_extra]
            order = np.argsort(np.take_along_axis(d, part, axis=1), axis=1)
            extra[s:s + chunk] = np.take_along_axis(part, order, axis=1)
        self.replica_assignments = extra
        self._rebuild_replica_lists()

    def _rebuild_replica_lists(self) -> None:
        extra = self.replica_assignments
        self.replica_lists = {}
        if extra is None or extra.shape[1] == 0:
            return
        flat_z = extra.reshape(-1)
        flat_v = np.repeat(np.arange(extra.shape[0], dtype=np.int64), extra.shape[1])
        order = np.argsort(flat_z, kind='stable')
        flat_z, flat_v = flat_z[order], flat_v[order]
        bounds = np.flatnonzero(np.diff(flat_z)) + 1
        for z_ids, v_ids in zip(np.split(flat_z, bounds), np.split(flat_v, bounds)):
            if len(z_ids):
                self.replica_lists[int(z_ids[0])] = v_ids

    def get_replica_overhead(self) -> Dict:
        extra = 0 if self.replica_assignments is None else int(self.replica_assignments.size)
        return {
            'n_replicas': self.n_replicas,
            'extra_ids': extra,
            'extra_bytes': extra * 4,  # int32 ids
            'extra_fraction': extra / max(1, self.n_vectors),
        }

    def get_zone_candidates(self, zone_ids) -> np.ndarray:
        """Unique vector ids in the given zones, including replica members."""
        parts = []
        for zone_id in zone_ids:
            zone_id = int(zone_id)
            parts.append(np.asarray(self.inverted_lists.get(zone_id, []), dtype=np.int64))
            replicas = self.replica_lists.get(zone_id)
            if replicas is not None:
                parts.append(replicas)
        if not parts:
            return np.zeros(0, dtype=np.int64)
        return np.unique(np.concatenate(parts))

    def get_zone_vectors(self, zone_id: int) -> List[int]:
        return self.inverted_lists.get(zone_id, [])

    def get_zone_neighbors(self, zone_id: int) -> List[int]:
        if zone_id < len(self.levels[-1]):
            return self.levels[-1][zone_id].neighbors
        return []

    def get_residual(
        self,
        vectors: np.ndarray,
        zone_ids: np.ndarray
    ) -> np.ndarray:
        return vectors - self.fine_centroids[zone_ids]

    def save_state(self) -> Dict:
        return {
            'config': {
                'auto_zones': self.config.auto_zones,
                'n_zones_coarse': self.config.n_zones_coarse,
                'n_zones_fine': self.config.n_zones_fine,
                'use_hierarchy': self.config.use_hierarchy,
                'random_state': self.config.random_state
            },
            'n_vectors': self.n_vectors,
            'dimension': self.dimension,
            'n_levels': self.n_levels,
            'coarse_centroids': self.coarse_centroids,
            'fine_centroids': self.fine_centroids,
            'assignments': self.assignments,
            'inverted_lists': {int(k): list(v) for k, v in self.inverted_lists.items()},
            'coarse_to_fine': {int(k): list(v) for k, v in self.coarse_to_fine.items()},
            'fine_to_coarse': {int(k): int(v) for k, v in self.fine_to_coarse.items()},
            'n_replicas': self.n_replicas,
            'replica_assignments': self.replica_assignments
        }

    @classmethod
    def load_state(cls, state: Dict) -> 'AdaptiveHierarchicalZones':
        config = ZoneConfig(**state['config'])
        zones = cls(config)
    
        zones.n_vectors = state['n_vectors']
        zones.dimension = state['dimension']
        zones.n_levels = state['n_levels']
        zones.coarse_centroids = state['coarse_centroids']
        zones.fine_centroids = state['fine_centroids']
        zones.assignments = state['assignments']
        zones.inverted_lists = {int(k): list(v) for k, v in state['inverted_lists'].items()}
        zones.coarse_to_fine = {int(k): list(v) for k, v in state['coarse_to_fine'].items()}
        zones.fine_to_coarse = {int(k): int(v) for k, v in state['fine_to_coarse'].items()}
        zones.n_replicas = state.get('n_replicas', 1)
        zones.replica_assignments = state.get('replica_assignments')
        zones._rebuild_replica_lists()
        zones.is_built = True
        return zones
