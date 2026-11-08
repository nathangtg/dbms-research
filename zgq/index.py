import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np

from zgq.core.zones import AdaptiveHierarchicalZones, ZoneConfig
from zgq.core.graph import ZoneGuidedGraph, GraphConfig
from zgq.core.build_order import STRATEGIES
from zgq.core.quantization import ResidualProductQuantizer, RPQConfig, suggest_pq_params
from zgq.core.distances import DistanceComputer
from zgq.core import compressed as _cmp_mod


def _batch_rerank_fast(queries, vectors, candidate_ids, k):
    n_queries = queries.shape[0]
    n_candidates = candidate_ids.shape[1]

    result_ids = np.zeros((n_queries, k), dtype=np.int64)
    result_distances = np.zeros((n_queries, k), dtype=np.float32)

    if n_candidates <= k:
        result_ids[:, :n_candidates] = candidate_ids
        candidate_vectors = vectors[candidate_ids]
        diff = queries[:, np.newaxis, :] - candidate_vectors
        result_distances[:, :n_candidates] = np.einsum('ijk,ijk->ij', diff, diff)
        return result_ids, result_distances

    candidate_vectors = vectors[candidate_ids]

    # ||q - c||^2 = ||q||^2 - 2*q.c + ||c||^2
    q_norm_sq = np.einsum('ij,ij->i', queries, queries)[:, np.newaxis]
    c_norm_sq = np.einsum('ijk,ijk->ij', candidate_vectors, candidate_vectors)
    qc_dot = np.einsum('ij,ikj->ik', queries, candidate_vectors)

    distances = q_norm_sq - 2 * qc_dot + c_norm_sq
    distances = distances.astype(np.float32)

    rows = np.arange(n_queries)[:, np.newaxis]
    if k < n_candidates // 2:
        top_k_indices = np.argpartition(distances, k-1, axis=1)[:, :k]
        top_k_dists = distances[rows, top_k_indices]
        sort_idx = np.argsort(top_k_dists, axis=1)
        top_k_indices = np.take_along_axis(top_k_indices, sort_idx, axis=1)
    else:
        top_k_indices = np.argsort(distances, axis=1)[:, :k]

    result_ids = candidate_ids[rows, top_k_indices]
    result_distances = distances[rows, top_k_indices]

    return result_ids, result_distances


def _map_queries(search_one, n_queries, n_jobs):
    if n_jobs == 1:
        return [search_one(i) for i in range(n_queries)]
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        return list(executor.map(search_one, range(n_queries)))


@dataclass
class ZGQConfig:
    n_zones: Union[str, int] = 'auto'
    use_hierarchy: bool = True

    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64

    use_pq: bool = True
    pq_m: Union[str, int] = 'auto'
    pq_bits: int = 8
    use_residual_pq: bool = True

    use_simd: bool = True
    use_zone_guidance: bool = True

    metric: str = 'l2'

    verbose: bool = False
    random_state: int = 42

    insertion_order: str = 'zone'
    skeleton_fraction: float = 0.05
    zone_replicas: int = 1
    adaptive_probe: bool = False
    min_probe: int = 1
    max_probe: int = 32
    probe_ratio: float = 1.3
    search_mode: str = 'graph'
    compressed_search: bool = False
    rerank_budget: int = 0
    drop_full_vectors: bool = False
    compressed_entry_zones: int = 4

    def __post_init__(self):
        if self.insertion_order not in STRATEGIES:
            raise ValueError(f"insertion_order must be one of {STRATEGIES}")
        if not 0.0 <= self.skeleton_fraction <= 1.0:
            raise ValueError("skeleton_fraction must be in [0, 1]")
        if self.search_mode not in ('graph', 'zone_scan', 'global_entry'):
            raise ValueError(
                "search_mode must be 'graph', 'zone_scan' or 'global_entry'"
            )
        if self.zone_replicas < 1:
            raise ValueError("zone_replicas must be >= 1")
        if not 1 <= self.min_probe <= self.max_probe:
            raise ValueError("require 1 <= min_probe <= max_probe")
        if self.probe_ratio < 1.0:
            raise ValueError("probe_ratio must be >= 1.0")
        if self.rerank_budget < 0:
            raise ValueError("rerank_budget must be >= 0")
        if self.compressed_entry_zones < 1:
            raise ValueError("compressed_entry_zones must be >= 1")
        if self.drop_full_vectors and not self.compressed_search:
            raise ValueError("drop_full_vectors requires compressed_search=True")
        if self.compressed_search and not self.use_pq:
            raise ValueError("compressed_search requires use_pq=True")


class ZGQIndex:
    def __init__(self, config: Optional[ZGQConfig] = None):
        self.config = config or ZGQConfig()

        self.zones: Optional[AdaptiveHierarchicalZones] = None
        self.graph: Optional[ZoneGuidedGraph] = None
        self.pq: Optional[ResidualProductQuantizer] = None
        self.pq_codes: Optional[np.ndarray] = None

        self.vectors: Optional[np.ndarray] = None
        self.vector_norms: Optional[np.ndarray] = None

        self.distance_computer = DistanceComputer(metric=self.config.metric)

        self.n_vectors: int = 0
        self.dimension: int = 0
        self.is_built: bool = False

        self.build_stats: Dict = {}
        self.last_batch_stats: Dict = {}
        self._cmp: Optional[Dict] = None

    def build(self, vectors: np.ndarray) -> 'ZGQIndex':
        build_start = time.time()

        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.vectors = vectors
        self.n_vectors, self.dimension = vectors.shape

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("Building ZGQ v8 Index")
            print(f"{'='*60}")
            print(f"Vectors: {self.n_vectors:,}")
            print(f"Dimension: {self.dimension}")
            print()

        self._auto_configure()

        step_start = time.time()
        if self.config.verbose:
            print("[1/4] Building Adaptive Hierarchical Zones...")

        zone_config = ZoneConfig(
            auto_zones=(self.config.n_zones == 'auto'),
            use_hierarchy=self.config.use_hierarchy,
            random_state=self.config.random_state,
            verbose=self.config.verbose
        )

        if self.config.n_zones != 'auto':
            zone_config.n_zones_fine = self.config.n_zones

        self.zones = AdaptiveHierarchicalZones(zone_config)
        self.zones.build(vectors)

        if self.config.zone_replicas > 1:
            self.zones.build_replicas(vectors, self.config.zone_replicas)
            self.build_stats['replica_overhead'] = self.zones.get_replica_overhead()

        self.build_stats['zones_time'] = time.time() - step_start
        if self.config.verbose:
            print(f"  ✓ Completed in {self.build_stats['zones_time']:.2f}s")

        step_start = time.time()
        if self.config.verbose:
            print("\n[2/4] Building Zone-Guided Graph...")

        graph_config = GraphConfig(
            M=self.config.M,
            ef_construction=self.config.ef_construction,
            ef_search=self.config.ef_search,
            use_zone_guidance=self.config.use_zone_guidance,
            space=self.config.metric,
            random_seed=self.config.random_state,
            insertion_order=self.config.insertion_order,
            skeleton_fraction=self.config.skeleton_fraction
        )

        self.graph = ZoneGuidedGraph(graph_config)
        self.graph.build(
            vectors,
            self.zones.assignments,
            self.zones.fine_centroids
        )

        self.build_stats['graph_time'] = time.time() - step_start
        if self.config.verbose:
            print(f"  ✓ Completed in {self.build_stats['graph_time']:.2f}s")

        step_start = time.time()
        if self.config.use_pq:
            if self.config.verbose:
                print("\n[3/4] Training Residual Product Quantization...")

            if self.config.pq_m == 'auto':
                pq_m, _ = suggest_pq_params(self.dimension)
            else:
                pq_m = self.config.pq_m

            pq_config = RPQConfig(
                m=pq_m,
                n_bits=self.config.pq_bits,
                use_residuals=self.config.use_residual_pq,
                random_state=self.config.random_state
            )

            self.pq = ResidualProductQuantizer(pq_config)
            self.pq.train(
                vectors,
                zone_centroids=self.zones.fine_centroids,
                zone_assignments=self.zones.assignments
            )

            if self.config.verbose:
                print("  Encoding vectors...")

            self.pq_codes = self.pq.encode(
                vectors,
                zone_centroids=self.zones.fine_centroids,
                zone_assignments=self.zones.assignments
            )

            self.build_stats['pq_time'] = time.time() - step_start
            if self.config.verbose:
                mem_info = self.pq.get_memory_usage(self.n_vectors)
                print(f"  Compression ratio: {mem_info['compression_ratio']:.1f}x")
                print(f"  ✓ Completed in {self.build_stats['pq_time']:.2f}s")
        else:
            self.build_stats['pq_time'] = 0
            if self.config.verbose:
                print("\n[3/4] Product Quantization: SKIPPED")

        step_start = time.time()
        if self.config.verbose:
            print("\n[4/4] Precomputing vector norms...")

        self.vector_norms = DistanceComputer.precompute_norms(vectors)

        self.build_stats['norms_time'] = time.time() - step_start
        if self.config.verbose:
            print(f"  ✓ Completed in {self.build_stats['norms_time']:.2f}s")

        if self.config.compressed_search:
            step_start = time.time()
            self._build_compressed()
            self.build_stats['compressed_time'] = time.time() - step_start

        self.is_built = True
        self.build_stats['total_time'] = time.time() - build_start

        if self.config.verbose:
            print(f"\n{'='*60}")
            print("✓ ZGQ Index Built Successfully")
            print(f"  Total time: {self.build_stats['total_time']:.2f}s")
            print(f"{'='*60}\n")

        return self

    def _auto_configure(self) -> None:
        if self.n_vectors < 10000:
            self.config.ef_search = min(self.config.ef_search, 32)
        elif self.n_vectors > 1000000:
            self.config.ef_search = max(self.config.ef_search, 100)

        if self.dimension > 256 and self.config.M < 24:
            self.config.M = 24

        if self.config.verbose and self.config.n_zones == 'auto':
            print("Auto-configuring parameters...")

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        use_pq_rerank: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.config.compressed_search:
            if not self.is_built:
                raise RuntimeError("Index must be built before searching")
            ids, dists = self._compressed_batch_search(
                np.asarray(query, dtype=np.float32)[None, :], k, ef_search
            )
            return ids[0], dists[0]
        ids, distances, _ = self._search_impl(query, k, n_probe, ef_search, use_pq_rerank)
        return ids, distances

    def _build_compressed(self) -> None:
        config = self.config
        links, _ = _cmp_mod.extract_level0_links(self.graph.hnsw, self.dimension)
        codebooks = _cmp_mod.effective_codebooks(self.pq)
        residual = bool(self.pq.config.use_residuals)
        centroids = np.ascontiguousarray(self.zones.fine_centroids, dtype=np.float32)
        n_zones = len(centroids)
        entry_ids = np.full(n_zones, -1, dtype=np.int32)
        for zone, points in self.graph.zone_entry_points.items():
            if len(points):
                entry_ids[int(zone)] = points[0]
        zone_dtype = np.uint16 if n_zones < 65536 else np.int32
        self._cmp = {
            'links': links,
            'cw': codebooks,
            'residual': residual,
            'centroids': centroids,
            'entry_ids': entry_ids,
            'cross': _cmp_mod.zone_cross_table(centroids, codebooks) if residual else None,
            'zone_of': (self.zones.assignments.astype(zone_dtype) if residual
                        else np.zeros(0, dtype=zone_dtype)),
        }
        if config.drop_full_vectors:
            self.graph = None
            self.vector_norms = None
            if config.rerank_budget == 0:
                self.vectors = None

    def _tables(self, queries: np.ndarray):
        compressed = self._cmp
        adc_tables, zone_distances = _cmp_mod.query_tables(
            queries, compressed['cw'], compressed['centroids'])
        if compressed['residual']:
            query_terms = zone_distances
        else:
            query_terms = np.einsum('ij,ij->i', queries, queries)[:, None].astype(np.float32)
        return adc_tables, query_terms, zone_distances

    def _entries(
        self, queries: np.ndarray, zone_distances: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if zone_distances is None:
            zone_distances = self._tables(queries)[2]
        n_entry_zones = min(self.config.compressed_entry_zones, zone_distances.shape[1])
        nearest = np.argpartition(zone_distances, n_entry_zones - 1, axis=1)[:, :n_entry_zones]
        return self._cmp['entry_ids'][nearest].astype(np.int64)

    def _compressed_batch_search(self, queries, k, ef_search=None, chunk=256):
        config = self.config
        ef = ef_search if ef_search is not None else config.ef_search
        ef = max(ef, k, config.rerank_budget)
        compressed = self._cmp
        n_queries = len(queries)
        out_ids = np.zeros((n_queries, k), dtype=np.int64)
        out_distances = np.zeros((n_queries, k), dtype=np.float32)
        for start in range(0, n_queries, chunk):
            chunk_queries = np.ascontiguousarray(queries[start:start + chunk], dtype=np.float32)
            adc_tables, query_terms, zone_distances = self._tables(chunk_queries)
            ids, dists = _cmp_mod.beam_search_batch(
                compressed['links'], self.pq_codes, compressed['zone_of'],
                adc_tables, query_terms, compressed['cross'],
                self._entries(chunk_queries, zone_distances), ef)
            chunk_slice = slice(start, start + chunk)
            if config.rerank_budget > 0 and self.vectors is not None:
                candidates = ids[:, :min(config.rerank_budget, ef)]
                safe_ids = np.where(candidates >= 0, candidates, 0)
                diff = chunk_queries[:, None, :] - self.vectors[safe_ids]
                exact = np.einsum('ijk,ijk->ij', diff, diff)
                exact[candidates < 0] = np.inf
                order = np.argsort(exact, axis=1)[:, :k]
                out_ids[chunk_slice] = np.take_along_axis(candidates, order, axis=1)
                out_distances[chunk_slice] = np.take_along_axis(exact, order, axis=1)
            else:
                out_ids[chunk_slice] = ids[:, :k]
                out_distances[chunk_slice] = dists[:, :k]
        return out_ids, out_distances

    def compressed_memory_bytes(self) -> Dict:
        compressed = self._cmp
        comps = {
            'level0_links': compressed['links'].nbytes,
            'pq_codes': self.pq_codes.nbytes,
            'pq_codebooks': self.pq.codebooks.nbytes,
            'zone_entry_points': compressed['entry_ids'].nbytes,
            'zone_centroids': compressed['centroids'].nbytes,
        }
        if compressed['residual']:
            comps['zone_ids'] = compressed['zone_of'].nbytes
            comps['zone_codebook_cross_table'] = compressed['cross'].nbytes
        if self.vectors is not None and self.config.rerank_budget > 0:
            comps['raw_vectors'] = self.vectors.nbytes
        total = sum(comps.values())
        return {'components': comps, 'total_bytes': total,
                'bytes_per_vector': total / self.n_vectors}

    def _choose_zones(self, query: np.ndarray, n_probe: int) -> np.ndarray:
        config = self.config
        if not config.adaptive_probe:
            return self.zones.select_zones(query, n_probe)
        zones = self.zones.select_zones(query, config.max_probe)
        squared_distances = np.sum((self.zones.fine_centroids[zones] - query) ** 2, axis=1)
        # squared L2, while probe_ratio applies to plain distances
        within_ratio = squared_distances <= (config.probe_ratio ** 2) * squared_distances[0]
        n_selected = int(np.sum(within_ratio))
        n_selected = max(config.min_probe, min(n_selected, config.max_probe, len(zones)))
        return zones[:n_selected]

    def _search_impl(self, query, k, n_probe, ef_search, use_pq_rerank):
        if not self.is_built:
            raise RuntimeError("Index must be built before searching")

        query = np.ascontiguousarray(query.astype(np.float32))

        if query.shape[0] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[0]} != index dimension {self.dimension}"
            )

        selected_zones = self._choose_zones(query, n_probe)
        n_probed = len(selected_zones)

        if ef_search is not None:
            self.graph.set_ef(ef_search)

        mode = self.config.search_mode
        if mode == 'zone_scan':
            # IVF-style: candidates are all members (and replicas) of probed zones
            candidate_ids = self.zones.get_zone_candidates(selected_zones)
            if len(candidate_ids) == 0:
                candidate_ids, _ = self.graph.search(query, k, selected_zones)
        else:
            k_candidates = min(k * n_probed, self.n_vectors)
            candidate_ids, candidate_dists = self.graph.search(
                query, k_candidates, selected_zones
            )
            if mode == 'global_entry':
                # Seed the candidate set with the entry points of probed zones
                seeds = [self.graph.zone_entry_points.get(int(z), []) for z in selected_zones]
                seeds = np.array([i for g in seeds for i in g], dtype=np.int64)
                candidate_ids = np.unique(
                    np.concatenate([np.asarray(candidate_ids, dtype=np.int64), seeds])
                )

        if self.config.use_pq and use_pq_rerank and len(candidate_ids) > k:
            query_zone = selected_zones[0]
            zone_centroid = self.zones.fine_centroids[query_zone]

            dist_table = self.pq.compute_distance_table(query, zone_centroid)
            pq_distances = self.pq.asymmetric_distance(
                self.pq_codes[candidate_ids], dist_table
            )

            k_rerank = min(k * 2, len(candidate_ids))
            top_indices = np.argpartition(pq_distances, k_rerank - 1)[:k_rerank]
            candidate_ids = candidate_ids[top_indices]

        candidate_vectors = self.vectors[candidate_ids]
        exact_distances = self.distance_computer.compute(
            query, candidate_vectors, self.vector_norms[candidate_ids]
        )

        if len(exact_distances) > k:
            top_k = np.argpartition(exact_distances, k - 1)[:k]
            top_k = top_k[np.argsort(exact_distances[top_k])]
        else:
            top_k = np.argsort(exact_distances)

        return candidate_ids[top_k], exact_distances[top_k], n_probed

    def batch_zone_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        use_pq_rerank: bool = True,
        n_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_built:
            raise RuntimeError("Index must be built before searching")
        queries = np.ascontiguousarray(queries.astype(np.float32))
        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {queries.shape[1]} != index dimension {self.dimension}"
            )
        n_queries = len(queries)
        if ef_search is not None:
            self.graph.set_ef(ef_search)

        def search_one(i):
            return self._search_impl(queries[i], k, n_probe, None, use_pq_rerank)

        results = _map_queries(search_one, n_queries, n_jobs)

        all_ids = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        probes = np.zeros(n_queries, dtype=np.int32)
        for i, (ids, dists, n_probed) in enumerate(results):
            n_results = min(k, len(ids))
            all_ids[i, :n_results] = ids[:n_results]
            all_distances[i, :n_results] = dists[:n_results]
            probes[i] = n_probed
        self.last_batch_stats = {
            'mean_probes': float(probes.mean()) if n_queries else 0.0,
            'probes': probes,
        }
        return all_ids, all_distances

    def batch_search(
        self,
        queries: np.ndarray,
        k: int = 10,
        n_probe: int = 8,
        ef_search: Optional[int] = None,
        n_jobs: int = 1,
        fast_mode: bool = True,
        rerank_factor: int = 3,
        use_zone_boost: bool = True,
        turbo_mode: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_built:
            raise RuntimeError("Index must be built before searching")

        queries = np.ascontiguousarray(queries.astype(np.float32))
        n_queries = len(queries)

        if queries.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {queries.shape[1]} != index dimension {self.dimension}"
            )

        if self.config.compressed_search:
            return self._compressed_batch_search(queries, k, ef_search)

        if turbo_mode and self.graph is not None:
            effective_ef = ef_search if ef_search is not None else self.config.ef_search
            turbo_ef = int(effective_ef * 1.1)
            self.graph.set_ef(turbo_ef)
            return self.graph.search_batch(queries, k, turbo_ef)

        if fast_mode and self.graph is not None:
            effective_ef = ef_search if ef_search is not None else self.config.ef_search
            scaled_ef = max(effective_ef, k * rerank_factor)
            self.graph.set_ef(scaled_ef)

            k_search = min(k * rerank_factor, self.n_vectors)
            all_ids, all_distances = self.graph.search_batch(queries, k_search, scaled_ef)
            if k_search > k:
                return _batch_rerank_fast(queries, self.vectors, all_ids, k)
            return all_ids, all_distances

        def search_one(i):
            return self.search(queries[i], k, n_probe, ef_search)

        results = _map_queries(search_one, n_queries, n_jobs)

        all_ids = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        for i, (ids, distances) in enumerate(results):
            n_results = min(k, len(ids))
            all_ids[i, :n_results] = ids[:n_results]
            all_distances[i, :n_results] = distances[:n_results]
        return all_ids, all_distances

    def get_stats(self) -> Dict:
        stats = {
            'n_vectors': self.n_vectors,
            'dimension': self.dimension,
            'is_built': self.is_built,
            'config': {
                'M': self.config.M,
                'ef_construction': self.config.ef_construction,
                'ef_search': self.config.ef_search,
                'use_pq': self.config.use_pq,
                'use_hierarchy': self.config.use_hierarchy,
                'metric': self.config.metric
            }
        }

        if self.is_built:
            stats['build_stats'] = self.build_stats
            stats['n_zones'] = len(self.zones.fine_centroids) if self.zones else 0
            if self.zones and self.zones.n_replicas > 1:
                stats['replica_overhead'] = self.zones.get_replica_overhead()

            if self.pq:
                stats['pq_info'] = self.pq.get_memory_usage(self.n_vectors)

        return stats

    def save(self, filepath: str) -> None:
        if not self.is_built:
            raise RuntimeError("Cannot save index that hasn't been built")
        if self.config.drop_full_vectors:
            raise RuntimeError(
                "Cannot save an index built with drop_full_vectors=True "
                "(the hnswlib graph was released)"
            )

        state = {
            'version': '8.0.0',
            'config': {
                'n_zones': self.config.n_zones,
                'use_hierarchy': self.config.use_hierarchy,
                'M': self.config.M,
                'ef_construction': self.config.ef_construction,
                'ef_search': self.config.ef_search,
                'use_pq': self.config.use_pq,
                'pq_m': self.config.pq_m,
                'pq_bits': self.config.pq_bits,
                'use_residual_pq': self.config.use_residual_pq,
                'metric': self.config.metric,
                'random_state': self.config.random_state,
                'insertion_order': self.config.insertion_order,
                'skeleton_fraction': self.config.skeleton_fraction,
                'zone_replicas': self.config.zone_replicas,
                'adaptive_probe': self.config.adaptive_probe,
                'min_probe': self.config.min_probe,
                'max_probe': self.config.max_probe,
                'probe_ratio': self.config.probe_ratio,
                'search_mode': self.config.search_mode
            },
            'data': {
                'vectors': self.vectors,
                'vector_norms': self.vector_norms,
                'n_vectors': self.n_vectors,
                'dimension': self.dimension
            },
            'zones': self.zones.save_state() if self.zones else None,
            'graph': self.graph.save_state() if self.graph else None,
            'pq': self.pq.save_state() if self.pq else None,
            'pq_codes': self.pq_codes,
            'build_stats': self.build_stats
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self.config.verbose:
            print(f"Index saved to {filepath}")

    @classmethod
    def load(cls, filepath: str, verbose: bool = False) -> 'ZGQIndex':
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        config = ZGQConfig(**state['config'], verbose=verbose)
        index = cls(config)

        index.vectors = state['data']['vectors']
        index.vector_norms = state['data']['vector_norms']
        index.n_vectors = state['data']['n_vectors']
        index.dimension = state['data']['dimension']

        if state['zones']:
            index.zones = AdaptiveHierarchicalZones.load_state(state['zones'])

        if state['graph']:
            index.graph = ZoneGuidedGraph.load_state(state['graph'])

        if state['pq']:
            index.pq = ResidualProductQuantizer.load_state(state['pq'])
            index.pq_codes = state['pq_codes']

        index.build_stats = state.get('build_stats', {})
        index.is_built = True

        if verbose:
            print(f"Index loaded from {filepath}")

        return index
