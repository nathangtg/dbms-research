"""
Insertion-order strategies for the zone-partitioned HNSW build.

Pure functions: given vectors and zone information they return a permutation
of ``range(n)`` (int64) giving the order in which ids are added to hnswlib.

Strategies
----------
zone        zones in id order, within a zone ascending squared distance to the
            zone centroid (the library default, bit-identical to the legacy loop)
random      ``RandomState(seed).permutation(n)``
lid_desc    descending local intrinsic dimension (stable)
lid_asc     ascending local intrinsic dimension (stable)
skeleton    a seeded random sample of ``skeleton_fraction * n`` ids first, in
            random order, then the remaining ids in zone order
roundrobin  one vector per zone in cyclic order, within-zone order by ascending
            distance to the centroid
"""

from typing import Optional

import numpy as np

STRATEGIES = ('zone', 'random', 'lid_desc', 'lid_asc', 'skeleton', 'roundrobin')


def estimate_lid(vectors: np.ndarray, k: int = 20,
                 max_reference: int = 200_000, seed: int = 0) -> np.ndarray:
    """
    Levina-Bickel maximum-likelihood estimate of local intrinsic dimension.

    For each point, with T_1 <= ... <= T_k the distances to its k nearest
    neighbors (self excluded), LID = [ mean_{j<k} log(T_k / T_j) ]^-1.

    Approximation: when n > max_reference, neighbors are searched in a seeded
    random reference subset of max_reference points instead of all points.
    Self matches are excluded either way. Distances to a subsample are larger,
    so absolute values shift, but the ranking used for ordering is what matters.
    """
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    n = len(x)
    if k < 2:
        raise ValueError("k must be >= 2")
    if n <= k:
        raise ValueError("need more than k vectors to estimate LID")

    ref_ids = np.arange(n)
    if n > max_reference:
        ref_ids = np.sort(np.random.RandomState(seed).choice(
            n, size=max_reference, replace=False))
    ref = np.ascontiguousarray(x[ref_ids])
    in_ref = np.full(n, -1, dtype=np.int64)
    in_ref[ref_ids] = np.arange(len(ref_ids))

    kq = k + 1
    try:
        import faiss
        index = faiss.IndexFlatL2(x.shape[1])
        index.add(ref)
        d2, nn = index.search(x, kq)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=kq).fit(ref)
        dist, nn = nbrs.kneighbors(x)
        d2 = dist.astype(np.float64) ** 2

    d2 = np.maximum(d2.astype(np.float64), 0.0)
    # Drop the self match (at most one per row); if self is not among the kq
    # results (ties), drop the farthest instead.
    self_pos = np.where(nn == in_ref[:, None], np.arange(kq)[None, :], kq)
    drop = np.minimum(self_pos.min(axis=1), kq - 1)
    keep = np.ones_like(d2, dtype=bool)
    keep[np.arange(n), drop] = False
    # Rows whose self is not in the reference subset keep k+1 candidates; trim.
    d2 = d2[keep].reshape(n, kq - 1)

    r = np.sqrt(np.maximum(d2, 1e-24))
    log_ratio = np.log(r[:, -1:] / r[:, :-1])
    mean = np.maximum(log_ratio.mean(axis=1), 1e-12)
    return 1.0 / mean


def _zone_sorted_lists(vectors, zone_assignments, zone_centroids):
    lists = []
    for zone_id in range(len(zone_centroids)):
        zone_vids = np.where(zone_assignments == zone_id)[0]
        if len(zone_vids) == 0:
            continue
        d = np.sum((vectors[zone_vids] - zone_centroids[zone_id]) ** 2, axis=1)
        lists.append(zone_vids[np.argsort(d)])
    return lists


def _zone_order(vectors, zone_assignments, zone_centroids) -> np.ndarray:
    if zone_centroids is None or len(zone_centroids) == 0:
        return np.arange(len(vectors), dtype=np.int64)
    lists = _zone_sorted_lists(vectors, zone_assignments, zone_centroids)
    return np.concatenate(lists).astype(np.int64)


def insertion_order(name: str, vectors: np.ndarray,
                    zone_assignments: Optional[np.ndarray],
                    zone_centroids: Optional[np.ndarray], seed: int,
                    skeleton_fraction: float = 0.05,
                    lid: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the int64 insertion permutation for strategy ``name``."""
    if name not in STRATEGIES:
        raise ValueError(
            f"insertion_order must be one of {STRATEGIES}, got {name!r}")
    if not 0.0 <= skeleton_fraction <= 1.0:
        raise ValueError("skeleton_fraction must be in [0, 1]")
    n = len(vectors)

    if name == 'random':
        return np.random.RandomState(seed).permutation(n).astype(np.int64)
    if name == 'zone':
        return _zone_order(vectors, zone_assignments, zone_centroids)
    if name == 'skeleton':
        perm = np.random.RandomState(seed).permutation(n).astype(np.int64)
        k = int(round(skeleton_fraction * n))
        sampled = np.zeros(n, dtype=bool)
        sampled[perm[:k]] = True
        zone = _zone_order(vectors, zone_assignments, zone_centroids)
        return np.concatenate([perm[:k], zone[~sampled[zone]]]).astype(np.int64)
    if name == 'roundrobin':
        lists = _zone_sorted_lists(vectors, zone_assignments, zone_centroids)
        out: list = []
        for i in range(max(len(a) for a in lists)):
            out.extend(a[i] for a in lists if i < len(a))
        return np.array(out, dtype=np.int64)

    # lid_desc / lid_asc
    if lid is None:
        lid = estimate_lid(vectors, seed=seed)
    if name == 'lid_desc':
        return np.argsort(-lid, kind='stable').astype(np.int64)
    return np.argsort(lid, kind='stable').astype(np.int64)
