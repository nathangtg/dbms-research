"""Structure analysis of a built hnswlib index.

hnswlib 0.8.0 exposes no neighbor lists, so this module parses the file that
``Index.save_index`` writes (``HierarchicalNSW::saveIndex`` in hnswalg.h).

File layout (little endian, native x86-64 / arm64 sizes):

Header, in this order:
    offsetLevel0 (size_t), max_elements (size_t), cur_element_count (size_t),
    size_data_per_element (size_t), label_offset (size_t), offsetData (size_t),
    maxlevel (int32), enterpoint_node (uint32), maxM (size_t), maxM0 (size_t),
    M (size_t), mult (double), ef_construction (size_t).

Level 0 block: cur_element_count records of size_data_per_element bytes:
    [uint32 link header: low 16 bits = neighbor count][maxM0 x uint32 ids]
    [vector data (offsetData .. label_offset)][label (size_t) at label_offset].
Ids at every level are internal ids (insertion order, 0..n-1).

Upper levels: for each element i in order, a uint32 byte size
``linkListSize`` (0 if level 0 only, else size_links_per_element * level_i,
with size_links_per_element = 4 + 4 * maxM) followed by that many bytes. Level
l (1-based) of the element occupies bytes [(l-1)*s, l*s) of the list, each in
the same [count header][maxM x uint32] format.

Internal ids are used everywhere; ``labels[internal_id]`` maps to the user label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numba import njit
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

_HEADER_FMT = [
    ("offset_level0", "<u8"),
    ("max_elements", "<u8"),
    ("cur_count", "<u8"),
    ("size_data_per_element", "<u8"),
    ("label_offset", "<u8"),
    ("offset_data", "<u8"),
    ("max_level", "<i4"),
    ("enter_point", "<u4"),
    ("max_m", "<u8"),
    ("max_m0", "<u8"),
    ("m", "<u8"),
    ("mult", "<f8"),
    ("ef_construction", "<u8"),
]
_HEADER_DTYPE = np.dtype(_HEADER_FMT)


@dataclass
class HNSWGraph:
    """Parsed hnswlib graph in internal ids."""

    level0: np.ndarray  # (n, maxM0) int32, padded with -1
    counts: np.ndarray  # (n,) int32 level-0 out-degree
    labels: np.ndarray  # (n,) int64
    levels: np.ndarray  # (n,) int32 top level of each node
    vectors: np.ndarray  # (n, dim) float32
    upper_row: np.ndarray  # (n,) int64 first row in upper_links (-1 if level 0 only)
    upper_links: (
        np.ndarray
    )  # (R, maxM) int32 padded with -1; row = upper_row[i] + level - 1
    upper_counts: np.ndarray  # (R,) int32
    entry_point: int
    max_level: int
    M: int
    max_m: int
    max_m0: int
    ef_construction: int

    @property
    def n(self) -> int:
        return int(self.level0.shape[0])

    def level_neighbors(self, node: int, level: int) -> np.ndarray:
        """Neighbor internal ids of ``node`` at ``level`` (0 = base layer)."""
        if level == 0:
            return self.level0[node, : self.counts[node]]
        if level > self.levels[node]:
            raise ValueError("node %d has no level %d" % (node, level))
        row = self.upper_row[node] + level - 1
        return self.upper_links[row, : self.upper_counts[row]]


def read_hnsw_index(path: str) -> HNSWGraph:
    """Parse a file written by ``hnswlib.Index.save_index``."""
    raw = np.fromfile(path, dtype=np.uint8)
    hdr = raw[: _HEADER_DTYPE.itemsize].view(_HEADER_DTYPE)[0]
    n = int(hdr["cur_count"])
    sz = int(hdr["size_data_per_element"])
    max_m, max_m0 = int(hdr["max_m"]), int(hdr["max_m0"])
    off_data, off_label = int(hdr["offset_data"]), int(hdr["label_offset"])
    dim = (off_label - off_data) // 4
    start = _HEADER_DTYPE.itemsize
    blk = raw[start : start + n * sz].reshape(n, sz)

    counts = np.ascontiguousarray(blk[:, 0:2]).view("<u2").reshape(n).astype(np.int32)
    level0 = (
        np.ascontiguousarray(blk[:, 4 : 4 + 4 * max_m0]).view("<u4").astype(np.int32)
    )
    level0 = level0.reshape(n, max_m0)
    level0[np.arange(max_m0)[None, :] >= counts[:, None]] = -1
    vectors = (
        np.ascontiguousarray(blk[:, off_data:off_label]).view("<f4").reshape(n, dim)
    )
    labels = (
        np.ascontiguousarray(blk[:, off_label : off_label + 8]).view("<u8").reshape(n)
    )
    labels = labels.astype(np.int64)

    # Upper levels: walk the variable-size records.
    per_level = 4 + 4 * max_m
    levels = np.zeros(n, dtype=np.int32)
    upper_row = np.full(n, -1, dtype=np.int64)
    pos = start + n * sz
    sizes_pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        size = int(raw[pos : pos + 4].view("<u4")[0])
        sizes_pos[i] = pos + 4
        levels[i] = size // per_level
        pos += 4 + size
    if pos != raw.size:
        raise ValueError(
            "unexpected file size: parsed %d of %d bytes" % (pos, raw.size)
        )
    total_rows = int(levels.sum())
    upper_links = np.full((total_rows, max_m), -1, dtype=np.int32)
    upper_counts = np.zeros(total_rows, dtype=np.int32)
    row = 0
    for i in np.nonzero(levels)[0]:
        lv = int(levels[i])
        upper_row[i] = row
        chunk = raw[sizes_pos[i] : sizes_pos[i] + lv * per_level].reshape(lv, per_level)
        cnt = (
            np.ascontiguousarray(chunk[:, 0:2]).view("<u2").reshape(lv).astype(np.int32)
        )
        ids = (
            np.ascontiguousarray(chunk[:, 4:])
            .view("<u4")
            .astype(np.int32)
            .reshape(lv, max_m)
        )
        ids[np.arange(max_m)[None, :] >= cnt[:, None]] = -1
        upper_links[row : row + lv] = ids
        upper_counts[row : row + lv] = cnt
        row += lv

    return HNSWGraph(
        level0=level0,
        counts=counts,
        labels=labels,
        levels=levels,
        vectors=vectors,
        upper_row=upper_row,
        upper_links=upper_links,
        upper_counts=upper_counts,
        entry_point=int(hdr["enter_point"]),
        max_level=int(hdr["max_level"]),
        M=int(hdr["m"]),
        max_m=max_m,
        max_m0=max_m0,
        ef_construction=int(hdr["ef_construction"]),
    )


# ----------------------------------------------------------------------------
# Elementary statistics (pure functions, tested on fixtures)
# ----------------------------------------------------------------------------


@njit(cache=True)
def _bfs(neighbors, counts, start):
    n = neighbors.shape[0]
    seen = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int32)
    head = 0
    tail = 1
    queue[0] = start
    seen[start] = True
    while head < tail:
        u = queue[head]
        head += 1
        for j in range(counts[u]):
            v = neighbors[u, j]
            if not seen[v]:
                seen[v] = True
                queue[tail] = v
                tail += 1
    return tail


def reachable_count(neighbors: np.ndarray, counts: np.ndarray, start: int) -> int:
    """Number of nodes reachable from ``start`` following directed edges."""
    return int(
        _bfs(
            np.ascontiguousarray(neighbors, dtype=np.int32),
            np.ascontiguousarray(counts, dtype=np.int32),
            int(start),
        )
    )


def _edge_arrays(
    neighbors: np.ndarray, counts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flat (src, dst) arrays of valid edges."""
    mask = np.arange(neighbors.shape[1])[None, :] < counts[:, None]
    src = np.repeat(np.arange(neighbors.shape[0], dtype=np.int32), counts)
    return src, neighbors[mask]


def cross_zone_fraction(
    neighbors: np.ndarray, counts: np.ndarray, zones: np.ndarray
) -> float:
    """Fraction of directed edges whose endpoints lie in different zones."""
    src, dst = _edge_arrays(neighbors, counts)
    if src.size == 0:
        return 0.0
    return float(np.mean(zones[src] != zones[dst]))


def edge_length_ratios(edge_dists: np.ndarray, kth_dist: np.ndarray) -> np.ndarray:
    """Edge length divided by the source node's exact k-th NN distance.

    ``edge_dists`` is (s, m) with NaN padding, ``kth_dist`` is (s,). A ratio of
    1 means the edge is as long as the k-th nearest neighbor; the result is the
    flat array of non-NaN ratios.
    """
    ratio = edge_dists / kth_dist[:, None]
    return ratio[~np.isnan(ratio)]


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float(2.0 * np.sum(np.arange(1, n + 1) * x) / (n * total) - (n + 1.0) / n)


def exact_knn_sample(
    vectors: np.ndarray, sample: np.ndarray, k: int, chunk: int = 20000
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbors (self excluded) of ``vectors[sample]``.

    Returns (ids (s, k), L2 distances (s, k)) sorted ascending. Brute force in
    chunks, so memory is O(s * chunk).
    """
    q = np.ascontiguousarray(vectors[sample], dtype=np.float32)
    qn = np.einsum("ij,ij->i", q, q)
    s = q.shape[0]
    best_d = np.full((s, k), np.inf, dtype=np.float32)
    best_i = np.full((s, k), -1, dtype=np.int64)
    for lo in range(0, vectors.shape[0], chunk):
        c = vectors[lo : lo + chunk]
        cn = np.einsum("ij,ij->i", c, c)
        d = qn[:, None] - 2.0 * (q @ c.T) + cn[None, :]
        ids = np.arange(lo, lo + c.shape[0])
        d[sample[:, None] == ids[None, :]] = np.inf
        cat_d = np.concatenate([best_d, d], axis=1)
        cat_i = np.concatenate([best_i, np.broadcast_to(ids, d.shape)], axis=1)
        part = np.argpartition(cat_d, k - 1, axis=1)[:, :k]
        best_d = np.take_along_axis(cat_d, part, axis=1)
        best_i = np.take_along_axis(cat_i, part, axis=1)
    order = np.argsort(best_d, axis=1)
    best_d = np.take_along_axis(best_d, order, axis=1)
    best_i = np.take_along_axis(best_i, order, axis=1)
    return best_i, np.sqrt(np.maximum(best_d, 0.0))


def _pct(x: np.ndarray, qs=(50, 90, 99)) -> Dict[str, float]:
    return {"p%d" % q: float(np.percentile(x, q)) for q in qs}


def analyze_graph(
    g: HNSWGraph,
    vectors_by_internal_id: np.ndarray,
    zone_of_internal_id: np.ndarray,
    seed: int = 0,
    n_sample: int = 2000,
    knn_k: int = 10,
) -> dict:
    """JSON-serializable structure statistics of ``g`` (level 0 unless noted)."""
    n = g.n
    zones = np.asarray(zone_of_internal_id)
    rng = np.random.RandomState(seed)
    out: dict = {"n": n, "M": g.M, "max_level": g.max_level}

    # Reachability and components.
    out["reachable_fraction_from_entry"] = (
        reachable_count(g.level0, g.counts, g.entry_point) / n
    )
    src, dst = _edge_arrays(g.level0, g.counts)
    adj = csr_matrix((np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n))
    out["n_weak_components"] = int(
        connected_components(adj, directed=True, connection="weak")[0]
    )
    del adj

    # Degrees.
    deg = g.counts
    out["out_degree"] = {
        "mean": float(deg.mean()),
        "p50": float(np.percentile(deg, 50)),
        "p99": float(np.percentile(deg, 99)),
    }
    indeg = np.bincount(dst, minlength=n)
    out["in_degree_zero_fraction"] = float(np.mean(indeg == 0))
    out["in_degree_max"] = int(indeg.max())
    out["in_degree_gini"] = gini(indeg)

    # Zones.
    out["cross_zone_edge_fraction"] = cross_zone_fraction(g.level0, g.counts, zones)

    # Sampled edge geometry against exact 10-NN.
    s = min(n_sample, n)
    sample = np.sort(rng.choice(n, size=s, replace=False)).astype(np.int64)
    knn_ids, knn_d = exact_knn_sample(vectors_by_internal_id, sample, knn_k)
    out["n_sample"] = int(s)
    out["cross_zone_fraction_exact_knn"] = float(
        np.mean(zones[knn_ids] != zones[sample][:, None])
    )
    nb = g.level0[sample]
    valid = nb >= 0
    gq = vectors_by_internal_id[sample].astype(np.float32)
    diff = vectors_by_internal_id[np.where(valid, nb, 0)] - gq[:, None, :]
    edge_d = np.sqrt(np.einsum("smd,smd->sm", diff, diff))
    edge_d[~valid] = np.nan
    ratios = edge_length_ratios(edge_d, knn_d[:, knn_k - 1])
    out["cross_zone_fraction_graph_sample"] = float(
        np.mean((zones[np.where(valid, nb, 0)] != zones[sample][:, None])[valid])
    )
    out["edge_length_ratio"] = {
        "reference": "distance to exact %dth neighbor" % knn_k,
        "frac_gt_2x": float(np.mean(ratios > 2.0)),
        "frac_gt_4x": float(np.mean(ratios > 4.0)),
        **_pct(ratios),
    }

    # Upper layers.
    max_lv = int(g.levels.max())
    out["nodes_per_level"] = [int(np.sum(g.levels >= lv)) for lv in range(max_lv + 1)]
    upper = np.nonzero(g.levels >= 1)[0]
    top = np.nonzero(g.levels == max_lv)[0]
    out["distinct_zones_upper"] = int(np.unique(zones[upper]).size)
    out["distinct_zones_top_level"] = int(np.unique(zones[top]).size)
    out["n_zones"] = int(np.unique(zones).size)
    out["entry_point_internal_id"] = int(g.entry_point)
    out["entry_point_zone"] = int(zones[g.entry_point])
    out["entry_point_insertion_rank"] = int(g.entry_point)
    return out


# ----------------------------------------------------------------------------
# Instrumented search
# ----------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _l2(vectors, q, i):
    s = 0.0
    for t in range(q.shape[0]):
        d = vectors[i, t] - q[t]
        s += d * d
    return s


@njit(cache=True)
def _heap_push(hd, hi, size, d, i, sign):
    # sign=+1: min-heap on d; sign=-1: max-heap on d (stored as keys sign*d)
    key = sign * d
    pos = size
    while pos > 0:
        parent = (pos - 1) >> 1
        if sign * hd[parent] <= key:
            break
        hd[pos] = hd[parent]
        hi[pos] = hi[parent]
        pos = parent
    hd[pos] = d
    hi[pos] = i
    return size + 1


@njit(cache=True)
def _heap_pop(hd, hi, size, sign):
    size -= 1
    d = hd[size]
    i = hi[size]
    key = sign * d
    pos = 0
    while True:
        child = 2 * pos + 1
        if child >= size:
            break
        if child + 1 < size and sign * hd[child + 1] < sign * hd[child]:
            child += 1
        if sign * hd[child] >= key:
            break
        hd[pos] = hd[child]
        hi[pos] = hi[child]
        pos = child
    if size > 0:
        hd[pos] = d
        hi[pos] = i
    return size


@njit(cache=True)
def _search_one(
    level0,
    counts,
    upper_row,
    upper_links,
    upper_counts,
    vectors,
    entry,
    max_level,
    q,
    ef,
    k,
    visited,
    stamp,
    cand_d,
    cand_i,
    res_d,
    res_i,
    out_ids,
):
    ndist = 1
    hops_up = 0
    cur = entry
    cur_d = _l2(vectors, q, cur)
    for lv in range(max_level, 0, -1):
        changed = True
        while changed:
            changed = False
            hops_up += 1
            row = upper_row[cur] + lv - 1
            for j in range(upper_counts[row]):
                v = upper_links[row, j]
                d = _l2(vectors, q, v)
                ndist += 1
                if d < cur_d:
                    cur_d = d
                    cur = v
                    changed = True
    # Level 0 best-first search (searchBaseLayerST).
    csize = 0
    rsize = 0
    csize = _heap_push(cand_d, cand_i, csize, cur_d, cur, 1)
    rsize = _heap_push(res_d, res_i, rsize, cur_d, cur, -1)
    visited[cur] = stamp
    lower = cur_d
    hops0 = 0
    while csize > 0:
        cd = cand_d[0]
        ci = cand_i[0]
        if cd > lower and rsize >= ef:
            break
        csize = _heap_pop(cand_d, cand_i, csize, 1)
        hops0 += 1
        for j in range(counts[ci]):
            v = level0[ci, j]
            if visited[v] == stamp:
                continue
            visited[v] = stamp
            d = _l2(vectors, q, v)
            ndist += 1
            if rsize < ef or lower > d:
                csize = _heap_push(cand_d, cand_i, csize, d, v, 1)
                rsize = _heap_push(res_d, res_i, rsize, d, v, -1)
                if rsize > ef:
                    rsize = _heap_pop(res_d, res_i, rsize, -1)
                lower = res_d[0]
    while rsize > k:
        rsize = _heap_pop(res_d, res_i, rsize, -1)
    for t in range(rsize - 1, -1, -1):
        out_ids[t] = res_i[0]
        rsize = _heap_pop(res_d, res_i, rsize, -1)
    return hops0, hops_up, ndist


@njit(cache=True)
def _search_all(
    level0,
    counts,
    upper_row,
    upper_links,
    upper_counts,
    vectors,
    labels,
    entry,
    max_level,
    queries,
    ef,
    k,
):
    nq = queries.shape[0]
    n = level0.shape[0]
    ids = np.full((nq, k), -1, dtype=np.int64)
    hops0 = np.zeros(nq, dtype=np.int64)
    hops_up = np.zeros(nq, dtype=np.int64)
    ndist = np.zeros(nq, dtype=np.int64)
    visited = np.zeros(n, dtype=np.int32)
    cap = n + 1
    cand_d = np.empty(cap, dtype=np.float32)
    cand_i = np.empty(cap, dtype=np.int32)
    res_d = np.empty(ef + 2, dtype=np.float32)
    res_i = np.empty(ef + 2, dtype=np.int32)
    tmp = np.full(k, -1, dtype=np.int32)
    for qi in range(nq):
        tmp[:] = -1
        h0, hu, nd = _search_one(
            level0,
            counts,
            upper_row,
            upper_links,
            upper_counts,
            vectors,
            entry,
            max_level,
            queries[qi],
            ef,
            k,
            visited,
            qi + 1,
            cand_d,
            cand_i,
            res_d,
            res_i,
            tmp,
        )
        hops0[qi] = h0
        hops_up[qi] = hu
        ndist[qi] = nd
        for t in range(k):
            if tmp[t] >= 0:
                ids[qi, t] = labels[tmp[t]]
    return ids, hops0, hops_up, ndist


def search_hops(
    graph: HNSWGraph,
    vectors: Optional[np.ndarray],
    queries: np.ndarray,
    ef: int,
    k: int,
) -> dict:
    """Reimplementation of hnswlib search that counts work per query.

    ``vectors`` are indexed by internal id (``None`` uses ``graph.vectors``).
    Returns result ``ids`` (labels), per-query ``hops_level0`` (expanded
    nodes), ``hops_upper``, ``dist_comps`` and ``summary`` aggregates.
    """
    vecs = np.ascontiguousarray(
        graph.vectors if vectors is None else vectors, dtype=np.float32
    )
    ef = max(int(ef), int(k))
    ids, h0, hu, nd = _search_all(
        graph.level0,
        graph.counts,
        graph.upper_row,
        graph.upper_links,
        graph.upper_counts,
        vecs,
        graph.labels,
        np.int32(graph.entry_point),
        graph.max_level,
        np.ascontiguousarray(queries, dtype=np.float32),
        ef,
        k,
    )

    def agg(x):
        return {
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)),
        }

    return {
        "ids": ids,
        "hops_level0": h0,
        "hops_upper": hu,
        "dist_comps": nd,
        "summary": {
            "hops_level0": agg(h0),
            "hops_upper": agg(hu),
            "dist_comps": agg(nd),
        },
    }
