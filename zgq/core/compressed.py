import os
import struct
import tempfile
from typing import Tuple

import numpy as np
from numba import njit


def extract_level0_links(hnsw, dim: int) -> Tuple[np.ndarray, int]:
    fd, path = tempfile.mkstemp(suffix=".hnsw")
    os.close(fd)
    try:
        hnsw.save_index(path)
        with open(path, "rb") as f:
            raw = f.read(96)
            (
                offset_level0,
                max_elements,
                cur_count,
                size_per_el,
                label_offset,
                offset_data,
                maxlevel,
                enter,
                maxM,
                maxM0,
                M,
            ) = struct.unpack("<QQQQQQiIQQQ", raw[:80])
            block = np.fromfile(f, dtype=np.uint8, count=cur_count * size_per_el)
    finally:
        os.remove(path)
    block = block.reshape(cur_count, size_per_el)
    counts = block[:, :2].copy().view(np.uint16).reshape(-1).astype(np.int64)
    links_internal = (
        block[:, 4 : 4 + 4 * maxM0].copy().view(np.int32).reshape(cur_count, maxM0)
    )
    labels = block[:, label_offset : label_offset + 8].copy().view(np.int64).reshape(-1)

    # Mask slots beyond each node's link count, then map internal ids to labels
    col = np.arange(maxM0)[None, :]
    links_internal = np.where(col < counts[:, None], links_internal, -1)
    mapped = np.where(links_internal >= 0, labels[np.clip(links_internal, 0, None)], -1)
    out = np.full((cur_count, maxM0), -1, dtype=np.int32)
    out[labels] = mapped.astype(np.int32)
    return out, int(maxM0)


def effective_codebooks(pq) -> np.ndarray:
    cb = pq.codebooks
    if pq.config.normalize_residuals and pq.mean_residual is not None:
        m, _, sub = cb.shape
        return (cb * pq.std_residual + pq.mean_residual.reshape(m, 1, sub)).astype(
            np.float32
        )
    return cb.astype(np.float32)


def zone_cross_table(centroids: np.ndarray, cw: np.ndarray) -> np.ndarray:
    m, k, sub = cw.shape
    cj = centroids.reshape(len(centroids), m, sub)
    return (2.0 * np.einsum("zjs,jks->zjk", cj, cw)).astype(np.float32)


def query_tables(queries: np.ndarray, cw: np.ndarray, centroids: np.ndarray):
    m, k, sub = cw.shape
    qj = queries.reshape(len(queries), m, sub)
    cw_norm = np.einsum("jks,jks->jk", cw, cw)
    A = (-2.0 * np.einsum("qjs,jks->qjk", qj, cw) + cw_norm[None]).astype(np.float32)
    qn = np.einsum("ij,ij->i", queries, queries)
    cn = np.einsum("ij,ij->i", centroids, centroids)
    Zd = qn[:, None] - 2.0 * (queries @ centroids.T) + cn[None, :]
    return A, np.maximum(Zd, 0).astype(np.float32)


@njit(cache=True)
def _beam_search(
    links, codes, zone_of, A, Qz, B, use_zones, entries, ef, out_ids, out_d
):
    n = links.shape[0]
    m = codes.shape[1]
    M0 = links.shape[1]
    visited = np.zeros(n, dtype=np.bool_)
    bd = np.empty(ef, dtype=np.float32)
    bi = np.empty(ef, dtype=np.int64)
    bx = np.zeros(ef, dtype=np.bool_)  # expanded flag
    cnt = 0

    for e in range(entries.shape[0]):
        v = entries[e]
        if v < 0 or visited[v]:
            continue
        visited[v] = True
        d = Qz[zone_of[v]] if use_zones else Qz[0]
        for j in range(m):
            c = codes[v, j]
            d += A[j, c]
            if use_zones:
                d += B[zone_of[v], j, c]
        if cnt < ef or d < bd[cnt - 1]:
            p = cnt if cnt < ef else ef - 1
            while p > 0 and bd[p - 1] > d:
                if p < ef:
                    bd[p] = bd[p - 1]
                    bi[p] = bi[p - 1]
                    bx[p] = bx[p - 1]
                p -= 1
            bd[p] = d
            bi[p] = v
            bx[p] = False
            if cnt < ef:
                cnt += 1

    cursor = 0
    while True:
        while cursor < cnt and bx[cursor]:
            cursor += 1
        if cursor >= cnt:
            break
        bx[cursor] = True
        u = bi[cursor]
        for t in range(M0):
            v = links[u, t]
            if v < 0:
                break
            if visited[v]:
                continue
            visited[v] = True
            d = Qz[zone_of[v]] if use_zones else Qz[0]
            for j in range(m):
                c = codes[v, j]
                d += A[j, c]
                if use_zones:
                    d += B[zone_of[v], j, c]
            if cnt < ef or d < bd[cnt - 1]:
                p = cnt if cnt < ef else ef - 1
                while p > 0 and bd[p - 1] > d:
                    bd[p] = bd[p - 1]
                    bi[p] = bi[p - 1]
                    bx[p] = bx[p - 1]
                    p -= 1
                bd[p] = d
                bi[p] = v
                bx[p] = False
                if cnt < ef:
                    cnt += 1
                if p < cursor:
                    cursor = p

    for i in range(cnt):
        out_ids[i] = bi[i]
        out_d[i] = bd[i]
    return cnt


@njit(cache=True)
def _batch_beam(
    links, codes, zone_of, A, Qz, B, use_zones, entries, ef, out_ids, out_d
):
    nq = A.shape[0]
    counts = np.zeros(nq, dtype=np.int64)
    for i in range(nq):
        counts[i] = _beam_search(
            links,
            codes,
            zone_of,
            A[i],
            Qz[i],
            B,
            use_zones,
            entries[i],
            ef,
            out_ids[i],
            out_d[i],
        )
    return counts


def beam_search_batch(links, codes, zone_of, A, Qz, B, entries, ef):
    nq = len(A)
    ids = np.full((nq, ef), -1, dtype=np.int64)
    dists = np.full((nq, ef), np.inf, dtype=np.float32)
    use_zones = B is not None
    if B is None:
        B = np.zeros((1, 1, 1), dtype=np.float32)
    _batch_beam(links, codes, zone_of, A, Qz, B, use_zones, entries, ef, ids, dists)
    return ids, dists
