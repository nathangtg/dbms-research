"""
Analytic index size model
=========================

RSS deltas are unreliable on macOS, so equal-memory comparisons use the bytes
each index structure occupies, computed from its parameters. Every function
returns ``{"components": {name: bytes}, "total_bytes": int, "bytes_per_vector": float}``.

Formulas (n vectors, dimension d, all floats are 4 bytes):

hnswlib HNSW (level-0 block per element + upper-level link lists)
    size_data_per_element = 4*(2M) + 4 + 4*d + 8      links, link count, vector, label
    upper levels          = n/(M-1) * (4M + 4)        E[level] = 1/(M-1) for mult = 1/ln(M)
    saved file            = 96 header + n*(size_data_per_element + 4) + upper levels
    The 4 bytes added per element are the link-list size word written for
    every element in the saved file; it is not resident for level-0 only elements
    but is included so the model can be checked against the file length.

ZGQ default index (what ``ZGQIndex.build`` keeps): the hnswlib index above, plus
    a second copy of the vectors (``vectors``), the squared norms (4n), PQ codes
    (n*pq_m*bits/8), PQ codebooks (pq_m*2^bits*d/pq_m*4 = 2^bits*d*4) and zone data.

ZGQ compressed (``compressed_search=True``): level-0 link matrix (n*2M*4, int32,
    -1 padded), PQ codes, zone id per vector (2 bytes), codebooks, zone centroids
    (Z*d*4), per-zone entry point (Z*4), and for residual PQ the zone/codebook
    cross table (Z*pq_m*2^bits*4). Optional raw vectors (n*d*4) for exact rerank.

FAISS IVF-PQ:   n*(pq_m*bits/8 + 8 id) + codebooks + nlist*d*4 coarse centroids
FAISS IVF-PQR:  IVF-PQ + n*refine_m*refine_bits/8 + refine codebooks
FAISS HNSW-PQ:  n*(pq_m*bits/8) + codebooks + graph
FAISS HNSW-SQ8: n*d + 2*d*4 + graph
FAISS HNSW graph: n*(2M*4 + 8 offset + 4 level) + n/(M-1)*(M*4); the upper
    term is the expected link storage above level 0.

``tests/test_memory_model.py`` checks the hnswlib and FAISS formulas against
real saved/serialized sizes.
"""

from typing import Dict, Optional

FLOAT = 4


def _result(components: Dict[str, float], n: int) -> Dict:
    comps = {k: int(round(v)) for k, v in components.items()}
    total = sum(comps.values())
    return {
        "components": comps,
        "total_bytes": total,
        "bytes_per_vector": total / n if n else 0.0,
    }


def hnswlib_bytes(n: int, dim: int, M: int = 16) -> Dict:
    """hnswlib HNSW file size model (see module docstring)."""
    level0 = n * (4 * 2 * M + 4 + FLOAT * dim + 8)
    upper = n / (M - 1) * (4 * M + 4)
    return _result(
        {
            "header": 96,
            "level0_block": level0,
            "link_list_size_words": 4 * n,
            "upper_levels": upper,
        },
        n,
    )


def pq_codebook_bytes(dim: int, pq_bits: int = 8) -> int:
    """pq_m * 2^bits * (d/pq_m) * 4 = 2^bits * d * 4, independent of pq_m."""
    return (2**pq_bits) * dim * FLOAT


def zgq_default_bytes(
    n: int, dim: int, M: int = 16, pq_m: int = 16, pq_bits: int = 8, n_zones: int = 0
) -> Dict:
    """Resident size of a default ``ZGQIndex`` (graph with its own vector copy
    plus ``self.vectors``, norms, PQ codes, codebooks, zone centroids and ids)."""
    g = hnswlib_bytes(n, dim, M)["components"]
    return _result(
        {
            "hnsw_graph_with_vectors": g["level0_block"] + g["upper_levels"],
            "vectors_copy": n * dim * FLOAT,
            "vector_norms": n * FLOAT,
            "pq_codes": n * pq_m * pq_bits / 8,
            "pq_codebooks": pq_codebook_bytes(dim, pq_bits),
            "zone_centroids": n_zones * dim * FLOAT,
            "zone_assignments": n * 8,
        },
        n,
    )


def zgq_compressed_bytes(
    n: int,
    dim: int,
    M: int = 16,
    pq_m: int = 16,
    pq_bits: int = 8,
    n_zones: int = 0,
    residual: bool = True,
    keep_vectors: bool = False,
) -> Dict:
    """Size of the compressed ZGQ structures (graph + codes, optional raw vectors)."""
    comps = {
        "level0_links": n * 2 * M * 4,
        "pq_codes": n * pq_m * pq_bits / 8,
        "pq_codebooks": pq_codebook_bytes(dim, pq_bits),
        "zone_entry_points": n_zones * 4,
        "zone_centroids": n_zones * dim * FLOAT,
    }
    if residual:
        comps["zone_ids"] = n * 2
        comps["zone_codebook_cross_table"] = n_zones * (2**pq_bits) * pq_m * FLOAT
    if keep_vectors:
        comps["raw_vectors"] = n * dim * FLOAT
    return _result(comps, n)


def faiss_ivfpq_bytes(
    n: int,
    dim: int,
    nlist: int,
    pq_m: int,
    pq_bits: int = 8,
    refine_m: Optional[int] = None,
    refine_bits: int = 8,
) -> Dict:
    """IndexIVFPQ, or IndexIVFPQR when ``refine_m`` is given."""
    comps = {
        "pq_codes": n * pq_m * pq_bits / 8,
        "ids": n * 8,
        "pq_codebooks": pq_codebook_bytes(dim, pq_bits),
        "coarse_centroids": nlist * dim * FLOAT,
    }
    if refine_m:
        comps["refine_codes"] = n * refine_m * refine_bits / 8
        comps["refine_codebooks"] = (2**refine_bits) * dim * FLOAT
    return _result(comps, n)


def _faiss_graph(n: int, M: int) -> Dict[str, float]:
    return {
        "graph_level0_links": n * 2 * M * 4,
        "graph_offsets_levels": n * (8 + 4),
        "graph_upper_links": n / (M - 1) * M * 4,
    }


def faiss_hnswpq_bytes(n: int, dim: int, M: int, pq_m: int, pq_bits: int = 8) -> Dict:
    comps: Dict[str, float] = {
        "pq_codes": n * pq_m * pq_bits / 8,
        "pq_codebooks": pq_codebook_bytes(dim, pq_bits),
    }
    comps.update(_faiss_graph(n, M))
    return _result(comps, n)


def faiss_hnswsq8_bytes(n: int, dim: int, M: int) -> Dict:
    comps: Dict[str, float] = {"sq8_codes": n * dim, "sq8_trained": 2 * dim * FLOAT}
    comps.update(_faiss_graph(n, M))
    return _result(comps, n)
