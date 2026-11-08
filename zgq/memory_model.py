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
    return (2**pq_bits) * dim * FLOAT


def zgq_default_bytes(
    n: int, dim: int, M: int = 16, pq_m: int = 16, pq_bits: int = 8, n_zones: int = 0
) -> Dict:
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
