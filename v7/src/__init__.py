"""
ZGQ V7 - Zonal Graph Quantization for Approximate Nearest Neighbor Search.

A next-generation ANNS algorithm combining zonal partitioning,
HNSW graphs, and product quantization.
"""

try:
    from .index import ZGQIndex
    from .search import ZGQSearch, compute_ground_truth
    from .core import (
        DistanceMetrics,
        PQDistanceMetrics,
        DistanceUtils,
        ZonalPartitioner,
        HNSWGraphManager,
        ProductQuantizer,
        suggest_n_zones,
        suggest_pq_parameters
    )
except ImportError:
    from index import ZGQIndex
    from search import ZGQSearch, compute_ground_truth
    from core import (
        DistanceMetrics,
        PQDistanceMetrics,
        DistanceUtils,
        ZonalPartitioner,
        HNSWGraphManager,
        ProductQuantizer,
        suggest_n_zones,
        suggest_pq_parameters
    )

__version__ = "7.0.0"

__all__ = [
    'ZGQIndex',
    'ZGQSearch',
    'compute_ground_truth',
    'DistanceMetrics',
    'PQDistanceMetrics',
    'DistanceUtils',
    'ZonalPartitioner',
    'HNSWGraphManager',
    'ProductQuantizer',
    'suggest_n_zones',
    'suggest_pq_parameters',
]
