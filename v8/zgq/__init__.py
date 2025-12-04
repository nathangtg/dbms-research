"""
ZGQ v8 - Zone-Guided Quantization for High-Performance ANNS
============================================================

A novel Approximate Nearest Neighbor Search algorithm that combines
hierarchical zonal partitioning with optimized graph navigation.

Example:
    >>> from zgq import ZGQIndex, ZGQConfig
    >>> config = ZGQConfig(verbose=True)
    >>> index = ZGQIndex(config)
    >>> index.build(vectors)
    >>> ids, distances = index.search(query, k=10)
"""

from zgq.index import ZGQIndex, ZGQConfig
from zgq.search import ZGQSearch

__version__ = "8.0.0"
__author__ = "ZGQ Research Team"
__all__ = ["ZGQIndex", "ZGQConfig", "ZGQSearch", "__version__"]
