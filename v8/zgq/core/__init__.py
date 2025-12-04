"""
ZGQ Core Components
===================

Core algorithms and data structures for Zone-Guided Quantization.
"""

from zgq.core.zones import AdaptiveHierarchicalZones, ZoneConfig
from zgq.core.graph import ZoneGuidedGraph
from zgq.core.distances import DistanceComputer, SIMDDistance
from zgq.core.quantization import ResidualProductQuantizer

__all__ = [
    "AdaptiveHierarchicalZones",
    "ZoneConfig", 
    "ZoneGuidedGraph",
    "DistanceComputer",
    "SIMDDistance",
    "ResidualProductQuantizer"
]
