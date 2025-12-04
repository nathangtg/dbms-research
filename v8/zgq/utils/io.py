"""
I/O Utilities for ZGQ v8
========================

Functions for saving and loading ZGQ indices.
"""

import pickle
from pathlib import Path
from typing import Union


def save_index(index, filepath: Union[str, Path]) -> None:
    """
    Save ZGQ index to file.
    
    Args:
        index: ZGQIndex instance
        filepath: Path to save file
    """
    index.save(str(filepath))


def load_index(filepath: Union[str, Path]):
    """
    Load ZGQ index from file.
    
    Args:
        filepath: Path to saved index
        
    Returns:
        Loaded ZGQIndex instance
    """
    from zgq.index import ZGQIndex
    return ZGQIndex.load(str(filepath))


def save_benchmark_results(results: dict, filepath: Union[str, Path]) -> None:
    """
    Save benchmark results to JSON file.
    
    Args:
        results: Benchmark results dictionary
        filepath: Output file path
    """
    import json
    
    # Convert numpy types to Python types
    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)


def load_benchmark_results(filepath: Union[str, Path]) -> dict:
    """
    Load benchmark results from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Benchmark results dictionary
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)
