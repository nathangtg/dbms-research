import json
from pathlib import Path
from typing import Union


def save_index(index, filepath: Union[str, Path]) -> None:
    index.save(str(filepath))


def load_index(filepath: Union[str, Path]):
    from zgq.index import ZGQIndex
    return ZGQIndex.load(str(filepath))


def save_benchmark_results(results: dict, filepath: Union[str, Path]) -> None:
    def convert(obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)


def load_benchmark_results(filepath: Union[str, Path]) -> dict:
    with open(filepath, 'r') as f:
        return json.load(f)
