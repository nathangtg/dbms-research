"""
Schema validator for ZGQ benchmark results.

Usage:
    python -m benchmarks.check_results results/2025-01-01/summary.json
    python -m benchmarks.check_results results/2025-01-01/summary.json --canonical

``--canonical`` additionally enforces the protocol floor required for numbers
that may be quoted in the paper (>= 1000 queries, >= 5 repeats).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCHEMA_VERSION = "zgq-bench/1"

STAT_KEYS = ("median", "q1", "q3", "iqr", "n")
ENV_KEYS = (
    "python",
    "platform",
    "machine",
    "cpu",
    "cpu_count",
    "ram_gb",
    "libraries",
    "git_commit",
    "git_dirty",
)
PROTOCOL_KEYS = (
    "seed",
    "k",
    "n_queries",
    "repeats",
    "warmup_passes",
    "threads",
    "memory_metric",
    "latency_metric",
)
ALGOS = ("hnsw", "zgq", "ivfpq")
MIN_CANONICAL_QUERIES = 1000
MIN_CANONICAL_REPEATS = 5


def _check_stat(obj: Any, where: str, errors: List[str]) -> None:
    if not isinstance(obj, dict):
        errors.append(f"{where}: expected object with {STAT_KEYS}")
        return
    ok = True
    for key in STAT_KEYS:
        if key not in obj:
            errors.append(f"{where}: missing '{key}'")
            ok = False
        elif not isinstance(obj[key], (int, float)) or isinstance(obj[key], bool):
            errors.append(f"{where}.{key}: must be a number")
            ok = False
    if ok and obj["q1"] > obj["q3"]:
        errors.append(f"{where}: q1 > q3")


def validate_summary(
    summary: Dict[str, Any],
    base_dir: Path = None,
    canonical: bool = False,
) -> List[str]:
    """Return a list of schema violations (empty list means valid)."""
    errors: List[str] = []
    if not isinstance(summary, dict):
        return ["summary must be a JSON object"]

    if summary.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be '{SCHEMA_VERSION}', got {summary.get('schema_version')!r}"
        )
    if not isinstance(summary.get("created_utc"), str):
        errors.append("created_utc: missing or not a string")

    env = summary.get("environment")
    if not isinstance(env, dict):
        errors.append("environment: missing")
    else:
        for key in ENV_KEYS:
            if key not in env:
                errors.append(f"environment: missing '{key}'")
        libs = env.get("libraries")
        if isinstance(libs, dict):
            for lib in ("numpy", "hnswlib", "zgq"):
                if lib not in libs:
                    errors.append(f"environment.libraries: missing '{lib}'")

    proto = summary.get("protocol")
    if not isinstance(proto, dict):
        errors.append("protocol: missing")
    else:
        for key in PROTOCOL_KEYS:
            if key not in proto:
                errors.append(f"protocol: missing '{key}'")
        if canonical:
            if proto.get("n_queries", 0) < MIN_CANONICAL_QUERIES:
                errors.append(
                    f"protocol.n_queries < {MIN_CANONICAL_QUERIES} (canonical)"
                )
            if proto.get("repeats", 0) < MIN_CANONICAL_REPEATS:
                errors.append(f"protocol.repeats < {MIN_CANONICAL_REPEATS} (canonical)")

    scales = summary.get("scales")
    if not isinstance(scales, dict) or not scales:
        errors.append("scales: must be a non-empty object")
        return errors

    for scale_name, scale in scales.items():
        sw = f"scales.{scale_name}"
        for key in ("n_vectors", "dimension", "indexes"):
            if key not in scale:
                errors.append(f"{sw}: missing '{key}'")
        indexes = scale.get("indexes")
        if not isinstance(indexes, list) or not indexes:
            errors.append(f"{sw}.indexes: must be a non-empty list")
            continue
        labels = set()
        for idx in indexes:
            label = idx.get("label")
            iw = f"{sw}.indexes[{label}]"
            if not isinstance(label, str):
                errors.append(f"{sw}: index without string label")
                continue
            if label in labels:
                errors.append(f"{iw}: duplicate label")
            labels.add(label)
            if idx.get("algo") not in ALGOS:
                errors.append(f"{iw}: algo must be one of {ALGOS}")
            if not isinstance(idx.get("build_params"), dict):
                errors.append(f"{iw}: missing build_params")
            build = idx.get("build")
            if not isinstance(build, dict):
                errors.append(f"{iw}: missing build")
            else:
                for key in ("time_s", "rss_delta_mb"):
                    if not isinstance(build.get(key), (int, float)):
                        errors.append(f"{iw}.build: '{key}' must be a number")
            # build time must live only on the index, never on sweep points
            points = idx.get("points")
            if not isinstance(points, list) or not points:
                errors.append(f"{iw}.points: must be a non-empty list")
                continue
            for pt in points:
                pw = f"{iw}.points[{pt.get('search_param')}]"
                if "build_time_s" in pt or "build" in pt:
                    errors.append(f"{pw}: build data must not be repeated per point")
                if not isinstance(pt.get("search_param"), dict):
                    errors.append(f"{pw}: missing search_param")
                for metric in ("recall_pct", "latency_ms_per_query", "qps"):
                    _check_stat(pt.get(metric), f"{pw}.{metric}", errors)
            raw = idx.get("raw_file")
            if not isinstance(raw, str):
                errors.append(f"{iw}: missing raw_file")
            elif base_dir is not None:
                raw_path = base_dir / raw
                if not raw_path.is_file():
                    errors.append(f"{iw}: raw file not found: {raw}")
                else:
                    with open(raw_path) as f:
                        records = json.load(f)
                    n_expected = (
                        sum(proto.get("repeats", 0) for _ in points)
                        if isinstance(proto, dict)
                        else 0
                    )
                    if not isinstance(records, dict) or "runs" not in records:
                        errors.append(f"{iw}: raw file lacks 'runs'")
                    elif len(records["runs"]) != n_expected:
                        errors.append(
                            f"{iw}: raw has {len(records['runs'])} runs, expected {n_expected}"
                        )
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate ZGQ benchmark summary.json")
    parser.add_argument("summary", type=Path)
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Also enforce >=1000 queries and >=5 repeats",
    )
    args = parser.parse_args(argv)

    with open(args.summary) as f:
        summary = json.load(f)
    errors = validate_summary(
        summary, base_dir=args.summary.parent, canonical=args.canonical
    )
    if errors:
        print(f"INVALID: {args.summary}")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK: {args.summary} (schema {SCHEMA_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
