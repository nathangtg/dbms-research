"""Compare opt-in ZGQ variants against the matched HNSW-M16 from one summary.json.

Reads only a committed ``summary.json`` (schema zgq-bench/1) and prints Markdown:

* equal-ef table for variants swept over ``ef`` (recall delta, latency ratio),
* equal-recall table (latency at a target recall by linear interpolation along
  each recall/latency curve; "not reached" when the curve tops out below it),
* build-time ratio and analytic index bytes ratio (RSS is never used here).

Usage: ``python -m benchmarks.compare_improvements results/<dir>/summary.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Optional

BASE = "HNSW-M16"
TARGETS = (80.0, 90.0, 95.0)
# Analytic bytes of the default ZGQ index equal those of the "+random" variant
# (same stored components, only insertion order differs).
ZGQ_BYTES_FROM = "ZGQ-M16+random"


def _curve(entry: Dict) -> List[tuple]:
    pts = []
    for p in entry["points"]:
        ((name, val),) = p["search_param"].items()
        pts.append(
            (p["recall_pct"]["median"], p["latency_ms_per_query"]["median"], name, val)
        )
    return sorted(pts)


def latency_at_recall(entry: Dict, target: float) -> Optional[float]:
    pts = _curve(entry)
    if pts[-1][0] < target:
        return None
    if pts[0][0] >= target:
        return pts[0][1]
    for (r0, l0, _, _), (r1, l1, _, _) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            if r1 == r0:
                return l1
            return l0 + (l1 - l0) * (target - r0) / (r1 - r0)
    return None


def _by_label(indexes: List[Dict]) -> Dict[str, Dict]:
    return {i["label"]: i for i in indexes}


def report(summary: Dict) -> str:
    out: List[str] = []
    for scale, sc in summary["scales"].items():
        idx = _by_label(sc["indexes"])
        base = idx[BASE]
        base_bytes = base["analytic_bytes"]["total_bytes"]
        zgq_bytes = idx[ZGQ_BYTES_FROM]["analytic_bytes"]["total_bytes"]
        out.append(f"### {scale} ({sc['n_vectors']:,} vectors)\n")
        out.append("Build time and analytic index bytes vs HNSW-M16:\n")
        out.append("| Index | Build s | Build ratio | Analytic MB | Bytes ratio |")
        out.append("|---|---:|---:|---:|---:|")
        for lab, e in idx.items():
            if e["algo"] == "ivfpq":
                continue
            b = e.get("analytic_bytes", {}).get("total_bytes")
            if b is None and lab == "ZGQ-M16":
                b = zgq_bytes
            bt = e["build"]["time_s"]
            out.append(
                f"| {lab} | {bt:.1f} | {bt / base['build']['time_s']:.2f} | "
                f"{b / 1e6:.1f} | {b / base_bytes:.2f} |"
            )
        out.append("")
        out.append(
            "Equal ef (variants swept over ef), recall delta in points and latency ratio:\n"
        )
        efs = sorted(p["search_param"]["ef"] for p in base["points"])
        out.append("| Index | " + " | ".join(f"ef={e}" for e in efs) + " |")
        out.append("|---|" + "---|" * len(efs))
        bmap = {p["search_param"]["ef"]: p for p in base["points"]}
        for lab, e in idx.items():
            if e["algo"] == "ivfpq" or lab == BASE:
                continue
            if "ef" not in e["points"][0]["search_param"]:
                continue
            row = []
            for ef in efs:
                p = next(
                    (q for q in e["points"] if q["search_param"]["ef"] == ef), None
                )
                if p is None:
                    row.append("-")
                    continue
                b = bmap[ef]
                dr = p["recall_pct"]["median"] - b["recall_pct"]["median"]
                lr = (
                    p["latency_ms_per_query"]["median"]
                    / b["latency_ms_per_query"]["median"]
                )
                row.append(f"{dr:+.1f} pt, {lr:.1f}x")
            out.append(f"| {lab} | " + " | ".join(row) + " |")
        out.append("")
        out.append(
            "HNSW-M16 reference: "
            + ", ".join(
                f"ef={ef} recall {bmap[ef]['recall_pct']['median']:.1f}% "
                f"{bmap[ef]['latency_ms_per_query']['median']:.3f} ms"
                for ef in efs
            )
            + "\n"
        )
        out.append(
            "Equal recall: latency ms (ratio to HNSW-M16 at the same recall); "
            "max recall of each curve in the last column:\n"
        )
        out.append(
            "| Index | "
            + " | ".join(f"R={t:.0f}%" for t in TARGETS)
            + " | Max recall (latency) |"
        )
        out.append("|---|" + "---|" * (len(TARGETS) + 1))
        for lab, e in idx.items():
            if e["algo"] == "ivfpq":
                continue
            cells = []
            for t in TARGETS:
                lv = latency_at_recall(e, t)
                lb = latency_at_recall(base, t)
                if lv is None:
                    cells.append("not reached")
                elif lb is None:
                    cells.append(f"{lv:.3f} (HNSW n/a)")
                else:
                    cells.append(f"{lv:.3f} ({lv / lb:.1f}x)")
            top = _curve(e)[-1]
            out.append(
                f"| {lab} | "
                + " | ".join(cells)
                + f" | {top[0]:.1f}% ({top[1]:.3f} ms) |"
            )
        out.append("")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("summary")
    args = ap.parse_args(argv)
    with open(args.summary) as f:
        print(report(json.load(f)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
