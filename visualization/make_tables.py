#!/usr/bin/env python3
"""Render Markdown and LaTeX result tables from a results summary.json only.

    python -m visualization.make_tables
    python -m visualization.make_tables --results results/2026-11-08/summary.json --output tables
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

DEFAULT_RESULTS = Path(__file__).resolve().parent.parent / "results" / "2026-11-08" / "summary.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "tables"
SCALE_ORDER = ["10k", "100k", "1m"]
EFS = (64, 256)


def load_summary(path: Path) -> Dict:
    with open(path) as f:
        data = json.load(f)
    if data.get("schema_version") != "zgq-bench/1":
        raise ValueError(f"{path}: unsupported schema_version {data.get('schema_version')!r}")
    return data


def _size(n: int) -> str:
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}K"


def _point(index: Dict, key: str, value: int):
    for p in index["points"]:
        if p["search_param"].get(key) == value:
            return p
    return None


def _cell(p) -> str:
    if p is None:
        return "n/a"
    return f"{p['recall_pct']['median']:.1f} / {p['latency_ms_per_query']['median']:.3f}"


def result_rows(summary: Dict, efs=EFS) -> List[List[str]]:
    """Rows: scale, index, comparison, build s, then recall/latency per ef; IVF-PQ at best recall."""
    rows = []
    for s in [k for k in SCALE_ORDER if k in summary["scales"]]:
        sc = summary["scales"][s]
        for idx in sc["indexes"]:
            if idx["algo"] == "ivfpq":
                best = max(idx["points"], key=lambda p: p["recall_pct"]["median"])
                cells = [_cell(best) + f" (nprobe={best['search_param']['nprobe']})"] + ["-"] * (len(efs) - 1)
            else:
                cells = [_cell(_point(idx, "ef", ef)) for ef in efs]
            rows.append([_size(sc["n_vectors"]), idx["label"], idx["comparison"],
                         f"{idx['build']['time_s']:.1f}"] + cells)
    return rows


def _header(efs=EFS) -> List[str]:
    return ["Scale", "Index", "Role", "Build (s)"] + [f"ef={e}: recall % / ms" for e in efs]


def to_markdown(summary: Dict) -> str:
    head = _header()
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(r) + " |" for r in result_rows(summary)]
    return "\n".join(out) + "\n"


def to_latex(summary: Dict) -> str:
    head = _header()

    def esc(text: str) -> str:
        return text.replace("%", r"\%").replace("_", r"\_")

    lines = [r"\begin{tabular}{lllr" + "l" * (len(head) - 4) + "}", r"\hline",
             " & ".join(esc(h) for h in head) + r" \\", r"\hline"]
    lines += [" & ".join(esc(c) for c in r) + r" \\" for r in result_rows(summary)]
    lines += [r"\hline", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def write_tables(results: Path = DEFAULT_RESULTS, output: Path = DEFAULT_OUTPUT) -> List[Path]:
    summary = load_summary(Path(results))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    md, tex = output / "results_table.md", output / "results_table.tex"
    md.write_text(to_markdown(summary))
    tex.write_text(to_latex(summary))
    return [md, tex]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    for p in write_tables(args.results, args.output):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
