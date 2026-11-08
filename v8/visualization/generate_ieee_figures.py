#!/usr/bin/env python3
"""Figure generator for ZGQ v8. Reads only a results summary.json.

No benchmarking happens here. Every plotted number comes from the
``zgq-bench/1`` summary produced by ``benchmarks/run_benchmarks.py``.

Usage:
    python -m visualization.generate_ieee_figures
    python -m visualization.generate_ieee_figures --results results/2026-11-08/summary.json \
        --output figures --format pdf
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IEEE_SINGLE_COL = 3.5
IEEE_DOUBLE_COL = 7.16
IEEE_DPI = 300

STYLE = {
    "HNSW-M16": dict(color="#A23B72", marker="o", ls="-"),
    "ZGQ-M16": dict(color="#2E86AB", marker="s", ls="-"),
    "ZGQ-M24": dict(color="#6BA8C9", marker="^", ls="--"),
    "ZGQ-M32": dict(color="#9CCBE3", marker="v", ls=":"),
    "IVFPQ": dict(color="#6C757D", marker="x", ls="-"),
}
DEFAULT_RESULTS = (
    Path(__file__).resolve().parent.parent / "results" / "2026-11-08" / "summary.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "figures"
SCALE_ORDER = ["10k", "100k", "1m"]


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
            "savefig.dpi": IEEE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def load_summary(path: Path) -> Dict:
    with open(path) as f:
        data = json.load(f)
    if data.get("schema_version") != "zgq-bench/1":
        raise ValueError(
            f"{path}: unsupported schema_version {data.get('schema_version')!r}"
        )
    return data


def _scales(summary: Dict) -> List[str]:
    return [s for s in SCALE_ORDER if s in summary["scales"]]


def _label(n: int) -> str:
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}K"


def _style(label: str) -> Dict:
    return STYLE.get(label, dict(color="black", marker=".", ls="-"))


def _graph_indexes(scale: Dict) -> List[Dict]:
    return [i for i in scale["indexes"] if i["algo"] in ("hnsw", "zgq")]


def _xy(index: Dict, xkey: str, ykey: str):
    pts = index["points"]
    return ([p[xkey]["median"] for p in pts], [p[ykey]["median"] for p in pts])


def _save(fig, output_dir: Path, name: str, fmt: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.{fmt}"
    fig.savefig(path, format=fmt)
    plt.close(fig)
    return path


def _footer(fig, summary: Dict) -> None:
    p = summary["protocol"]
    fig.text(
        0.5,
        -0.02,
        f"Synthetic clustered data, dim {p.get('dimension', 128)}, "
        f"{p.get('n_queries', '?')} queries, single thread, median of repeats. "
        "ZGQ is Python+numba+hnswlib; HNSW is hnswlib (C++).",
        ha="center",
        fontsize=6,
        color="#555555",
    )


def _frontier(
    summary: Dict,
    output_dir: Path,
    fmt: str,
    xkey: str,
    ykey: str,
    xlabel: str,
    ylabel: str,
    name: str,
    logx: bool = False,
) -> Path:
    scales = _scales(summary)
    fig, axes = plt.subplots(
        1, len(scales), figsize=(IEEE_DOUBLE_COL, 2.8), squeeze=False
    )
    for ax, s in zip(axes[0], scales):
        sc = summary["scales"][s]
        for idx in _graph_indexes(sc):
            x, y = _xy(idx, xkey, ykey)
            ax.plot(x, y, label=idx["label"], markersize=4, **_style(idx["label"]))
        ax.set_title(f"{_label(sc['n_vectors'])} vectors")
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(loc="best")
    _footer(fig, summary)
    fig.tight_layout()
    return _save(fig, output_dir, name, fmt)


def fig_recall_vs_qps(summary: Dict, output_dir: Path, fmt: str = "pdf") -> Path:
    return _frontier(
        summary,
        output_dir,
        fmt,
        "recall_pct",
        "qps",
        "Recall@10 (%)",
        "Queries per second",
        "fig_recall_vs_qps",
    )


def fig_latency_comparison(summary: Dict, output_dir: Path, fmt: str = "pdf") -> Path:
    return _frontier(
        summary,
        output_dir,
        fmt,
        "recall_pct",
        "latency_ms_per_query",
        "Recall@10 (%)",
        "Latency (ms/query)",
        "fig_latency_comparison",
    )


def fig_pareto_frontier(
    summary: Dict, output_dir: Path, fmt: str = "pdf", scale: Optional[str] = None
) -> Path:
    """Recall vs latency at one scale (default: 100k), all indexes including IVF-PQ."""
    scale = scale or ("100k" if "100k" in summary["scales"] else _scales(summary)[0])
    sc = summary["scales"][scale]
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL * 1.4, 3.0))
    for idx in sc["indexes"]:
        x, y = _xy(idx, "recall_pct", "latency_ms_per_query")
        ax.plot(x, y, label=idx["label"], markersize=4, **_style(idx["label"]))
    ax.set_yscale("log")
    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("Latency (ms/query, log)")
    ax.set_title(f"Recall vs latency, {_label(sc['n_vectors'])} vectors")
    ax.legend(loc="best")
    _footer(fig, summary)
    fig.tight_layout()
    return _save(fig, output_dir, "fig_pareto_frontier", fmt)


def fig_build_time(summary: Dict, output_dir: Path, fmt: str = "pdf") -> Path:
    scales = _scales(summary)
    labels = [i["label"] for i in _graph_indexes(summary["scales"][scales[0]])]
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL * 1.4, 3.0))
    width = 0.8 / len(labels)
    for j, lab in enumerate(labels):
        vals = []
        for s in scales:
            idx = next(i for i in summary["scales"][s]["indexes"] if i["label"] == lab)
            vals.append(idx["build"]["time_s"])
        xs = [k + (j - (len(labels) - 1) / 2) * width for k in range(len(scales))]
        ax.bar(xs, vals, width, label=lab, color=_style(lab)["color"])
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([_label(summary["scales"][s]["n_vectors"]) for s in scales])
    ax.set_yscale("log")
    ax.set_ylabel("Build time (s, log)")
    ax.set_title("Index build time (single thread)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, output_dir, "fig_build_time", fmt)


def fig_scaling_analysis(
    summary: Dict, output_dir: Path, fmt: str = "pdf", ef: int = 64
) -> Path:
    scales = _scales(summary)
    ns = [summary["scales"][s]["n_vectors"] for s in scales]
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL * 1.4, 3.0))
    for idx0 in _graph_indexes(summary["scales"][scales[0]]):
        lab = idx0["label"]
        ys = []
        for s in scales:
            idx = next(i for i in summary["scales"][s]["indexes"] if i["label"] == lab)
            pt = next(p for p in idx["points"] if p["search_param"].get("ef") == ef)
            ys.append(pt["recall_pct"]["median"])
        ax.plot(ns, ys, label=lab, markersize=4, **_style(lab))
    ax.set_xscale("log")
    ax.set_xlabel("Dataset size (vectors)")
    ax.set_ylabel(f"Recall@10 (%) at ef={ef}")
    ax.set_title("Recall vs dataset size at equal ef")
    ax.legend(loc="best")
    fig.tight_layout()
    return _save(fig, output_dir, "fig_scaling_analysis", fmt)


def fig_ivf_pq_latency(summary: Dict, output_dir: Path, fmt: str = "pdf") -> Path:
    """Latency at the highest-recall setting of each index, per scale."""
    scales = _scales(summary)
    labels = [i["label"] for i in summary["scales"][scales[0]]["indexes"]]
    fig, ax = plt.subplots(figsize=(IEEE_DOUBLE_COL * 0.8, 3.0))
    width = 0.8 / len(labels)
    for j, lab in enumerate(labels):
        vals, recs = [], []
        for s in scales:
            idx = next(i for i in summary["scales"][s]["indexes"] if i["label"] == lab)
            best = max(idx["points"], key=lambda p: p["recall_pct"]["median"])
            vals.append(best["latency_ms_per_query"]["median"])
            recs.append(best["recall_pct"]["median"])
        xs = [k + (j - (len(labels) - 1) / 2) * width for k in range(len(scales))]
        ax.bar(xs, vals, width, label=lab, color=_style(lab)["color"])
        for x, v, r in zip(xs, vals, recs):
            ax.text(x, v, f"{r:.0f}%", ha="center", va="bottom", fontsize=5)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([_label(summary["scales"][s]["n_vectors"]) for s in scales])
    ax.set_yscale("log")
    ax.set_ylabel("Latency (ms/query, log)")
    ax.set_title("Latency at each index's highest-recall setting (recall labeled)")
    ax.legend(loc="upper left", ncol=2)
    _footer(fig, summary)
    fig.tight_layout()
    return _save(fig, output_dir, "fig_ivf_pq_latency", fmt)


def fig_ivf_pq_tradeoff(summary: Dict, output_dir: Path, fmt: str = "pdf") -> Path:
    """Recall vs QPS for HNSW-M16, ZGQ-M16 and IVF-PQ at every scale."""
    scales = _scales(summary)
    fig, ax = plt.subplots(figsize=(IEEE_SINGLE_COL * 1.4, 3.0))
    shades = {"10k": 1.0, "100k": 0.65, "1m": 0.35}
    for s in scales:
        for idx in summary["scales"][s]["indexes"]:
            if idx["label"] not in ("HNSW-M16", "ZGQ-M16", "IVFPQ"):
                continue
            x, y = _xy(idx, "recall_pct", "qps")
            st = _style(idx["label"])
            ax.plot(
                x,
                y,
                color=st["color"],
                marker=st["marker"],
                ls=st["ls"],
                alpha=shades.get(s, 1.0),
                markersize=3,
                label=f"{idx['label']} {_label(summary['scales'][s]['n_vectors'])}",
            )
    ax.set_yscale("log")
    ax.set_xlabel("Recall@10 (%)")
    ax.set_ylabel("Queries per second (log)")
    ax.set_title("Recall vs throughput (darker = smaller scale)")
    ax.legend(loc="best", ncol=2, fontsize=5)
    _footer(fig, summary)
    fig.tight_layout()
    return _save(fig, output_dir, "fig_ivf_pq_tradeoff", fmt)


FIGURES = [
    fig_recall_vs_qps,
    fig_latency_comparison,
    fig_pareto_frontier,
    fig_build_time,
    fig_scaling_analysis,
    fig_ivf_pq_latency,
    fig_ivf_pq_tradeoff,
]


def generate_all_figures(
    results_path: Path = DEFAULT_RESULTS,
    output_dir: Path = DEFAULT_OUTPUT,
    fmt: str = "pdf",
) -> List[Path]:
    setup_style()
    summary = load_summary(Path(results_path))
    return [fn(summary, Path(output_dir), fmt) for fn in FIGURES]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate ZGQ figures from results/summary.json"
    )
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--format", "-f", default="pdf", choices=["pdf", "png", "svg"])
    args = parser.parse_args(argv)
    for p in generate_all_figures(args.results, args.output, args.format):
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
