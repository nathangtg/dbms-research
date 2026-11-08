"""
Generate comparison.md and figures from a build-order study_summary.json.

Every number comes from the summary file; nothing is typed by hand.

    python -m benchmarks.build_order_report results/<date>-novelty/study_summary.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from benchmarks.build_order_study import SCHEMA, check_summary

STRUCT_COLS = (
    ("reachable_fraction_from_entry", "reachable"),
    ("cross_zone_edge_fraction", "cross-zone"),
    ("long_edge_fraction", "long-edge"),
)


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _med(s, nd=3):
    return "n/a" if not s else f"{s['median']:.{nd}f} ({s['iqr']:.{nd}f})"


def _by_exp(summary, exp) -> List[Dict]:
    return [g for g in summary["groups"] if g["experiment"] == exp]


def _scopes(groups):
    seen = []
    for g in groups:
        key = (g["dataset"], g["n_base"], g["threads"])
        if key not in seen:
            seen.append(key)
    return seen


def _ci(d):
    if not d:
        return "n/a"
    return f"{d['diff']:+.2f} [{d['ci_low']:+.2f}, {d['ci_high']:+.2f}] ({d['method']}, n={d['n']})"


def _table(header, rows) -> List[str]:
    out = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out + [""]


def build_markdown(summary: Dict) -> str:
    L = [
        "# Build-order study",
        "",
        f"Generated from `study_summary.json` (schema `{summary['schema']}`, "
        f"{summary['n_runs']} runs). Medians over seeds with IQR in parentheses.",
        "",
        summary.get("note", ""),
        "",
    ]
    e1 = _by_exp(summary, "e1")

    L += ["## Recall@10 (%) by ef", ""]
    for ds, n, th in _scopes(e1):
        gs = [g for g in e1 if (g["dataset"], g["n_base"], g["threads"]) == (ds, n, th)]
        efs = gs[0]["ef"]
        L += [f"### {ds}, n={n}", ""]
        L += _table(
            ["strategy", "seeds"] + [f"ef={e}" for e in efs],
            [
                [g["strategy"], g["n_seeds"]]
                + [_med(g["recall_pct"][str(e)], 2) for e in efs]
                for g in gs
            ],
        )

    L += [
        "## Recall gap vs random at ef=256 (percentage points, strategy minus random)",
        "",
    ]
    rows = []
    for g in e1:
        d = (g.get("paired_vs_random") or {}).get("recall_pct", {}).get("256")
        if g["strategy"] != "random":
            rows.append([g["dataset"], g["n_base"], g["strategy"], _ci(d)])
    L += _table(["dataset", "n", "strategy", "difference [95% CI]"], rows)

    L += [
        "## Equal-recall latency (ms/query, linear interpolation; n/a = target not reached)",
        "",
    ]
    targets = sorted({t for g in e1 for t in g["equal_recall_latency_ms"]}, key=float)
    L += _table(
        ["dataset", "n", "strategy"] + [f"{t}% recall" for t in targets],
        [
            [g["dataset"], g["n_base"], g["strategy"]]
            + [_f(g["equal_recall_latency_ms"].get(t), 4) for t in targets]
            for g in e1
        ],
    )

    L += [
        "## Build time (s, add_items only) and zone time",
        "",
        "Timings are valid only when `concurrent jobs` is 0. Larger values mean other "
        "study processes shared the machine and timings are contaminated.",
        "",
    ]
    L += _table(
        [
            "dataset",
            "n",
            "strategy",
            "threads",
            "build s",
            "order s",
            "zone s",
            "concurrent jobs (max)",
        ],
        [
            [
                g["dataset"],
                g["n_base"],
                g["strategy"],
                g["threads"],
                _med(g["build_time_s"]),
                _med(g["order_time_s"], 4),
                _med(g["zone_time_s"]),
                g["concurrent_jobs_max"],
            ]
            for g in summary["groups"]
            if g["experiment"] in ("e1", "e2")
        ],
    )

    L += ["## Graph structure (level 0, median over seeds)", ""]
    L += _table(
        ["dataset", "n", "strategy"]
        + [c[1] for c in STRUCT_COLS]
        + ["hops L0 @ef64", "hops L0 @ef256"],
        [
            [g["dataset"], g["n_base"], g["strategy"]]
            + [_f(g["structure"].get(c[0]), 4) for c in STRUCT_COLS]
            + [
                _f(g["hops"].get(ef, {}).get("hops_level0.mean"), 1)
                for ef in ("64", "256")
            ]
            for g in e1
        ],
    )

    e2 = _by_exp(summary, "e2")
    if e2:
        L += ["## E2: multi-thread build", ""]
        L += _table(
            [
                "dataset",
                "n",
                "strategy",
                "threads",
                "build s",
                "recall@ef64",
                "recall@ef256",
                "concurrent jobs (max)",
            ],
            [
                [
                    g["dataset"],
                    g["n_base"],
                    g["strategy"],
                    g["threads"],
                    _med(g["build_time_s"]),
                    _med(g.get("build_repeat_recall_pct", {}).get("64")),
                    _med(g.get("build_repeat_recall_pct", {}).get("256")),
                    g["concurrent_jobs_max"],
                ]
                for g in e2
            ],
        )
    e3 = _by_exp(summary, "e3")
    if e3:
        L += ["## E3: incremental insertion of the last 20%", ""]
        L += _table(
            [
                "dataset",
                "n",
                "arrival order",
                "insert s",
                "recall@ef64",
                "recall@ef256",
                "gap@ef256 vs random arrivals",
            ],
            [
                [
                    g["dataset"],
                    g["n_base"],
                    g["strategy"],
                    _med(g.get("insertion_time_s")),
                    _med(g["recall_pct"].get("64"), 2),
                    _med(g["recall_pct"].get("256"), 2),
                    _ci(
                        ((g.get("paired_vs_random") or {}).get("recall_pct") or {}).get(
                            "256"
                        )
                    ),
                ]
                for g in e3
            ],
        )
    return "\n".join(L) + "\n"


def make_figures(summary: Dict, out_dir: Path) -> List[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    e1 = _by_exp(summary, "e1")
    for ds, n, th in _scopes(e1):
        gs = [g for g in e1 if (g["dataset"], g["n_base"], g["threads"]) == (ds, n, th)]
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for g in gs:
            ax.plot(
                g["ef"],
                [g["recall_pct"][str(e)]["median"] for e in g["ef"]],
                marker="o",
                label=g["strategy"],
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("ef")
        ax.set_ylabel("recall@10 (%)")
        ax.set_title(f"{ds}, n={n}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f"recall_vs_ef_{ds}_n{n}.png"
        fig.savefig(p, dpi=150)
        fig.savefig(p.with_suffix(".pdf"))
        plt.close(fig)
        made.append(p)

    for ds in sorted(
        {
            g["dataset"].split("-")[0]
            if g["dataset"].startswith("synthetic")
            else g["dataset"]
            for g in e1
        }
    ):
        gs = [
            g
            for g in e1
            if (
                g["dataset"].startswith("synthetic")
                and ds == "synthetic"
                or g["dataset"] == ds
            )
            and g["strategy"] != "random"
        ]
        if not gs:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for s in sorted({g["strategy"] for g in gs}):
            pts = sorted(
                (g["n_base"], g["paired_vs_random"]["recall_pct"]["256"])
                for g in gs
                if g["strategy"] == s
                and g.get("paired_vs_random")
                and "256" in g["paired_vs_random"]["recall_pct"]
            )
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ax.errorbar(
                xs,
                [p[1]["diff"] for p in pts],
                yerr=[
                    [p[1]["diff"] - p[1]["ci_low"] for p in pts],
                    [p[1]["ci_high"] - p[1]["diff"] for p in pts],
                ],
                marker="o",
                capsize=2,
                label=s,
            )
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("n")
        ax.set_ylabel("recall gap vs random at ef=256 (pp)")
        ax.set_title(ds)
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / f"recall_gap_vs_scale_{ds}.png"
        fig.savefig(p, dpi=150)
        fig.savefig(p.with_suffix(".pdf"))
        plt.close(fig)
        made.append(p)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary")
    ap.add_argument("--out-dir", default=None, help="default: directory of the summary")
    args = ap.parse_args(argv)
    summary = json.loads(Path(args.summary).read_text())
    errs = check_summary(summary)
    if errs:
        print(errs)
        return 1
    out = Path(args.out_dir or Path(args.summary).parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.md").write_text(build_markdown(summary))
    figs = make_figures(summary, out)
    print(f"wrote {out / 'comparison.md'} and {len(figs)} figures ({SCHEMA})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
