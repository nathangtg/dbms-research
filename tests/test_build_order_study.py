import json

import numpy as np
import pytest
from benchmarks import build_order_study as bos

EF = [16, 64, 256]


def _cfg(strategy, seed=0, **kw):
    cfg = bos.make_config(
        experiment="e1",
        dataset="synthetic",
        n_base=2000,
        strategy=strategy,
        seed=seed,
        ef=EF,
        n_queries=100,
        repeats=2,
        **kw,
    )
    return cfg


@pytest.fixture(scope="module")
def runs():
    return {
        s: bos.run_one(_cfg(s, skeleton_fraction=0.05))
        for s in ("zone", "random", "skeleton")
    }


def test_run_has_required_keys_and_validates(runs):
    for name, run in runs.items():
        assert bos.check_run(run) == [], name
        assert run["schema"] == bos.SCHEMA == "zgq-build-order/1"
        for key in (
            "environment",
            "dataset",
            "config",
            "build_time_s",
            "zone_time_s",
            "order_time_s",
            "n_zones",
            "index_file_bytes",
            "analytic_bytes",
            "points",
            "graph_stats",
            "hops",
            "concurrent_jobs",
        ):
            assert key in run, key
        assert run["config"]["strategy"] == name
        assert set(run["hops"]) == {"64", "256"}
        json.dumps(run)


def test_recall_at_largest_ef_is_high(runs):
    for run in runs.values():
        last = run["points"][-1]
        assert last["ef"] == max(EF)
        assert last["recall_pct"]["median"] > 90.0


def test_checker_rejects_bad_run(runs):
    bad = dict(runs["zone"])
    bad["schema"] = "other"
    assert bos.check_run(bad)
    bad = dict(runs["zone"])
    del bad["points"]
    assert bos.check_run(bad)


def test_data_seed_is_independent_of_run_seed(monkeypatch):
    seen = []

    real = bos.load_dataset

    def spy(name, n_base, n_queries, k, seed):
        seen.append(seed)
        return real(name, n_base, n_queries, k, seed)

    monkeypatch.setattr(bos, "load_dataset", spy)
    bos.run_one(_cfg("random", seed=3))
    bos.run_one(_cfg("random", seed=7))
    assert seen == [0, 0]


def test_seed_changes_random_order():
    v = np.zeros((50, 4), dtype=np.float32)
    a = bos.compute_order("random", v, None, None, seed=1, skeleton_fraction=0.05)
    b = bos.compute_order("random", v, None, None, seed=2, skeleton_fraction=0.05)
    assert not np.array_equal(a, b)


def test_resume_skips_existing_and_writes_atomically(tmp_path, monkeypatch):
    cfg = _cfg("random")
    path = bos.execute(cfg, str(tmp_path))
    assert path.exists() and not list(tmp_path.rglob("*.tmp"))
    mtime = path.stat().st_mtime_ns
    monkeypatch.setattr(bos, "run_one", lambda c: pytest.fail("should skip"))
    assert bos.execute(cfg, str(tmp_path)) == path
    assert path.stat().st_mtime_ns == mtime


def _fake_run(strategy, seed, shift, n_q=50, exp="e1"):
    rng = np.random.RandomState(seed * 7 + (1 if strategy != "random" else 0))
    points = []
    hits_all = {}
    for ef, base in zip(EF, (0.6, 0.85, 0.97)):
        p = min(1.0, base + shift)
        hits = (rng.rand(n_q, 10) < p).sum(axis=1)
        hits_all[str(ef)] = hits.tolist()
        r = float(hits.sum()) / (n_q * 10) * 100
        lat = 0.01 * ef**0.5

        def stat(v):
            return {"median": v, "q1": v, "q3": v, "iqr": 0.0, "n": 1}

        points.append(
            {"ef": ef, "recall_pct": stat(r), "latency_ms_per_query": stat(lat)}
        )
    return {
        "schema": bos.SCHEMA,
        "kind": "run",
        "config": {
            "experiment": exp,
            "dataset": "synthetic-2000",
            "n_base": 2000,
            "strategy": strategy,
            "skeleton_fraction": 0.05,
            "seed": seed,
            "threads": 1,
        },
        "strategy_key": strategy,
        "concurrent_jobs": 0,
        "build_time_s": 1.0 + seed,
        "zone_time_s": 0.5,
        "order_time_s": 0.01,
        "points": points,
        "per_query_hits": hits_all,
        "graph_stats": {"reachable_fraction_from_entry": 1.0},
        "hops": {"64": {"hops_level0": {"mean": 10.0}}},
    }


def test_aggregate_multi_seed_uses_seed_bootstrap_and_median_iqr():
    runs = [
        _fake_run(s, seed, sh)
        for seed in (0, 1, 2)
        for s, sh in (("random", 0.0), ("zone", 0.02))
    ]
    summary = bos.aggregate(runs)
    assert bos.check_summary(summary) == []
    grp = {g["strategy"]: g for g in summary["groups"]}
    assert grp["zone"]["n_seeds"] == 3
    r = grp["zone"]["recall_pct"]["256"]
    assert r["q1"] <= r["median"] <= r["q3"]
    d = grp["zone"]["paired_vs_random"]["recall_pct"]["256"]
    assert d["method"] == "bootstrap over seeds"
    assert d["ci_low"] <= d["diff"] <= d["ci_high"] or d["ci_low"] <= d["ci_high"]
    assert (
        "paired_vs_random" not in grp["random"]
        or grp["random"]["paired_vs_random"] is None
    )


def test_aggregate_single_seed_bootstraps_over_queries():
    runs = [_fake_run("random", 0, 0.0, n_q=400), _fake_run("zone", 0, 0.05, n_q=400)]
    summary = bos.aggregate(runs)
    grp = {g["strategy"]: g for g in summary["groups"]}
    d = grp["zone"]["paired_vs_random"]["recall_pct"]["64"]
    assert d["method"] == "bootstrap over queries"
    assert d["diff"] > 0 and d["ci_low"] < d["ci_high"]


def test_equal_recall_latency_interpolates():
    recalls = [60.0, 80.0, 100.0]
    lats = [1.0, 2.0, 4.0]
    assert bos.latency_at_recall(recalls, lats, 70.0) == pytest.approx(1.5)
    assert bos.latency_at_recall(recalls, lats, 90.0) == pytest.approx(3.0)
    assert bos.latency_at_recall(recalls, lats, 101.0) is None
    assert bos.latency_at_recall(recalls, lats, 50.0) == pytest.approx(1.0)


def test_check_summary_rejects_wrong_schema():
    summary = bos.aggregate([_fake_run("random", 0, 0.0)])
    summary["schema"] = "zgq-build-order/0"
    assert bos.check_summary(summary)


def test_e3_runs_and_records_insertion_time():
    cfg = bos.make_config(
        experiment="e3",
        dataset="synthetic",
        n_base=2000,
        strategy="e3_zone",
        seed=0,
        ef=[64, 256],
        n_queries=100,
        repeats=1,
    )
    run = bos.run_one(cfg)
    assert bos.check_run(run) == []
    assert run["insertion_time_s"] > 0
    assert run["points"][-1]["recall_pct"]["median"] > 80.0


def test_report_generates_markdown_and_figures(tmp_path):
    from benchmarks import build_order_report as rep

    runs = [_fake_run(s, seed, sh) for seed in (0, 1)
            for s, sh in (("random", 0.0), ("zone", 0.02))]
    path = tmp_path / "study_summary.json"
    path.write_text(json.dumps(bos.aggregate(runs)))
    assert rep.main([str(path)]) == 0
    text = (tmp_path / "comparison.md").read_text()
    for section in ("Recall@10", "Equal-recall latency", "concurrent jobs", "Graph structure"):
        assert section in text
