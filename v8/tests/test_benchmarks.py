import copy
import json

import pytest

from benchmarks import run_benchmarks as rb
from benchmarks.check_results import main as check_main
from benchmarks.check_results import validate_summary


def _args(out_dir, seed=42):
    return rb.build_parser().parse_args([
        "--scale", "tiny", "--out-dir", str(out_dir), "--seed", str(seed),
        "--n-queries", "40", "--repeats", "2", "--ef", "32", "64",
        "--nprobe", "4", "16", "--zgq-m", "16", "24",
    ])


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    summary = rb.run(_args(out))
    return out, summary


def test_smoke_run_writes_valid_summary(smoke):
    out, summary = smoke
    assert (out / "summary.json").is_file()
    assert validate_summary(json.loads((out / "summary.json").read_text()), base_dir=out) == []
    labels = [i["label"] for i in summary["scales"]["tiny"]["indexes"]]
    assert "HNSW-M16" in labels and "ZGQ-M16" in labels and "ZGQ-M24" in labels
    assert check_main([str(out / "summary.json")]) == 0


def test_build_recorded_once_per_index(smoke):
    _, summary = smoke
    for idx in summary["scales"]["tiny"]["indexes"]:
        assert idx["build"]["time_s"] > 0
        for pt in idx["points"]:
            assert "build" not in pt and "build_time_s" not in pt


def test_raw_records_per_run(smoke):
    out, summary = smoke
    idx = summary["scales"]["tiny"]["indexes"][0]
    raw = json.loads((out / idx["raw_file"]).read_text())
    assert len(raw["runs"]) == len(idx["points"]) * 2


def test_schema_rejects_bad_summaries(smoke):
    _, summary = smoke
    bad = copy.deepcopy(summary)
    bad["schema_version"] = "nope"
    assert validate_summary(bad)
    bad = copy.deepcopy(summary)
    del bad["environment"]["git_commit"]
    assert validate_summary(bad)
    bad = copy.deepcopy(summary)
    bad["scales"]["tiny"]["indexes"][0]["points"][0]["build_time_s"] = 1.0
    assert validate_summary(bad)
    bad = copy.deepcopy(summary)
    del bad["scales"]["tiny"]["indexes"][0]["points"][0]["recall_pct"]["median"]
    assert validate_summary(bad)
    # smoke protocol is below the canonical floor
    assert validate_summary(summary, canonical=True)


def test_data_and_recall_deterministic(tmp_path, smoke):
    _, first = smoke
    second = rb.run(_args(tmp_path))

    def recalls(s):
        return {
            (i["label"], tuple(p["search_param"].items())): p["recall_pct"]["median"]
            for i in s["scales"]["tiny"]["indexes"] for p in i["points"]
        }

    assert recalls(first) == recalls(second)


def test_generate_data_seeded():
    v1, q1 = rb.generate_test_data(1000, 20, seed=7)
    v2, q2 = rb.generate_test_data(1000, 20, seed=7)
    v3, _ = rb.generate_test_data(1000, 30, seed=7)
    v4, _ = rb.generate_test_data(1000, 20, seed=8)
    assert (v1 == v2).all() and (q1 == q2).all()
    assert (v1 == v3).all()  # database does not depend on n_queries
    assert not (v1 == v4).all()


def test_default_specs_unchanged_without_improvements(smoke):
    _, summary = smoke
    labels = [i["label"] for i in summary["scales"]["tiny"]["indexes"]]
    assert labels == ["HNSW-M16", "ZGQ-M16", "ZGQ-M24", "IVFPQ"]


@pytest.fixture(scope="module")
def improved(tmp_path_factory):
    out = tmp_path_factory.mktemp("improve")
    args = rb.build_parser().parse_args([
        "--scale", "tiny", "--out-dir", str(out), "--n-queries", "30",
        "--repeats", "2", "--ef", "32", "64", "--zgq-m", "16", "--no-ivfpq",
        "--improvements", "--improve-nprobe", "2", "8", "--improve-ratio-pct", "110", "150",
    ])
    return out, rb.run(args)


def test_improvement_variants_present_and_valid(improved):
    out, summary = improved
    idx = {i["label"]: i for i in summary["scales"]["tiny"]["indexes"]}
    for label in ("HNSW-M16", "ZGQ-M16", "ZGQ-M16+random",
                  "ZGQ-M16+overlap2-nopq", "ZGQ-M16+overlap2", "ZGQ-M16+adaptive-nopq",
                  "ZGQ-M16+global-entry-nopq", "ZGQ-M16+random+global-entry-nopq"):
        assert label in idx, label
    assert validate_summary(json.loads((out / "summary.json").read_text()), base_dir=out) == []


def test_improvement_analytic_bytes_and_probes(improved):
    _, summary = improved
    idx = {i["label"]: i for i in summary["scales"]["tiny"]["indexes"]}
    base = idx["ZGQ-M16+zonescan-nopq"]["analytic_bytes"]["total_bytes"]
    n = summary["scales"]["tiny"]["n_vectors"]
    assert idx["ZGQ-M16+overlap2-nopq"]["analytic_bytes"]["total_bytes"] == base + 4 * n
    assert idx["HNSW-M16"]["analytic_bytes"]["bytes_per_vector"] > 0
    for pt in idx["ZGQ-M16+adaptive-nopq"]["points"]:
        assert 1 <= pt["mean_probes"] <= rb.IMPROVE_MAX_PROBE
    # replicas never reduce zone-scan recall
    r1 = [p["recall_pct"]["median"] for p in idx["ZGQ-M16+zonescan-nopq"]["points"]]
    r2 = [p["recall_pct"]["median"] for p in idx["ZGQ-M16+overlap2-nopq"]["points"]]
    assert all(b >= a - 1e-9 for a, b in zip(r1, r2))
