import json

from benchmarks import run_compression as rc


def test_tiny_run_tables_and_adc(tmp_path):
    argv = ["--scale", "tiny", "--out-dir", str(tmp_path), "--n-queries", "40", "--repeats", "2",
            "--build-threads", "2", "--only", "HNSW-M16", "ZGQ-C-RPQ-M8-pq16", "HNSWPQ-M8-pq16",
            "IVFPQ-pq16", "IVFPQR-pq16-r64"]
    assert rc.main(argv) == 0
    s = json.loads((tmp_path / "compression_summary.json").read_text())
    assert s["schema_version"] == rc.SCHEMA
    idx = s["scales"]["tiny"]["indexes"]
    labels = [i["label"] for i in idx]
    assert "ZGQ-C-RPQ-M8-pq16" in labels and "ZGQ-C-RPQ-M8-pq16+rerank50" in labels
    for i in idx:
        assert i["memory"]["bytes_per_vector"] > 0
        assert i["points"] and (tmp_path / i["raw_file"]).is_file()
        # analytic model agrees with the measured/serialized size
        assert 0.9 < i["memory"]["crosscheck_ratio"] < 1.1, (i["label"], i["memory"])
    assert rc.main(["--out-dir", str(tmp_path), "--tables-only"]) == 0
    assert "Model cross-check" in (tmp_path / "equal_memory_tiny.md").read_text()
    assert rc.main(["--scale", "tiny", "--out-dir", str(tmp_path), "--n-queries", "40",
                    "--adc-check", "--adc-queries", "10", "--build-threads", "2"]) == 0
    adc = json.loads((tmp_path / "adc_check_tiny.json").read_text())
    assert len(adc["rows"]) == 4
