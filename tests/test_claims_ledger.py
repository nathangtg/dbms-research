"""Check the 'verified' matched-comparison rows of docs/CLAIMS.md against the canonical summary."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "2026-11-08" / "summary.json"
CLAIMS = ROOT / "docs" / "CLAIMS.md"

ROW = re.compile(
    r"^\| (?P<scale>10K|100K|1M) ef=(?P<ef>\d+)[^:]*: "
    r"HNSW-M16 (?P<hr>[\d.]+)% / (?P<hl>[\d.]+) ms; ZGQ-M16 (?P<zr>[\d.]+)% / (?P<zl>[\d.]+) ms"
)


def _point(summary, scale, label, ef):
    idx = next(i for i in summary["scales"][scale.lower()]["indexes"] if i["label"] == label)
    return next(p for p in idx["points"] if p["search_param"]["ef"] == ef)


def test_matched_rows_match_summary():
    summary = json.loads(SUMMARY.read_text())
    rows = [m for m in map(ROW.match, CLAIMS.read_text().splitlines()) if m]
    assert len(rows) >= 6
    for m in rows:
        ef = int(m["ef"])
        for label, rec, lat in (("HNSW-M16", m["hr"], m["hl"]), ("ZGQ-M16", m["zr"], m["zl"])):
            p = _point(summary, m["scale"], label, ef)
            assert round(p["recall_pct"]["median"], 1) == float(rec), (m.group(0), label)
            assert round(p["latency_ms_per_query"]["median"], 3) == float(lat), (m.group(0), label)


def test_build_time_row_matches_summary():
    summary = json.loads(SUMMARY.read_text())
    text = CLAIMS.read_text()
    m = re.search(r"Build times HNSW-M16 / ZGQ-M16: ([\d.]+)/([\d.]+) s \(10K\), ([\d.]+)/([\d.]+) s \(100K\), "
                  r"([\d.]+)/([\d.]+) s \(1M\)", text)
    assert m
    vals = list(map(float, m.groups()))
    for i, scale in enumerate(("10k", "100k", "1m")):
        idx = {x["label"]: x for x in summary["scales"][scale]["indexes"]}
        assert round(idx["HNSW-M16"]["build"]["time_s"], 1) == vals[2 * i]
        assert round(idx["ZGQ-M16"]["build"]["time_s"], 1) == vals[2 * i + 1]
