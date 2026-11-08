from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from visualization import generate_ieee_figures as figs  # noqa: E402
from visualization import make_tables  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "mini_summary.json"


def test_all_figures_generated_from_json_only(tmp_path):
    paths = figs.generate_all_figures(FIXTURE, tmp_path, "png")
    assert len(paths) == len(figs.FIGURES)
    for p in paths:
        assert p.is_file() and p.stat().st_size > 0
    assert (tmp_path / "fig_ivf_pq_latency.png").exists()
    assert (tmp_path / "fig_ivf_pq_tradeoff.png").exists()


def test_figure_script_has_no_benchmark_code():
    src = Path(figs.__file__).read_text()
    assert (
        "import hnswlib" not in src
        and "skip-benchmark" not in src
        and "import time" not in src
    )


def test_rejects_wrong_schema(tmp_path):
    bad = tmp_path / "s.json"
    bad.write_text('{"schema_version": "other"}')
    with pytest.raises(ValueError):
        figs.load_summary(bad)


def test_tables_markdown_and_latex(tmp_path):
    md, tex = make_tables.write_tables(FIXTURE, tmp_path)
    text = md.read_text()
    assert "| 10K | HNSW-M16 | baseline | 1.0 | 90.0 / 0.050 | 98.0 / 0.100 |" in text
    assert "60.0 / 0.050 (nprobe=16)" in text
    assert r"\begin{tabular}" in tex.read_text() and r"\%" in tex.read_text()
