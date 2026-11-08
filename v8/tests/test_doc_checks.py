import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


check_claims = _load("check_claims")
check_links = _load("check_links")


def test_claims_check_flags_retired_numbers(tmp_path):
    (tmp_path / "a.md").write_text("ZGQ reaches 94.8% recall in 11.4 ms with 43,794 QPS.\n")
    (tmp_path / "b.tex").write_text("a 28\\% increase and 13.3x speedup, wins 3 out of 4 metrics\n")
    names = {n for _, _, n in check_claims.find_violations(tmp_path)}
    assert {"94.8% recall", "11.4 ms latency", "43,794 QPS", "28% QPS",
            "13.3x vs IVF-PQ", "wins 3 of 4 metrics"} <= names
    assert check_claims.main(["--root", str(tmp_path)]) == 1


def test_claims_check_exempts_ledger_and_passes_clean(tmp_path):
    (tmp_path / "CLAIMS.md").write_text("removed: 94.8% recall\n")
    (tmp_path / "ok.md").write_text("ZGQ-M16 reaches 78.9% at ef=64.\n")
    assert check_claims.main(["--root", str(tmp_path)]) == 0


def test_repo_docs_have_no_retired_claims():
    assert check_claims.find_violations(ROOT.parent) == []


def test_links_check_detects_broken(tmp_path):
    (tmp_path / "x.md").write_text("[ok](y.md) [bad](nope.md) [web](https://example.com) [a](#anchor)\n")
    (tmp_path / "y.md").write_text("hi\n")
    (tmp_path / "p.tex").write_text("\\includegraphics[width=1cm]{fig/missing.pdf}\n")
    assert check_links.check_file(tmp_path / "x.md") == ["nope.md"]
    assert check_links.check_file(tmp_path / "p.tex") == ["fig/missing.pdf"]
    assert check_links.main(["--root", str(tmp_path)]) == 1


def test_repo_links_resolve():
    assert check_links.main(["--root", str(ROOT.parent)]) == 0
