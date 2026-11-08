#!/usr/bin/env python3
"""Fail if a retired benchmark number reappears in the docs or LaTeX sources.

The retired numbers came from unreproducible runs and are listed, with their
status, in docs/CLAIMS.md (which is exempt from this check).

    python scripts/check_claims.py [--root PATH]
"""

import argparse
import re
import sys
from pathlib import Path

RETIRED = {
    "94.8% recall": r"94\.8\d?\s*\\?%",
    "11.4 ms latency": r"11\.4\d?\s*(\\,|~)?\s*ms",
    "43,794 QPS": r"43,?794",
    "13.6% throughput": r"13\.6\s*\\?%",
    "0.048 ms latency": r"0\.048\s*ms",
    "68.2% recall": r"68\.2\d?\s*\\?%",
    "580 MB at 1M": r"~?\s*580\s*MB",
    "13.3x vs IVF-PQ": r"13\.3\s*(x|\\times|×)",
    "3.0x faster than IVF-PQ": r"3\.0\s*(x|×)\s+faster",
    "22% latency": r"22\s*\\?%\s*(reduction|lower|latency|faster)",
    "28% QPS": r"28\s*\\?%\s*(increase|higher|throughput|QPS|more)",
    "wins 3 of 4 metrics": r"\b3\s+(out\s+)?of\s+(the\s+)?4\b",
    "93.2% vs 92.9%": r"\b93\.2\d?\s*\\?%|\b92\.9\d?\s*\\?%",
    "ZGQ uses less memory than HNSW": r"(19\s*\\?%\s*less memory|less memory than HNSW|lower memory than HNSW)",
    "Expected improvements": r"Expected improvements",
    "15-20% RPQ error": r"15\s*-\s*20\s*\\?%\s*(lower|less|reduc)",
    "0.071 ms latency": r"0\.071\s*(\\,|~)?\s*ms",
    "64.6% recall": r"64\.6\d?\s*\\?%",
    "35% faster queries": r"35\s*\\?%\s*faster\s+quer",
    "20% memory reduction": r"20\s*\\?%\s*(memory|reduction in memory)",
    "610 MB at 1M": r"~?\s*610\s*MB",
    "alpha 0.74": r"(alpha|\\alpha)[^.\n]{0,20}0\.74",
    "empirical alpha": r"empirical[^.\n]{0,20}\\alpha",
}
EXEMPT = {"CLAIMS.md"}
# The archived v1 preprint is kept verbatim; its retired numbers are logged in
# docs/CLAIMS.md. Only this exact path (relative to the scan root) is exempt.
EXEMPT_PATHS = {"docs/preprint/v1.md"}
SKIP_DIRS = {".git", ".agents", "results", "References", "node_modules", ".venv"}


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent


def scan_files(root: Path):
    for ext in ("*.md", "*.tex"):
        for p in sorted(root.rglob(ext)):
            rel = p.relative_to(root)
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
                continue
            if p.name in EXEMPT or rel.as_posix() in EXEMPT_PATHS:
                continue
            yield p


def find_violations(root: Path):
    out = []
    compiled = {k: re.compile(v, re.IGNORECASE) for k, v in RETIRED.items()}
    for p in scan_files(root):
        for n, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            for name, rx in compiled.items():
                if rx.search(line):
                    out.append((p.relative_to(root), n, name))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=default_root())
    args = ap.parse_args(argv)
    bad = find_violations(args.root)
    for path, n, name in bad:
        print(f"{path}:{n}: retired claim ({name})")
    if bad:
        print(f"claims check FAILED: {len(bad)} retired number(s) found")
        return 1
    print("claims check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
