#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote

MD_LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|!\[[^\]]*\]\(([^)\s]+)\)")
TEX_GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
SKIP_DIRS = {".git", ".agents", "results", "References", ".venv", "zgq.egg-info"}


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent


def iter_files(root: Path):
    for ext in ("*.md", "*.tex"):
        for p in sorted(root.rglob(ext)):
            rel = p.relative_to(root)
            if any(part in SKIP_DIRS or part.startswith(".") for part in rel.parts):
                continue
            yield p


def _strip_fences(text: str) -> str:
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def check_file(path: Path):
    text = path.read_text(errors="replace")
    missing = []
    if path.suffix == ".md":
        for m in MD_LINK.finditer(_strip_fences(text)):
            target = m.group(1) or m.group(2)
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            target = unquote(target.split("#", 1)[0])
            if target and not (path.parent / target).exists():
                missing.append(target)
    else:
        for line in text.splitlines():
            if line.lstrip().startswith("%"):
                continue
            for m in TEX_GRAPHIC.finditer(line):
                if not (path.parent / m.group(1)).exists():
                    missing.append(m.group(1))
    return missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=default_root())
    args = ap.parse_args(argv)
    failures = 0
    for p in iter_files(args.root):
        for target in check_file(p):
            print(f"{p.relative_to(args.root)}: broken link or path: {target}")
            failures += 1
    if failures:
        print(f"link check FAILED: {failures} broken")
        return 1
    print("link check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
