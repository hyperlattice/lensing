"""Read-only checks of maintained configuration paths and documentation links.

Historical archive/output records are not rewritten. Missing links to optional
private data/results are reported separately from broken source references.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = re.compile(r"(?<![\w/])configs/[A-Za-z0-9_./-]+\.yaml")
PROTOCOL_PATH = re.compile(r"(?<![\w/])(?:archive/)?docs/reliability_study/protocol_[\w.]+\.md")
LINK = re.compile(r"\[[^\]\n]*\]\(([^)\n]+)\)")


def check():
    errors, optional = [], []
    config_count = 0
    import yaml

    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            errors.append(f"{path.relative_to(ROOT)}: expected a configuration mapping")
        config_count += 1
    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    for directory in ("src", "scripts", "tests", "configs", "docs"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.suffix in {".py", ".yaml", ".md"} and p.is_file()
                     and "__pycache__" not in p.parts
                     and not p.is_relative_to(ROOT / "docs/archive"))
    for path in sorted(set(paths)):
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            ast.parse(text, filename=str(path))
        for pattern in (CONFIG_PATH, PROTOCOL_PATH):
            for target in set(pattern.findall(text)):
                if not (ROOT / target).is_file():
                    errors.append(f"{path.relative_to(ROOT)}: missing {target}")
        if path.suffix != ".md":
            continue
        for target in LINK.findall(text):
            target = unquote(target.split("#", 1)[0].split("?", 1)[0]).strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if resolved.exists():
                continue
            message = f"{path.relative_to(ROOT)}: missing link {target}"
            if resolved.is_relative_to(ROOT / "data") or resolved.is_relative_to(ROOT / "results"):
                optional.append(message)
            else:
                errors.append(message)
    for message in sorted(set(errors)):
        print(f"FAIL: {message}")
    for message in sorted(set(optional)):
        print(f"OPTIONAL ARTIFACT: {message}")
    print(f"Checked {config_count} YAML configurations and {len(set(paths))} maintained text files")
    print(f"Broken references: {len(set(errors))}; unavailable optional artifacts: {len(set(optional))}")
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
