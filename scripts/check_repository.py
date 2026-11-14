"""Check the maintained workflow; historical documents use their recorded checkout."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = re.compile(r"(?<![\w/])configs/[A-Za-z0-9_./-]+\.yaml")
LINK = re.compile(r"\[[^\]\n]*\]\(([^)\n]+)\)")
CURRENT_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "src/README.md",
    "scripts/README.md",
    "configs/README.md",
    "docs/README.md",
    "docs/study.md",
    "docs/reproduction.md",
    "docs/data_contract.md",
    "docs/rendering.md",
    "docs/migration.md",
    "docs/hst_study.md",
)


def check() -> int:
    """Validate active imports/configuration references without opening scientific data."""
    errors = []
    optional = []
    configurations = sorted((ROOT / "configs").rglob("*.yaml"))
    for path in configurations:
        if path.parent != ROOT / "configs":
            errors.append(f"Nested active configuration: {path.relative_to(ROOT)}")
        if not isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict):
            errors.append(f"Expected configuration mapping: {path.relative_to(ROOT)}")
    paths = [ROOT / name for name in CURRENT_DOCS]
    for directory in ("src", "scripts", "tests"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend(configurations)
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.module:
                    modules.append(node.module)
                elif isinstance(node, ast.Import):
                    modules.extend(alias.name for alias in node.names)
                for module in modules:
                    if not module.startswith(("src.", "scripts.")):
                        continue
                    target = ROOT.joinpath(*module.split("."))
                    if not target.with_suffix(".py").is_file() and not target.is_dir():
                        errors.append(f"{path.relative_to(ROOT)}: missing import {module}")
        for target in set(CONFIG_PATH.findall(text)):
            if not (ROOT / target).is_file():
                errors.append(f"{path.relative_to(ROOT)}: missing {target}")
        if path.suffix != ".md":
            continue
        for target in LINK.findall(text):
            target = unquote(target.split("#", 1)[0].split("?", 1)[0]).strip("<>")
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                message = f"{path.relative_to(ROOT)}: missing link {target}"
                if resolved.is_relative_to(ROOT / "data") or resolved.is_relative_to(
                    ROOT / "results"
                ):
                    optional.append(message)
                else:
                    errors.append(message)
    for message in sorted(set(errors)):
        print(f"FAIL: {message}")
    for message in sorted(set(optional)):
        print(f"OPTIONAL ARTIFACT: {message}")
    print(f"Checked {len(configurations)} configurations and {len(paths)} maintained files")
    print(
        f"Broken references: {len(set(errors))}; unavailable optional artifacts: {len(set(optional))}"
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(check())
