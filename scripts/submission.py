"""Validate or byte-verify an export of the canonical reviewer bundle."""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission" / "aaai27" / "reviewer_bundle"
PACKAGE = BUNDLE
PAPER = PACKAGE / "paper.tex"
BIB = PACKAGE / "paper.bib"
ALLOWED = {
    "paper.tex",
    "paper.bib",
    "aaai2027.sty",
    "aaai2027.bst",
    "README.md",
    "REVIEWER_CONTENTS.md",
    "reproducibility_checklist.md",
    "figures/equal_budget_mae_by_condition.png",
    "figures/equal_budget_paired_contrasts.png",
    "figures/simulated_lens_examples_and_results.pdf",
}


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def check_boundary() -> int:
    if not BUNDLE.is_dir():
        fail(f"missing reviewer bundle: {BUNDLE}")

    actual = {path.relative_to(BUNDLE).as_posix() for path in BUNDLE.rglob("*") if path.is_file()}
    unexpected = sorted(actual - ALLOWED)
    missing = sorted(ALLOWED - actual)
    if unexpected:
        fail("unexpected reviewer-bundle files: " + ", ".join(unexpected))
    if missing:
        fail("missing reviewer-bundle files: " + ", ".join(missing))

    forbidden_text = re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|results/|data/|docs/private|working_build|AuthorKit27)"
    )
    for relative in sorted(actual):
        path = BUNDLE / Path(relative)
        if path.suffix.lower() in {".tex", ".bib", ".md"}:
            text = path.read_text(encoding="utf-8")
            if forbidden_text.search(text) and path.name not in {
                "README.md",
                "REVIEWER_CONTENTS.md",
            }:
                fail(f"private-path marker found in reviewer file: {relative}")

    print("PASS: reviewer boundary is clean")
    print(f"bundle={BUNDLE.relative_to(ROOT)}")
    print(f"files={len(actual)}")
    return 0


def submission_fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def validate_submission() -> int:
    required = [
        PAPER,
        BIB,
        PACKAGE / "aaai2027.sty",
        PACKAGE / "aaai2027.bst",
        PACKAGE / "README.md",
        PACKAGE / "REVIEWER_CONTENTS.md",
        PACKAGE / "reproducibility_checklist.md",
        PACKAGE / "figures" / "equal_budget_mae_by_condition.png",
        PACKAGE / "figures" / "equal_budget_paired_contrasts.png",
        PACKAGE / "figures" / "simulated_lens_examples_and_results.pdf",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        submission_fail("missing required package files: " + ", ".join(missing))
    if (PACKAGE / "submission_manifest.md").exists():
        submission_fail("internal submission manifest is present in reviewer bundle")

    paper = PAPER.read_text(encoding="utf-8")
    bib = BIB.read_text(encoding="utf-8")

    required_patterns = {
        "letterpaper class": r"\\documentclass\[letterpaper\]\{article\}",
        "AAAI submission style": r"\\usepackage\[submission\]\{aaai2027\}",
        "anonymous author": r"\\author\{Anonymous Submission\}",
        "bibliography": r"\\bibliography\{paper\}",
    }
    for name, pattern in required_patterns.items():
        if not re.search(pattern, paper):
            submission_fail(f"missing {name}")

    forbidden = (
        r"\\usepackage\{hyperref\}",
        r"\\usepackage\{geometry\}",
        r"\\usepackage\{balance\}",
        r"\\clearpage",
        r"\\newpage",
        r"\\pagebreak",
        r"\\pagestyle",
        r"\\tiny",
    )
    found_forbidden = [pattern for pattern in forbidden if re.search(pattern, paper)]
    if found_forbidden:
        submission_fail("forbidden template command/package(s): " + ", ".join(found_forbidden))

    figure_paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", paper)
    for figure in figure_paths:
        figure_path = PACKAGE / figure
        if not figure_path.exists():
            submission_fail(f"missing included figure: {figure}")

    citation_keys = set()
    for match in re.finditer(r"\\cite[pt]?\{([^}]+)\}", paper):
        citation_keys.update(key.strip() for key in match.group(1).split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_keys = sorted(citation_keys - bib_keys)
    if missing_keys:
        submission_fail("citation keys missing from paper.bib: " + ", ".join(missing_keys))

    print("PASS: static AI2ASE package validation")
    print(f"package={PACKAGE.relative_to(ROOT)}")
    print(f"figures={len(figure_paths)}")
    print(f"citations={len(citation_keys)}")
    print("NOTE: compile with the official AAAI-27 TeX toolchain for page count and visual checks")
    return 0


def export(output):
    check_boundary()
    validate_submission()
    output = Path(output).resolve()
    if output == BUNDLE.resolve() or output.is_relative_to(BUNDLE.resolve()):
        raise ValueError("Exports must be outside the canonical reviewer bundle")
    if output.suffix.lower() != ".zip":
        raise ValueError("Expected a .zip output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation avoids accidentally replacing a hand-maintained file.
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(ALLOWED):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, (BUNDLE / relative).read_bytes())
    with zipfile.ZipFile(output) as archive:
        if set(archive.namelist()) != ALLOWED or archive.testzip() is not None:
            raise RuntimeError("Export verification failed")
        for relative in ALLOWED:
            if archive.read(relative) != (BUNDLE / relative).read_bytes():
                raise RuntimeError(f"Export differs from canonical source: {relative}")
    print(f"PASS: exported and verified {len(ALLOWED)} files to {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "export"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "check":
        check_boundary()
        validate_submission()
    else:
        if args.output is None:
            parser.error("export requires --output")
        export(args.output)


if __name__ == "__main__":
    main()
