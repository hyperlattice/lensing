"""Static validation for the anonymous AAAI-27 AI2ASE source package.

This deliberately does not replace compiling with the official TeX toolchain.
It catches missing local assets, template-contract mistakes, forbidden packages,
and citation-key mismatches before a human performs the final PDF check.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission" / "aaai27" / "reviewer_bundle"
PAPER = PACKAGE / "paper.tex"
BIB = PACKAGE / "paper.bib"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> int:
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
        fail("missing required package files: " + ", ".join(missing))
    if (PACKAGE / "submission_manifest.md").exists():
        fail("internal submission manifest is present in reviewer bundle")

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
            fail(f"missing {name}")

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
        fail("forbidden template command/package(s): " + ", ".join(found_forbidden))

    figure_paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", paper)
    for figure in figure_paths:
        figure_path = PACKAGE / figure
        if not figure_path.exists():
            fail(f"missing included figure: {figure}")

    citation_keys = set()
    for match in re.finditer(r"\\cite[pt]?\{([^}]+)\}", paper):
        citation_keys.update(key.strip() for key in match.group(1).split(","))
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_keys = sorted(citation_keys - bib_keys)
    if missing_keys:
        fail("citation keys missing from paper.bib: " + ", ".join(missing_keys))

    print("PASS: static AI2ASE package validation")
    print(f"package={PACKAGE.relative_to(ROOT)}")
    print(f"figures={len(figure_paths)}")
    print(f"citations={len(citation_keys)}")
    print("NOTE: compile with the official AAAI-27 TeX toolchain for page count and visual checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
