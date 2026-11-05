"""Guard the curated AI2ASE reviewer bundle from private project leakage."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "submission" / "aaai27" / "reviewer_bundle"

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


def main() -> int:
    if not BUNDLE.is_dir():
        fail(f"missing reviewer bundle: {BUNDLE}")

    actual = {
        path.relative_to(BUNDLE).as_posix()
        for path in BUNDLE.rglob("*")
        if path.is_file()
    }
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


if __name__ == "__main__":
    raise SystemExit(main())
