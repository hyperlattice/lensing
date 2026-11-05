"""Build a deterministic Overleaf ZIP from the canonical reviewer bundle."""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

from scripts.check_reviewer_boundary_v1 import ALLOWED, BUNDLE, ROOT, main as check_boundary
from scripts.validate_ai2ase_submission_v1 import main as validate_submission


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
    parser.add_argument("--output", type=Path,
                        default=ROOT / "submission/aaai27/ai2ase_overleaf_project.zip")
    export(parser.parse_args().output)


if __name__ == "__main__":
    main()
