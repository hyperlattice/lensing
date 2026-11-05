# AI2ASE-27 submission source and export

Edit only `reviewer_bundle/`. Generate an Overleaf ZIP from repository root:

```powershell
python -m scripts.export_submission
```

The exporter validates the canonical bundle and verifies every exported byte.
An existing ZIP is protected; use `--output` with a fresh path for a new
export. ZIPs and expanded exports are ignored delivery products, not editable
second sources. Static validation does not replace compiling and inspecting
the manuscript PDF.

Only `reviewer_bundle/` is intended for workshop sharing. It contains the
anonymous paper source, required AAAI style files, local figures, and a
reviewer-safe reproducibility summary.

The private author-kit and working-build material is retained under
`../../archive/private/submission_aaai27/`. It is not part of the upload and is
ignored by Git. Do not zip or upload this parent directory wholesale.

For Overleaf, use `ai2ase_overleaf_project.zip`; it contains only the curated
reviewer bundle contents at the archive root.
