# Shared accurate point-Jacobian qualification

Date 2026-10-27, before corrected-image qualification execution. V3's
engineering magnification preflight found stable external derivatives at
steps 1e-4/5e-5 (relative changes <2e-10), but agreement with original A
amplitudes did not improve. Inspection of installed Lenstronomy 1.14.2 found
that its default SIE/NIE Hessian uses forward differences at step 1e-10.
Roundoff in that calculation is a plausible source of the 1e-5-level image
disagreements. This diagnosis is numerical development, not CNN evidence.

Apply a five-point fourth-order deflection stencil at step 1e-4 in BOTH new
branches. Each uses its own package's deflections and independently solved
positions. This adds a shared Jacobian recipe to the already shared quadrature,
PSF and detector; it does not make their mass/light evaluators identical.
Keep original simulation sources and reported results unchanged.

Reuse v2's A adaptive CLEAN arrays only: clean light does not depend on point
magnifications. Verify their immutable hashes, then regenerate expected scenes
and point amplitudes using the corrected A Jacobian in a new v3 release.
Rerun the selected 16/32/64 fine-reference comparisons with corrected point
amplitudes, and the complete adaptive B comparison with corrected B amplitudes.
V2 records remain preserved; no old expected-image array is relabeled as new.

All controls and tolerances from protocols v1/v2 remain: 315 renderable views,
the same selected fine-reference cases, 0.005 clean/expected accuracy, 1e-5
cross-engine image L1, 1e-5 arcsec root and 1e-3 amplitude differences. Retain
the v3 step-stability fixture, and record new A/B amplitude agreement. The
single-image proposal remains an explicit unconditional rejection only after
the independent classification. A failure blocks qualification.

The execution entry point is `scripts/qualify_adaptive_renderer_v3.py`; source
hashes bind the precise point renderer, inherited physical renderer, adaptive
integrator and this protocol. Final eligibility depends on the combined
`results/reports/renderer-qualification-v3/qualification.json`, not on the
engineering point preflight or a passing loader alone.
