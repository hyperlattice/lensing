# Sparse exact-profile root recovery

Declared before the next root qualification or bank retry. The analytical-only
low-memory prototype passed 312/315 view comparisons but missed one faint
inner image in parent 005. The v4 qualification failed and is preserved; no
bank generation is allowed under it.

The new low-memory v2 A engine supplements analytical candidates with 48
angular starts at nine radii .005/.02/.1/.35/.65/.9/1.1/1.4/1.8 times theta_E.
All starts are independently solved against A's actual SIE deflections. Require
residual <=1e-8, merge roots only within 1e-6 arcsec, and retain roots within
four model radii. Exclude radii <1e-7 arcsec around the singular center to avoid
the numerical NIE-core artifact of the intended singular SIE; this is far
below pixel sampling and does not alter resolved image positions. This
finite-start method is not a proof of completeness over every possible lens.
B's solver, the physical profiles, quadrature, Jacobians, PSFs and detector
are unchanged.

Repeat the SAME 105-parent/315-view comparison against immutable v3 references
with unchanged root/count/amplitude/image tolerances. Preserve all failures.
Combine a passing sparse-root component with the passing parent v3 numerical
qualification in `results/reports/renderer-qualification-v5/qualification.json`.
Do not revise v3 or the failed v4 reports. Use v2 bank output directories and
the original fixed parent/seed cohorts only after this new gate passes.

The data build now runs roles sequentially with at most two rendering workers
globally. Read-only memory-mapped bank arrays bound training private memory.
These resource corrections do not authorize dropping numerical failures or
selecting favorable physical parents; only the already declared physical
single-image adaptation rejection is permitted.
