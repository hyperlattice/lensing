# Rendering

There is one `SceneEngine` in `src/simulation/render.py`, not a chain of benchmark,
precise and low-memory version classes. It incorporates the qualified sparse
Lenstronomy root search, fourth-order deflection Jacobians and bounded-memory
midpoint integration. Adaptive integration retains its original brightness-based
refinement rule. Root tolerances, deduplication, numerical differences and PSF
sampling are part of the scientific contract, not style choices.

`backends.py` contains lazy AutoLens physical profiles and coordinate/light
conversions. Lenstronomy profiles are constructed by the scene engine. Each
backend evaluates its own deflections, light and roots. Shared rasterization,
PSF convolution and detector sampling remain explicit limitations of independence.
`observation.py` is the sole definition of conditions, PSFs and detector draws.
`independent.py` retains the analytic SIE check used to classify root failures.

Old qualification reports bind old source hashes. They must not be rewritten to
approve this checkout. `study qualify` explicitly compares all 315 saved numerical
controls under `data/simulations/renderer-qualification-v3/` and writes to the
existing `results/reports/renderer-qualification-v5/` destination only if absent.
It rerenders adaptive clean light and compares root positions, magnifications and
expected images. Both physical backends must qualify before new paired generation.
Reference metadata/products and current source hashes are recorded. Existing
qualification outputs are protected, including failed reports.

The refactor's small reference tests are regression checks, not a claim that this
complete scientific qualification or AutoLens validation has been rerun.
