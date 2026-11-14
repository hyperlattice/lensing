# Confirmatory analysis specification

Declared before new scientific fits, method freeze, power pilot or final
generation. This refines the proposed plan; it is not an executed freeze.

The primary question is whether fixed source-failure allocation R improves
over uniform allocation G under the same backbone, initialization, parent
schedule, source exposures, optimizer, objective and update count. G's
source-development-selected optimizer is used for both G and R. Independently
tuned R is a development diagnostic. The same sharing applies within each
G/R+MMD information-access pair. A common Huber objective makes these
point-regression method ports, not exact published MVE reproductions.

The nine conditions and renderer order are defined in
`src/processing/evaluation_conditions_v1.py`. The primary statistic averages
single-model absolute errors over the three B Moffat conditions, parents and
paired seeds. Never average predictions first. Preserve all conditions of a
parent and all methods during independent seed/parent resampling.

Allocate one-sided alpha .025 to the primary R-minus-G meaningful-superiority
test and .025 to a separate secondary family. This conservative allocation
replaces the proposed plan's .05 primary plus unspecified secondary family,
and allows reporting a secondary contribution even if the primary fails.
The primary passes only if its crossed-bootstrap upper bound is below
-0.005 arcsec AND the two canonical R-minus-G MAE bounds (A and B) are each
below +0.005 arcsec AND the B-Moffat tail bound is below +0.010 arcsec. These
are joint requirements; each guardrail uses alpha .025. The tail statistic
is the linear-interpolated 95th percentile over parent/condition errors
within each seed, averaged across seeds. Recompute it inside each crossed
bootstrap resample. These are engineering margins, not cosmological accuracy
requirements. Report them regardless of whether they pass.

The 5-mas margin is one tenth of the canonical 0.05-arcsec pixel and 0.25–1%
of the declared 0.5–2-arcsec radius range. It is a concrete benchmark target;
it does not imply recovery of subpixel image detail or a cosmological error
budget. The 10-mas P95 allowance is one fifth of a canonical pixel. These
interpretations motivate an explicit accuracy/utility tradeoff before outcomes.

Secondary superiority contrasts, all on the same B Moffat MAE endpoint:
R-minus-C, R-minus-N, R-minus-T; R+MMD-minus-G+MMD for Gaussian-only target
access; and R+MMD-representative-minus-G+MMD-representative. Use Bonferroni
alpha .025/5 for each upper bound and the same -0.005 arcsec margin. For each
secondary claim require its corresponding canonical and tail guardrails at
that contrast's adjusted alpha. Report all five, including failed tests.
No target-free versus target-aware comparison establishes an advantage in
information access without its stated error result. Comparisons across those
access levels, renderer interactions, radius bins, bias and relative errors
are descriptive, with no additional confirmatory superiority claims.

Use 10,000 crossed paired percentile-bootstrap draws, NumPy linear quantiles,
fixed seed 720901. One-sided bounds are not two-sided 95% intervals; label
them explicitly. Use a paired seed-level Student t sensitivity interval,
conditional on sampled parents, alongside the crossed bound. This sensitivity
cannot substitute for the crossed analysis or make an unsuccessful test pass.
Report per-seed effects and both raw and nonnegative planning variance
components. Few-seed uncertainty and finite-pilot variance remain limitations.

Freeze methods before a fresh 120-parent, five-pilot-seed evaluation. Pilot
scripts export centered paired-error variance summaries and power only; no
method ranking is used for tuning. Choose among S=10/15/20 and
N=600/1000/1500/2000 for 90% primary power at true delta=-.010 arcsec, tested
against -.005. Estimate power under a declared additive Gaussian crossed
random-effects planning model, using actual crossed-bootstrap rejection in
simulation. Record Monte Carlo uncertainty, raw/clipped components and a
variance-inflation sensitivity. A planning result is model-dependent, not
guaranteed power. If the capped design falls short, report the limitation;
do not relax the margin or change the endpoint. Select by estimated training
and rendering cost among designs meeting the target, with smaller S then N
as deterministic ties. Freeze actual checkpoints, N/S and seed lists before
final generation. No final outcome-driven sample enlargement.

Only independently agreed physical single-image systems may be rejected from
the declared two/four-image population. Numerical disagreements block the
release and require an auditable correction/new version, never silent sample
replacement. Preserve every proposal and failure. Check identities, seeds,
latent fingerprints and ancestry against all previous scientific roles.
Renderer agreement is a numerical control; substantive shift is the specified
PSF family. Shared quadrature, detector and PSF code limit independence.

Final prediction files contain no truth; verify axes and seal every method
before opening the isolated sidecar once. No silent inference or analysis
restart. Infrastructure recovery may reuse only hash-verified completed
outputs under the same identity; scientific code/data corrections require a
new documented release and invalidate any affected pilot/freeze. Algorithmic
fit failures remain outcomes and cannot be replaced with favorable seeds.
