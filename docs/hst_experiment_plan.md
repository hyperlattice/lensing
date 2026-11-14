# C/G/R synthetic-to-HST experiment plan

Prepared 12 November 2026. This document specifies proposed follow-up experiments;
it does not report a real-image accuracy result or authorize changes to the
original training design. No new experiments were launched while writing it.

## Framing and research questions

The central question is whether changing the synthetic training distribution
improves the same Einstein-radius CNN, and whether those benefits persist when
the model is evaluated on actual telescope observations.

- **RQ1, synthetic robustness:** Does general observing-condition variation (G)
  reduce error compared with canonical training (C)? The current manuscript has
  already tested this on its declared synthetic holdout.
- **RQ2, targeted allocation:** Does development-error-weighted variation (R)
  improve upon G? Existing evidence is inconclusive. Preserve that result.
- **RQ3, observational transfer:** Do G and R agree more closely than C with
  independent classical radius estimates for selected HST lenses?
- **RQ4, mechanisms:** Which controlled physical or observational changes alter
  predictions, and do those sensitivities explain failures? This is an optional,
  exploratory extension, not an established contribution of the current pilot.

The five-system compatibility pilot addresses a prerequisite: whether differences
in geometry, units, centering, and reference definitions can be controlled well
enough to interpret an observational comparison.

## Existing evidence and constraints

The current manuscript uses 697 qualified training parents, 150 synthetic holdout
parents, ten paired seeds, the same CNN, and 880 optimizer updates per run. Its
four-stress MAEs are C 0.141251, G 0.111507, and R 0.107385 arcsec. The G-minus-C
interval supports improvement under the tested conditions; the R-minus-G interval
does not establish improvement. These results do not establish real-image utility.

The HST pilot has audited five fixed systems and obtained a wider archive cutout
for one. Remaining issues are in the [pilot protocol](hst_compatibility_pilot.md).
The configured `results/ml/equal-budget-final-v1` output contains no `.pt` files
in the current checkout. Checkpoint availability is a practical dependency.

The screened Dinos-I table contains 76 systems. Removing all five development IDs
leaves at most **71 metadata candidates**: 31 SLACS, 22 SL2S, and 18 BELLS. This is
not the final eligible sample and does not establish statistical power.

## Stage 0: recover and verify the experimental assets

**Action:** locate the original 30 checkpoints (three arms times ten seeds), their
stored training-only normalization, model settings, and training metadata. Verify
the seed/arm identities and saved configuration against `configs/study.yaml` and
the existing results. Check for missing runs or mismatched preprocessing.

**Control:** do not choose the best seed, substitute a new architecture, select a
checkpoint using HST errors, or alter R's allocation policy. If originals cannot
be recovered, plan an explicitly identified reproduction of the same training
design after checking compute and source-data availability. A reproduced run is
not silently presented as an original checkpoint.

**Output:** an asset manifest recording paths, hashes, settings, and availability.
Do not start real-image model comparisons until this dependency is resolved.

## Experiment 1: qualify one observational input, then the five-system pilot

**Question:** can real images enter the current model without untracked changes
in physical scale, intensity units, or centering?

Start with SDSSJ0037-0942. Retain the original HDF5 audit and separately qualify
the wider MAST product; different processing versions are not interchangeable.

1. Determine the foreground-galaxy center using an explicit light-profile method,
   with visual review and synthetic recovery checks. Investigate the approximately
   0.74 arcsec catalogue-to-brightest-pixel difference. Brightest-pixel centering
   alone is insufficient.
2. Verify the archive's DN/s interpretation, gain conversion, and background
   processing. Preserve original arrays and record each numerical transformation.
   Applying training-only normalization does not require estimating statistics
   on the HST sample. Do not introduce per-image normalization or arbitrary
   histogram matching to make predictions look plausible.
3. Obtain or construct a PSF appropriate to the selected archive processing and
   native pixel grid. Validate any PSF resampling with flux/centering checks and
   point-source injection tests. Do not transfer the old Dinos kernel unchanged.
4. Validate WCS, support, masks, contaminants, and field size. Retain native pixels
   where possible. Positive weights are coverage evidence, not calibrated inverse
   variance or absence of artifacts.
5. Audit the reference convention. The SIE/EPL slope-two control checks a limiting
   convention; it does not establish equivalence of the observed mass models.
   Record the mass family, normalization, shear, median, and interval semantics.

Use the PSF/noise description to assess compatibility and classical fitting; these
are not extra CNN predictor inputs. Tolerances for numerical conversion and center
recovery should be established on synthetic controls before reviewing CNN results.
Do not require a perfect detector-noise model to inspect predictions, but do not
claim calibrated observational likelihoods without an adequate one.

**Output:** versioned native image products, transformations, reference metadata,
reviewed centers/masks, and an explicit readiness decision. Extend the same rules
to the remaining four development systems. They remain excluded from confirmation
in every band and in overlapping releases.

## Experiment 2: test the preprocessing bridge on synthetic development scenes

**Question:** can the proposed real-data preparation itself create radius errors?

Use independent source-development parents, never the synthetic final holdout.
Begin with a bounded smoke subset to validate execution, then use the available
development population for descriptive checks. Match parent IDs across all
conditions and retain true simulator radii.

Compare a canonical reference with the actual proposed HST geometry and
processing. Include matched controls for field size, known coordinate offsets,
and validated count-unit conversion. Implement transformations through the
existing renderer where appropriate; a new noise realization or an interpolated
image must not be represented as an independent lens.

For geometry changes, distinguish changes in pixel sampling from lost sky
coverage. For calibration, the correctly converted representation must recover
the reference units; do not demand invariance to a genuinely brighter physical
source. Assess combinations only after individual transformations are understood.

**Measurements:** flux and WCS consistency, recovered centers, prediction changes,
and paired changes in absolute radius error for each model. Report whether an
effect is attributable to the preparation pipeline or to information actually
removed by cropping/noise. Use existing image-only forwards and checkpoint
normalization; no training or real-data parameter fitting occurs here.

**Gate:** repair identified preparation bugs before interpreting real predictions.
Model sensitivity to valid observational changes is a result, not a reason to
modify preprocessing until one arm wins. Crop-induced information loss is not
automatically a learned shortcut.

## Experiment 3: exploratory C/G/R predictions on the five HST development lenses

**Question:** after qualification, are outputs usable, and which remaining failure
mechanisms deserve controlled investigation?

Evaluate the same selected observation of each lens with all available verified
arm/seed pairs. Use deterministic evaluation mode and identical prepared images.
Keep reference metadata outside model inputs. Store predictions before joining
reference values to compute disagreements.

Select one primary band per system using a rule based on availability and
image/calibration quality, fixed before CNN outcomes. Additional bands are
within-system sensitivity checks. Flag finite but implausible outputs without
clipping them to the training range or dropping their systems after scoring.

**Output:** per-lens, per-arm, per-seed predictions, signed and absolute reference
disagreements, and clearly labelled diagnostic plots. Five lenses support case
studies and debugging, not a reliable population-level ranking. No significance
or generalization claim will be based on this stage.

Reference strategy: use published EPL estimates only as explicitly model-dependent
reference agreement after checking the parameter convention. For the pilot, test
sensitivity to compatible classical SIE fits where available, or a separately
validated SIE refit with the selected image/PSF/noise model. Do not relabel EPL
disagreement as error against exact SIE truth. If a defensible primary reference
cannot be established, keep observational results descriptive.

## Experiment 4: freeze and test transfer on untouched eligible HST lenses

**Question:** does the C/G/R ranking persist in an observational cohort whose
selection and preparation did not depend on their predictions?

Before running models on the remaining candidates, freeze:

- Checkpoint hashes and the complete set of paired seeds.
- Image preparation, centering, PSF handling, primary-band selection, and reference
  convention. Record every choice learned during the five-system pilot.
- Eligibility rules for field coverage, calibration, single-deflector suitability,
  usable reference estimates, target-range support, and image quality.
- The cohort ID list and exclusions, primary contrast, metrics, and resampling.

Deduplicate by physical lens, including alternate names and overlapping releases.
Include all systems passing the rules; do not choose a convenient final count or
exclude lenses because a model has high error. Report numbers excluded for each
reason. The final sample may be much smaller than 71.

**Primary contrast:** G minus C in mean absolute disagreement with the declared
classical radius reference. **Secondary contrast:** R minus G, reported with
uncertainty as a secondary comparison. R minus C is supplementary. If the goal is
formal claims about multiple contrasts, specify multiplicity handling before
execution; otherwise keep secondary comparisons descriptive.

For lens i, arm a, and paired training seed s, define

`L[a,i] = mean_s(abs(prediction[a,i,s] - reference[i]))`.

Then average `L[G,i] - L[C,i]` equally over eligible systems. This assesses typical
single-trained-model loss across the declared seeds; it is not the loss of an
ensemble formed by averaging predictions first.

Use 10,000 paired bootstrap resamples of lens systems, keeping all arms and seeds
of a selected lens together. This interval is conditional on the evaluated seed
set. Report paired training-seed variation separately; 10 seeds times N lenses
does not create 10N independent lenses. Do not pool HST and Euclid or combine
multiple bands as independent rows. Record the bootstrap seed in the frozen plan.

Also report signed bias, median absolute disagreement, individual-lens changes,
and descriptive survey breakdowns with sample counts. Show published reference
intervals and model-family sensitivity separately; formal posterior errors do
not cover all reference systematics. Do not use inverse-error weighting to make
precisely fitted lenses dominate the primary cohort result.

This estimates observational agreement for the eligible, selected cohort. It
does not prove exact parameter accuracy or universal telescope performance. A
negative or uncertain transfer result is scientifically meaningful. Statistical
power cannot be claimed from the metadata count; precision assessment follows
actual eligibility and development evidence, before final outcomes are examined.

## Optional Experiment 5: physically controlled feature/failure analysis

Run this only after the input and asset checks, and keep it distinct from the
primary observational comparison.

Use matched synthetic development scenes with known radius. Start with a small
set of interpretable interventions: foreground light, PSF width, and detector
noise. Change one declared family at a time, keep the true mass parameter and
parent ancestry fixed, and use matched nuisance/noise draws where appropriate.
Also include a physical positive control that changes the true Einstein radius
and rerenders the image; its prediction should respond to changed lens geometry.

Measure paired prediction changes and paired error changes by arm/seed/parent.
Choose intervention strengths on development support and report absolute
sensitivities; different perturbation sizes do not yield comparable feature
shares automatically. Interactions and changed information content must be
considered before interpreting a dependence as a shortcut.

Use Integrated Gradients or other visual explanations only as supporting evidence
after sanity checks. Do not infer causal feature reliance from heatmaps or from
correlations between physical parameters and error alone. With this sample size,
avoid a high-dimensional learned failure predictor or a new synthetic-data-size
sweep unless the simpler controlled study produces a clear, reproducible effect.

## Execution order and claim boundaries

1. Recover verified model assets and qualify the one wider HST input.
2. Validate the preprocessing bridge on synthetic development data.
3. Finish the five-system diagnostic pilot and resolve reference interpretation.
4. Freeze the observational cohort and analysis before untouched predictions.
5. Run the final paired observational comparison and report uncertainty.
6. Add controlled feature analysis if it contributes an identifiable mechanism.

No new CNN architecture, HST fine-tuning, modified R policy, sample-size sweep,
or Euclid extension is needed for the core HST transfer question. Any of those
would constitute a separately specified experiment. The existing manuscript and
its reported synthetic findings remain unchanged while this follow-up is developed.
