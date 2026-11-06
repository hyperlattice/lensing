# Data feasibility and options for a synthetic lens-generation benchmark

**Discussion recorded: 6 November 2026. Status: planning; no route selected and no new generation or training runs performed.**

This note records the dataset and generation choices raised after the [novelty review](review.md). It complements the [explainability proposal](../../explainability_synthetic_data_proposal.md), rather than replacing its intervention study. The regression target remains `theta_E_arcsec`.

## 1. What qualifies as a training sample?

For supervised radius regression, a sample needs a complete lens image and a valid Einstein-radius label for the configuration depicted. Other physical and observation metadata support controlled comparisons and failure analysis; matching column names alone does not establish compatibility.

A galaxy is a physical object; an image is a rendered or observed patch of sky. Our lens images combine a foreground lens, distorted background-source light, foreground light, and observation effects. Several bands or noisy views of one object do not represent independent physical systems. An ordinary galaxy cutout can supply a source component, but it is not already a labelled lens-regression example. Measured light ellipticity and position angle are also not automatically the foreground mass parameters `q` and `phi_rad`.

## 2. External dataset assessment

| Dataset | Available images and information | Direct expansion of our labelled lens data? |
|---|---|---|
| DeepSkies / Agarwal et al. | The paper reports 100,000 simulated lenses in each of low-noise and DES-like observation domains, with Einstein-radius regression labels. Images are three-band 40 x 40 arrays at 0.263 arcsec/pixel, with radius proposals of 1-3 arcsec. | **Potentially, after compatibility work.** Our configuration uses a single channel, 256 x 256 pixels, 0.05 arcsec/pixel, and 0.5-2 arcsec radius proposals. Label conventions, saved parameter coverage, and ancestry need inspection before integration. |
| GalSim COSMOS | Approximately 56,000 galaxies in a brighter selection or 87,000 in a deeper selection; galaxy cutouts with original PSF and noise information support realistic rendering. The selections overlap and must not be added as independent counts. | **No, as supplied.** It is a source-galaxy library; physical lensing must be applied to create images with simulator-defined radius labels. |
| GalaxiesML | 286,401 galaxies observed in five bands, with photometry, spectroscopic redshifts, and measured structural parameters including half-light radius, ellipticity, and Sérsic index. | **No, as supplied.** These are galaxy properties, not ground-truth lens configurations. Source preparation and physical rendering would be additional work. |

Sources: [DeepSkies paper, Section 3 and Table 1](https://ml4physicalsciences.github.io/2024/files/NeurIPS_ML4PS_2024_58.pdf), [DeepSkies release](https://zenodo.org/records/13647416), [GalSim COSMOS documentation](https://galsim-developers.github.io/GalSim/_build/html/real_gal.html), and [GalaxiesML release](https://zenodo.org/records/13878122). The [archived DeepSkies intake](../../../archive/docs/agarwal_same_holdout_intake_v1.md) documents earlier resource and code checks. This assessment does not establish that the external archives have passed local integration or label validation.

Resizing an external image does not recover angular detail absent from its original observation. Image units, passbands, PSF, noise, mass-model conventions, source populations, and selection need assessment alongside radius labels. No dataset in this shortlist is established as a drop-in expansion.

## 3. How dataset mismatch could create a shortcut

If our images have radii of 0.5-2 arcsec and recognizably different external images have radii of 1-3 arcsec, dataset appearance provides information about the target. The CNN could exploit that association without correctly extracting all the relevant lens geometry. This is a possible confound, not a measured finding.

A random split of pooled data can preserve the association. Matching minimum and maximum radius alone is insufficient: the full target distribution can differ within the common range. For a comparison intended to isolate generation method, match target distributions and declared observation conditions, use the same independent evaluation population, and retain dataset origin for stratified diagnostics. Restricting to overlapping support can help but changes the population being studied and leaves other differences unresolved. Converting units is appropriate; relabelling physically different radii to make their numerical ranges agree is not.

## 4. Sample size: limited data, not a demonstrated impossibility

The [engineering release record](../../../data/manifests/sim-eng1000-v1/engineering_release.json) records 1,000 parents, allocated as 700 training, 150 validation, and 150 historical test systems. Thus 700 systems would be available to a generator fitted on that training split. Validation and test systems must remain excluded from generator fitting.

This is a limited-data setting with risks of memorization, missing modes, and weak continuous conditioning. It does not establish that learned generation is impossible. Earlier conversational ranges such as 10,000-100,000 images were planning estimates, not validated minimums. [Karras et al. (2020), abstract](https://arxiv.org/abs/2006.06676) report successful data-efficient GAN training with a few thousand images in other domains; that does not validate labelled lens generation at our sample size.

A small-data study would ask which implementations work under a declared data constraint. Learning curves over independent training-system counts could help distinguish data limitation from method limitation. Equal generated-image counts do not establish equal effective diversity, and a large output bank does not replace independent evaluation systems.

## 5. Four routes and their claim boundaries

| Route | Benefit | Limitation and supported scope |
|---|---|---|
| Fit generators on the current training set | Consistent labels and image conventions; bounded pilot. | Limited coverage; conclusions concern generation under the current data constraint. |
| Render more independent parents with our simulator | Matching parameter conventions and controllable target distributions. | Explores the simulator's assumed population; additional samples do not establish observational realism. |
| Incorporate external labelled lenses | More samples and potentially broader conditions. | Requires compatibility work; use a separate robustness domain if pooling would obscure interpretation. |
| Generate background galaxies, then apply a common lens and instrument renderer | Empirical morphology can inform source generation while the renderer supplies lens labels. | Requires source/PSF/noise preparation and changes the question to source-generation effects on inference. |

Expanding our simulator was recommended in the discussion as the most direct continuation of complete-image generation, conditional on solving output-label validation. This is an advisory recommendation, not an adopted experiment. The source-generation route remains an alternative when empirical morphology is the priority.

## 6. What the outputs would mean

Both our simulator and a learned whole-image generator can output the same pixel-array format. Their evidence differs. The simulator uses the chosen radius in an explicit lens calculation and retains physical and observation parameters. A conditional learned generator receives a requested radius; that request alone does not establish that its output depicts the corresponding configuration. Whole-image generation therefore needs independent label-fidelity checks, rather than acceptance by the same CNN whose training utility is being tested.

A generator trained exclusively on our simulator primarily approximates that simulator's distribution. It has no empirical basis for recovering real structures omitted from those training images; added texture is not evidence of greater realism. Such a benchmark can assess preservation of simulated structure and labels, downstream utility, and cost. Claims about observational realism need separate evidence. Generating source components before shared physical rendering preserves simulator-defined lens parameters, while source realism remains a distinct validation question.

The detailed simulator mechanics are already covered in [simulation_pipeline.md](../../simulation_pipeline.md). No simulator changes are proposed here.

## 7. Decisions still open

- Choose complete lens-image generation or source generation as the comparison's main scope.
- Choose the data route, eligible generator-training systems, target distribution, and training-size conditions.
- Define independent whole-image label checks if that route is retained.
- Freeze a common evaluation population and any external-data compatibility criteria before comparing methods.

These are design choices. The session did not select generator implementations, approve a final run matrix, or demonstrate a generator benefit.
