# Week 10 next — CASTLES domain-gap analysis

Week 9 found no consistent evaluation-count or runtime advantage from the ML initializer for the observation-only point-image fit. The final Week 8 randomized CNN remains the frozen model choice because it passed the preregistered synthetic mitigation gate; Week 9 does not justify changing its architecture or training.

Protocol v1.10 now freezes a descriptive comparison between the 1,400-view Week 8 training distribution and the 137 available processed CASTLES observations from 69 lens systems. The analysis uses pixel scale, field of view, shape/cropping, metadata completeness, robust within-image intensity measures, fixed-cutout radial profiles, concentration, and an uncalibrated outer-region brightness proxy. It records PSF, mask, and multiplicity as unavailable where products/labels are absent. Einstein-radius/redshift fields and all real-image model accuracy scoring are excluded.

After documenting what the simulator covers and leaves unsupported, Week 11 freezes the final synthetic evaluation method and generates a new independent blind holdout. The historical synthetic test remains quarantined.
