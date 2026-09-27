# Week 9 next — hybrid classical/ML initialization

Week 8 closed with a pass for its frozen targeted-randomization rule: the equal-weight four-condition stress MAE change was -0.0290 arcsec (95% interval [-0.0351, -0.0231]), and canonical MAE improved by -0.0109 arcsec (upper interval bound -0.0031). The Week 8 randomized checkpoints are therefore the frozen ML starts for Week 9.

Protocol v1.9 compares training-mean, image-radius heuristic, and CNN-predicted `theta_E` starts across the same 150 validation parents in canonical, 0.04/0.06 pixel-scale, half-exposure, and double-read-noise conditions. It uses the established observation-only peak detector and bounded SIS/SIE point-image geometry estimator. Truth-sidecar-initialized full pixel fitting is excluded. The run records convergence, final residual/error, evaluations, runtime, boundaries, and local Jacobian conditioning when available. Predictions and fit records must be saved before target joins.

After the Week 9 result, close its status and registry entries, then freeze the Week 10 CASTLES/simulation coverage analysis. Real observations may be compared only on measurable metadata and image statistics; no real-data parameter-accuracy claim is available without a blinded truth benchmark.
