# Simulation pilot report: sim-p100-cumulative-v1

- Samples: **100**
- Independent parents: **100**
- Artifact size at report time: **219.82 MiB**
- Rejected stage draws: **0**

## Validation

- Validator passed: **True**
- Validator errors: **0**
- Stage counts: `{'sim-p10-v1': 10, 'sim-p100-v1': 40, 'sim-p30-v1': 20, 'sim-p60-v1': 30}`
- Physical multiplicities: `{2.0: 54, 4.0: 26}`
- Detectable multiplicities: `{2.0: 54, 4.0: 26}`

## Numerical and noise checks

- Supersampling control: `{'status': 'measured_single_control', 'sample_id': 'sim-p30-v1-0001', 'factor_one': 1, 'factor_two': 2, 'relative_l2_change': 0.018191667273640633}`
- Noise moments: `{'sample_count': 70, 'status': 'measured_engineering_moment', 'mean_residual_mean': 1.0812200404762532e-05, 'mean_residual_std': 0.03079134849693549, 'mean_predicted_pixel_std': 0.03078456712620599, 'noise_policy': 'Poisson in electrons plus Gaussian read noise before background subtraction'}`

## Limitations

- The Gaussian PSF is an analytic control profile, not an empirical CASTLES/HST kernel.
- Noise uses a toy independent-pixel model with sampled exposure/background/read-noise parameters.
- The pilot is an angular engineering dataset; no calibrated physical mass likelihood is claimed.
- The cumulative release contains stage increments with different model complexity and should retain release_stage in evaluation.
