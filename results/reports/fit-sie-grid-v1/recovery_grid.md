# SIE recovery grid

- Scenarios: **6**
- Fits: **13**
- Optimizer successes: **13**
- Locally identifiable: **13**
- Baseline maximum theta_E relative error: **0.1171**
- Scientific gate passed: **True**

## Conditions

- `p30_noiseless_control`: 3/3 optimizer successes; 3 locally identifiable; max theta_E relative error 0.0003.
- `p60_low_snr`: 2/2 optimizer successes; 2 locally identifiable; max theta_E relative error 0.1171.
- `p60_high_snr`: 2/2 optimizer successes; 2 locally identifiable; max theta_E relative error 0.0179.
- `p100_stress_baseline`: 2/2 optimizer successes; 2 locally identifiable; max theta_E relative error 0.1144.
- `p100_psf_ten_percent_wide`: 2/2 optimizer successes; 2 locally identifiable; max theta_E relative error 0.0978.
- `p100_source_profile_mismatch`: 2/2 optimizer successes; 2 locally identifiable; max theta_E relative error 0.0957.

## Limitations

- Point-image positions remain conditioned on saved roots in observation-mode fits.
- Local Jacobian rank/conditioning is not a posterior uncertainty or coverage analysis.
- The source mismatch grid perturbs a fixed analytic Sersic profile; it does not span real morphology.
- Low-SNR error is an experiment result, not an implementation failure; noisy cases are not required to meet the 1% analytic-control threshold.
