# Week 2 — Matched-counterfactual infrastructure

**Status:** gate passed on 2026-09-25  
**Protocol:** `protocol_v1.md`  
**Code:** `src/simulation/counterfactual.py`

## Implemented

- Read frozen parent parameter sidecars and save a canonical state with parent/split identity and source hashes.
- Rerender the same parent under one declared parameter or preprocessing intervention. The intervention API rejects changes to `theta_E_arcsec`, model identity, seeds, and ancestry fields.
- Preserve `parent_sample_id`, `parent_system_id`, `split_group_id`, source/background ancestry, and the original train/validation assignment on every descendant.
- Re-solve point-image positions when source position or shear changes; preserve the saved roots for changes that do not alter image positions.
- Write a release-style manifest, train/validation manifests, parameter sidecars, detector/render products, rejection log, and per-family preview contact sheets.
- Validate the manifest against the existing schema and strict dataset loader. Variable image shapes can be explicitly enabled for later pixel-scale work; the default loader remains 256 x 256.
- Added tests for forbidden target changes, override handling, cross-split ancestry detection, deterministic rerendering, label invariance, and product loading.

The initial implemented render controls cover pixel scale, PSF width, detector noise/exposure, source brightness/position/Sersic properties, lens-light properties, shear, and preprocessing settings. Alternate source families and deliberate counterimage suppression are not implemented because the current renderer does not yet provide validated physical implementations for those changes.

## Smoke benchmark

Command:

```powershell
python -B -m src.simulation.counterfactual --output data/simulations/reliability-counterfactual-smoke-v2 --max-parents-per-split 2
```

It selected two independent train and two independent validation parents and rendered the unchanged canonical scene plus a 1.25x lens-light-amplitude descendant for each parent.

| Check | Result |
|---|---|
| Descendants | 8 from 4 parents; 4 train and 4 validation |
| Rejected renders | 0 |
| Canonical scene identity | Exact for all 4 parents; max absolute difference 0 |
| Detector observation identity | Exact for all 4 parents; max absolute difference 0 |
| Preprocessed image identity | Exact for all 4 parents; max absolute difference 0 |
| `theta_E_arcsec` invariance | Passed |
| Ancestry/split isolation | Passed; no ancestry field crossed train/validation |
| Strict dataset loader | Passed for 4 train and 4 validation descendants |
| Preview contact sheets | Written separately for `canonical` and `lens_light` |
| Focused tests | `4 passed in 4.90s` |

Generated smoke artifacts are local and ignored under `data/simulations/reliability-counterfactual-smoke-v2/`. The smoke benchmark made no model predictions and is infrastructure evidence, not a robustness result.

## Gate decision

Week 2 passes: unchanged rerenders reproduce the stored parent products exactly; labels remain fixed; descendants inherit their parent's split; and the release is readable through the existing strict loader. Proceed to the Week 3 pixel-scale audit only after the pixel-scale levels and field-of-view convention are frozen in the protocol.
