# Reproduction

Run commands from the repository root after installing the project as described
in the README. `configs/study.yaml`, `configs/simulation.yaml` and
`configs/development.yaml` define the maintained synthetic configuration surface.
The separate [HST workflow](hst_study.md) uses `configs/hst_study.yaml` and reviewed
observation/reference ledgers; its default scientific qualification choices are unresolved.
There is no experiment registry or YAML inheritance.

## Existing input/output identities

| Stage | Inputs | Existing destination |
|---|---|---|
| Source generation | Original engineering source parents, exclusion identities, numerical references and current qualification | `data/simulations/robustness-study-source-train-v4/` or `robustness-study-source-development-v4/`; reports under `results/reports/robustness-study-banks-v4/` |
| Policy development | Qualified v4 source train/development banks | `results/ml/robustness-development-v4/` |
| C/G/R training | Same source banks and frozen allocation policy | `results/ml/equal-budget-final-v1/fits/` |
| Primary prediction/analysis | Fixed checkpoints and the existing blind-holdout release | `results/ml/equal-budget-final-v1/` |
| Paired generation | Frozen design, excluded banks and two qualified backends | `data/simulations/cross-simulator-rapid-v2/` |
| Secondary prediction/analysis | Paired release and fixed primary checkpoints | `results/ml/cross-simulator-rapid-v2/` |

Large inputs/checkpoints are not bundled. Missing inputs are errors, not an
instruction to silently substitute another dataset or regenerate final data.
Existing completed identities are not overwritten. Historical runs missing a
resolved configuration still require their original runner.

## Commands

```powershell
python -m scripts.study qualify --renderers lenstronomy autolens
python -m scripts.study generate --role source_train --workers 2
python -m scripts.study generate --role source_development --workers 2
python -m scripts.study develop reference
python -m scripts.study develop weights
python -m scripts.study train --config configs/study.yaml
python -m scripts.study predict --config configs/study.yaml
python -m scripts.study analyze --config configs/study.yaml
python -m scripts.study plot --config configs/study.yaml
```

Qualification is an explicit expensive numerical workflow; see [rendering](rendering.md).
The existing source proposal identities depend on `sim-eng1000-v1` and numerical
exclusion records. Creating the original engineering release from raw proposals
is a historical workflow, not silently substituted by this refactor.

For the secondary study, use `generate --paired prepare --design <generation_design.json>`,
then `generate --paired render --renderer lenstronomy` and the corresponding
AutoLens phase, then `generate --paired assemble`. The design must be bound to a
write-once method record before preparation. These commands preserve the original
rapid-comparison release identity by default. They do not implement the retired
formal Freeze-2 experiment. `predict --paired` and `analyze --paired` retain the
fixed secondary destinations and primary checkpoint paths.

Development sweeps use `develop tune`; `develop review` summarizes completed
source-only trials. Review never updates the fixed final configuration automatically.

```powershell
python -m scripts.submission check
python -m scripts.submission export --output submission/aaai27/ai2ase_overleaf_project.zip
```

Only the existing canonical reviewer bundle is exported. Its contents are unchanged.
