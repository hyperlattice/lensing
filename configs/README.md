# Configuration

- `study.yaml`: fixed primary C/G/R experiment, including existing artifact paths.
- `simulation.yaml`: original engineering population/observation proposal, unchanged.
- `development.yaml`: source-only v4 reference fits and hyperparameter sweeps.
- `hst_compatibility_pilot.json`: five fixed Dinos-I HST development systems and
  commit-pinned file identities; reserved from subsequent confirmation data.
- `hst_study.yaml`: staged HST preparation, development validation and paired transfer;
  numerical tolerances and scientific qualification rules require development evidence.

Each YAML is explicit and self-contained; there is no registry or inheritance.
The secondary comparison retains its existing frozen design and artifact identities.
See [migration](../docs/migration.md) for old configurations and reproduction boundaries.
