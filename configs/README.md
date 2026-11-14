# Configuration

- `study.yaml` is the shared synthetic-study configuration. Its `profiles.primary`
  and `profiles.seed_replication` sections keep their separate seeds, outputs, and
  experiment IDs while sharing the C/G/R matrix and training controls. The
  `development` section uses those same shared controls.
- `simulation.yaml` defines the physical parameter proposal used by simulation
  generation and renderer comparisons.
- `hst_study.yaml` configures the separate observational preparation and transfer
  workflow.

The fixed five-system HST compatibility sample is a provenance manifest, not a
run configuration. It lives at
[`data/manifests/hst_compatibility_pilot_v1.json`](../data/manifests/hst_compatibility_pilot_v1.json)
and remains reserved from HST confirmation.

See [reproduction](../docs/reproduction.md) for commands and output identities.
