# Study simplification

The clean pre-refactor baseline is Git revision
`aac2e583b32ddea3e7f93ed23485e0bf395eb1b4`. Use that checkout in a separate directory
for the original 193-test suite and historical workflows. Archived scripts are
historical records, not a second maintained package. The complete retired-file
inventory is in [migration_inventory.json](migration_inventory.json).

## Ownership

| Earlier implementation | Maintained owner |
|---|---|
| Equal-budget runner | `src/ml/train.py`, `predict.py`, `analysis.py`; `scripts.study` dispatches |
| Robustness training and source scoring | One C/G/R trainer, `ml/data.py` and `ml/policy.py` |
| Benchmark/precise/low-memory renderers and adaptive integration | One `simulation/render.py` |
| AutoLens v2 adapter | `simulation/backends.py` plus shared `observation.py` |
| Repeated condition definitions | `simulation/observation.py` |
| Source/paired bank interfaces | `data/dataset.py`; lineage/path contracts in `data/schema.py` |
| Archived v4 source generator | `simulation/generate.py` with self-contained source helpers |
| Archived paired generation/evaluation | `simulation/paired.py`, `ml/predict.py` and `ml/analysis.py` |
| Many configuration directories | Three active YAML files: shared study profiles, simulation settings, and HST settings |
| Submission checks and exporter | `scripts/submission.py` |

Original reliability phases, metadata/CASTLES preprocessing, scale-aware variants,
classical fitting, MMD/alternative-arm studies, Huber baseline suites and formal
confirmatory/power machinery are retired from active source. Their code/tests are
available in Git. Existing datasets, curated labels, reports and manuscript assets
remain in their original locations. Older documentation remains historical.

The maintained source-bank generator no longer requires the historical T-arm
randomized training release. C/G/R never used its historical-view metadata; new
source banks retain canonical compatibility fields. Source-only tuning preserves
its original learning-rate/weight-decay grid; MMD and N/T arms are removed.
Method records now hash their named inputs: archived paired generators expected
those hashes, but the baseline helper stored paths only. This is an explicit
boundary repair, not a claim that old manifests have new provenance.

## Validation

Baseline: 193 tests passed, one optional AutoLens test skipped. Before editing,
small C/G/R runs and renderer products were saved for direct comparison. On the
local CPU environment, the consolidated trainer matched checkpoint tensors,
histories, schedules and predictions exactly; the consolidated renderer matched
all saved native reference fields exactly. Compact reference fixtures remain in
`tests/fixtures/`, with their source revision recorded.

The current suite covers retained contracts, three-arm pairing, source-only
normalization, label exclusion, bootstrap parent units, checkpoint regression,
rendering regression, submission exports and a temporary end-to-end workflow.
A smaller suite reflects retired scientific workflows, not an assertion that
all 193 historical tests are still applicable. Full production training,
315-case requalification and alternate-backend scientific runs are not performed
as part of maintenance.

## Final maintenance checks

The maintained suite passed **47 tests**, with one optional AutoLens physical
backend test skipped because the package is unavailable locally. Ruff lint and
format checks, maintained-reference/import checks and reviewer submission checks
passed. All 2,506 pre-existing data/result/reviewer files retained their original
hashes. Ten disposable `.pytest-work*` folders were removed. Active source changed
from 87 to 22 Python files (19,683 to 4,142 lines before final documentation and
interface polish); ML implementation modules changed from 31 to 8.

The local Python 3.10 CPU environment validates the primary workflow, not a full
Python 3.12 AutoLens installation. Full rendering qualification remains an
explicit separate action, requiring the complete numerical reference set.
