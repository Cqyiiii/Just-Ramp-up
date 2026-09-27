# Running and extending experiments

## Installation

Python 3.10 or newer, Linux or macOS (the run writer uses POSIX advisory locks):

```bash
python -m pip install -e ".[dev]"
python -m rampup demo --output outputs/demo
```

OLS, graph processing, reports and core tests use NumPy, SciPy and NetworkX. They do not import Torch/DGL. For GNN, first install a compatible CPU Torch/DGL pair for your platform using the [official DGL installer](https://www.dgl.ai/pages/start.html), then verify:

```bash
python -c "import torch, dgl; print(torch.__version__, dgl.__version__)"
python -m rampup demo --estimator gnn --repetitions 2 --epochs 8 --output outputs/gnn-demo
```

The optional `gnn` extra declares library dependencies but does not guarantee binary compatibility between every Torch and DGL release. DGL's GraphBolt component can require a matching Torch build even though this project uses ChebConv. The local validation environment successfully imported and tested **Python 3.11, NumPy 1.26.4, SciPy 1.16.0, NetworkX 3.6.1, Torch 2.3.1 and DGL 2.3.0**. Reported versions come from imported modules, not stale package metadata. CPU execution is sufficient; no CUDA or downloaded model weights are required.

Maintainer commands use the existing Conda `ml` environment, for example `conda run -n ml python -m pytest -q`. CI installs only the core dependencies; the GNN test is skipped there when DGL is absent. It is executed in the local validation environment.

## Configuration

JSON configurations are validated before execution. Unknown fields, nonfinite numbers, invalid stage schedules and incompatible baseline schemes fail explicitly. Dataset paths are resolved relative to the **current working directory**, so run the examples from the repository root or supply an absolute dataset path. The complete resolved configuration is saved in each output directory.

| Field | Values / default |
|---|---|
| `dataset` | Repository `.mtx` path, standard Matrix Market path, or `demo` (60-node synthetic graph) |
| `outcome` | `linear`, `multiplicative`, `quadratic`, `sqrt`, `two_hop`, `dynamic` |
| `level` | `unit` or `cluster` |
| `randomization` | `complete` or `bernoulli` |
| `temporal` | `staggered` or `independent` |
| `merge` | `suffix`, `baseline_pair`, `final_pair`, `repeated` |
| `stages` | Strictly increasing proportions; default `[0.02,0.05,0.1,0.25,0.5]` |
| `baseline` | `one` (one treated entity) or `zero`; used only for baseline pairs |
| `estimator` | `gnn` (paper presets) or `ols` (quick diagnostics) |
| `repetitions`, `seed` | 100, 2026 |
| `epochs`, `learning_rate`, `model_seed`, `threads` | 400, 0.004, 2, 1 |
| `resolution`, `cluster_seed`, `graph_seed` | 10, 10, 0 |
| `alpha`, `beta`, `gamma`, `sigma`, `r2` | 1, 1, 1, 0.1, 0.5 |
| `purpose` | `simulation` or `smoke`; controls the report's claim boundary, not the computation |

The fixed GNN has three ChebConv layers `(2,16,2)`, `(16,16,1)`, `(16,1,1)`, default ReLU (including the final layer, as in the notebooks), and Adam betas `(0.9,0.999)`. Each epoch performs one optimizer step per selected stage, in chronological order. Node features are treatment and raw degree. OLS fits only intercept and treatment. Model initialization is fixed separately from assignment and noise randomness.

For linear/square-root/quadratic/dynamic outcomes without isolates, true GATE is `beta + gamma`; multiplicative truth is `alpha*(beta + gamma)`. Two-hop truth is computed from the actual topology after removing the diagonal of `W²`. Noise is averaged out in the target estimand; multiplicative noise follows the degree-weighted outcome model.

### Merge definitions

- `suffix`: report final-only, last two stages, ..., all stages, ordered by the number merged.
- `baseline_pair`: combine each stage with the baseline. The one-entity baseline follows the current manuscript and requires complete randomization. For tiny graphs, a staggered stage must treat at least as many entities as the baseline. Use `zero` when exact global control is intended.
- `final_pair`: report final-only plus each earlier stage paired with the final stage.
- `repeated`: replace every configured stage proportion with the final proportion and merge 1..T repetitions. `temporal=staggered` reuses assignment scores (historical behavior); `independent` redraws assignments each time.

Complete randomization selects `floor(p*m)` of the `m` entities. Cluster-level entity fractions and node fractions can differ. Bernoulli assignment uses independent uniform entity scores thresholded at p; staggered stages share those scores. OLS refuses a group with no treatment variation. A single-stage all-treated design is likewise unidentified for OLS. The graph loader refuses isolates because these degree-normalized outcome models are undefined there.

## Commands

```bash
# Inspect the supplied real graph, optionally calculate the partition.
python -m rampup inspect Dataset/socfb-Stanford3.mtx --clusters

# Quick sparse two-hop pipeline check.
python -m rampup run --config configs/smoke-ols.json --output outputs/smoke-two-hop

# Paper-sized configurations: 100 repetitions × 400 GNN epochs per fitted group.
python -m rampup run --config configs/paper-linear-unit.json --output outputs/linear-unit
python -m rampup run --config configs/paper-two-hop-cluster.json --output outputs/two-hop-cluster

# Resume using exactly the original effective configuration and overrides.
python -m rampup run --config configs/paper-linear-unit.json --output outputs/linear-unit --resume

# Rebuild the report without fitting any models.
python -m rampup report outputs/linear-unit
```

`run` accepts overrides for `--dataset`, `--estimator`, `--repetitions`, `--epochs`, `--seed` and `--threads`. For other choices, copy and edit a small JSON config. Use a different output directory for every changed design. No output directory is overwritten implicitly. To batch studies, loop over config files and give each run a unique output path; the runner itself is intentionally serial and resumable.

## Outputs

```text
outputs/my-study/
├── config.json          # effective frozen configuration
├── manifest.json        # versions, code/run identity, counts, attempts and failures
├── node_ids.json        # original ID for each canonical row
├── clusters.json        # baseline membership, only for cluster assignment
├── replicates/
│   ├── 000000.json      # run identity, seeds, diagnostics and estimates
│   └── ...
└── report.html          # offline report with embedded evidence and handoff block
```

Reports include exact estimates, graph-specific truth, signed bias, population standard deviation, empirical MSE and Monte Carlo intervals. Student-t intervals for bias use the sample standard error; MSE intervals use a normal approximation. These quantify Monte Carlo uncertainty, not uncertainty about deploying a real treatment. Comparisons share stage data within each repetition. Do not interpret marginal intervals as independent pairwise tests. The visible machine-readable handoff is in `<pre id="handoff">` and is derived from the same records as the tables.

Checkpoints and reports are written by atomic replacement and protected by the same advisory run lock. A process crash releases the lock. On resume, a missing full repetition is deterministically recomputed; corrupt or mismatched checkpoints fail rather than being silently discarded. Every checkpoint carries an identity derived from its frozen configuration and execution metadata. One small package-source fingerprint detects changed implementations, including uncommitted edits and installed packages without Git metadata. Dataset identity remains a lightweight filename/size/mtime check, not a content hash or tamper-proof archive. Failures produce a report when the run configuration and manifest remain readable. Fix the underlying failure and use `--resume`; do not fill rows by hand.

`python scripts/validate_refactor.py --reuse` rebuilds the engineering report only from complete validated case records, with the same implementation, runtime and data identity. Missing repetitions fail this gate even if an old manifest says `complete`. Historical reports can still be rebuilt with `python -m rampup report`; their recorded provenance remains visible. Schema 2 checkpoints require execution identities; development-era schema 1 outputs cannot be resumed and need a fresh output directory.

## Extending the system

1. Add the scientific choice to the validated config.
2. Implement the formula in `outcomes.py` or the estimator in `estimators.py`.
3. Add a hand-calculated small-graph test, including its global counterfactual truth.
4. Add a short end-to-end scenario test and document its estimand/limitations.
5. Run `python -m pytest -q`, then a small run and inspect the resulting report before launching full experiments.

The [migration notes](migration.md) distinguish intentional corrections from historical behavior. Supplementary estimators and missing datasets are not silently represented by placeholder implementations.
