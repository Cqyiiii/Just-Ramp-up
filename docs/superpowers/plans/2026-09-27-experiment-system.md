# Experiment System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. The user authorized autonomous execution without further staged approvals.

**Goal:** Ship and push a maintainable, reproducible experiment runner plus polished paper README.

**Architecture:** Keep historical notebooks intact; add a tested Python package with configuration-driven execution. Sparse numerical kernels feed OLS or the existing DGL GNN, with atomic repetition records and standalone reports.

**Tech Stack:** Python >=3.10, NumPy, SciPy, NetworkX, optional Torch/DGL; pytest; standard-library JSON/HTML/CLI.

**Spec:** `docs/superpowers/specs/2026-09-27-experiment-system-design.md`

## Global Constraints

- Conda ml; preserve legacy files and results; no LLM training or rollouts.
- Paper defaults: stages=(.02,.05,.10,.25,.50), repetitions=100, epochs=400, lr=.004; explicit one-entity near-zero baseline.
- Canonical node alignment, sparse two-hop computation, no filled-in missing outcomes.
- Offline self-contained HTML; no secrets or absolute local metadata in shared reports.
- One final independent whole-branch code review; push is already authorized.

## Review Focus

1. Config typos/nonfinite numbers and graph files whose header silently drops vertices must fail clearly.
2. Tiny graphs where floor(p*m)=0 and Bernoulli stages lack treatment variation must not yield fabricated OLS estimates.
3. Reordering groups or resuming a partial run must not alter completed repetition values or noise assignments.
4. Dynamic stage graphs must travel with their own features into training and evaluation.
5. Interrupted, malformed, mixed-version or concurrent runs must not be summarized as completed experiments.

### Task 1: Scientific core

**Files:** `pyproject.toml`, `.gitignore`, `rampup/{__init__,config,graphs,design,outcomes}.py`, `tests/test_core.py`.

**Interfaces:** `ExperimentConfig.from_dict(dict)` and `.to_dict()`; `GraphData` with adjacency, node_ids, degrees, n_nodes/n_edges; `load_graph(path)` and `from_networkx(graph)`; `assignments(config, n_nodes, membership, rng)`; `merge_groups(config)` yielding `(label, indices)`; `mean_outcome(graph,z,config)` and `sample_outcome(...,rng)` and `true_gate(graph,config)`.

- [ ] Write failing tests for invalid config, file dimensions, canonical IDs, exact cluster assignments, near-zero baseline, monotonicity, and hand-computed two-hop counterfactuals.
- [ ] Run `conda run -n ml python -m pytest tests/test_core.py -q`; expect failures on missing package/behavior.
- [ ] Implement validated config and sparse kernels; dynamic evolution preserves node IDs.
- [ ] Run core tests; expect all pass. Commit the core.

### Task 2: Estimation and summaries

**Files:** `rampup/{estimators,metrics}.py`, `tests/test_estimators.py`.

**Interfaces:** `StageData(graph,treatment,outcome)`; `fit_estimate(stages, evaluation_graph, config)` returns estimate and train_loss. `summarize(records)` returns one metric object per merge group; records contain replicate, group, estimate and truth.

- [ ] Test OLS on hand-derived pooled data; fail on rank-deficient treatment and nonfinite estimates. Test signed bias/MSE using fixed errors [-1,1]. Test tiny GNN determinism and graph-specific stage conversion with real DGL when installed.
- [ ] Run targeted tests, observe missing behavior; implement lazy optional GNN imports and exact ChebConv architecture.
- [ ] Run full pytest suite; expect pass. Commit estimation.

### Task 3: Execution, resume, report and CLI

**Files:** `rampup/{runner,report,cli,__main__}.py`, `configs/*.json`, `tests/test_runner.py`.

**Interfaces:** `run_experiment(config, output_dir, resume=False)` returns run path; `render_report(run_dir)` writes `report.html`; CLI uses these exact public functions.

- [ ] Write tests for deterministic repeat/resume, refusal to overwrite or mix configurations, invalid checkpoints, nonempty output directories, standalone and sanitized HTML, CLI exit status.
- [ ] Run tests and observe missing runner/report behavior. Implement per-repetition atomic JSON and advisory locking, version/data checks, status/attempt tracking and report generation.
- [ ] Run all tests, including CLI demo and report rebuild; expect pass. Commit the runner.

### Task 4: Documentation and validation delivery

**Files:** `README.md`, `READEME.md` redirect, `docs/{migration,experiments}.md`, `docs/assets/overview.svg`, `.github/workflows/tests.yml`, `reports/refactor-validation.html`.

- [ ] Write the README around the actual CLI and tested defaults. Document every scientifically material difference and the legacy-only supplementary scope.
- [ ] Run full tests, demo OLS, tiny GNN smoke, real-data inspection and real-data OLS smoke. Save command/exit/version records; generate final standalone validation report.
- [ ] Open report and visually check its key sections; fix material issues. Check README links and repository diff.
- [ ] Dispatch one fresh whole-branch reviewer per executing-plans; fix important findings with regression tests, then rerun affected checks.
- [ ] Commit, push without force, and verify the remote SHA. Deliver repository link and absolute report path.
