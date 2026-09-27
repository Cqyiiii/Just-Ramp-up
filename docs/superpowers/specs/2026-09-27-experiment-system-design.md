# Reproducible experiment system

## Brief and authority

Refactor the experiment system for maintainability and reproducibility, then push it to Cqyiiii/Just-Ramp-up. The user authorized autonomous design, implementation, validation and push on 2026-09-27. Also create an attractive README marked **NeurIPS 2026**, linking https://arxiv.org/abs/2410.12740. The supplied latest manuscript is titled *Just Ramp-Up: Debiasing Regression-based Estimator for A/B Tests under Network Interference*. Its simulation section and appendices take precedence over older notebook conventions. The arXiv landing page currently retains the earlier title.

## Architecture

Keep the original notebooks and Result files at their original paths for traceability. Add an installable `rampup` package and a single `python -m rampup` CLI as the maintained execution path. Separate configuration, sparse graph loading/evolution, assignments, potential outcomes, merging, estimation, orchestration, metrics, and standalone reporting. No new service, LLM API, model downloads, or notebook execution engine is required.

The six supported outcome settings are linear, multiplicative, quadratic, square-root, two-hop, and linear with dynamic graphs. Support unit/cluster assignment, complete/Bernoulli randomization, staggered/independent assignment over time, suffix merging, baseline pairs, fixed final-stage pairs, and repeated experiments. Estimators are graph-agnostic OLS and the existing DGL ChebConv GNN. Historical HT notebooks remain available; MII, Hájek, Lagrange and MIT8 reproduction are explicitly outside the implemented runner because complete source/data is absent from this checkout. Do not claim their reproduction.

## Scientific contracts

- Default stages: 0.02, 0.05, 0.10, 0.25, 0.50. Paper configuration: 100 repetitions, 400 epochs, Adam lr=0.004 and betas=(0.9,0.999); ChebConv dimensions/filter sizes (2,16,2), (16,16,1), (16,1,1), default ReLU; features treatment and unnormalized degree. Louvain resolution=10, seed=10.
- Maintain one canonical sorted node order for adjacency, treatment, outcomes, features and cluster membership. Reject loops, empty graphs, isolates, malformed graph headers, invalid endpoints and inconsistent edge counts.
- Complete randomization chooses exactly floor(p*m) assignment entities, where m is the number of units or clusters. Record realized unit proportions separately from entity proportions. Staggered assignment shares one ordering/uniform vector. Independent assignment draws separately per stage. Repeated designs expose either temporal choice explicitly, because historical repeat notebooks reuse the same treated units.
- Baseline pairs default to exactly one treated entity, matching the latest paper. Explicit `zero` baseline is also supported. This is a design correction relative to some historical notebooks.
- Generate a stage's outcome once per repetition and reuse it across merge comparisons. Use independent deterministic random streams for assignment and noise. Fix model initialization separately. This makes comparisons paired; it does not reproduce historical random streams.
- Outcomes default to alpha=beta=gamma=1, sigma=0.1. Two-hop spillover is gamma*(Wz + r2*(W(Wz)-diag(W^2)z)), r2=0.5, W=D^-1 A. Never materialize dense n-by-n or squared adjacency matrices. Derive the true noise-averaged GATE on the actual graph from counterfactual means.
- Dynamic graphs evolve exogenously once per stage using graph_seed=0: each node at or above the median degree gets five degree-weighted candidate neighbors sampled with replacement; discard loops/existing edges. Preserve initial clusters and canonical node IDs. Train each stage on its own graph; evaluate on the last selected stage's graph. Store per-stage edge counts. The exact historical edge counts are not promised under changed node ordering/RNG.
- Report signed bias, population standard deviation (ddof=0), empirical MSE, MC standard errors and clearly labelled 95% Monte Carlo intervals. Never include uncompleted repetitions as zeros. Full-paper numerical equality is not an acceptance criterion for a corrected implementation.

## Reproducibility and artifacts

Validated JSON configuration, deterministic seed schedule, source Git commit, imported dependency versions, data size/mtime and graph counts, node IDs, cluster labels, stage diagnostics, and per-repetition records. Atomic completion records permit exact resume with unchanged configuration, data identity, graph metadata and runtime versions. Reject malformed/incompatible records; guard concurrent writes. No exhaustive file hashes or repeated data hashing.

Every run produces an offline standalone HTML report, readable without JavaScript or sibling assets. Include study identity/status, purpose/claim boundaries, configuration, methods, execution attempts/failures, exact summaries/uncertainty, diagnostics, raw records, reproducibility index, and a visible machine-readable handoff block. Redact absolute local paths and host metadata from the shareable HTML. Reports are engineering/Monte Carlo outputs, not evidence that paper tables have been independently reproduced.

## User surface

CLI: `demo`, `run --config ... --output ... [--resume]`, `report RUN_DIR`, `inspect DATASET`. Demo works without Torch/DGL by default. Config examples cover paper GNN and smoke OLS. Heavy dependencies are optional and imported only for GNN. README supplies tested install/run commands, a repo-native SVG overview, scientific defaults, supported scope, output format, migration notes, NeurIPS 2026 badge and arXiv citation. No invented acceptance, award or benchmark claims.

## Verification and delivery gates

Use Conda `ml`; interpreter verified as `/opt/anaconda3/envs/ml/bin/python`. Tests must cover nonconsecutive node IDs, header rows and isolated vertices, exact/staggered/cluster assignment, true zero versus one-entity baseline, sparse two-hop diagonal removal, graph-specific GNN stage inputs, pooling, outcome formulas, resume equivalence, corrupted/incompatible records and HTML sanitization. Run an OLS demo, tiny real GNN demo, real Stanford graph inspection and short OLS smoke. Deliver a self-contained validation HTML with execution records and honest limits; do not rerun the entire 100x400 paper suite as part of this refactor.

Work in the fresh dedicated clone on `refactor/experiment-system`. Review the full change, then fast-forward the authorized remote main without force if its base remains unchanged; otherwise preserve remote changes and integrate safely. Never overwrite historical simulation outputs.
